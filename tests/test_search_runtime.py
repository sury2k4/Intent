from pathlib import Path
from dataclasses import replace

from model_intent.osm_ingest import load_property_catalog
from model_intent.poi_ranking import load_ranking_policy
from model_intent.poi_repository import PoiRepository
from model_intent.poi_schemas import PoiRecord
from model_intent.property_extraction import PropertyExtractor
from model_intent.rendering import load_enrichment
from model_intent.search_runtime import SearchRuntime
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


class CategoryStub:
    def __init__(self, action="accept", category_id="23"):
        self.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
        self.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
        self.action = action
        self.category_id = category_id
        self.queries = []
        self.database = None

    def query(self, text, **kwargs):
        self.queries.append(text)
        return {"prediction": {"query": text, "hits": []}, "decision": {"action": self.action, "accepted_item_id": self.category_id if self.action == "accept" else None}, "runtime": {}}

    def resolve_selection(self, *, query, selected_id, **kwargs):
        return {"query": query, "decision": {"action": "accept", "accepted_item_id": selected_id}, "runtime": {}}


class CountingRepository(PoiRepository):
    def __init__(self, records):
        super().__init__(records, generation_id=10)
        self.calls = 0

    def for_category(self, category_id):
        self.calls += 1
        return super().for_category(category_id)


def make_poi(poi_id, has_ac=True):
    return PoiRecord(
        poi_id=poi_id,
        osm_type="node",
        osm_id=int(poi_id.rsplit(":", 1)[1]),
        category_id="23",
        category_code="coffee",
        name="Cafe",
        display_name="Cafe",
        names={},
        latitude=10.78,
        longitude=106.69,
        location_method="node",
        mapping_rule_id="amenity-cafe",
        mapping_version="v1",
        effective_properties={"air_conditioning": {"value": has_ac, "status": "known", "source": "osm"}},
        rating=4.6,
        review_count=100,
    )


def runtime(action="accept"):
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v1.json")
    repo = CountingRepository([make_poi("osm:node:1", True), make_poi("osm:node:2", False)])
    category = CategoryStub(action=action)
    search = SearchRuntime(
        category_runtime=category,
        property_extractor=PropertyExtractor(catalog),
        poi_repository=repo,
        ranking_policy=load_ranking_policy(ROOT / "data/policies/poi_ranking.poc.v1.json"),
        property_catalog=catalog,
    )
    return search, category, repo


def test_osm_only_runtime_forces_verified_evidence_and_returns_null_rating():
    search, _category, _repository = runtime()
    search.ranking_policy = load_ranking_policy(ROOT / "data/policies/poi_ranking.poc.v3.json")
    tagged = replace(make_poi("osm:node:1"), rating=None, review_count=0, rating_source="unknown")
    generated = replace(make_poi("osm:node:3"), rating=None, review_count=0, rating_source="unknown",
                        name=None, display_name="[Generated OSM label] coffee (osm:node:3)")
    search.poi_repository.records = (tagged, generated)
    accepted = search.search("quán cà phê có máy lạnh")
    assert accepted["poi_search"]["verified_only"] is True
    assert accepted["poi_search"]["ranking_schema_version"] == 3
    assert {row["poi_id"] for row in accepted["poi_search"]["hits"]} == {"osm:node:1", "osm:node:3"}
    assert all(hit["score"]["adjusted_rating"] is None for hit in accepted["poi_search"]["hits"])
    assert next(hit for hit in accepted["poi_search"]["hits"] if hit["poi_id"] == "osm:node:3")["name"] is None
    selected = search.resolve_selection_and_search(
        query="quán cà phê hoặc nhà thuốc có máy lạnh", selected_id="23", allowed_option_ids=["23", "34"],
    )
    assert selected["poi_search"]["verified_only"] is True


def test_osm_only_filters_unknown_negatives_in_runtime():
    search, _category, _repository = runtime()
    search.ranking_policy = load_ranking_policy(ROOT / "data/policies/poi_ranking.poc.v3.json")
    known_false = replace(make_poi("osm:node:3"), rating=None, review_count=0, rating_source="unknown",
                          effective_properties={"internet_access": {
        "value": False, "status": "known", "source": "osm",
    }})
    unknown = replace(make_poi("osm:node:4"), rating=None, review_count=0,
                      rating_source="unknown", effective_properties={})
    search.poi_repository.records = (known_false, unknown)
    result = search.search("quán cà phê không có wifi")
    assert [hit["poi_id"] for hit in result["poi_search"]["hits"]] == [known_false.poi_id]
    assert result["poi_search"]["hard_filter_match_count"] == 1


def test_zero_minimum_profiles_reconcile_nonapplicable_filters_in_runtime():
    from model_intent.poi_properties import load_property_profiles

    search, category, _repository = runtime()
    category.category_id = "114"
    profiles = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v2.json", category.taxonomy)
    search.property_profiles = replace(profiles, minimum_known=0)
    search.property_catalog = profiles.definitions
    search.property_extractor = PropertyExtractor(profiles.definitions)
    bus_stop = replace(make_poi("osm:node:5"), category_id="114", category_code="bus_stop", effective_properties={})
    search.poi_repository.records = (bus_stop,)
    result = search._search_from_category(
        "điểm dừng xe buýt có máy lạnh", category.query("điểm dừng xe buýt"),
        poi_top_k=5, language="vi", origin=None, radius_km=None, verified_only=False,
    )
    assert result["poi_search"]["returned_count"] == 1
    assert result["property_extraction"]["properties"][0]["ignored_reason"] == "category_inapplicable"


def test_search_preserves_raw_category_query_and_filters_required_poi_property():
    search, category, repo = runtime()
    query = "quán cà phê có máy lạnh"
    result = search.search(query)
    assert category.queries == [query]
    assert result["intent"]["decision"]["accepted_item_id"] == "23"
    assert result["poi_search"]["status"] == "ok"
    assert result["poi_search"]["hard_filter_match_count"] == 1
    assert [hit["poi_id"] for hit in result["poi_search"]["hits"]] == ["osm:node:1"]
    assert repo.calls == 1


def test_single_query_resolves_category_then_returns_filtered_pois():
    search, category, repo = runtime(action="reject")
    query = "quán cà phê có máy lạnh, ưu tiên thoáng mát"
    result = search.search(query)
    assert category.queries == [query]
    assert result["intent"]["decision"]["accepted_item_id"] == "23"
    assert result["intent"]["category_resolution"]["source"] == "explicit_phrase"
    assert result["poi_search"]["status"] == "ok"
    assert [item["poi_id"] for item in result["poi_search"]["hits"]] == ["osm:node:1"]
    assert repo.calls == 1


def test_multiple_explicit_categories_require_clarification_before_pois():
    search, _category, repo = runtime(action="reject")
    query = "quán cà phê hoặc nhà thuốc"
    result = search.search(query)
    assert result["intent"]["decision"]["action"] == "clarify"
    assert {item["item_id"] for item in result["intent"]["decision"]["options"]} == {"23", "34"}
    assert result["poi_search"]["status"] == "skipped_category_clarify"
    assert repo.calls == 0
    selected = search.resolve_selection_and_search(
        query=query, selected_id="23", allowed_option_ids=["23", "34"],
    )
    assert selected["intent"]["decision"]["accepted_item_id"] == "23"
    assert selected["property_extraction"]["properties"] == []
    assert selected["poi_search"]["category_candidate_count"] == 2
    assert selected["poi_search"]["hard_filter_match_count"] == 2


def test_clarification_selection_does_not_recast_an_alternative_category_as_property():
    search, _category, repository = runtime(action="reject")
    query = "quán cà phê hoặc bãi đỗ xe"
    initial = search.search(query)
    assert initial["intent"]["decision"]["action"] == "clarify"
    assert initial["property_extraction"]["properties"][0]["ignored_reason"] == "category_component"
    allowed = [option["item_id"] for option in initial["intent"]["decision"]["options"]]
    selected = search.resolve_selection_and_search(
        query=query, selected_id="23", allowed_option_ids=allowed,
    )
    assert selected["property_extraction"]["properties"][0]["ignored_reason"] == "category_component"
    assert selected["poi_search"]["hard_filter_match_count"] == 2
    assert selected["poi_search"]["returned_count"] == 2
    assert repository.calls == 1


def test_clarification_skips_poi_repository():
    search, _category, repo = runtime(action="clarify")
    result = search.search("chỗ ngồi uống gì đó")
    assert result["poi_search"]["status"] == "skipped_category_clarify"
    assert repo.calls == 0


def test_conflicting_semantic_and_explicit_category_asks_instead_of_searching_wrong_pois():
    search, _category, repo = runtime(action="accept")
    search.category_runtime.category_id = "34"
    result = search.search("quán cà phê có máy lạnh")
    assert result["intent"]["decision"]["action"] == "clarify"
    assert {item["item_id"] for item in result["intent"]["decision"]["options"]} == {"23", "34"}
    assert repo.calls == 0


def test_informational_query_with_category_name_is_rejected():
    search, _category, repo = runtime(action="reject")
    result = search.search("cách mở quán cà phê")
    assert result["intent"]["decision"]["action"] == "reject"
    assert result["poi_search"]["status"] == "skipped_category_reject"
    assert repo.calls == 0


def test_real_model_software_request_cannot_search_physical_pois():
    search, category, repo = runtime(action="accept")
    category.model_key = "e5-base-multilingual"
    result = search.search("tìm phần mềm quản lý bán hàng cho laptop")
    assert result["intent"]["decision"]["reason"] == "digital_product_not_place_search"
    assert result["poi_search"]["status"] == "skipped_category_reject"
    assert repo.calls == 0


def test_missing_radius_origin_is_reported_as_not_applied():
    search, _category, _repo = runtime()
    result = search.search("quán cà phê trong 2 km")
    radius = next(row for row in result["property_extraction"]["properties"] if row["property_id"] == "radius_km")
    assert radius["applied"] is False
    assert radius["ignored_reason"] == "missing_origin"
    assert result["poi_search"]["hard_filter_match_count"] == 2
    assert any("origin" in warning for warning in result["poi_search"]["warnings"])


def test_sushi_query_includes_japanese_restaurant_with_explicit_sushi_cuisine_tag():
    search, category, _repo = runtime(action="accept")
    category.category_id = "65"
    source = replace(
        make_poi("osm:node:101"), category_id="57", category_code="japanese_restaurant",
        osm_tags={"amenity": "restaurant", "cuisine": "japanese;sushi"},
    )
    other = replace(
        make_poi("osm:node:102"), category_id="57", category_code="japanese_restaurant",
        osm_tags={"amenity": "restaurant", "cuisine": "japanese"},
    )
    search.poi_repository.records = (source, other)
    result = search.search("nhà hàng Sushi")
    assert result["intent"]["decision"]["accepted_item_id"] == "65"
    assert result["poi_search"]["category_candidate_count"] == 1
    assert [hit["poi_id"] for hit in result["poi_search"]["hits"]] == [source.poi_id]


def test_combined_radius_constraints_use_the_narrower_limit():
    search, _category, _repository = runtime()
    nearby = make_poi("osm:node:8")
    farther = replace(make_poi("osm:node:9"), latitude=10.81)
    search.poi_repository.records = (nearby, farther)
    result = search.search("quán cà phê trong 1 km và trong 5 km", origin=(10.78, 106.69))
    assert result["poi_search"]["effective_radius_km"] == 1.0
    assert [hit["poi_id"] for hit in result["poi_search"]["hits"]] == [nearby.poi_id]
    assert all(row["applied"] for row in result["property_extraction"]["properties"] if row["property_id"] == "radius_km")


def test_explicit_radius_override_is_reported_without_claiming_query_radius_applied():
    search, _category, _repo = runtime()
    result = search.search("quán cà phê trong 2 km", origin=(10.78, 106.69), radius_km=10)
    radius = next(row for row in result["property_extraction"]["properties"] if row["property_id"] == "radius_km")
    assert radius["applied"] is False
    assert radius["ignored_reason"] == "overridden_by_explicit_radius"
    assert result["poi_search"]["effective_radius_km"] == 10
    assert "query radius was replaced" in " ".join(result["poi_search"]["warnings"])


def test_real_model_rejection_cannot_be_promoted_by_explicit_category_alias():
    search, category, repository = runtime(action="reject")
    category.model_key = "e5-base-multilingual"
    result = search.search("quán cà phê có máy lạnh")
    assert result["intent"]["decision"]["action"] == "reject"
    assert result["intent"]["category_resolution"]["source"] == "semantic_model"
    assert result["poi_search"]["status"] == "skipped_category_reject"
    assert repository.calls == 0


def test_real_model_acceptance_conflicting_with_explicit_alias_requires_clarification():
    search, category, repository = runtime(action="accept")
    category.model_key = "e5-base-multilingual"
    category.category_id = "23"
    result = search.search("nhà thuốc")
    assert result["intent"]["decision"]["action"] == "clarify"
    assert {option["item_id"] for option in result["intent"]["decision"]["options"]} == {"23", "34"}
    assert result["intent"]["category_resolution"]["source"] == "semantic_guard_clarify"
    assert repository.calls == 0


def test_empty_poi_results_do_not_change_category_acceptance():
    search, _category, _repo = runtime()
    result = search.search("quán cà phê có máy lạnh")
    # Replace with a constraint guaranteed to fail against both fixture POIs.
    search.poi_repository.records = tuple(
        make_poi(f"osm:node:{index}", False) for index in (3, 4)
    )
    result = search.search("quán cà phê có máy lạnh")
    assert result["intent"]["decision"]["action"] == "accept"
    assert result["poi_search"]["status"] == "no_matches"


def test_branded_category_guard_does_not_search_generic_charging_pois():
    search, category, repository = runtime(action="accept")
    category.model_key = "e5-base-multilingual"
    category.category_id = "19"
    result = search.search("trạm sạc nhanh ,có nhà vệ sinh")
    assert result["intent"]["query_interpretation"]["category_text"] == "trạm sạc nhanh"
    assert result["intent"]["decision"]["reason"] == "unsupported_branded_category"
    assert result["poi_search"]["status"] == "unsupported_category"
    assert repository.calls == 0


def test_named_branded_category_does_not_require_the_parent_brand_word():
    search, category, repository = runtime(action="accept")
    category.model_key = "e5-base-multilingual"
    category.category_id = "234"
    result = search.search("Vinschool")
    assert result["intent"]["decision"]["action"] == "accept"
    assert result["intent"]["decision"]["accepted_item_id"] == "234"
    assert result["poi_search"]["status"] == "no_pois_for_category"
    assert repository.calls == 1


def test_real_model_focused_query_does_not_erase_category_component():
    search, category, _repository = runtime(action="accept")
    category.model_key = "e5-base-multilingual"
    category.category_id = "52"
    search.property_catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v2.json")
    search.property_extractor = PropertyExtractor(search.property_catalog)
    result = search.search("máy rút tiền")
    assert category.queries == ["máy rút tiền"]
    assert result["intent"]["query_interpretation"]["category_text"] == "máy rút tiền"
    assert result["property_extraction"]["properties"][0]["ignored_reason"] == "category_component"


def test_real_model_uses_focused_category_when_raw_query_is_distracted():
    search, category, _repository = runtime(action="accept")
    category.model_key = "e5-base-multilingual"
    category.category_id = "23"
    result = search.search("quán cà phê có máy lạnh")
    assert category.queries == ["quán cà phê", "quán cà phê có máy lạnh"]
    assert result["intent"]["category_resolution"]["semantic_query"] == "quán cà phê"
    assert result["intent"]["decision"]["accepted_item_id"] == "23"
    assert result["property_extraction"]["properties"][0]["applied"]
