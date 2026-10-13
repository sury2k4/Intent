from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_intent.category_query_aliases import CategoryQueryAliases
from model_intent.poi_properties import load_property_definitions
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


def runtime(action="accept"):
    catalog = load_property_definitions(ROOT / "data/properties/poi_properties.poc.v1.json")
    category = CategoryStub(action=action)
    search = SearchRuntime(
        category_runtime=category,
        property_extractor=PropertyExtractor(catalog),
        property_catalog=catalog,
        property_semantic_path=ROOT / "data/properties/property_semantic_profiles.poc.v2.json",
        property_generation_id=4,
    )
    return search, category, SimpleNamespace(calls=0)


def test_catalog_property_and_pending_request_share_one_list():
    search, category, repository = runtime()
    category.model_key = "e5-base-multilingual"
    result = search.analyze_intent("quán cà phê có máy lạnh gần nhất")
    assert category.queries == ["quán cà phê"]
    properties = result["property_extraction"]["properties"]
    assert [(row["property_id"], row["applied"]) for row in properties] == [
        ("air_conditioning", True), ("near", False),
    ]
    assert "status" not in properties[0]
    assert (properties[1]["status"], properties[1]["requires"]) == (
        "pending_location", ["user_location", "poi_coordinates"],
    )
    assert [row["property_id"] for row in result["query_parts"]["filter_candidates"]] == ["air_conditioning"]
    assert result["query_parts"]["requested_search"]["sort_candidates"][0]["kind"] == "distance_asc"
    assert "filter_candidates" not in result["query_parts"]["requested_search"]
    assert "poi_search" not in result and repository.calls == 0


def test_pending_requested_filters_are_exposed_without_poi_execution():
    search, category, repository = runtime()
    category.model_key = "e5-base-multilingual"
    result = search.analyze_intent("quán ăn ngon gần nhất")
    assert category.queries == ["quán ăn"]
    assert result["query_parts"]["category_text"] == "quán ăn"
    requested = result["query_parts"]["requested_search"]
    properties = result["property_extraction"]["properties"]
    assert [row["property_id"] for row in properties] == ["quality_by_reviews", "near"]
    assert [row["status"] for row in properties] == ["pending_review_data", "pending_location"]
    assert all(not row["applied"] for row in properties)
    assert "filter_candidates" not in requested
    assert [row["kind"] for row in requested["sort_candidates"]] == ["distance_asc", "review_score_desc"]
    assert result["query_parts"]["filter_candidates"] == []
    assert result["query_parts"]["preference_candidates"] == []
    assert result["query_parts"]["unresolved_property_clauses"] == []
    assert "poi_search" not in result and repository.calls == 0
    assert result["intent"]["query_interpretation"]["requested_search"] == requested


def test_explicit_review_rating_is_pending_not_an_applied_filter():
    search, category, repository = runtime()
    category.model_key = "e5-base-multilingual"
    result = search.analyze_intent("quán cà phê rating above 4.5")
    assert category.queries == ["quán cà phê"]
    rating = next(row for row in result["property_extraction"]["properties"] if row["property_id"] == "rating")
    assert rating["applied"] is False and rating["ignored_reason"] == "rating_unverified"
    assert (rating["operator"], rating["value"]) == ("gt", 4.5)
    assert result["query_parts"]["filter_candidates"] == []
    assert rating["status"] == "pending_review_data"
    assert "filter_candidates" not in result["query_parts"]["requested_search"]
    assert repository.calls == 0


def test_selected_category_preserves_pending_search_request_without_reretrieval():
    search, category, repository = runtime(action="reject")
    category.model_key = "e5-base-multilingual"
    initial = search.analyze_intent("quán cà phê gần nhất")
    selected = search.resolve_selection_and_analyze_intent(
        query="quán cà phê gần nhất", selected_id="23", allowed_option_ids=["23"],
    )
    assert len(category.queries) == 1
    assert selected["query_parts"]["requested_search"] == initial["query_parts"]["requested_search"]
    assert selected["property_extraction"]["properties"] == initial["property_extraction"]["properties"]
    assert selected["property_extraction"]["properties"][0]["status"] == "pending_location"
    assert selected["query_parts"]["filter_candidates"] == []
    assert "poi_search" not in selected and repository.calls == 0


def test_intent_analysis_returns_category_and_properties_without_poi_search():
    search, category, repository = runtime()
    query = "quán cà phê có máy lạnh"
    result = search.analyze_intent(query)
    assert category.queries == [query]
    assert result["intent"]["decision"]["accepted_item_id"] == "23"
    assert result["property_extraction"]["properties"][0]["property_id"] == "air_conditioning"
    assert result["property_extraction"]["properties"][0]["applied"] is True
    assert result["timing"]["path"] == "intent_analysis"
    assert not any(row["name"].startswith("poi_") for row in result["timing"]["stages"])
    assert "poi_search" not in result
    assert repository.calls == 0


def test_intent_analysis_uses_only_category_text_for_real_model_and_exposes_filter_candidates():
    search, category, repository = runtime()
    category.model_key = "e5-base-multilingual"
    query = "quán cà phê có máy lạnh"
    result = search.analyze_intent(query)
    assert category.queries == ["quán cà phê"]
    assert result["query_parts"]["category_text"] == "quán cà phê"
    assert result["query_parts"]["pgvector_queries"] == ["quán cà phê"]
    assert result["query_parts"]["vector_query_mode"] == "focused"
    assert [(row["property_id"], row["mode"], row["value"]) for row in result["query_parts"]["filter_candidates"]] == [
        ("air_conditioning", "required", True),
    ]
    assert "poi_search" not in result
    assert repository.calls == 0



def test_intent_filter_candidates_respect_negation_and_ignored_properties():
    search, category, _repository = runtime()
    category.model_key = "e5-base-multilingual"
    excluded = search.analyze_intent("quán cà phê không có wifi, có máy lạnh")
    assert category.queries == ["quán cà phê"]
    assert {(row["property_id"], row["mode"]) for row in excluded["query_parts"]["filter_candidates"]} == {
        ("internet_access", "excluded"), ("air_conditioning", "required"),
    }
    not_requested = search.analyze_intent("quán cà phê không cần wifi, có máy lạnh")
    assert {row["property_id"] for row in not_requested["query_parts"]["filter_candidates"]} == {
        "air_conditioning",
    }
    assert next(row for row in not_requested["property_extraction"]["properties"]
                if row["property_id"] == "internet_access")["ignored_reason"] == "not_requested"
    preferred = search.analyze_intent("quán cà phê ưu tiên máy lạnh")
    assert preferred["query_parts"]["filter_candidates"] == []
    assert [(row["property_id"], row["mode"]) for row in preferred["query_parts"]["preference_candidates"]] == [
        ("air_conditioning", "preferred"),
    ]


def test_intent_analysis_does_not_vector_search_property_only_query():
    search, category, repository = runtime()
    category.model_key = "e5-base-multilingual"
    result = search.analyze_intent("có máy lạnh")
    assert category.queries == []
    assert result["intent"]["decision"]["reason"] == "no_category_phrase"
    assert result["query_parts"]["category_text"] is None
    assert result["query_parts"]["pgvector_queries"] == []
    assert result["query_parts"]["filter_candidates"][0]["property_id"] == "air_conditioning"
    assert result["query_parts"]["applicability"] == "pending_category"
    assert repository.calls == 0


def test_intent_analysis_marks_original_query_fallback_when_focus_is_too_weak():
    search, category, _repository = runtime()
    category.model_key = "e5-base-multilingual"
    result = search.analyze_intent("shop co may lanh")
    assert category.queries == ["shop co may lanh"]
    assert result["query_parts"]["category_text"] == "shop"
    assert result["query_parts"]["pgvector_queries"] == ["shop co may lanh"]
    assert result["query_parts"]["vector_query_mode"] == "original_fallback"


def test_lodging_typo_splits_filters_and_requires_user_category_confirmation():
    from model_intent.poi_properties import load_property_profiles

    search, category, repository = runtime(action="reject")
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    profiles = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v5.json", category.taxonomy)
    search.property_profiles = profiles
    search.property_catalog = profiles.definitions
    search.property_extractor = PropertyExtractor(profiles.definitions)

    def unrelated_top(text, **kwargs):
        category.queries.append(text)
        return {
            "prediction": {"query": text, "hits": [{"item_id": "76", "score": .8078}]},
            "decision": {"action": "reject", "reason": "low_calibrated_confidence",
                         "accepted_item_id": None, "calibrated_confidence": .263},
            "runtime": {},
        }

    category.query = unrelated_top
    query = "nhà nghĩ có bãi đỗ ô tô, không có chỗ cho xe lăng"
    result = search.analyze_intent(query)
    decision = result["intent"]["decision"]
    assert category.queries == ["nhà nghĩ"]
    assert result["query_parts"]["category_boundary"]["cue"] == "có"
    assert result["query_parts"]["unresolved_property_clauses"] == []
    assert [(row["property_id"], row["mode"]) for row in result["query_parts"]["filter_candidates"]] == [
        ("parking", "required"), ("wheelchair", "excluded"),
    ]
    assert decision["action"] == "clarify" and decision["reason"] == "folded_category_name_confirmation"
    assert decision["accepted_item_id"] is None
    assert [row["item_id"] for row in decision["options"]] == ["137"]
    assert decision["calibrated_confidence"] == .263
    assert result["intent"]["category_resolution"]["original_semantic_decision"]["action"] == "reject"
    assert repository.calls == 0 and "poi_search" not in result

    selected = search.resolve_selection_and_analyze_intent(
        query=query, selected_id="137", allowed_option_ids=["137"],
    )
    assert selected["intent"]["decision"]["accepted_item_id"] == "137"
    assert [(row["property_id"], row["mode"]) for row in selected["query_parts"]["filter_candidates"]] == [
        ("parking", "required"), ("wheelchair", "excluded"),
    ]
    assert repository.calls == 0
    assert search.analyze_intent("không phải nhà nghĩ")["intent"]["decision"]["action"] == "reject"
    assert search.analyze_intent("cách mở nhà nghĩ")["intent"]["decision"]["action"] == "reject"


def test_short_wifi_typo_stays_a_suggestion_until_user_confirms():
    from model_intent.poi_properties import load_property_profiles

    search, category, repository = runtime()
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    profiles = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v5.json", category.taxonomy)
    search.property_profiles = profiles
    search.property_catalog = profiles.definitions
    search.property_extractor = PropertyExtractor(profiles.definitions)
    query = "quán cà phê không có wifa"
    result = search.analyze_intent(query)
    assert category.queries == ["quán cà phê"]
    assert result["query_parts"]["category_text"] == "quán cà phê"
    assert result["query_parts"]["filter_candidates"] == []
    assert result["query_parts"]["unresolved_property_clauses"][0]["text"] == "không có wifa"
    assert "poi_search" not in result and repository.calls == 0


def test_intent_near_tie_offers_real_leaf_choices_without_accepting_top1():
    search, category, repository = runtime(action="reject")
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    category.policy = {"max_options": 4, "thresholds": {"e5-base-multilingual": {
        "clarify_min_score": .81, "coherent_cluster_delta": .012, "ambiguity_delta": .032,
    }}}
    category.confidence_calibration = {"action_thresholds": {
        "clarify_min_in_domain_confidence": .695, "reject_min_top_confidence": .605,
    }}
    catalog = load_property_definitions(ROOT / "data/properties/poi_properties.poc.v5.json")
    search.property_catalog = catalog
    search.property_extractor = PropertyExtractor(catalog)
    sample = {"top_id": "242", "second_id": "19", "third_id": "229", "score2": .8698, "score3": .8633,
              "domain": .994, "confidence": .732, "reason": "insufficient_calibrated_evidence"}

    def uncertain_query(text, **kwargs):
        category.queries.append(text)
        return {
            "prediction": {"query": text, "hits": [
                {"item_id": sample["top_id"], "score": .8719}, {"item_id": sample["second_id"], "score": sample["score2"]},
                {"item_id": sample["third_id"], "score": sample["score3"]},
            ]},
            "decision": {
                "action": "reject", "reason": sample["reason"], "accepted_item_id": None,
                "calibrated_confidence": sample["confidence"],
                "in_domain_confidence": sample["domain"], "ambiguity_score": .024,
            },
            "runtime": {},
        }

    category.query = uncertain_query
    query = "trạm sạc có sạc nhanh, nhà vệ sinh"
    result = search.analyze_intent(query)
    decision = result["intent"]["decision"]
    assert category.queries == ["trạm sạc"]
    assert decision["action"] == "clarify"
    assert decision["reason"] == "near_tied_category_candidates"
    assert decision["accepted_item_id"] is None
    assert [option["item_id"] for option in decision["options"]] == ["242", "19", "229"]
    assert [option["score"] for option in decision["options"]] == [.8719, .8698, .8633]
    assert decision["calibrated_confidence"] == .732
    assert decision["ambiguity_score"] == .024
    resolution = result["intent"]["category_resolution"]
    assert resolution["original_semantic_decision"]["reason"] == "insufficient_calibrated_evidence"
    assert resolution["source"] == "near_tie_clarification"
    assert {row["property_id"] for row in result["query_parts"]["filter_candidates"]} == {"fast_charging", "toilets"}
    assert "poi_search" not in result and repository.calls == 0

    selected = search.resolve_selection_and_analyze_intent(
        query=query, selected_id="242", allowed_option_ids=[option["item_id"] for option in decision["options"]],
    )
    assert selected["intent"]["decision"]["accepted_item_id"] == "242"
    assert selected["query_parts"]["pgvector_queries"] == []
    assert {row["property_id"] for row in selected["property_extraction"]["properties"]} == {"fast_charging", "toilets"}
    assert repository.calls == 0

    sample["score2"] = .83
    sample["score3"] = .82
    assert search.analyze_intent(query)["intent"]["decision"]["action"] == "reject"
    sample["score2"] = .8698
    sample["score3"] = .8633
    sample["domain"] = .64
    assert search.analyze_intent(query)["intent"]["decision"]["action"] == "reject"
    sample["domain"] = .994
    sample["reason"] = "low_calibrated_confidence"
    assert search.analyze_intent(query)["intent"]["decision"]["action"] == "reject"
    sample["reason"] = "insufficient_calibrated_evidence"
    assert search.analyze_intent("cách mở trạm sạc")["intent"]["decision"]["reason"] == "informational_not_place_search"
    assert search.analyze_intent("không phải trạm sạc")["intent"]["decision"]["reason"] == "negated_place_only"
    assert repository.calls == 0
    sample.update({"top_id": "34", "second_id": "154", "third_id": "242", "score3": .82})
    unrelated = search.analyze_intent("Muộn rồi, tìm tiệm thuốc tây còn mở khuya giúp mình")
    assert unrelated["intent"]["query_interpretation"]["category_matches"] == []
    assert unrelated["intent"]["decision"]["action"] == "reject"
    assert not unrelated["intent"]["decision"].get("options")


def test_confirm_semantic_property_requires_explicit_valid_candidate_and_mode():
    search, _category, repository = runtime()
    query = "quán cà phê có mát rượi"
    result = search.analyze_intent(query)
    assert result["property_extraction"]["properties"] == []
    span = {"text": "có mát rượi", "start_offset": query.index("có mát rượi"), "end_offset": len(query)}
    result["query_parts"]["semantic_property_suggestions"] = [{
        "span": span, "query_text": "mát rượi", "status": "needs_confirmation",
        "candidates": [{"property_id": "air_conditioning", "cosine": .82, "applicable": True}],
    }]
    with pytest.raises(ValueError, match="not offered"):
        search.confirm_semantic_property(result, span_index=0, property_id="parking", mode="required")
    with pytest.raises(ValueError, match="choose"):
        search.confirm_semantic_property(result, span_index=0, property_id="air_conditioning", mode="invalid")
    assert result["query_parts"]["filter_candidates"] == []
    confirmed = search.confirm_semantic_property(
        result, span_index=0, property_id="air_conditioning", mode="excluded",
    )
    row = confirmed["query_parts"]["filter_candidates"][0]
    assert (row["property_id"], row["mode"], row["value"]) == ("air_conditioning", "excluded", True)
    assert row["evidence_text"] == query[span["start_offset"]:span["end_offset"]]
    assert row["cue"] == "semantic_user_confirmed" and row["confidence"] == 0.0
    assert confirmed["query_parts"]["confirmed_semantic_choices"][0]["mode"] == "excluded"
    assert result["query_parts"]["semantic_property_suggestions"][0]["status"] == "needs_confirmation"
    assert "poi_search" not in confirmed and repository.calls == 0
    with pytest.raises(ValueError, match="not offered"):
        search.confirm_semantic_property(confirmed, span_index=0, property_id="air_conditioning", mode="required")


def test_semantic_selection_respects_category_applicability():
    from model_intent.poi_properties import load_property_profiles

    search, category, _repository = runtime()
    profiles = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v2.json", category.taxonomy)
    search.property_profiles = profiles
    search.property_catalog = profiles.definitions
    search.property_extractor = PropertyExtractor(profiles.definitions)
    query = "quán cà phê có mát rượi"
    result = search.analyze_intent(query)
    result["query_parts"]["semantic_property_suggestions"] = [{
        "span": {"text": "có mát rượi", "start_offset": 12, "end_offset": len(query)},
        "status": "needs_confirmation",
        "candidates": [{"property_id": "wheelchair", "cosine": .85, "applicable": False}],
    }]
    with pytest.raises(ValueError, match="inapplicable"):
        search.confirm_semantic_property(result, span_index=0, property_id="wheelchair", mode="required")


def test_intent_analysis_exposes_properties_before_clarification_and_after_selection():
    search, _category, repository = runtime(action="reject")
    query = "quán cà phê hoặc nhà thuốc có máy lạnh"
    result = search.analyze_intent(query)
    assert result["intent"]["decision"]["action"] == "clarify"
    assert any(row["property_id"] == "air_conditioning" for row in result["property_extraction"]["properties"])
    allowed = [option["item_id"] for option in result["intent"]["decision"]["options"]]
    selected = search.resolve_selection_and_analyze_intent(
        query=query, selected_id="23", allowed_option_ids=allowed,
    )
    assert selected["intent"]["decision"]["accepted_item_id"] == "23"
    assessment = selected["intent"]["confidence_assessment"]
    assert assessment["final_decision_source"] == "explicit_user_selection"
    assert assessment["semantic_top_item_id"] is None
    assert assessment["quality"] == "none"
    assert assessment["is_probability_of_final_category"] is False
    assert selected["query_parts"]["pgvector_queries"] == []
    assert selected["query_parts"]["vector_query_mode"] == "user_selection"
    assert selected["query_parts"]["filter_candidates"]
    assert any(row["property_id"] == "air_conditioning" for row in selected["property_extraction"]["properties"])
    assert "poi_search" not in selected
    assert repository.calls == 0


def test_intent_analysis_marks_inapplicable_properties_without_poi_lookup():
    from model_intent.poi_properties import load_property_profiles

    search, category, repository = runtime()
    category.category_id = "114"
    profiles = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v2.json", category.taxonomy)
    search.property_profiles = replace(profiles, minimum_known=0)
    search.property_catalog = profiles.definitions
    search.property_extractor = PropertyExtractor(profiles.definitions)
    result = search.analyze_intent("điểm dừng xe buýt có máy lạnh")
    assert result["intent"]["decision"]["accepted_item_id"] == "114"
    assert result["property_extraction"]["properties"][0]["ignored_reason"] == "category_inapplicable"
    assert "poi_search" not in result
    assert repository.calls == 0


def test_intent_only_runtime_rejects_poi_search():
    search, _category, _repository = runtime()
    result = search.analyze_intent("quán cà phê")
    assert result["intent"]["decision"]["action"] == "accept"
    assert "poi_search" not in result
    assert not hasattr(search, "search")
    assert not hasattr(search, "resolve_selection_and_search")
    search.close()


def test_intent_factory_uses_property_profiles_without_poi_generation(monkeypatch):
    import model_intent.search_runtime as module

    category = CategoryStub()
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:fake@127.0.0.1:5433/model_intent_unified")
    monkeypatch.setattr(module.MatcherRuntime, "from_config", lambda **kwargs: category)
    monkeypatch.setattr(module.SearchRuntime, "_validate_e5_category_runtime", lambda *args: None)

    analysis = SearchRuntime.from_intent_config(
        config_path=ROOT / "configs/benchmark.poi.pg.v3.yaml",
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v3.json",
        model_key="e5-base-multilingual",
    )
    assert not hasattr(analysis, "poi_repository")
    assert analysis.property_profiles is not None
    assert analysis.analyze_intent("quán cà phê có máy lạnh")["property_extraction"]["properties"]
    analysis.close()


def test_intent_factory_closes_category_database_on_property_error(monkeypatch, tmp_path):
    import model_intent.search_runtime as module

    category = CategoryStub()
    closed = []
    category.database = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(module.MatcherRuntime, "from_config", lambda **kwargs: category)
    monkeypatch.setattr(module.SearchRuntime, "_validate_e5_category_runtime", lambda *args: None)
    monkeypatch.setattr(module, "validate_database_target", lambda config: None)
    config_path = tmp_path / "configs/benchmark.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("property_intent:\n  property_path: missing.json\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Property catalog"):
        SearchRuntime.from_intent_config(
            config_path=config_path,
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
            model_key="e5-base-multilingual",
        )
    assert closed == [True]


def test_digital_property_after_explicit_place_does_not_veto_category():
    search, category, _repository = runtime(action="accept")
    category.model_key = "e5-base-multilingual"
    category.category_id = "34"
    allowed = search.analyze_intent("tìm nhà thuốc có app giao hàng")
    assert allowed["intent"]["decision"]["action"] == "accept"
    assert allowed["intent"]["decision"]["accepted_item_id"] == "34"
    blocked = search.analyze_intent("tìm phần mềm quản lý nhà thuốc")
    assert blocked["intent"]["decision"]["action"] == "reject"
    assert blocked["intent"]["decision"]["reason"] == "digital_product_not_place_search"


def test_anchored_place_name_never_promotes_near_tied_categories():
    search, category, _repository = runtime(action="reject")
    category.model_key = "e5-base-multilingual"

    def close_candidates(text, **kwargs):
        return {
            "prediction": {"query": text, "hits": [
                {"item_id": "23", "score": 0.790}, {"item_id": "34", "score": 0.786},
            ]},
            "decision": {"action": "reject", "reason": "low_calibrated_confidence",
                         "accepted_item_id": None},
            "runtime": {},
        }

    category.query = close_candidates
    result = search.analyze_intent("Tui kiếm quán cà phê ở Thủ Đức")
    assert result["intent"]["decision"]["action"] == "reject"
    assert result["intent"]["category_resolution"]["source"] == "semantic_model"


def test_real_model_anchored_place_name_rescues_rejected_correct_top1_without_changing_score():
    search, category, repository = runtime(action="reject")
    category.model_key = "e5-base-multilingual"

    def low_confidence(text, **kwargs):
        return {
            "prediction": {"query": text, "hits": [{"item_id": "23", "score": 0.79}]},
            "decision": {"action": "reject", "reason": "low_calibrated_confidence",
                         "accepted_item_id": None, "calibrated_confidence": 0.25,
                         "in_domain_confidence": 0.42},
            "runtime": {},
        }

    category.query = low_confidence
    result = search.analyze_intent("Tui kiếm quán cà phê ở Thủ Đức")
    assert result["intent"]["decision"]["accepted_item_id"] == "23"
    assert result["intent"]["decision"]["reason"] == "anchored_place_name_evidence"
    assert result["intent"]["category_resolution"]["semantic_decision"]["action"] == "reject"
    assert result["intent"]["decision"]["calibrated_confidence"] == 0.25
    assert repository.calls == 0

    bare_seeking = search.analyze_intent("Tui kiếm quán cà phê tư vấn pha chế")
    assert bare_seeking["intent"]["decision"]["action"] == "reject"
    negated = search.analyze_intent("Không tìm quán cà phê")
    assert negated["intent"]["decision"]["action"] == "reject"
    negated_prefixed = search.analyze_intent("Mình không muốn tìm quán cà phê")
    assert negated_prefixed["intent"]["decision"]["action"] == "reject"
    informational = search.analyze_intent("Tui không tìm quán cà phê; muốn hỏi cách tự pha cà phê")
    assert informational["intent"]["decision"]["action"] == "reject"


def test_curated_hospital_alias_protects_parking_as_required_property():
    search, category, repository = runtime(action="reject")
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    search.query_aliases = CategoryQueryAliases(
        category.taxonomy, category.enrichment,
        overrides_path=ROOT / "data/taxonomies/category_query_aliases.poc.v3.json",
    )
    search.property_catalog = load_property_definitions(ROOT / "data/properties/poi_properties.poc.v5.json")
    search.property_extractor = PropertyExtractor(search.property_catalog)

    def query(text, **kwargs):
        category.queries.append(text)
        canonical = text == "Bệnh viện"
        return {
            "prediction": {"query": text, "hits": [{"item_id": "42" if canonical else "86", "score": 0.86}]},
            "decision": {"action": "accept" if canonical else "reject",
                         "reason": "calibrated_high_confidence" if canonical else "low_calibrated_confidence",
                         "accepted_item_id": "42" if canonical else None},
            "runtime": {},
        }

    category.query = query
    result = search.analyze_intent("nhà thương có bãi đỗ xe")
    assert result["intent"]["decision"]["accepted_item_id"] == "42"
    assert result["intent"]["query_interpretation"]["category_text"] == "nhà thương"
    assert result["property_extraction"]["properties"][0]["property_id"] == "parking"
    assert result["property_extraction"]["properties"][0]["applied"] is True
    assert repository.calls == 0


def test_unique_reviewed_alias_requires_canonical_e5_confirmation_and_preserves_original_score():
    search, category, repository = runtime(action="reject")
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    search.query_aliases = CategoryQueryAliases(
        category.taxonomy, category.enrichment,
        overrides_path=ROOT / "data/taxonomies/category_query_aliases.poc.v3.json",
    )

    def query(text, **kwargs):
        category.queries.append(text)
        is_canonical = text == "Bệnh viện"
        top_id = "42" if is_canonical else "125"
        return {
            "prediction": {"query": text, "hits": [{"item_id": top_id, "score": 0.86 if is_canonical else 0.84}]},
            "decision": {
                "action": "accept" if is_canonical else "reject",
                "reason": "calibrated_high_confidence" if is_canonical else "low_calibrated_confidence",
                "accepted_item_id": "42" if is_canonical else None,
                "calibrated_confidence": 0.9 if is_canonical else 0.4,
            },
            "runtime": {},
        }

    category.query = query
    result = search.analyze_intent("nhà thương có máy lạnh")
    decision = result["intent"]["decision"]
    resolution = result["intent"]["category_resolution"]
    assert decision["action"] == "accept" and decision["accepted_item_id"] == "42"
    assert decision["calibrated_confidence"] == 0.9
    assert resolution["source"] == "alias_canonical_probe"
    assert resolution["original_semantic_top_id"] == "125"
    assert resolution["original_semantic_decision"]["calibrated_confidence"] == 0.4
    assert result["intent"]["confidence_assessment"]["semantic_variant_query"] == "Bệnh viện"
    assert [row["property_id"] for row in result["property_extraction"]["properties"]] == ["air_conditioning"]
    assert category.queries == ["nhà thương", "Bệnh viện"]
    assert result["query_parts"]["pgvector_queries"] == ["nhà thương", "Bệnh viện"]
    assert result["query_parts"]["category_text"] == "nhà thương"
    assert repository.calls == 0

    negated = search.analyze_intent("không phải nhà thương")
    assert negated["intent"]["decision"]["action"] == "reject"
    assert negated["intent"]["category_resolution"].get("alias_probe") is None
    commercial = search.analyze_intent("nhà thương mại")
    assert commercial["intent"]["decision"]["action"] == "reject"
    assert commercial["intent"]["category_resolution"].get("alias_probe") is None


def test_unique_reviewed_alias_does_not_override_conflicting_semantic_accept():
    search, category, repository = runtime(action="accept")
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    search.query_aliases = CategoryQueryAliases(
        category.taxonomy, category.enrichment,
        overrides_path=ROOT / "data/taxonomies/category_query_aliases.poc.v3.json",
    )

    def query(text, **kwargs):
        is_canonical = text == "Bệnh viện"
        item_id = "42" if is_canonical else "125"
        return {
            "prediction": {"query": text, "hits": [{"item_id": item_id, "score": 0.86}]},
            "decision": {"action": "accept", "reason": "calibrated_high_confidence", "accepted_item_id": item_id},
            "runtime": {},
        }

    category.query = query
    result = search.analyze_intent("nhà thương")
    decision = result["intent"]["decision"]
    assert decision["action"] == "clarify" and decision["reason"] == "alias_semantic_conflict"
    assert {str(option["item_id"]) for option in decision["options"]} == {"42", "125"}
    assert repository.calls == 0


def test_curated_category_selection_preserves_unresolved_property_span():
    search, category, _ = runtime(action="reject")
    category.taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    category.enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    category.model_key = "e5-base-multilingual"
    catalog = load_property_definitions(ROOT / "data/properties/poi_properties.poc.v8.json")
    search.property_catalog = catalog
    search.property_extractor = PropertyExtractor(catalog)
    search.query_aliases = CategoryQueryAliases(
        category.taxonomy, category.enrichment,
        overrides_path=ROOT / "data/taxonomies/category_query_aliases.poc.v3.json",
    )
    query = "nhà thương có mát rượi"
    original = search.analyze_intent(query)
    selected = search.resolve_selection_and_analyze_intent(
        query=query, selected_id="42", allowed_option_ids=["42"],
    )
    spans = original["query_parts"]["unresolved_property_clauses"]
    assert spans and spans == selected["query_parts"]["unresolved_property_clauses"]
    assert spans[0]["text"] == "có mát rượi"
