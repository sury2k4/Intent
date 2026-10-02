from pathlib import Path

import pytest

from model_intent.search_explanation import hit_steps, search_steps
from model_intent.search_runtime import SearchRuntime

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/osm/hcmc-poc-v2-queryfix/pois.jsonl"


@pytest.fixture(scope="module")
def runtime():
    if not ARTIFACT.exists():
        pytest.skip("local HCMC artifact is not available")
    search = SearchRuntime.from_config(
        config_path=ROOT / "configs/benchmark.poi.yaml",
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
        model_key="fake",
        local_files_only=True,
    )
    yield search
    search.close()


def test_osm_only_explanation_avoids_rating_claims_and_labels_unnamed_pois():
    from model_intent.poi_ranking import load_ranking_policy, rank_pois
    from model_intent.poi_schemas import PoiRecord

    policy = load_ranking_policy(ROOT / "data/policies/poi_ranking.poc.v3.json")
    poi = PoiRecord(
        poi_id="osm:node:12", osm_type="node", osm_id=12, category_id="23", category_code="coffee",
        name=None, display_name="[Generated OSM label] coffee (osm:node:12)", names={}, latitude=10.78,
        longitude=106.69, location_method="node", mapping_rule_id="amenity-cafe", mapping_version="v3",
        rating_source="unknown",
    )
    hits, _ = rank_pois((poi,), policy=policy)
    hit = hits[0].as_dict()
    assert hit["name"] is None
    assert hit["score"]["adjusted_rating"] is None
    details = " ".join(hit_steps(hit, {"properties": []}, policy))
    assert "Tên OSM chưa có" in details
    assert "Rating và lượt review không tham gia" in details
    assert "adjusted rating 3." not in details
    assert "điểm đã trả: 0.000" in details
    steps = search_steps({"intent": {"decision": {"action": "accept"}},
                          "poi_search": {"status": "ok", "ranking_schema_version": 3,
                                         "returned_count": 1, "hard_filter_match_count": 1}})
    ranking = " ".join(steps[3]["details"])
    assert "rating tiên nghiệm" in ranking
    assert "Cùng điểm thì xét adjusted rating" not in ranking


def test_explanation_uses_recorded_category_constraints_and_counts(runtime):
    result = runtime.search("quán cà phê có máy lạnh, ưu tiên thoáng mát", poi_top_k=1)
    steps = search_steps(result)
    assert [step["title"] for step in steps] == [
        "1. Xác định loại địa điểm",
        "2. Tách yêu cầu thuộc tính",
        "3. Lọc POI",
        "4. Xếp hạng và trả kết quả",
    ]
    assert "quán cà phê" in " ".join(steps[0]["details"])
    assert "air_conditioning" in " ".join(steps[1]["details"])
    assert f"{result['poi_search']['category_candidate_count']:,}" in " ".join(steps[2]["details"])
    assert "mock" in " ".join(steps[2]["details"])


def test_per_hit_explanation_matches_actual_score_and_mock_provenance(runtime):
    result = runtime.search("quán cà phê có máy lạnh, ưu tiên thoáng mát", poi_top_k=1)
    hit = result["poi_search"]["hits"][0]
    details = hit_steps(hit, result["property_extraction"], runtime.ranking_policy)
    explanation = " ".join(details)
    assert "air_conditioning" in explanation
    assert "mock" in explanation
    assert f"điểm đã trả: {hit['score']['final_score']:.3f}" in explanation
    assert "Không có tọa độ gốc" in explanation


def test_verified_only_explanation_does_not_claim_mock_evidence(runtime):
    result = runtime.search("quán cà phê có máy lạnh", verified_only=True, poi_top_k=1)
    hit = result["poi_search"]["hits"][0]
    details = hit_steps(hit, result["property_extraction"], runtime.ranking_policy, verified_only=True)
    assert hit["required_property_sources"]["air_conditioning"] == "osm"
    assert "nguồn OSM" in " ".join(details)
    assert "Chỉ dùng bằng chứng OSM" in " ".join(search_steps(result)[2]["details"])


def test_clarify_and_reject_explain_why_poi_search_was_skipped(runtime):
    for query, action in (
        ("quán cà phê hoặc nhà thuốc", "clarify"),
        ("mai trời có mưa không", "reject"),
    ):
        result = runtime.search(query)
        assert result["intent"]["decision"]["action"] == action
        assert "Không truy vấn POI" in " ".join(search_steps(result)[2]["details"])
        assert "Chưa có POI" in " ".join(search_steps(result)[3]["details"])


def test_ignored_radius_appears_as_ignored_not_applied(runtime):
    result = runtime.search("quán cà phê trong 2 km", poi_top_k=1)
    assert "missing_origin" in " ".join(search_steps(result)[1]["details"])
    assert "radius constraint was not applied" in " ".join(search_steps(result)[2]["details"])
