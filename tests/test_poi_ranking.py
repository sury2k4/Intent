from dataclasses import replace
from pathlib import Path

import pytest

from model_intent.poi_ranking import filter_pois, load_ranking_policy, rank_pois
from model_intent.poi_schemas import ExtractedProperty, PoiRecord


def poi(poi_id, rating, reviews, properties=None, lat=10.78, lon=106.69):
    return PoiRecord(
        poi_id=poi_id,
        osm_type="node",
        osm_id=int(poi_id.split(":")[-1]),
        category_id="23",
        category_code="coffee",
        name=poi_id,
        display_name=poi_id,
        names={},
        latitude=lat,
        longitude=lon,
        location_method="node",
        mapping_rule_id="amenity-cafe",
        mapping_version="v1",
        effective_properties=properties or {},
        rating=rating,
        review_count=reviews,
    )


def test_required_property_filter_precedes_ranking():
    good = poi("osm:node:1", 4.6, 120, {"air_conditioning": {"value": True, "status": "known", "source": "mock"}})
    bad = poi("osm:node:2", 5.0, 10000, {"air_conditioning": {"value": False, "status": "known", "source": "osm"}})
    constraint = ExtractedProperty("air_conditioning", "required", value=True)
    hits, counts = rank_pois([bad, good], [constraint])
    assert [hit.poi.poi_id for hit in hits] == [good.poi_id]
    assert counts["required_rejected"] == 1


def test_preferred_property_and_bayesian_rating_are_ranked_deterministically():
    low_evidence = poi("osm:node:1", 5.0, 1, {"internet_access": {"value": True, "status": "known", "source": "osm"}})
    strong_evidence = poi("osm:node:2", 4.7, 800, {"internet_access": {"value": True, "status": "known", "source": "osm"}})
    preferred = ExtractedProperty("internet_access", "preferred", value=True)
    hits, _ = rank_pois([low_evidence, strong_evidence], [preferred])
    assert hits[0].poi.poi_id == strong_evidence.poi_id
    assert hits[0].score.final_score >= hits[1].score.final_score


def test_ordinal_ambience_constraint_compares_levels():
    medium = poi("osm:node:1", 4.5, 20, {"airy": {"value": "medium", "status": "known", "source": "mock"}})
    high = poi("osm:node:2", 4.5, 20, {"airy": {"value": "high", "status": "known", "source": "mock"}})
    required = ExtractedProperty("airy", "required", operator="gte", value="high")
    hits, counts = rank_pois([medium, high], [required])
    assert [hit.poi.poi_id for hit in hits] == [high.poi_id]
    assert counts["required_rejected"] == 1


def test_radius_filter_uses_exact_distance():
    near = poi("osm:node:1", 4.5, 20, lat=10.78, lon=106.69)
    far = poi("osm:node:2", 4.5, 20, lat=11.0, lon=106.69)
    hits, counts = rank_pois([near, far], origin=(10.78, 106.69), radius_km=2.0)
    assert [hit.poi.poi_id for hit in hits] == [near.poi_id]
    assert counts["distance_rejected"] == 1


def test_excluded_property_removes_known_true_only():
    present = poi("osm:node:1", 4.0, 1, {"outdoor_seating": {"value": True, "status": "known", "source": "osm"}})
    unknown = poi("osm:node:2", 4.1, 1, {})
    constraint = ExtractedProperty("outdoor_seating", "excluded", value=True)
    filtered, counts = filter_pois([present, unknown], [constraint])
    assert [record.poi_id for record in filtered] == [unknown.poi_id]
    assert counts["excluded_rejected"] == 1


def test_verified_only_mode_separates_mock_matches():
    fake = poi("osm:node:1", 4.9, 500, {"air_conditioning": {"value": True, "status": "known", "source": "mock"}})
    real = poi("osm:node:2", 4.8, 100, {"air_conditioning": {"value": True, "status": "known", "source": "osm"}})
    required = ExtractedProperty("air_conditioning", "required", value=True)
    policy = load_ranking_policy(Path(__file__).parents[1] / "data/policies/poi_ranking.poc.v2.json")
    demo, counts = rank_pois([fake, real], [required], policy=policy)
    verified, strict_counts = rank_pois([fake, real], [required], policy=policy, verified_only=True)
    assert counts["verified_required_count"] == 1
    assert counts["mock_supported_count"] == 1
    assert strict_counts["mock_supported_count"] == 0
    assert {hit.poi.poi_id for hit in demo} == {fake.poi_id, real.poi_id}
    assert [hit.poi.poi_id for hit in verified] == [real.poi_id]
    assert next(hit for hit in demo if hit.poi.poi_id == fake.poi_id).required_property_sources["air_conditioning"] == "mock"


def test_verified_only_exclusion_requires_verified_negative_evidence():
    confirmed_no = poi("osm:node:1", 4.2, 20, {"outdoor_seating": {"value": False, "status": "known", "source": "osm"}})
    mock_no = poi("osm:node:2", 4.2, 20, {"outdoor_seating": {"value": False, "status": "known", "source": "mock"}})
    unknown = poi("osm:node:3", 4.2, 20)
    confirmed_yes = poi("osm:node:4", 4.2, 20, {"outdoor_seating": {"value": True, "status": "known", "source": "osm"}})
    excluded = ExtractedProperty("outdoor_seating", "excluded", value=True)
    demo, demo_counts = filter_pois([confirmed_no, mock_no, unknown, confirmed_yes], [excluded])
    strict, strict_counts = filter_pois([confirmed_no, mock_no, unknown, confirmed_yes], [excluded], verified_only=True)
    assert {item.poi_id for item in demo} == {confirmed_no.poi_id, mock_no.poi_id, unknown.poi_id}
    assert [item.poi_id for item in strict] == [confirmed_no.poi_id]
    assert strict_counts["excluded_rejected"] == 3


def test_rating_gt_excludes_boundary_while_gte_includes_it():
    at_limit = poi("osm:node:1", 4.5, 50)
    above_limit = poi("osm:node:2", 4.6, 50)
    strict = ExtractedProperty("rating", "required", operator="gt", value=4.5)
    inclusive = ExtractedProperty("rating", "required", operator="gte", value=4.5)
    assert [item.poi_id for item in filter_pois([at_limit, above_limit], [strict])[0]] == [above_limit.poi_id]
    assert {item.poi_id for item in filter_pois([at_limit, above_limit], [inclusive])[0]} == {at_limit.poi_id, above_limit.poi_id}


def test_zero_bayesian_prior_is_rejected_before_no_review_venue_is_ranked(tmp_path):
    import json
    import pytest

    policy = load_ranking_policy(Path(__file__).parents[1] / "data/policies/poi_ranking.poc.v2.json")
    policy["bayesian_prior_reviews"] = 0
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        load_ranking_policy(path)


def test_osm_only_ranking_ignores_synthetic_ratings_and_uses_stable_name_tiebreak():
    policy = load_ranking_policy(Path(__file__).parents[1] / "data/policies/poi_ranking.poc.v3.json")
    assert "fallback_prior_rating" not in policy
    assert "bayesian_prior_reviews" not in policy
    lower_name = replace(poi("osm:node:2", 1.0, 1), name="Alpha", display_name="Alpha", rating_source="osm")
    higher_rating = replace(poi("osm:node:1", 5.0, 10_000), name="Zulu", display_name="Zulu", rating_source="osm")
    hits, counts = rank_pois((higher_rating, lower_name), policy=policy)
    assert [hit.poi.poi_id for hit in hits] == [lower_name.poi_id, higher_rating.poi_id]
    assert counts["category_mean_rating"] is None
    assert all(hit.score.adjusted_rating is None for hit in hits)
    assert all(hit.score.rating_score == hit.score.review_volume_score == 0 for hit in hits)
    assert hits[0].score.final_score == 0
    reranked, _ = rank_pois((lower_name, higher_rating), policy=policy)
    assert [hit.poi.poi_id for hit in hits] == [hit.poi.poi_id for hit in reranked]


def test_osm_only_enforces_verified_positive_and_negative_evidence():
    policy = load_ranking_policy(Path(__file__).parents[1] / "data/policies/poi_ranking.poc.v3.json")
    confirmed = replace(poi("osm:node:1", None, 0, {"seating": {"value": False, "status": "known", "source": "osm"},
                                                        "shade": {"value": True, "status": "known", "source": "osm"}}), rating_source="unknown")
    unknown = replace(poi("osm:node:2", None, 0), rating_source="unknown")
    mock = poi("osm:node:3", 5.0, 10_000, {"seating": {"value": False, "status": "known", "source": "mock"},
                                            "shade": {"value": True, "status": "known", "source": "mock"}})
    with pytest.raises(ValueError, match="mock or unverified"):
        rank_pois((confirmed, unknown, mock), policy=policy)
    filtered, counts = rank_pois((confirmed, unknown),
                                 (ExtractedProperty("seating", "excluded", value=True),), policy=policy)
    assert [hit.poi.poi_id for hit in filtered] == [confirmed.poi_id]
    assert counts["excluded_rejected"] == 1
    preferred, _ = rank_pois((confirmed, unknown),
                             (ExtractedProperty("shade", "preferred"),), policy=policy)
    assert preferred[0].poi.poi_id == confirmed.poi_id
    assert preferred[0].score.final_score == 1.0
    assert next(hit for hit in preferred if hit.poi.poi_id == unknown.poi_id).score.final_score == 0.0


def test_osm_only_preferred_rating_never_influences_score():
    policy = load_ranking_policy(Path(__file__).parents[1] / "data/policies/poi_ranking.poc.v3.json")
    real_rating = replace(poi("osm:node:1", 5.0, 100), rating_source="osm")
    hits, _ = rank_pois((real_rating,), (ExtractedProperty("rating", "preferred", value=5.0),), policy=policy)
    assert hits[0].score.final_score == 0.0
    assert hits[0].score.adjusted_rating is None


def test_osm_only_policy_rejects_nonzero_rating_weights(tmp_path):
    import json
    import pytest

    policy = load_ranking_policy(Path(__file__).parents[1] / "data/policies/poi_ranking.poc.v3.json")
    policy["weights"]["rating"] = 0.1
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="OSM-only"):
        load_ranking_policy(path)
    with pytest.raises(ValueError, match="OSM-only"):
        rank_pois((replace(poi("osm:node:1", None, 0), rating_source="unknown"),), policy=policy)
    policy["weights"]["rating"] = 0.0
    policy["fallback_prior_rating"] = 3.8
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="must not configure synthetic rating priors"):
        load_ranking_policy(path)


def test_v2_mock_rating_influences_rank_at_reduced_weight_without_distance():
    base = poi("osm:node:1", 4.8, 100)
    better = poi("osm:node:2", 5.0, 100)
    policy = load_ranking_policy(Path(__file__).parents[1] / "data/policies/poi_ranking.poc.v2.json")
    hits, _ = rank_pois([base, better], policy=policy)
    assert hits[0].poi.poi_id == better.poi_id
    assert hits[0].score.rating_source_factor == policy["mock_rating_factor"]
    assert hits[0].score.review_source_factor == policy["mock_review_volume_factor"]
    # An unprovided origin contributes no constant distance bonus in v2.
    expected = (policy["weights"]["rating"] * hits[0].score.rating_score +
                policy["weights"]["review_volume"] * hits[0].score.review_volume_score) / (
                    policy["weights"]["rating"] + policy["weights"]["review_volume"])
    assert abs(hits[0].score.final_score - expected) < 1e-12
    real = replace(better, rating_source="osm")
    real_hits, _ = rank_pois([base, real], policy=policy)
    assert next(hit for hit in real_hits if hit.poi.poi_id == real.poi_id).score.rating_source_factor == 1.0
