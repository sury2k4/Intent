from pathlib import Path
import unicodedata

import pytest

from model_intent.poi_properties import load_property_definitions
from model_intent.property_extraction import PropertyExtractor
from model_intent.query_interpretation import interpret_query
from model_intent.rendering import load_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def parts():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v3.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    extractor = PropertyExtractor(load_property_definitions(ROOT / "data/properties/poi_properties.poc.v8.json"))
    return taxonomy, enrichment, extractor


def interpret(query, parts):
    taxonomy, enrichment, extractor = parts
    return interpret_query(query, extractor.extract(query), taxonomy, enrichment)


@pytest.mark.parametrize("query,category,properties,sorts", [
    ("quán ăn ngon gần nhất", "quán ăn", ["quality_by_reviews", "near"], ["distance_asc", "review_score_desc"]),
    ("quán cà phê gần đây", "quán cà phê", ["near"], []),
    ("nhà hàng quanh đây", "nhà hàng", ["near"], []),
    ("quán ăn xịn", "quán ăn", ["quality_by_reviews"], ["review_score_desc"]),
    ("quán ăn có ngon", "quán ăn", ["quality_by_reviews"], ["review_score_desc"]),
    ("khách sạn 5 sao", "khách sạn", ["accommodation_star_class"], []),
    ("khách sạn đánh giá 5 sao", "khách sạn", ["rating"], []),
    ("nhà hàng 5 sao", "nhà hàng", ["ambiguous_stars"], []),
    ("5 sao", "5 sao", ["ambiguous_stars"], []),
    ("cafe rating above 4.5", "cafe", ["rating"], []),
    ("restaurant tasty nearest", "restaurant", ["quality_by_reviews", "near"], ["distance_asc", "review_score_desc"]),
])
def test_pending_properties_share_one_extraction_list(query, category, properties, sorts, parts):
    result = interpret(query, parts)
    assert result.category_text == category
    pending = [row for row in result.extraction.properties if row.status is not None]
    assert [row.property_id for row in pending] == properties
    assert [row["kind"] for row in result.requested_search["sort_candidates"]] == sorts
    for row in pending:
        assert query[row.start_offset:row.end_offset] == row.evidence_text
        assert row.status != "applied" and not row.applied
        assert row.requires
        assert row.as_dict()["status"] == row.status
        assert row.as_dict()["requires"] == list(row.requires)
    for row in result.requested_search["sort_candidates"]:
        assert query[row["start_offset"]:row["end_offset"]] == row["evidence_text"]
    assert not result.unresolved_property_clauses
    assert result.as_dict()["requested_search"] == {"sort_candidates": result.requested_search["sort_candidates"]}
    assert "filter_candidates" not in result.as_dict()["requested_search"]


def test_location_quality_and_star_requests_keep_distinct_required_data(parts):
    result = interpret("quán ăn ngon gần nhất", parts)
    quality, nearby = result.extraction.properties
    assert (quality.mode, quality.status, quality.requires) == (
        "preferred", "pending_review_data", ("verified_reviews",),
    )
    assert (nearby.mode, nearby.status, nearby.requires) == (
        "required", "pending_location", ("user_location", "poi_coordinates"),
    )
    assert [(row["kind"], row["priority"]) for row in result.requested_search["sort_candidates"]] == [
        ("distance_asc", 1), ("review_score_desc", 2),
    ]
    hotel = interpret("khách sạn 5 sao", parts).extraction.properties[0]
    review_rows = interpret("khách sạn đánh giá 5 sao", parts).extraction.properties
    ambiguous = interpret("5 sao", parts).extraction.properties[0]
    assert (hotel.property_id, hotel.operator, hotel.value) == ("accommodation_star_class", "eq", 5.0)
    assert len(review_rows) == 1
    assert (review_rows[0].property_id, review_rows[0].operator, review_rows[0].value) == ("rating", "gte", 5.0)
    assert (ambiguous.property_id, ambiguous.status) == ("ambiguous_stars", "needs_clarification")


@pytest.mark.parametrize("query", [
    "nhà hàng mới mở gần đây",
    "Nhà hàng Ngon",
    "quán Ngon cho gia đình",
    "nhà hàng ngon cho quán ăn",
    "nhà hàng không ngon",
    "khách sạn không cần 5 sao",
    "quán cafe xin wifi",
    "quán ăn đánh giá 5.5 sao",
    "cách mở nhà hàng ngon",
])
def test_no_false_pending_property(query, parts):
    assert not any(row.status for row in interpret(query, parts).extraction.properties)


def test_numeric_rating_is_parsed_but_never_applied_without_reviews(parts):
    extractor = parts[2]
    rows = extractor.extract("cafe rating above 4.5 trong 2 km").properties
    rating = next(row for row in rows if row.property_id == "rating")
    radius = next(row for row in rows if row.property_id == "radius_km")
    assert (rating.operator, rating.value, rating.applied, rating.ignored_reason) == (
        "gt", 4.5, False, "rating_unverified",
    )
    assert (radius.value, radius.applied) == (2.0, True)
    pending = interpret("cafe rating above 4.5", parts).extraction.properties
    assert len(pending) == 1
    assert (pending[0].property_id, pending[0].operator, pending[0].value) == ("rating", "gt", 4.5)
    assert pending[0].status == "pending_review_data"
    invalid = extractor.extract("cafe rating above 0.5").properties[0]
    assert invalid.ignored_reason == "rating_out_of_range" and not invalid.applied
    assert not any(row.status for row in interpret("cafe rating above 0.5", parts).extraction.properties)


def test_intent_only_queries_and_decomposed_vietnamese_preserve_evidence(parts):
    alone = interpret("gần nhất", parts)
    assert alone.focused_reason == "no_category_text_after_property_removal"
    assert [row.property_id for row in alone.extraction.properties] == ["near"]
    query = unicodedata.normalize("NFD", "quán ăn ngon gần nhất")
    result = interpret(query, parts)
    for row in result.extraction.properties:
        assert query[row.start_offset:row.end_offset] == row.evidence_text
    assert unicodedata.normalize("NFC", result.category_text) == "quán ăn"


def test_catalog_properties_still_work_beside_pending_requests(parts):
    result = interpret("quán cà phê có máy lạnh gần nhất", parts)
    assert [(row.property_id, row.applied) for row in result.extraction.properties] == [
        ("air_conditioning", True), ("near", False),
    ]
    assert "status" not in result.extraction.properties[0].as_dict()
    assert result.extraction.properties[1].as_dict()["status"] == "pending_location"
    assert result.category_text == "quán cà phê"
    assert [row["kind"] for row in result.requested_search["sort_candidates"]] == ["distance_asc"]
    assert not result.unresolved_property_clauses
