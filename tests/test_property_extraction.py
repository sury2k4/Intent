from pathlib import Path
import unicodedata

import pytest

from model_intent.osm_ingest import load_property_catalog
from model_intent.property_extraction import PropertyExtractor, active_constraints, reconcile_for_category
from model_intent.poi_properties import load_property_profiles
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def extractor():
    return PropertyExtractor(load_property_catalog(ROOT / "data/properties/poi_properties.poc.v1.json"))


def test_extracts_category_properties_separately_in_vietnamese():
    result = extractor().extract("quán cà phê thoáng mát có máy lạnh")
    assert [(row.property_id, row.mode) for row in result.properties] == [
        ("airy", "required"),
        ("air_conditioning", "required"),
    ]
    assert all(row.applied for row in result.properties)


def test_accentless_aliases_and_preference_and_exclusion():
    result = extractor().extract("quan cafe uu tien wifi, khong co cho ngoi ngoai troi")
    assert [(row.property_id, row.mode) for row in result.properties] == [
        ("internet_access", "preferred"),
        ("outdoor_seating", "excluded"),
    ]
    assert all(row.applied for row in result.properties)


def test_without_a_preference_is_not_misread_as_an_exclusion():
    result = extractor().extract("quan cafe khong can wifi")
    row = result.properties[0]
    assert row.property_id == "internet_access"
    assert row.applied is False
    assert row.ignored_reason == "not_requested"
    assert active_constraints(result) == ()


def test_english_property_extraction():
    result = extractor().extract("air-conditioned cafe with outdoor seating")
    assert {(row.property_id, row.mode) for row in result.properties} == {
        ("air_conditioning", "required"),
        ("outdoor_seating", "required"),
    }


def test_numeric_rating_and_radius_are_structured():
    result = extractor().extract("cafe rating từ 4.5 trong 2 km")
    assert [(row.property_id, row.value, row.operator) for row in result.properties] == [
        ("rating", 4.5, "gte"),
        ("radius_km", 2.0, "lte"),
    ]


def test_category_inapplicable_property_is_not_applied():
    result = extractor().extract("quán cà phê có wifi")
    reconciled = reconcile_for_category(
        result,
        category_code="pharmacy",
        property_catalog=extractor().property_catalog,
    )
    assert reconciled.properties[0].applied is False
    assert reconciled.properties[0].ignored_reason == "category_inapplicable"


def test_v2_category_profile_controls_query_applicability():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    profiles = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v2.json", taxonomy)
    extractor = PropertyExtractor(profiles.definitions)
    parsed = extractor.extract("điểm dừng xe buýt có mái che và máy lạnh")
    reconciled = reconcile_for_category(
        parsed,
        category_code="bus_stop",
        category_id="114",
        taxonomy=taxonomy,
        profiles=profiles,
        property_catalog=profiles.definitions,
    )
    assert {item.property_id for item in active_constraints(reconciled)} == {"covered_area"}
    assert any(item.property_id == "air_conditioning" and item.ignored_reason == "category_inapplicable" for item in reconciled.properties)


def test_zero_minimum_profiles_still_reconcile_category_applicability():
    from dataclasses import replace

    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    profiles = replace(load_property_profiles(ROOT / "data/properties/poi_properties.poc.v2.json", taxonomy), minimum_known=0)
    parsed = PropertyExtractor(profiles.definitions).extract("điểm dừng xe buýt có mái che và máy lạnh")
    reconciled = reconcile_for_category(
        parsed, category_code="bus_stop", category_id="114", taxonomy=taxonomy,
        profiles=profiles, property_catalog=profiles.definitions,
    )
    assert {row.property_id for row in active_constraints(reconciled)} == {"covered_area"}
    assert any(row.property_id == "air_conditioning" and row.ignored_reason == "category_inapplicable"
               for row in reconciled.properties)


def test_enum_and_unspecified_string_do_not_become_broken_hard_filters():
    parsed = extractor().extract("quán cà phê cho xe lăn và có món ăn")
    wheelchair = next(item for item in parsed.properties if item.property_id == "wheelchair")
    cuisine = next(item for item in parsed.properties if item.property_id == "cuisine")
    assert wheelchair.value == "yes" and wheelchair.applied
    assert not cuisine.applied and cuisine.ignored_reason == "value_unspecified"


@pytest.mark.parametrize("query", ["hoặc", "pharmacy", "các nhà thuốc", "bác sĩ"])
def test_short_ac_alias_does_not_match_inside_other_words(query):
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v2.json")
    assert all(row.property_id != "air_conditioning" for row in PropertyExtractor(catalog).extract(query).properties)


@pytest.mark.parametrize("query, expected", [
    ("coffee with no wifi", [("internet_access", "excluded")]),
    ("công viên có bóng mát và ghế ngồi", [("shade", "required"), ("seating", "required")]),
    ("công viên không có ghế ngồi", [("seating", "excluded")]),
    ("công viên nếu có ghế ngồi", [("seating", "preferred")]),
    ("quán cà phê không có wifi, có máy lạnh", [("internet_access", "excluded"), ("air_conditioning", "required")]),
    ("quán cà phê không cần wifi nhưng có máy lạnh", [("internet_access", "preferred"), ("air_conditioning", "required")]),
    ("điểm dừng xe buýt có mái che và ghế ngồi", [("covered_area", "required"), ("seating", "required")]),
    ("ATM rút tiền có mái che", [("cash_withdrawal", "required"), ("covered_area", "required")]),
])
def test_queried_cues_and_properties_are_scoped(query, expected):
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v2.json")
    result = PropertyExtractor(catalog).extract(query)
    assert [(row.property_id, row.mode) for row in result.properties] == expected
    assert all(query[row.start_offset:row.end_offset] == row.evidence_text for row in result.properties)


def test_decomposed_vietnamese_preserves_original_evidence_offsets():
    query = unicodedata.normalize("NFD", "quán cà phê có máy lạnh")
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v2.json")
    result = PropertyExtractor(catalog).extract(query)
    assert len(result.properties) == 1
    row = result.properties[0]
    assert row.property_id == "air_conditioning"
    assert query[row.start_offset:row.end_offset] == row.evidence_text
    assert unicodedata.normalize("NFC", row.evidence_text) == "máy lạnh"


def test_gym_bench_exercise_is_not_seating_but_park_benches_are():
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v3.json")
    extractor = PropertyExtractor(catalog)
    assert all(row.property_id != "seating" for row in extractor.extract("phòng gym squat bench deadlift").properties)
    assert any(row.property_id == "seating" for row in extractor.extract("công viên có ghế đá").properties)


def test_duplicate_aliases_do_not_create_duplicate_constraints():
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v3.json")
    rows = PropertyExtractor(catalog).extract("bảo tàng có thuyết minh tự động audio guide").properties
    assert sum(row.property_id == "audio_guide" for row in rows) == 1


def test_rating_strict_and_inclusive_comparators():
    rows = extractor().extract("cafe rating above 4.5").properties
    assert [(row.property_id, row.operator, row.value) for row in rows] == [("rating", "gt", 4.5)]
    rows = extractor().extract("cafe rating từ 4.5").properties
    assert [(row.property_id, row.operator, row.value) for row in rows] == [("rating", "gte", 4.5)]
