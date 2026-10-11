from pathlib import Path
import unicodedata

import pytest

from model_intent.poi_properties import load_property_definitions as load_property_catalog
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
    ("bệnh viện có lối tiếp cận xe lăn", [("wheelchair", "required")]),
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


def test_natural_user_phrases_and_negation_are_scoped_with_v5_catalog():
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v5.json")
    extractor = PropertyExtractor(catalog)
    cases = [
        ("Tìm cây xăng có xăng, nhưng đừng có dầu diesel", "fuel_diesel", "excluded", True),
        ("Chỗ nào không wifi cho mình ngồi?", "internet_access", "excluded", True),
        ("Không cần bãi đậu xe, chỉ cần nhà thuốc", "parking", "preferred", False),
        ("Wifi không cần nhưng máy lạnh phải có", "internet_access", "preferred", False),
        ("Cần phòng chờ cho người già", "waiting_area", "required", True),
        ("Phòng khám đặt lịch khám trước", "appointment", "required", True),
        ("Bank branch where I can pay by card", "accepts_cards", "required", True),
        ("Có bảng thông tin tuyến để mình xem", "route_information", "required", True),
        ("Cửa hàng có chỗ đậu xe không?", "parking", "required", True),
        ("Shop giao đến nhà", "home_delivery", "required", True),
    ]
    for query, prop_id, mode, applied in cases:
        properties = extractor.extract(query).properties
        row = next(row for row in properties if row.property_id == prop_id)
        assert row.mode == mode and row.applied is applied, (query, row)
    negative_preference = extractor.extract("Motel without a car park, if possible").properties
    parking = next(row for row in negative_preference if row.property_id == "parking")
    assert parking.mode == "preferred" and parking.value is False and parking.applied


def test_lodging_parking_and_wheelchair_cues_include_bounded_misspellings():
    parser = PropertyExtractor(load_property_catalog(ROOT / "data/properties/poi_properties.poc.v5.json"))
    for query in (
        "nhà nghĩ có bãi đỗ ô tô, không có chỗ cho xe lăng",
        "nhà nghỉ có bãi đỗ ô tô không có chỗ cho xe lăn",
        "nha nghi co bai do o to khong co cho cho xe lang",
        "Motel with car parking, no wheelchair access",
    ):
        rows = parser.extract(query).properties
        assert [(row.property_id, row.mode) for row in rows] == [
            ("parking", "required"), ("wheelchair", "excluded"),
        ], query
        assert all(row.applied and query[row.start_offset:row.end_offset] == row.evidence_text for row in rows)

    not_needed = parser.extract("nhà nghỉ có bãi đỗ ô tô, không cần chỗ cho xe lăn").properties
    assert not_needed[0].mode == "required"
    assert not_needed[1].mode == "preferred" and not_needed[1].applied is False
    assert not_needed[1].ignored_reason == "not_requested"
    wider_cue = parser.extract("nhà nghỉ không có lối đi cho xe lăn").properties
    assert [(row.property_id, row.mode) for row in wider_cue] == [("wheelchair", "excluded")]
    used = parser.extract("nhà nghỉ có dùng máy lạnh").properties
    assert [(row.property_id, row.mode) for row in used] == [("air_conditioning", "required")]
    excluded_wifi = parser.extract("nhà nghỉ đừng wifi").properties
    assert [(row.property_id, row.mode) for row in excluded_wifi] == [("internet_access", "excluded")]
    decomposed = parser.extract(unicodedata.normalize("NFD", "nhà nghỉ đừng wifi")).properties
    assert [(row.property_id, row.mode) for row in decomposed] == [("internet_access", "excluded")]
    optional_access = parser.extract("nhà nghỉ không cần lối đi cho xe lăn").properties
    assert optional_access[0].ignored_reason == "not_requested"
    assert all(row.property_id != "wheelchair" for row in parser.extract("nhà lăng tẩm có bãi đỗ xe").properties)
    motorbike_rows = parser.extract("bãi đỗ xe máy có mái che").properties
    assert all(row.matched_alias != "bãi đỗ ô tô" for row in motorbike_rows)


def test_natural_property_cues_do_not_retroactively_modify_previous_requirement():
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v5.json")
    parser = PropertyExtractor(catalog)
    cases = [
        ("Homestay có wifi, không cần bữa sáng", "internet_access", "required"),
        ("Quán cà phê cần có máy lạnh, nếu có wifi càng tốt", "air_conditioning", "required"),
        ("Bãi đỗ xe máy có mái che, không cần nhà vệ sinh", "covered_area", "required"),
        ("Quán cà phê có máy lạnh, không cần wifi", "air_conditioning", "required"),
        ("Quán cà phê có máy lạnh, không cần wifi", "internet_access", "preferred"),
    ]
    for query, prop_id, mode in cases:
        row = next(row for row in parser.extract(query).properties if row.property_id == prop_id)
        assert row.mode == mode, (query, prop_id, row.mode)


def test_rating_strict_and_inclusive_comparators():
    rows = extractor().extract("cafe rating above 4.5").properties
    assert [(row.property_id, row.operator, row.value) for row in rows] == [("rating", "gt", 4.5)]
    rows = extractor().extract("cafe rating từ 4.5").properties
    assert [(row.property_id, row.operator, row.value) for row in rows] == [("rating", "gte", 4.5)]
