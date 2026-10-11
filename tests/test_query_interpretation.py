from pathlib import Path
import unicodedata

import pytest

from model_intent.poi_properties import load_property_definitions as load_property_catalog
from model_intent.property_extraction import PropertyExtractor
from model_intent.query_interpretation import interpret_query
from model_intent.rendering import load_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def parts():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    extractor = PropertyExtractor(load_property_catalog(ROOT / "data/properties/poi_properties.poc.v2.json"))
    return taxonomy, enrichment, extractor


def interpret(text, parts):
    taxonomy, enrichment, extractor = parts
    return interpret_query(text, extractor.extract(text), taxonomy, enrichment)


def test_charging_modifier_stays_in_category_text(parts):
    text = "trạm sạc nhanh ,có nhà vệ sinh"
    result = interpret(text, parts)
    assert result.category_text == "trạm sạc nhanh"
    assert result.variants == ("trạm sạc nhanh", text)
    assert [(row["property_id"], row["role"]) for row in result.property_roles] == [
        ("fast_charging", "dual_role_category_modifier"),
        ("toilets", "constraint"),
    ]
    assert {row.property_id for row in result.extraction.properties if row.applied} == {"fast_charging", "toilets"}


@pytest.mark.parametrize("text, category_id, property_id", [
    ("bãi đỗ xe", "86", "parking"),
    ("máy rút tiền", "52", "cash_withdrawal"),
    ("nhà vệ sinh công cộng", "240", "toilets"),
])
def test_category_words_are_not_turned_into_requirements(text, category_id, property_id, parts):
    result = interpret(text, parts)
    assert result.category_text == text
    assert [match.item_id for match in result.category_matches] == [category_id]
    row = next(row for row in result.extraction.properties if row.property_id == property_id)
    assert not row.applied and row.ignored_reason == "category_component"


def test_toilet_phrase_is_a_property_when_cafe_is_target(parts):
    result = interpret("quán cà phê có nhà vệ sinh công cộng", parts)
    assert result.category_text == "quán cà phê"
    assert [match.item_id for match in result.category_matches] == ["23"]
    assert [(row.property_id, row.applied) for row in result.extraction.properties] == [("toilets", True)]


@pytest.mark.parametrize("text, category_text, key", [
    ("nhà hàng Nhật có máy lạnh", "nhà hàng Nhật", "air_conditioning"),
    ("coffee with parking", "coffee", "parking"),
    ("quán cà phê có máy lạnh, ưu tiên thoáng mát", "quán cà phê", "air_conditioning"),
    ("ATM rút tiền có mái che", "ATM", "cash_withdrawal"),
])
def test_category_target_survives_removing_only_constraints(text, category_text, key, parts):
    result = interpret(text, parts)
    assert result.category_text == category_text
    assert any(row.property_id == key and row.applied for row in result.extraction.properties)
    assert result.focused_is_usable


def test_optional_absence_of_parking_is_property_not_second_category(parts):
    result = interpret("Motel without a car park, if possible", parts)
    parking = next(row for row in result.extraction.properties if row.property_id == "parking")
    assert parking.mode == "preferred" and parking.value is False and parking.applied
    assert "parking" not in {match.item_id for match in result.category_matches}


def test_property_only_query_has_no_category_text(parts):
    result = interpret("có máy lạnh", parts)
    assert not result.focused_is_usable
    assert result.focused_reason == "no_category_text_after_property_removal"
    assert [row.property_id for row in result.extraction.properties] == ["air_conditioning"]
    unknown = interpret("có mát rượi", parts)
    assert unknown.focused_reason == "no_category_text_after_property_removal"
    assert unknown.unresolved_property_clauses[0]["text"] == "có mát rượi"
    for place_request in (
        "Có dãy nhà phố vừa ở vừa kinh doanh nào không?",
        "Có nhà thương nào gần đây không, tôi cần đưa người nhà tới.",
        "Không cần phòng khám thú y, tìm cửa hàng bán đồ cho mèo.",
    ):
        parsed = interpret(place_request, parts)
        assert parsed.category_boundary is None
        assert parsed.focused_reason != "no_category_text_after_property_removal"
    assert interpret("không phải nhà nghỉ", parts).category_boundary is None


def test_searching_for_property_named_place_keeps_category_target(parts):
    result = interpret("tìm bãi đỗ ô tô", parts)
    assert result.category_text == "tìm bãi đỗ ô tô"
    assert result.extraction.properties[0].ignored_reason == "possible_category_target"
    assert not result.extraction.properties[0].applied


def test_property_cue_splits_unknown_clause_without_inventing_a_filter(parts):
    text = "nhà nghỉ có máy giặt rệt"
    result = interpret(text, parts)
    assert result.category_text == "nhà nghỉ"
    assert not result.extraction.properties
    assert result.category_boundary["cue"] == "có"
    assert text[result.category_boundary["start_offset"]:result.category_boundary["end_offset"]] == "có"
    unresolved = result.unresolved_property_clauses
    assert len(unresolved) == 1
    assert text[unresolved[0]["start_offset"]:unresolved[0]["end_offset"]] == "có máy giặt rệt"


def test_unknown_property_spans_keep_independent_offsets(parts):
    text = "nhà nghỉ có máy giặt rệt và sân thượng riêng"
    result = interpret(text, parts)
    assert result.category_text == "nhà nghỉ"
    assert [row["text"] for row in result.unresolved_property_clauses] == [
        "có máy giặt rệt", "sân thượng riêng",
    ]
    assert all(text[row["start_offset"]:row["end_offset"]] == row["text"] for row in result.unresolved_property_clauses)
    after_known = interpret("nhà nghỉ có wifi và máy giặt rệt", parts)
    assert [row["text"] for row in after_known.unresolved_property_clauses] == ["máy giặt rệt"]
    negative = interpret("nhà nghỉ có wifi nhưng đừng có hồ sục", parts)
    assert [row["text"] for row in negative.unresolved_property_clauses] == ["đừng có hồ sục"]


def test_property_cue_does_not_remove_category_after_cue_or_explicit_alternative(parts):
    assert interpret("Có nhà nghỉ gần đây không?", parts).category_boundary is None
    alternative = interpret("nhà nghỉ có wifi hoặc khách sạn", parts)
    assert alternative.category_boundary is None
    assert "khách sạn" in alternative.category_text
    yes_no = interpret("nhà nghỉ có wifi hay không?", parts)
    assert yes_no.category_text == "nhà nghỉ"
    assert yes_no.category_boundary["cue"] == "có"


def test_property_alternatives_keep_category_text_clean(parts):
    for text in ("nhà nghỉ có wifi hoặc máy lạnh", "nhà nghỉ có wifi hoặc có máy lạnh"):
        result = interpret(text, parts)
        assert result.category_text == "nhà nghỉ"
        assert result.unresolved_property_clauses == ()
        assert {row.property_id for row in result.extraction.properties} == {"internet_access", "air_conditioning"}
    category_alternative = interpret("nhà nghỉ có wifi hoặc nhà nghĩ", parts)
    assert "nhà nghĩ" in category_alternative.category_text
    assert category_alternative.category_boundary is None
    decomposed = interpret(unicodedata.normalize("NFD", "nhà nghỉ có wifi hoặc nhà nghĩ"), parts)
    assert "nghĩ" in decomposed.category_text
    assert decomposed.category_boundary is None


def test_mixed_unicode_preserves_category_and_negation(parts):
    prefix = unicodedata.normalize("NFD", "nhà nghỉ")
    mixed = interpret(prefix + " không wifi", parts)
    assert mixed.category_text == prefix
    assert mixed.extraction.properties[0].mode == "excluded"
    text = unicodedata.normalize("NFD", "nhà nghỉ đừng có wifi")
    result = interpret(text, parts)
    assert result.category_text == unicodedata.normalize("NFD", "nhà nghỉ")
    assert result.extraction.properties[0].mode == "excluded"
    assert text[result.category_boundary["start_offset"]:result.category_boundary["end_offset"]] == result.category_boundary["cue"]


def test_ambiguous_category_is_not_silently_chosen(parts):
    result = interpret("quán cà phê hoặc nhà thuốc", parts)
    assert result.variants == ("quán cà phê hoặc nhà thuốc",)
    assert {match.item_id for match in result.category_matches} == {"23", "34"}
