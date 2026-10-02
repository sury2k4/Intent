from pathlib import Path

import pytest

from model_intent.osm_ingest import load_property_catalog
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


def test_ambiguous_category_is_not_silently_chosen(parts):
    result = interpret("quán cà phê hoặc nhà thuốc", parts)
    assert result.variants == ("quán cà phê hoặc nhà thuốc",)
    assert {match.item_id for match in result.category_matches} == {"23", "34"}
