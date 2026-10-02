from pathlib import Path

from model_intent.category_resolution import explicit_categories
from model_intent.rendering import load_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def resolve(query):
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    return explicit_categories(query, taxonomy, enrichment)


def test_explicit_category_is_found_in_property_rich_query():
    matches = resolve("quán cà phê thoáng mát có máy lạnh")
    assert [match.item_id for match in matches] == ["23"]


def test_multiple_distinct_place_types_remain_ambiguous():
    matches = resolve("quán cà phê và nhà thuốc")
    assert {match.item_id for match in matches} == {"23", "34"}


def test_unrelated_query_does_not_fabricate_a_category():
    assert resolve("mai trời có mưa không") == ()
    assert resolve("cửa hàng bán máy lạnh") == ()


def test_short_atm_alias_is_allowed_without_matching_other_short_words():
    assert [match.item_id for match in resolve("ATM rút tiền có mái che")] == ["52"]
    assert resolve("các") == ()


def test_location_and_negated_category_mentions_are_not_targets():
    assert [match.item_id for match in resolve("quán cà phê gần nhà thuốc")] == ["23"]
    assert [match.item_id for match in resolve("quán cà phê không phải nhà thuốc")] == ["23"]


def test_hyphenated_category_alias_matches_normalized_phrase():
    assert [match.item_id for match in resolve("fast-food")] == ["75"]
    assert [match.item_id for match in resolve("ice-cream shop")] == ["25"]


def test_explicit_category_fallback_can_skip_property_spans():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    text = "coffee with parking"
    assert [match.item_id for match in explicit_categories(
        text, taxonomy, enrichment, property_spans=((12, 19),),
    )] == ["23"]
