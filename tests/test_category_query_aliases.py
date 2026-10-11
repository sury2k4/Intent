import json
from pathlib import Path

import pytest

from model_intent.category_query_aliases import CategoryQueryAliases
from model_intent.normalization import fold_vietnamese, normalize_text
from model_intent.rendering import load_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]
TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
ENRICHMENT = ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json"
OVERRIDES = ROOT / "data/taxonomies/category_query_aliases.poc.v3.json"


def catalog():
    taxonomy = Taxonomy.load(TAXONOMY)
    enrichment = load_enrichment(ENRICHMENT)
    return taxonomy, enrichment, CategoryQueryAliases(taxonomy, enrichment, overrides_path=OVERRIDES)


def test_unique_existing_names_and_aliases_across_every_v3_leaf():
    taxonomy, enrichment, aliases = catalog()
    assert sum(not taxonomy.has_children(item_id) for item_id in taxonomy.items) == 384
    for item_id, item in taxonomy.items.items():
        if taxonomy.has_children(item_id):
            continue
        phrases = (
            *item.names.values(),
            *(enrichment[item_id].get("aliases_vi") or ()),
            *(enrichment[item_id].get("aliases_en") or ()),
        )
        for phrase in phrases:
            normalized = normalize_text(fold_vietnamese(phrase))
            matched = aliases.match_whole_phrase(phrase, include_canonical=True)
            if (len(normalized) >= 4 or normalized == "atm") and aliases.owners.get(normalized) == item_id:
                assert matched is not None and matched.category_id == item_id
            else:
                assert matched is None


@pytest.mark.parametrize("query,expected", [
    ("nhà thương", "42"), ("nha thuong", "42"),
    ("tìm nhà thương gần đây", "42"),
    ("tiệm thuốc tây", "34"), ("nhà băng", "243"), ("phi trường", "1"), ("tiệm hớt tóc", "40"),
    ("chemist", "34"), ("public house", "26"), ("upper-secondary school", "46"),
    ("trường trung học phổ thông", "46"),
])
def test_reviewed_alias_has_one_leaf_and_canonical_query(query, expected):
    _taxonomy, _enrichment, aliases = catalog()
    candidate = aliases.match_whole_phrase(query)
    assert candidate is not None
    assert candidate.category_id == expected
    assert candidate.canonical_name


@pytest.mark.parametrize("query", [
    "nhà thương mại", "cách mở nhà thương", "không phải nhà thương",
    "nhà thương hay nhà thuốc", "shoe store", "trạm nghỉ",
])
def test_unrelated_or_ambiguous_phrases_have_no_single_alias(query):
    _taxonomy, _enrichment, aliases = catalog()
    assert aliases.match_whole_phrase(query) is None


def test_explicit_alternatives_remain_ambiguous():
    _taxonomy, _enrichment, aliases = catalog()
    assert {row.category_id for row in aliases.match_alternatives("nhà thương hoặc nhà thuốc")} == {"42", "34"}
    assert {row.category_id for row in aliases.match_alternatives("nhà thương và nhà thuốc")} == {"42", "34"}
    assert {row.category_id for row in aliases.match_alternatives("ATM hoặc nhà băng")} == {"52", "243"}
    assert aliases.match_alternatives("nhà thương mại hoặc nhà thuốc") == ()


def test_leading_curated_alias_protects_property_category_overlap():
    _taxonomy, _enrichment, aliases = catalog()
    match = aliases.leading_curated_match("nhà thương có bãi đỗ xe")
    assert match is not None and match.item_id == "42" and match.evidence == "nhà thương"
    assert aliases.leading_curated_match("nhà thương mại có bãi đỗ xe") is None
    assert aliases.leading_curated_match("không phải nhà thương có bãi đỗ xe") is None


def test_reviewed_alias_cannot_collide_with_another_leaf(tmp_path):
    taxonomy, enrichment, _aliases = catalog()
    payload = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    payload["aliases"].append({
        "category_id": "34", "language": "vi", "phrase": "bệnh viện", "evidence": "invalid overlap",
    })
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="collides with another category"):
        CategoryQueryAliases(taxonomy, enrichment, overrides_path=path)


def test_override_manifest_is_version_pinned(tmp_path):
    taxonomy, enrichment, _aliases = catalog()
    payload = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    payload["taxonomy_version"] = "unrelated"
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="active taxonomy"):
        CategoryQueryAliases(taxonomy, enrichment, overrides_path=path)
