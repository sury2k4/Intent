import json
from pathlib import Path

import pytest

from model_intent.osm_ingest import load_mapping_catalog, mapping_coverage, match_mapping
from model_intent.poi_properties import load_property_definitions, load_property_profiles
from model_intent.rendering import load_enrichment, render_category_profiles, validate_enrichment
from model_intent.taxonomy import Taxonomy
from scripts.prepare_poi_v3_resources import (
    ENRICHMENT, EXTRA_RULES, MAPPING, NEW_CATEGORIES, PROPERTIES, REGISTRY, TAXONOMY,
    add_observed_types, resources, write_new,
)


def test_v3_resource_generator_is_additive_with_stable_numeric_ids():
    taxonomy, enrichment, mapping, properties = resources()
    old = Taxonomy.load(TAXONOMY.with_name("place_types.poc.v2.json"))
    by_id = {item["id"]: item for item in taxonomy["items"]}
    assert {item.id for item in old.document.items} <= set(by_id)
    assert sorted(int(row[0]) for row in NEW_CATEGORIES) == list(range(246, 284))
    assert len(by_id) == len(taxonomy["items"])
    assert "*" not in mapping["intentionally_unmapped"]
    assert properties["minimum_known_properties"] == 0
    assert properties["property_taxonomy_version"].endswith("osm-only")
    assert "base_catalog" not in properties and "property_overrides" not in properties
    v4_definitions = load_property_definitions(PROPERTIES.with_name("poi_properties.poc.v4.json"))
    assert {row["id"]: row for row in properties["properties"]} == v4_definitions
    old_leaves = {item_id for item_id in old.items if not old.has_children(item_id)}
    next_taxonomy = Taxonomy.from_payload(taxonomy)
    assert all(not next_taxonomy.has_children(item_id) for item_id in old_leaves)
    assert not validate_enrichment(next_taxonomy, enrichment["items"], require_rich_fields=True)
    assert render_category_profiles(next_taxonomy, enrichment=enrichment["items"], require_rich_fields=True)
    for category_id, parent_id, code, en, vi, _meaning in NEW_CATEGORIES:
        assert by_id[category_id]["parent_id"] == parent_id
        assert by_id[category_id]["code"] == code
        assert by_id[category_id]["names"] == {"en": en, "vi": vi}
        row = enrichment["items"][category_id]
        assert all(row[key] for key in ("description_en", "description_vi", "aliases_en", "aliases_vi"))
        assert "Provisional" in row["description_en"]
        assert "tạm thời" in row["description_vi"]


def test_observed_expansion_creates_tag_specific_provisional_leaf_with_stable_id():
    taxonomy, enrichment, mapping, _properties = resources()
    report = {"signatures": [
        {"signature": "shop=tyres", "count": 3, "dispositions": {"unmapped_needs_review": 3}},
        {"signature": "shop=tyres|brand=Sample", "count": 2, "dispositions": {"unmapped_needs_review": 2}},
        {"signature": "shop=one_off", "count": 1, "dispositions": {"unmapped_needs_review": 1}},
        {"signature": "amenity=yes", "count": 4, "dispositions": {"unmapped_needs_review": 4}},
    ]}
    summary = add_observed_types(report, taxonomy, enrichment, mapping)
    assert summary["added_leaf_count"] == 1
    assert summary["selected_candidate_count"] == 5
    assert summary["deferred_clear_candidate_count"] == 1
    assert taxonomy["items"][-1]["id"] == "284"
    assert taxonomy["items"][-1]["names"] == {"en": "Tyre shop", "vi": "Cửa hàng lốp xe"}
    assert "Provisional" in enrichment["items"]["284"]["description_en"]
    assert "shop=tyres" in enrichment["items"]["284"]["description_en"]
    assert mapping["rules"][-1]["match"] == {"tag": "shop", "op": "eq", "value": "tyres"}
    assert match_mapping({"shop": "one_off"}, {"rules": sorted(mapping["rules"], key=lambda rule: -rule["priority"])}) is None


def test_actual_v3_rich_enrichment_preserves_v2_leaves():
    if not TAXONOMY.exists() or not ENRICHMENT.exists():
        pytest.skip("v3 resources not generated")
    v2 = Taxonomy.load(TAXONOMY.with_name("place_types.poc.v2.json"))
    v3 = Taxonomy.load(TAXONOMY)
    assert all(not v3.has_children(item_id) for item_id in v2.items if not v2.has_children(item_id))
    rich = load_enrichment(ENRICHMENT)
    assert validate_enrichment(v3, rich, require_rich_fields=True) == []
    assert len(render_category_profiles(v3, enrichment=rich, require_rich_fields=True)) == (
        2 * sum(not v3.has_children(item_id) for item_id in v3.items)
    )


def test_v3_mapping_precedence_and_explicit_leaf_dispositions():
    if not MAPPING.exists():
        pytest.skip("v3 resources not generated")
    taxonomy = Taxonomy.load(TAXONOMY)
    catalog = load_mapping_catalog(MAPPING, taxonomy)
    assert mapping_coverage(catalog, taxonomy)["complete"]
    assert "*" not in catalog["intentionally_unmapped"]
    assert "*" not in catalog["intentionally_excluded_signatures"]
    assert match_mapping({"amenity": "charging_station", "brand": "VinFast"}, catalog)["category_id"] == "19"
    assert match_mapping({"amenity": "charging_station"}, catalog)["category_id"] == "242"
    assert match_mapping({"amenity": "restaurant", "cuisine": "japanese"}, catalog)["category_id"] == "57"
    assert match_mapping({"office": "lawyer"}, catalog)["category_id"] == "101"
    assert match_mapping({"shop": "hairdresser"}, catalog)["category_id"] == "40"
    assert match_mapping({"public_transport": "station"}, catalog)["category_id"] == "283"
    assert match_mapping({"public_transport": "station", "amenity": "ferry_terminal"}, catalog)["category_id"] == "113"
    assert match_mapping({"amenity": "place_of_worship"}, catalog)["category_id"] == "275"
    assert match_mapping({"amenity": "place_of_worship", "religion": "christian"}, catalog)["category_id"] == "140"
    assert match_mapping({"office": "hypothetical_new_kind"}, catalog) is None
    assert load_property_profiles(PROPERTIES, taxonomy).minimum_known == 0


def test_registry_is_complete_and_counts_are_accounted_for():
    if not REGISTRY.exists():
        pytest.skip("v3 registry not generated")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert registry["observed_type_expansion"]["added_leaf_count"] <= 120
    assert registry["signatures"] == sorted(registry["signatures"], key=lambda row: row["signature"])
    assert len({row["signature"] for row in registry["signatures"]}) == len(registry["signatures"])
    assert sum(row["count"] for row in registry["signatures"]) == registry["counts"]["inside_hcmc_tagged_candidates"]
    for row in registry["signatures"]:
        assert sum(row["dispositions"].values()) == row["count"]
        assert "unmapped_needs_review" not in row["dispositions"]
        if row["dispositions"].get("intentionally_excluded_ambiguous") or row["dispositions"].get("intentionally_excluded_unmodelled"):
            excluded = json.loads(MAPPING.read_text(encoding="utf-8"))["intentionally_excluded_signatures"]
            assert row["signature"] in excluded
            assert excluded[row["signature"]]["disposition"] in row["dispositions"]


def test_resource_writer_never_overwrites_an_existing_different_file(tmp_path: Path):
    target = tmp_path / "immutable.json"
    write_new(target, {"version": 1})
    write_new(target, {"version": 1})
    with pytest.raises(FileExistsError):
        write_new(target, {"version": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
