from pathlib import Path
import json

from model_intent.osm_ingest import build_artifact, load_mapping_catalog, match_mapping, validate_catalogs
from model_intent.rendering import load_enrichment, validate_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def _v2():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v2.json")
    mapping = load_mapping_catalog(ROOT / "data/osm/mappings/place_types.poc.v2.json", taxonomy)
    return taxonomy, mapping


def test_new_taxonomy_is_additive_and_every_leaf_has_enrichment_and_properties():
    old = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    new, mapping = _v2()
    assert set(old.items) <= set(new.items)
    assert {item_id: new.items[item_id] for item_id in old.items} == old.items
    assert len(new.items) == 242
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v2.json")
    assert validate_enrichment(new, enrichment) == []
    catalogs = validate_catalogs(
        new,
        ROOT / "data/osm/mappings/place_types.poc.v2.json",
        ROOT / "data/properties/poi_properties.poc.v3.json",
    )
    assert catalogs["coverage"]["complete"]
    assert {"242", "243", "244", "245", "83"} <= set(catalogs["coverage"]["mapped_category_ids"])
    for item_id in ("242", "243", "244", "245"):
        profile = catalogs["profiles"].applicable_ids(new, item_id)
        assert len(profile) >= 5


def test_brand_rule_precedes_generic_ev_charging_and_battery_swap_is_not_mapped():
    _taxonomy, mapping = _v2()
    assert match_mapping({"amenity": "charging_station", "brand": "VinFast"}, mapping)["category_id"] == "19"
    assert match_mapping({"amenity": "charging_station", "brand": "Tesla"}, mapping)["category_id"] == "242"
    assert match_mapping({"amenity": "charging_station"}, mapping)["category_id"] == "242"
    assert match_mapping({"amenity": "battery_swap"}, mapping) is None


def test_bank_branch_and_atm_remain_distinct():
    _taxonomy, mapping = _v2()
    assert match_mapping({"amenity": "bank", "name": "Demo bank"}, mapping)["category_id"] == "243"
    assert match_mapping({"amenity": "atm"}, mapping)["category_id"] == "52"


def test_school_grade_specific_precedes_generic_fallback():
    _taxonomy, mapping = _v2()
    assert match_mapping({"amenity": "school", "isced:level": "1"}, mapping)["category_id"] == "44"
    assert match_mapping({"amenity": "school"}, mapping)["category_id"] == "244"


def test_versioned_poi_fixture_builds_without_ignored_real_artifact(tmp_path):
    taxonomy, _mapping = _v2()
    elements = [json.loads(line) for line in (ROOT / "tests/fixtures/hcmc_pois_v2.raw.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = build_artifact(
        elements, taxonomy=taxonomy,
        mapping_path=ROOT / "data/osm/mappings/place_types.poc.v2.json",
        property_path=ROOT / "data/properties/poi_properties.poc.v4.json",
        boundary_path=ROOT / "tests/fixtures/hcmc_mini_boundary.geojson",
        output_dir=tmp_path / "fixture_artifact",
    )
    rows = [json.loads(line) for line in (tmp_path / "fixture_artifact/pois.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["poi_count"] == 8
    assert manifest["rejected_counts"]["unmapped_tags"] == 1
    assert {"19", "242", "243", "52", "244", "44", "245", "83"} == {row["category_id"] for row in rows}
    generic = next(row for row in rows if row["category_id"] == "242")
    assert generic["properties"]["osm"]["fast_charging"]["value"] is True
    assert generic["properties"]["osm"]["fast_charging"]["source"] == "osm"
    assert min(sum(value["status"] == "known" for value in row["properties"]["effective"].values()) for row in rows) >= 5


def test_tire_retail_is_not_misrepresented_as_tire_repair():
    _taxonomy, mapping = _v2()
    assert match_mapping({"shop": "car_repair"}, mapping)["category_id"] == "83"
    assert match_mapping({"shop": "motorcycle_repair"}, mapping)["category_id"] == "245"
    assert match_mapping({"shop": "tyres"}, mapping) is None
