from pathlib import Path

from model_intent.osm_ingest import validate_catalogs
from model_intent.poi_properties import load_property_profiles
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def test_poi_catalogs_cover_taxonomy_with_valid_leaf_rules():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    result = validate_catalogs(
        taxonomy,
        ROOT / "data/osm/mappings/place_types.poc.v1.json",
        ROOT / "data/properties/poi_properties.poc.v1.json",
    )
    assert result["coverage"]["complete"] is True
    assert result["coverage"]["leaf_count"] == 222
    assert len(result["mapping"]["rules"]) >= 50
    assert {"air_conditioning", "internet_access", "outdoor_seating", "airy"} <= set(result["properties"])


def test_v2_profiles_cover_every_mappable_leaf_with_relevant_properties():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    catalogs = validate_catalogs(
        taxonomy,
        ROOT / "data/osm/mappings/place_types.poc.v1.json",
        ROOT / "data/properties/poi_properties.poc.v2.json",
    )
    profiles = catalogs["profiles"]
    mapped = set(catalogs["coverage"]["mapped_category_ids"])
    resolved = profiles.validate_coverage(taxonomy, mapped)
    assert len(profiles.definitions) >= 60
    assert len(resolved) == len(mapped)
    assert profiles.minimum_known == 5
    assert {"shelter", "route_information", "tactile_paving"} <= set(resolved["114"])
    assert "air_conditioning" not in resolved["114"]
    assert {"cash_withdrawal", "cash_deposit"} <= set(resolved["52"])
    assert "outdoor_seating" not in resolved["52"]
    assert {"shade", "drinking_water", "playground"} <= set(resolved["74"])
    assert {"air_conditioning", "indoor_seating"} <= set(resolved["57"])
