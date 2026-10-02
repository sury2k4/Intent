import json
from pathlib import Path

from model_intent.osm_census import census_hcmc_candidates
from model_intent.osm_ingest import Boundary, validate_catalogs
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def test_hcmc_census_counts_unmapped_candidates_before_mapping_filter():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v2.json")
    catalogs = validate_catalogs(
        taxonomy, ROOT / "data/osm/mappings/place_types.poc.v2.json",
        ROOT / "data/properties/poi_properties.poc.v4.json",
    )
    elements = [json.loads(line) for line in (ROOT / "tests/fixtures/hcmc_pois_v2.raw.jsonl").read_text(encoding="utf-8").splitlines()]
    elements += [
        {"type": "node", "id": 200, "lat": 12.0, "lon": 106.69,
         "tags": {"amenity": "bank", "name": "Bank outside HCMC"}},
        {"type": "node", "id": 201, "lat": 10.78, "lon": 106.69,
         "tags": {"highway": "primary", "name": "A road, not a business"}},
    ]
    report = census_hcmc_candidates(
        elements, boundary=Boundary.from_geojson(ROOT / "tests/fixtures/hcmc_mini_boundary.geojson"),
        taxonomy=taxonomy, mapping_catalog=catalogs["mapping"], property_profiles=catalogs["profiles"],
    )
    counts = report["counts"]
    assert counts["reader_emitted"] == 11
    assert counts["poi_like_tagged_candidates_nationwide"] == 10
    assert counts["outside_hcmc"] == 1
    assert counts["inside_hcmc_tagged_candidates"] == 9
    assert counts["inside_hcmc_unmapped"] == 1
    assert counts["inside_hcmc_importable"] == 8


def test_complete_signature_census_reports_every_in_boundary_disposition():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v2.json")
    catalogs = validate_catalogs(
        taxonomy, ROOT / "data/osm/mappings/place_types.poc.v2.json",
        ROOT / "data/properties/poi_properties.poc.v4.json",
    )
    rows = [
        {"type": "node", "id": 1, "lat": 10.78, "lon": 106.69,
         "tags": {"amenity": "cafe", "name": "Cafe"}},
        {"type": "node", "id": 2, "lat": 10.78, "lon": 106.69,
         "tags": {"amenity": "bench"}},
        {"type": "node", "id": 3, "lat": 10.78, "lon": 106.69,
         "tags": {"highway": "residential", "name": "Road"}},
        {"type": "node", "id": 4, "lat": 12.0, "lon": 106.69,
         "tags": {"amenity": "cafe", "name": "Outside"}},
    ]
    report = census_hcmc_candidates(
        rows, boundary=Boundary.from_geojson(ROOT / "tests/fixtures/hcmc_mini_boundary.geojson"),
        taxonomy=taxonomy, mapping_catalog=catalogs["mapping"],
        property_profiles=catalogs["profiles"], osm_only=True, complete_signatures=True,
    )
    assert report["counts"]["inside_hcmc_tagged_candidates"] == 2
    assert report["counts"]["not_poi_like"] == 1
    assert report["counts"]["inside_hcmc_imported"] == 1
    assert report["counts"]["inside_hcmc_unmapped_needs_review"] == 1
    assert {row["signature"]: row["dispositions"] for row in report["signatures"]} == {
        "amenity=bench": {"unmapped_needs_review": 1},
        "amenity=cafe": {"imported": 1},
    }
