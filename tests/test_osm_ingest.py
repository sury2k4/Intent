import json
from pathlib import Path

from model_intent.osm_ingest import Boundary, build_artifact, candidate_signature, candidate_tags, enrich_artifact, file_sha256, load_mapping_catalog, load_property_catalog, match_mapping, mock_property, normalize_element, stable_hash
from model_intent.poi_properties import load_property_profiles
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def _catalogs():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    mapping = load_mapping_catalog(ROOT / "data/osm/mappings/place_types.poc.v1.json", taxonomy)
    properties = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v1.json")
    boundary = Boundary.from_geojson(ROOT / "tests/fixtures/hcmc_mini_boundary.geojson")
    return taxonomy, mapping, properties, boundary


def test_mapping_specific_rule_precedes_restaurant_fallback():
    _taxonomy, mapping, _properties, _boundary = _catalogs()
    selected = match_mapping({"amenity": "restaurant", "cuisine": "japanese;sushi"}, mapping)
    assert selected["id"] == "restaurant-japanese"
    assert selected["category_id"] == "57"


def test_ambiguous_school_tags_are_not_guessed():
    _taxonomy, mapping, _properties, _boundary = _catalogs()
    assert match_mapping({"amenity": "school"}, mapping) is None
    primary = match_mapping({"amenity": "school", "isced:level": "1"}, mapping)
    assert primary["category_id"] == "44"
    assert match_mapping({"amenity": "place_of_worship"}, mapping) is None
    pagoda = match_mapping({"amenity": "place_of_worship", "religion": "buddhist"}, mapping)
    assert pagoda["category_id"] == "141"


def test_real_properties_are_not_overwritten_by_mock_values():
    taxonomy, mapping, properties, boundary = _catalogs()
    row, error = normalize_element(
        {"type": "node", "id": 101, "lat": 10.78, "lon": 106.69, "tags": {"amenity": "cafe", "name": "Cafe", "air_conditioning": "no"}},
        taxonomy=taxonomy, mapping_catalog=mapping, property_catalog=properties, boundary=boundary,
    )
    assert error is None
    assert row.effective_properties["air_conditioning"]["value"] is False
    assert row.effective_properties["air_conditioning"]["source"] == "osm"
    assert "air_conditioning" not in row.properties_mock


def test_unknown_osm_value_is_retained_and_not_mocked():
    taxonomy, mapping, properties, boundary = _catalogs()
    row, error = normalize_element(
        {"type": "node", "id": 102, "lat": 10.78, "lon": 106.69, "tags": {"amenity": "cafe", "name": "Cafe", "air_conditioning": "customers"}},
        taxonomy=taxonomy, mapping_catalog=mapping, property_catalog=properties, boundary=boundary,
    )
    assert error is None
    value = row.effective_properties["air_conditioning"]
    assert value["status"] == "unknown" and value["source"] == "osm"
    assert "air_conditioning" not in row.properties_mock


def test_open_way_is_rejected_and_polygon_way_gets_representative_location():
    taxonomy, mapping, properties, boundary = _catalogs()
    open_way, open_error = normalize_element(
        {"type": "way", "id": 120, "geometry": [{"lat": 10.78, "lon": 106.69}, {"lat": 10.79, "lon": 106.70}], "tags": {"amenity": "cafe", "name": "Open way"}},
        taxonomy=taxonomy, mapping_catalog=mapping, property_catalog=properties, boundary=boundary,
    )
    assert open_way is None and open_error == "unsupported_open_way"
    area, area_error = normalize_element(
        {"type": "way", "id": 121, "geometry": {"type": "Polygon", "coordinates": [[[106.68, 10.77], [106.70, 10.77], [106.70, 10.79], [106.68, 10.79], [106.68, 10.77]]]}, "tags": {"amenity": "cafe", "name": "Area Cafe"}},
        taxonomy=taxonomy, mapping_catalog=mapping, property_catalog=properties, boundary=boundary,
    )
    assert area_error is None
    assert area.location_method == "representative_point"
    assert 106.0 <= area.longitude <= 107.0 and 10.0 <= area.latitude <= 11.0


def test_mock_properties_are_stable_and_seeded():
    definition = {"value_type": "boolean"}
    first = mock_property("wifi", definition, poi_id="osm:node:1", category_id="23", seed=17)
    second = mock_property("wifi", definition, poi_id="osm:node:1", category_id="23", seed=17)
    different = mock_property("wifi", definition, poi_id="osm:node:1", category_id="23", seed=18)
    assert first == second
    assert first["source"] == "mock"
    assert first["mock_seed"] == 17
    assert first != different


def test_artifact_is_canonical_across_input_order(tmp_path):
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    elements = [
        {"type": "node", "id": 101, "lat": 10.78, "lon": 106.69, "tags": {"amenity": "cafe", "name": "Cafe A"}},
        {"type": "node", "id": 102, "lat": 10.781, "lon": 106.691, "tags": {"amenity": "pharmacy", "name": "Pharmacy"}},
    ]
    outputs = [tmp_path / "one", tmp_path / "two"]
    manifests = []
    for rows, output in ((elements, outputs[0]), (list(reversed(elements)), outputs[1])):
        manifests.append(build_artifact(
            rows,
            taxonomy=taxonomy,
            mapping_path=ROOT / "data/osm/mappings/place_types.poc.v1.json",
            property_path=ROOT / "data/properties/poi_properties.poc.v1.json",
            boundary_path=ROOT / "tests/fixtures/hcmc_mini_boundary.geojson",
            output_dir=output,
        ))
    assert manifests[0]["content_set_hash"] == manifests[1]["content_set_hash"]
    assert manifests[0]["pois_jsonl_sha256"] == manifests[1]["pois_jsonl_sha256"]
    assert (outputs[0] / "pois.jsonl").read_bytes() == (outputs[1] / "pois.jsonl").read_bytes()
    row = json.loads((outputs[0] / "pois.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["poi_id"] == "osm:node:101"
    assert row["rating"]["source"] == "mock"
    assert row["rating"]["mock_schema_version"] == "poi-mock-v1"
    assert row["rating"]["mock_seed"] == 17


def test_v2_enrichment_preserves_real_values_and_fills_every_poi(tmp_path):
    taxonomy, mapping, _properties, boundary = _catalogs()
    profiles = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v2.json", taxonomy)
    for osm_id, category_tag, property_tag in (
        (100, "cafe", {"air_conditioning": "no", "internet_access": "wlan"}),
        (101, "bus_stop", {"shelter": "unexpected"}),
    ):
        tags = {"name": f"Place {osm_id}", **property_tag}
        tags["highway" if category_tag == "bus_stop" else "amenity"] = category_tag
        row, error = normalize_element(
            {"type": "node", "id": osm_id, "lat": 10.78, "lon": 106.69, "tags": tags},
            taxonomy=taxonomy,
            mapping_catalog=mapping,
            property_catalog=profiles.definitions,
            property_profiles=profiles,
            boundary=boundary,
            mock_schema_version="poi-mock-v2",
        )
        assert error is None
        suitable = profiles.applicable_ids(taxonomy, row.category_id)
        assert sum(row.effective_properties[key]["status"] == "known" for key in suitable) >= 5
        assert all(row.properties_mock[key]["mock_schema_version"] == "poi-mock-v2" for key in row.properties_mock)
        if category_tag == "cafe":
            assert row.effective_properties["air_conditioning"]["value"] is False
            assert "air_conditioning" not in row.properties_mock
        else:
            assert row.effective_properties["shelter"]["status"] == "unknown"
            assert "shelter" not in row.properties_mock
            assert "air_conditioning" not in row.properties_mock

    source_dir = tmp_path / "v1"
    base_rows = [
        {"type": "node", "id": 101, "lat": 10.78, "lon": 106.69, "tags": {"highway": "bus_stop", "name": "Bus stop"}},
        {"type": "node", "id": 102, "lat": 10.78, "lon": 106.69, "tags": {"amenity": "cafe", "name": "Cafe", "air_conditioning": "no"}},
    ]
    build_artifact(
        base_rows,
        taxonomy=taxonomy,
        mapping_path=ROOT / "data/osm/mappings/place_types.poc.v1.json",
        property_path=ROOT / "data/properties/poi_properties.poc.v1.json",
        boundary_path=ROOT / "tests/fixtures/hcmc_mini_boundary.geojson",
        output_dir=source_dir,
    )
    v2_path = tmp_path / "v2"
    result = enrich_artifact(
        source_dir / "manifest.json",
        output_dir=v2_path,
        taxonomy=taxonomy,
        mapping_path=ROOT / "data/osm/mappings/place_types.poc.v1.json",
        property_path=ROOT / "data/properties/poi_properties.poc.v2.json",
    )
    assert result["poi_count"] == 2
    assert min(item["minimum_known"] for item in result["property_coverage"].values()) >= 5
    assert result["property_taxonomy_version"] == "poi-properties-poc-v2"
    assert result["parent_artifact_id"] != result["artifact_id"]
    assert (source_dir / "pois.jsonl").exists()


def test_fast_charging_requires_explicit_power_not_connector_count():
    from model_intent.osm_ingest import normalize_osm_properties

    definitions = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v4.json")
    assert "fast_charging" not in normalize_osm_properties({"socket:ccs": "2"}, definitions)
    fast = normalize_osm_properties({"socket:ccs:output": "50 kW"}, definitions)["fast_charging"]
    assert fast["status"] == "known" and fast["source"] == "osm" and fast["value"] is True
    slow = normalize_osm_properties({"socket:ccs:output": "22 kW"}, definitions)["fast_charging"]
    assert slow["status"] == "known" and slow["value"] is False
    mixed = normalize_osm_properties({"socket:ccs:output": "22 kW", "socket:chademo:output": "50 kW"}, definitions)["fast_charging"]
    assert mixed["status"] == "known" and mixed["value"] is True
    assert len(mixed["raw"]) == 2
    partially_unknown = normalize_osm_properties({"socket:ccs:output": "22 kW", "socket:chademo:output": "unknown"}, definitions)["fast_charging"]
    assert partially_unknown["status"] == "unknown"
    unknown = normalize_osm_properties({"socket:ccs:output": "unknown"}, definitions)["fast_charging"]
    assert unknown["status"] == "unknown"


def test_enrichment_regenerates_mock_rating_when_seed_changes(tmp_path):
    from model_intent.poi_mocking import mock_rating

    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    source = tmp_path / "source"
    build_artifact(
        [{"type": "node", "id": 999, "lat": 10.78, "lon": 106.69,
          "tags": {"amenity": "cafe", "name": "Seeded cafe"}}],
        taxonomy=taxonomy,
        mapping_path=ROOT / "data/osm/mappings/place_types.poc.v1.json",
        property_path=ROOT / "data/properties/poi_properties.poc.v1.json",
        boundary_path=ROOT / "tests/fixtures/hcmc_mini_boundary.geojson",
        output_dir=source,
    )
    target = tmp_path / "enriched"
    enrich_artifact(
        source / "manifest.json", output_dir=target, taxonomy=taxonomy,
        mapping_path=ROOT / "data/osm/mappings/place_types.poc.v1.json",
        property_path=ROOT / "data/properties/poi_properties.poc.v2.json",
        seed=18,
    )
    row = json.loads((target / "pois.jsonl").read_text(encoding="utf-8"))
    rating, count = mock_rating("osm:node:999", "23", 18, "poi-mock-v2")
    assert row["rating"]["value"] == rating
    assert row["rating"]["review_count"] == count
    assert row["rating"]["mock_seed"] == 18
    assert row["rating"]["mock_schema_version"] == "poi-mock-v2"


def test_unparseable_osm_values_reject_only_the_problem_poi(tmp_path):
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v2.json")
    tags = {"amenity": "atm", "name": "Unreliable ATM", "cash_out": "sometimes", "cash_in": "sometimes",
            "wheelchair": "unknown", "covered": "partial", "lit": "unknown"}
    rows = [
        {"type": "node", "id": 300, "lat": 10.78, "lon": 106.69, "tags": tags},
        {"type": "node", "id": 301, "lat": 10.78, "lon": 106.69, "tags": {"amenity": "atm", "name": "Working ATM"}},
    ]
    result = build_artifact(
        rows, taxonomy=taxonomy,
        mapping_path=ROOT / "data/osm/mappings/place_types.poc.v2.json",
        property_path=ROOT / "data/properties/poi_properties.poc.v4.json",
        boundary_path=ROOT / "tests/fixtures/hcmc_mini_boundary.geojson",
        output_dir=tmp_path / "output",
    )
    assert result["poi_count"] == 1
    assert result["rejected_counts"]["insufficient_known_properties"] == 1


def test_v3_osm_only_has_no_synthetic_assertions_or_rating():
    taxonomy, mapping, properties, boundary = _catalogs()
    record, reason = normalize_element(
        {"type": "node", "id": 777, "lat": 10.78, "lon": 106.69,
         "tags": {"amenity": "cafe", "name": "Existing Cafe", "wheelchair": "yes", "rating:count": "900"}},
        taxonomy=taxonomy, mapping_catalog=mapping, property_catalog=properties,
        boundary=boundary, osm_only=True,
    )
    assert reason is None
    assert record.properties_mock == {}
    assert record.properties_osm == record.effective_properties
    assert record.effective_properties["wheelchair"]["source"] == "osm"
    assert "air_conditioning" not in record.effective_properties
    assert record.rating is None and record.review_count == 0 and record.rating_source == "unknown"
    assert record.rating_mock_seed is None and record.rating_mock_schema_version is None


def test_v3_unnamed_facility_has_explicit_generated_label_and_no_fake_name():
    taxonomy, mapping, properties, boundary = _catalogs()
    record, reason = normalize_element(
        {"type": "node", "id": 888, "lat": 10.78, "lon": 106.69,
         "tags": {"amenity": "parking"}},
        taxonomy=taxonomy, mapping_catalog=mapping, property_catalog=properties,
        boundary=boundary, osm_only=True,
    )
    assert reason is None
    assert record.name is None and record.names == {"default": None, "vi": None, "en": None}
    assert record.display_name.startswith("[Nhãn tạo từ loại OSM]")
    assert "osm:node:888" in record.display_name
    assert record.rating_source == "unknown"


def test_v3_all_classified_unnamed_businesses_are_included_but_v2_are_not():
    taxonomy, mapping, properties, boundary = _catalogs()
    element = {"type": "node", "id": 889, "lat": 10.78, "lon": 106.69,
               "tags": {"amenity": "cafe"}}
    omitted, legacy_reason = normalize_element(
        element, taxonomy=taxonomy, mapping_catalog=mapping,
        property_catalog=properties, boundary=boundary,
    )
    assert omitted is None and legacy_reason == "unnamed_business"
    record, reason = normalize_element(
        element, taxonomy=taxonomy, mapping_catalog=mapping,
        property_catalog=properties, boundary=boundary, osm_only=True,
    )
    assert reason is None and record.name is None
    assert record.display_name.startswith("[Nhãn tạo từ loại OSM]")


def test_reader_candidate_filter_does_not_admit_road_barrier_without_poi():
    assert candidate_tags({"highway": "primary", "barrier": "gate", "name": "Road"}) == {}
    assert candidate_tags({"railway": "rail", "waterway": "river", "name": "Rail"}) == {}
    assert candidate_tags({"highway": "bus_stop", "name": "Bus stop"}) == {"highway": "bus_stop"}
    assert candidate_signature({"amenity": "restaurant", "cuisine": "sushi", "name": "Sushi"}) == "amenity=restaurant|cuisine=sushi"


def test_v3_artifact_rejects_unreviewed_signatures_before_writing(tmp_path):
    import pytest
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v2.json")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "mapping_version": "hcmc-osm-place-types-poc-v2",
        "taxonomy_version": taxonomy.document.taxonomy_version,
        "boundary_sha256": file_sha256(ROOT / "tests/fixtures/hcmc_mini_boundary.geojson"),
        "pbf_sha256": None, "audit_sha256": stable_hash([]),
        "signatures": [],
    }), encoding="utf-8")
    target = tmp_path / "new-artifact"
    with pytest.raises(ValueError, match="needs review"):
        build_artifact(
            [{"type": "node", "id": 999, "lat": 10.78, "lon": 106.69,
              "tags": {"amenity": "cafe", "name": "New cafe"}}],
            taxonomy=taxonomy, mapping_path=ROOT / "data/osm/mappings/place_types.poc.v2.json",
            property_path=ROOT / "data/properties/poi_properties.poc.v4.json",
            boundary_path=ROOT / "tests/fixtures/hcmc_mini_boundary.geojson",
            output_dir=target, osm_only=True, signature_registry_path=registry,
        )
    assert not target.exists()


def test_pyosmium_streams_nodes_closed_ways_and_multipolygon_areas():
    import pytest
    from collections import Counter

    pytest.importorskip("osmium")
    from model_intent.osm_ingest import iter_pbf_elements

    stats = Counter()
    rows = list(iter_pbf_elements(ROOT / "tests/fixtures/hcmc-mini.osm", reader_stats=stats))
    identities = {(row["type"], row["id"]) for row in rows}
    assert ("node", 1) in identities
    assert ("way", 20) in identities
    assert ("relation", 30) in identities
    assert all(row.get("geometry") for row in rows if row["type"] in {"way", "relation"})
    assert stats["emitted_node"] >= 1
    assert stats["emitted_way_area"] >= 1
    assert stats["emitted_relation_area"] >= 1
