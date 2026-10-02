from __future__ import annotations

from contextlib import contextmanager

from model_intent.poi_repository import PoiRepository


class GenerationConnection:
    def __init__(self):
        self.queries = []

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        return self

    def fetchone(self):
        return ("ready", "taxonomy", "mapping-hash", "property-hash", "hcmc-osm-place-types-poc-v2")


class GenerationDatabase:
    def __init__(self):
        self.conn = GenerationConnection()

    @contextmanager
    def connection(self):
        yield self.conn, None


def test_generated_display_label_keeps_nullable_name_and_no_extra_v2_hash_fields():
    repo = PoiRepository(())
    row = (
        "osm:node:1", "node", 1, 23, None, "[Generated OSM label] coffee (osm:node:1)",
        {}, 10.78, 106.69, "node", "amenity-cafe", {}, {}, {}, {},
        None, 0, "unknown", None, None, "0" * 64, 1, None,
    )
    record = repo._record_from_row(row)
    payload = record.as_dict()
    assert payload["name"] is None
    assert payload["display_name"] == row[5]
    assert "display_name_source" not in payload
    assert payload["rating"]["value"] is None


def test_cuisine_filter_matches_artifact_candidates_without_dropping_direct_matches():
    from dataclasses import replace

    base = PoiRepository(())._record_from_row((
        "osm:node:1", "node", 1, 23, None, "Cafe", {}, 10.78, 106.69, "node", "amenity-cafe",
        {}, {}, {}, {}, None, 0, "unknown", None, None, "0" * 64, 1, None,
    ))
    direct = replace(base, category_id="65")
    tagged = replace(base, poi_id="osm:node:2", category_id="57", osm_tags={"cuisine": "japanese ; SUSHI"})
    irrelevant = replace(base, poi_id="osm:node:3", category_id="57", osm_tags={"cuisine": "japanese"})
    repo = PoiRepository((direct, tagged, irrelevant))
    assert [record.poi_id for record in repo.for_categories(("57", "65"),
           cuisine_filter={"sushi"}, always_include_category_id="65")] == ["osm:node:1", "osm:node:2"]


def test_postgres_cuisine_filter_is_pushed_down_and_keeps_generation_scope():
    class Connection:
        def __init__(self):
            self.query = None
            self.params = None

        def execute(self, query, params):
            self.query, self.params = query.as_string(), params
            return self

        def fetchall(self):
            return []

    class Database:
        def __init__(self):
            self.conn = Connection()

        @contextmanager
        def connection(self):
            yield self.conn, None

    database = Database()
    repo = PoiRepository((), generation_id=7, source="postgres")
    repo.database = database
    assert repo.for_categories(("20", "57", "65"), cuisine_filter={"sushi"}, always_include_category_id="65") == ()
    assert "place_type_id IN (%s, %s, %s)" in database.conn.query
    assert "regexp_split_to_table" in database.conn.query
    assert "ORDER BY poi_id" in database.conn.query
    assert database.conn.params == (7, 20, 57, 65, 65, ["sushi"])


def test_database_generation_preserves_mapping_version_on_poi_records():
    database = GenerationDatabase()
    repo = PoiRepository.from_database(
        database, generation_id=7, taxonomy_fingerprint="taxonomy", mapping_sha256="mapping-hash",
        property_taxonomy_sha256="property-hash",
    )
    assert repo.mapping_version == "hcmc-osm-place-types-poc-v2"
    record = repo._record_from_row((
        "osm:node:1", "node", 1, 23, "Cafe", "Cafe", {}, 10.78, 106.69, "node", "amenity-cafe",
        {}, {}, {}, {}, 4.5, 12, "mock", "poi-mock-v2", 17, "0" * 64, 1, None,
    ))
    assert record.mapping_version == "hcmc-osm-place-types-poc-v2"
