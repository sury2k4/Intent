import hashlib
import json
from contextlib import contextmanager

import pytest

from model_intent.osm_ingest import stable_hash
from model_intent.poi_sync import activate_poi_generation, sync_poi_artifact, validate_poi_database_schema


UNIFIED_URL = "postgresql://model_intent@127.0.0.1:5433/model_intent_unified"


def _manifest(tmp_path, artifact_id="conflicting-artifact"):
    payload = {"poi_id": "osm:node:1", "category_id": "23", "content_hash": ""}
    payload["content_hash"] = stable_hash(payload)
    raw = (json.dumps(payload) + "\n").encode()
    (tmp_path / "pois.jsonl").write_bytes(raw)
    manifest = {
        "artifact_id": artifact_id, "poi_count": 1,
        "pois_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "content_set_hash": stable_hash([(payload["poi_id"], payload["content_hash"])]),
        "taxonomy_fingerprint": "4" * 64,
        "mapping_sha256": "2" * 64, "property_taxonomy_sha256": "3" * 64,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, path


class ReadyConnection:
    def __init__(self, content_hash):
        self.content_hash = content_hash
        self.statements = []

    def execute(self, statement, params=None):
        self.statements.append((statement, params))
        self.statement = statement
        return self

    def fetchone(self):
        if "SELECT count(*) FROM pois" in self.statement:
            return (1,)
        return (
            17, "ready", 1, 1, "0" * 64, self.content_hash,
            "4" * 64, "2" * 64, "3" * 64, "a" * 64,
            "unknown", "unknown", "unknown", 17,
        )

    def fetchall(self):
        return [
            ("poi_dataset_generations", key) for key in (
                "id", "dataset_key", "artifact_id", "state", "expected_count", "completed_count",
                "content_set_hash", "taxonomy_fingerprint", "mapping_sha256", "property_taxonomy_sha256",
            )
        ] + [
            ("pois", key) for key in (
                "generation_id", "poi_id", "place_type_id", "osm_tags", "properties_osm",
                "properties_mock", "effective_properties", "rating", "review_count", "content_hash",
            )
        ] + [
            ("active_poi_dataset_generation", key) for key in ("dataset_key", "generation_id")
        ]

    @contextmanager
    def transaction(self):
        yield

    def commit(self):
        pass

    def rollback(self):
        pass


class ReadyDatabase:
    database_url = UNIFIED_URL

    def __init__(self, connection):
        self.conn = connection

    @contextmanager
    def connection(self):
        yield self.conn, 0


def test_legacy_poi_schema_is_rejected_before_any_database_writes(tmp_path):
    class OldPoiConnection(ReadyConnection):
        def fetchall(self):
            return [("pois", "id"), ("pois", "import_id"), ("pois", "source_tags")]

    manifest, path = _manifest(tmp_path, artifact_id="legacy-blocked")
    connection = OldPoiConnection(manifest["content_set_hash"])
    with pytest.raises(RuntimeError, match="schema does not match"):
        sync_poi_artifact(ReadyDatabase(connection), path)
    assert all("pg_advisory_lock" not in statement for statement, _ in connection.statements)
    assert all("DELETE FROM pois" not in statement for statement, _ in connection.statements)


def test_direct_activation_is_disabled_before_connection():
    connection = ReadyConnection("1" * 64)
    with pytest.raises(RuntimeError, match="unilateral POI activation"):
        activate_poi_generation(ReadyDatabase(connection), 17, dataset_key="hcmc-osm")
    assert connection.statements == []


def test_conflicting_manifest_does_not_mark_ready_generation_failed(tmp_path):
    manifest, path = _manifest(tmp_path)
    connection = ReadyConnection("f" * 64)
    with pytest.raises(RuntimeError, match="immutable and conflicts"):
        sync_poi_artifact(ReadyDatabase(connection), path)
    statements = "\n".join(statement for statement, _ in connection.statements)
    assert "UPDATE poi_dataset_generations SET state='failed'" not in statements
    assert "DELETE FROM pois" not in statements


def test_artifact_inventory_checks_counts_hashes_and_streaming_batches(tmp_path):
    from model_intent.poi_sync import _artifact_inventory, _iter_batches, _iter_rows

    manifest, path = _manifest(tmp_path)
    count, content_set_hash = _artifact_inventory(tmp_path / "pois.jsonl")
    assert count == manifest["poi_count"]
    assert content_set_hash == manifest["content_set_hash"]
    assert [len(batch) for batch in _iter_batches(_iter_rows(tmp_path / "pois.jsonl"), 1)] == [1]

    manifest["content_set_hash"] = "f" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content set hash"):
        sync_poi_artifact(ReadyDatabase(ReadyConnection(content_set_hash)), path, activate=False)


@pytest.mark.parametrize("osm_only", [False, True])
def test_new_poi_generation_stages_in_bounded_batches_without_activation(tmp_path, osm_only):
    from model_intent.poi_sync import _database_poi_content_hash

    rows = []
    for number in range(1, 4):
        row = {
            "poi_id": f"osm:node:{number}", "category_id": "23",
            "osm": {"type": "node", "id": number},
            "location": {"lat": 10.0, "lon": 106.0},
            "properties": {"osm": {"parking": {"source": "osm", "value": True}},
                           "mock": {}, "effective": {"parking": {"source": "osm", "value": True}}},
            "rating": {"value": None, "review_count": 0, "source": "unknown"},
            "content_hash": "",
        }
        row["content_hash"] = stable_hash(row)
        rows.append(row)
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (tmp_path / "pois.jsonl").write_bytes(raw)
    manifest = {
        "artifact_id": "batch-staging", "poi_count": len(rows),
        "pois_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "content_set_hash": stable_hash([(row["poi_id"], row["content_hash"]) for row in rows]),
        "taxonomy_fingerprint": "a" * 64,
        "mapping_sha256": "b" * 64, "property_taxonomy_sha256": "c" * 64,
    }
    if osm_only:
        manifest.update({
            "artifact_kind": "hcmc_osm_poi_v4_osm_only", "osm_only": True,
            "mock_schema_version": None, "mock_seed": None,
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_bytes = path.read_bytes()
    artifact_bytes = (tmp_path / "pois.jsonl").read_bytes()

    class BuildingConnection(ReadyConnection):
        def __init__(self):
            super().__init__(manifest["content_set_hash"])
            self.inserted = []
            self.row_mock_seeds = []
            self.batch_lengths = []

        def fetchone(self):
            if "FROM poi_dataset_generations WHERE dataset_key" in self.statement:
                return None
            if "INSERT INTO poi_dataset_generations" in self.statement:
                return (42,)
            if "SELECT count(*) FROM pois" in self.statement:
                return (len(self.inserted),)
            return None

        def cursor(self, *, name=None):
            connection = self

            class Cursor:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    pass

                def executemany(self, sql, args):
                    connection.batch_lengths.append(len(args))
                    connection.inserted.extend((item[1], item[21]) for item in args)
                    connection.row_mock_seeds.extend(item[20] for item in args)

                def execute(self, sql, params):
                    assert "ORDER BY CASE osm_type" in sql

                def __iter__(self):
                    return iter(connection.inserted)

            return Cursor()

    connection = BuildingConnection()
    result = sync_poi_artifact(ReadyDatabase(connection), path, batch_size=2, activate=False)
    assert result["completed_count"] == 3 and not result["activated"]
    assert connection.batch_lengths == [2, 1]
    assert connection.row_mock_seeds == [None, None, None]
    inserted = [params for sql, params in connection.statements if "INSERT INTO poi_dataset_generations" in sql]
    assert len(inserted) == 1
    assert inserted[0][10:12] == (("none", 0) if osm_only else ("unknown", 17))
    assert path.read_bytes() == manifest_bytes
    assert (tmp_path / "pois.jsonl").read_bytes() == artifact_bytes
    assert _database_poi_content_hash(connection, 42, 3) == manifest["content_set_hash"]
    assert not any("INSERT INTO active_poi_dataset_generation" in sql for sql, _ in connection.statements)


@pytest.mark.parametrize("osm_only", [False, True])
def test_reused_poi_generation_no_activate_does_not_write_active_pointer(tmp_path, monkeypatch, osm_only):
    from model_intent.osm_ingest import file_sha256

    manifest, path = _manifest(tmp_path)
    if osm_only:
        row = json.loads((tmp_path / "pois.jsonl").read_text(encoding="utf-8"))
        row["properties"] = {"osm": {}, "mock": {}, "effective": {}}
        row["rating"] = {"source": "unknown", "value": None, "review_count": 0}
        row["content_hash"] = ""
        row["content_hash"] = stable_hash(row)
        data = (json.dumps(row) + "\n").encode()
        (tmp_path / "pois.jsonl").write_bytes(data)
        manifest.update({
            "pois_jsonl_sha256": hashlib.sha256(data).hexdigest(),
            "content_set_hash": stable_hash([(row["poi_id"], row["content_hash"])]),
            "artifact_kind": "hcmc_osm_poi_v3_osm_only", "osm_only": True,
            "mock_schema_version": None, "mock_seed": None,
        })
        path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_bytes = path.read_bytes()
    artifact_bytes = (tmp_path / "pois.jsonl").read_bytes()
    monkeypatch.setattr(
        "model_intent.poi_sync._database_poi_content_hash",
        lambda *args: manifest["content_set_hash"],
    )
    connection = ReadyConnection(manifest["content_set_hash"])
    connection.fetchone = lambda: (
        (17, "ready", 1, 1, manifest["pois_jsonl_sha256"], manifest["content_set_hash"],
         manifest["taxonomy_fingerprint"], manifest["mapping_sha256"],
         manifest["property_taxonomy_sha256"], file_sha256(path),
         "unknown", "unknown", "none" if osm_only else "unknown", 0 if osm_only else 17)
        if "FROM poi_dataset_generations" in connection.statement else (1,)
    )
    result = sync_poi_artifact(ReadyDatabase(connection), path, activate=False)
    assert result["already_ready"] and not result["activated"]
    assert path.read_bytes() == manifest_bytes
    assert (tmp_path / "pois.jsonl").read_bytes() == artifact_bytes
    assert not any("INSERT INTO active_poi_dataset_generation" in sql for sql, _ in connection.statements)
    assert not any("DELETE FROM pois" in sql for sql, _ in connection.statements)
    assert not any("UPDATE poi_dataset_generations" in sql for sql, _ in connection.statements)


@pytest.mark.parametrize("stored_schema,stored_seed", [("none", 17), ("unknown", 0), ("None", 0)])
def test_osm_only_ready_reuse_rejects_conflicting_mock_metadata_without_writes(
    tmp_path, stored_schema, stored_seed,
):
    from model_intent.osm_ingest import file_sha256

    manifest, path = _manifest(tmp_path)
    manifest.update({
        "artifact_kind": "hcmc_osm_poi_v4_osm_only", "osm_only": True,
        "mock_schema_version": None, "mock_seed": None,
    })
    row = json.loads((tmp_path / "pois.jsonl").read_text(encoding="utf-8"))
    row["properties"] = {"osm": {}, "mock": {}, "effective": {}}
    row["rating"] = {"source": "unknown", "value": None, "review_count": 0}
    row["content_hash"] = ""
    row["content_hash"] = stable_hash(row)
    data = (json.dumps(row) + "\n").encode()
    (tmp_path / "pois.jsonl").write_bytes(data)
    manifest["pois_jsonl_sha256"] = hashlib.sha256(data).hexdigest()
    manifest["content_set_hash"] = stable_hash([(row["poi_id"], row["content_hash"])])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_bytes = path.read_bytes()
    artifact_bytes = (tmp_path / "pois.jsonl").read_bytes()
    connection = ReadyConnection(manifest["content_set_hash"])
    connection.fetchone = lambda: (
        (17, "ready", 1, 1, manifest["pois_jsonl_sha256"], manifest["content_set_hash"],
         manifest["taxonomy_fingerprint"], manifest["mapping_sha256"],
         manifest["property_taxonomy_sha256"], file_sha256(path),
         "unknown", "unknown", stored_schema, stored_seed)
        if "FROM poi_dataset_generations" in connection.statement else (1,)
    )
    with pytest.raises(RuntimeError, match="immutable and conflicts"):
        sync_poi_artifact(ReadyDatabase(connection), path, activate=False)
    assert not any("INSERT INTO active_" in sql or "UPDATE poi_dataset_generations" in sql
                   or "DELETE FROM pois" in sql for sql, _ in connection.statements)
    assert path.read_bytes() == manifest_bytes
    assert (tmp_path / "pois.jsonl").read_bytes() == artifact_bytes


def test_osm_only_rejects_mock_or_conflicting_row_evidence_before_db_access(tmp_path):
    from copy import deepcopy

    manifest, path = _manifest(tmp_path)
    manifest.update({
        "artifact_kind": "hcmc_osm_poi_v4_osm_only", "osm_only": True,
        "mock_schema_version": None, "mock_seed": None,
    })
    valid = json.loads((tmp_path / "pois.jsonl").read_text(encoding="utf-8"))
    valid["properties"] = {"osm": {}, "mock": {}, "effective": {}}
    valid["rating"] = {"source": "unknown", "value": None, "review_count": 0}
    modifications = [
        lambda row: row["properties"]["mock"].update({"parking": {"source": "mock", "value": True}}),
        lambda row: row["properties"]["effective"].update({"parking": {"source": "mock", "value": True}}),
        lambda row: row["rating"].update({"source": "mock", "value": 5.0}),
        lambda row: row["rating"].update({"source": "unknown", "value": 4.5}),
        lambda row: row["rating"].update({"mock_seed": 17}),
        lambda row: row.pop("rating"),
    ]
    for modify in modifications:
        row = deepcopy(valid)
        modify(row)
        row["content_hash"] = ""
        row["content_hash"] = stable_hash(row)
        raw = (json.dumps(row) + "\n").encode()
        (tmp_path / "pois.jsonl").write_bytes(raw)
        manifest["pois_jsonl_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest["content_set_hash"] = stable_hash([(row["poi_id"], row["content_hash"])])
        path.write_text(json.dumps(manifest), encoding="utf-8")
        connection = ReadyConnection(manifest["content_set_hash"])
        with pytest.raises(ValueError, match="OSM-only POI row"):
            sync_poi_artifact(ReadyDatabase(connection), path, activate=False)
        assert connection.statements == []


@pytest.mark.parametrize("fields", [
    {"mock_schema_version": None, "mock_seed": None},
    {"artifact_kind": "hcmc_osm_poi_v3_osm_only", "osm_only": True,
     "mock_schema_version": None, "mock_seed": 17},
    {"artifact_kind": "hcmc_osm_poi_v4_osm_only", "osm_only": True,
     "mock_schema_version": "poi-mock-v2", "mock_seed": None},
])
def test_inconsistent_osm_only_mock_metadata_rejected_before_db_access(tmp_path, fields):
    manifest, path = _manifest(tmp_path)
    manifest.update(fields)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    connection = ReadyConnection(manifest["content_set_hash"])
    with pytest.raises(ValueError, match="OSM-only POI manifest"):
        sync_poi_artifact(ReadyDatabase(connection), path, activate=False)
    assert connection.statements == []
