"""Offline contract tests: no Postgres connection or migration is opened."""
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from model_intent.embedding_sync import stable_hash
from model_intent.poi_sync import promote_generation_pair, rollback_generation_pair
from model_intent.rendering import PROFILE_RENDERER_VERSION, PROFILE_SCHEMA_VERSION


URL = "postgresql://model_intent@127.0.0.1:5433/model_intent_unified"
TAXONOMY = "a" * 64
MAPPING = "b" * 64
PROPERTY = "c" * 64
MODEL_METADATA = {
    "adapter": "sentence-transformer-v1", "model_key": "e5-base-multilingual",
    "model_id": "intfloat/multilingual-e5-base", "revision": "pinned-revision",
    "query_prefix": "query: ", "document_prefix": "passage: ",
    "normalize_embeddings": True, "dimension": 768, "max_seq_length": 128,
}
MODEL = stable_hash(MODEL_METADATA)
SOURCE = "d" * 64
PROFILE_TEXT = "healthy place"
PROFILE_HASH = stable_hash({"item_id": "23", "language": "en", "text": PROFILE_TEXT})
PROFILE_SET = stable_hash([("23", "en", PROFILE_HASH)])
PROVENANCE = {
    "taxonomy_fingerprint": TAXONOMY, "mapping_sha256": MAPPING,
    "property_taxonomy_sha256": PROPERTY, "model_config_hash": MODEL,
    "source_fingerprint": SOURCE, "embedding_content_set_hash": PROFILE_SET,
    "renderer_version": PROFILE_RENDERER_VERSION,
    "profile_schema_version": PROFILE_SCHEMA_VERSION,
    "expected_profile_count": 1, "expected_category_count": 1,
}


class PromotionConnection:
    def __init__(self, *, fail_poi_swap=False):
        self.pointers = {"model_key": 11, "dataset_key": 12}
        self.activated_at = {"model_key": None, "dataset_key": None}
        self.stmts = []
        self.fail_poi_swap = fail_poi_swap
        self._result = None
        self._time = datetime(2026, 10, 2, tzinfo=timezone.utc)

    @contextmanager
    def transaction(self):
        old = dict(self.pointers)
        old_time = dict(self.activated_at)
        try:
            yield
        except Exception:
            self.pointers = old
            self.activated_at = old_time
            raise

    def execute(self, statement, params=None):
        self.stmts.append((statement, params))
        if "SELECT generation_id FROM active_category_profile_generation_v3" in statement:
            self._result = (self.pointers["model_key"],) if self.pointers["model_key"] else None
        elif "SELECT generation_id FROM active_poi_dataset_generation" in statement:
            self._result = (self.pointers["dataset_key"],) if self.pointers["dataset_key"] else None
        elif "SELECT activated_at FROM active_category_profile_generation_v3" in statement:
            self._result = (self.activated_at["model_key"],)
        elif "SELECT activated_at FROM active_poi_dataset_generation" in statement:
            self._result = (self.activated_at["dataset_key"],)
        elif "SELECT clock_timestamp()" in statement:
            self._time += timedelta(seconds=1)
            self._result = (self._time,)
        elif "INSERT INTO active_category_profile_generation_v3" in statement:
            self._result = self._swap("model_key", params)
        elif "INSERT INTO active_poi_dataset_generation" in statement:
            self._result = None if self.fail_poi_swap else self._swap("dataset_key", params)
        elif "UPDATE active_category_profile_generation_v3" in statement:
            self._result = self._restore("model_key", params)
        elif "UPDATE active_poi_dataset_generation" in statement:
            self._result = self._restore("dataset_key", params)
        elif "DELETE FROM active_" in statement:
            key = "model_key" if "active_category_profile" in statement else "dataset_key"
            self._result = (params[1],) if self.pointers[key] == params[1] else None
            if self._result:
                self.pointers[key] = None
        else:
            self._result = (1,)
        return self

    def _swap(self, key, params):
        _, target, activated_at, previous = params
        if self.pointers[key] != previous:
            return None
        self.pointers[key] = target
        self.activated_at[key] = activated_at
        return (target,)

    def _restore(self, key, params):
        original, _, promoted = params
        if self.pointers[key] != promoted:
            return None
        self.pointers[key] = original
        self.activated_at[key] = None
        return (original,)

    def fetchone(self):
        return self._result


class PromotionDatabase:
    database_url = URL

    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connection(self):
        yield self.conn, None


def kwargs(tmp_path):
    return {**PROVENANCE, "rollback_record_path": tmp_path / "rollback.json"}


def test_promote_and_rollback_both_pointers_in_one_transaction(monkeypatch, tmp_path):
    conn = PromotionConnection()
    monkeypatch.setattr("model_intent.poi_sync._validated_pair", lambda *args, **kw: kw)
    monkeypatch.setattr("model_intent.poi_sync._previous_pair_provenance", lambda *args, **kw: {
        **PROVENANCE, "dataset_key": "hcmc-osm",
    } if args[1] is not None else None)
    record = promote_generation_pair(PromotionDatabase(conn), 21, 22, **kwargs(tmp_path))
    assert conn.pointers == {"model_key": 21, "dataset_key": 22}
    assert record["previous"] == {"embedding_generation_id": 11, "poi_generation_id": 12}
    assert json.loads((tmp_path / "rollback.json").read_text())["promoted"] == record["promoted"]
    assert "DATABASE_URL" not in (tmp_path / "rollback.json").read_text()
    assert rollback_generation_pair(PromotionDatabase(conn), tmp_path / "rollback.json")["restored"] == record["previous"]
    assert conn.pointers == {"model_key": 11, "dataset_key": 12}
    assert len([sql for sql, params in conn.stmts if "pg_advisory_xact_lock" in sql and params == ("model-intent:e5-poi-paired-promotion",)]) == 2


def test_promotion_refuses_existing_record_before_any_pointer_writes(monkeypatch, tmp_path):
    conn = PromotionConnection()
    monkeypatch.setattr("model_intent.poi_sync._validated_pair", lambda *args, **kw: kw)
    monkeypatch.setattr("model_intent.poi_sync._previous_pair_provenance", lambda *args, **kw: {
        **PROVENANCE, "dataset_key": "hcmc-osm",
    } if args[1] is not None else None)
    path = tmp_path / "rollback.json"
    path.write_text("already here")
    with pytest.raises(ValueError, match="overwrite"):
        promote_generation_pair(PromotionDatabase(conn), 21, 22, **kwargs(tmp_path))
    assert path.read_text() == "already here"
    assert conn.stmts == []


def test_pair_swap_failure_is_transactionally_rolled_back(monkeypatch, tmp_path):
    conn = PromotionConnection(fail_poi_swap=True)
    monkeypatch.setattr("model_intent.poi_sync._validated_pair", lambda *args, **kw: kw)
    monkeypatch.setattr("model_intent.poi_sync._previous_pair_provenance", lambda *args, **kw: {
        **PROVENANCE, "dataset_key": "hcmc-osm",
    } if args[1] is not None else None)
    with pytest.raises(RuntimeError, match="both updates rolled back"):
        promote_generation_pair(PromotionDatabase(conn), 21, 22, **kwargs(tmp_path))
    assert conn.pointers == {"model_key": 11, "dataset_key": 12}
    assert (tmp_path / "rollback.json").exists()  # Unused record is harmless: CAS fails.
    with pytest.raises(RuntimeError, match="active pair differs"):
        rollback_generation_pair(PromotionDatabase(conn), tmp_path / "rollback.json")


def test_stale_failed_record_cannot_rollback_later_promotion_to_same_pair(monkeypatch, tmp_path):
    conn = PromotionConnection(fail_poi_swap=True)
    monkeypatch.setattr("model_intent.poi_sync._validated_pair", lambda *args, **kw: kw)
    monkeypatch.setattr("model_intent.poi_sync._previous_pair_provenance", lambda *args, **kw: {
        **PROVENANCE, "dataset_key": "hcmc-osm",
    } if args[1] is not None else None)
    with pytest.raises(RuntimeError, match="both updates rolled back"):
        promote_generation_pair(PromotionDatabase(conn), 21, 22, **kwargs(tmp_path))
    stale = tmp_path / "rollback.json"
    assert stale.exists() and conn.pointers == {"model_key": 11, "dataset_key": 12}

    conn.pointers = {"model_key": 31, "dataset_key": 32}
    conn.fail_poi_swap = False
    fresh = tmp_path / "fresh.json"
    promote_generation_pair(PromotionDatabase(conn), 21, 22, **{
        **kwargs(tmp_path), "rollback_record_path": fresh,
    })
    assert conn.pointers == {"model_key": 21, "dataset_key": 22}
    with pytest.raises(RuntimeError, match="promotion event"):
        rollback_generation_pair(PromotionDatabase(conn), stale)
    assert conn.pointers == {"model_key": 21, "dataset_key": 22}
    rollback_generation_pair(PromotionDatabase(conn), fresh)
    assert conn.pointers == {"model_key": 31, "dataset_key": 32}


def test_promote_from_empty_pointers_and_rollback_deletes_only_new_pair(monkeypatch, tmp_path):
    conn = PromotionConnection()
    conn.pointers = {"model_key": None, "dataset_key": None}
    monkeypatch.setattr("model_intent.poi_sync._validated_pair", lambda *args, **kw: kw)
    monkeypatch.setattr("model_intent.poi_sync._previous_pair_provenance", lambda *args, **kw: {
        **PROVENANCE, "dataset_key": "hcmc-osm",
    } if args[1] is not None else None)
    promote_generation_pair(PromotionDatabase(conn), 21, 22, **kwargs(tmp_path))
    assert conn.pointers == {"model_key": 21, "dataset_key": 22}
    rollback_generation_pair(PromotionDatabase(conn), tmp_path / "rollback.json")
    assert conn.pointers == {"model_key": None, "dataset_key": None}


def test_rollback_refuses_stale_or_diverged_pair(monkeypatch, tmp_path):
    conn = PromotionConnection()
    monkeypatch.setattr("model_intent.poi_sync._validated_pair", lambda *args, **kw: kw)
    monkeypatch.setattr("model_intent.poi_sync._previous_pair_provenance", lambda *args, **kw: {
        **PROVENANCE, "dataset_key": "hcmc-osm",
    } if args[1] is not None else None)
    promote_generation_pair(PromotionDatabase(conn), 21, 22, **kwargs(tmp_path))
    conn.pointers["dataset_key"] = 31
    with pytest.raises(RuntimeError, match="active pair differs"):
        rollback_generation_pair(PromotionDatabase(conn), tmp_path / "rollback.json")
    assert conn.pointers == {"model_key": 21, "dataset_key": 31}


def test_validation_rejects_wrong_provenance_before_pointer_swap(monkeypatch, tmp_path):
    from model_intent.poi_sync import _validated_pair

    class VerificationConnection:
        def __init__(self, embedded_fingerprint=TAXONOMY, poi_mapping=MAPPING):
            self.embedded_fingerprint = embedded_fingerprint
            self.poi_mapping = poi_mapping
            self.executed = []

        def execute(self, sql, params):
            self.executed.append(sql)
            if "FROM embedding_generations g" in sql:
                self.result = (
                    "ready", "category_profile_v3", 1, 1, "e" * 64,
                    self.embedded_fingerprint, "e5-base-multilingual", MODEL, 768,
                    1, 1, "e" * 64, 1,
                    MODEL_METADATA["adapter"], MODEL_METADATA["model_id"], MODEL_METADATA["revision"],
                    MODEL_METADATA["query_prefix"], MODEL_METADATA["document_prefix"],
                    MODEL_METADATA["normalize_embeddings"], MODEL_METADATA["max_seq_length"],
                    SOURCE, PROFILE_SCHEMA_VERSION, PROFILE_RENDERER_VERSION, PROFILE_RENDERER_VERSION,
                )
            elif "FROM category_search_profiles_v3" in sql:
                self.result = []
            elif "FROM poi_dataset_generations" in sql:
                self.result = (
                    "ready", "hcmc-osm", 1, 1, "f" * 64, TAXONOMY,
                    self.poi_mapping, PROPERTY, "1" * 64, "2" * 64, "artifact",
                )
            return self

        def fetchone(self):
            return self.result

        def fetchall(self):
            return self.result

    parameters = {**PROVENANCE, "dataset_key": "hcmc-osm"}
    with pytest.raises(ValueError, match="taxonomy/model"):
        _validated_pair(VerificationConnection(embedded_fingerprint="0" * 64), 21, 22, **parameters)

    class DriftedModelConnection(VerificationConnection):
        def execute(self, sql, params):
            super().execute(sql, params)
            if "FROM embedding_generations g" in sql:
                fields = list(self.result)
                fields[15] = "unexpected-revision"
                self.result = tuple(fields)
            return self

    with pytest.raises(ValueError, match="taxonomy/model"):
        _validated_pair(DriftedModelConnection(), 21, 22, **parameters)

    profile_text = PROFILE_TEXT
    profile_hash = PROFILE_HASH
    profiles_hash = PROFILE_SET

    class ValidProfilesConnection(VerificationConnection):
        def execute(self, sql, params):
            super().execute(sql, params)
            if "FROM embedding_generations g" in sql:
                data = list(self.result)
                data[4] = profiles_hash
                data[11] = profiles_hash
                self.result = tuple(data)
            elif "FROM category_search_profiles_v3" in sql:
                self.result = [(23, "en", profile_text, profile_hash)]
            return self

    with pytest.raises(ValueError, match="mapping/property"):
        _validated_pair(ValidProfilesConnection(poi_mapping="0" * 64), 21, 22, **parameters)

    class DriftedSourceConnection(ValidProfilesConnection):
        def execute(self, sql, params):
            super().execute(sql, params)
            if "FROM embedding_generations g" in sql:
                data = list(self.result)
                data[20] = "0" * 64
                self.result = tuple(data)
            return self

    with pytest.raises(ValueError, match="taxonomy/model"):
        _validated_pair(DriftedSourceConnection(), 21, 22, **parameters)

    class DriftedRendererConnection(ValidProfilesConnection):
        def execute(self, sql, params):
            super().execute(sql, params)
            if "FROM embedding_generations g" in sql:
                data = list(self.result)
                data[22] = "wrong-renderer"
                self.result = tuple(data)
            return self

    with pytest.raises(ValueError, match="taxonomy/model"):
        _validated_pair(DriftedRendererConnection(), 21, 22, **parameters)

    monkeypatch.setattr("model_intent.poi_sync._database_poi_content_hash", lambda *args: "f" * 64)
    provenance = _validated_pair(ValidProfilesConnection(), 21, 22, **parameters)
    assert provenance["embedding_content_set_hash"] == profiles_hash
    assert provenance["poi_content_set_hash"] == "f" * 64
    monkeypatch.setattr("model_intent.poi_sync._database_poi_content_hash", lambda *args: "0" * 64)
    with pytest.raises(ValueError, match="content set hash"):
        _validated_pair(ValidProfilesConnection(), 21, 22, **parameters)


def test_rollback_rejects_tampered_record_before_database_calls(monkeypatch, tmp_path):
    conn = PromotionConnection()
    monkeypatch.setattr("model_intent.poi_sync._validated_pair", lambda *args, **kw: kw)
    monkeypatch.setattr("model_intent.poi_sync._previous_pair_provenance", lambda *args, **kw: {
        **PROVENANCE, "dataset_key": "hcmc-osm",
    } if args[1] is not None else None)
    promote_generation_pair(PromotionDatabase(conn), 21, 22, **kwargs(tmp_path))
    record_path = tmp_path / "rollback.json"
    record = json.loads(record_path.read_text())
    record["previous"]["poi_generation_id"] = 999
    record_path.write_text(json.dumps(record))
    previous_count = len(conn.stmts)
    with pytest.raises(ValueError, match="checksum mismatch"):
        rollback_generation_pair(PromotionDatabase(conn), record_path)
    assert len(conn.stmts) == previous_count


def test_database_inventory_streams_rows_and_rejects_wrong_count():
    from model_intent.poi_sync import _database_poi_content_hash

    rows = [("osm:node:1", "1" * 64), ("osm:way:2", "2" * 64)]

    class StreamingConnection:
        def cursor(self, *, name):
            assert name == "verify_poi_generation_42"

            class Cursor:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    pass

                def execute(self, sql, params):
                    assert "ORDER BY CASE osm_type" in sql
                    assert params == (42,)

                def __iter__(self):
                    return iter(rows)

            return Cursor()

    assert _database_poi_content_hash(StreamingConnection(), 42, 2) == stable_hash(rows)
    with pytest.raises(ValueError, match="row count mismatch"):
        _database_poi_content_hash(StreamingConnection(), 42, 3)


def test_guard_refuses_legacy_url_without_database_calls(tmp_path):
    db = PromotionDatabase(PromotionConnection())
    db.database_url = "postgresql://model_intent@127.0.0.1:5432/model_intent"
    with pytest.raises(RuntimeError, match="isolated unified database"):
        promote_generation_pair(db, 21, 22, **kwargs(tmp_path))
    assert db.conn.stmts == []
