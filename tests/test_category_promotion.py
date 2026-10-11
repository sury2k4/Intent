"""Offline category promotion contract tests: never connect to or write a database."""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from model_intent.category_promotion import (
    _pinned_provenance,
    _validated_category,
    promote_category_generation,
    rollback_category_generation,
)
from model_intent.embedding_sync import MODEL_KEY, stable_hash
from model_intent.rendering import PROFILE_RENDERER_VERSION, PROFILE_SCHEMA_VERSION


URL = "postgresql://model_intent@127.0.0.1:5433/model_intent_unified"
MODEL = {
    "adapter": "sentence-transformer-v1", "model_key": MODEL_KEY,
    "model_id": "intfloat/multilingual-e5-base", "revision": "pinned-revision",
    "query_prefix": "query: ", "document_prefix": "passage: ",
    "normalize_embeddings": True, "dimension": 768, "max_seq_length": 128,
}
TEXT = "Category: sample"
ROW_HASH = stable_hash({"item_id": "23", "language": "en", "text": TEXT})
SET_HASH = stable_hash([("23", "en", ROW_HASH)])
PINNED = {
    "taxonomy_fingerprint": "a" * 64,
    "model_config_hash": stable_hash(MODEL),
    "source_fingerprint": "b" * 64,
    "content_set_hash": SET_HASH,
    "renderer_version": PROFILE_RENDERER_VERSION,
    "profile_schema_version": PROFILE_SCHEMA_VERSION,
    "profile_count": 1, "category_count": 1,
    "model_id": MODEL["model_id"], "revision": MODEL["revision"], "dimension": 768,
}
RENDERED = {(23, "en"): (TEXT, ROW_HASH)}


def generation_row(**changes):
    row = [
        "ready", "category_profile_v3", 1, 1, SET_HASH,
        PINNED["taxonomy_fingerprint"], MODEL_KEY, PINNED["model_config_hash"],
        768, 1, 1, SET_HASH, 1, MODEL["adapter"], MODEL["model_id"],
        MODEL["revision"], MODEL["query_prefix"], MODEL["document_prefix"],
        MODEL["normalize_embeddings"], MODEL["max_seq_length"],
        PINNED["source_fingerprint"], PROFILE_SCHEMA_VERSION,
        PROFILE_RENDERER_VERSION, PROFILE_RENDERER_VERSION,
    ]
    columns = {"state": 0, "kind": 1, "expected": 2, "completed": 3,
               "content_set_hash": 4, "taxonomy": 5, "key": 6,
               "model_hash": 7, "dimension": 8, "set_profiles": 9,
               "categories": 10, "set_hash": 11, "set_taxonomy_id": 12,
               "revision": 15, "source": 20, "schema": 21,
               "renderer": 22, "set_renderer": 23}
    for name, value in changes.items():
        row[columns[name]] = value
    return tuple(row)


class FakeConnection:
    def __init__(self, *, previous=11, fail_swap=False):
        self.pointer = previous
        self.poi_pointer = 77
        self.timestamp = None
        self.fail_swap = fail_swap
        self.rows = {11: generation_row(), 21: generation_row()}
        self.profiles = {11: [(23, "en", TEXT, ROW_HASH)], 21: [(23, "en", TEXT, ROW_HASH)]}
        self.statements = []
        self._result = None
        self.now = datetime(2026, 10, 7, tzinfo=timezone.utc)

    @contextmanager
    def transaction(self):
        prior = (self.pointer, self.timestamp, self.poi_pointer)
        try:
            yield
        except Exception:
            self.pointer, self.timestamp, self.poi_pointer = prior
            raise

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if "FROM embedding_generations g" in sql:
            self._result = self.rows.get(params[0])
        elif "FROM category_search_profiles_v3" in sql:
            self._result = self.profiles.get(params[0], [])
        elif "SELECT generation_id FROM active_category_profile_generation_v3" in sql:
            self._result = (self.pointer,) if self.pointer is not None else None
        elif "SELECT activated_at FROM active_category_profile_generation_v3" in sql:
            self._result = (self.timestamp,) if self.pointer is not None else None
        elif "SELECT clock_timestamp()" in sql:
            self.now += timedelta(seconds=1)
            self._result = (self.now,)
        elif "INSERT INTO active_category_profile_generation_v3" in sql:
            _, target, activated_at, expected = params
            if self.fail_swap or self.pointer != expected:
                self._result = None
            else:
                self.pointer, self.timestamp = target, activated_at
                self._result = (target,)
        elif "DELETE FROM active_category_profile_generation_v3" in sql:
            _, target, activated_at = params
            self._result = (target,) if (self.pointer, self.timestamp) == (target, activated_at) else None
            if self._result:
                self.pointer, self.timestamp = None, None
        elif "UPDATE active_category_profile_generation_v3" in sql:
            original, _, target, activated_at = params
            self._result = (original,) if (self.pointer, self.timestamp) == (target, activated_at) else None
            if self._result:
                self.pointer, self.timestamp = original, self.now
        else:
            self._result = (1,)
        return self

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result


class FakeDatabase:
    database_url = URL

    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connection(self):
        yield self.conn, None


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(
        "model_intent.category_promotion._pinned_provenance",
        lambda *args: (dict(PINNED), dict(RENDERED)),
    )


def promote(db, record):
    return promote_category_generation(
        db, 21, config_path="pinned.yaml", taxonomy_path="taxonomy.json",
        rollback_record_path=record,
    )


def test_promote_and_rollback_only_category_pointer(pinned, tmp_path):
    conn = FakeConnection()
    db = FakeDatabase(conn)
    record_path = tmp_path / "rollback.json"
    saved = promote(db, record_path)
    assert conn.pointer == 21 and conn.poi_pointer == 77
    assert saved["previous_generation_id"] == 11
    assert saved["provenance"] == PINNED
    assert json.loads(record_path.read_text()) == {k: v for k, v in saved.items() if k != "rollback_record"}
    assert rollback_category_generation(db, record_path)["restored_generation_id"] == 11
    assert conn.pointer == 11 and conn.poi_pointer == 77
    assert not any("active_poi_dataset_generation" in sql or "UPDATE embedding_generations" in sql
                   or "DELETE FROM category_search_profiles_v3" in sql for sql, _ in conn.statements)
    assert sum("pg_advisory_xact_lock" in sql and params == ("model-intent:e5-poi-paired-promotion",)
               for sql, params in conn.statements) == 2


def test_promote_from_empty_pointer_and_rollback_deletes_only_category(pinned, tmp_path):
    conn = FakeConnection(previous=None)
    db = FakeDatabase(conn)
    promote(db, tmp_path / "rollback.json")
    assert conn.pointer == 21
    rollback_category_generation(db, tmp_path / "rollback.json")
    assert conn.pointer is None and conn.poi_pointer == 77


def test_existing_record_is_never_overwritten(pinned, tmp_path):
    conn = FakeConnection()
    record_path = tmp_path / "rollback.json"
    record_path.write_text("original")
    with pytest.raises(ValueError, match="overwrite"):
        promote(FakeDatabase(conn), record_path)
    assert record_path.read_text() == "original"
    assert conn.statements == []


def test_swap_failure_keeps_pointer_and_unused_record_cannot_rollback(pinned, tmp_path):
    conn = FakeConnection(fail_swap=True)
    db = FakeDatabase(conn)
    record_path = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="active pointer changed"):
        promote(db, record_path)
    assert conn.pointer == 11 and conn.poi_pointer == 77
    assert record_path.exists()
    with pytest.raises(RuntimeError, match="pointer differs"):
        rollback_category_generation(db, record_path)


def test_rollback_refuses_changed_pointer_and_stale_event(pinned, tmp_path):
    conn = FakeConnection()
    db = FakeDatabase(conn)
    record_path = tmp_path / "record.json"
    promote(db, record_path)
    conn.pointer = 32
    with pytest.raises(RuntimeError, match="pointer differs"):
        rollback_category_generation(db, record_path)
    conn.pointer, conn.timestamp = 21, conn.timestamp + timedelta(seconds=10)
    with pytest.raises(RuntimeError, match="promotion event"):
        rollback_category_generation(db, record_path)
    assert conn.pointer == 21 and conn.poi_pointer == 77


def test_rollback_refuses_tampered_record_before_database_calls(pinned, tmp_path):
    conn = FakeConnection()
    record_path = tmp_path / "record.json"
    promote(FakeDatabase(conn), record_path)
    record = json.loads(record_path.read_text())
    record["previous_generation_id"] = 9000
    record_path.write_text(json.dumps(record))
    before = len(conn.statements)
    with pytest.raises(ValueError, match="checksum mismatch"):
        rollback_category_generation(FakeDatabase(conn), record_path)
    assert len(conn.statements) == before and conn.pointer == 21


@pytest.mark.parametrize("problem", [
    {"state": "failed"}, {"kind": "variant_v1"}, {"expected": 2},
    {"completed": 0}, {"set_profiles": 0}, {"categories": 2},
    {"set_hash": "0" * 64}, {"taxonomy": "0" * 64},
    {"source": "0" * 64}, {"renderer": "old"},
    {"set_renderer": "old"}, {"schema": 2},
    {"revision": "changed"}, {"model_hash": "0" * 64},
    {"dimension": 384}, {"set_taxonomy_id": 0},
])
def test_target_rejects_incompatible_generation_metadata(problem, pinned, tmp_path):
    conn = FakeConnection(previous=None)
    conn.rows[21] = generation_row(**problem)
    with pytest.raises(ValueError):
        promote(FakeDatabase(conn), tmp_path / "record.json")
    assert conn.pointer is None and conn.poi_pointer == 77
    assert not (tmp_path / "record.json").exists()


@pytest.mark.parametrize("profiles", [
    [], [(23, "en", TEXT, "0" * 64)],
    [(23, "en", TEXT, ROW_HASH), (23, "en", TEXT, ROW_HASH)],
    [(24, "en", TEXT, ROW_HASH)],
])
def test_target_rejects_missing_or_corrupt_candidate_rows(profiles, pinned, tmp_path):
    conn = FakeConnection(previous=None)
    conn.profiles[21] = profiles
    with pytest.raises(ValueError, match="candidate"):
        promote(FakeDatabase(conn), tmp_path / "record.json")
    assert conn.pointer is None and not (tmp_path / "record.json").exists()


def test_previous_generation_drift_refuses_rollback(pinned, tmp_path):
    conn = FakeConnection()
    db = FakeDatabase(conn)
    record_path = tmp_path / "record.json"
    promote(db, record_path)
    conn.profiles[11] = []
    with pytest.raises(RuntimeError, match="previous category generation"):
        rollback_category_generation(db, record_path)
    assert conn.pointer == 21 and conn.poi_pointer == 77


def test_rollback_restores_previous_even_if_promoted_rows_are_damaged(pinned, tmp_path):
    conn = FakeConnection()
    db = FakeDatabase(conn)
    record_path = tmp_path / "record.json"
    promote(db, record_path)
    conn.profiles[21] = []
    restored = rollback_category_generation(db, record_path)
    assert restored["restored_generation_id"] == 11
    assert conn.pointer == 11 and conn.poi_pointer == 77


@pytest.mark.parametrize("invalid_id", ["01", "+1", "1.0"])
def test_pinned_provenance_refuses_noncanonical_category_ids(monkeypatch, invalid_id):
    from model_intent import category_promotion as promotion

    root = Path(__file__).resolve().parents[1]
    config = root / "configs/benchmark.poi.pg.v3.yaml"
    taxonomy = root / "data/taxonomies/place_types.poc.v3.json"
    monkeypatch.setattr(promotion, "render_category_profiles", lambda *args, **kwargs: [
        type("Profile", (), {"item_id": invalid_id, "language": "en", "text": TEXT})(),
    ])
    with pytest.raises(ValueError, match="canonical numeric"):
        _pinned_provenance(config, taxonomy, FakeDatabase(FakeConnection()))


def test_invalid_database_target_rejected_before_sql(pinned, tmp_path):
    db = FakeDatabase(FakeConnection())
    db.database_url = "postgresql://model_intent@127.0.0.1:5432/model_intent"
    with pytest.raises(RuntimeError, match="isolated unified database"):
        promote(db, tmp_path / "rollback.json")
    assert db.conn.statements == []


def test_real_pinned_provenance_matches_renderer_without_database_access(tmp_path):
    root = Path(__file__).resolve().parents[1]
    taxonomy = root / "data" / "taxonomies" / "place_types.poc.v3.json"
    enrichment = root / "data" / "taxonomies" / "place_type_enrichment.rich.poc.v3.json"
    config = tmp_path / "category.pg.yaml"
    config.write_text(
        "retrieval_backend: pgvector\nretrieval_mode: pg_exact\nlanguages: [vi, en]\n"
        f"taxonomy_path: '{taxonomy.as_posix()}'\n"
        f"enrichment_path: '{enrichment.as_posix()}'\n"
        "models:\n  - key: e5-base-multilingual\n"
        "    model_id: intfloat/multilingual-e5-base\n"
        "    expected_dimension: 768\n"
        "    revision: d128750597153bb5987e10b1c3493a34e5a4502a\n"
        "    query_prefix: 'query: '\n    document_prefix: 'passage: '\n"
        "    max_seq_length: 192\n",
        encoding="utf-8",
    )
    expected, rendered = _pinned_provenance(config, taxonomy, FakeDatabase(FakeConnection()))
    assert expected["profile_count"] == len(rendered)
    assert expected["category_count"] == len({item_id for item_id, _ in rendered})
    assert expected["dimension"] == 768
    assert expected["model_config_hash"] == stable_hash({
        "adapter": MODEL["adapter"], "model_key": MODEL_KEY,
        "model_id": MODEL["model_id"], "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "query_prefix": MODEL["query_prefix"], "document_prefix": MODEL["document_prefix"],
        "normalize_embeddings": True, "dimension": 768, "max_seq_length": 192,
    })


def test_direct_generation_validation_matches_pinned_and_actual_rows():
    conn = FakeConnection(previous=None)
    assert _validated_category(conn, 21, expected=PINNED, rendered_rows=RENDERED) == PINNED
    assert any("FOR SHARE OF g" in sql for sql, _ in conn.statements)
