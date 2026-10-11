"""No real database, migration, model download, or live writes in this suite."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from model_intent import property_embedding_sync as sync
from model_intent.migrations import discover_migrations

URL = "postgresql://model_intent@127.0.0.1:5433/model_intent_unified"


def corpus_fixture() -> SimpleNamespace:
    rows = []
    for number in range(70):
        for language in ("en", "vi"):
            property_id = f"property_{number:02d}"
            text = f"{language} definition of {property_id}"
            content_hash = sync.stable_hash({
                "property_id": property_id, "language": language,
                "profile_key": "primary", "text": text,
            })
            rows.append(SimpleNamespace(
                property_id=property_id, language=language, profile_key="primary",
                text=text, content_hash=content_hash,
            ))
    content_set_hash = sync.stable_hash([
        (row.property_id, row.language, row.profile_key, row.content_hash) for row in rows
    ])
    return SimpleNamespace(
        rows=tuple(rows), property_count=70, profile_count=len(rows),
        catalog_hash="a" * 64, source_hash="b" * 64,
        taxonomy_fingerprint="c" * 64, renderer_version="property-semantic-v1",
        content_set_hash=content_set_hash,
    )


class FakeConnection:
    def __init__(self, corpus, *, state=None, active=None, ledger_bad=None, schema_missing=None):
        self.corpus = corpus
        self.state = state
        self.active = active
        self.ledger_bad = ledger_bad
        self.schema_missing = schema_missing
        self.profile_set = state is not None
        self.written = {
            (row.property_id, row.language, row.profile_key): (row.text, row.content_hash, 768, 1.0)
            for row in corpus.rows
        } if state == "ready" else {}
        self.completed = len(self.written)
        self.statements = []
        self.locked = False
        self.model_inserted = False
        self.fail_after_batches = False

    @contextmanager
    def transaction(self):
        yield

    def execute(self, statement, params=None):
        sql = " ".join(statement.split())
        self.statements.append((sql, params))
        result = None
        if "to_regclass(" in sql:
            result = (None if params[0] == self.schema_missing else params[0],)
        elif sql.startswith("SELECT version, checksum FROM schema_migrations"):
            root = {migration.version: migration.checksum for migration in discover_migrations(sync.ROOT_MIGRATIONS)}
            scoped = discover_migrations(sync.PROPERTY_MIGRATIONS)
            root[scoped[0].version] = scoped[0].checksum
            self.result = [(version, "0" * 64 if version == self.ledger_bad else root[version])
                           for version in params[0] if version != self.ledger_bad or self.ledger_bad != "missing"]
            return self
        elif "pg_advisory_lock" in sql:
            self.locked = True
        elif "pg_advisory_unlock" in sql:
            self.locked = False
        elif sql.startswith("INSERT INTO embedding_model_configs("):
            self.model_inserted = True
            result = (1,)
        elif sql.startswith("INSERT INTO property_profile_sets("):
            result = None if self.profile_set else (3,)
            self.profile_set = True
        elif sql.startswith("SELECT id, property_count, profile_count FROM property_profile_sets"):
            result = (3, 70, len(self.corpus.rows))
        elif sql.startswith("INSERT INTO property_embedding_generations("):
            if self.state is None:
                self.state = "building"
                self.completed = 0
                result = (17, "building", len(self.corpus.rows), 0, self.corpus.content_set_hash)
        elif sql.startswith("SELECT id, state, expected_count, completed_count, content_set_hash"):
            result = (17, self.state, len(self.corpus.rows), self.completed, self.corpus.content_set_hash)
        elif sql.startswith("SELECT generation_id FROM active_property_embedding_generation"):
            result = (self.active,) if self.active is not None else None
        elif sql.startswith("SELECT property_id, language, profile_key, search_text, content_hash,"):
            self.result = [(*key, *data) for key, data in self.written.items()]
            return self
        elif sql.startswith("SELECT property_id, language, profile_key FROM property_semantic_profiles"):
            self.result = list(self.written)
            return self
        elif sql.startswith("SELECT count(*) FROM property_semantic_profiles"):
            result = (len(self.written),)
        elif sql.startswith("UPDATE property_embedding_generations SET state='building'"):
            self.state = "building"
        elif sql.startswith("UPDATE property_embedding_generations SET completed_count="):
            self.completed = params[0]
            if self.fail_after_batches:
                raise RuntimeError("injected batch failure")
        elif sql.startswith("UPDATE property_embedding_generations SET state='failed'"):
            self.state = "failed"
        elif sql.startswith("UPDATE property_embedding_generations SET state='ready'"):
            self.state = "ready"
            self.completed = params[0]
        elif sql.startswith("SELECT g.id, g.state, g.expected_count"):
            result = None
        elif sql.startswith("SELECT g.id, g.profile_set_id"):
            self.result = [(17, 3, self.state, len(self.corpus.rows), self.completed,
                            None, None, None, self.corpus.content_set_hash, "d" * 64)] if self.state else []
            return self
        else:
            raise AssertionError(f"unexpected SQL: {sql}")
        self.result = result
        return self

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result

    def cursor(self):
        connection = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def executemany(self, sql, values):
                connection.statements.append((" ".join(sql.split()), values))
                for generation_id, property_id, language, profile_key, text, content_hash, embedding in values:
                    assert generation_id == 17
                    assert np.isfinite(embedding).all() and len(embedding) == 768
                    assert abs(float(np.linalg.norm(embedding)) - 1) < 1e-5
                    connection.written[(property_id, language, profile_key)] = (
                        text, content_hash, 768, float(np.linalg.norm(embedding)),
                    )
        return Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass


class FakeDatabase:
    def __init__(self, connection, url=URL):
        self.conn = connection
        self.database_url = url

    @contextmanager
    def connection(self):
        yield self.conn, None


def prepare(monkeypatch, corpus, vectors=None):
    profile = SimpleNamespace(
        key=sync.MODEL_KEY, model_id="intfloat/multilingual-e5-base", revision="pinned",
        query_prefix="query: ", document_prefix="passage: ",
        expected_dimension=768, max_seq_length=128, batch_size=16,
    )
    calls = []

    class FakeEmbedder:
        def metadata(self):
            return {**sync.semantic_model_metadata(profile), "batch_size": 16, "device": "cpu"}

        def encode_documents(self, texts):
            calls.append(tuple(texts))
            if vectors is not None:
                return vectors(len(texts))
            result = np.zeros((len(texts), 768), dtype=np.float32)
            result[:, 0] = 1
            return result

    FakeEmbedder.profile = profile
    FakeEmbedder.model = SimpleNamespace(tokenizer=lambda texts, **_: {
        "input_ids": [[1] * len(text.split()) for text in texts],
    })
    monkeypatch.setattr(sync, "load_yaml", lambda *_: {
        "taxonomy_path": str(Path("taxonomy.json").resolve()),
        "property_intent": {"property_path": "data/properties/property.json"},
    })
    monkeypatch.setattr(sync, "validate_database_target", lambda *_, **__: None)
    monkeypatch.setattr(sync, "model_profile", lambda *_: profile)
    monkeypatch.setattr(sync, "load_property_definitions", lambda *_: {})
    monkeypatch.setattr(sync.Taxonomy, "load", lambda *_: SimpleNamespace(fingerprint=lambda: "c" * 64))
    monkeypatch.setattr(sync.property_semantic, "load_property_semantic_corpus", lambda *args: corpus, raising=False)
    monkeypatch.setattr(sync, "configure_runtime", lambda *_: None)
    monkeypatch.setattr(sync, "make_embedder", lambda *args: FakeEmbedder())
    return calls


def stage(db, **kwargs):
    return sync.stage_property_embeddings(
        db, config_path="configs/benchmark.poi.pg.v3.yaml", taxonomy_path="taxonomy.json",
        semantic_path="semantic.json", **kwargs,
    )


def test_migration_is_scoped_and_independent_of_existing_category_and_poi_tables():
    names = {item.version for item in discover_migrations(sync.PROPERTY_MIGRATIONS)}
    assert names == {sync.PROPERTY_MIGRATION_VERSION}
    sql = (sync.PROPERTY_MIGRATIONS / "property_001_semantic_profiles.sql").read_text(encoding="utf-8")
    assert "embedding vector(768)" in sql
    assert "REFERENCES embedding_model_configs(id)" in sql
    assert "UNIQUE (generation_id, property_id, language, profile_key)" in sql
    assert not any(f"ALTER TABLE {name}" in sql for name in ("pois", "embedding_generations", "category_profile_sets"))
    assert "INSERT INTO active_property_embedding_generation" not in sql


def test_repository_corpus_matches_renderer_and_config_without_model_or_database():
    root = sync.ROOT_MIGRATIONS.parent
    config = sync.load_yaml(root / "configs/benchmark.poi.pg.v3.yaml")
    taxonomy = sync.Taxonomy.load(root / config["taxonomy_path"])
    from model_intent.poi_properties import load_property_definitions

    definitions = load_property_definitions(root / config["property_intent"]["property_path"])
    metadata = sync.semantic_model_metadata(sync.model_profile(config))
    corpus = sync.property_semantic.load_property_semantic_corpus(
        root / "data/properties/property_semantic_profiles.poc.v2.json", definitions, taxonomy, metadata,
    )
    assert len(sync._verified_corpus(corpus)) == 212
    assert corpus.property_count == 106


def test_stage_fails_without_schema_or_valid_base_checksum_before_any_write(monkeypatch):
    corpus = corpus_fixture()
    calls = prepare(monkeypatch, corpus)
    for invalid in (
        FakeConnection(corpus, schema_missing="property_profile_sets"),
        FakeConnection(corpus, ledger_bad="005_category_search_profiles_v3"),
        FakeConnection(corpus, ledger_bad=sync.PROPERTY_MIGRATION_VERSION),
    ):
        with pytest.raises(RuntimeError, match="schema is not installed|ledger mismatch"):
            stage(FakeDatabase(invalid))
        assert not invalid.model_inserted and not invalid.locked and not calls
        assert not any("CREATE TABLE" in sql for sql, _ in invalid.statements)


def test_stage_is_idempotent_ready_reuse_and_never_changes_pointer(monkeypatch):
    corpus = corpus_fixture()
    calls = prepare(monkeypatch, corpus)
    conn = FakeConnection(corpus)
    result = stage(FakeDatabase(conn), batch_size=17)
    assert result.state == "ready" and result.embedded_count == len(corpus.rows)
    assert result.expected_count == result.completed_count == 140 and not result.active
    assert len(calls) == 9 and not conn.locked
    result = stage(FakeDatabase(conn))
    assert result.already_ready and result.embedded_count == 0 and len(calls) == 9
    assert not any("INSERT INTO active_property_embedding_generation" in sql or
                   "UPDATE active_property_embedding_generation" in sql for sql, _ in conn.statements)
    assert not any("008_poi_category_provenance" in str(params) for _, params in conn.statements)


def test_ready_rows_are_immutable_even_if_text_or_hash_is_tampered(monkeypatch):
    corpus = corpus_fixture()
    prepare(monkeypatch, corpus)
    conn = FakeConnection(corpus, state="ready")
    row = corpus.rows[0]
    conn.written[(row.property_id, row.language, row.profile_key)] = ("wrong text", row.content_hash, 768, 1.0)
    with pytest.raises(RuntimeError, match="conflict"):
        stage(FakeDatabase(conn))
    assert conn.state == "ready"
    assert not any("INSERT INTO property_semantic_profiles" in sql for sql, _ in conn.statements)


def test_resume_failed_generation_only_embeds_missing_rows(monkeypatch):
    corpus = corpus_fixture()
    calls = prepare(monkeypatch, corpus)
    conn = FakeConnection(corpus, state="failed", active=5)
    for row in corpus.rows[:7]:
        conn.written[(row.property_id, row.language, row.profile_key)] = (row.text, row.content_hash, 768, 1.0)
    conn.completed = 7
    result = stage(FakeDatabase(conn), batch_size=16)
    assert result.embedded_count == 133 and result.completed_count == 140
    assert sum(len(texts) for texts in calls) == 133 and conn.active == 5
    assert conn.state == "ready"


def test_active_building_generation_is_never_overwritten(monkeypatch):
    corpus = corpus_fixture()
    calls = prepare(monkeypatch, corpus)
    conn = FakeConnection(corpus, state="building", active=17)
    with pytest.raises(RuntimeError, match="active property generation"):
        stage(FakeDatabase(conn))
    assert not calls and conn.state == "building"


@pytest.mark.parametrize("vectors", [
    lambda count: np.zeros((count, 768), dtype=np.float32),
    lambda count: np.full((count, 768), np.nan, dtype=np.float32),
    lambda count: np.ones((count, 384), dtype=np.float32),
])
def test_invalid_vectors_mark_building_generation_failed(monkeypatch, vectors):
    corpus = corpus_fixture()
    prepare(monkeypatch, corpus, vectors=vectors)
    conn = FakeConnection(corpus)
    with pytest.raises(ValueError, match="shape/finite|norm"):
        stage(FakeDatabase(conn))
    assert conn.state == "failed" and not conn.locked


def test_bad_corpus_hash_or_missing_property_rejected_before_writing(monkeypatch):
    corpus = corpus_fixture()
    prepare(monkeypatch, corpus)
    conn = FakeConnection(corpus)
    corpus.rows[0].content_hash = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        stage(FakeDatabase(conn))
    assert not conn.statements
    corpus = corpus_fixture()
    corpus.rows = corpus.rows[2:]
    corpus.profile_count = len(corpus.rows)
    prepare(monkeypatch, corpus)
    with pytest.raises(ValueError, match="70 queryable"):
        stage(FakeDatabase(conn))
    assert not conn.statements


def test_status_is_read_only_and_requires_safe_database(monkeypatch):
    corpus = corpus_fixture()
    conn = FakeConnection(corpus, state="ready")
    payload = sync.property_embedding_status(FakeDatabase(conn))
    assert payload["active"] is None and payload["generations"][0]["state"] == "ready"
    assert all(sql.startswith("SELECT") for sql, _ in conn.statements)
    unsafe = FakeConnection(corpus)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        sync.property_embedding_status(FakeDatabase(unsafe, url="postgresql://localhost:5432/legacy"))
    assert not unsafe.statements
