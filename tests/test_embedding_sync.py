"""Unit-only E5 generation staging tests; never connect to Postgres."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from model_intent import embedding_sync as module


URL = "postgresql://model_intent@127.0.0.1:5433/model_intent_unified"


class EmbeddingConnection:
    def __init__(self, *, ready=False, active=7):
        self.ready = ready
        self.active = active
        self.statements = []
        self.profile = SimpleNamespace(item_id="23", language="en", text="test medical profile")
        self.result = None
        self.fail_ready_content = False
        self.rows_inserted = False

    @contextmanager
    def transaction(self):
        yield

    def execute(self, statement, params=None):
        self.statements.append((statement, params))
        if "INSERT INTO embedding_model_configs" in statement:
            self.result = (1,)
        elif "INSERT INTO taxonomy_snapshots" in statement:
            self.result = (2,)
        elif "INSERT INTO category_profile_sets" in statement:
            self.result = (3,)
        elif "INSERT INTO embedding_generations" in statement:
            self.result = (17, "ready" if self.ready else "building", 1, "category_profile_v3", 3)
        elif "SELECT generation_id FROM active_category_profile_generation_v3" in statement:
            self.result = (self.active,) if self.active is not None else None
        elif "SELECT count(*) FROM category_search_profiles_v3" in statement:
            self.result = (1 if self.ready or "content_hash IS NOT NULL" in statement or self.rows_inserted else 0,)
        elif "SELECT place_type_id, language, content_hash, search_text" in statement:
            self.result = [(23, "en", "f" * 64 if self.fail_ready_content else module.profile_content_hash(self.profile), self.profile.text)]
        elif "SELECT place_type_id, language, content_hash, id" in statement:
            self.result = []
        elif "SELECT count(*) FROM category_search_profiles_v3 WHERE generation_id=%s" in statement:
            self.result = (1,)
        elif "SELECT count(*) FROM category_search_profiles_v3" in statement and "content_hash IS NOT NULL" in statement:
            self.result = (1,)
        elif "INSERT INTO active_category_profile_generation_v3" in statement:
            self.active = params[1]
            self.result = None
        else:
            self.result = None
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
                pass

            def executemany(self, statement, rows):
                connection.statements.append((statement, rows))
                connection.rows_inserted = True

        return Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass


class EmbeddingDatabase:
    database_url = URL

    def __init__(self, connection):
        self.conn = connection

    @contextmanager
    def connection(self):
        yield self.conn, None


def _stub_corpus(monkeypatch, connection):
    profile = SimpleNamespace(
        key=module.MODEL_KEY, model_id="intfloat/multilingual-e5-base", revision="pinned-revision",
        query_prefix="query: ", document_prefix="passage: ",
        expected_dimension=768, max_seq_length=128, batch_size=4,
    )
    taxonomy = SimpleNamespace(
        document=SimpleNamespace(items=[SimpleNamespace(id="23")], taxonomy_version="v3", schema_version=1, source="test"),
        has_children=lambda item_id: False, fingerprint=lambda: "a" * 64,
    )
    monkeypatch.setattr(module, "load_yaml", lambda *_: {"languages": ["en"], "enrichment_path": "ignored"})
    monkeypatch.setattr(module.Taxonomy, "load", lambda *_: taxonomy)
    monkeypatch.setattr(module, "_configure_runtime", lambda *_: None)
    monkeypatch.setattr(module, "validate_database_target", lambda *_: None)
    monkeypatch.setattr(module, "load_enrichment", lambda *_: {})
    monkeypatch.setattr(module, "validate_enrichment", lambda *_ , **__: None)
    monkeypatch.setattr(module, "render_category_profiles", lambda *_ , **__: [connection.profile])
    monkeypatch.setattr(module, "model_profile", lambda *_: profile)
    monkeypatch.setattr(module, "make_embedder", lambda *_: SimpleNamespace(
        dimension=768, encode_documents=lambda texts: np.ones((len(texts), 768), dtype=np.float32),
    ))


def test_ready_generation_reuse_no_activate_never_writes_pointer(monkeypatch):
    connection = EmbeddingConnection(ready=True)
    _stub_corpus(monkeypatch, connection)
    result = module.sync_embeddings(EmbeddingDatabase(connection), config_path="config.yml", taxonomy_path="taxonomy.json", activate=False)
    assert result.already_ready and result.active is False and result.embedded_count == 0
    assert connection.active == 7
    assert not any("INSERT INTO active_category_profile_generation_v3" in sql for sql, _ in connection.statements)
    assert not any("UPDATE embedding_generations SET state='building'" in sql for sql, _ in connection.statements)
    assert all("DO UPDATE" not in sql for sql, _ in connection.statements)


def test_no_activate_reports_true_when_same_ready_generation_is_already_active(monkeypatch):
    connection = EmbeddingConnection(ready=True, active=17)
    _stub_corpus(monkeypatch, connection)
    result = module.sync_embeddings(EmbeddingDatabase(connection), config_path="config.yml", taxonomy_path="taxonomy.json", activate=False)
    assert result.active and result.already_ready
    assert not any("INSERT INTO active_category_profile_generation_v3" in sql for sql, _ in connection.statements)


def test_new_generation_staging_keeps_existing_active_pointer(monkeypatch):
    connection = EmbeddingConnection(ready=False)
    _stub_corpus(monkeypatch, connection)
    result = module.sync_embeddings(EmbeddingDatabase(connection), config_path="config.yml", taxonomy_path="taxonomy.json", activate=False)
    assert result.state == "ready" and not result.active and not result.already_ready
    assert result.embedded_count == 1
    assert connection.active == 7
    assert not any("INSERT INTO active_category_profile_generation_v3" in sql for sql, _ in connection.statements)


@pytest.mark.parametrize("ready", [False, True])
def test_default_sync_stages_new_and_reused_without_switching_pointers(monkeypatch, ready):
    connection = EmbeddingConnection(ready=ready)
    _stub_corpus(monkeypatch, connection)
    result = module.sync_embeddings(EmbeddingDatabase(connection), config_path="config.yml", taxonomy_path="taxonomy.json")
    assert result.already_ready is ready
    assert not result.active
    assert connection.active == 7
    assert not any("INSERT INTO active_category_profile_generation_v3" in sql for sql, _ in connection.statements)


def test_explicit_unpaired_activate_refused_before_connection(monkeypatch):
    connection = EmbeddingConnection(ready=True)
    _stub_corpus(monkeypatch, connection)
    database = EmbeddingDatabase(connection)
    with pytest.raises(RuntimeError, match="unilateral E5 activation"):
        module.sync_embeddings(database, config_path="config.yml", taxonomy_path="taxonomy.json", activate=True)
    with pytest.raises(RuntimeError, match="unilateral E5 activation"):
        module.activate_generation(database, 17)
    assert connection.statements == []


def test_cleanup_refuses_unified_db_before_connection():
    connection = EmbeddingConnection()
    with pytest.raises(RuntimeError, match="cleanup is disabled"):
        module.cleanup_generations(EmbeddingDatabase(connection))
    assert connection.statements == []


def test_conflicting_ready_generation_is_never_rebuilt_or_marked_failed(monkeypatch):
    connection = EmbeddingConnection(ready=True)
    connection.fail_ready_content = True
    _stub_corpus(monkeypatch, connection)
    with pytest.raises(RuntimeError, match="content hashes/text"):
        module.sync_embeddings(EmbeddingDatabase(connection), config_path="config.yml", taxonomy_path="taxonomy.json", activate=False)
    assert connection.active == 7
    assert not any("UPDATE embedding_generations SET state='failed'" in sql for sql, _ in connection.statements)
