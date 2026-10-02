import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

import model_intent.db as db_module
from model_intent.db import Database
from model_intent.embedding_sync import embedding_status, sync_embeddings
from model_intent.migrations import apply_migrations
from model_intent.retrieval import SemanticIndex
from model_intent.runtime import MatcherRuntime
from model_intent.schemas import CategorySearchProfile

ROOT = Path(__file__).parents[2]
pytestmark = [pytest.mark.pgvector, pytest.mark.pgvector_write]


@pytest.fixture(autouse=True)
def disposable_database_only(monkeypatch):
    url = os.getenv("MODEL_INTENT_WRITE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("write tests require MODEL_INTENT_WRITE_TEST_DATABASE_URL pointing to a disposable DB")
    parsed = urlparse(url)
    try:
        valid = (
            parsed.scheme in ("postgres", "postgresql")
            and parsed.hostname == "127.0.0.1"
            and parsed.port not in (5432, 5433, None)
            and parsed.path == "/model_intent_write_test"
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        pytest.fail("write tests require a separate 127.0.0.1 port and model_intent_write_test database")

    def scratch_url(database_url=None):
        if database_url is not None and database_url != url:
            raise RuntimeError("write tests cannot connect outside the disposable database")
        return url

    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(db_module, "require_database_url", scratch_url)


def database_or_skip():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL is not set")
    try:
        database = Database(statement_timeout_ms=120000)
    except RuntimeError as exc:
        if "optional 'db' dependencies" in str(exc):
            pytest.skip("optional database dependencies are not installed")
        raise
    database.open()
    return database


def test_profile_migrations_and_sync_are_idempotent():
    database = database_or_skip()
    try:
        applied = apply_migrations(database, ROOT / "migrations")
        assert not applied or any(
            name in applied
            for name in ("005_category_search_profiles_v3", "006_category_search_profiles_bge_m3", "007_poi_dataset_generations")
        )
        result = sync_embeddings(
            database,
            config_path=ROOT / "configs/benchmark.cpu.yaml",
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
            local_files_only=True,
        )
        assert result.state == "ready"
        assert result.profile_count == 444
        assert result.category_count == 222
        second = sync_embeddings(
            database,
            config_path=ROOT / "configs/benchmark.cpu.yaml",
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
            local_files_only=True,
        )
        assert second.already_ready is True
        assert second.embedded_count == 0
        status = embedding_status(database)
        assert status["active"]["state"] == "ready"
        assert status["active"]["profile_count"] == 444
        assert status["active"]["category_count"] == 222
    finally:
        database.close()


def test_pg_exact_matches_numpy_for_top_twenty_profiles():
    database = database_or_skip()
    runtime = MatcherRuntime.from_config(
        config_path=ROOT / "configs/benchmark.cpu.yaml",
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
        model_key="e5-base-multilingual",
        local_files_only=True,
    )
    try:
        with database.connection() as (connection, _):
            rows = connection.execute(
                """
                SELECT csp.place_type_id, csp.language, csp.search_text, csp.embedding
                FROM active_category_profile_generation_v3 a
                JOIN category_search_profiles_v3 csp ON csp.generation_id=a.generation_id
                WHERE a.model_key=%s ORDER BY csp.id
                """,
                ("e5-base-multilingual",),
            ).fetchall()
        profiles = [
            CategorySearchProfile(str(row[0]), row[1], row[2])
            for row in rows
        ]
        assert all("\nMục đích sử dụng:" not in row[2] for row in rows)
        assert all("\nTiện ích thường gặp:" not in row[2] for row in rows)
        assert all("\nÝ định phổ biến:" not in row[2] for row in rows)
        vectors = np.asarray([row[3].to_numpy() for row in rows], dtype=np.float32)
        numpy_index = SemanticIndex(runtime.taxonomy, profiles, vectors, runtime.model_key)
        for query in ("đau đầu cần mua thuốc", "chỗ ngồi làm việc có wifi"):
            pg_result = runtime.index.search(query, runtime.embedder, top_k=20)
            np_result = numpy_index.search(query, runtime.embedder, top_k=20)
            assert [hit.item_id for hit in pg_result.hits] == [hit.item_id for hit in np_result.hits]
            assert max(abs(a.score - b.score) for a, b in zip(pg_result.hits, np_result.hits)) < 1e-5
    finally:
        database.close()
        if runtime.database:
            runtime.database.close()
