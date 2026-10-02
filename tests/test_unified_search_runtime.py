from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from model_intent.search_runtime import (
    E5_POI_MODEL_ID,
    E5_POI_MODEL_KEY,
    E5_POI_REVISION,
    SearchRuntime,
)
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def test_unified_serving_requires_explicit_database_url(monkeypatch):
    import model_intent.search_runtime as module

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        module.MatcherRuntime, "from_config",
        lambda **kwargs: pytest.fail("must reject before contacting any database"),
    )
    with pytest.raises(RuntimeError, match="explicit DATABASE_URL"):
        SearchRuntime.from_config(
            config_path=ROOT / "configs/benchmark.poi.pg.yaml",
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v2.json",
            model_key=E5_POI_MODEL_KEY,
        )


def test_unified_serving_rejects_url_to_legacy_database(monkeypatch):
    import model_intent.search_runtime as module

    monkeypatch.setenv("DATABASE_URL", "postgresql://model_intent@localhost:5432/model_intent")
    monkeypatch.setattr(
        module.MatcherRuntime, "from_config",
        lambda **kwargs: pytest.fail("must reject legacy database before opening pool"),
    )
    with pytest.raises(RuntimeError, match="isolated unified database"):
        SearchRuntime.from_config(
            config_path=ROOT / "configs/benchmark.poi.pg.yaml",
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v2.json",
            model_key=E5_POI_MODEL_KEY,
        )


@pytest.mark.parametrize("backend,mode,dimension,generation,db,valid", [
    ("pgvector", "pg_exact", 768, 77, object(), True),
    ("pgvector", "pg_exact", 765, 77, object(), False),
    ("pgvector", "pg_exact", 768, None, object(), False),
    ("pgvector", "pg_ann", 768, 77, object(), False),
])
def test_search_runtime_accepts_only_pinned_exact_768d_backend(
    monkeypatch, backend, mode, dimension, generation, db, valid,
):
    import model_intent.search_runtime as module

    monkeypatch.setenv("DATABASE_URL", "postgresql://model_intent@127.0.0.1:5433/model_intent_unified")

    class StubRuntime:
        metadata = SimpleNamespace(
            model_id=E5_POI_MODEL_ID, model_revision=E5_POI_REVISION,
            backend=backend, retrieval_mode=mode, generation_id=generation,
        )
        embedder = SimpleNamespace(metadata=lambda: {"dimension": 768})
        index = SimpleNamespace(dimension=dimension, vectors=SimpleNamespace(shape=(452, dimension)))
        confidence_calibration = {"calibration_version": "e5-base-confidence-v2"}
        database = db
        taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v2.json")

    monkeypatch.setattr(module.MatcherRuntime, "from_config", lambda **kwargs: StubRuntime())
    if backend == "pgvector" and valid:
        class StubPoiRepository:
            def __init__(self):
                self.generation_id = 77

        monkeypatch.setattr(module.PoiRepository, "from_database", lambda *args, **kwargs: StubPoiRepository())
        search = SearchRuntime.from_config(
            config_path=ROOT / "configs/benchmark.poi.pg.yaml",
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v2.json",
            model_key=E5_POI_MODEL_KEY,
        )
        assert search.poi_repository.generation_id == 77
    elif not valid:
        with pytest.raises(RuntimeError, match="pinned E5-base 768D"):
            SearchRuntime.from_config(
                config_path=ROOT / "configs/benchmark.poi.pg.yaml",
                taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v2.json",
                model_key=E5_POI_MODEL_KEY,
            )
