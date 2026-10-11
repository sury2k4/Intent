from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from model_intent.search_runtime import E5_POI_MODEL_ID, E5_POI_MODEL_KEY, E5_POI_REVISION, SearchRuntime

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("database_url,error", [
    (None, "explicit DATABASE_URL"),
    ("postgresql://model_intent@localhost:5432/model_intent", "isolated unified database"),
])
def test_intent_serving_validates_database_before_loading_model(monkeypatch, database_url, error):
    import model_intent.search_runtime as module

    if database_url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(module.MatcherRuntime, "from_config", lambda **kwargs: pytest.fail("model must not load"))
    with pytest.raises(RuntimeError, match=error):
        SearchRuntime.from_intent_config(
            config_path=ROOT / "configs/benchmark.poi.pg.v3.yaml",
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v3.json",
            model_key=E5_POI_MODEL_KEY,
        )


@pytest.mark.parametrize("backend,mode,dimension,generation,valid", [
    ("pgvector", "pg_exact", 768, 77, True),
    ("pgvector", "pg_exact", 765, 77, False),
    ("pgvector", "pg_exact", 768, None, False),
    ("pgvector", "pg_ann", 768, 77, False),
    ("numpy", "numpy_exact", 768, None, False),
])
def test_intent_runtime_requires_pinned_e5_category_generation(backend, mode, dimension, generation, valid):
    category = SimpleNamespace(
        metadata=SimpleNamespace(
            model_id=E5_POI_MODEL_ID, model_revision=E5_POI_REVISION,
            backend=backend, retrieval_mode=mode, generation_id=generation,
        ),
        embedder=SimpleNamespace(metadata=lambda: {"dimension": 768}),
        index=SimpleNamespace(dimension=dimension, vectors=SimpleNamespace(shape=(452, dimension))),
        database=object() if backend == "pgvector" else None,
        confidence_calibration={"calibration_version": "e5-base-confidence-v3"},
    )
    if valid:
        SearchRuntime._validate_e5_category_runtime(category, E5_POI_MODEL_KEY)
    else:
        with pytest.raises(RuntimeError, match="pinned E5-base 768D"):
            SearchRuntime._validate_e5_category_runtime(category, E5_POI_MODEL_KEY)
