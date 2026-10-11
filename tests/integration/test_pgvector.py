"""Opt-in read-only smoke test against the existing unified serving database."""

import os
from pathlib import Path

import pytest

from model_intent.search_runtime import E5_POI_MODEL_KEY, SearchRuntime

ROOT = Path(__file__).parents[2]
pytestmark = [pytest.mark.pgvector, pytest.mark.real_model]


def test_serving_intent_uses_category_and_property_pgvector_without_pois(monkeypatch):
    if os.getenv("MODEL_INTENT_READONLY_DB_TEST") != "1":
        pytest.skip("set MODEL_INTENT_READONLY_DB_TEST=1 and DATABASE_URL to opt into read-only serving test")
    monkeypatch.setenv("PGOPTIONS", "-c default_transaction_read_only=on")
    runtime = SearchRuntime.from_intent_config(
        config_path=ROOT / "configs/benchmark.poi.pg.v3.yaml",
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v3.json",
        model_key=E5_POI_MODEL_KEY,
        local_files_only=True,
    )
    try:
        category = runtime.analyze_intent("quán cà phê có máy lạnh")
        assert category["intent"]["runtime"]["backend"] == "pgvector"
        assert category["intent"]["runtime"]["generation_id"] is not None
        assert "poi_search" not in category
        assert any(row["property_id"] == "air_conditioning" for row in category["property_extraction"]["properties"])

        unresolved = runtime.analyze_intent("nhà nghỉ có máy giặt rệt")
        parts = unresolved["query_parts"]
        assert "poi_search" not in unresolved
        assert parts["unresolved_property_clauses"]
        assert parts["semantic_property_metadata"]["status"] == "suggestions_only"
        assert parts["semantic_property_metadata"]["backend"] == "pg_exact_v1"
        assert parts["semantic_property_metadata"]["generation_id"] == 4
        assert not parts["filter_candidates"] or all(
            row["cue"] != "semantic_user_confirmed" for row in parts["filter_candidates"]
        )
    finally:
        runtime.close()
