from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from model_intent.category_resolution import ExplicitCategory, explicit_categories, unique_leaf_name
from model_intent.pgvector_index import PgVectorIndex
from model_intent.rendering import load_enrichment
from model_intent.search_timing import SearchTiming
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


class Connection:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def execute(self, sql, params):
        self.calls += 1
        return self

    def fetchall(self):
        return self.rows


class DatabaseStub:
    def __init__(self, rows):
        self.conn = Connection(rows)

    @contextmanager
    def connection(self):
        yield self.conn, SimpleNamespace(wait_ms=1.25)


class EmbedderStub:
    def encode_queries(self, queries):
        return np.ones((len(queries), 2), dtype=np.float32)


def test_timing_marks_failed_spans_without_leaking_to_next_request():
    timing = SearchTiming("intent_analysis")
    with pytest.raises(RuntimeError), timing.measure("category_db_execute_fetch"):
        raise RuntimeError("simulated database failure")
    first = timing.as_dict()["stages"][0]
    assert first["status"] == "error" and first["duration_ms"] >= 0
    other = SearchTiming("intent_analysis")
    assert other.as_dict()["stages"] == []


def test_unique_leaf_name_only_matches_a_full_unambiguous_name():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    matches = explicit_categories("quán cà phê", taxonomy, enrichment)
    assert unique_leaf_name("quán cà phê", taxonomy, matches).item_id == "23"
    assert unique_leaf_name("tìm quán cà phê", taxonomy, matches).item_id == "23"
    assert unique_leaf_name("quán cà phê ở bên cạnh nhà thuốc", taxonomy, matches) is None
    assert unique_leaf_name("không phải quán cà phê", taxonomy, matches) is None
    assert unique_leaf_name("quán cà phê", taxonomy, matches + (ExplicitCategory("34", "nhà thuốc"),)) is None


def test_pgvector_trace_records_all_variants_without_changing_scores():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    db = DatabaseStub([(42, 23, "vi", "Category: Quán cà phê", 0.82)])
    index = PgVectorIndex(taxonomy, db, generation_id=42, dimension=2)
    timing = SearchTiming("intent_analysis")
    with timing.measure("category_variant", variant_index=0):
        prediction = index.search("quán cà phê", EmbedderStub(), top_k=2, timing=timing)
    assert prediction.hits[0].score == 0.82
    assert db.conn.calls == 2
    stages = timing.as_dict()["stages"]
    assert [row["name"] for row in stages].count("category_db_variant") == 2
    assert [row["duration_ms"] for row in stages if row["name"] == "category_pool_wait"] == [1.25, 1.25]
    assert [row["name"] for row in stages].count("category_db_execute_fetch") == 2
    assert [row["name"] for row in stages].count("category_hydrate_merge") == 2
    assert next(row for row in stages if row["name"] == "category_sort_hits")["duration_ms"] >= 0
    assert index.last_search_metadata["query_variant_count"] == 2
