from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_intent.search_runtime import SearchRuntime

ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.pgvector


def test_unified_e5_and_poi_generations_match_artifact_results_without_resync():
    url = os.environ.get("DATABASE_URL", "")
    if not url.endswith(":5433/model_intent_unified"):
        pytest.skip("requires isolated unified Postgres port 5433; will not touch legacy database")
    if not (ROOT / "artifacts/osm/hcmc-osm-v3/pois.jsonl").exists():
        pytest.skip("requires the ignored v3 HCMC artifact for read-only parity")
    artifact = SearchRuntime.from_config(
        config_path=ROOT / "configs/benchmark.poi.v3.yaml",
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v3.json",
        model_key="e5-base-multilingual", local_files_only=True,
    )
    postgres = SearchRuntime.from_config(
        config_path=ROOT / "configs/benchmark.poi.pg.v3.yaml",
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v3.json",
        model_key="e5-base-multilingual", local_files_only=True,
    )
    try:
        assert postgres.category_runtime.metadata.backend == "pgvector"
        assert postgres.category_runtime.metadata.profile_count == 768
        assert postgres.poi_repository.source == "postgres"
        assert postgres.poi_repository.count() == artifact.poi_repository.count() == 40639
        for query in (
            "trạm sạc nhanh có nhà vệ sinh", "ngân hàng", "nhà thuốc có điều hòa",
            "quán cà phê ưu tiên thoáng mát", "quán cà phê hoặc nhà thuốc", "mai trời có mưa không",
        ):
            baseline = artifact.search(query, poi_top_k=10)
            unified = postgres.search(query, poi_top_k=10)
            assert unified["intent"]["decision"]["action"] == baseline["intent"]["decision"]["action"]
            assert unified["intent"]["decision"].get("accepted_item_id") == baseline["intent"]["decision"].get("accepted_item_id")
            assert unified["poi_search"]["hard_filter_match_count"] == baseline["poi_search"]["hard_filter_match_count"]
            assert [item["poi_id"] for item in unified["poi_search"]["hits"]] == [item["poi_id"] for item in baseline["poi_search"]["hits"]]
    finally:
        artifact.close()
        postgres.close()
