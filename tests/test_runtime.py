from pathlib import Path

import pytest
import yaml

from model_intent.runtime import MatcherRuntime

ROOT = Path(__file__).parents[1]


def write_fake_config(tmp_path):
    config = {
        "seed": 17,
        "threads": 2,
        "cache_dir": str(tmp_path / "cache"),
        "languages": ["vi", "en"],
        "enrichment_path": str(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json"),
        "policy_path": str(ROOT / "data/policies/place_type_decision_policy.poc.v1.json"),
        "models": [],
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_runtime_builds_then_reuses_candidate_cache(tmp_path):
    config = write_fake_config(tmp_path)
    first = MatcherRuntime.from_config(
        config_path=config,
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
        model_key="fake",
        local_files_only=True,
    )
    second = MatcherRuntime.from_config(
        config_path=config,
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
        model_key="fake",
        local_files_only=True,
    )
    assert first.metadata.cache_hit is False
    assert second.metadata.cache_hit is True
    assert first.metadata.candidate_count == second.metadata.candidate_count == 444
    assert first.metadata.profile_count == second.metadata.profile_count == 444
    assert first.metadata.indexed_category_count == second.metadata.indexed_category_count == 222


def test_runtime_query_slices_display_but_keeps_decision_depth(tmp_path):
    runtime = MatcherRuntime.from_config(
        config_path=write_fake_config(tmp_path),
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
        model_key="fake",
        local_files_only=True,
    )
    result = runtime.query("mai trời có mưa không", top_k=3, language="vi")
    assert len(result["prediction"]["hits"]) == 3
    assert result["decision"]["action"] == "reject"
    assert result["runtime"]["candidate_count"] > 100


def test_runtime_selection_is_restricted_to_offered_leaf_ids(tmp_path):
    runtime = MatcherRuntime.from_config(
        config_path=write_fake_config(tmp_path),
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v1.json",
        model_key="fake",
        local_files_only=True,
    )
    selected = runtime.resolve_selection(
        query="đau đầu",
        selected_id="34",
        allowed_option_ids=["34", "144"],
        language="vi",
    )
    assert selected["decision"]["action"] == "accept"
    assert selected["decision"]["accepted_code"] == "pharmacy"
    with pytest.raises(ValueError, match="not offered"):
        runtime.resolve_selection(query="q", selected_id="85", allowed_option_ids=["34"])
    with pytest.raises(ValueError, match="leaf"):
        runtime.resolve_selection(query="q", selected_id="41", allowed_option_ids=["41"])
