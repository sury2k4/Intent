from pathlib import Path

from model_intent.benchmark import load_yaml, run_model
from model_intent.evaluation import load_cases
from model_intent.schemas import ModelProfile
from model_intent.taxonomy import Taxonomy


def test_fake_benchmark_produces_metrics(tmp_path):
    root = Path(__file__).parents[1]
    taxonomy = Taxonomy.load(root / "data/taxonomies/place_types.poc.v1.json")
    cases = load_cases(root / "data/eval/place_type_queries.poc.v1.jsonl")[:4]
    config = {
        "languages": ["vi", "en"], "cache_dir": str(tmp_path), "top_k": 5,
        "warmup_queries": 1, "latency_repeats": 1,
        "policy_path": str(root / "data/policies/place_type_decision_policy.poc.v1.json"),
    }
    result = run_model(ModelProfile("fake", "fake"), taxonomy, cases, config)
    assert result["error"] is None
    assert result["metrics"]["overall"]["n"] == 4
    assert len(result["predictions"]) == 4
    assert len(result["decisions"]) == 4
