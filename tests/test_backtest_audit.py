from pathlib import Path

from scripts.audit_backtest_5698 import audit, category_metrics, read_rows, sha256

ROOT = Path(__file__).parents[1]
PATH = ROOT / "artifacts/backtest_port_8502/backtest_predictions_5698.jsonl"


def test_saved_backtest_numbers_and_proxy_labels_are_not_confused():
    import json

    if not PATH.exists():
        return
    rows = read_rows(PATH)
    policy = json.loads((ROOT / "data/eval/backtest_gold_semantics.poc.v1.json").read_text(encoding="utf-8"))
    report = audit(rows, policy, predictions_sha256=sha256(PATH))
    assert report["rows"] == 5698
    assert report["in_domain_reported_gold_proxy"]["raw_top1_correct"] == 2763
    assert report["in_domain_reported_gold_proxy"]["raw_top3_correct"] == 3681
    assert report["in_domain_reported_gold_proxy"]["accepted_correct"] == 1093
    assert report["ood"]["false_accepts"] == 12
    assert report["known_invalid_or_unsupported"]["count"] == 616
    assert report["property_predictions"]["gold_available"] is False
    assert report["top_confidence_0_55_counterfactual_upper_bound"]["rejected_insufficient"] == 134
