from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "artifacts/backtest_port_8502/backtest_predictions_5698.jsonl"
DEFAULT_GOLD_POLICY = ROOT / "data/eval/backtest_gold_semantics.poc.v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("backtest contains duplicate query IDs")
    return rows


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [row for row in rows if row["query_type"] == "in_domain"]
    def matched(row: dict[str, Any], limit: int) -> bool:
        targets = {str(item) for item in row.get("target_category_ids", [])}
        return any(str(candidate.get("id")) in targets for candidate in (row.get("top3_candidates") or [])[:limit])

    raw_top1 = sum(matched(row, 1) for row in relevant)
    raw_top3 = sum(matched(row, 3) for row in relevant)
    accepts = [row for row in relevant if row["action"] == "accept"]
    correct_accepts = sum(str(row.get("accepted_category", {}).get("id")) in
                          {str(item) for item in row.get("target_category_ids", [])} for row in accepts)
    return {
        "count": len(relevant),
        "raw_top1_correct": raw_top1,
        "raw_top3_correct": raw_top3,
        "accepted": len(accepts),
        "accepted_correct": correct_accepts,
        "raw_top1_accuracy": raw_top1 / len(relevant) if relevant else None,
        "raw_top3_recall": raw_top3 / len(relevant) if relevant else None,
        "correct_accept_rate": correct_accepts / len(relevant) if relevant else None,
        "accept_precision": correct_accepts / len(accepts) if accepts else None,
    }


def audit(rows: list[dict[str, Any]], gold_policy: dict[str, Any], *, predictions_sha256: str) -> dict[str, Any]:
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[(row["split"], row["query_type"])].append(row)
    flagged = set(gold_policy["unsupported_or_proxy_labels"])
    valid = [row for row in rows if row["query_type"] == "in_domain" and row["gold_label"] not in flagged]
    unsupported = [row for row in rows if row["query_type"] == "in_domain" and row["gold_label"] in flagged]
    properties = [item for row in rows for item in row.get("extracted_properties", [])]
    latencies = sorted(float(row["elapsed_ms"]) for row in rows)
    def percentile(fraction: float) -> float:
        return latencies[min(len(latencies) - 1, int((len(latencies) - 1) * fraction))]

    thresholds = {"top_correct_accept": 0.60, "in_domain_gate": 0.705}
    band = [row for row in rows
            if row.get("scores", {}).get("calibrated_confidence") is not None
            and row.get("scores", {}).get("in_domain_confidence") is not None
            and 0.55 <= float(row["scores"]["calibrated_confidence"]) < thresholds["top_correct_accept"]
            and float(row["scores"]["in_domain_confidence"]) >= thresholds["in_domain_gate"]]
    newly_eligible = [row for row in band if row["action"] == "reject"
                      and row["reason"] == "insufficient_calibrated_evidence"]
    report = {
        "schema_version": 1,
        "source_sha256": predictions_sha256,
        "gold_policy_version": gold_policy["gold_policy_version"],
        "rows": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "cohorts": {f"{split}/{query_type}": len(group) for (split, query_type), group in sorted(cohorts.items())},
        "in_domain_reported_gold_proxy": category_metrics(rows),
        "in_domain_excluding_known_invalid_proxies": category_metrics(valid),
        "known_invalid_or_unsupported": {
            "count": len(unsupported),
            "labels": dict(sorted(Counter(row["gold_label"] for row in unsupported).items())),
            "reasons": gold_policy["unsupported_or_proxy_labels"],
        },
        "ood": {
            "count": sum(row["query_type"] == "ood" for row in rows),
            "rejected": sum(row["query_type"] == "ood" and row["action"] == "reject" for row in rows),
            "false_accepts": sum(row["query_type"] == "ood" and row["action"] == "accept" for row in rows),
        },
        "property_predictions": {
            "count": len(properties),
            "queries_with_any": sum(bool(row.get("extracted_properties")) for row in rows),
            "mode_counts": dict(Counter(str(item.get("mode")) for item in properties)),
            "applied": sum(item.get("applied") is True for item in properties),
            "ignored": sum(item.get("applied") is False for item in properties),
            "gold_available": any("expected_properties" in row or "gold_properties" in row for row in rows),
        },
        "latency_ms": {
            "mean": mean(latencies), "p50": percentile(.50),
            "p90": percentile(.90), "p99": percentile(.99),
            "max": max(latencies), "above_150": sum(value > 150 for value in latencies),
        },
        "top_confidence_0_55_counterfactual_upper_bound": {
            "status": "exploratory_only_rounded_scores_top3_no_exact_replay",
            "band_rows_passing_domain": len(band),
            "existing_clarifications": sum(row["action"] == "clarify" for row in band),
            "rejected_insufficient": len(newly_eligible),
            "eligible_by_type": dict(sorted(Counter(row["query_type"] for row in newly_eligible).items())),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit recorded 5,698-query backtest without changing policy")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gold-policy", type=Path, default=DEFAULT_GOLD_POLICY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.predictions)
    policy = json.loads(args.gold_policy.read_text(encoding="utf-8"))
    report = audit(rows, policy, predictions_sha256=sha256(args.predictions))
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite existing backtest audit: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
