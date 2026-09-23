from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .retrieval import prediction_to_dict


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "query", "rank", "item_id", "code", "cosine_similarity", "language", "matched_text", "margin"])
        for row in predictions:
            for hit in row.get("hits", []):
                writer.writerow([row["query_id"], row["query"], hit["rank"], hit["item_id"], hit["code"], hit["score"], hit["language"], hit["matched_text"], row.get("margin")])


def write_decisions(path: str | Path, decisions: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id", "query", "action", "reason", "accepted_item_id", "accepted_code",
        "question", "option_ids", "cosine_similarity", "margin", "calibrated_confidence",
        "in_domain_confidence", "ambiguity_score", "calibration_version", "policy_version",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in decisions:
            writer.writerow({
                "query_id": row.get("query_id"), "query": row.get("query"),
                "action": row.get("action"), "reason": row.get("reason"),
                "accepted_item_id": row.get("accepted_item_id"), "accepted_code": row.get("accepted_code"),
                "question": row.get("question"),
                "option_ids": ",".join(str(option.get("item_id")) for option in row.get("options", [])),
                "cosine_similarity": row.get("top1_score"), "margin": row.get("margin"),
                "calibrated_confidence": row.get("calibrated_confidence"),
                "in_domain_confidence": row.get("in_domain_confidence"),
                "ambiguity_score": row.get("ambiguity_score"),
                "calibration_version": row.get("calibration_version"),
                "policy_version": row.get("policy_version"),
            })


def write_model_report(directory: str | Path, result: dict[str, Any]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", result.get("metrics", {}))
    write_json(directory / "resource.json", result.get("resource", {}))
    write_predictions(directory / "predictions.csv", result.get("predictions", []))
    write_decisions(directory / "decisions.csv", result.get("decisions", []))
    lines = [f"# Model report: {result.get('model_key')}", "", f"Model: `{result.get('model_id')}`", ""]
    if result.get("error"):
        lines += [f"**Error:** {result['error']}", ""]
    overall = result.get("metrics", {}).get("overall", {})
    lines += ["## Overall", "", "| Metric | Value |", "|---|---:|"]
    for key in ("strict_hit@1", "strict_hit@3", "acceptable_hit@1", "mrr"):
        value = overall.get(key)
        if isinstance(value, dict):
            value = value.get("mean")
        lines.append(f"| {key} | {value if value is not None else 'n/a'} |")
    decision = overall.get("decision", {})
    lines += ["", "## Decision quality", "", "| Metric | Value |", "|---|---:|"]
    for key in ("action_accuracy", "accept_coverage", "accepted_accuracy", "clarification_f1", "option_recall", "over_clarification_rate"):
        lines.append(f"| {key} | {decision.get(key, 'n/a')} |")
    ood = overall.get("ood", {})
    lines.append(f"| ood_reject_rate | {ood.get('ood_reject_rate', 'n/a')} |")
    lines.append(f"| ood_false_accept_rate | {ood.get('ood_false_accept_rate', 'n/a')} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(directory: str | Path, results: list[dict[str, Any]]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        overall = result.get("metrics", {}).get("overall", {})
        def value(key):
            raw = overall.get(key)
            return raw.get("mean") if isinstance(raw, dict) else raw
        decision = overall.get("decision", {})
        ood = overall.get("ood", {})
        rows.append({
            "model_key": result.get("model_key"), "model_id": result.get("model_id"), "error": result.get("error"),
            "strict_hit@1": value("strict_hit@1"), "strict_hit@3": value("strict_hit@3"),
            "acceptable_hit@1": value("acceptable_hit@1"), "mrr": value("mrr"),
            "decision_accuracy": decision.get("action_accuracy"),
            "accepted_accuracy": decision.get("accepted_accuracy"),
            "clarification_f1": decision.get("clarification_f1"),
            "ood_reject_rate": ood.get("ood_reject_rate"),
            "query_p50_ms": result.get("resource", {}).get("query_p50_ms"),
            "query_p95_ms": result.get("resource", {}).get("query_p95_ms"),
            "peak_rss_mb": result.get("resource", {}).get("peak_rss_mb"),
        })
    write_json(directory / "comparison.json", rows)
    with (directory / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model_key"])
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Model comparison", "", "| Model | Hit@1 | MRR | Decision acc. | Accepted acc. | Clarify F1 | OOD reject | p50 ms | RSS MB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {model_key} | {strict_hit@1} | {mrr} | {decision_accuracy} | {accepted_accuracy} | {clarification_f1} | {ood_reject_rate} | {query_p50_ms} | {peak_rss_mb} |".format(**row))
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
