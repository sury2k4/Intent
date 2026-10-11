from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import EvaluationCase, IntentDecision, Prediction


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        case_id = str(row["id"])
        if case_id in seen:
            raise ValueError(f"duplicate evaluation case id: {case_id}")
        seen.add(case_id)
        cases.append(EvaluationCase(
            id=case_id, track=str(row["track"]), split=str(row.get("split", "test")),
            query=str(row["query"]), language=row.get("language"),
            gold_ids=tuple(str(x) for x in row.get("gold_ids", [])),
            acceptable_ids=tuple(str(x) for x in row.get("acceptable_ids", [])),
            difficulty=str(row.get("difficulty", "normal")), is_ood=bool(row.get("is_ood", False)),
            notes=str(row.get("notes", "")),
            expected_action=row.get("expected_action"),
            clarification_ids=tuple(str(x) for x in row.get("clarification_ids", [])),
            near_group=row.get("near_group"),
        ))
    return cases


def validate_cases(cases: list[EvaluationCase], taxonomy_ids: set[str]) -> list[str]:
    warnings: list[str] = []
    for case in cases:
        referenced = set(case.gold_ids + case.acceptable_ids + case.clarification_ids)
        unknown = sorted(referenced - taxonomy_ids)
        if unknown:
            raise ValueError(f"evaluation case {case.id} has unknown IDs: {unknown}")
        expected = _expected_action(case)
        if expected not in {"accept", "clarify", "reject"}:
            raise ValueError(f"evaluation case {case.id} has invalid expected_action: {expected}")
        if expected == "clarify" and len(set(case.clarification_ids or case.gold_ids)) < 2:
            raise ValueError(f"clarification case {case.id} needs at least two option IDs")
        if case.is_ood and expected != "reject":
            warnings.append(f"OOD case {case.id} expects {expected}")
    return warnings


def _rank(pred: Prediction | None, ids: Iterable[str]) -> int | None:
    if pred is None:
        return None
    wanted = set(ids)
    for hit in pred.hits:
        if hit.item_id in wanted:
            return hit.rank
    return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _bootstrap(values: list[float], seed: int = 17, rounds: int = 500) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "low": None, "high": None}
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    return {"mean": statistics.fmean(values), "low": means[int(0.025 * rounds)], "high": means[int(0.975 * rounds) - 1]}


def _expected_action(case: EvaluationCase) -> str:
    if case.expected_action:
        return case.expected_action
    return "reject" if case.is_ood else "accept"


def evaluate(
    cases: list[EvaluationCase],
    predictions: dict[str, Prediction],
    top_ks: tuple[int, ...] = (1, 3, 5),
    seed: int = 17,
    decisions: dict[str, IntentDecision] | None = None,
) -> dict[str, Any]:
    decisions = decisions or {}
    if not cases:
        return {"n": 0, "tracks": {}, "overall": _evaluate_subset([], predictions, decisions, top_ks, seed)}
    metrics: dict[str, Any] = {"n": len(cases), "tracks": {}}
    for split in sorted({c.split for c in cases}):
        subset = [c for c in cases if c.split == split]
        metrics["tracks"][split] = _evaluate_subset(subset, predictions, decisions, top_ks, seed)
    headline = [c for c in cases if c.split == "test"] or cases
    metrics["overall"] = _evaluate_subset(headline, predictions, decisions, top_ks, seed)
    return metrics


def _evaluate_subset(
    cases: list[EvaluationCase],
    predictions: dict[str, Prediction],
    decisions: dict[str, IntentDecision],
    top_ks: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        pred = predictions.get(case.id)
        decision = decisions.get(case.id)
        strict_rank = _rank(pred, case.gold_ids)
        acceptable = tuple(dict.fromkeys(case.gold_ids + case.acceptable_ids))
        acceptable_rank = _rank(pred, acceptable)
        rr = 1.0 / strict_rank if strict_rank else 0.0
        rows.append({
            "case": case, "prediction": pred, "decision": decision,
            "strict_rank": strict_rank, "acceptable_rank": acceptable_rank, "rr": rr,
            "expected_action": _expected_action(case),
        })
    result: dict[str, Any] = {
        "n": len(cases),
        "predicted_n": sum(r["prediction"] is not None for r in rows),
        "missing_predictions": sum(r["prediction"] is None for r in rows),
        "ood_n": sum(r["case"].is_ood for r in rows),
    }
    for k in top_ks:
        strict = [1.0 if r["strict_rank"] and r["strict_rank"] <= k else 0.0 for r in rows if not r["case"].is_ood]
        acceptable = [1.0 if r["acceptable_rank"] and r["acceptable_rank"] <= k else 0.0 for r in rows if not r["case"].is_ood]
        result[f"strict_hit@{k}"] = _bootstrap(strict, seed)
        result[f"acceptable_hit@{k}"] = _bootstrap(acceptable, seed)
    in_domain = [r for r in rows if not r["case"].is_ood]
    result["mrr"] = _bootstrap([r["rr"] for r in in_domain], seed)
    result["ood"] = _ood_metrics(rows)
    result["decision"] = _decision_metrics(rows)
    result["by_track"] = _group_metrics(rows)
    result["by_language"] = _group_metrics(rows, lambda r: r["case"].language or "unknown")
    result["by_difficulty"] = _group_metrics(rows, lambda r: r["case"].difficulty)
    return result


def _group_metrics(rows: list[dict[str, Any]], key_fn=lambda r: r["case"].track) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    output = {}
    for key, group in groups.items():
        in_domain = [r for r in group if not r["case"].is_ood]
        decision_rows = [r for r in group if r["decision"] is not None]
        output[key] = {
            "n": len(group),
            "predicted_n": sum(r["prediction"] is not None for r in group),
            "strict_hit@1": _mean([1.0 if r["strict_rank"] == 1 else 0.0 for r in in_domain]),
            "acceptable_hit@1": _mean([1.0 if r["acceptable_rank"] == 1 else 0.0 for r in in_domain]),
            "mrr": _mean([r["rr"] for r in in_domain]),
            "decision_accuracy": _mean([1.0 if r["decision"].action == r["expected_action"] else 0.0 for r in decision_rows]),
        }
    return output


def _probability_metrics(probabilities: list[float], labels: list[float], bins: int = 10) -> dict[str, Any]:
    if not probabilities:
        return {"n": 0, "brier": None, "log_loss": None, "ece": None, "reliability": []}
    clipped = [min(1.0 - 1e-9, max(1e-9, float(value))) for value in probabilities]
    brier = statistics.fmean((probability - label) ** 2 for probability, label in zip(clipped, labels))
    log_loss = statistics.fmean(
        -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
        for probability, label in zip(clipped, labels)
    )
    reliability = []
    weighted_gap = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            (probability, label)
            for probability, label in zip(clipped, labels)
            if low <= probability < high or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        mean_confidence = statistics.fmean(value[0] for value in members)
        accuracy = statistics.fmean(value[1] for value in members)
        weighted_gap += len(members) / len(clipped) * abs(mean_confidence - accuracy)
        reliability.append({
            "low": low, "high": high, "n": len(members),
            "mean_confidence": mean_confidence, "accuracy": accuracy,
        })
    return {"n": len(clipped), "brier": brier, "log_loss": log_loss, "ece": weighted_gap, "reliability": reliability}


def probability_metrics(probabilities: list[float], labels: list[float], bins: int = 10) -> dict[str, Any]:
    """Reliability diagnostics for independently labeled probabilities."""
    return _probability_metrics(probabilities, labels, bins)


def _decision_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decided = [r for r in rows if r["decision"] is not None]
    action_counts = Counter(r["decision"].action for r in decided)
    accepted = [r for r in decided if r["decision"].action == "accept"]
    accepted_correct = []
    for row in accepted:
        valid = set(row["case"].gold_ids + row["case"].acceptable_ids)
        accepted_correct.append(1.0 if row["decision"].accepted_item_id in valid else 0.0)
    expected_clarify = [r for r in decided if r["expected_action"] == "clarify"]
    actual_clarify = [r for r in decided if r["decision"].action == "clarify"]
    true_clarify = [r for r in actual_clarify if r["expected_action"] == "clarify"]
    option_recall = []
    option_precision = []
    option_exact = []
    for row in expected_clarify:
        wanted = set(row["case"].clarification_ids or row["case"].gold_ids + row["case"].acceptable_ids)
        offered = {option.item_id for option in row["decision"].options}
        overlap = wanted & offered
        option_recall.append(len(overlap) / len(wanted) if wanted else 0.0)
        option_precision.append(len(overlap) / len(offered) if offered else 0.0)
        option_exact.append(1.0 if wanted == offered else 0.0)
    precision = len(true_clarify) / len(actual_clarify) if actual_clarify else None
    recall = len(true_clarify) / len(expected_clarify) if expected_clarify else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    confidence_rows = [r for r in decided if r["decision"].calibrated_confidence is not None]
    top_probabilities = [float(r["decision"].calibrated_confidence) for r in confidence_rows]
    top_labels = []
    domain_probabilities = []
    domain_labels = []
    for row in confidence_rows:
        valid = set(row["case"].gold_ids + row["case"].acceptable_ids + row["case"].clarification_ids)
        top_id = row["prediction"].hits[0].item_id if row["prediction"] and row["prediction"].hits else None
        top_labels.append(1.0 if (not row["case"].is_ood and top_id in valid) else 0.0)
        domain_probabilities.append(float(row["decision"].in_domain_confidence))
        domain_labels.append(0.0 if row["case"].is_ood else 1.0)
    return {
        "n": len(decided),
        "action_counts": dict(action_counts),
        "action_accuracy": _mean([1.0 if r["decision"].action == r["expected_action"] else 0.0 for r in decided]),
        "accept_coverage": len(accepted) / len(decided) if decided else None,
        "accepted_accuracy": _mean(accepted_correct),
        "incorrect_accept_rate": _mean([1.0 - value for value in accepted_correct]),
        "clarification_precision": precision,
        "clarification_recall": recall,
        "clarification_f1": f1,
        "option_recall": _mean(option_recall),
        "option_precision": _mean(option_precision),
        "option_exact_match": _mean(option_exact),
        "mean_option_count": _mean([float(len(r["decision"].options)) for r in actual_clarify]),
        "over_clarification_rate": _mean([1.0 if r["decision"].action == "clarify" else 0.0 for r in decided if r["expected_action"] == "accept"]),
        "top_correct_calibration": _probability_metrics(top_probabilities, top_labels),
        "in_domain_calibration": _probability_metrics(domain_probabilities, domain_labels),
    }


def _ood_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ood = [r for r in rows if r["case"].is_ood and r["prediction"] is not None]
    in_domain = [r for r in rows if not r["case"].is_ood and r["prediction"] is not None]
    def score(row):
        return row["prediction"].top1_score if row["prediction"].top1_score is not None else -1.0
    def margin(row):
        return row["prediction"].margin if row["prediction"].margin is not None else 0.0
    ood_decided = [r for r in rows if r["case"].is_ood and r["decision"] is not None]
    return {
        "ood_n": sum(r["case"].is_ood for r in rows),
        "predicted_ood_n": len(ood),
        "ood_reject_rate": _mean([1.0 if r["decision"].action == "reject" else 0.0 for r in ood_decided]),
        "ood_false_accept_rate": _mean([1.0 if r["decision"].action == "accept" else 0.0 for r in ood_decided]),
        "in_domain_score_mean": _mean([score(r) for r in in_domain]),
        "ood_score_mean": _mean([score(r) for r in ood]),
        "in_domain_margin_mean": _mean([margin(r) for r in in_domain]),
        "ood_margin_mean": _mean([margin(r) for r in ood]),
        "max_score": {"in_domain": [score(r) for r in in_domain], "ood": [score(r) for r in ood]},
    }
