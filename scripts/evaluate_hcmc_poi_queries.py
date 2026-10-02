from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_intent.poi_ranking import satisfies_constraint, satisfies_exclusion
from model_intent.poi_schemas import ExtractedProperty
from model_intent.search_runtime import SearchRuntime

ROOT = Path(__file__).resolve().parents[1]


def _property_key(row: dict) -> tuple[str, str, str, str]:
    return (str(row["property_id"]), str(row["mode"]), str(row.get("operator", "eq")), repr(row.get("value")))


def _full_key(row: dict) -> tuple[str, ...]:
    return (*_property_key(row), str(row.get("applied", True)), str(row.get("ignored_reason")))


def evaluate(cases: list[dict], runtime: SearchRuntime) -> dict:
    true_positive = false_positive = false_negative = full_exact = 0
    action_correct = category_correct = status_correct = clarification_correct = interpretation_correct = 0
    violations = 0
    failures = []
    records_by_id = {record.poi_id: record for record in runtime.poi_repository.records}
    for case in cases:
        query = case["query"]
        result = runtime.search(
            query,
            language=case.get("language", "vi"),
            origin=tuple(case["origin"]) if case.get("origin") else None,
            verified_only=bool(case.get("verified_only", False)),
            poi_top_k=5,
        )
        decision = result["intent"]["decision"]
        search = result["poi_search"]
        properties = result["property_extraction"]["properties"]
        expected = case.get("expected_properties", [])
        predicted_keys = {_property_key(row) for row in properties if row["applied"]}
        expected_keys = {_property_key(row) for row in expected if row.get("applied", True)}
        true_positive += len(predicted_keys & expected_keys)
        false_positive += len(predicted_keys - expected_keys)
        false_negative += len(expected_keys - predicted_keys)
        exact_properties = sorted(_full_key(row) for row in properties) == sorted(_full_key(row) for row in expected)
        full_exact += int(exact_properties)
        action_ok = decision.get("action") == case["expected_action"]
        category_ok = decision.get("accepted_item_id") == case.get("expected_category_id")
        status_ok = search["status"] == case["expected_status"]
        if case.get("expected_options"):
            option_ids = {str(row["item_id"]) for row in decision.get("options", [])}
            clarification_ok = option_ids == set(case["expected_options"])
        else:
            clarification_ok = True
        action_correct += int(action_ok)
        category_correct += int(category_ok)
        status_correct += int(status_ok)
        clarification_correct += int(clarification_ok)
        interpretation = result["intent"].get("query_interpretation") or {}
        focused_ok = (
            case.get("expected_category_text") is None
            or interpretation.get("category_text") == case["expected_category_text"]
        )
        roles_ok = (
            case.get("expected_property_roles") is None
            or [row["role"] for row in interpretation.get("property_roles", [])] == case["expected_property_roles"]
        )
        interpretation_ok = focused_ok and roles_ok
        interpretation_correct += int(interpretation_ok)

        current_violations = 0
        if decision.get("action") in {"reject", "clarify"} and search["hits"]:
            current_violations += len(search["hits"])
        for hit in search["hits"]:
            poi = records_by_id[hit["poi_id"]]
            for row in properties:
                if not row["applied"] or row["property_id"] == "radius_km":
                    continue
                constraint = ExtractedProperty(
                    property_id=row["property_id"], mode=row["mode"], operator=row["operator"], value=row["value"],
                )
                if row["mode"] == "required" and not satisfies_constraint(poi, constraint, verified_only=case.get("verified_only", False)):
                    current_violations += 1
                if row["mode"] == "excluded" and not satisfies_exclusion(poi, constraint, verified_only=case.get("verified_only", False)):
                    current_violations += 1
        violations += current_violations
        if not all((action_ok, category_ok, status_ok, clarification_ok, exact_properties, interpretation_ok)) or current_violations:
            failures.append({
                "id": case["id"], "query": query,
                "action": decision.get("action"), "category_id": decision.get("accepted_item_id"),
                "status": search["status"], "predicted_properties": [_full_key(row) for row in properties],
                "category_text": interpretation.get("category_text"),
                "property_roles": [row.get("role") for row in interpretation.get("property_roles", [])],
                "filter_violations": current_violations,
            })
    total = len(cases)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    return {
        "cases": total,
        "category_action_accuracy": action_correct / total if total else 1.0,
        "category_id_accuracy": category_correct / total if total else 1.0,
        "clarification_option_accuracy": clarification_correct / total if total else 1.0,
        "query_interpretation_accuracy": interpretation_correct / total if total else 1.0,
        "poi_search_status_accuracy": status_correct / total if total else 1.0,
        "property_precision": precision,
        "property_recall": recall,
        "property_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "property_full_exact_accuracy": full_exact / total if total else 1.0,
        "hard_filter_violations": violations,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate query-to-POI behavior against the pinned HCMC artifact")
    parser.add_argument("--evaluation", default=str(ROOT / "data/eval/hcmc_poi_queries.v3.jsonl"))
    parser.add_argument("--config", default=str(ROOT / "configs/benchmark.poi.v2.yaml"))
    parser.add_argument("--taxonomy", default=str(ROOT / "data/taxonomies/place_types.poc.v2.json"))
    parser.add_argument("--model", default="fake")
    args = parser.parse_args(argv)
    cases = [
        case for line in Path(args.evaluation).read_text(encoding="utf-8").splitlines()
        if line.strip() and (case := json.loads(line)).get("model", args.model) == args.model
    ]
    runtime = SearchRuntime.from_config(
        config_path=args.config, taxonomy_path=args.taxonomy, model_key=args.model, local_files_only=True,
    )
    try:
        result = evaluate(cases, runtime)
    finally:
        runtime.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["failures"] and result["hard_filter_violations"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
