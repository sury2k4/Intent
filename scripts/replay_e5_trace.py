from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from model_intent.decision import decide_intent, decision_to_dict, load_decision_policy, validate_decision_policy
from model_intent.osm_ingest import load_property_catalog
from model_intent.poi_repository import PoiRepository
from model_intent.property_extraction import PropertyExtractor
from model_intent.query_interpretation import InterpretedQuery, interpret_query
from model_intent.rendering import load_enrichment
from model_intent.schemas import Prediction, RetrievalHit
from model_intent.search_runtime import SearchRuntime
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_from_trace(row: dict[str, Any]) -> Prediction:
    hits = tuple(RetrievalHit(
        item_id=str(hit["item_id"]), code=str(hit["code"]), score=float(hit["score"]),
        rank=int(hit["rank"]), language=str(hit["language"]),
        matched_text="", query_variant_kind=str(hit.get("query_variant_kind", "original")),
        profile_kind=str(hit.get("profile_kind", "primary")),
    ) for hit in row["hits"])
    return Prediction(
        query_id=str(row.get("query_id", "query")), query=str(row["query"]), hits=hits,
        top1_score=row.get("top1_score"), margin=row.get("margin"),
        model_key=str(row["model_key"]),
        query_variants=tuple(row.get("query_variants") or ()),
        variant_top_ids=tuple(row.get("variant_top_ids") or ()),
    )


def replay_row(
    row: dict[str, Any], runtime: SearchRuntime, calibration: dict[str, Any],
    policy: dict[str, Any], *,
    interpretation: InterpretedQuery | None = None,
    predictions: tuple[Prediction, ...] | None = None,
) -> dict[str, Any]:
    taxonomy = runtime.category_runtime.taxonomy
    enrichment = runtime.category_runtime.enrichment
    if interpretation is None:
        interpretation = interpret_query(
            row["query"], runtime.property_extractor.extract(row["query"]), taxonomy, enrichment,
        )
    stored_predictions = row["predictions"]
    if tuple(pred["query"] for pred in stored_predictions) != interpretation.variants:
        raise RuntimeError(f"trace variant drift at query {row['query_id']}")
    if predictions is None:
        predictions = tuple(prediction_from_trace(stored) for stored in stored_predictions)
    outcomes = []
    for stored, prediction in zip(stored_predictions, predictions):
        decision = decide_intent(
            prediction, taxonomy, policy, enrichment, "vi", None, calibration,
            policy_prevalidated=True,
        )
        outcomes.append({"prediction": stored, "decision": decision_to_dict(decision)})
    return runtime._choose_real_category(row["query"], outcomes, interpretation, "vi")["decision"]


def summarize(rows: list[dict[str, Any]], decisions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    categories = [row for row in rows if row["query_type"] == "in_domain"]
    ood = [row for row in rows if row["query_type"] == "ood"]
    ambiguous = [row for row in rows if row["query_type"] == "ambiguous"]
    accepted = [row for row in categories if decisions[row["query_id"]]["action"] == "accept"]
    correct = sum(
        str(decisions[row["query_id"]]["accepted_item_id"]) in
        {str(item) for item in row["target_category_ids"]}
        for row in accepted
    )
    def retrieved(row: dict[str, Any], top_k: int) -> bool:
        selected_query = row.get("category_resolution", {}).get("semantic_query")
        prediction = next(
            (item for item in row["predictions"] if item["query"] == selected_query),
            row["predictions"][0],
        )
        targets = {str(item) for item in row["target_category_ids"]}
        return any(str(hit["item_id"]) in targets for hit in prediction["hits"][:top_k])
    return {
        "in_domain_count": len(categories),
        "raw_top1_correct_proxy_gold": sum(retrieved(row, 1) for row in categories),
        "raw_top3_correct_proxy_gold": sum(retrieved(row, 3) for row in categories),
        "accepted": len(accepted),
        "accepted_correct_proxy_gold": correct,
        "correct_accept_rate_proxy_gold": correct / len(categories) if categories else None,
        "accept_precision_proxy_gold": correct / len(accepted) if accepted else None,
        "ood_count": len(ood),
        "ood_false_accepts": sum(decisions[row["query_id"]]["action"] == "accept" for row in ood),
        "ambiguous_count": len(ambiguous),
        "ambiguous_clarify": sum(decisions[row["query_id"]]["action"] == "clarify" for row in ambiguous),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact offline E5 policy replay from full-precision traces")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "data/taxonomies/place_types.poc.v2.json")
    parser.add_argument("--enrichment", type=Path, default=ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v2.json")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/properties/poi_properties.poc.v3.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "data/policies/place_type_decision_policy.poc.v1.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "data/policies/e5_base_confidence_calibration.v2.json")
    parser.add_argument("--gold-policy", type=Path, default=ROOT / "data/eval/backtest_gold_semantics.poc.v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError(f"refusing to overwrite existing policy report: {args.output}")
    taxonomy = Taxonomy.load(args.taxonomy)
    enrichment = load_enrichment(args.enrichment)
    property_catalog = load_property_catalog(args.properties)
    policy = load_decision_policy(args.policy)
    validate_decision_policy(policy, taxonomy)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("taxonomy_fingerprint") != taxonomy.fingerprint():
        raise ValueError("calibration belongs to a different taxonomy")
    category_stub = SimpleNamespace(taxonomy=taxonomy, enrichment=enrichment)
    runtime = SearchRuntime(
        category_runtime=category_stub, property_extractor=PropertyExtractor(property_catalog),
        poi_repository=PoiRepository(()), ranking_policy={}, property_catalog=property_catalog,
    )
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    trace_manifest_path = args.trace.parent / (args.trace.stem + ".manifest.json")
    trace_manifest = json.loads(trace_manifest_path.read_text(encoding="utf-8"))
    if trace_manifest["trace_sha256"] != file_sha256(args.trace):
        raise ValueError("E5 trace does not match its captured checksum")
    if trace_manifest["taxonomy_fingerprint"] != taxonomy.fingerprint():
        raise ValueError("E5 trace belongs to a different taxonomy")
    if int(trace_manifest["count"]) != len(rows):
        raise ValueError("E5 trace row count differs from its manifest")
    prepared = {
        row["query_id"]: (
            interpret_query(row["query"], runtime.property_extractor.extract(row["query"]), taxonomy, enrichment),
            tuple(prediction_from_trace(pred) for pred in row["predictions"]),
        )
        for row in rows
    }
    def run(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        interpretation, predictions = prepared[row["query_id"]]
        return replay_row(
            row, runtime, candidate, policy,
            interpretation=interpretation, predictions=predictions,
        )

    baseline = {row["query_id"]: run(row, calibration) for row in rows}
    mismatch = [row["query_id"] for row in rows if any(
        baseline[row["query_id"]].get(key) != row["final_decision"].get(key)
        for key in ("action", "reason", "accepted_item_id")
    )]
    if mismatch:
        raise RuntimeError(f"cannot sweep: replay differs from captured runtime for {len(mismatch)} cases, first={mismatch[:5]}")
    invalid = set(json.loads(args.gold_policy.read_text(encoding="utf-8"))["unsupported_or_proxy_labels"])
    valid_validation = [row for row in rows if row["split"] == "validation" and row["query_type"] == "in_domain" and row["gold_label"] not in invalid]
    valid_test = [row for row in rows if row["split"] == "test" and row["query_type"] == "in_domain" and row["gold_label"] not in invalid]
    test_ood = [row for row in rows if row["split"] == "test" and row["query_type"] == "ood"]
    ambiguous_validation = [row for row in rows if row["split"] == "val" and row["query_type"] == "ambiguous"]
    ambiguous_test = [row for row in rows if row["split"] == "test" and row["query_type"] == "ambiguous"]
    current_threshold = float(calibration["action_thresholds"]["accept_min_top_confidence"])
    candidates = sorted({0.55, 0.575, 0.60, current_threshold, 0.625, 0.65, 0.70})
    sweep = []
    for threshold in candidates:
        adjusted = dict(calibration)
        adjusted["action_thresholds"] = {
            **calibration["action_thresholds"], "accept_min_top_confidence": threshold,
        }
        chosen_rows = valid_validation + ambiguous_validation
        decisions = {row["query_id"]: run(row, adjusted) for row in chosen_rows}
        sweep.append({
            "accept_min_top_confidence": threshold,
            "validation_supported": summarize(valid_validation, decisions),
            "validation_ambiguous": summarize(ambiguous_validation, decisions),
        })
    report = {
        "schema_version": 2,
        "trace_count": len(rows),
        "baseline_replay_mismatches": len(mismatch),
        "calibration_version": calibration["calibration_version"],
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "baseline_all_proxy_gold": summarize(rows, baseline),
        "baseline_validation_supported": summarize(valid_validation, baseline),
        "baseline_blind_test_supported": summarize(valid_test, baseline),
        "baseline_blind_test_ood": summarize(test_ood, baseline),
        "baseline_blind_test_ambiguous": summarize(ambiguous_test, baseline),
        "selection_split": "validation and val only",
        "excluded_proxy_labels": sorted(invalid),
        "ood_tuning_available": False,
        "decision": "retain_baseline_until_independent_ood_validation_and_adjudicated_gold_are_available",
        "validation_sweep": sweep,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
