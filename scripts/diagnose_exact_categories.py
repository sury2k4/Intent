from __future__ import annotations

"""Read-only v3 category-name diagnostics; report JSON to stdout, never tune serving."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from model_intent.serving_config import load_yaml
from model_intent.category_resolution import explicit_categories
from model_intent.db import validate_database_target
from model_intent.normalization import fold_vietnamese, normalize_text
from model_intent.rendering import load_enrichment
from model_intent.runtime import MatcherRuntime
from model_intent.search_runtime import SearchRuntime
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/benchmark.poi.pg.v3.yaml"
PREFIXES = {"vi": "tìm ", "en": "find "}
NEGATIVES = (
    ("vi", "thời tiết ngày mai thế nào?"),
    ("vi", "cách nấu cơm ngon"),
    ("vi", "cách mở nhà thuốc"),
    ("vi", "không phải quán cà phê"),
    ("vi", "nhà hàng hoặc nhà thuốc"),
    ("vi", "tìm phần mềm quản lý nhà thuốc"),
    ("en", "what is the weather tomorrow?"),
    ("en", "how to cook rice"),
    ("en", "how to open a pharmacy"),
    ("en", "not a hospital"),
)
CALIBRATION_REASONS = {"low_calibrated_confidence", "insufficient_calibrated_evidence"}


@dataclass(frozen=True)
class Case:
    query: str
    language: str
    kind: str
    expected_item_id: str | None


def _key(text: str) -> str:
    # The explicit-name guard folds accents; collisions must use the same key.
    return normalize_text(fold_vietnamese(text))


def build_cases(
    taxonomy: Taxonomy,
    enrichment: Mapping[str, Mapping[str, Any]],
    *,
    item_ids: set[str] | None = None,
    include_aliases: bool = False,
    include_negatives: bool = True,
) -> tuple[list[Case], list[dict[str, Any]]]:
    """Exclude globally ambiguous names (including collisions with other leaf aliases)."""
    leaves = {item_id: item for item_id, item in taxonomy.items.items() if not taxonomy.has_children(item_id)}
    if item_ids is not None:
        unknown = item_ids - leaves.keys()
        if unknown:
            raise ValueError(f"unknown or non-leaf item IDs: {sorted(unknown)}")
    phrases: dict[str, set[str]] = defaultdict(set)
    for item_id, item in leaves.items():
        for name in item.names.values():
            if _key(name):
                phrases[_key(name)].add(item_id)
        for language in PREFIXES:
            for alias in (enrichment.get(item_id) or {}).get(f"aliases_{language}", ()):
                if _key(alias):
                    phrases[_key(alias)].add(item_id)

    cases: list[Case] = []
    collisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item_id in sorted(item_ids if item_ids is not None else leaves, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        item = leaves[item_id]
        for language, prefix in PREFIXES.items():
            name = item.names.get(language)
            candidates = [("name", name)] if name else []
            if include_aliases:
                candidates.extend(("alias", alias) for alias in (enrichment.get(item_id) or {}).get(f"aliases_{language}", ()))
            for kind, phrase in candidates:
                phrase = str(phrase).strip()
                key = _key(phrase)
                if not key:
                    continue
                owners = phrases[key]
                if len(owners) != 1:
                    collisions.append({
                        "item_id": item_id, "language": language, "kind": kind,
                        "phrase": phrase, "colliding_item_ids": sorted(owners),
                    })
                    continue
                # Avoid repeating an alias that is just a canonical name under accent folding.
                if (item_id, language, key) in seen:
                    continue
                seen.add((item_id, language, key))
                cases.append(Case(phrase, language, f"{kind}_bare", item_id))
                cases.append(Case(f"{prefix}{phrase}", language, f"{kind}_prefixed", item_id))
    if include_negatives:
        cases.extend(Case(query, language, "negative", None) for language, query in NEGATIVES)
    return cases, collisions


def _decision_fields(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: decision.get(key) for key in (
            "action", "reason", "accepted_item_id", "calibrated_confidence",
            "in_domain_confidence", "ambiguity_score", "calibration_version", "policy_version",
        )
    }


def _metadata(runtime: MatcherRuntime | SearchRuntime) -> dict[str, Any]:
    category = runtime.category_runtime if isinstance(runtime, SearchRuntime) else runtime
    result = asdict(category.metadata)
    result["calibration_version"] = (
        (category.confidence_calibration or {}).get("calibration_version")
    )
    return result


def diagnose(cases: list[Case], runtime: MatcherRuntime | SearchRuntime, *, top_k: int) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    rows: list[dict[str, Any]] = []
    stages: Counter[str] = Counter()
    final_correct = final_false_accept = 0
    for case in cases:
        if isinstance(runtime, SearchRuntime):
            intent = runtime.analyze_intent(
                case.query, category_top_k=top_k, language=case.language,
            )["intent"]
            resolution = intent.get("category_resolution") or {}
            semantic = resolution.get("semantic_decision") or resolution.get("original_semantic_decision")
            if semantic is not None:
                # Literal rescue changes only action/reason/accepted ID; the
                # calibrated scores on the final decision still describe E5.
                semantic = {**intent["decision"], "accepted_item_id": None, "accepted_code": None, **semantic}
            elif resolution.get("source") in {"semantic_model", "alias_canonical_probe"}:
                semantic = intent["decision"]
            if semantic is None:
                # A guard may have changed the decision. Re-query the *chosen*
                # semantic variant, not the raw query, to separate its decision.
                semantic_query = resolution.get("semantic_query") or case.query
                semantic = runtime.category_runtime.query(
                    semantic_query, top_k=top_k, language=case.language,
                )["decision"]
        else:
            intent = runtime.query(case.query, top_k=top_k, language=case.language)
            matches = explicit_categories(case.query, runtime.taxonomy, runtime.enrichment)
            resolution = {
                "source": "matcher_only",
                "explicit_matches": [{"item_id": match.item_id, "evidence": match.evidence} for match in matches],
            }
            semantic = intent["decision"]
        prediction = intent["prediction"]
        decision = intent["decision"]
        hits = prediction.get("hits") or []
        top_id = str(hits[0]["item_id"]) if hits else None
        accepted_id = str(decision["accepted_item_id"]) if decision.get("accepted_item_id") is not None else None
        correct = case.expected_item_id is not None and decision.get("action") == "accept" and accepted_id == case.expected_item_id
        if case.expected_item_id is None:
            stage = "negative_false_accept" if decision.get("action") == "accept" else "negative_not_accepted"
            final_false_accept += int(stage == "negative_false_accept")
        elif top_id != case.expected_item_id:
            stage = "retrieval_miss"
        elif semantic.get("action") == "reject" and semantic.get("reason") in CALIBRATION_REASONS:
            stage = "calibration_reject_rescued" if correct else "calibration_reject"
        else:
            stage = "correct" if correct else "other_decision_miss"
        stages[stage] += 1
        final_correct += int(correct)
        expected_hit = next((hit for hit in hits if str(hit["item_id"]) == case.expected_item_id), None)
        rows.append({
            "query": case.query, "language": case.language, "kind": case.kind,
            "expected_item_id": case.expected_item_id,
            "expected_rank": expected_hit["rank"] if expected_hit else None,
            "expected_cosine": expected_hit["score"] if expected_hit else None,
            "top1_matches_expected": top_id == case.expected_item_id if case.expected_item_id else None,
            "hits": [{"item_id": hit["item_id"], "cosine": hit["score"]} for hit in hits],
            "top1_cosine": prediction.get("top1_score"), "margin": prediction.get("margin"),
            "semantic_decision": _decision_fields(semantic),
            "final_decision": _decision_fields(decision),
            "category_resolution": resolution,
            "runtime": intent.get("runtime"),
            "stage": stage,
        })
    positives = sum(case.expected_item_id is not None for case in cases)
    return {
        "runtime_metadata": _metadata(runtime),
        "summary": {
            "positive_cases": positives, "negative_cases": len(cases) - positives,
            "correct_final_accept": final_correct,
            "negative_false_accept": final_false_accept,
            "retrieval_miss": stages["retrieval_miss"],
            "calibration_reject": stages["calibration_reject"],
            "calibration_reject_rescued": stages["calibration_reject_rescued"],
            "other_decision_miss": stages["other_decision_miss"],
            "stages": dict(stages),
        },
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Diagnose exact v3 leaf-name category retrieval and decisions (JSON on stdout)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--taxonomy", type=Path, help="default: taxonomy_path in config")
    parser.add_argument("--model", default="e5-base-multilingual")
    parser.add_argument("--runtime", choices=("intent", "matcher"), default="intent")
    parser.add_argument("--generation-id", type=int, help="pin category generation; default: active")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--item-id", action="append", dest="item_ids", help="diagnose selected leaf ID (repeatable)")
    parser.add_argument("--limit", type=int, help="limit number of leaves, not cases")
    parser.add_argument("--include-aliases", action="store_true", help="include globally unique leaf aliases")
    parser.add_argument("--no-negatives", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.top_k < 1 or args.limit is not None and args.limit < 1:
            raise ValueError("--top-k and --limit must be positive")
        if args.model == "fake":
            raise ValueError("fake model can build/write a NumPy cache; use a pgvector-backed model")
        config_path = args.config.resolve()
        config = load_yaml(config_path)
        # NumPy initialization writes missing embedding caches. Keep this tool
        # strictly read-only even when the caller overrides its config.
        if config.get("retrieval_backend") != "pgvector" or config.get("retrieval_mode") != "pg_exact":
            raise ValueError("diagnostic requires pgvector pg_exact; NumPy can write a missing cache")
        validate_database_target(config)  # Fail before loading the model when DATABASE_URL is absent/unsafe.
        taxonomy_path = args.taxonomy or (config_path.parent.parent / str(config["taxonomy_path"]))
        taxonomy = Taxonomy.load(taxonomy_path)
        enrichment_path = config_path.parent.parent / str(config["enrichment_path"])
        enrichment = load_enrichment(enrichment_path)
        leaf_ids = [item_id for item_id in taxonomy.items if not taxonomy.has_children(item_id)]
        leaf_ids.sort(key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        selected = set(args.item_ids) if args.item_ids else None
        if selected is not None and (invalid := selected - set(leaf_ids)):
            raise ValueError(f"unknown or non-leaf item IDs: {sorted(invalid)}")
        if args.limit is not None:
            selected = set([item_id for item_id in leaf_ids if selected is None or item_id in selected][:args.limit])
        cases, collisions = build_cases(
            taxonomy, enrichment, item_ids=selected,
            include_aliases=args.include_aliases, include_negatives=not args.no_negatives,
        )
        if args.include_aliases and args.runtime != "matcher" and config.get("category_query_aliases_path"):
            alias_path = Path(config["category_query_aliases_path"])
            if not alias_path.is_absolute():
                alias_path = (config_path.parent.parent / alias_path).resolve()
            curated = json.loads(alias_path.read_text(encoding="utf-8"))
            existing = {(case.language, _key(case.query)) for case in cases}
            for entry in curated["aliases"]:
                item_id, language, phrase = str(entry["category_id"]), entry["language"], entry["phrase"]
                if selected is not None and item_id not in selected:
                    continue
                for kind, query in (("curated_alias_bare", phrase), ("curated_alias_prefixed", PREFIXES[language] + phrase)):
                    signature = (language, _key(query))
                    if signature not in existing:
                        cases.append(Case(query, language, kind, item_id))
                        existing.add(signature)
        if args.runtime == "intent":
            runtime = SearchRuntime.from_intent_config(
                config_path=config_path, taxonomy_path=taxonomy_path, model_key=args.model,
                local_files_only=True, generation_id=args.generation_id,
            )
        else:
            runtime = MatcherRuntime.from_config(
                config_path=config_path, taxonomy_path=taxonomy_path, model_key=args.model,
                local_files_only=True, generation_id=args.generation_id,
            )
        try:
            result = diagnose(cases, runtime, top_k=args.top_k)
        finally:
            if isinstance(runtime, SearchRuntime):
                runtime.close()
            elif runtime.database is not None:
                runtime.database.close()
        result["config"] = str(config_path)
        result["taxonomy"] = str(taxonomy_path.resolve())
        result["runtime_kind"] = args.runtime
        result["skipped_collisions"] = collisions
        result["summary"]["skipped_collision_phrases"] = len(collisions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        print(f"diagnose_exact_categories: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
