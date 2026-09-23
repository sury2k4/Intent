from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .evaluation import load_cases, validate_cases
from .runtime import MatcherRuntime


def run_pgvector_benchmark(
    *,
    config_path: str | Path,
    taxonomy_path: str | Path,
    evaluation_path: str | Path,
    local_files_only: bool = True,
    output_path: str | Path | None = None,
    generation_id: int | None = None,
) -> dict[str, Any]:
    runtime = MatcherRuntime.from_config(
        config_path=config_path,
        taxonomy_path=taxonomy_path,
        model_key="e5-base-multilingual",
        local_files_only=local_files_only,
        generation_id=generation_id,
    )
    cases = [case for case in load_cases(evaluation_path) if case.split == "test"]
    taxonomy_ids = set(runtime.taxonomy.items)
    validate_cases(cases, taxonomy_ids)
    if not cases:
        raise ValueError("evaluation file has no test cases")
    rows = []
    try:
        for case in cases:
            result = runtime.query(case.query, top_k=5, language=case.language if case.language in {"vi", "en"} else "vi")
            top_id = result["prediction"]["hits"][0]["item_id"] if result["prediction"]["hits"] else None
            acceptable = set(case.gold_ids + case.acceptable_ids)
            rows.append({
                "case_id": case.id,
                "action": result["decision"]["action"],
                "top_item_id": top_id,
                "top1_correct": top_id in acceptable if not case.is_ood else None,
                "query_encode_ms": result["runtime"].get("query_encode_ms"),
                "pool_wait_ms": result["runtime"].get("pool_wait_ms"),
                "db_search_ms": result["runtime"].get("db_search_ms"),
                "decision_ms": result["runtime"].get("decision_ms"),
                "total_ms": result["runtime"].get("query_elapsed_ms"),
            })
        def percentiles(key: str) -> dict[str, float]:
            values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
            return {
                "mean": float(fmean(values)), "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            }
        in_domain = [row for row, case in zip(rows, cases) if not case.is_ood]
        report = {
            "backend": runtime.metadata.backend,
            "mode": runtime.metadata.retrieval_mode,
            "generation_id": runtime.metadata.generation_id,
            "model_id": runtime.metadata.model_id,
            "model_revision": runtime.metadata.model_revision,
            "taxonomy_fingerprint": runtime.metadata.taxonomy_fingerprint,
            "candidate_count": runtime.metadata.candidate_count,
            "profile_count": runtime.metadata.profile_count,
            "indexed_category_count": runtime.metadata.indexed_category_count,
            "profile_set_id": runtime.metadata.profile_set_id,
            "artifact_kind": runtime.metadata.artifact_kind,
            "case_count": len(rows),
            "top1_accuracy": fmean([1.0 if row["top1_correct"] else 0.0 for row in in_domain]),
            "action_counts": dict(Counter(row["action"] for row in rows)),
            "latency_ms": {
                key: percentiles(key)
                for key in ("query_encode_ms", "pool_wait_ms", "db_search_ms", "decision_ms", "total_ms")
            },
            "rows": rows,
        }
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        if runtime.database:
            runtime.database.close()
