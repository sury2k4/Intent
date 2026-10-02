from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from model_intent.retrieval import prediction_to_dict
from model_intent.search_runtime import SearchRuntime

ROOT = Path(__file__).resolve().parents[1]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record complete E5 category predictions for exact offline policy replay")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.poi.v2.yaml")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "data/taxonomies/place_types.poc.v2.json")
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/backtest_port_8502/backtest_predictions_5698.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing E5 trace: {args.output}")
    runtime = SearchRuntime.from_config(
        config_path=args.config, taxonomy_path=args.taxonomy,
        model_key="e5-base-multilingual", local_files_only=True,
    )
    meta = runtime.category_runtime.metadata
    source_sha = checksum(args.input)
    config_sha = checksum(args.config)
    taxonomy_sha = checksum(args.taxonomy)
    predictions: list[dict] = []
    index = runtime.category_runtime.index
    original_search = index.search

    def recording_search(*args_, **kwargs_):
        prediction = original_search(*args_, **kwargs_)
        data = prediction_to_dict(prediction)
        for hit in data["hits"]:
            hit.pop("matched_text", None)
            hit.pop("path", None)
        predictions.append(data)
        return prediction

    index.search = recording_search
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                predictions.clear()
                result = runtime.search(row["query"], language="vi", poi_top_k=1)
                target.write(json.dumps({
                    "query_id": row["query_id"], "split": row["split"],
                    "query_type": row["query_type"], "query": row["query"],
                    "gold_label": row["gold_label"], "target_category_ids": row.get("target_category_ids", []),
                    "group_id": (row.get("raw_item_metadata") or {}).get("group_id"),
                    "raw_item_metadata": row.get("raw_item_metadata", {}),
                    "predictions": list(predictions),
                    "final_decision": result["intent"]["decision"],
                    "query_interpretation": result["intent"]["query_interpretation"],
                    "category_resolution": result["intent"].get("category_resolution", {}),
                    "poi_status": result["poi_search"]["status"],
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                if count % 200 == 0:
                    target.flush()
                    print(f"captured {count} queries", file=sys.stderr, flush=True)
                if args.limit and count >= args.limit:
                    break
    finally:
        index.search = original_search
        runtime.close()
    manifest = {
        "schema_version": 1,
        "capture_kind": "e5-full-category-predictions-v1",
        "source_sha256": source_sha,
        "trace_sha256": checksum(args.output),
        "count": count,
        "taxonomy_fingerprint": meta.taxonomy_fingerprint,
        "model_id": meta.model_id,
        "model_revision": meta.model_revision,
        "embedding_dimension": runtime.category_runtime.embedder.metadata()["dimension"],
        "profile_count": meta.profile_count,
        "calibration_version": runtime.category_runtime.confidence_calibration["calibration_version"],
        "config_sha256": config_sha,
        "taxonomy_sha256": taxonomy_sha,
    }
    (args.output.parent / (args.output.stem + ".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
