from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_intent.serving_config import load_yaml
from model_intent.category_confidence_eval import (
    evaluate_reviewed_cases,
    load_reviewed_cases,
    serving_identity,
    validate_review_manifest,
)
from model_intent.db import validate_database_target
from model_intent.search_runtime import SearchRuntime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/benchmark.poi.pg.v3.yaml"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Read-only category confidence review; requires independently adjudicated real queries")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--taxonomy", type=Path, help="default: taxonomy_path from config")
    parser.add_argument("--cases", type=Path, help="privacy-reviewed, human-adjudicated JSONL")
    parser.add_argument("--manifest", type=Path, help="frozen dataset and serving identity JSON")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--describe-runtime", action="store_true", help="print serving identity for preparing a review manifest")
    args = parser.parse_args(argv)
    try:
        if not args.describe_runtime and (args.cases is None or args.manifest is None):
            raise ValueError("--cases and --manifest are required for evaluation")
        config_path = args.config.resolve()
        config = load_yaml(config_path)
        if config.get("retrieval_backend") != "pgvector" or config.get("retrieval_mode") != "pg_exact":
            raise ValueError("confidence evaluation requires read-only exact pgvector retrieval")
        validate_database_target(config)
        taxonomy = args.taxonomy or config_path.parent.parent / str(config["taxonomy_path"])
        runtime = SearchRuntime.from_intent_config(
            config_path=config_path, taxonomy_path=taxonomy,
            model_key="e5-base-multilingual", local_files_only=True,
        )
        try:
            identity = serving_identity(runtime, config_path)
            if args.describe_runtime:
                print(json.dumps({"serving_identity": identity}, ensure_ascii=False, indent=2))
                return 0
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            validate_review_manifest(manifest, identity, args.cases)
            cases = load_reviewed_cases(args.cases, runtime.category_runtime.taxonomy)
            report = evaluate_reviewed_cases(runtime, cases, split=args.split)
            print(json.dumps({
                "dataset_version": manifest["dataset_version"],
                "serving_identity": identity,
                **report,
            }, ensure_ascii=False, indent=2))
        finally:
            runtime.close()
        return 0
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        print(f"evaluate_category_confidence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
