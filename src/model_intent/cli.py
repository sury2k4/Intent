from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import _worker_command, load_yaml, make_embedder, profiles_from_config, run_benchmark
from .db import Database
from .embedding_sync import activate_generation, cleanup_generations, embedding_status, sync_embeddings
from .embedders import FakeEmbedder
from .migrations import apply_migrations, migration_status
from .pg_benchmark import run_pgvector_benchmark
from .runtime import MatcherRuntime
from .taxonomy import Taxonomy, TaxonomyError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v1.json"
DEFAULT_EVAL = ROOT / "data/eval/place_type_queries.poc.v1.jsonl"
DEFAULT_CONFIG = ROOT / "configs/benchmark.cpu.yaml"
DEFAULT_MIGRATIONS = ROOT / "migrations"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-intent")
    sub = parser.add_subparsers(dest="command", required=True)

    taxonomy = sub.add_parser("taxonomy")
    tsub = taxonomy.add_subparsers(dest="taxonomy_command", required=True)
    validate = tsub.add_parser("validate"); _common(validate)
    inspect = tsub.add_parser("inspect"); _common(inspect); inspect.add_argument("--id", required=True)

    models = sub.add_parser("models")
    msub = models.add_subparsers(dest="models_command", required=True)
    smoke = msub.add_parser("smoke-test")
    smoke.add_argument("--config", default=str(DEFAULT_CONFIG)); smoke.add_argument("--model", required=True); smoke.add_argument("--local-files-only", action="store_true")

    query = sub.add_parser("query")
    _common(query); query.add_argument("--config", default=str(DEFAULT_CONFIG)); query.add_argument("--model", default="e5-base-multilingual"); query.add_argument("--query", required=True); query.add_argument("--top-k", type=int, default=5); query.add_argument("--language", choices=("vi", "en"), default="vi"); query.add_argument("--generation-id", type=int); query.add_argument("--selected-id"); query.add_argument("--allowed-option-id", action="append", default=[]); query.add_argument("--min-score", type=float); query.add_argument("--min-margin", type=float); query.add_argument("--local-files-only", action="store_true")

    db = sub.add_parser("db")
    dbsub = db.add_subparsers(dest="db_command", required=True)
    migrate = dbsub.add_parser("migrate"); migrate.add_argument("--migrations", default=str(DEFAULT_MIGRATIONS))
    dbstatus = dbsub.add_parser("status"); dbstatus.add_argument("--migrations", default=str(DEFAULT_MIGRATIONS))

    embeddings = sub.add_parser("embeddings")
    esub = embeddings.add_subparsers(dest="embeddings_command", required=True)
    sync = esub.add_parser("sync"); _common(sync); sync.add_argument("--config", default=str(DEFAULT_CONFIG)); sync.add_argument("--local-files-only", action="store_true")
    esub.add_parser("status")
    activate = esub.add_parser("activate"); activate.add_argument("generation_id", type=int)
    cleanup = esub.add_parser("cleanup"); cleanup.add_argument("--keep", type=int, default=3)

    benchmark = sub.add_parser("benchmark")
    bsub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    run = bsub.add_parser("run")
    _common(run); run.add_argument("--config", default=str(DEFAULT_CONFIG)); run.add_argument("--evaluation", default=str(DEFAULT_EVAL)); run.add_argument("--include-stretch", action="store_true"); run.add_argument("--local-files-only", action="store_true"); run.add_argument("--model")
    worker = bsub.add_parser("worker", help=argparse.SUPPRESS)
    _common(worker); worker.add_argument("--config", required=True); worker.add_argument("--evaluation", required=True); worker.add_argument("--model", required=True); worker.add_argument("--result", required=True); worker.add_argument("--local-files-only", action="store_true")
    pgrun = bsub.add_parser("pgvector")
    _common(pgrun); pgrun.add_argument("--config", default=str(DEFAULT_CONFIG)); pgrun.add_argument("--evaluation", default=str(DEFAULT_EVAL)); pgrun.add_argument("--output", default=str(ROOT / "artifacts/pgvector/benchmark.json")); pgrun.add_argument("--generation-id", type=int); pgrun.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "taxonomy":
            taxonomy = Taxonomy.load(args.taxonomy)
            if args.taxonomy_command == "validate":
                warnings = taxonomy.validate()
                print(json.dumps({"status": "ok", "items": len(taxonomy.items), "fingerprint": taxonomy.fingerprint(), "warnings": warnings}, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(taxonomy.inspect(str(args.id)), ensure_ascii=False, indent=2))
            return 0
        if args.command == "models" and args.models_command == "smoke-test":
            config = load_yaml(args.config)
            profiles = [p for p in profiles_from_config(config, include_stretch=True) if p.key == args.model]
            if args.model == "fake":
                embedder = FakeEmbedder(); print(json.dumps(embedder.metadata(), indent=2)); return 0
            if not profiles: raise ValueError(f"unknown model profile: {args.model}")
            embedder = make_embedder(profiles[0], local_files_only=args.local_files_only)
            print(json.dumps(embedder.metadata(), ensure_ascii=False, indent=2)); return 0
        if args.command == "db":
            database = Database(statement_timeout_ms=120000); database.open()
            try:
                if args.db_command == "migrate":
                    payload = {"applied": apply_migrations(database, args.migrations), "database": database.check()}
                else:
                    payload = {"database": database.check(), "migrations": migration_status(database, args.migrations)}
                print(json.dumps(payload, ensure_ascii=False, indent=2, default=str)); return 0
            finally:
                database.close()
        if args.command == "embeddings":
            database = Database(statement_timeout_ms=120000); database.open()
            try:
                if args.embeddings_command == "sync":
                    payload = sync_embeddings(database, config_path=args.config, taxonomy_path=args.taxonomy, local_files_only=args.local_files_only)
                    payload = payload.__dict__
                elif args.embeddings_command == "activate":
                    activate_generation(database, args.generation_id); payload = {"activated": args.generation_id}
                elif args.embeddings_command == "cleanup":
                    payload = {"removed_generation_ids": cleanup_generations(database, args.keep), "keep": args.keep}
                else:
                    payload = embedding_status(database)
                print(json.dumps(payload, ensure_ascii=False, indent=2, default=str)); return 0
            finally:
                database.close()
        if args.command == "query":
            runtime = MatcherRuntime.from_config(
                config_path=args.config,
                taxonomy_path=args.taxonomy,
                model_key=args.model,
                local_files_only=args.local_files_only,
                generation_id=args.generation_id,
            )
            if args.selected_id:
                payload = runtime.resolve_selection(
                    query=args.query,
                    selected_id=args.selected_id,
                    allowed_option_ids=args.allowed_option_id,
                    language=args.language,
                )
            else:
                overrides = {}
                if args.min_score is not None:
                    overrides["accept_min_score"] = args.min_score
                if args.min_margin is not None:
                    overrides["accept_min_margin"] = args.min_margin
                payload = runtime.query(
                    args.query,
                    top_k=args.top_k,
                    language=args.language,
                    overrides=overrides,
                )
            print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0
        if args.command == "benchmark" and args.benchmark_command == "worker":
            return _worker_command(args.taxonomy, args.evaluation, args.config, args.model, args.result, args.local_files_only)
        if args.command == "benchmark" and args.benchmark_command == "pgvector":
            payload = run_pgvector_benchmark(
                config_path=args.config, taxonomy_path=args.taxonomy,
                evaluation_path=args.evaluation, local_files_only=args.local_files_only,
                output_path=args.output, generation_id=args.generation_id,
            )
            print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, ensure_ascii=False, indent=2, default=str)); return 0
        if args.command == "benchmark" and args.benchmark_command == "run":
            results = run_benchmark(args.taxonomy, args.evaluation, args.config, args.include_stretch, args.local_files_only, args.model)
            print(json.dumps([{"model_key": r["model_key"], "error": r.get("error"), "metrics": r.get("metrics", {}).get("overall", {})} for r in results], ensure_ascii=False, indent=2, default=str))
            return 0 if all(not r.get("error") for r in results) else 2
    except (ValueError, TaxonomyError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
