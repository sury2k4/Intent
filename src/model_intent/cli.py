from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import _worker_command, load_yaml, make_embedder, profiles_from_config, run_benchmark
from .db import Database
from .embedding_sync import (
    embedding_status, model_profile, profile_content_hash, profile_source_fingerprint,
    semantic_model_metadata, stable_hash, sync_embeddings,
)
from .embedders import FakeEmbedder
from .migrations import apply_migrations, migration_status
from .osm_ingest import (
    build_artifact,
    enrich_artifact,
    file_sha256,
    iter_pbf_elements,
    validate_catalogs,
)
from .poi_sync import poi_status, promote_generation_pair, rollback_generation_pair, sync_poi_artifact
from .rendering import (
    PROFILE_RENDERER_VERSION, PROFILE_SCHEMA_VERSION, load_enrichment,
    render_category_profiles, validate_enrichment,
)
from .search_runtime import SearchRuntime
from .pg_benchmark import run_pgvector_benchmark
from .runtime import MatcherRuntime
from .taxonomy import Taxonomy, TaxonomyError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v1.json"
DEFAULT_POI_TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
DEFAULT_EVAL = ROOT / "data/eval/place_type_queries.poc.v1.jsonl"
DEFAULT_CONFIG = ROOT / "configs/benchmark.cpu.yaml"
DEFAULT_SERVING_CONFIG = ROOT / "configs/benchmark.poi.pg.v3.yaml"
DEFAULT_MIGRATIONS = ROOT / "migrations"


def _common(parser: argparse.ArgumentParser, *, taxonomy_path: Path = DEFAULT_TAXONOMY) -> None:
    parser.add_argument("--taxonomy", default=str(taxonomy_path))


def _promotion_provenance(config_path: str, taxonomy_path: str, dataset_key: str) -> dict[str, str]:
    """Pin a candidate pair to the files actually selected for serving."""
    config = load_yaml(config_path)
    poi_config = config.get("poi_search") or {}
    if (config.get("retrieval_backend") != "pgvector"
            or poi_config.get("backend") != "postgres"
            or poi_config.get("dataset_key") != dataset_key):
        raise ValueError("paired promotion requires a matching pgvector/Postgres serving configuration")
    root = Path(config_path).resolve().parent.parent
    configured_taxonomy = config.get("taxonomy_path")
    if configured_taxonomy and (root / configured_taxonomy).resolve() != Path(taxonomy_path).resolve():
        raise ValueError("--taxonomy does not match the taxonomy_path pinned by --config")
    enrichment_path = config.get("enrichment_path")
    if not enrichment_path:
        raise ValueError("serving configuration must pin enrichment_path for E5 promotion")
    taxonomy = Taxonomy.load(taxonomy_path)
    enrichment = load_enrichment(root / enrichment_path)
    validate_enrichment(taxonomy, enrichment, require_rich_fields=True)
    profiles = render_category_profiles(
        taxonomy, tuple(config.get("languages", ["vi", "en"])), enrichment,
        require_rich_fields=True,
    )
    category_ids = {profile.item_id for profile in profiles}
    expected_category_ids = {item.id for item in taxonomy.document.items if not taxonomy.has_children(item.id)}
    if not profiles or category_ids != expected_category_ids or any(not item.isdigit() for item in category_ids):
        raise ValueError("serving category corpus must cover exactly the numeric taxonomy leaf IDs")
    mapping_path = poi_config.get("mapping_path")
    property_path = poi_config.get("property_path")
    if not mapping_path or not property_path:
        raise ValueError("serving configuration must pin both POI mapping and property taxonomy files")
    profile = model_profile(config)
    return {
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "mapping_sha256": file_sha256(root / mapping_path),
        "property_taxonomy_sha256": file_sha256(root / property_path),
        "model_config_hash": stable_hash(semantic_model_metadata(profile)),
        "source_fingerprint": profile_source_fingerprint(enrichment),
        "embedding_content_set_hash": stable_hash(sorted(
            (row.item_id, row.language, profile_content_hash(row)) for row in profiles
        )),
        "renderer_version": PROFILE_RENDERER_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "expected_profile_count": len(profiles),
        "expected_category_count": len(category_ids),
    }


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
    _common(query, taxonomy_path=DEFAULT_POI_TAXONOMY); query.add_argument("--config", default=str(DEFAULT_SERVING_CONFIG)); query.add_argument("--model", default="e5-base-multilingual"); query.add_argument("--query", required=True); query.add_argument("--top-k", type=int, default=5); query.add_argument("--language", choices=("vi", "en"), default="vi"); query.add_argument("--generation-id", type=int); query.add_argument("--selected-id"); query.add_argument("--allowed-option-id", action="append", default=[]); query.add_argument("--min-score", type=float); query.add_argument("--min-margin", type=float); query.add_argument("--local-files-only", action="store_true")

    pois = sub.add_parser("pois")
    psub = pois.add_subparsers(dest="pois_command", required=True)
    fetch = psub.add_parser("fetch")
    fetch.add_argument("--url", default="https://download.geofabrik.de/asia/vietnam-latest.osm.pbf")
    fetch.add_argument("--output", required=True)
    pvalidate = psub.add_parser("validate")
    _common(pvalidate); pvalidate.add_argument("--mapping", default=str(ROOT / "data/osm/mappings/place_types.poc.v1.json")); pvalidate.add_argument("--properties", default=str(ROOT / "data/properties/poi_properties.poc.v1.json"))
    build = psub.add_parser("build")
    _common(build); build.add_argument("--mapping", default=str(ROOT / "data/osm/mappings/place_types.poc.v1.json")); build.add_argument("--properties", default=str(ROOT / "data/properties/poi_properties.poc.v1.json")); build.add_argument("--boundary", default=str(ROOT / "data/osm/boundaries/hcmc.admin.v1.geojson")); build.add_argument("--pbf"); build.add_argument("--elements-jsonl"); build.add_argument("--source-manifest", default=str(ROOT / "data/osm/sources/hcmc.poc.v1.json")); build.add_argument("--output", required=True); build.add_argument("--seed", type=int, default=17)
    enrich = psub.add_parser("enrich")
    _common(enrich); enrich.add_argument("--artifact", required=True); enrich.add_argument("--output", required=True)
    enrich.add_argument("--mapping", default=str(ROOT / "data/osm/mappings/place_types.poc.v1.json"))
    enrich.add_argument("--properties", default=str(ROOT / "data/properties/poi_properties.poc.v2.json"))
    enrich.add_argument("--seed", type=int)
    sync_pois = psub.add_parser("sync")
    sync_pois.add_argument("--manifest", required=True); sync_pois.add_argument("--dataset-key", default="hcmc-osm"); sync_pois.add_argument("--no-activate", action="store_true")
    status_pois = psub.add_parser("status"); status_pois.add_argument("--dataset-key", default="hcmc-osm")
    activate_pois = psub.add_parser("activate"); activate_pois.add_argument("generation_id", type=int); activate_pois.add_argument("--dataset-key", default="hcmc-osm")
    promote_pois = psub.add_parser("promote", help="Atomically activate a verified E5/POI pair")
    promote_pois.add_argument("--embedding-generation-id", type=int, required=True)
    promote_pois.add_argument("--poi-generation-id", type=int, required=True)
    promote_pois.add_argument("--config", default=str(DEFAULT_SERVING_CONFIG))
    promote_pois.add_argument("--taxonomy", default=str(DEFAULT_POI_TAXONOMY))
    promote_pois.add_argument("--dataset-key", default="hcmc-osm")
    promote_pois.add_argument("--rollback-record", required=True)
    rollback_pois = psub.add_parser("rollback", help="Restore both pointers from a paired promotion record")
    rollback_pois.add_argument("--record", required=True)

    search = sub.add_parser("search")
    search.add_argument("--taxonomy", default=str(DEFAULT_POI_TAXONOMY))
    search.add_argument("--config", default=str(DEFAULT_SERVING_CONFIG)); search.add_argument("--model", default="e5-base-multilingual"); search.add_argument("--query", required=True); search.add_argument("--category-top-k", type=int, default=5); search.add_argument("--poi-top-k", type=int, default=20); search.add_argument("--language", choices=("vi", "en"), default="vi"); search.add_argument("--generation-id", type=int); search.add_argument("--poi-generation-id", type=int); search.add_argument("--selected-id"); search.add_argument("--allowed-option-id", action="append", default=[]); search.add_argument("--min-score", type=float); search.add_argument("--min-margin", type=float); search.add_argument("--lat", type=float); search.add_argument("--lon", type=float); search.add_argument("--radius-km", type=float); search.add_argument("--local-files-only", action="store_true")
    search.add_argument("--verified-only", action="store_true", help="Do not treat mocked properties as evidence for search filters")

    db = sub.add_parser("db")
    dbsub = db.add_subparsers(dest="db_command", required=True)
    migrate = dbsub.add_parser("migrate"); migrate.add_argument("--migrations", default=str(DEFAULT_MIGRATIONS))
    dbstatus = dbsub.add_parser("status"); dbstatus.add_argument("--migrations", default=str(DEFAULT_MIGRATIONS))

    embeddings = sub.add_parser("embeddings")
    esub = embeddings.add_subparsers(dest="embeddings_command", required=True)
    sync = esub.add_parser("sync"); _common(sync, taxonomy_path=DEFAULT_POI_TAXONOMY); sync.add_argument("--config", default=str(DEFAULT_SERVING_CONFIG)); sync.add_argument("--local-files-only", action="store_true"); sync.add_argument("--no-activate", action="store_true")
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
        if args.command == "pois":
            if args.pois_command == "fetch":
                import shutil
                import urllib.request

                output = Path(args.output).resolve()
                if output.exists():
                    raise ValueError(f"refusing to overwrite existing OSM source file: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                if str(args.url).startswith(("http://", "https://")):
                    urllib.request.urlretrieve(args.url, output)
                else:
                    shutil.copyfile(args.url, output)
                print(json.dumps({"status": "ok", "output": str(output), "url": args.url, "sha256": file_sha256(output), "bytes": output.stat().st_size}, ensure_ascii=False, indent=2))
                return 0
            if args.pois_command == "validate":
                taxonomy = Taxonomy.load(args.taxonomy)
                result = validate_catalogs(taxonomy, args.mapping, args.properties, require_complete=True)
                print(json.dumps({"status": "ok", "coverage": result["coverage"], "mapping_rules": len(result["mapping"]["rules"]), "properties": len(result["properties"])}, ensure_ascii=False, indent=2))
                return 0
            if args.pois_command == "build":
                taxonomy = Taxonomy.load(args.taxonomy)
                if bool(args.pbf) == bool(args.elements_jsonl):
                    raise ValueError("pois build requires exactly one of --pbf or --elements-jsonl")
                if args.elements_jsonl:
                    def elements():
                        with Path(args.elements_jsonl).open(encoding="utf-8") as handle:
                            for line in handle:
                                if line.strip():
                                    yield json.loads(line)
                else:
                    elements = lambda: iter_pbf_elements(args.pbf)
                manifest = build_artifact(
                    elements(),
                    taxonomy=taxonomy,
                    mapping_path=args.mapping,
                    property_path=args.properties,
                    boundary_path=args.boundary,
                    output_dir=args.output,
                    seed=args.seed,
                    source_manifest_path=args.source_manifest,
                    source_pbf_path=args.pbf,
                )
                print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
                return 0
            if args.pois_command == "enrich":
                taxonomy = Taxonomy.load(args.taxonomy)
                manifest = enrich_artifact(
                    args.artifact,
                    output_dir=args.output,
                    taxonomy=taxonomy,
                    mapping_path=args.mapping,
                    property_path=args.properties,
                    seed=args.seed,
                )
                print(json.dumps({
                    "artifact_id": manifest["artifact_id"],
                    "poi_count": manifest["poi_count"],
                    "minimum_properties": min(
                        (row["minimum_known"] for row in manifest["property_coverage"].values()),
                        default=0,
                    ),
                    "property_taxonomy_version": manifest["property_taxonomy_version"],
                    "output": str(Path(args.output).resolve()),
                }, ensure_ascii=False, indent=2))
                return 0
            if args.pois_command == "activate":
                raise RuntimeError("unilateral POI activation is disabled; use pois promote for a verified pair")
            if args.pois_command == "promote":
                provenance = _promotion_provenance(args.config, args.taxonomy, args.dataset_key)
            database = Database(statement_timeout_ms=120000)
            database.open()
            try:
                if args.pois_command == "sync":
                    payload = sync_poi_artifact(
                        database, args.manifest, dataset_key=args.dataset_key, activate=False,
                    )
                elif args.pois_command == "status":
                    payload = poi_status(database, dataset_key=args.dataset_key)
                elif args.pois_command == "promote":
                    payload = promote_generation_pair(
                        database, args.embedding_generation_id, args.poi_generation_id,
                        **provenance, dataset_key=args.dataset_key,
                        rollback_record_path=args.rollback_record,
                    )
                elif args.pois_command == "rollback":
                    payload = rollback_generation_pair(database, args.record)
                else:
                    raise ValueError(f"unsupported POI command: {args.pois_command}")
                print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                return 0
            finally:
                database.close()
        if args.command == "search":
            runtime = SearchRuntime.from_config(
                config_path=args.config,
                taxonomy_path=args.taxonomy,
                model_key=args.model,
                local_files_only=args.local_files_only,
                generation_id=args.generation_id,
                poi_generation_id=args.poi_generation_id,
            )
            try:
                if (args.lat is None) != (args.lon is None):
                    raise ValueError("--lat and --lon must be provided together")
                origin = (args.lat, args.lon) if args.lat is not None else None
                if args.selected_id:
                    payload = runtime.resolve_selection_and_search(
                        query=args.query,
                        selected_id=args.selected_id,
                        allowed_option_ids=args.allowed_option_id,
                        poi_top_k=args.poi_top_k,
                        language=args.language,
                        origin=origin,
                        radius_km=args.radius_km,
                        verified_only=args.verified_only,
                    )
                else:
                    overrides = {}
                    if args.min_score is not None:
                        overrides["accept_min_score"] = args.min_score
                    if args.min_margin is not None:
                        overrides["accept_min_margin"] = args.min_margin
                    payload = runtime.search(
                        args.query,
                        category_top_k=args.category_top_k,
                        poi_top_k=args.poi_top_k,
                        language=args.language,
                        overrides=overrides,
                        origin=origin,
                        radius_km=args.radius_km,
                        verified_only=args.verified_only,
                    )
                print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                return 0
            finally:
                runtime.close()
        if args.command == "db":
            database = Database(statement_timeout_ms=120000); database.open()
            try:
                if args.db_command == "migrate":
                    payload = {"applied": apply_migrations(database, args.migrations), "database": database.check()}
                else:
                    migrations = migration_status(database, args.migrations)
                    payload = {"database": database.check(), "migrations": migrations}
                print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                return 2 if args.db_command == "status" and any(not item["checksum_matches"] for item in migrations) else 0
            finally:
                database.close()
        if args.command == "embeddings":
            if args.embeddings_command in ("activate", "cleanup"):
                raise RuntimeError(
                    "unpaired E5 activate/cleanup is disabled on unified DB; use pois promote/rollback"
                )
            database = Database(statement_timeout_ms=120000); database.open()
            try:
                if args.embeddings_command == "sync":
                    payload = sync_embeddings(
                        database, config_path=args.config, taxonomy_path=args.taxonomy,
                        local_files_only=args.local_files_only, activate=False,
                    )
                    payload = payload.__dict__
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
