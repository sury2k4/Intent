from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import Database
from .embedding_sync import embedding_status, sync_embeddings
from .migrations import apply_migrations, migration_status
from .property_embedding_sync import (
    BASE_MIGRATION_VERSIONS, PROPERTY_MIGRATIONS,
    property_embedding_status, stage_property_embeddings,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
DEFAULT_CONFIG = ROOT / "configs/benchmark.poi.pg.v3.yaml"
DEFAULT_PROPERTY_SEMANTIC = ROOT / "data/properties/property_semantic_profiles.poc.v2.json"
DEFAULT_MIGRATIONS = ROOT / "migrations"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-intent", description="Category/property vector generation administration")
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="Inspect checksum-verified schema migration history")
    dbsub = db.add_subparsers(dest="db_command", required=True)
    dbsub.add_parser("status").add_argument("--migrations", default=str(DEFAULT_MIGRATIONS))

    embeddings = sub.add_parser("embeddings", help="Stage and inspect category E5 generations")
    esub = embeddings.add_subparsers(dest="embeddings_command", required=True)
    esub.add_parser("status")
    stage = esub.add_parser("sync", help="Stage category vectors without activation")
    stage.add_argument("--config", default=str(DEFAULT_CONFIG))
    stage.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    stage.add_argument("--local-files-only", action="store_true")
    promote = esub.add_parser("promote", help="Verify and activate a ready category generation")
    promote.add_argument("generation_id", type=int)
    promote.add_argument("--config", default=str(DEFAULT_CONFIG))
    promote.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    promote.add_argument("--rollback-record", required=True)
    rollback = esub.add_parser("rollback", help="Restore a verified previous category generation")
    rollback.add_argument("--record", required=True)

    properties = sub.add_parser("properties", help="Stage and inspect separate property E5 generations")
    psub = properties.add_subparsers(dest="properties_command", required=True)
    psub.add_parser("migrate", help="Apply only the scoped property migration after root 001-007")
    psub.add_parser("status")
    sync = psub.add_parser("sync", help="Stage property vectors without activation")
    sync.add_argument("--config", default=str(DEFAULT_CONFIG))
    sync.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    sync.add_argument("--semantic", default=str(DEFAULT_PROPERTY_SEMANTIC))
    sync.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        database = Database(statement_timeout_ms=120000)
        database.open()
        try:
            if args.command == "db":
                migrations = migration_status(database, args.migrations)
                payload = {"database": database.check(), "migrations": migrations}
                if any(not item["checksum_matches"] for item in migrations):
                    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                    return 2
            elif args.command == "embeddings":
                if args.embeddings_command == "sync":
                    payload = sync_embeddings(
                        database, config_path=args.config, taxonomy_path=args.taxonomy,
                        local_files_only=args.local_files_only, activate=False,
                    ).__dict__
                elif args.embeddings_command == "promote":
                    from .category_promotion import promote_category_generation

                    payload = promote_category_generation(
                        database, args.generation_id, config_path=args.config,
                        taxonomy_path=args.taxonomy, rollback_record_path=args.rollback_record,
                    )
                elif args.embeddings_command == "rollback":
                    from .category_promotion import rollback_category_generation

                    payload = rollback_category_generation(database, args.record)
                else:
                    payload = embedding_status(database)
            else:
                if args.properties_command == "migrate":
                    roots = migration_status(database, DEFAULT_MIGRATIONS)
                    by_version = {row["version"]: row for row in roots}
                    if any(
                        version not in by_version or not by_version[version]["applied"]
                        or not by_version[version]["checksum_matches"]
                        for version in BASE_MIGRATION_VERSIONS
                    ):
                        raise RuntimeError("property migration requires verified root migrations 001-007")
                    payload = {"applied": apply_migrations(database, PROPERTY_MIGRATIONS)}
                elif args.properties_command == "sync":
                    payload = stage_property_embeddings(
                        database, config_path=args.config, taxonomy_path=args.taxonomy,
                        semantic_path=args.semantic, local_files_only=True, batch_size=args.batch_size,
                    ).__dict__
                else:
                    payload = property_embedding_status(database)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            return 0
        finally:
            database.close()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
