from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .db import Database


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str


def discover_migrations(directory: str | Path) -> list[Migration]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"migration directory does not exist: {root}")
    migrations = []
    for path in sorted(root.glob("*.sql")):
        raw = path.read_bytes()
        migrations.append(Migration(path.stem, path, hashlib.sha256(raw).hexdigest()))
    if not migrations:
        raise ValueError(f"no SQL migrations found in: {root}")
    return migrations


def ensure_ledger(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            checksum CHAR(64) NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def migration_status(database: Database, directory: str | Path) -> list[dict[str, object]]:
    migrations = discover_migrations(directory)
    with database.connection() as (connection, _):
        ledger = connection.execute("SELECT to_regclass('schema_migrations')").fetchone()[0]
        applied = dict(connection.execute("SELECT version, checksum FROM schema_migrations").fetchall()) if ledger else {}
    return [
        {
            "version": migration.version,
            "path": str(migration.path),
            "checksum": migration.checksum,
            "applied": migration.version in applied,
            "checksum_matches": applied.get(migration.version, migration.checksum) == migration.checksum,
        }
        for migration in migrations
    ]


def apply_migrations(database: Database, directory: str | Path) -> list[str]:
    migrations = discover_migrations(directory)
    applied_now: list[str] = []
    with database.connection() as (connection, _):
        ensure_ledger(connection)
        connection.commit()
        for migration in migrations:
            existing = connection.execute(
                "SELECT checksum FROM schema_migrations WHERE version = %s", (migration.version,)
            ).fetchone()
            if existing:
                if existing[0] != migration.checksum:
                    raise RuntimeError(f"migration checksum mismatch: {migration.version}")
                continue
            sql = migration.path.read_text(encoding="utf-8")
            try:
                with connection.transaction():
                    connection.execute(sql)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, checksum) VALUES (%s, %s)",
                        (migration.version, migration.checksum),
                    )
            except Exception:
                connection.rollback()
                raise
            applied_now.append(migration.version)
    if applied_now:
        database.reopen()
    return applied_now
