from __future__ import annotations

import os
import time
from urllib.parse import urlparse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, Any


UNIFIED_DATABASE_HOST = "127.0.0.1"
UNIFIED_DATABASE_PORT = 5433
UNIFIED_DATABASE_NAME = "model_intent_unified"


def require_database_url(database_url: str | None = None) -> str:
    """Allow only the explicit local unified database, never the legacy instance."""
    url = database_url if database_url is not None else os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("unified Postgres serving requires explicit DATABASE_URL; refusing the legacy default database")
    try:
        parsed = urlparse(url)
        valid = (
            parsed.scheme in ("postgres", "postgresql")
            and parsed.hostname == UNIFIED_DATABASE_HOST
            and parsed.port == UNIFIED_DATABASE_PORT
            and parsed.path == f"/{UNIFIED_DATABASE_NAME}"
            and not parsed.query
            and not parsed.fragment
            and not parsed.params
        )
    except ValueError:
        valid = False
    if not valid:
        raise RuntimeError("DATABASE_URL must target the isolated unified database at 127.0.0.1:5433/model_intent_unified")
    return url


def validate_database_target(config: Mapping[str, Any], database_url: str | None = None) -> str | None:
    """Reject an unsafe target before loading a database-backed runtime."""
    poi_config = config.get("poi_search") or {}
    if (
        config.get("require_explicit_database_url")
        or config.get("retrieval_backend") == "pgvector"
        or poi_config.get("backend") == "postgres"
    ):
        return require_database_url(database_url)
    return database_url if database_url is not None else os.getenv("DATABASE_URL")


@dataclass(frozen=True)
class PoolMetrics:
    wait_ms: float


class Database:
    def __init__(
        self,
        database_url: str | None = None,
        *,
        min_size: int = 1,
        max_size: int = 8,
        timeout: float = 10.0,
        statement_timeout_ms: int = 5000,
    ) -> None:
        self.database_url = require_database_url(database_url)
        try:
            from pgvector.psycopg import register_vector
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError("Install the optional 'db' dependencies") from exc
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.pool_timeout = float(timeout)
        self.statement_timeout_ms = int(statement_timeout_ms)

        def configure(connection) -> None:
            has_vector = connection.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector')"
            ).fetchone()[0]
            if has_vector:
                register_vector(connection)
            connection.execute("SET application_name = 'model-intent'")
            connection.execute(f"SET statement_timeout = {self.statement_timeout_ms}")
            connection.commit()

        self.pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=self.min_size,
            max_size=self.max_size,
            timeout=self.pool_timeout,
            configure=configure,
            kwargs={"autocommit": True},
            open=False,
            name="model-intent-pool",
        )

    def open(self, wait: bool = True, timeout: float = 30.0) -> None:
        if not wait:
            raise ValueError("database target verification requires wait=True")
        self.pool.open(wait=True)
        try:
            self.pool.wait(timeout=timeout)
            with self.connection() as (connection, _):
                current_database = connection.execute("SELECT current_database()").fetchone()[0]
            if current_database != UNIFIED_DATABASE_NAME:
                raise RuntimeError("connected database is not model_intent_unified")
        except Exception:
            self.pool.close()
            raise

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def connection(self) -> Iterator[tuple[object, PoolMetrics]]:
        started = time.perf_counter()
        with self.pool.connection() as connection:
            wait_ms = (time.perf_counter() - started) * 1000
            yield connection, PoolMetrics(wait_ms=wait_ms)

    def reopen(self) -> None:
        # psycopg_pool instances cannot be opened again after close(). Build a
        # fresh pool so migrations can re-register extensions created during the
        # migration transaction.
        self.pool.close()
        replacement = type(self)(
            database_url=self.database_url,
            min_size=self.min_size,
            max_size=self.max_size,
            timeout=self.pool_timeout,
            statement_timeout_ms=self.statement_timeout_ms,
        )
        replacement.open()
        self.pool = replacement.pool

    def check(self) -> dict[str, object]:
        with self.connection() as (connection, metrics):
            row = connection.execute(
                "SELECT current_database(), current_user, version(), "
                "(SELECT extversion FROM pg_extension WHERE extname = 'vector')"
            ).fetchone()
        return {
            "database": row[0],
            "user": row[1],
            "server_version": row[2],
            "pgvector_version": row[3],
            "pool_wait_ms": metrics.wait_ms,
        }
