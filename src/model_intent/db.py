from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


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
        try:
            from pgvector.psycopg import register_vector
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError("Install the optional 'db' dependencies") from exc
        self.database_url = database_url or os.getenv("DATABASE_URL") or "postgresql://model_intent:model_intent_dev@localhost:5432/model_intent"
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
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
        self.pool.open(wait=wait)
        if wait:
            self.pool.wait(timeout=timeout)

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
