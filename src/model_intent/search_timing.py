from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class SearchTiming:
    """Request-local wall-clock spans; nested durations must not be added together."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self._parents: list[int] = []

    @contextmanager
    def measure(self, name: str, **counts: int) -> Iterator[None]:
        parent = self._parents[-1] if self._parents else None
        row: dict[str, Any] = {"name": name, "parent": parent, "status": "ok", "duration_ms": None}
        row.update(counts)
        index = len(self.stages)
        self.stages.append(row)
        self._parents.append(index)
        started = time.perf_counter()
        try:
            yield
        except BaseException:
            row["status"] = "error"
            raise
        finally:
            row["duration_ms"] = (time.perf_counter() - started) * 1000
            self._parents.pop()

    def record(self, name: str, duration_ms: float, **counts: int) -> None:
        parent = self._parents[-1] if self._parents else None
        self.stages.append({
            "name": name, "parent": parent, "status": "ok", "duration_ms": duration_ms, **counts,
        })

    def skip(self, *names: str) -> None:
        for name in names:
            self.stages.append({"name": name, "parent": None, "status": "skipped", "duration_ms": None})

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "path": self.path,
            "total_ms": (time.perf_counter() - self.started) * 1000,
            "stages": [dict(row) for row in self.stages],
        }
