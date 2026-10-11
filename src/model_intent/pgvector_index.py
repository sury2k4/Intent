from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from typing import Any

import numpy as np

from .db import Database
from .normalization import text_variants
from .schemas import Prediction, RetrievalHit
from .search_timing import SearchTiming
from .taxonomy import Taxonomy


class PgVectorIndex:
    """Exact cosine retrieval over the v3 single-row text+embedding table."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        database: Database,
        *,
        model_key: str = "e5-base-multilingual",
        mode: str = "pg_exact",
        ef_search: int = 100,
        overfetch: int = 200,
        generation_id: int | None = None,
        dimension: int = 768,
    ) -> None:
        if mode != "pg_exact":
            raise ValueError("category profile v3 serving supports pg_exact only")
        self.taxonomy = taxonomy
        self.database = database
        self.model_key = model_key
        self.mode = mode
        self.ef_search = int(ef_search)
        self.overfetch = int(overfetch)
        self.generation_id = generation_id
        self.dimension = int(dimension)
        if self.model_key == "bge-m3":
            self.active_table = "active_category_profile_generation_bge_m3"
            self.table_name = "category_search_profiles_bge_m3"
            self.artifact_kind = "category_profile_bge_m3_v1"
        else:
            self.active_table = "active_category_profile_generation_v3"
            self.table_name = "category_search_profiles_v3"
            self.artifact_kind = "category_profile_v3"
        self._embedding_lock = threading.RLock()
        self._search_local = threading.local()

    @property
    def last_search_metadata(self) -> dict[str, Any]:
        return getattr(self._search_local, "metadata", {})

    def active_generation(self) -> dict[str, Any]:
        with self.database.connection() as (connection, pool_metrics):
            row = connection.execute(
                f"""
                SELECT g.id, g.expected_count, m.model_id, m.revision, m.config_hash,
                       t.fingerprint, g.renderer_version, g.profile_set_id,
                       g.content_set_hash, m.dimension, ps.source_fingerprint,
                       ps.profile_schema_version, ps.category_count, ps.profile_count,
                       g.artifact_kind
                FROM {self.active_table} a
                JOIN embedding_generations g
                  ON g.id=a.generation_id
                 AND g.state='ready'
                 AND g.artifact_kind='{self.artifact_kind}'
                JOIN embedding_model_configs m ON m.id=g.model_config_id
                JOIN taxonomy_snapshots t ON t.id=g.taxonomy_snapshot_id
                JOIN category_profile_sets ps ON ps.id=g.profile_set_id
                WHERE a.model_key=%s
                """,
                (self.model_key,),
            ).fetchone()
        if not row:
            raise RuntimeError(f"no active ready category profile generation for {self.model_key}")
        return {
            "generation_id": int(row[0]), "candidate_count": int(row[1]),
            "profile_count": int(row[1]), "model_id": row[2], "revision": row[3],
            "config_hash": row[4], "taxonomy_fingerprint": row[5],
            "renderer_version": row[6], "profile_set_id": int(row[7]),
            "content_set_hash": row[8], "dimension": int(row[9]),
            "source_fingerprint": row[10], "profile_schema_version": int(row[11]),
            "indexed_category_count": int(row[12]), "profile_count_from_set": int(row[13]),
            "artifact_kind": row[14], "pool_wait_ms": pool_metrics.wait_ms,
        }

    def generation_metadata(self, generation_id: int) -> dict[str, Any]:
        with self.database.connection() as (connection, pool_metrics):
            row = connection.execute(
                f"""
                SELECT g.id, g.expected_count, m.model_id, m.revision, m.config_hash,
                       t.fingerprint, g.renderer_version, g.profile_set_id,
                       g.content_set_hash, m.dimension, ps.source_fingerprint,
                       ps.profile_schema_version, ps.category_count, ps.profile_count,
                       g.artifact_kind
                FROM embedding_generations g
                JOIN embedding_model_configs m ON m.id=g.model_config_id
                JOIN taxonomy_snapshots t ON t.id=g.taxonomy_snapshot_id
                JOIN category_profile_sets ps ON ps.id=g.profile_set_id
                WHERE g.id=%s AND m.model_key=%s
                  AND g.state='ready' AND g.artifact_kind='{self.artifact_kind}'
                """,
                (int(generation_id), self.model_key),
            ).fetchone()
        if not row:
            raise RuntimeError(f"generation {generation_id} is not a ready category profile generation")
        return {
            "generation_id": int(row[0]), "candidate_count": int(row[1]),
            "profile_count": int(row[1]), "model_id": row[2], "revision": row[3],
            "config_hash": row[4], "taxonomy_fingerprint": row[5],
            "renderer_version": row[6], "profile_set_id": int(row[7]),
            "content_set_hash": row[8], "dimension": int(row[9]),
            "source_fingerprint": row[10], "profile_schema_version": int(row[11]),
            "indexed_category_count": int(row[12]), "profile_count_from_set": int(row[13]),
            "artifact_kind": row[14], "pool_wait_ms": pool_metrics.wait_ms,
        }

    def _search_vector(
        self, vector: np.ndarray, limit: int, timing: SearchTiming | None = None,
    ) -> tuple[list[tuple], dict[str, float]]:
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"query vector must be a finite {self.dimension}-dimensional vector")
        generation_id = self.generation_id
        if generation_id is None:
            generation_id = int(self.active_generation()["generation_id"])
            self.generation_id = generation_id
        sql = f"""
            WITH ranked AS (
                SELECT csp.generation_id,
                       csp.place_type_id,
                       csp.language,
                       csp.search_text,
                       1 - (csp.embedding <=> %s) AS score,
                       row_number() OVER (
                            PARTITION BY csp.place_type_id
                            ORDER BY csp.embedding <=> %s, csp.id
                       ) AS item_rank
                FROM {self.table_name} csp
                JOIN embedding_generations eg
                  ON eg.id=csp.generation_id
                 AND eg.state='ready'
                 AND eg.artifact_kind='{self.artifact_kind}'
                WHERE csp.generation_id=%s
            )
            SELECT generation_id, place_type_id, language, search_text, score
            FROM ranked
            WHERE item_rank=1
            ORDER BY score DESC, place_type_id ASC
            LIMIT %s
        """
        with self.database.connection() as (connection, pool_metrics):
            if timing is not None:
                timing.record("category_pool_wait", pool_metrics.wait_ms)
            sql_started = time.perf_counter()
            with (timing.measure("category_db_execute_fetch") if timing else nullcontext()):
                rows = connection.execute(sql, (vector, vector, generation_id, limit)).fetchall()
        elapsed_ms = (time.perf_counter() - sql_started) * 1000
        return rows, {"pool_wait_ms": pool_metrics.wait_ms, "db_search_ms": elapsed_ms}

    def search(
        self, query: str, embedder, top_k: int = 5, query_id: str = "query",
        *, timing: SearchTiming | None = None,
    ) -> Prediction:
        with (timing.measure("category_query_variants") if timing else nullcontext()):
            query_entries = text_variants(query)
            query_texts = [value for _, value in query_entries]
        encode_started = time.perf_counter()
        with (timing.measure("category_encode_normalize") if timing else nullcontext()):
            with self._embedding_lock:
                query_matrix = np.asarray(embedder.encode_queries(query_texts), dtype=np.float32)
            norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            query_matrix = np.divide(query_matrix, np.maximum(norms, 1e-12), dtype=np.float32)
        encode_ms = (time.perf_counter() - encode_started) * 1000

        by_item: dict[str, RetrievalHit] = {}
        pool_wait_ms = 0.0
        db_search_ms = 0.0
        generation_ids: set[int] = set()
        variant_top_ids: list[str] = []
        for variant_index, ((query_kind, _), vector) in enumerate(zip(query_entries, query_matrix)):
            with (timing.measure("category_db_variant", variant_index=variant_index) if timing else nullcontext()):
                rows, metrics = self._search_vector(vector, max(top_k, 20), timing=timing)
                if rows:
                    variant_top_ids.append(str(rows[0][1]))
                pool_wait_ms += metrics["pool_wait_ms"]
                db_search_ms += metrics["db_search_ms"]
                with (timing.measure("category_hydrate_merge", row_count=len(rows)) if timing else nullcontext()):
                    for row in rows:
                        generation_ids.add(int(row[0]))
                        item_id = str(row[1])
                        item = self.taxonomy.items.get(item_id)
                        if item is None:
                            continue
                        language = str(row[2])
                        hit = RetrievalHit(
                            item_id=item_id,
                            code=item.code,
                            score=float(row[4]),
                            rank=0,
                            language=language,
                            matched_text=str(row[3]),
                            path=tuple(self.taxonomy.path_names(item_id, language)),
                            profile_kind="primary",
                            query_variant_kind=query_kind,
                        )
                        previous = by_item.get(item_id)
                        if previous is None or hit.score > previous.score:
                            by_item[item_id] = hit
        if generation_ids and self.generation_id not in generation_ids:
            raise RuntimeError("search returned vectors outside the runtime-pinned generation")
        with (timing.measure("category_sort_hits", candidate_count=len(by_item)) if timing else nullcontext()):
            hits = sorted(by_item.values(), key=lambda hit: (-hit.score, hit.item_id))[:top_k]
            ranked = tuple(
                RetrievalHit(
                    hit.item_id, hit.code, hit.score, rank + 1, hit.language,
                    hit.matched_text, hit.path, hit.profile_kind, hit.query_variant_kind,
                )
                for rank, hit in enumerate(hits)
            )
        top1 = ranked[0].score if ranked else None
        margin = top1 - ranked[1].score if len(ranked) > 1 else None
        self._search_local.metadata = {
            "backend": "pgvector", "mode": self.mode,
            "generation_id": next(iter(generation_ids), None),
            "query_encode_ms": encode_ms, "pool_wait_ms": pool_wait_ms,
            "db_search_ms": db_search_ms, "query_variant_count": len(query_entries),
            "returned_count": len(ranked), "dimension": self.dimension,
        }
        return Prediction(
            query_id, query, ranked, top1, margin, self.model_key,
            tuple(query_texts), tuple(variant_top_ids),
        )
