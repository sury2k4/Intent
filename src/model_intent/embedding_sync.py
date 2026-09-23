from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import _configure_runtime, load_yaml, make_embedder, profiles_from_config
from .db import Database
from .rendering import (
    PROFILE_RENDERER_VERSION,
    PROFILE_SCHEMA_VERSION,
    load_enrichment,
    render_category_profiles,
    validate_enrichment,
)
from .schemas import CategorySearchProfile, ModelProfile
from .taxonomy import Taxonomy

MODEL_KEY = "e5-base-multilingual"
MODEL_DIMENSION = 768
ARTIFACT_KIND = "category_profile_v3"
ADVISORY_LOCK_NAME = "model-intent:e5-base:category-profile-sync-v3"


@dataclass(frozen=True)
class SyncResult:
    generation_id: int
    state: str
    expected_count: int
    embedded_count: int
    already_ready: bool
    active: bool
    model_config_hash: str
    taxonomy_fingerprint: str
    content_set_hash: str
    elapsed_seconds: float
    profile_set_id: int | None = None
    profile_count: int = 0
    category_count: int = 0
    source_fingerprint: str = ""
    artifact_kind: str = ARTIFACT_KIND


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def profile_identity(profile: CategorySearchProfile) -> dict[str, Any]:
    return {
        "item_id": profile.item_id,
        "language": profile.language,
        "text": profile.text,
        "renderer_version": PROFILE_RENDERER_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
    }


def profile_hash(profile: CategorySearchProfile) -> str:
    return stable_hash(profile_identity(profile))


def profile_content_hash(profile: CategorySearchProfile) -> str:
    return stable_hash({
        "item_id": profile.item_id,
        "language": profile.language,
        "text": profile.text,
    })


def variant_identity(profile: CategorySearchProfile) -> dict[str, Any]:
    return profile_identity(profile)


def variant_hash(profile: CategorySearchProfile) -> str:
    return profile_hash(profile)


def content_hash(profile: CategorySearchProfile) -> str:
    return profile_content_hash(profile)


def _source_without_ignored_fields(enrichment: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item_id: {
            key: value
            for key, value in profile.items()
            if (key.endswith("_vi") or key.endswith("_en"))
            and (key.startswith("description_") or key.startswith("aliases_"))
        }
        for item_id, profile in sorted(enrichment.items())
    }


def profile_source_fingerprint(enrichment: dict[str, dict[str, Any]]) -> str:
    return stable_hash({
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "items": _source_without_ignored_fields(enrichment),
    })


def model_profile(config: dict[str, Any]) -> ModelProfile:
    profile = next(
        (item for item in profiles_from_config(config, include_stretch=True) if item.key == MODEL_KEY),
        None,
    )
    if profile is None:
        raise ValueError(f"config does not contain {MODEL_KEY}")
    if not profile.revision:
        raise ValueError("E5-base revision must be pinned before embedding sync")
    expected = profile.expected_dimension or MODEL_DIMENSION
    if expected != MODEL_DIMENSION:
        raise ValueError(f"pgvector profile storage is fixed to {MODEL_DIMENSION} dimensions")
    return profile


def semantic_model_metadata(profile: ModelProfile) -> dict[str, Any]:
    return {
        "adapter": "sentence-transformer-v1",
        "model_key": profile.key,
        "model_id": profile.model_id,
        "revision": profile.revision,
        "query_prefix": profile.query_prefix,
        "document_prefix": profile.document_prefix,
        "normalize_embeddings": True,
        "dimension": profile.expected_dimension or MODEL_DIMENSION,
        "max_seq_length": profile.max_seq_length,
    }


def embedding_metadata(profile: ModelProfile) -> dict[str, Any]:
    return {**semantic_model_metadata(profile), "batch_size": profile.batch_size, "device": "cpu"}


def _resolve_path(project_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _validate_token_lengths(embedder: Any, texts: list[str], max_seq_length: int) -> None:
    tokenizer = getattr(getattr(embedder, "model", None), "tokenizer", None)
    if tokenizer is None or not texts:
        return
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    lengths = [len(ids) for ids in encoded.get("input_ids", [])]
    if lengths and max(lengths) > max_seq_length:
        raise ValueError(
            f"category profile has {max(lengths)} tokens, exceeding max_seq_length={max_seq_length}; "
            "increase the configured limit or shorten the profile"
        )


def sync_embeddings(
    database: Database,
    *,
    config_path: str | Path,
    taxonomy_path: str | Path,
    local_files_only: bool = False,
    batch_size: int | None = None,
) -> SyncResult:
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_yaml(config_path)
    _configure_runtime(config, local_files_only)
    taxonomy = Taxonomy.load(taxonomy_path)
    enrichment = load_enrichment(_resolve_path(project_root, config.get("enrichment_path")))
    validate_enrichment(taxonomy, enrichment, require_rich_fields=True)
    profile = model_profile(config)
    metadata = embedding_metadata(profile)
    semantic_metadata = semantic_model_metadata(profile)
    profiles = render_category_profiles(
        taxonomy,
        tuple(config.get("languages", ["vi", "en"])),
        enrichment,
        require_rich_fields=True,
    )
    expected_categories = {item.id for item in taxonomy.document.items if not taxonomy.has_children(item.id)}
    category_ids = {item.item_id for item in profiles}
    if category_ids != expected_categories:
        raise RuntimeError("profile corpus does not contain exactly one profile set per leaf category")
    if any(not profile.item_id.isdigit() for profile in profiles):
        raise ValueError("v3 place_type_id requires numeric taxonomy IDs")
    profile_rows = [{"profile": item, "content_hash": profile_content_hash(item)} for item in profiles]
    content_set_hash = stable_hash(sorted(
        (row["profile"].item_id, row["profile"].language, row["content_hash"])
        for row in profile_rows
    ))
    source_fingerprint = profile_source_fingerprint(enrichment)
    model_config_hash = stable_hash(semantic_metadata)
    batch_size = int(batch_size or profile.batch_size)

    with database.connection() as (connection, _):
        connection.execute("SELECT pg_advisory_lock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
        generation_id: int | None = None
        try:
            with connection.transaction():
                model_id = connection.execute(
                    """
                    INSERT INTO embedding_model_configs(
                        model_key, model_id, revision, adapter_version, dimension,
                        query_prefix, document_prefix, normalize_embeddings,
                        max_seq_length, config_hash
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (config_hash) DO UPDATE SET config_hash = EXCLUDED.config_hash
                    RETURNING id
                    """,
                    (
                        MODEL_KEY, metadata["model_id"], metadata["revision"], metadata["adapter"],
                        int(metadata["dimension"]), metadata["query_prefix"], metadata["document_prefix"],
                        bool(metadata["normalize_embeddings"]), int(metadata["max_seq_length"]), model_config_hash,
                    ),
                ).fetchone()[0]
                snapshot_id = connection.execute(
                    """
                    INSERT INTO taxonomy_snapshots(taxonomy_version, fingerprint, schema_version, source)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (fingerprint) DO UPDATE SET fingerprint = EXCLUDED.fingerprint
                    RETURNING id
                    """,
                    (
                        taxonomy.document.taxonomy_version, taxonomy.fingerprint(),
                        taxonomy.document.schema_version, taxonomy.document.source,
                    ),
                ).fetchone()[0]
                profile_set_id = connection.execute(
                    """
                    INSERT INTO category_profile_sets(
                        taxonomy_snapshot_id, profile_schema_version, source_fingerprint,
                        renderer_version, content_set_hash, category_count, profile_count
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (
                        taxonomy_snapshot_id, profile_schema_version, source_fingerprint,
                        renderer_version, content_set_hash
                    ) DO UPDATE SET profile_count = EXCLUDED.profile_count
                    RETURNING id
                    """,
                    (
                        snapshot_id, PROFILE_SCHEMA_VERSION, source_fingerprint,
                        PROFILE_RENDERER_VERSION, content_set_hash, len(category_ids), len(profile_rows),
                    ),
                ).fetchone()[0]
                generation = connection.execute(
                    """
                    INSERT INTO embedding_generations(
                        model_config_id, taxonomy_snapshot_id, renderer_version,
                        content_set_hash, state, expected_count, artifact_kind, profile_set_id
                    ) VALUES (%s,%s,%s,%s,'building',%s,%s,%s)
                    ON CONFLICT (model_config_id, taxonomy_snapshot_id, renderer_version, content_set_hash)
                    DO UPDATE SET expected_count=EXCLUDED.expected_count,
                                  artifact_kind=EXCLUDED.artifact_kind,
                                  profile_set_id=EXCLUDED.profile_set_id
                    RETURNING id, state
                    """,
                    (
                        model_id, snapshot_id, PROFILE_RENDERER_VERSION, content_set_hash,
                        len(profile_rows), ARTIFACT_KIND, profile_set_id,
                    ),
                ).fetchone()
                generation_id, state = int(generation[0]), str(generation[1])

            active_row = connection.execute(
                "SELECT generation_id FROM active_category_profile_generation_v3 WHERE model_key=%s",
                (MODEL_KEY,),
            ).fetchone()
            existing = connection.execute(
                "SELECT count(*) FROM category_search_profiles_v3 WHERE generation_id=%s",
                (generation_id,),
            ).fetchone()[0]
            if state == "ready" and existing == len(profile_rows):
                if not active_row or int(active_row[0]) != generation_id:
                    with connection.transaction():
                        connection.execute(
                            """
                            INSERT INTO active_category_profile_generation_v3(model_key, generation_id)
                            VALUES (%s,%s)
                            ON CONFLICT (model_key) DO UPDATE
                            SET generation_id=EXCLUDED.generation_id, activated_at=now()
                            """,
                            (MODEL_KEY, generation_id),
                        )
                return SyncResult(
                    generation_id, "ready", len(profile_rows), 0, True, True,
                    model_config_hash, taxonomy.fingerprint(), content_set_hash,
                    time.perf_counter() - started, int(profile_set_id), len(profile_rows),
                    len(category_ids), source_fingerprint, ARTIFACT_KIND,
                )

            with connection.transaction():
                connection.execute(
                    "UPDATE embedding_generations SET state='building', error=NULL WHERE id=%s",
                    (generation_id,),
                )
            existing_rows = {
                (int(row[0]), str(row[1]), str(row[2])): int(row[3])
                for row in connection.execute(
                    """
                    SELECT place_type_id, language, content_hash, id
                    FROM category_search_profiles_v3
                    WHERE generation_id=%s
                    """,
                    (generation_id,),
                ).fetchall()
            }
            missing = [
                row for row in profile_rows
                if (int(row["profile"].item_id), row["profile"].language, row["content_hash"])
                not in existing_rows
            ]
            embedded_count = 0
            embedder = make_embedder(profile, local_files_only) if missing else None
            if embedder is not None:
                actual_dimension = int(getattr(embedder, "dimension", 0))
                if actual_dimension != MODEL_DIMENSION:
                    raise ValueError(f"loaded embedder dimension {actual_dimension} != {MODEL_DIMENSION}")
                _validate_token_lengths(embedder, [row["profile"].text for row in missing], profile.max_seq_length)
            for offset in range(0, len(missing), batch_size):
                batch = missing[offset:offset + batch_size]
                vectors = np.asarray(
                    embedder.encode_documents([row["profile"].text for row in batch]), dtype=np.float32
                )
                if vectors.shape != (len(batch), MODEL_DIMENSION) or not np.isfinite(vectors).all():
                    raise ValueError("embedding batch failed shape/finite validation")
                if np.any(np.linalg.norm(vectors, axis=1) <= 0):
                    raise ValueError("embedding batch contains a zero vector")
                rows = [
                    (
                        generation_id,
                        int(row["profile"].item_id),
                        row["profile"].language,
                        row["profile"].text,
                        vector,
                        row["content_hash"],
                    )
                    for row, vector in zip(batch, vectors)
                ]
                with connection.transaction():
                    with connection.cursor() as cursor:
                        cursor.executemany(
                            """
                            INSERT INTO category_search_profiles_v3(
                                generation_id, place_type_id, language, search_text,
                                embedding, content_hash
                            ) VALUES (%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (generation_id, place_type_id, language) DO UPDATE
                            SET search_text=EXCLUDED.search_text,
                                embedding=EXCLUDED.embedding,
                                content_hash=EXCLUDED.content_hash,
                                created_at=now()
                            """,
                            rows,
                        )
                    embedded_count += len(batch)
                    connection.execute(
                        "UPDATE embedding_generations SET completed_count=%s WHERE id=%s",
                        (len(existing_rows) + embedded_count, generation_id),
                    )

            final_count = connection.execute(
                "SELECT count(*) FROM category_search_profiles_v3 WHERE generation_id=%s",
                (generation_id,),
            ).fetchone()[0]
            matching_count = connection.execute(
                """
                SELECT count(*) FROM category_search_profiles_v3
                WHERE generation_id=%s AND content_hash IS NOT NULL
                """,
                (generation_id,),
            ).fetchone()[0]
            if final_count != len(profile_rows) or matching_count != final_count:
                raise RuntimeError(
                    f"profile generation integrity mismatch: rows={final_count}, "
                    f"hashed={matching_count}, expected={len(profile_rows)}"
                )
            with connection.transaction():
                connection.execute(
                    """
                    UPDATE embedding_generations
                    SET state='ready', completed_count=%s, finished_at=now(), error=NULL
                    WHERE id=%s
                    """,
                    (final_count, generation_id),
                )
                connection.execute(
                    """
                    INSERT INTO active_category_profile_generation_v3(model_key, generation_id)
                    VALUES (%s,%s)
                    ON CONFLICT (model_key) DO UPDATE
                    SET generation_id=EXCLUDED.generation_id, activated_at=now()
                    """,
                    (MODEL_KEY, generation_id),
                )
            return SyncResult(
                generation_id, "ready", len(profile_rows), embedded_count, False, True,
                model_config_hash, taxonomy.fingerprint(), content_set_hash,
                time.perf_counter() - started, int(profile_set_id), len(profile_rows),
                len(category_ids), source_fingerprint, ARTIFACT_KIND,
            )
        except Exception as exc:
            if generation_id is not None:
                try:
                    with connection.transaction():
                        connection.execute(
                            "UPDATE embedding_generations SET state='failed', error=%s WHERE id=%s",
                            (f"{type(exc).__name__}: {exc}", generation_id),
                        )
                except Exception:
                    connection.rollback()
            raise
        finally:
            connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
            connection.commit()


def _generation_payload(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "generation_id": int(row[0]), "state": row[1], "expected_count": int(row[2]),
        "completed_count": int(row[3]), "started_at": row[4], "finished_at": row[5],
        "error": row[6], "artifact_kind": row[7], "profile_set_id": row[8],
    }


def embedding_status(database: Database) -> dict[str, Any]:
    with database.connection() as (connection, _):
        active = connection.execute(
            """
            SELECT g.id, g.state, g.expected_count, g.completed_count, g.started_at,
                   g.finished_at, g.error, g.artifact_kind, g.profile_set_id,
                   m.model_id, m.revision, m.config_hash, t.fingerprint, g.renderer_version,
                   ps.source_fingerprint, ps.category_count, ps.profile_count, m.dimension
            FROM active_category_profile_generation_v3 a
            JOIN embedding_generations g ON g.id=a.generation_id
            JOIN embedding_model_configs m ON m.id=g.model_config_id
            JOIN taxonomy_snapshots t ON t.id=g.taxonomy_snapshot_id
            LEFT JOIN category_profile_sets ps ON ps.id=g.profile_set_id
            WHERE a.model_key=%s
            """,
            (MODEL_KEY,),
        ).fetchone()
        generations = connection.execute(
            """
            SELECT id, state, expected_count, completed_count, started_at, finished_at,
                   error, artifact_kind, profile_set_id
            FROM embedding_generations WHERE artifact_kind=%s ORDER BY id DESC
            """,
            (ARTIFACT_KIND,),
        ).fetchall()
    return {
        "model_key": MODEL_KEY,
        "artifact_kind": ARTIFACT_KIND,
        "dimension": MODEL_DIMENSION,
        "active": None if active is None else {
            "generation_id": active[0], "state": active[1], "expected_count": active[2],
            "completed_count": active[3], "started_at": active[4], "finished_at": active[5],
            "error": active[6], "artifact_kind": active[7], "profile_set_id": active[8],
            "model_id": active[9], "revision": active[10], "config_hash": active[11],
            "taxonomy_fingerprint": active[12], "renderer_version": active[13],
            "source_fingerprint": active[14], "category_count": active[15],
            "profile_count": active[16], "dimension": active[17],
        },
        "generations": [_generation_payload(row) for row in generations],
    }


def cleanup_generations(database: Database, keep: int = 3) -> list[int]:
    if keep < 1:
        raise ValueError("keep must be at least one")
    with database.connection() as (connection, _):
        active_ids = {
            int(row[0]) for row in connection.execute(
                """
                SELECT generation_id FROM active_category_profile_generation_v3
                UNION SELECT generation_id FROM active_category_profile_generation_v2
                UNION SELECT generation_id FROM active_category_profile_generation
                UNION SELECT generation_id FROM active_embedding_generation
                """
            ).fetchall()
        }
        rows = connection.execute(
            """
            SELECT g.id FROM embedding_generations g
            JOIN embedding_model_configs m ON m.id=g.model_config_id
            WHERE m.model_key=%s AND g.state IN ('ready','failed','retired')
            ORDER BY g.id DESC
            """,
            (MODEL_KEY,),
        ).fetchall()
        retained = {int(row[0]) for row in rows[:keep]} | active_ids
        remove = [int(row[0]) for row in rows if int(row[0]) not in retained]
        if remove:
            with connection.transaction():
                connection.execute("DELETE FROM embedding_generations WHERE id=ANY(%s)", (remove,))
        return remove


def activate_generation(database: Database, generation_id: int) -> None:
    with database.connection() as (connection, _):
        row = connection.execute(
            """
            SELECT g.state, g.artifact_kind, g.expected_count, g.completed_count,
                   m.model_key, count(csp.id)
            FROM embedding_generations g
            JOIN embedding_model_configs m ON m.id=g.model_config_id
            LEFT JOIN category_search_profiles_v3 csp ON csp.generation_id=g.id
            WHERE g.id=%s GROUP BY g.id, m.model_key
            """,
            (generation_id,),
        ).fetchone()
        if not row or row[0] != "ready" or row[1] != ARTIFACT_KIND or row[4] != MODEL_KEY:
            raise ValueError("generation must be a ready category profile v3 generation")
        if int(row[2]) != int(row[3]) or int(row[2]) != int(row[5]):
            raise ValueError("generation is incomplete and cannot be activated")
        with connection.transaction():
            connection.execute(
                """
                INSERT INTO active_category_profile_generation_v3(model_key, generation_id)
                VALUES (%s,%s)
                ON CONFLICT (model_key) DO UPDATE
                SET generation_id=EXCLUDED.generation_id, activated_at=now()
                """,
                (MODEL_KEY, generation_id),
            )
