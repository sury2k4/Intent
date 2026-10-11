"""Stage independently versioned property embeddings; never migrate or activate here."""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import property_semantic
from .serving_config import configure_runtime, load_yaml, make_embedder
from .db import Database, require_database_url, validate_database_target
from .embedding_sync import (
    MODEL_DIMENSION,
    MODEL_KEY,
    model_profile,
    semantic_model_metadata,
    stable_hash,
)
from .migrations import discover_migrations
from .poi_properties import load_property_definitions
from .taxonomy import Taxonomy

ADVISORY_LOCK_NAME = "model-intent:e5-base:property-profile-sync-v1"
ROOT_MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
PROPERTY_MIGRATIONS = ROOT_MIGRATIONS / "property"
PROPERTY_MIGRATION_VERSION = "property_001_semantic_profiles"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BASE_MIGRATION_VERSIONS = (
    "001_core", "002_hnsw", "003_category_search_profiles",
    "004_compact_category_profiles", "005_category_search_profiles_v3",
    "006_category_search_profiles_bge_m3", "007_poi_dataset_generations",
)


@dataclass(frozen=True)
class PropertyStageResult:
    generation_id: int
    profile_set_id: int
    state: str
    expected_count: int
    completed_count: int
    embedded_count: int
    already_ready: bool
    active: bool
    model_config_hash: str
    catalog_hash: str
    source_hash: str
    taxonomy_fingerprint: str
    renderer_version: str
    content_set_hash: str
    elapsed_seconds: float


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _verify_schema(connection: Any) -> None:
    """Read the ledger without creating it; 008 is intentionally not required."""
    required = (
        "schema_migrations", "embedding_model_configs", "property_profile_sets",
        "property_embedding_generations", "property_semantic_profiles",
        "active_property_embedding_generation",
    )
    if any(connection.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None for table in required):
        raise RuntimeError(
            "property embedding schema is not installed; explicitly apply migrations/property "
            "after verifying root migrations 001-007 (do not apply pending root 008)"
        )
    root = {migration.version: migration for migration in discover_migrations(ROOT_MIGRATIONS)}
    if any(version not in root for version in BASE_MIGRATION_VERSIONS):
        raise RuntimeError("this checkout is missing a required root migration (001-007)")
    expected = {version: root[version].checksum for version in BASE_MIGRATION_VERSIONS}
    scoped = discover_migrations(PROPERTY_MIGRATIONS)
    if len(scoped) != 1 or scoped[0].version != PROPERTY_MIGRATION_VERSION:
        raise RuntimeError("property migration directory must contain only property_001_semantic_profiles.sql")
    expected[PROPERTY_MIGRATION_VERSION] = scoped[0].checksum
    rows = connection.execute(
        "SELECT version, checksum FROM schema_migrations WHERE version = ANY(%s)",
        (list(expected),),
    ).fetchall()
    applied = {str(version): str(checksum) for version, checksum in rows}
    if applied != expected:
        missing = sorted(expected.keys() - applied.keys())
        mismatched = sorted(name for name in applied if applied[name] != expected[name])
        raise RuntimeError(f"property embedding migration ledger mismatch: missing={missing}, checksums={mismatched}")


def _resolve_config_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _verified_corpus(corpus: Any) -> tuple[Any, ...]:
    rows = tuple(corpus.rows)
    if not rows or len(rows) != int(getattr(corpus, "profile_count", len(rows))):
        raise ValueError("property semantic corpus has an invalid profile count")
    unique_ids = len({row.property_id for row in rows})
    if int(corpus.property_count) < 70 or unique_ids != int(corpus.property_count):
        if int(corpus.property_count) == 70 or unique_ids < 70:
            raise ValueError("property semantic corpus must contain all 70 queryable property IDs")
        raise ValueError("property semantic corpus property count mismatch")
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        identity = (row.property_id, row.language, row.profile_key)
        if (identity in identities or not all(isinstance(value, str) and value.strip() for value in identity)
                or row.language not in ("vi", "en") or not isinstance(row.text, str) or not row.text.strip()):
            raise ValueError("property semantic corpus has a duplicate or invalid profile")
        identities.add(identity)
        _require_hash(row.content_hash, "property profile content hash")
        if row.content_hash != stable_hash({
            "property_id": row.property_id, "language": row.language,
            "profile_key": row.profile_key, "text": row.text,
        }):
            raise ValueError("property semantic row text/content hash mismatch")
    for field in ("catalog_hash", "source_hash", "taxonomy_fingerprint", "content_set_hash"):
        _require_hash(getattr(corpus, field), field)
    expected_set_hash = stable_hash([
        (row.property_id, row.language, row.profile_key, row.content_hash)
        for row in sorted(rows, key=lambda row: (row.property_id, row.language, row.profile_key))
    ])
    if corpus.content_set_hash != expected_set_hash:
        raise ValueError("property semantic content set hash does not match profile rows")
    if not isinstance(corpus.renderer_version, str) or not corpus.renderer_version.strip():
        raise ValueError("property corpus renderer version must be nonempty")
    return rows


def _verify_persisted(connection: Any, generation_id: int, rows: tuple[Any, ...]) -> None:
    stored = connection.execute(
        """
        SELECT property_id, language, profile_key, search_text, content_hash,
               vector_dims(embedding), vector_norm(embedding)
        FROM property_semantic_profiles WHERE generation_id=%s
        """,
        (generation_id,),
    ).fetchall()
    expected = {(row.property_id, row.language, row.profile_key): (row.text, row.content_hash) for row in rows}
    if len(stored) > len(expected):
        raise RuntimeError("property generation contains extra rows")
    seen: set[tuple[str, str, str]] = set()
    for property_id, language, profile_key, text, content_hash, dimension, norm in stored:
        key = (str(property_id), str(language), str(profile_key))
        if (key in seen or expected.get(key) != (str(text), str(content_hash))
                or int(dimension) != MODEL_DIMENSION or not math.isfinite(float(norm))
                or not 0.99 < float(norm) < 1.01):
            raise RuntimeError("property generation rows conflict with immutable text, hash or embedding shape")
        seen.add(key)


def _validate_tokens(embedder: Any, texts: list[str], max_seq_length: int) -> None:
    tokenizer = getattr(getattr(embedder, "model", None), "tokenizer", None)
    if tokenizer is not None and texts:
        # The embedder adds the passage prefix before tokenizing.
        prepared = [embedder.profile.document_prefix + text for text in texts]
        tokens = tokenizer(prepared, add_special_tokens=True, truncation=False, padding=False)
        lengths = [len(ids) for ids in tokens.get("input_ids", [])]
        if len(lengths) != len(texts) or (lengths and max(lengths) > max_seq_length):
            raise ValueError(f"property profile exceeds max_seq_length={max_seq_length}")


def _model_config_id(connection: Any, metadata: dict[str, Any], config_hash: str) -> int:
    fields = (
        MODEL_KEY, metadata["model_id"], metadata["revision"], metadata["adapter"],
        MODEL_DIMENSION, metadata["query_prefix"], metadata["document_prefix"],
        True, metadata["max_seq_length"],
    )
    row = connection.execute(
        """
        INSERT INTO embedding_model_configs(
            model_key, model_id, revision, adapter_version, dimension,
            query_prefix, document_prefix, normalize_embeddings, max_seq_length, config_hash
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (config_hash) DO NOTHING RETURNING id
        """,
        (*fields, config_hash),
    ).fetchone()
    if row is not None:
        return int(row[0])
    existing = connection.execute(
        """
        SELECT id, model_key, model_id, revision, adapter_version, dimension,
               query_prefix, document_prefix, normalize_embeddings, max_seq_length, status
        FROM embedding_model_configs WHERE config_hash=%s
        """,
        (config_hash,),
    ).fetchone()
    if existing is None or tuple(existing[1:10]) != fields or existing[10] != "active":
        raise RuntimeError("existing embedding_model_configs row conflicts with pinned E5 profile")
    return int(existing[0])


def stage_property_embeddings(
    database: Database,
    *,
    config_path: str | Path,
    taxonomy_path: str | Path,
    semantic_path: str | Path,
    local_files_only: bool = True,
    batch_size: int = 16,
) -> PropertyStageResult:
    """Stage a complete E5 property corpus, without applying a migration or switching pointers."""
    started = time.perf_counter()
    require_database_url(database.database_url)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("property embedding batch_size must be positive")
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    validate_database_target(config, database.database_url)
    configured_taxonomy = config.get("taxonomy_path")
    if not configured_taxonomy or _resolve_config_path(config_path, configured_taxonomy) != Path(taxonomy_path).resolve():
        raise ValueError("--taxonomy must match the taxonomy_path pinned by the serving config")
    profile = model_profile(config)
    metadata = semantic_model_metadata(profile)
    if (metadata["model_id"] != "intfloat/multilingual-e5-base" or metadata["query_prefix"] != "query: "
            or metadata["document_prefix"] != "passage: " or not metadata["revision"]
            or metadata["dimension"] != MODEL_DIMENSION):
        raise ValueError("property embeddings require the pinned multilingual E5-base 768D query/passage profile")
    taxonomy = Taxonomy.load(taxonomy_path)
    property_path = (config.get("property_intent") or {}).get("property_path")
    if not property_path:
        raise ValueError("config property_intent.property_path is required for property semantic profiles")
    definitions = load_property_definitions(_resolve_config_path(config_path, property_path))
    corpus = property_semantic.load_property_semantic_corpus(
        semantic_path, definitions, taxonomy, metadata,
    )
    rows = _verified_corpus(corpus)
    if corpus.taxonomy_fingerprint != taxonomy.fingerprint():
        raise ValueError("property corpus taxonomy fingerprint does not match the configured taxonomy")
    config_hash = stable_hash(metadata)
    with database.connection() as (connection, _):
        _verify_schema(connection)
        connection.execute("SELECT pg_advisory_lock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
        generation_id: int | None = None
        building = False
        try:
            with connection.transaction():
                model_id = _model_config_id(connection, metadata, config_hash)
                profile_set = connection.execute(
                    """
                    INSERT INTO property_profile_sets(
                        catalog_hash, source_hash, taxonomy_fingerprint, renderer_version,
                        content_set_hash, property_count, profile_count
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (catalog_hash, source_hash, taxonomy_fingerprint, renderer_version, content_set_hash)
                    DO NOTHING RETURNING id
                    """,
                    (corpus.catalog_hash, corpus.source_hash, corpus.taxonomy_fingerprint,
                     corpus.renderer_version, corpus.content_set_hash, corpus.property_count, len(rows)),
                ).fetchone()
                if profile_set is None:
                    profile_set = connection.execute(
                        """
                        SELECT id, property_count, profile_count FROM property_profile_sets
                        WHERE catalog_hash=%s AND source_hash=%s AND taxonomy_fingerprint=%s
                          AND renderer_version=%s AND content_set_hash=%s
                        """,
                        (corpus.catalog_hash, corpus.source_hash, corpus.taxonomy_fingerprint,
                         corpus.renderer_version, corpus.content_set_hash),
                    ).fetchone()
                    if profile_set is None or tuple(profile_set[1:]) != (corpus.property_count, len(rows)):
                        raise RuntimeError("existing property profile set conflicts with rendered corpus")
                profile_set_id = int(profile_set[0])
                generation = connection.execute(
                    """
                    INSERT INTO property_embedding_generations(
                        profile_set_id, model_config_id, content_set_hash, state, expected_count
                    ) VALUES (%s,%s,%s,'building',%s)
                    ON CONFLICT (profile_set_id, model_config_id) DO NOTHING
                    RETURNING id, state, expected_count, completed_count, content_set_hash
                    """,
                    (profile_set_id, model_id, corpus.content_set_hash, len(rows)),
                ).fetchone()
                if generation is None:
                    generation = connection.execute(
                        """
                        SELECT id, state, expected_count, completed_count, content_set_hash
                        FROM property_embedding_generations WHERE profile_set_id=%s AND model_config_id=%s
                        FOR UPDATE
                        """,
                        (profile_set_id, model_id),
                    ).fetchone()
                if generation is None:
                    raise RuntimeError("property generation disappeared during staging")
                generation_id = int(generation[0])
                state = str(generation[1])
                if int(generation[2]) != len(rows) or str(generation[4]) != corpus.content_set_hash:
                    raise RuntimeError("existing property generation conflicts with rendered corpus")
                if state == "retired":
                    raise RuntimeError("retired property generation is immutable")
            active = connection.execute(
                "SELECT generation_id FROM active_property_embedding_generation WHERE model_key=%s",
                (MODEL_KEY,),
            ).fetchone()
            is_active = active is not None and int(active[0]) == generation_id
            _verify_persisted(connection, generation_id, rows)
            existing = connection.execute(
                "SELECT property_id, language, profile_key FROM property_semantic_profiles WHERE generation_id=%s",
                (generation_id,),
            ).fetchall()
            existing_keys = {(str(property_id), str(language), str(profile_key)) for property_id, language, profile_key in existing}
            if state == "ready":
                if len(existing_keys) != len(rows) or int(generation[3]) != len(rows):
                    raise RuntimeError("ready property generation is incomplete and immutable")
                return PropertyStageResult(
                    generation_id, profile_set_id, "ready", len(rows), len(rows), 0, True, is_active,
                    config_hash, corpus.catalog_hash, corpus.source_hash, corpus.taxonomy_fingerprint,
                    corpus.renderer_version, corpus.content_set_hash, time.perf_counter() - started,
                )
            if is_active:
                raise RuntimeError("refusing to rebuild an active property generation")
            building = True
            with connection.transaction():
                connection.execute(
                    "UPDATE property_embedding_generations SET state='building', finished_at=NULL, error=NULL WHERE id=%s",
                    (generation_id,),
                )
            missing = [row for row in rows if (row.property_id, row.language, row.profile_key) not in existing_keys]
            embedded = 0
            if missing:
                configure_runtime(config, local_files_only)
                embedder = make_embedder(profile, local_files_only)
                actual = embedder.metadata()
                if (int(actual.get("dimension", 0)) != MODEL_DIMENSION
                        or any(actual.get(key) != metadata[key] for key in (
                            "model_key", "model_id", "revision", "query_prefix", "document_prefix", "adapter",
                            "max_seq_length", "normalize_embeddings",
                        ))):
                    raise ValueError("loaded embedder does not match pinned E5 metadata")
                _validate_tokens(embedder, [row.text for row in missing], int(metadata["max_seq_length"]))
                encode_lock = threading.RLock()
                for offset in range(0, len(missing), batch_size):
                    batch = missing[offset:offset + batch_size]
                    with encode_lock:
                        vectors = np.asarray(embedder.encode_documents([row.text for row in batch]), dtype=np.float32)
                    if vectors.shape != (len(batch), MODEL_DIMENSION) or not np.isfinite(vectors).all():
                        raise ValueError("property embedding batch failed shape/finite validation")
                    norms = np.linalg.norm(vectors, axis=1)
                    if not np.isfinite(norms).all() or np.any(norms <= 1e-8):
                        raise ValueError("property embedding batch contains a zero or invalid norm")
                    vectors = np.asarray(vectors / norms[:, None], dtype=np.float32)
                    with connection.transaction():
                        with connection.cursor() as cursor:
                            cursor.executemany(
                                """
                                INSERT INTO property_semantic_profiles(
                                    generation_id, property_id, language, profile_key,
                                    search_text, content_hash, embedding
                                ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                                """,
                                [(generation_id, row.property_id, row.language, row.profile_key,
                                  row.text, row.content_hash, vector) for row, vector in zip(batch, vectors)],
                            )
                        embedded += len(batch)
                        connection.execute(
                            "UPDATE property_embedding_generations SET completed_count=%s WHERE id=%s",
                            (len(existing_keys) + embedded, generation_id),
                        )
            with connection.transaction():
                _verify_persisted(connection, generation_id, rows)
                final = connection.execute(
                    "SELECT count(*) FROM property_semantic_profiles WHERE generation_id=%s",
                    (generation_id,),
                ).fetchone()
                if int(final[0]) != len(rows):
                    raise RuntimeError("property generation has an incomplete profile corpus")
                connection.execute(
                    """
                    UPDATE property_embedding_generations
                    SET state='ready', completed_count=%s, finished_at=now(), error=NULL
                    WHERE id=%s AND state='building'
                    """,
                    (len(rows), generation_id),
                )
            return PropertyStageResult(
                generation_id, profile_set_id, "ready", len(rows), len(rows), embedded, False, False,
                config_hash, corpus.catalog_hash, corpus.source_hash, corpus.taxonomy_fingerprint,
                corpus.renderer_version, corpus.content_set_hash, time.perf_counter() - started,
            )
        except Exception as exc:
            if building and generation_id is not None:
                try:
                    with connection.transaction():
                        connection.execute(
                            """
                            UPDATE property_embedding_generations SET state='failed', error=%s
                            WHERE id=%s AND state='building'
                            """,
                            (f"{type(exc).__name__}: {exc}"[:2000], generation_id),
                        )
                except Exception:
                    connection.rollback()
            raise
        finally:
            connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
            connection.commit()


def property_embedding_status(database: Database) -> dict[str, Any]:
    """Read-only view of the property pointer and all staged generations."""
    require_database_url(database.database_url)
    with database.connection() as (connection, _):
        _verify_schema(connection)
        active = connection.execute(
            """
            SELECT g.id, g.state, g.expected_count, g.completed_count, a.activated_at,
                   m.model_id, m.revision, m.config_hash, p.catalog_hash, p.source_hash,
                   p.taxonomy_fingerprint, p.renderer_version, p.content_set_hash,
                   p.property_count, p.profile_count
            FROM active_property_embedding_generation a
            JOIN property_embedding_generations g ON g.id=a.generation_id
            JOIN embedding_model_configs m ON m.id=g.model_config_id
            JOIN property_profile_sets p ON p.id=g.profile_set_id
            WHERE a.model_key=%s
            """,
            (MODEL_KEY,),
        ).fetchone()
        generations = connection.execute(
            """
            SELECT g.id, g.profile_set_id, g.state, g.expected_count, g.completed_count,
                   g.started_at, g.finished_at, g.error, g.content_set_hash, m.config_hash
            FROM property_embedding_generations g
            JOIN embedding_model_configs m ON m.id=g.model_config_id
            ORDER BY g.id DESC
            """
        ).fetchall()
    return {
        "model_key": MODEL_KEY,
        "dimension": MODEL_DIMENSION,
        "active": None if active is None else dict(zip((
            "generation_id", "state", "expected_count", "completed_count", "activated_at",
            "model_id", "revision", "model_config_hash", "catalog_hash", "source_hash",
            "taxonomy_fingerprint", "renderer_version", "content_set_hash",
            "property_count", "profile_count",
        ), active)),
        "generations": [dict(zip((
            "generation_id", "profile_set_id", "state", "expected_count", "completed_count",
            "started_at", "finished_at", "error", "content_set_hash", "model_config_hash",
        ), row)) for row in generations],
    }
