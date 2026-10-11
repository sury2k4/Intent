"""Administrative E5 v3 category pointer promotion and event-bound rollback."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .db import Database, require_database_url, validate_database_target
from .embedding_sync import (
    ADVISORY_LOCK_NAME as E5_SYNC_LOCK,
    ARTIFACT_KIND,
    MODEL_DIMENSION,
    MODEL_KEY,
    model_profile,
    profile_content_hash,
    profile_source_fingerprint,
    semantic_model_metadata,
    stable_hash,
)
from .rendering import (
    PROFILE_RENDERER_VERSION,
    PROFILE_SCHEMA_VERSION,
    load_enrichment,
    render_category_profiles,
    validate_enrichment,
)
from .serving_config import E5_MODEL_ID, E5_MODEL_REVISION, load_yaml
from .taxonomy import Taxonomy

_ACTIVE_TABLE = "active_category_profile_generation_v3"
# Preserve lock order and names while legacy sync/promotion writers may coexist.
PAIR_LOCK_NAME = "model-intent:e5-poi-paired-promotion"
_LEGACY_SYNC_LOCK = "model-intent-poi-sync"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROVENANCE_KEYS = (
    "taxonomy_fingerprint", "model_config_hash", "source_fingerprint",
    "content_set_hash", "renderer_version", "profile_schema_version",
    "profile_count", "category_count", "model_id", "revision", "dimension",
)


def _require_hash(value: Any, label: str) -> str:
    result = str(value or "")
    if not _HEX_SHA256.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return result


def _lock_generation_syncs(connection: Any) -> None:
    """Serialize pointer changes with existing generation writers."""
    for lock_name in (E5_SYNC_LOCK, _LEGACY_SYNC_LOCK):
        connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_name,))


def _write_rollback_record(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path).resolve()
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing rollback record: {target}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot persist rollback record before promotion: {target}") from exc


def _switch_pointer(
    connection: Any, *, expected: int | None, target: int, activated_at: datetime,
) -> None:
    row = connection.execute(
        f"""
        INSERT INTO {_ACTIVE_TABLE}(model_key, generation_id, activated_at) VALUES (%s,%s,%s)
        ON CONFLICT (model_key) DO UPDATE
        SET generation_id=EXCLUDED.generation_id, activated_at=EXCLUDED.activated_at
        WHERE {_ACTIVE_TABLE}.generation_id IS NOT DISTINCT FROM %s
        RETURNING generation_id
        """,
        (MODEL_KEY, target, activated_at, expected),
    ).fetchone()
    if row is None:
        raise RuntimeError("active pointer changed during category promotion; update rolled back")


def _pinned_provenance(config_path: str | Path, taxonomy_path: str | Path, database: Database) -> tuple[dict[str, Any], dict[tuple[int, str], tuple[str, str]]]:
    """Compute the serving runtime's E5 contract without loading a model or opening a DB."""
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    validate_database_target(config, database.database_url)
    if config.get("retrieval_backend") != "pgvector" or config.get("retrieval_mode", "pg_exact") != "pg_exact":
        raise ValueError("category promotion requires a pg_exact pgvector serving config")
    configured_taxonomy = config.get("taxonomy_path")
    if not configured_taxonomy:
        raise ValueError("serving config must pin taxonomy_path")
    configured_path = Path(configured_taxonomy)
    if not configured_path.is_absolute():
        configured_path = config_path.parent.parent / configured_path
    if configured_path.resolve() != Path(taxonomy_path).resolve():
        raise ValueError("--taxonomy does not match the taxonomy_path pinned by --config")
    taxonomy = Taxonomy.load(taxonomy_path)
    enrichment_path = config.get("enrichment_path")
    enrichment = load_enrichment(
        None if enrichment_path is None else (
            Path(enrichment_path) if Path(enrichment_path).is_absolute()
            else (config_path.parent.parent / enrichment_path).resolve()
        )
    )
    validate_enrichment(taxonomy, enrichment, require_rich_fields=True)
    profile = model_profile(config)
    if (profile.model_id != E5_MODEL_ID or profile.revision != E5_MODEL_REVISION
            or profile.expected_dimension != MODEL_DIMENSION
            or profile.query_prefix != "query: " or profile.document_prefix != "passage: "):
        raise ValueError("category promotion requires the exact pinned E5-base 768D serving profile")
    rendered = render_category_profiles(
        taxonomy, tuple(config.get("languages", ["vi", "en"])), enrichment,
        require_rich_fields=True,
    )
    if not rendered or any(
        not item.item_id.isdigit() or item.item_id != str(int(item.item_id)) for item in rendered
    ):
        raise ValueError("configured category profiles require canonical numeric leaf IDs")
    rows = {
        (int(item.item_id), item.language): (item.text, profile_content_hash(item))
        for item in rendered
    }
    if len(rows) != len(rendered):
        raise ValueError("configured category profiles require unique canonical numeric leaf IDs")
    leaves = {int(item.id) for item in taxonomy.document.items if not taxonomy.has_children(item.id)}
    if {item_id for item_id, _ in rows} != leaves:
        raise ValueError("configured category profiles must cover exactly the leaf taxonomy")
    return {
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "model_config_hash": stable_hash(semantic_model_metadata(profile)),
        "source_fingerprint": profile_source_fingerprint(enrichment),
        "content_set_hash": stable_hash(sorted(
            (item.item_id, item.language, profile_content_hash(item)) for item in rendered
        )),
        "renderer_version": PROFILE_RENDERER_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_count": len(rendered),
        "category_count": len(leaves),
        "model_id": profile.model_id,
        "revision": profile.revision,
        "dimension": MODEL_DIMENSION,
    }, rows


def _validated_category(
    connection: Any, generation_id: int, *,
    expected: dict[str, Any] | None = None,
    rendered_rows: dict[tuple[int, str], tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify immutable E5 provenance and actual candidate rows, under a transaction.

    expected=None is reserved for recording the existing active generation; its
    persisted, self-consistent provenance is rechecked on rollback. The target
    additionally matches the exact profiles rendered from the pinned config.
    """
    row = connection.execute(
        """
        SELECT g.state, g.artifact_kind, g.expected_count, g.completed_count,
               g.content_set_hash, t.fingerprint, m.model_key, m.config_hash,
               m.dimension, ps.profile_count, ps.category_count, ps.content_set_hash,
               ps.taxonomy_snapshot_id, m.adapter_version, m.model_id, m.revision,
               m.query_prefix, m.document_prefix, m.normalize_embeddings,
               m.max_seq_length, ps.source_fingerprint, ps.profile_schema_version,
               g.renderer_version, ps.renderer_version
        FROM embedding_generations g
        JOIN embedding_model_configs m ON m.id=g.model_config_id
        JOIN taxonomy_snapshots t ON t.id=g.taxonomy_snapshot_id
        JOIN category_profile_sets ps ON ps.id=g.profile_set_id
        WHERE g.id=%s AND ps.taxonomy_snapshot_id=g.taxonomy_snapshot_id
        FOR SHARE OF g
        """,
        (generation_id,),
    ).fetchone()
    if not row or (str(row[0]), str(row[1]), str(row[6]), int(row[8])) != (
        "ready", ARTIFACT_KIND, MODEL_KEY, MODEL_DIMENSION,
    ):
        raise ValueError("generation must be a ready 768D E5 category_profile_v3")
    model_hash = stable_hash({
        "adapter": row[13], "model_key": row[6], "model_id": row[14],
        "revision": row[15], "query_prefix": row[16], "document_prefix": row[17],
        "normalize_embeddings": row[18], "dimension": row[8],
        "max_seq_length": row[19],
    })
    provenance = {
        "taxonomy_fingerprint": str(row[5]), "model_config_hash": str(row[7]),
        "source_fingerprint": str(row[20]), "content_set_hash": str(row[4]),
        "renderer_version": str(row[22]), "profile_schema_version": int(row[21]),
        "profile_count": int(row[2]), "category_count": int(row[10]),
        "model_id": str(row[14]), "revision": str(row[15]),
        "dimension": int(row[8]),
    }
    if (provenance["profile_count"] < 1 or provenance["category_count"] < 1
            or int(row[3]) != provenance["profile_count"]
            or int(row[9]) != provenance["profile_count"]
            or str(row[11]) != provenance["content_set_hash"]
            or model_hash != provenance["model_config_hash"]
            or int(row[12]) < 1
            or provenance["renderer_version"] != PROFILE_RENDERER_VERSION
            or str(row[23]) != PROFILE_RENDERER_VERSION
            or provenance["profile_schema_version"] != PROFILE_SCHEMA_VERSION):
        raise ValueError("generation provenance, renderer, or profile counts mismatch")
    for key in ("taxonomy_fingerprint", "model_config_hash", "source_fingerprint", "content_set_hash"):
        _require_hash(provenance[key], key)
    if expected is not None and provenance != expected:
        raise ValueError("generation does not match pinned serving config/provenance")

    profiles = connection.execute(
        """
        SELECT place_type_id, language, search_text, content_hash
        FROM category_search_profiles_v3 WHERE generation_id=%s
        """,
        (generation_id,),
    ).fetchall()
    if (len(profiles) != provenance["profile_count"]
            or len({int(item[0]) for item in profiles}) != provenance["category_count"]):
        raise ValueError("generation candidate row/category count mismatch")
    fingerprints = []
    actual_rows = {}
    for item_id, language, search_text, content_hash in profiles:
        key = (int(item_id), str(language))
        content_hash = str(content_hash)
        if content_hash != stable_hash({
            "item_id": str(item_id), "language": str(language), "text": str(search_text),
        }) or key in actual_rows:
            raise ValueError("generation candidate content hash or identity mismatch")
        fingerprints.append((str(item_id), str(language), content_hash))
        actual_rows[key] = (str(search_text), content_hash)
    if stable_hash(sorted(fingerprints)) != provenance["content_set_hash"]:
        raise ValueError("generation candidate content set hash mismatch")
    if rendered_rows is not None and actual_rows != rendered_rows:
        raise ValueError("generation candidate rows differ from pinned serving profiles")
    return provenance


def _active_pointer(connection: Any) -> int | None:
    row = connection.execute(
        f"SELECT generation_id FROM {_ACTIVE_TABLE} WHERE model_key=%s FOR UPDATE",
        (MODEL_KEY,),
    ).fetchone()
    return int(row[0]) if row is not None else None


def _generation_id(value: Any, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("category generation IDs must be positive integers")
    return value


def promote_category_generation(
    database: Database, generation_id: int, *, config_path: str | Path,
    taxonomy_path: str | Path, rollback_record_path: str | Path,
) -> dict[str, Any]:
    """Promote only the category pointer and persist an exclusive rollback record."""
    require_database_url(database.database_url)
    _generation_id(generation_id)
    record_path = Path(rollback_record_path).resolve()
    if record_path.exists():
        raise ValueError(f"refusing to overwrite existing rollback record: {record_path}")
    pinned, rows = _pinned_provenance(config_path, taxonomy_path, database)
    with database.connection() as (connection, _):
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (PAIR_LOCK_NAME,))
            _lock_generation_syncs(connection)
            previous = _active_pointer(connection)
            if previous == generation_id:
                raise ValueError("requested category generation is already active")
            previous_provenance = (
                _validated_category(connection, previous) if previous is not None else None
            )
            provenance = _validated_category(
                connection, generation_id, expected=pinned, rendered_rows=rows,
            )
            activated_at = connection.execute("SELECT clock_timestamp()").fetchone()[0]
            if not isinstance(activated_at, datetime) or activated_at.tzinfo is None:
                raise RuntimeError("database did not provide a timezone-aware promotion timestamp")
            record = {
                "schema_version": 1, "model_key": MODEL_KEY,
                "previous_generation_id": previous, "promoted_generation_id": generation_id,
                "previous_provenance": previous_provenance, "provenance": provenance,
                "promotion_timestamp": activated_at.isoformat(),
            }
            record["record_sha256"] = stable_hash(record)
            # Persist before swapping: failed commits can leave an unused record, but
            # ID and promotion timestamp checks prevent its use against another event.
            _write_rollback_record(record_path, record)
            _switch_pointer(
                connection, expected=previous, target=generation_id,
                activated_at=activated_at,
            )
    return {**record, "rollback_record": str(record_path)}


def rollback_category_generation(database: Database, record_path: str | Path) -> dict[str, Any]:
    """Restore only the prior category pointer if this event is still active."""
    require_database_url(database.database_url)
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("model_key") != MODEL_KEY:
        raise ValueError("invalid category promotion rollback record")
    checksum = record.get("record_sha256")
    if not isinstance(checksum, str) or checksum != stable_hash({
        key: value for key, value in record.items() if key != "record_sha256"
    }):
        raise ValueError("category promotion rollback record checksum mismatch")
    before = _generation_id(record.get("previous_generation_id"), nullable=True)
    after = _generation_id(record.get("promoted_generation_id"))
    if before == after:
        raise ValueError("rollback record does not describe a distinct promoted generation")
    previous_provenance = record.get("previous_provenance")
    provenance = record.get("provenance")
    if (not isinstance(provenance, dict) or set(provenance) != set(_PROVENANCE_KEYS)
            or (before is None and previous_provenance is not None)
            or (before is not None and (
                not isinstance(previous_provenance, dict)
                or set(previous_provenance) != set(_PROVENANCE_KEYS)
            ))):
        raise ValueError("rollback record provenance is missing or inconsistent")
    try:
        promotion_timestamp = datetime.fromisoformat(record["promotion_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rollback record has an invalid promotion timestamp") from exc
    if promotion_timestamp.tzinfo is None:
        raise ValueError("rollback record requires a timezone-aware promotion timestamp")
    with database.connection() as (connection, _):
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (PAIR_LOCK_NAME,))
            _lock_generation_syncs(connection)
            if _active_pointer(connection) != after:
                raise RuntimeError("active category pointer differs from rollback record; refusing rollback")
            active_at = connection.execute(
                f"SELECT activated_at FROM {_ACTIVE_TABLE} WHERE model_key=%s",
                (MODEL_KEY,),
            ).fetchone()
            if not active_at or active_at[0] != promotion_timestamp:
                raise RuntimeError("active category pointer is not this promotion event; refusing stale rollback")
            # The promoted generation may be broken; that is one reason to roll it back.
            # Pointer ID and activation timestamp bind this record to the exact event.
            # Only the generation being restored must still be complete.
            if before is not None:
                try:
                    validated = _validated_category(connection, before)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError("previous category generation is no longer complete/compatible") from exc
                if validated != previous_provenance:
                    raise RuntimeError("previous category provenance changed; refusing rollback")
            if before is None:
                result = connection.execute(
                    f"DELETE FROM {_ACTIVE_TABLE} WHERE model_key=%s AND generation_id=%s "
                    "AND activated_at=%s RETURNING generation_id",
                    (MODEL_KEY, after, promotion_timestamp),
                ).fetchone()
            else:
                result = connection.execute(
                    f"UPDATE {_ACTIVE_TABLE} SET generation_id=%s, activated_at=now() "
                    "WHERE model_key=%s AND generation_id=%s AND activated_at=%s "
                    "RETURNING generation_id",
                    (before, MODEL_KEY, after, promotion_timestamp),
                ).fetchone()
            if result is None:
                raise RuntimeError("active category pointer changed during rollback")
    return {"restored_generation_id": before, "replaced_generation_id": after, "model_key": MODEL_KEY}
