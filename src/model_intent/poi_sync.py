from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .db import Database, require_database_url
from .osm_ingest import file_sha256, stable_hash

ADVISORY_LOCK_NAME = "model-intent-poi-sync"
PAIR_LOCK_NAME = "model-intent:e5-poi-paired-promotion"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def validate_poi_database_schema(connection: Any) -> None:
    """Fail before writes when a legacy POI schema occupies the table names."""
    expected = {
        "poi_dataset_generations": {
            "id", "dataset_key", "artifact_id", "state", "expected_count", "completed_count",
            "content_set_hash", "taxonomy_fingerprint", "mapping_sha256", "property_taxonomy_sha256",
        },
        "pois": {
            "generation_id", "poi_id", "place_type_id", "osm_tags", "properties_osm",
            "properties_mock", "effective_properties", "rating", "review_count", "content_hash",
        },
        "active_poi_dataset_generation": {"dataset_key", "generation_id"},
    }
    rows = connection.execute(
        """
        SELECT table_name, column_name FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = ANY(%s)
        """,
        (list(expected),),
    ).fetchall()
    actual: dict[str, set[str]] = {name: set() for name in expected}
    for table, column in rows:
        actual[str(table)].add(str(column))
    missing = {
        name: sorted(columns - actual[name])
        for name, columns in expected.items() if columns - actual[name]
    }
    if missing:
        raise RuntimeError(
            f"POI database schema does not match this checkout: {missing}; "
            "use a fresh isolated migrated database, not the existing local volume"
        )


def _load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("POI manifest must be an object")
    artifact_path = manifest_path.parent / "pois.jsonl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"POI artifact not found next to manifest: {artifact_path}")
    expected = str(manifest.get("pois_jsonl_sha256", ""))
    actual = file_sha256(artifact_path)
    if len(expected) != 64 or expected != actual:
        raise ValueError(f"POI artifact hash is missing or mismatched: expected {expected}, got {actual}")
    return manifest, artifact_path


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _validate_osm_only_row(row: dict[str, Any], number: int) -> None:
    """Fail closed if a purported OSM-only artifact carries fabricated evidence."""
    properties = row.get("properties")
    rating = row.get("rating")
    if not isinstance(properties, dict) or not isinstance(rating, dict):
        raise ValueError(f"OSM-only POI row {number} must include properties and rating provenance")
    osm = properties.get("osm")
    effective = properties.get("effective")
    mock = properties.get("mock")
    if (not isinstance(osm, dict) or not isinstance(effective, dict)
            or not isinstance(mock, dict) or mock or effective != osm
            or any(not isinstance(value, dict) or value.get("source") not in ("osm", "unknown")
                   for value in effective.values())
            or rating.get("source") not in ("osm", "unknown")
            or rating.get("mock_schema_version") is not None
            or rating.get("mock_seed") is not None
            or (rating.get("source") == "unknown"
                and (rating.get("value") is not None or rating.get("review_count", 0) != 0))):
        raise ValueError(f"OSM-only POI row {number} contains mock or conflicting evidence")


def _artifact_inventory(path: Path, *, osm_only: bool = False) -> tuple[int, str]:
    """Verify row hashes and content-set provenance without retaining the corpus."""
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    for row in _iter_rows(path):
        if not isinstance(row, dict) or not row.get("poi_id"):
            raise ValueError(f"invalid POI artifact row at row {count + 1}")
        if osm_only:
            _validate_osm_only_row(row, count + 1)
        content_hash = str(row.get("content_hash", ""))
        if not _HEX_SHA256.fullmatch(content_hash) or content_hash != stable_hash({**row, "content_hash": ""}):
            raise ValueError(f"invalid POI row content hash at row {count + 1}")
        if count:
            digest.update(b",")
        digest.update(json.dumps([str(row["poi_id"]), content_hash], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        count += 1
    digest.update(b"]")
    return count, digest.hexdigest()


def _iter_batches(rows: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _require_hash(value: Any, label: str) -> str:
    result = str(value or "")
    if not _HEX_SHA256.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return result


def _manifest_mock_metadata(manifest: dict[str, Any]) -> tuple[str, int]:
    """Map explicit OSM-only nulls to DB sentinels without rewriting the manifest.

    Migration 007 requires non-NULL generation metadata: schema ``none`` and seed
    ``0`` mean no mock data was generated. Per-POI mock fields remain NULL.
    Older mocked manifests retain their existing schema/default seed behavior.
    """
    schema = manifest.get("mock_schema_version", "unknown")
    seed = manifest.get("mock_seed", 17)
    osm_only = manifest.get("osm_only") is True
    osm_only_kind = str(manifest.get("artifact_kind", "")).endswith("_osm_only")
    if osm_only or osm_only_kind or seed is None:
        if not (osm_only and osm_only_kind and schema is None and seed is None):
            raise ValueError("OSM-only POI manifest requires osm_only=true and null mock schema/seed")
        return "none", 0
    return str(schema), int(seed)


def _row_args(generation_id: int, payload: dict[str, Any]) -> tuple[Any, ...]:
    osm = payload.get("osm") or {}
    mapping = payload.get("mapping") or {}
    location = payload.get("location") or {}
    properties = payload.get("properties") or {}
    rating = payload.get("rating") or {}
    return (
        generation_id,
        str(payload["poi_id"]),
        str(osm.get("type", "unknown")),
        int(osm.get("id", 0)),
        int(payload["category_id"]),
        payload.get("name"),
        str(payload.get("display_name", payload["poi_id"])),
        json.dumps(payload.get("names") or {}, ensure_ascii=False),
        float(location["lat"]),
        float(location["lon"]),
        str(location.get("method", "unknown")),
        str(mapping.get("rule_id", "unknown")),
        json.dumps(osm.get("tags") or {}, ensure_ascii=False),
        json.dumps(properties.get("osm") or {}, ensure_ascii=False),
        json.dumps(properties.get("mock") or {}, ensure_ascii=False),
        json.dumps(properties.get("effective") or {}, ensure_ascii=False),
        float(rating["value"]) if rating.get("value") is not None else None,
        int(rating.get("review_count", 0)),
        str(rating.get("source", "unknown")),
        rating.get("mock_schema_version"),
        int(rating["mock_seed"]) if rating.get("mock_seed") is not None else None,
        str(payload.get("content_hash", "")),
        int(osm["version"]) if osm.get("version") is not None else None,
        str(osm["timestamp"]) if osm.get("timestamp") else None,
    )


def sync_poi_artifact(
    database: Database,
    manifest_path: str | Path,
    *,
    dataset_key: str = "hcmc-osm",
    activate: bool = False,
    batch_size: int = 500,
) -> dict[str, Any]:
    if activate:
        raise RuntimeError("unilateral POI activation is disabled on unified DB; use paired promotion")
    started = time.perf_counter()
    if batch_size < 1:
        raise ValueError("POI sync batch size must be positive")
    require_database_url(database.database_url)
    manifest, artifact_path = _load_manifest(manifest_path)
    artifact_id = str(manifest.get("artifact_id", ""))
    if not artifact_id:
        raise ValueError("POI manifest requires artifact_id")
    if "poi_count" not in manifest:
        raise ValueError("POI manifest requires poi_count")
    expected_count = int(manifest["poi_count"])
    if expected_count < 0:
        raise ValueError("POI manifest poi_count must be nonnegative")
    mock_schema_version, mock_seed = _manifest_mock_metadata(manifest)
    osm_only = mock_schema_version == "none" and mock_seed == 0
    row_count, content_set_hash = _artifact_inventory(artifact_path, osm_only=osm_only)
    if row_count != expected_count:
        raise ValueError(f"POI manifest row count mismatch: {row_count} != {expected_count}")
    if content_set_hash != _require_hash(manifest.get("content_set_hash"), "POI content set hash"):
        raise ValueError("POI content set hash does not match artifact rows")
    source_manifest_hash = file_sha256(manifest_path)
    taxonomy_fingerprint = _require_hash(manifest.get("taxonomy_fingerprint"), "taxonomy fingerprint")
    mapping_sha256 = _require_hash(manifest.get("mapping_sha256"), "mapping hash")
    property_sha256 = _require_hash(manifest.get("property_taxonomy_sha256"), "property taxonomy hash")
    with database.connection() as (connection, _):
        validate_poi_database_schema(connection)
        connection.execute("SELECT pg_advisory_lock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
        generation_id: int | None = None
        building_generation = False
        try:
            with connection.transaction():
                existing = connection.execute(
                    """
                    SELECT id, state, expected_count, completed_count, artifact_sha256,
                           content_set_hash, taxonomy_fingerprint, mapping_sha256,
                           property_taxonomy_sha256, source_manifest_hash, mapping_version,
                           property_taxonomy_version, mock_schema_version, mock_seed
                    FROM poi_dataset_generations WHERE dataset_key=%s AND artifact_id=%s
                    """,
                    (dataset_key, artifact_id),
                ).fetchone()
                if existing and str(existing[1]) == "ready":
                    generation_id = int(existing[0])
                    if (
                        int(existing[2]) != expected_count or int(existing[3]) != expected_count
                        or str(existing[4] or "") != str(manifest["pois_jsonl_sha256"])
                        or str(existing[5] or "") != content_set_hash
                        or str(existing[6]) != taxonomy_fingerprint
                        or str(existing[7]) != mapping_sha256
                        or str(existing[8]) != property_sha256
                        or str(existing[9]) != source_manifest_hash
                        or str(existing[10]) != str(manifest.get("mapping_version", "unknown"))
                        or str(existing[11]) != str(manifest.get("property_taxonomy_version", "unknown"))
                        or str(existing[12]) != mock_schema_version
                        or int(existing[13]) != mock_seed
                    ):
                        raise RuntimeError("ready POI generation is immutable and conflicts with this artifact manifest")
                    stored_count = int(connection.execute(
                        "SELECT count(*) FROM pois WHERE generation_id=%s", (generation_id,)
                    ).fetchone()[0])
                    if stored_count != expected_count or _database_poi_content_hash(
                        connection, generation_id, expected_count,
                    ) != content_set_hash:
                        raise RuntimeError("ready POI generation rows are incomplete or conflict with the manifest")
                    return {
                        "generation_id": generation_id,
                        "state": "ready",
                        "expected_count": expected_count,
                        "completed_count": int(existing[3]),
                        "already_ready": True,
                        "activated": False,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                if existing:
                    if str(existing[1]) == "retired":
                        raise RuntimeError("retired POI generation is immutable")
                    active = connection.execute(
                        "SELECT generation_id FROM active_poi_dataset_generation WHERE dataset_key=%s",
                        (dataset_key,),
                    ).fetchone()
                    if active and int(active[0]) == int(existing[0]):
                        raise RuntimeError("refusing to rebuild an active POI generation")
                row = connection.execute(
                    """
                    INSERT INTO poi_dataset_generations(
                        dataset_key, artifact_id, state, source_manifest_hash,
                        artifact_sha256, content_set_hash, taxonomy_fingerprint,
                        mapping_version, mapping_sha256, property_taxonomy_version,
                        property_taxonomy_sha256, mock_schema_version, mock_seed,
                        expected_count, completed_count
                    ) VALUES (%s,%s,'building',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)
                    ON CONFLICT (dataset_key, artifact_id) DO UPDATE
                    SET state='building', error=NULL, expected_count=EXCLUDED.expected_count,
                        completed_count=0, source_manifest_hash=EXCLUDED.source_manifest_hash,
                        artifact_sha256=EXCLUDED.artifact_sha256,
                        content_set_hash=EXCLUDED.content_set_hash,
                        taxonomy_fingerprint=EXCLUDED.taxonomy_fingerprint,
                        mapping_version=EXCLUDED.mapping_version,
                        mapping_sha256=EXCLUDED.mapping_sha256,
                        property_taxonomy_version=EXCLUDED.property_taxonomy_version,
                        property_taxonomy_sha256=EXCLUDED.property_taxonomy_sha256,
                        mock_schema_version=EXCLUDED.mock_schema_version,
                        mock_seed=EXCLUDED.mock_seed
                    RETURNING id
                    """,
                    (
                        dataset_key,
                        artifact_id,
                        source_manifest_hash,
                        str(manifest["pois_jsonl_sha256"]),
                        content_set_hash,
                        taxonomy_fingerprint,
                        str(manifest.get("mapping_version", "unknown")),
                        mapping_sha256,
                        str(manifest.get("property_taxonomy_version", "unknown")),
                        property_sha256,
                        mock_schema_version,
                        mock_seed,
                        expected_count,
                    ),
                ).fetchone()
                generation_id = int(row[0])
                building_generation = True
                connection.execute("DELETE FROM pois WHERE generation_id=%s", (generation_id,))
            insert_sql = """
                INSERT INTO pois(
                    generation_id, poi_id, osm_type, osm_id, place_type_id, name,
                    display_name, names, latitude, longitude, location_method,
                    mapping_rule_id, osm_tags, properties_osm, properties_mock,
                    effective_properties, rating, review_count, rating_source,
                    rating_mock_schema_version, rating_mock_seed, content_hash,
                    source_version, source_timestamp
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,
                    %s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,
                    %s,%s,%s,%s,%s,%s,%s,%s
                )
                ON CONFLICT (generation_id, poi_id) DO UPDATE SET
                    osm_type=EXCLUDED.osm_type, osm_id=EXCLUDED.osm_id,
                    place_type_id=EXCLUDED.place_type_id, name=EXCLUDED.name,
                    display_name=EXCLUDED.display_name, names=EXCLUDED.names,
                    latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude,
                    location_method=EXCLUDED.location_method,
                    mapping_rule_id=EXCLUDED.mapping_rule_id,
                    osm_tags=EXCLUDED.osm_tags, properties_osm=EXCLUDED.properties_osm,
                    properties_mock=EXCLUDED.properties_mock,
                    effective_properties=EXCLUDED.effective_properties,
                    rating=EXCLUDED.rating, review_count=EXCLUDED.review_count,
                    rating_source=EXCLUDED.rating_source,
                    rating_mock_schema_version=EXCLUDED.rating_mock_schema_version,
                    rating_mock_seed=EXCLUDED.rating_mock_seed,
                    content_hash=EXCLUDED.content_hash,
                    source_version=EXCLUDED.source_version, source_timestamp=EXCLUDED.source_timestamp
            """
            completed = 0
            for batch in _iter_batches(_iter_rows(artifact_path), batch_size):
                with connection.transaction():
                    with connection.cursor() as cursor:
                        cursor.executemany(insert_sql, [_row_args(generation_id, row) for row in batch])
                    completed += len(batch)
                    connection.execute(
                        "UPDATE poi_dataset_generations SET completed_count=%s WHERE id=%s",
                        (completed, generation_id),
                    )
            with connection.transaction():
                # The file was read twice; fail closed if it changed during the import.
                if completed != expected_count or file_sha256(artifact_path) != manifest["pois_jsonl_sha256"]:
                    raise RuntimeError("POI artifact changed during sync")
                final_count = int(connection.execute("SELECT count(*) FROM pois WHERE generation_id=%s", (generation_id,)).fetchone()[0])
                if final_count != expected_count:
                    raise RuntimeError(f"POI generation integrity mismatch: {final_count} != {expected_count}")
                if _database_poi_content_hash(connection, generation_id, expected_count) != content_set_hash:
                    raise RuntimeError("POI generation persisted content set hash does not match manifest")
                connection.execute(
                    "UPDATE poi_dataset_generations SET state='ready', completed_count=%s, finished_at=now(), error=NULL WHERE id=%s",
                    (final_count, generation_id),
                )
            return {
                "generation_id": generation_id,
                "state": "ready",
                "expected_count": expected_count,
                "completed_count": expected_count,
                "already_ready": False,
                "activated": False,
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            if building_generation and generation_id is not None:
                try:
                    with connection.transaction():
                        connection.execute(
                            "UPDATE poi_dataset_generations SET state='failed', error=%s WHERE id=%s AND state='building'",
                            (f"{type(exc).__name__}: {exc}", generation_id),
                        )
                except Exception:
                    connection.rollback()
            raise
        finally:
            connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
            connection.commit()


def activate_poi_generation(database: Database, generation_id: int, *, dataset_key: str = "hcmc-osm") -> None:
    """Unilateral POI activation is unsafe; paired promotion owns both pointers."""
    raise RuntimeError("unilateral POI activation is disabled on unified DB; use paired promotion")


def _lock_generation_syncs(connection: Any) -> None:
    """Serialize pointer swaps with writers that can rebuild staged generations."""
    from .embedding_sync import ADVISORY_LOCK_NAME as E5_SYNC_LOCK

    for lock_name in (E5_SYNC_LOCK, ADVISORY_LOCK_NAME):
        connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_name,))


def _read_active_pair(connection: Any, dataset_key: str) -> tuple[int | None, int | None]:
    from .embedding_sync import MODEL_KEY

    embedding = connection.execute(
        "SELECT generation_id FROM active_category_profile_generation_v3 WHERE model_key=%s FOR UPDATE",
        (MODEL_KEY,),
    ).fetchone()
    poi = connection.execute(
        "SELECT generation_id FROM active_poi_dataset_generation WHERE dataset_key=%s FOR UPDATE",
        (dataset_key,),
    ).fetchone()
    return (int(embedding[0]) if embedding else None, int(poi[0]) if poi else None)


def _validated_pair(
    connection: Any,
    embedding_generation_id: int,
    poi_generation_id: int,
    *,
    dataset_key: str,
    taxonomy_fingerprint: str,
    mapping_sha256: str,
    property_taxonomy_sha256: str,
    model_config_hash: str,
    source_fingerprint: str,
    embedding_content_set_hash: str,
    renderer_version: str,
    profile_schema_version: int,
    expected_profile_count: int,
    expected_category_count: int,
) -> dict[str, Any]:
    from .embedding_sync import ARTIFACT_KIND, MODEL_DIMENSION, MODEL_KEY, stable_hash
    from .rendering import PROFILE_RENDERER_VERSION, PROFILE_SCHEMA_VERSION

    if (renderer_version != PROFILE_RENDERER_VERSION or profile_schema_version != PROFILE_SCHEMA_VERSION
            or expected_profile_count < 1 or expected_category_count < 1):
        raise ValueError("expected E5 renderer/schema or profile counts are invalid")

    embedding = connection.execute(
        """
        SELECT g.state, g.artifact_kind, g.expected_count, g.completed_count,
               g.content_set_hash, t.fingerprint, m.model_key, m.config_hash, m.dimension,
               ps.profile_count, ps.category_count, ps.content_set_hash,
               ps.taxonomy_snapshot_id, m.adapter_version, m.model_id, m.revision,
               m.query_prefix, m.document_prefix, m.normalize_embeddings, m.max_seq_length,
               ps.source_fingerprint, ps.profile_schema_version, g.renderer_version,
               ps.renderer_version
        FROM embedding_generations g
        JOIN embedding_model_configs m ON m.id=g.model_config_id
        JOIN taxonomy_snapshots t ON t.id=g.taxonomy_snapshot_id
        JOIN category_profile_sets ps ON ps.id=g.profile_set_id
        WHERE g.id=%s AND ps.taxonomy_snapshot_id=g.taxonomy_snapshot_id
        FOR SHARE OF g
        """,
        (embedding_generation_id,),
    ).fetchone()
    if not embedding or (str(embedding[0]), str(embedding[1]), str(embedding[6]), int(embedding[8])) != (
        "ready", ARTIFACT_KIND, MODEL_KEY, MODEL_DIMENSION,
    ):
        raise ValueError("embedding generation must be a ready 768D E5 category_profile_v3")
    stored_model_hash = stable_hash({
        "adapter": embedding[13], "model_key": embedding[6],
        "model_id": embedding[14], "revision": embedding[15],
        "query_prefix": embedding[16], "document_prefix": embedding[17],
        "normalize_embeddings": embedding[18], "dimension": embedding[8],
        "max_seq_length": embedding[19],
    })
    if (str(embedding[5]) != taxonomy_fingerprint or str(embedding[7]) != model_config_hash
            or stored_model_hash != model_config_hash
            or str(embedding[20]) != source_fingerprint
            or int(embedding[21]) != profile_schema_version
            or str(embedding[22]) != renderer_version or str(embedding[23]) != renderer_version
            or str(embedding[4]) != embedding_content_set_hash
            or int(embedding[9]) != expected_profile_count
            or int(embedding[10]) != expected_category_count
            or int(embedding[2]) < 1 or int(embedding[2]) != int(embedding[3])
            or int(embedding[2]) != int(embedding[9])
            or str(embedding[4]) != str(embedding[11])
            or not _HEX_SHA256.fullmatch(str(embedding[4]))):
        raise ValueError("E5 generation taxonomy/model fingerprint or profile counts/hashes mismatch")
    profiles = connection.execute(
        """
        SELECT place_type_id, language, search_text, content_hash
        FROM category_search_profiles_v3 WHERE generation_id=%s
        """,
        (embedding_generation_id,),
    ).fetchall()
    if len(profiles) != int(embedding[2]) or len({int(row[0]) for row in profiles}) != int(embedding[10]):
        raise ValueError("E5 generation is missing category profiles")
    verified_profiles = []
    for place_type_id, language, search_text, content_hash in profiles:
        expected_hash = stable_hash({
            "item_id": str(place_type_id), "language": str(language), "text": str(search_text),
        })
        if str(content_hash) != expected_hash:
            raise ValueError("E5 generation contains a profile with a mismatched content hash")
        verified_profiles.append((str(place_type_id), str(language), expected_hash))
    if stable_hash(sorted(verified_profiles)) != str(embedding[4]):
        raise ValueError("E5 generation profile content set hash mismatch")

    poi = connection.execute(
        """
        SELECT state, dataset_key, expected_count, completed_count, content_set_hash,
               taxonomy_fingerprint, mapping_sha256, property_taxonomy_sha256,
               artifact_sha256, source_manifest_hash, artifact_id
        FROM poi_dataset_generations WHERE id=%s FOR SHARE
        """,
        (poi_generation_id,),
    ).fetchone()
    if not poi or str(poi[0]) != "ready" or str(poi[1]) != dataset_key:
        raise ValueError("POI generation must be ready and belong to the requested dataset")
    if (str(poi[5]) != taxonomy_fingerprint
            or str(poi[6]) != mapping_sha256 or str(poi[7]) != property_taxonomy_sha256
            or int(poi[2]) < 1 or int(poi[2]) != int(poi[3])
            or not all(_HEX_SHA256.fullmatch(str(poi[index])) for index in (4, 8, 9))):
        raise ValueError("POI generation taxonomy/mapping/property provenance or counts/hashes mismatch")
    if _database_poi_content_hash(connection, poi_generation_id, int(poi[2])) != str(poi[4]):
        raise ValueError("POI generation stored content set hash mismatch")
    return {
        "taxonomy_fingerprint": taxonomy_fingerprint,
        "model_config_hash": model_config_hash,
        "mapping_sha256": mapping_sha256,
        "property_taxonomy_sha256": property_taxonomy_sha256,
        "source_fingerprint": source_fingerprint,
        "embedding_content_set_hash": embedding_content_set_hash,
        "renderer_version": renderer_version,
        "profile_schema_version": profile_schema_version,
        "expected_profile_count": expected_profile_count,
        "expected_category_count": expected_category_count,
        "poi_artifact_id": str(poi[10]),
        "poi_content_set_hash": str(poi[4]),
    }


def _previous_pair_provenance(
    connection: Any, embedding_id: int | None, poi_id: int | None, dataset_key: str,
) -> dict[str, Any] | None:
    """Pin the previous pair's verified rows/hashes so rollback cannot trust stale IDs."""
    if embedding_id is None and poi_id is None:
        return None
    if embedding_id is None or poi_id is None:
        raise RuntimeError("cannot promote over incomplete prior E5/POI active pointers")
    e5 = connection.execute(
        """
        SELECT t.fingerprint, m.config_hash, ps.source_fingerprint,
               g.content_set_hash, g.renderer_version, ps.profile_schema_version,
               ps.profile_count, ps.category_count
        FROM embedding_generations g
        JOIN embedding_model_configs m ON m.id=g.model_config_id
        JOIN taxonomy_snapshots t ON t.id=g.taxonomy_snapshot_id
        JOIN category_profile_sets ps ON ps.id=g.profile_set_id
        WHERE g.id=%s FOR SHARE OF g
        """,
        (embedding_id,),
    ).fetchone()
    poi = connection.execute(
        """
        SELECT mapping_sha256, property_taxonomy_sha256
        FROM poi_dataset_generations WHERE id=%s AND dataset_key=%s FOR SHARE
        """,
        (poi_id, dataset_key),
    ).fetchone()
    if not e5 or not poi:
        raise RuntimeError("previous E5/POI pair is missing; refusing promotion")
    return _validated_pair(
        connection, embedding_id, poi_id, dataset_key=dataset_key,
        taxonomy_fingerprint=str(e5[0]), model_config_hash=str(e5[1]),
        source_fingerprint=str(e5[2]), embedding_content_set_hash=str(e5[3]),
        renderer_version=str(e5[4]), profile_schema_version=int(e5[5]),
        expected_profile_count=int(e5[6]), expected_category_count=int(e5[7]),
        mapping_sha256=str(poi[0]), property_taxonomy_sha256=str(poi[1]),
    )


def _database_poi_content_hash(connection: Any, generation_id: int, expected_count: int) -> str:
    """Stream the persisted rows in producer order, not just generation metadata."""
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    with connection.cursor(name=f"verify_poi_generation_{generation_id}") as cursor:
        cursor.execute(
            """
            SELECT poi_id, content_hash FROM pois WHERE generation_id=%s
            ORDER BY CASE osm_type WHEN 'node' THEN 0 WHEN 'way' THEN 1
                                   WHEN 'relation' THEN 2 ELSE 9 END,
                     osm_id, place_type_id
            """,
            (generation_id,),
        )
        for poi_id, row_hash in cursor:
            row_hash = str(row_hash)
            if not _HEX_SHA256.fullmatch(row_hash):
                raise ValueError("POI generation has an invalid row content hash")
            if count:
                digest.update(b",")
            digest.update(json.dumps([str(poi_id), row_hash], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            count += 1
    digest.update(b"]")
    if count != expected_count:
        raise ValueError(f"POI generation row count mismatch: {count} != {expected_count}")
    return digest.hexdigest()


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
    connection: Any, *, table: str, key_name: str, key: str,
    expected: int | None, target: int, activated_at: datetime,
) -> None:
    # The table/key are internal constants, not user-provided SQL identifiers.
    row = connection.execute(
        f"""
        INSERT INTO {table}({key_name}, generation_id, activated_at) VALUES (%s,%s,%s)
        ON CONFLICT ({key_name}) DO UPDATE
        SET generation_id=EXCLUDED.generation_id, activated_at=EXCLUDED.activated_at
        WHERE {table}.generation_id IS NOT DISTINCT FROM %s
        RETURNING generation_id
        """,
        (key, target, activated_at, expected),
    ).fetchone()
    if row is None:
        raise RuntimeError("active pointer changed during paired promotion; both updates rolled back")


def promote_generation_pair(
    database: Database,
    embedding_generation_id: int,
    poi_generation_id: int,
    *,
    taxonomy_fingerprint: str,
    mapping_sha256: str,
    property_taxonomy_sha256: str,
    model_config_hash: str,
    source_fingerprint: str,
    embedding_content_set_hash: str,
    renderer_version: str,
    profile_schema_version: int,
    expected_profile_count: int,
    expected_category_count: int,
    rollback_record_path: str | Path,
    dataset_key: str = "hcmc-osm",
) -> dict[str, Any]:
    """Promote a config-validated pair atomically; preserve prior generations."""
    require_database_url(database.database_url)
    for label, value in (
        ("taxonomy fingerprint", taxonomy_fingerprint), ("mapping hash", mapping_sha256),
        ("property taxonomy hash", property_taxonomy_sha256), ("model config hash", model_config_hash),
        ("E5 profile source fingerprint", source_fingerprint),
        ("E5 profile content set hash", embedding_content_set_hash),
    ):
        _require_hash(value, label)
    if embedding_generation_id < 1 or poi_generation_id < 1 or not dataset_key:
        raise ValueError("generation IDs must be positive and dataset_key must be nonempty")
    from .rendering import PROFILE_RENDERER_VERSION, PROFILE_SCHEMA_VERSION

    if (renderer_version != PROFILE_RENDERER_VERSION or profile_schema_version != PROFILE_SCHEMA_VERSION
            or type(expected_profile_count) is not int or expected_profile_count < 1
            or type(expected_category_count) is not int or expected_category_count < 1):
        raise ValueError("expected E5 renderer/schema/profile counts must match the serving renderer")
    if Path(rollback_record_path).exists():
        raise ValueError(f"refusing to overwrite existing rollback record: {rollback_record_path}")
    with database.connection() as (connection, _):
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (PAIR_LOCK_NAME,))
            _lock_generation_syncs(connection)
            old_embedding, old_poi = _read_active_pair(connection, dataset_key)
            if (old_embedding, old_poi) == (embedding_generation_id, poi_generation_id):
                raise ValueError("requested E5/POI pair is already active")
            previous_provenance = _previous_pair_provenance(
                connection, old_embedding, old_poi, dataset_key,
            )
            provenance = _validated_pair(
                connection, embedding_generation_id, poi_generation_id,
                dataset_key=dataset_key, taxonomy_fingerprint=taxonomy_fingerprint,
                mapping_sha256=mapping_sha256, property_taxonomy_sha256=property_taxonomy_sha256,
                model_config_hash=model_config_hash,
                source_fingerprint=source_fingerprint,
                embedding_content_set_hash=embedding_content_set_hash,
                renderer_version=renderer_version,
                profile_schema_version=profile_schema_version,
                expected_profile_count=expected_profile_count,
                expected_category_count=expected_category_count,
            )
            activation_at = connection.execute("SELECT clock_timestamp()").fetchone()[0]
            if not isinstance(activation_at, datetime) or activation_at.tzinfo is None:
                raise RuntimeError("database did not provide a timezone-aware promotion timestamp")
            record = {
                "schema_version": 2, "dataset_key": dataset_key,
                "previous": {"embedding_generation_id": old_embedding, "poi_generation_id": old_poi},
                "previous_provenance": previous_provenance,
                "promoted": {"embedding_generation_id": embedding_generation_id, "poi_generation_id": poi_generation_id},
                "provenance": provenance, "promotion_timestamp": activation_at.isoformat(),
            }
            record["record_sha256"] = stable_hash(record)
            # Persist before the pointer swap. An aborted transaction may leave an
            # unused record; rollback always checks BOTH currently active IDs.
            _write_rollback_record(rollback_record_path, record)
            from .embedding_sync import MODEL_KEY

            _switch_pointer(
                connection, table="active_category_profile_generation_v3",
                key_name="model_key", key=MODEL_KEY,
                expected=old_embedding, target=embedding_generation_id,
                activated_at=activation_at,
            )
            _switch_pointer(
                connection, table="active_poi_dataset_generation",
                key_name="dataset_key", key=dataset_key,
                expected=old_poi, target=poi_generation_id,
                activated_at=activation_at,
            )
    return {**record, "rollback_record": str(Path(rollback_record_path).resolve())}


def rollback_generation_pair(database: Database, record_path: str | Path) -> dict[str, Any]:
    """Compare both live pointers to a saved promotion before restoring either."""
    require_database_url(database.database_url)
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != 2:
        raise ValueError("invalid paired promotion rollback record (requires event-bound schema v2)")
    checksum = record.get("record_sha256")
    if not isinstance(checksum, str) or checksum != stable_hash({
        key: value for key, value in record.items() if key != "record_sha256"
    }):
        raise ValueError("paired promotion rollback record checksum mismatch")
    dataset_key = record.get("dataset_key")
    previous, promoted = record.get("previous"), record.get("promoted")
    provenance, previous_provenance = record.get("provenance"), record.get("previous_provenance")
    if (not isinstance(dataset_key, str) or not dataset_key
            or not isinstance(previous, dict) or not isinstance(promoted, dict)
            or not isinstance(provenance, dict)):
        raise ValueError("invalid paired promotion rollback record")
    try:
        promotion_timestamp = datetime.fromisoformat(record["promotion_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rollback record has invalid promotion timestamp") from exc
    if promotion_timestamp.tzinfo is None:
        raise ValueError("rollback record requires a timezone-aware promotion timestamp")
    for label, value in (
        ("taxonomy fingerprint", provenance.get("taxonomy_fingerprint")),
        ("mapping hash", provenance.get("mapping_sha256")),
        ("property taxonomy hash", provenance.get("property_taxonomy_sha256")),
        ("model config hash", provenance.get("model_config_hash")),
        ("E5 source fingerprint", provenance.get("source_fingerprint")),
        ("E5 content set hash", provenance.get("embedding_content_set_hash")),
    ):
        _require_hash(value, label)
    before = (previous.get("embedding_generation_id"), previous.get("poi_generation_id"))
    after = (promoted.get("embedding_generation_id"), promoted.get("poi_generation_id"))
    if any(value is not None and (type(value) is not int or value < 1) for value in (*before, *after)):
        raise ValueError("rollback record has invalid generation IDs")
    if None in after or before == after:
        raise ValueError("rollback record does not describe a distinct promoted pair")
    if (before[0] is None) != (before[1] is None):
        raise ValueError("rollback record has incomplete previous active pair")
    if (None in before and previous_provenance is not None) or (
        None not in before and not isinstance(previous_provenance, dict)
    ):
        raise ValueError("rollback record previous pair provenance is missing/inconsistent")
    with database.connection() as (connection, _):
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (PAIR_LOCK_NAME,))
            _lock_generation_syncs(connection)
            if _read_active_pair(connection, dataset_key) != after:
                raise RuntimeError("active pair differs from rollback record; refusing rollback")
            from .embedding_sync import MODEL_KEY

            e5_active_at = connection.execute(
                "SELECT activated_at FROM active_category_profile_generation_v3 WHERE model_key=%s",
                (MODEL_KEY,),
            ).fetchone()
            poi_active_at = connection.execute(
                "SELECT activated_at FROM active_poi_dataset_generation WHERE dataset_key=%s",
                (dataset_key,),
            ).fetchone()
            if (not e5_active_at or not poi_active_at
                    or e5_active_at[0] != promotion_timestamp
                    or poi_active_at[0] != promotion_timestamp):
                raise RuntimeError("active pair does not belong to this promotion event; refusing stale rollback")
            if previous_provenance is not None:
                expected_keys = (
                    "taxonomy_fingerprint", "mapping_sha256", "property_taxonomy_sha256",
                    "model_config_hash", "source_fingerprint", "embedding_content_set_hash",
                    "renderer_version", "profile_schema_version", "expected_profile_count",
                    "expected_category_count",
                )
                try:
                    validated = _validated_pair(
                        connection, before[0], before[1], dataset_key=dataset_key,
                        **{key: previous_provenance[key] for key in expected_keys},
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    raise RuntimeError("previous E5/POI pair is no longer complete/compatible") from exc
                if validated != previous_provenance:
                    raise RuntimeError("previous E5/POI provenance changed; refusing rollback")

            for table, key_name, key, original, current in (
                ("active_category_profile_generation_v3", "model_key", MODEL_KEY, before[0], after[0]),
                ("active_poi_dataset_generation", "dataset_key", dataset_key, before[1], after[1]),
            ):
                if original is None:
                    result = connection.execute(
                        f"DELETE FROM {table} WHERE {key_name}=%s AND generation_id=%s RETURNING generation_id",
                        (key, current),
                    ).fetchone()
                else:
                    result = connection.execute(
                        f"UPDATE {table} SET generation_id=%s, activated_at=now() "
                        f"WHERE {key_name}=%s AND generation_id=%s RETURNING generation_id",
                        (original, key, current),
                    ).fetchone()
                if result is None:
                    raise RuntimeError("active pair changed during rollback; both updates rolled back")
    return {"restored": record["previous"], "replaced": record["promoted"], "dataset_key": dataset_key}


def poi_status(database: Database, *, dataset_key: str = "hcmc-osm") -> dict[str, Any]:
    with database.connection() as (connection, _):
        active = connection.execute(
            """
            SELECT g.id, g.state, g.expected_count, g.completed_count,
                   g.artifact_id, g.content_set_hash, a.activated_at
            FROM active_poi_dataset_generation a
            JOIN poi_dataset_generations g ON g.id=a.generation_id
            WHERE a.dataset_key=%s
            """,
            (dataset_key,),
        ).fetchone()
        generations = connection.execute(
            """
            SELECT id, state, expected_count, completed_count, artifact_id, started_at, finished_at, error
            FROM poi_dataset_generations WHERE dataset_key=%s ORDER BY id DESC
            """,
            (dataset_key,),
        ).fetchall()
    return {
        "dataset_key": dataset_key,
        "active": {
            "generation_id": int(active[0]),
            "state": str(active[1]),
            "expected_count": int(active[2]),
            "completed_count": int(active[3]),
            "artifact_id": str(active[4]),
            "content_set_hash": str(active[5]),
            "activated_at": active[6],
        } if active else None,
        "generations": [
            {"generation_id": int(row[0]), "state": str(row[1]), "expected_count": int(row[2]), "completed_count": int(row[3]), "artifact_id": str(row[4]), "started_at": row[5], "finished_at": row[6], "error": row[7]}
            for row in generations
        ],
    }
