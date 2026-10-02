from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .db import Database
from .poi_ranking import filter_pois
from .poi_schemas import ExtractedProperty, PoiRecord


class PoiRepository:
    """Small artifact-backed repository used by the POC and network-free tests."""

    def __init__(self, records: Iterable[PoiRecord], *, generation_id: int | None = None, source: str = "artifact") -> None:
        self.records = tuple(records)
        self.generation_id = generation_id
        self.source = source
        self.database: Database | None = None
        self.owns_database = False
        self.taxonomy: Any = None
        self.mapping_version: str | None = None

    @classmethod
    def from_jsonl(cls, path: str | Path, *, generation_id: int | None = None) -> "PoiRepository":
        records: list[PoiRecord] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            osm = payload.get("osm") or {}
            location = payload.get("location") or {}
            rating = payload.get("rating") or {}
            records.append(PoiRecord(
                poi_id=str(payload["poi_id"]),
                osm_type=str(osm.get("type", "unknown")),
                osm_id=int(osm.get("id", 0)),
                category_id=str(payload["category_id"]),
                category_code=str(payload.get("category_code", "")),
                name=payload.get("name"),
                display_name=str(payload.get("display_name", payload.get("name") or payload["poi_id"])),
                names=payload.get("names") or {},
                latitude=float(location["lat"]),
                longitude=float(location["lon"]),
                location_method=str(location.get("method", "unknown")),
                mapping_rule_id=str((payload.get("mapping") or {}).get("rule_id", "unknown")),
                mapping_version=str((payload.get("mapping") or {}).get("version", "unknown")),
                osm_tags=osm.get("tags") or {},
                properties_osm=(payload.get("properties") or {}).get("osm") or {},
                properties_mock=(payload.get("properties") or {}).get("mock") or {},
                effective_properties=(payload.get("properties") or {}).get("effective") or {},
                rating=float(rating["value"]) if rating.get("value") is not None else None,
                review_count=int(rating.get("review_count", 0)),
                rating_source=str(rating.get("source", "unknown")),
                rating_mock_schema_version=rating.get("mock_schema_version"),
                rating_mock_seed=int(rating["mock_seed"]) if rating.get("mock_seed") is not None else None,
                content_hash=str(payload.get("content_hash", "")),
                source_version=int(osm["version"]) if osm.get("version") is not None else None,
                source_timestamp=str(osm["timestamp"]) if osm.get("timestamp") else None,
                category_source=payload.get("category_source"),
                category_evidence=payload.get("category_evidence"),
            ))
        return cls(records, generation_id=generation_id)

    @classmethod
    def from_database(
        cls,
        database: Database,
        *,
        dataset_key: str = "hcmc-osm",
        generation_id: int | None = None,
        taxonomy_fingerprint: str | None = None,
        mapping_sha256: str | None = None,
        property_taxonomy_sha256: str | None = None,
        taxonomy: Any = None,
        owns_database: bool = False,
    ) -> "PoiRepository":
        with database.connection() as (connection, _):
            if generation_id is None:
                active = connection.execute(
                    "SELECT generation_id FROM active_poi_dataset_generation WHERE dataset_key=%s",
                    (dataset_key,),
                ).fetchone()
                if active is None:
                    raise RuntimeError(f"no active POI dataset generation for {dataset_key}")
                generation_id = int(active[0])
            metadata = connection.execute(
                """SELECT state, taxonomy_fingerprint, mapping_sha256,
                          property_taxonomy_sha256, mapping_version
                   FROM poi_dataset_generations WHERE id=%s AND dataset_key=%s""",
                (generation_id, dataset_key),
            ).fetchone()
            if metadata is None or str(metadata[0]) != "ready":
                raise RuntimeError(f"POI generation {generation_id} is missing or not ready")
            if taxonomy_fingerprint and str(metadata[1]) != taxonomy_fingerprint:
                raise RuntimeError("active POI generation taxonomy fingerprint does not match the category taxonomy")
            if mapping_sha256 and str(metadata[2]) != mapping_sha256:
                raise RuntimeError("active POI generation mapping hash does not match current configuration")
            if property_taxonomy_sha256 and str(metadata[3]) != property_taxonomy_sha256:
                raise RuntimeError("active POI generation property hash does not match current configuration")
        repository = cls((), generation_id=generation_id, source="postgres")
        repository.database = database
        repository.owns_database = owns_database
        repository.taxonomy = taxonomy
        repository.mapping_version = str(metadata[4])
        return repository

    def for_categories(
        self,
        category_ids: Iterable[str],
        *,
        cuisine_filter: Iterable[str] | None = None,
        always_include_category_id: str | None = None,
    ) -> tuple[PoiRecord, ...]:
        ids = tuple(dict.fromkeys(str(category_id) for category_id in category_ids))
        if not ids:
            return ()
        cuisines = tuple(sorted({str(value).casefold().strip() for value in cuisine_filter or () if str(value).strip()}))
        if cuisines and always_include_category_id not in ids:
            raise ValueError("always_include_category_id must be one of the selected categories")
        if len(ids) == 1 and not cuisines:
            return self.for_category(ids[0])
        if self.source == "postgres" and self.database is not None:
            from psycopg import sql

            placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in ids)
            query = sql.SQL(
                """
                SELECT poi_id, osm_type, osm_id, place_type_id, name, display_name,
                       names, latitude, longitude, location_method, mapping_rule_id,
                       osm_tags, properties_osm, properties_mock, effective_properties,
                       rating, review_count, rating_source, rating_mock_schema_version,
                       rating_mock_seed, content_hash, source_version, source_timestamp,
                       to_jsonb(pois)->>'category_source' AS category_source,
                       to_jsonb(pois)->'category_evidence' AS category_evidence
                FROM pois WHERE generation_id=%s AND place_type_id IN ({})
                """
            ).format(placeholders)
            params: tuple[Any, ...] = (self.generation_id, *(int(category_id) for category_id in ids))
            if cuisines:
                query += sql.SQL(
                    """ AND (place_type_id=%s OR EXISTS (
                        SELECT 1 FROM regexp_split_to_table(COALESCE(osm_tags->>'cuisine', ''), '[;,]') AS cuisine_tag(tag)
                        WHERE lower(btrim(cuisine_tag.tag)) = ANY(%s)
                    ))"""
                )
                params += (int(always_include_category_id), list(cuisines))
            query += sql.SQL(" ORDER BY poi_id")
            with self.database.connection() as (connection, _):
                rows = connection.execute(query, params).fetchall()
            return tuple(self._record_from_row(row) for row in rows)
        eligible = set(ids)
        if cuisines:
            accepted = set(cuisines)
            return tuple(
                record for record in self.records
                if record.category_id in eligible and (
                    record.category_id == always_include_category_id
                    or bool(accepted & {
                        part.strip().casefold()
                        for part in str(record.osm_tags.get("cuisine") or "").replace(",", ";").split(";")
                        if part.strip()
                    })
                )
            )
        return tuple(record for record in self.records if record.category_id in eligible)

    def _record_from_row(self, row: tuple[Any, ...]) -> PoiRecord:
        item_id = str(row[3])
        category_code = self.taxonomy.items[item_id].code if self.taxonomy and item_id in self.taxonomy.items else ""
        return PoiRecord(
            poi_id=str(row[0]), osm_type=str(row[1]), osm_id=int(row[2]),
            category_id=item_id, category_code=category_code, name=row[4],
            display_name=str(row[5]), names=row[6] or {}, latitude=float(row[7]),
            longitude=float(row[8]), location_method=str(row[9]),
            mapping_rule_id=str(row[10]), mapping_version=self.mapping_version or "unknown",
            osm_tags=row[11] or {}, properties_osm=row[12] or {},
            properties_mock=row[13] or {}, effective_properties=row[14] or {},
            rating=float(row[15]) if row[15] is not None else None,
            review_count=int(row[16]), rating_source=str(row[17]),
            rating_mock_schema_version=row[18],
            rating_mock_seed=int(row[19]) if row[19] is not None else None,
            content_hash=str(row[20]),
            source_version=int(row[21]) if row[21] is not None else None,
            source_timestamp=str(row[22]) if row[22] is not None else None,
            category_source=row[23] if len(row) > 23 else None,
            category_evidence=row[24] if len(row) > 24 else None,
        )

    def for_category(self, category_id: str) -> tuple[PoiRecord, ...]:
        if self.source == "postgres" and self.database is not None:
            with self.database.connection() as (connection, _):
                rows = connection.execute(
                    """
                    SELECT poi_id, osm_type, osm_id, place_type_id, name, display_name,
                           names, latitude, longitude, location_method, mapping_rule_id,
                           osm_tags, properties_osm, properties_mock, effective_properties,
                           rating, review_count, rating_source, rating_mock_schema_version,
                           rating_mock_seed, content_hash, source_version, source_timestamp,
                       to_jsonb(pois)->>'category_source' AS category_source,
                       to_jsonb(pois)->'category_evidence' AS category_evidence
                    FROM pois WHERE generation_id=%s AND place_type_id=%s
                    ORDER BY poi_id
                    """,
                    (self.generation_id, int(category_id)),
                ).fetchall()
            return tuple(self._record_from_row(row) for row in rows)
        return tuple(record for record in self.records if record.category_id == str(category_id))

    def candidates(
        self,
        category_id: str,
        constraints: Sequence[ExtractedProperty] = (),
        *,
        origin: tuple[float, float] | None = None,
        radius_km: float | None = None,
    ) -> tuple[list[PoiRecord], dict[str, Any]]:
        category_records = list(self.for_category(category_id))
        filtered, counts = filter_pois(category_records, constraints, origin=origin, radius_km=radius_km)
        counts["category_candidates"] = len(category_records)
        return filtered, counts

    def count(self) -> int:
        if self.source == "postgres" and self.database is not None:
            with self.database.connection() as (connection, _):
                return int(connection.execute(
                    "SELECT count(*) FROM pois WHERE generation_id=%s", (self.generation_id,)
                ).fetchone()[0])
        return len(self.records)

    def close(self) -> None:
        if self.owns_database and self.database is not None:
            self.database.close()
