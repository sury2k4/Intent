from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PropertyAssertion:
    """A normalized POI property with explicit provenance."""

    value: Any = None
    status: str = "unknown"  # known, unknown, inapplicable
    source: str = "osm"  # osm, mock, derived
    raw: Any = None
    confidence: float | None = None
    mock_schema_version: str | None = None
    mock_seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": self.value,
            "status": self.status,
            "source": self.source,
        }
        if self.raw is not None:
            payload["raw"] = self.raw
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.mock_schema_version is not None:
            payload["mock_schema_version"] = self.mock_schema_version
        if self.mock_seed is not None:
            payload["mock_seed"] = self.mock_seed
        return payload


@dataclass(frozen=True)
class ExtractedProperty:
    """A property constraint extracted from a user query."""

    property_id: str
    mode: str  # required, preferred, excluded
    operator: str = "eq"
    value: Any = True
    evidence_text: str = ""
    start_offset: int | None = None
    end_offset: int | None = None
    matched_alias: str = ""
    cue: str = ""
    confidence: float = 1.0
    applied: bool = True
    ignored_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "mode": self.mode,
            "operator": self.operator,
            "value": self.value,
            "evidence_text": self.evidence_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "matched_alias": self.matched_alias,
            "cue": self.cue,
            "confidence": self.confidence,
            "applied": self.applied,
            "ignored_reason": self.ignored_reason,
        }


@dataclass(frozen=True)
class PropertyExtraction:
    query: str
    schema_version: str
    properties: tuple[ExtractedProperty, ...] = ()
    warnings: tuple[str, ...] = ()
    failed_open: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "schema_version": self.schema_version,
            "properties": [item.as_dict() for item in self.properties],
            "warnings": list(self.warnings),
            "failed_open": self.failed_open,
        }


@dataclass(frozen=True)
class PoiRecord:
    poi_id: str
    osm_type: str
    osm_id: int
    category_id: str
    category_code: str
    name: str | None
    display_name: str
    names: Mapping[str, str | None]
    latitude: float
    longitude: float
    location_method: str
    mapping_rule_id: str
    mapping_version: str
    osm_tags: Mapping[str, Any] = field(default_factory=dict)
    properties_osm: Mapping[str, Any] = field(default_factory=dict)
    properties_mock: Mapping[str, Any] = field(default_factory=dict)
    effective_properties: Mapping[str, Any] = field(default_factory=dict)
    rating: float | None = None
    review_count: int = 0
    rating_source: str = "mock"
    rating_mock_schema_version: str | None = None
    rating_mock_seed: int | None = None
    content_hash: str = ""
    source_version: int | None = None
    source_timestamp: str | None = None
    category_source: str | None = None
    category_evidence: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "poi_id": self.poi_id,
            "osm": {
                "type": self.osm_type,
                "id": self.osm_id,
                "version": self.source_version,
                "timestamp": self.source_timestamp,
                "tags": dict(self.osm_tags),
            },
            "category_id": self.category_id,
            "category_code": self.category_code,
            "mapping": {
                "version": self.mapping_version,
                "rule_id": self.mapping_rule_id,
            },
            "name": self.name,
            "display_name": self.display_name,
            "names": dict(self.names),
            "location": {
                "lat": self.latitude,
                "lon": self.longitude,
                "method": self.location_method,
            },
            "properties": {
                "osm": dict(self.properties_osm),
                "mock": dict(self.properties_mock),
                "effective": dict(self.effective_properties),
            },
            "rating": {
                "value": self.rating,
                "review_count": self.review_count,
                "source": self.rating_source,
                **({
                    "mock_schema_version": self.rating_mock_schema_version,
                    "mock_seed": self.rating_mock_seed,
                } if self.rating_source == "mock" else {}),
            },
            "content_hash": self.content_hash,
        }
        # Legacy v3 content hashes are calculated over the old wire shape.
        if self.category_source is not None:
            payload["category_source"] = self.category_source
        if self.category_evidence is not None:
            payload["category_evidence"] = dict(self.category_evidence)
        return payload


@dataclass(frozen=True)
class PoiScoreBreakdown:
    preferred_property_score: float
    rating_score: float
    review_volume_score: float
    distance_score: float
    adjusted_rating: float | None
    final_score: float
    rating_source_factor: float = 1.0
    review_source_factor: float = 1.0

    def as_dict(self) -> dict[str, float | None]:
        return {
            "preferred_property_score": self.preferred_property_score,
            "rating_score": self.rating_score,
            "review_volume_score": self.review_volume_score,
            "distance_score": self.distance_score,
            "adjusted_rating": self.adjusted_rating,
            "final_score": self.final_score,
            "rating_source_factor": self.rating_source_factor,
            "review_source_factor": self.review_source_factor,
        }


@dataclass(frozen=True)
class PoiHit:
    poi: PoiRecord
    score: PoiScoreBreakdown
    distance_km: float | None = None
    matched_properties: tuple[str, ...] = ()
    unknown_properties: tuple[str, ...] = ()
    required_property_sources: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = self.poi.as_dict()
        payload["distance_km"] = self.distance_km
        payload["score"] = self.score.as_dict()
        payload["matched_properties"] = list(self.matched_properties)
        payload["unknown_properties"] = list(self.unknown_properties)
        payload["required_property_sources"] = dict(self.required_property_sources)
        return payload


@dataclass(frozen=True)
class PoiSearchResult:
    status: str
    category_id: str | None
    generation_id: int | None = None
    category_candidate_count: int = 0
    hard_filter_match_count: int = 0
    verified_required_count: int | None = None
    mock_supported_count: int | None = None
    verified_only: bool = False
    effective_radius_km: float | None = None
    returned_count: int = 0
    ranking_policy_version: str | None = None
    ranking_schema_version: int | None = None
    hits: tuple[PoiHit, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "category_id": self.category_id,
            "generation_id": self.generation_id,
            "category_candidate_count": self.category_candidate_count,
            "hard_filter_match_count": self.hard_filter_match_count,
            "verified_required_count": self.verified_required_count,
            "mock_supported_count": self.mock_supported_count,
            "verified_only": self.verified_only,
            "effective_radius_km": self.effective_radius_km,
            "returned_count": self.returned_count,
            "ranking_policy_version": self.ranking_policy_version,
            "warnings": list(self.warnings),
            "hits": [hit.as_dict() for hit in self.hits],
        }
        if self.ranking_schema_version is not None:
            payload["ranking_schema_version"] = self.ranking_schema_version
        return payload
