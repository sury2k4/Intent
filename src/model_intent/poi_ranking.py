from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .poi_schemas import ExtractedProperty, PoiHit, PoiRecord, PoiScoreBreakdown

DEFAULT_POLICY = {
    "schema_version": 1,
    "policy_version": "poi-ranking-poc-v1",
    "weights": {"preferred_property": 0.35, "rating": 0.35, "review_volume": 0.10, "distance": 0.20},
    "bayesian_prior_reviews": 20,
    "fallback_prior_rating": 3.8,
    "review_count_cap": 1000,
    "mock_preferred_source_factor": 0.65,
    "distance_decay_km": 5.0,
}


def load_ranking_policy(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_POLICY)
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ranking policy must be an object")
    policy = dict(DEFAULT_POLICY)
    policy.update(payload)
    weights = dict(DEFAULT_POLICY["weights"])
    weights.update(payload.get("weights") or {})
    policy["weights"] = weights
    total = sum(float(value) for value in weights.values())
    if not math.isfinite(total) or total <= 0 or any(
        not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()
    ):
        raise ValueError("ranking weights must be finite and non-negative with a positive sum")
    schema_version = int(policy.get("schema_version", 1))
    if schema_version >= 3:
        ranking_mode = "OSM-only" if schema_version == 3 else "mixed-evidence"
        if float(weights.get("rating", 0)) != 0 or float(weights.get("review_volume", 0)) != 0:
            raise ValueError(f"{ranking_mode} ranking must give rating and review_volume zero weight")
        prior_keys = ("bayesian_prior_reviews", "fallback_prior_rating", "review_count_cap")
        if any(key in payload for key in prior_keys):
            raise ValueError(f"{ranking_mode} ranking must not configure synthetic rating priors")
        for key in prior_keys:
            policy.pop(key, None)
    elif float(policy["bayesian_prior_reviews"]) <= 0:
        raise ValueError("bayesian_prior_reviews must be positive")
    for key in ("mock_rating_factor", "mock_review_volume_factor", "mock_preferred_source_factor"):
        factor = float(policy.get(key, 1.0))
        if not math.isfinite(factor) or not 0 <= factor <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    return policy


def haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    radius = 6371.0088
    lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_lat = math.radians(latitude_b - latitude_a)
    delta_lon = math.radians(longitude_b - longitude_a)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    return radius * 2 * math.asin(min(1.0, math.sqrt(value)))


def _property_value(record: PoiRecord, property_id: str) -> tuple[Any, str, str]:
    if property_id not in record.effective_properties:
        return None, "unknown", "unknown"
    assertion = record.effective_properties[property_id]
    if isinstance(assertion, Mapping):
        return assertion.get("value"), str(assertion.get("status", "unknown")), str(assertion.get("source", "unknown"))
    return assertion, "known", "unknown"


def satisfies_constraint(record: PoiRecord, constraint: ExtractedProperty, *, verified_only: bool = False) -> bool:
    if constraint.property_id == "rating":
        value = record.rating
        status = "known" if value is not None else "unknown"
        source = record.rating_source
    else:
        value, status, source = _property_value(record, constraint.property_id)
    if status != "known" or value is None or (verified_only and source != "osm"):
        return False
    expected = constraint.value
    if constraint.operator == "eq":
        return value == expected
    if constraint.operator in {"gt", "gte", "lt", "lte"}:
        try:
            actual_order = float(value)
            expected_order = float(expected)
        except (TypeError, ValueError):
            ordinal = {"low": 0, "medium": 1, "high": 2}
            if str(value) not in ordinal or str(expected) not in ordinal:
                return False
            actual_order = float(ordinal[str(value)])
            expected_order = float(ordinal[str(expected)])
        if constraint.operator == "gt":
            return actual_order > expected_order
        if constraint.operator == "gte":
            return actual_order >= expected_order
        if constraint.operator == "lt":
            return actual_order < expected_order
        return actual_order <= expected_order
    if constraint.operator == "one_of":
        return value in expected
    if constraint.operator == "contains":
        return str(expected).casefold() in str(value).casefold()
    raise ValueError(f"unsupported POI constraint operator: {constraint.operator}")


def satisfies_exclusion(record: PoiRecord, constraint: ExtractedProperty, *, verified_only: bool = False) -> bool:
    if not verified_only:
        # Demo mode treats unknown evidence as eligible; this is not a verified negative.
        return not satisfies_constraint(record, constraint)
    if constraint.property_id == "rating":
        known = record.rating is not None and record.rating_source == "osm"
    else:
        value, status, source = _property_value(record, constraint.property_id)
        known = status == "known" and value is not None and source == "osm"
    return known and not satisfies_constraint(record, constraint, verified_only=True)


def filter_pois(
    records: Iterable[PoiRecord],
    constraints: Sequence[ExtractedProperty],
    *,
    origin: tuple[float, float] | None = None,
    radius_km: float | None = None,
    verified_only: bool = False,
) -> tuple[list[PoiRecord], dict[str, Any]]:
    required = [row for row in constraints if row.mode == "required" and row.applied]
    excluded = [row for row in constraints if row.mode == "excluded" and row.applied]
    output: list[PoiRecord] = []
    counts = {"input": 0, "required_rejected": 0, "excluded_rejected": 0, "distance_rejected": 0}
    for record in records:
        counts["input"] += 1
        if required and not all(satisfies_constraint(record, row, verified_only=verified_only) for row in required):
            counts["required_rejected"] += 1
            continue
        if not all(satisfies_exclusion(record, row, verified_only=verified_only) for row in excluded):
            counts["excluded_rejected"] += 1
            continue
        if origin is not None and radius_km is not None:
            distance = haversine_km(origin[0], origin[1], record.latitude, record.longitude)
            if distance > radius_km:
                counts["distance_rejected"] += 1
                continue
        output.append(record)
    counts["output"] = len(output)
    return output, counts


def _preferred_score(
    record: PoiRecord,
    constraints: Sequence[ExtractedProperty],
    policy: Mapping[str, Any],
    *,
    verified_only: bool = False,
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    preferred = [
        row for row in constraints if row.mode == "preferred" and row.applied
        and (int(policy.get("schema_version", 1)) < 3 or row.property_id != "rating")
    ]
    if not preferred:
        return 1.0, (), ()
    source_factor = float(policy.get("mock_preferred_source_factor", 0.65))
    weighted_total = 0.0
    weighted_match = 0.0
    matched: list[str] = []
    unknown: list[str] = []
    for row in preferred:
        weight = max(0.0, float(row.confidence))
        weighted_total += weight
        if satisfies_constraint(record, row, verified_only=verified_only):
            value, _status, source = _property_value(record, row.property_id)
            factor = source_factor if source == "mock" else 1.0
            weighted_match += weight * factor
            matched.append(row.property_id)
        else:
            value, status, _source = _property_value(record, row.property_id)
            if status != "known" or value is None:
                unknown.append(row.property_id)
    return (weighted_match / weighted_total if weighted_total else 0.0), tuple(sorted(set(matched))), tuple(sorted(set(unknown)))


def _distance_score(distance_km: float | None, policy: Mapping[str, Any]) -> float:
    if distance_km is None:
        return 1.0
    decay = max(0.001, float(policy.get("distance_decay_km", 5.0)))
    return math.exp(-max(0.0, distance_km) / decay)


def rank_pois(
    records: Iterable[PoiRecord],
    constraints: Sequence[ExtractedProperty] = (),
    *,
    policy: Mapping[str, Any] | None = None,
    origin: tuple[float, float] | None = None,
    radius_km: float | None = None,
    top_k: int = 20,
    verified_only: bool = False,
) -> tuple[tuple[PoiHit, ...], dict[str, Any]]:
    policy = policy or DEFAULT_POLICY
    osm_only = int(policy.get("schema_version", 1)) >= 3
    verified_only = verified_only or osm_only
    if osm_only and (
        float(policy["weights"].get("rating", 0)) != 0
        or float(policy["weights"].get("review_volume", 0)) != 0
    ):
        raise ValueError("OSM-only ranking must give rating and review_volume zero weight")
    candidate_records = tuple(records)
    if osm_only:
        # Immutable content hashes must match the dataset. Refuse a mixed/older mock
        # generation rather than silently editing assertions in a search response.
        if any(
            record.properties_mock or record.rating_source == "mock"
            or (record.rating_source != "osm" and (record.rating is not None or record.review_count))
            or any(not isinstance(assertion, Mapping) or assertion.get("source") != "osm"
                   for assertion in record.effective_properties.values())
            for record in candidate_records
        ):
            raise ValueError("OSM-only ranking cannot serve mock or unverified POI assertions")
        category_mean = None
    else:
        real_ratings = [float(record.rating) for record in candidate_records if record.rating is not None and record.rating_source != "mock"]
        ratings = real_ratings or [float(record.rating) for record in candidate_records if record.rating is not None]
        category_mean = sum(ratings) / len(ratings) if ratings else float(policy.get("fallback_prior_rating", 3.8))
    filtered, filter_counts = filter_pois(
        candidate_records, constraints, origin=origin, radius_km=radius_km, verified_only=verified_only,
    )
    required = [row for row in constraints if row.mode == "required" and row.applied]
    excluded = [row for row in constraints if row.mode == "excluded" and row.applied]
    verified_count = sum(
        all(satisfies_constraint(record, row, verified_only=True) for row in required)
        and all(satisfies_exclusion(record, row, verified_only=True) for row in excluded)
        for record in filtered
    ) if required or excluded else None
    filter_counts["verified_required_count"] = verified_count
    filter_counts["mock_supported_count"] = len(filtered) - verified_count if verified_count is not None else None
    prior_reviews = float(policy.get("bayesian_prior_reviews", 20)) if not osm_only else 0.0
    rating_weight = float(policy["weights"].get("rating", 0.35))
    preferred_weight = float(policy["weights"].get("preferred_property", 0.35))
    volume_weight = float(policy["weights"].get("review_volume", 0.10))
    distance_weight = float(policy["weights"].get("distance", 0.20))
    v2 = int(policy.get("schema_version", 1)) >= 2
    has_preference = any(
        row.mode == "preferred" and row.applied and (not osm_only or row.property_id != "rating")
        for row in constraints
    )
    active_weight = (
        rating_weight + volume_weight
        + (preferred_weight if has_preference else 0)
        + (distance_weight if origin is not None else 0)
    ) if v2 else 1.0
    active_weight = active_weight or 1.0
    review_cap = max(1, int(policy.get("review_count_cap", 1000)))
    hits: list[PoiHit] = []
    for record in filtered:
        if osm_only:
            adjusted = None
            rating_score = 0.0
            volume_score = 0.0
            rating_factor = review_factor = 0.0
        else:
            review_count = max(0, int(record.review_count or 0))
            rating = float(record.rating if record.rating is not None else category_mean)
            adjusted = ((review_count * rating) + (prior_reviews * category_mean)) / (review_count + prior_reviews)
            mock_rating = record.rating_source == "mock"
            rating_factor = float(policy.get("mock_rating_factor", 1.0)) if mock_rating else 1.0
            review_factor = float(policy.get("mock_review_volume_factor", 1.0)) if mock_rating else 1.0
            adjusted = category_mean + (adjusted - category_mean) * rating_factor
            rating_score = (adjusted - 1.0) / 4.0
            volume_score = (math.log1p(min(review_count, review_cap)) / math.log1p(review_cap)) * review_factor
        distance = None
        if origin is not None:
            distance = haversine_km(origin[0], origin[1], record.latitude, record.longitude)
        distance_score = _distance_score(distance, policy)
        preferred_score, matched, unknown = _preferred_score(record, constraints, policy, verified_only=verified_only)
        final = (
            (preferred_weight * preferred_score if has_preference or not v2 else 0)
            + rating_weight * rating_score
            + volume_weight * volume_score
            + (distance_weight * distance_score if origin is not None or not v2 else 0)
        ) / active_weight
        breakdown = PoiScoreBreakdown(
            preferred_property_score=preferred_score,
            rating_score=rating_score,
            review_volume_score=volume_score,
            distance_score=distance_score,
            adjusted_rating=adjusted,
            final_score=final,
            rating_source_factor=rating_factor,
            review_source_factor=review_factor,
        )
        required_sources = {
            row.property_id: (record.rating_source if row.property_id == "rating"
                              else _property_value(record, row.property_id)[2])
            for row in required
        }
        hits.append(PoiHit(record, breakdown, distance, matched, unknown, required_sources))
    if osm_only:
        # Ratings and review counts are neither ranking evidence nor tie-breakers.
        hits.sort(key=lambda hit: (
            -hit.score.final_score,
            (hit.poi.name or hit.poi.display_name).casefold(),
            hit.poi.poi_id,
        ))
    else:
        hits.sort(
            key=lambda hit: (
                -hit.score.final_score,
                -hit.score.adjusted_rating,
                -int(hit.poi.review_count or 0),
                (hit.poi.name or hit.poi.display_name).casefold(),
                hit.poi.poi_id,
            )
        )
    limited = tuple(hits[: max(0, int(top_k))])
    filter_counts.update({"category_mean_rating": category_mean, "returned": len(limited)})
    return limited, filter_counts
