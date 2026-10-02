from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from .poi_schemas import PropertyAssertion

MOCK_SCHEMA_VERSION = "poi-mock-v1"


def _digest_int(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def mock_property(
    property_id: str,
    definition: Mapping[str, Any],
    *,
    poi_id: str,
    category_id: str,
    category_code: str | None = None,
    seed: int,
    schema_version: str = MOCK_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Generate a stable mock assertion independent of traversal order."""
    value_type = str(definition["value_type"])
    digest = _digest_int(schema_version, seed, poi_id, category_id, property_id)
    if value_type == "boolean":
        rates = definition.get("mock_true_rate_by_code") or {}
        rate = float(rates.get(category_code, definition.get("mock_true_rate", 0.65)))
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"mock true-rate for {property_id} must be between 0 and 1")
        value = (digest / float(2**64)) < rate
    elif value_type == "enum":
        values = list(definition.get("values", [])) or ["yes", "no"]
        value = values[digest % len(values)]
    elif value_type == "ordinal":
        values = list(definition.get("values", [])) or ["low", "medium", "high"]
        value = values[min(len(values) - 1, (digest % 100) // max(1, 100 // len(values)))]
    else:
        raise ValueError(f"cannot fabricate a value for non-categorical property {property_id}")
    return PropertyAssertion(
        value=value,
        status="known",
        source="mock",
        confidence=0.5,
        mock_schema_version=schema_version,
        mock_seed=seed,
    ).as_dict()


def mock_rating(
    poi_id: str,
    category_id: str,
    seed: int,
    schema_version: str = MOCK_SCHEMA_VERSION,
) -> tuple[float, int]:
    digest = _digest_int(schema_version, seed, poi_id, category_id, "rating")
    unit = digest / float(2**64)
    # Skew synthetic ratings toward the upper-middle while retaining plausible
    # low-rated outliers for POC ranking tests.
    rating = round(min(5.0, max(1.0, 3.0 + (2.0 * unit**0.55))), 1)
    volume_digest = _digest_int(schema_version, seed, poi_id, category_id, "review_count")
    volume = volume_digest / float(2**64)
    review_count = max(1, int(math.exp(volume * math.log(2501))))
    return rating, min(review_count, 2500)
