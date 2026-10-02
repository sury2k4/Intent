from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .normalization import fold_vietnamese, normalize_text
from .poi_mocking import MOCK_SCHEMA_VERSION, mock_property, mock_rating
from .poi_properties import PropertyProfiles, load_property_definitions, load_property_profiles
from .poi_schemas import PoiRecord, PropertyAssertion
from .taxonomy import Taxonomy

PIPELINE_VERSION = "osm-poi-pipeline-v1"

# Keys with a place/facility meaning are admitted regardless of name. Linear
# transport/utility features are admitted only for the explicitly listed POIs.
POI_PRIMARY_KEYS = frozenset({
    "amenity", "shop", "tourism", "leisure", "office", "healthcare", "craft",
    "historic", "sport", "emergency", "government", "club", "public_transport",
})
POI_QUALIFIED_VALUES = {
    "aeroway": frozenset({"terminal", "helipad"}),
    "highway": frozenset({"bus_stop", "rest_area", "services"}),
    "railway": frozenset({"station", "halt", "tram_stop", "subway_entrance"}),
    "man_made": frozenset({"lighthouse", "tower", "water_tower", "pier"}),
    "natural": frozenset({"beach", "spring", "cave_entrance", "peak"}),
    "waterway": frozenset({"dock", "waterfall"}),
    "building": frozenset({"retail", "commercial", "hospital", "school", "train_station", "transportation"}),
    "landuse": frozenset({"cemetery", "recreation_ground"}),
}
SIGNATURE_MODIFIERS = frozenset({"religion", "isced:level", "brand", "cuisine", "station", "service", "vehicle"})


def candidate_tags(tags: Mapping[str, Any]) -> dict[str, str]:
    """Only place-like primary tags; never admit a road/barrier by mere presence."""
    return {
        key: str(tags[key]).strip()
        for key in sorted(POI_PRIMARY_KEYS | POI_QUALIFIED_VALUES.keys())
        if key in tags and str(tags[key]).strip() and (
            key in POI_PRIMARY_KEYS or _norm_tag(tags[key]) in POI_QUALIFIED_VALUES[key]
        )
    }


def candidate_signature(tags: Mapping[str, Any]) -> str:
    """Auditable signature including modifiers that affect category precedence."""
    primary = candidate_tags(tags)
    modifiers = {
        key: str(tags[key]).strip()
        for key in sorted(SIGNATURE_MODIFIERS)
        if key in tags and str(tags[key]).strip() and (
            key != "cuisine" or "amenity" in primary and _norm_tag(primary["amenity"]) == "restaurant"
        )
    }
    return "|".join(f"{key}={value}" for key, value in sorted({**primary, **modifiers}.items()))


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def load_property_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    return load_property_definitions(path)


def _leaf_ids(taxonomy: Taxonomy) -> set[str]:
    return {item.id for item in taxonomy.document.items if not taxonomy.has_children(item.id)}


def load_mapping_catalog(path: str | Path, taxonomy: Taxonomy) -> dict[str, Any]:
    payload = load_json(path)
    declared_taxonomy = payload.get("taxonomy_version")
    if declared_taxonomy != taxonomy.document.taxonomy_version:
        raise ValueError(
            f"OSM mapping taxonomy mismatch: {declared_taxonomy} != {taxonomy.document.taxonomy_version}"
        )
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError("mapping catalog requires a rules list")
    ids: set[str] = set()
    priorities: set[int] = set()
    leaves = _leaf_ids(taxonomy)
    for row in rules:
        if not isinstance(row, dict):
            raise ValueError("mapping rules must be objects")
        rule_id = str(row.get("id", "")).strip()
        category_id = str(row.get("category_id", "")).strip()
        priority = int(row.get("priority", 0))
        if not rule_id or rule_id in ids:
            raise ValueError(f"duplicate or missing mapping rule id: {rule_id!r}")
        if priority in priorities:
            raise ValueError(f"duplicate mapping rule priority: {priority}")
        if category_id not in leaves:
            raise ValueError(f"mapping rule {rule_id} targets unknown/non-leaf category {category_id}")
        ids.add(rule_id)
        priorities.add(priority)
    unmapped = payload.get("intentionally_unmapped", {})
    if not isinstance(unmapped, dict):
        raise ValueError("intentionally_unmapped must be an object")
    if str(payload.get("mapping_version", "")).endswith("v3") and "*" in unmapped:
        raise ValueError("v3 mappings require explicit leaf dispositions, not a wildcard")
    excluded = payload.get("intentionally_excluded_signatures", {})
    if not isinstance(excluded, dict) or "*" in excluded:
        raise ValueError("signature exclusions must be explicit, without a wildcard")
    valid_dispositions = {"intentionally_excluded_ambiguous", "intentionally_excluded_unmodelled"}
    for signature, disposition in excluded.items():
        if not signature or not isinstance(disposition, dict) or (
            disposition.get("disposition") not in valid_dispositions or not disposition.get("reason")
        ):
            raise ValueError(f"invalid explicit signature exclusion: {signature}")
    payload["rules"] = sorted(rules, key=lambda row: (-int(row.get("priority", 0)), str(row.get("id", ""))))
    payload["_leaf_ids"] = sorted(leaves, key=lambda value: int(value))
    return payload


def mapping_coverage(catalog: Mapping[str, Any], taxonomy: Taxonomy) -> dict[str, Any]:
    leaves = _leaf_ids(taxonomy)
    mapped = {str(row["category_id"]) for row in catalog.get("rules", [])}
    unmapped = catalog.get("intentionally_unmapped", {}) or {}
    wildcard = "*" in unmapped
    explicit_unmapped = {str(key) for key in unmapped if key != "*"}
    missing = sorted(leaves - mapped - explicit_unmapped, key=lambda value: int(value))
    if wildcard:
        explicit_unmapped |= leaves - mapped
        missing = []
    return {
        "leaf_count": len(leaves),
        "mapped_category_ids": sorted(mapped, key=lambda value: int(value)),
        "intentionally_unmapped_category_ids": sorted(explicit_unmapped, key=lambda value: int(value)),
        "missing_disposition_category_ids": missing,
        "complete": not missing,
    }


def validate_catalogs(
    taxonomy: Taxonomy,
    mapping_path: str | Path,
    property_path: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    mapping = load_mapping_catalog(mapping_path, taxonomy)
    profiles = load_property_profiles(property_path, taxonomy)
    properties = profiles.definitions
    coverage = mapping_coverage(mapping, taxonomy)
    if require_complete and not coverage["complete"]:
        raise ValueError(
            "mapping catalog has leaves without a mapped/unmapped disposition: "
            + ", ".join(coverage["missing_disposition_category_ids"])
        )
    if profiles.minimum_known:
        profiles.validate_coverage(taxonomy, set(coverage["mapped_category_ids"]))
    return {"mapping": mapping, "properties": properties, "coverage": coverage, "profiles": profiles}


def _norm_tag(value: Any) -> str:
    return normalize_text(fold_vietnamese(str(value)))


def _tag_value(tags: Mapping[str, Any], key: str) -> str:
    return str(tags.get(key, "")).strip()


def _predicate_matches(tags: Mapping[str, Any], predicate: Mapping[str, Any]) -> bool:
    if "tag" not in predicate:
        # A compact {"amenity": "parking"} predicate is accepted for hand-authored POC rules.
        if len(predicate) == 1:
            key, value = next(iter(predicate.items()))
            return _norm_tag(tags.get(str(key), "")) == _norm_tag(value)
        return False
    key = str(predicate.get("tag", ""))
    op = str(predicate.get("op", "eq"))
    actual = _tag_value(tags, key)
    normalized_actual = _norm_tag(actual)
    if op == "present":
        return bool(actual)
    if op == "eq":
        return normalized_actual == _norm_tag(predicate.get("value", ""))
    if op == "in":
        return normalized_actual in {_norm_tag(item) for item in predicate.get("values", [])}
    if op == "token":
        tokens = {_norm_tag(item) for item in re.split(r"[;,]", actual) if item.strip()}
        return _norm_tag(predicate.get("value", "")) in tokens
    if op == "any_token":
        tokens = {_norm_tag(item) for item in re.split(r"[;,]", actual) if item.strip()}
        return bool(tokens & {_norm_tag(item) for item in predicate.get("values", [])})
    if op == "normalized_eq":
        return _norm_tag(actual) == _norm_tag(predicate.get("value", ""))
    if op == "not_eq":
        return normalized_actual != _norm_tag(predicate.get("value", ""))
    raise ValueError(f"unsupported OSM mapping predicate operator: {op}")


def _match_expression(tags: Mapping[str, Any], expression: Any) -> bool:
    if expression is None:
        return True
    if not isinstance(expression, Mapping):
        raise ValueError("mapping expressions must be objects")
    checks: list[bool] = []
    if "all" in expression:
        checks.append(all(_match_expression(tags, item) for item in expression["all"]))
    if "any" in expression:
        checks.append(any(_match_expression(tags, item) for item in expression["any"]))
    if "none" in expression:
        checks.append(not any(_match_expression(tags, item) for item in expression["none"]))
    if checks:
        return all(checks)
    return _predicate_matches(tags, expression)


def match_mapping(tags: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any] | None:
    for rule in catalog.get("rules", []):
        if _match_expression(tags, rule.get("match")):
            return rule
    return None


def _parse_bool(value: Any) -> bool | None:
    normalized = _norm_tag(value)
    if normalized in {"yes", "true", "1", "y", "on"}:
        return True
    if normalized in {"no", "false", "0", "n", "off"}:
        return False
    return None


def _property_value(property_def: Mapping[str, Any], tags: Mapping[str, Any]) -> tuple[bool, PropertyAssertion | None]:
    key = str(property_def["id"])
    source_tags = [str(item) for item in property_def.get("osm_tags", [])]
    if key == "fast_charging":
        raw_values = {item: tags[item] for item in source_tags if item in tags}
        if not raw_values:
            return False, None
        watts: list[float] = []
        all_parsed = True
        for raw in raw_values.values():
            values = re.findall(r"(\d+(?:[\.,]\d+)?)\s*(kW|MW|W)\b", str(raw), flags=re.IGNORECASE)
            if not values:
                all_parsed = False
            watts.extend(
                float(amount.replace(",", ".")) * {"w": 1, "kw": 1000, "mw": 1000000}[unit.lower()]
                for amount, unit in values
            )
        if any(value >= 50000 for value in watts):
            parsed = True
        elif watts and all_parsed:
            parsed = False
        else:
            parsed = None
        return True, PropertyAssertion(
            value=parsed, status="known" if parsed is not None else "unknown", source="osm", raw=raw_values,
        )
    raw_key = next((item for item in source_tags if item in tags), None)
    if raw_key is None:
        return False, None
    raw = tags[raw_key]
    value_type = str(property_def["value_type"])
    if value_type == "boolean":
        normalized = _norm_tag(raw)
        if key == "free_entry" and raw_key == "fee":
            fee = _parse_bool(raw)
            parsed = None if fee is None else not fee
        elif key == "disabled_parking" and raw_key == "capacity:disabled":
            try:
                parsed = int(raw) > 0
            except (TypeError, ValueError):
                parsed = None
        elif key == "convenience_shop" and raw_key == "shop":
            parsed = normalized == "convenience" if normalized else None
        elif key == "reception_24h" and raw_key == "reception:opening_hours":
            parsed = True if normalized == "24 7" else None
        elif key == "internet_access" and normalized in {"wlan", "customers"}:
            parsed = True
        else:
            parsed = _parse_bool(raw)
        if parsed is None:
            return True, PropertyAssertion(value=None, status="unknown", source="osm", raw={raw_key: raw})
        return True, PropertyAssertion(value=parsed, status="known", source="osm", raw={raw_key: raw})
    if value_type == "enum":
        normalized = _norm_tag(raw)
        values = {_norm_tag(item) for item in property_def.get("values", [])}
        if normalized in values:
            return True, PropertyAssertion(value=normalized, status="known", source="osm", raw={raw_key: raw})
        return True, PropertyAssertion(value=None, status="unknown", source="osm", raw={raw_key: raw})
    if value_type == "ordinal":
        normalized = _norm_tag(raw)
        if normalized in {"low", "medium", "high"}:
            return True, PropertyAssertion(value=normalized, status="known", source="osm", raw={raw_key: raw})
        return True, PropertyAssertion(value=None, status="unknown", source="osm", raw={raw_key: raw})
    return True, PropertyAssertion(value=str(raw), status="known", source="osm", raw={raw_key: raw})


def normalize_osm_properties(tags: Mapping[str, Any], property_catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key, definition in sorted(property_catalog.items()):
        _, assertion = _property_value(definition, tags)
        if assertion is not None:
            output[key] = assertion.as_dict()
    return output


def _mock_properties(
    *,
    category_code: str,
    poi_id: str,
    category_id: str,
    real_properties: Mapping[str, Any],
    property_catalog: Mapping[str, Mapping[str, Any]],
    seed: int,
    mock_schema_version: str = MOCK_SCHEMA_VERSION,
    applicable_ids: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    candidate_ids = sorted(applicable_ids) if applicable_ids is not None else sorted(property_catalog)
    for key in candidate_ids:
        definition = property_catalog[key]
        applicable = applicable_ids is not None or category_code in definition.get("mock_applicable_codes", ())
        if not applicable or key in real_properties or definition.get("value_type") == "string":
            continue
        output[key] = mock_property(
            key,
            definition,
            poi_id=poi_id,
            category_id=category_id,
            category_code=category_code,
            seed=seed,
            schema_version=mock_schema_version,
        )
    return output


def _extract_name(tags: Mapping[str, Any]) -> tuple[str | None, dict[str, str | None]]:
    name = tags.get("name:vi") or tags.get("name") or tags.get("official_name") or tags.get("name:en")
    names = {
        "default": str(tags.get("name")) if tags.get("name") else None,
        "vi": str(tags.get("name:vi")) if tags.get("name:vi") else None,
        "en": str(tags.get("name:en")) if tags.get("name:en") else None,
    }
    return (str(name).strip() if name else None), names


class Boundary:
    def __init__(self, geometry: Any):
        self.geometry = geometry
        self._shape = None
        try:
            from shapely.geometry import shape

            self._shape = shape(geometry)
            self._bounds = self._shape.bounds
        except ImportError:
            self._shape = None
            self._bounds = None

    @classmethod
    def from_geojson(cls, path: str | Path) -> "Boundary":
        payload = load_json(path)
        if payload.get("type") == "FeatureCollection":
            features = payload.get("features") or []
            if not features:
                raise ValueError("boundary FeatureCollection has no features")
            geometry = features[0].get("geometry")
        elif payload.get("type") == "Feature":
            geometry = payload.get("geometry")
        else:
            geometry = payload
        if not isinstance(geometry, dict) or not geometry.get("type"):
            raise ValueError("invalid boundary GeoJSON geometry")
        return cls(geometry)

    def representative_geometry(self, geometry: Mapping[str, Any]) -> tuple[float, float] | None:
        if self._shape is not None:
            # Reject nationwide areas outside HCMC before constructing/intersecting
            # expensive Shapely geometries; the bbox check is conservative.
            def vertices(value: Any) -> Iterator[tuple[float, float]]:
                if isinstance(value, (tuple, list)):
                    if len(value) >= 2 and all(isinstance(v, (float, int)) for v in value[:2]):
                        yield float(value[0]), float(value[1])
                    else:
                        for part in value:
                            yield from vertices(part)

            left, bottom, right, top = self._bounds
            coordinates = list(vertices(geometry.get("coordinates", [])))
            if coordinates and (
                max(x for x, _ in coordinates) < left or min(x for x, _ in coordinates) > right
                or max(y for _, y in coordinates) < bottom or min(y for _, y in coordinates) > top
            ):
                return None
            try:
                from shapely.geometry import shape

                feature = shape(geometry)
                if not feature.is_valid:
                    feature = feature.buffer(0)
                clipped = feature.intersection(self._shape)
                if clipped.is_empty:
                    return None
                point = clipped.representative_point()
                return float(point.x), float(point.y)
            except Exception:
                return None
        coordinates = _polygon_coordinates(geometry)
        return self.representative_point(coordinates) if coordinates else None

    def representative_point(self, coordinates: list[tuple[float, float]]) -> tuple[float, float]:
        if not coordinates:
            raise ValueError("cannot find a representative point for empty geometry")
        if self._shape is not None and len(coordinates) >= 3:
            try:
                from shapely.geometry import Polygon

                polygon = Polygon(coordinates)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                clipped = polygon.intersection(self._shape)
                if not clipped.is_empty:
                    point = clipped.representative_point()
                    return float(point.x), float(point.y)
            except Exception:
                pass
        return (
            sum(point[0] for point in coordinates) / len(coordinates),
            sum(point[1] for point in coordinates) / len(coordinates),
        )

    def covers(self, longitude: float, latitude: float) -> bool:
        if self._shape is not None:
            left, bottom, right, top = self._bounds
            if not (left <= longitude <= right and bottom <= latitude <= top):
                return False
            try:
                from shapely.geometry import Point

                return bool(self._shape.covers(Point(longitude, latitude)))
            except Exception:
                pass
        coordinates = _polygon_coordinates(self.geometry)
        if not coordinates:
            return False
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        if not (min(xs) <= longitude <= max(xs) and min(ys) <= latitude <= max(ys)):
            return False
        inside = False
        j = len(coordinates) - 1
        for i, (xi, yi) in enumerate(coordinates):
            xj, yj = coordinates[j]
            intersects = ((yi > latitude) != (yj > latitude)) and (
                longitude <= (xj - xi) * (latitude - yi) / ((yj - yi) or 1e-15) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside


def _polygon_coordinates(geometry: Mapping[str, Any]) -> list[tuple[float, float]]:
    coords = geometry.get("coordinates")
    if geometry.get("type") == "Polygon" and coords:
        return [(float(x), float(y)) for x, y in coords[0]]
    if geometry.get("type") == "MultiPolygon" and coords and coords[0]:
        return [(float(x), float(y)) for x, y in coords[0][0]]
    return []


def _element_point(element: Mapping[str, Any], boundary: Boundary) -> tuple[float, float] | None:
    if element.get("lat") is not None and element.get("lon") is not None:
        return float(element["lon"]), float(element["lat"])
    geometry = element.get("geometry") or []
    if isinstance(geometry, Mapping):
        return boundary.representative_geometry(geometry)
    coordinates: list[tuple[float, float]] = []
    for item in geometry:
        if isinstance(item, Mapping):
            if item.get("lon") is not None and item.get("lat") is not None:
                coordinates.append((float(item["lon"]), float(item["lat"])))
            elif item.get("x") is not None and item.get("y") is not None:
                coordinates.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            coordinates.append((float(item[0]), float(item[1])))
    if len(coordinates) < 3:
        return None
    return boundary.representative_point(coordinates)


def normalize_element(
    element: Mapping[str, Any],
    *,
    taxonomy: Taxonomy,
    mapping_catalog: Mapping[str, Any],
    property_catalog: Mapping[str, Mapping[str, Any]],
    boundary: Boundary,
    seed: int = 17,
    mock_schema_version: str = MOCK_SCHEMA_VERSION,
    property_profiles: PropertyProfiles | None = None,
    osm_only: bool = False,
) -> tuple[PoiRecord | None, str | None]:
    osm_type = str(element.get("type", "")).strip().lower()
    try:
        osm_id = int(element["id"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_identity"
    tags = {str(key): value for key, value in (element.get("tags") or {}).items()}
    rule = match_mapping(tags, mapping_catalog)
    if rule is None:
        return None, "unmapped_tags"
    name, names = _extract_name(tags)
    if not name and not (osm_only or rule.get("allow_unnamed", False)):
        return None, "unnamed_business"
    if osm_type == "way" and isinstance(element.get("geometry"), list):
        points = element.get("geometry") or []
        if len(points) < 4 or points[0] != points[-1]:
            return None, "unsupported_open_way"
    point = _element_point(element, boundary)
    if point is None:
        return None, "missing_geometry"
    longitude, latitude = point
    if not boundary.covers(longitude, latitude):
        return None, "outside_boundary"
    category_id = str(rule["category_id"])
    item = taxonomy.items[category_id]
    poi_id = f"osm:{osm_type}:{osm_id}"
    real_properties = normalize_osm_properties(tags, property_catalog)
    applicable_ids = property_profiles.applicable_ids(taxonomy, category_id) if property_profiles else None
    synthetic_properties = {} if osm_only else _mock_properties(
        category_code=item.code,
        poi_id=poi_id,
        category_id=category_id,
        real_properties=real_properties,
        property_catalog=property_catalog,
        seed=seed,
        mock_schema_version=mock_schema_version,
        applicable_ids=applicable_ids,
    )
    effective = dict(real_properties)
    effective.update(synthetic_properties)
    if not osm_only and property_profiles and property_profiles.minimum_known:
        known_count = sum(
            effective.get(key, {}).get("status") == "known" for key in applicable_ids or ()
        )
        if known_count < property_profiles.minimum_known:
            return None, "insufficient_known_properties"
    raw_rating = tags.get("rating")
    rating: float | None = None
    rating_source = "unknown" if osm_only else "mock"
    if raw_rating is not None:
        try:
            parsed = float(raw_rating)
            if 1.0 <= parsed <= 5.0:
                rating = round(parsed, 1)
                rating_source = "osm"
        except (TypeError, ValueError):
            pass
    if rating is None:
        if osm_only:
            review_count = 0
        else:
            rating, review_count = mock_rating(poi_id, category_id, seed, mock_schema_version)
    else:
        try:
            review_count = max(0, int(tags.get("rating:count", 0)))
        except (TypeError, ValueError):
            review_count = 0
    location_method = "node" if osm_type == "node" else "representative_point"
    record_without_hash = PoiRecord(
        poi_id=poi_id,
        osm_type=osm_type,
        osm_id=osm_id,
        category_id=category_id,
        category_code=item.code,
        name=name,
        display_name=name or (
            f"[Nhãn tạo từ loại OSM] {item.names.get('vi') or item.code} ({poi_id})"
            if osm_only else f"{item.names.get('vi') or item.code} ({poi_id})"
        ),
        names=names,
        latitude=round(latitude, 7),
        longitude=round(longitude, 7),
        location_method=location_method,
        mapping_rule_id=str(rule["id"]),
        mapping_version=str(mapping_catalog.get("mapping_version", "unknown")),
        osm_tags=dict(sorted(tags.items())),
        properties_osm=real_properties,
        properties_mock=synthetic_properties,
        effective_properties=effective,
        rating=rating,
        review_count=review_count,
        rating_source=rating_source,
        rating_mock_schema_version=mock_schema_version if rating_source == "mock" else None,
        rating_mock_seed=seed if rating_source == "mock" else None,
        source_version=int(element["version"]) if element.get("version") is not None else None,
        source_timestamp=str(element["timestamp"]) if element.get("timestamp") else None,
    )
    content_hash = stable_hash(record_without_hash.as_dict())
    return PoiRecord(**{**asdict(record_without_hash), "content_hash": content_hash}), None


def _canonical_line(record: PoiRecord) -> bytes:
    return (json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_artifact(
    elements: Iterable[Mapping[str, Any]],
    *,
    taxonomy: Taxonomy,
    mapping_path: str | Path,
    property_path: str | Path,
    boundary_path: str | Path,
    output_dir: str | Path,
    seed: int = 17,
    source_manifest_path: str | Path | None = None,
    source_pbf_path: str | Path | None = None,
    mock_schema_version: str = MOCK_SCHEMA_VERSION,
    osm_only: bool = False,
    signature_registry_path: str | Path | None = None,
    reader_stats: Counter[str] | None = None,
) -> dict[str, Any]:
    catalogs = validate_catalogs(taxonomy, mapping_path, property_path, require_complete=True)
    if catalogs["profiles"].minimum_known and mock_schema_version == MOCK_SCHEMA_VERSION and not osm_only:
        mock_schema_version = "poi-mock-v2"
    boundary_hash = file_sha256(boundary_path)
    manifest_source = load_json(source_manifest_path) if source_manifest_path else {}
    expected_boundary_hash = manifest_source.get("boundary_sha256")
    if expected_boundary_hash and expected_boundary_hash != boundary_hash:
        raise ValueError(f"boundary hash mismatch: expected {expected_boundary_hash}, got {boundary_hash}")
    source_hash = file_sha256(source_pbf_path) if source_pbf_path else manifest_source.get("expected_sha256")
    expected_source_hash = manifest_source.get("expected_sha256")
    if expected_source_hash and source_hash and expected_source_hash != source_hash:
        raise ValueError(f"source PBF hash mismatch: expected {expected_source_hash}, got {source_hash}")
    if source_pbf_path:
        manifest_source["observed_sha256"] = source_hash
        manifest_source["observed_bytes"] = Path(source_pbf_path).stat().st_size
        manifest_source["local_filename"] = Path(source_pbf_path).name
    boundary = Boundary.from_geojson(boundary_path)
    output = Path(output_dir)
    if output.exists() and (osm_only or any(output.iterdir())):
        raise ValueError(f"refusing to overwrite existing POI artifact: {output}")
    registry = load_json(signature_registry_path) if signature_registry_path else None
    if registry and not osm_only:
        raise ValueError("signature registry requires osm_only mode")
    if registry and registry.get("mapping_version") != catalogs["mapping"].get("mapping_version"):
        raise ValueError("signature registry mapping version differs from artifact mapping")
    if registry and (
        registry.get("pbf_sha256") != source_hash or registry.get("boundary_sha256") != boundary_hash
        or registry.get("taxonomy_version") != taxonomy.document.taxonomy_version
    ):
        raise ValueError("signature registry does not match pinned source, boundary or taxonomy")
    expected_signatures = {row["signature"]: row for row in registry.get("signatures", [])} if registry else None
    if registry and len(expected_signatures) != len(registry["signatures"]):
        raise ValueError("duplicate candidate signature in registry")
    if registry and registry.get("audit_sha256") != stable_hash(registry["signatures"]):
        raise ValueError("candidate signature registry audit hash mismatch")
    records_by_id: dict[str, PoiRecord] = {}
    rejected: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    signature_dispositions: dict[str, Counter[str]] = {}
    unmapped_signatures: Counter[str] = Counter()
    unmapped_examples: list[dict[str, Any]] = []
    for element in elements:
        if osm_only:
            tags = element.get("tags") or {}
            if not candidate_tags(tags):
                rejected["not_poi_candidate"] += 1
                continue
            point = _element_point(element, boundary)
            if point is not None and boundary.covers(*point):
                signature = candidate_signature(tags)
                signature_counts[signature] += 1
                if expected_signatures is not None and signature not in expected_signatures:
                    raise ValueError(f"new in-boundary candidate signature needs review: {signature}")
            else:
                signature = None
        record, reason = normalize_element(
            element,
            taxonomy=taxonomy,
            mapping_catalog=catalogs["mapping"],
            property_catalog=catalogs["properties"],
            boundary=boundary,
            seed=seed,
            mock_schema_version=mock_schema_version,
            property_profiles=catalogs["profiles"] if osm_only or catalogs["profiles"].minimum_known else None,
            osm_only=osm_only,
        )
        if osm_only and signature is not None:
            disposition = ("imported_unnamed" if record.name is None else "imported") if record else str(reason)
            if disposition == "unmapped_tags" and signature in catalogs["mapping"].get("intentionally_excluded_signatures", {}):
                disposition = catalogs["mapping"]["intentionally_excluded_signatures"][signature]["disposition"]
            signature_dispositions.setdefault(signature, Counter())[disposition] += 1
        if record is not None:
            previous = records_by_id.get(record.poi_id)
            if previous is not None:
                rejected["duplicate_identity"] += 1
                if record.content_hash <= previous.content_hash:
                    continue
            records_by_id[record.poi_id] = record
        else:
            rejected[str(reason)] += 1
            if reason == "unmapped_tags":
                tags = element.get("tags", {}) or {}
                signature = "|".join(f"{key}={tags[key]}" for key in sorted(tags) if key in {"amenity", "shop", "tourism", "leisure", "office", "healthcare", "craft"})
                unmapped_signatures[signature or "(no primary category tag)"] += 1
                if len(unmapped_examples) < 100:
                    unmapped_examples.append({"type": element.get("type"), "id": element.get("id"), "tags": dict(tags)})
    if expected_signatures is not None:
        for signature, row in expected_signatures.items():
            if signature_counts[signature] != int(row["count"]):
                raise ValueError(f"candidate signature count differs from pinned registry: {signature}")
            if dict(sorted(signature_dispositions.get(signature, {}).items())) != row["dispositions"]:
                raise ValueError(f"candidate signature disposition differs from pinned registry: {signature}")
    type_order = {"node": 0, "way": 1, "relation": 2}
    records = sorted(records_by_id.values(), key=lambda row: (type_order.get(row.osm_type, 9), row.osm_id, int(row.category_id)))
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "pois.jsonl"
    digest = hashlib.sha256()
    with artifact_path.open("wb") as handle:
        for record in records:
            line = _canonical_line(record)
            handle.write(line)
            digest.update(line)
    category_count_values = Counter(record.category_id for record in records)
    category_counts = {
        category_id: category_count_values.get(category_id, 0)
        for category_id in catalogs["coverage"]["mapped_category_ids"]
    }
    rule_counts = Counter(record.mapping_rule_id for record in records)
    manifest_source = load_json(source_manifest_path) if source_manifest_path else {}
    mapping_hash = file_sha256(mapping_path)
    property_hash = file_sha256(property_path)
    source_hash = file_sha256(source_pbf_path) if source_pbf_path else manifest_source.get("expected_sha256")
    if source_pbf_path:
        manifest_source["observed_sha256"] = source_hash
        manifest_source["observed_bytes"] = Path(source_pbf_path).stat().st_size
        manifest_source["local_filename"] = Path(source_pbf_path).name
    content_set_hash = stable_hash([(record.poi_id, record.content_hash) for record in records])
    input_fingerprint = stable_hash({
        "source_sha256": source_hash,
        "content_set_hash": content_set_hash,
        "boundary_sha256": boundary_hash,
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "mapping_sha256": mapping_hash,
        "property_sha256": property_hash,
        "pipeline_version": "osm-poi-pipeline-v3-osm-only" if osm_only else PIPELINE_VERSION,
        **({"osm_only": True, "signature_registry_sha256": file_sha256(signature_registry_path) if signature_registry_path else None}
           if osm_only else {"mock_schema_version": mock_schema_version, "seed": seed}),
    })
    manifest = {
        "schema_version": 1,
        "artifact_kind": "hcmc_osm_poi_v3_osm_only" if osm_only else "hcmc_osm_poi_v1",
        "artifact_id": input_fingerprint[:24],
        "input_fingerprint": input_fingerprint,
        "pipeline_version": "osm-poi-pipeline-v3-osm-only" if osm_only else PIPELINE_VERSION,
        "source": manifest_source,
        "boundary_sha256": boundary_hash,
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "taxonomy_version": taxonomy.document.taxonomy_version,
        "mapping_version": catalogs["mapping"].get("mapping_version"),
        "mapping_sha256": mapping_hash,
        "property_taxonomy_version": load_json(property_path).get("property_taxonomy_version"),
        "property_taxonomy_sha256": property_hash,
        "mock_schema_version": None if osm_only else mock_schema_version,
        "mock_seed": None if osm_only else seed,
        "poi_count": len(records),
        "category_counts": dict(sorted(category_counts.items(), key=lambda row: int(row[0]))),
        "mapping_rule_counts": dict(sorted(rule_counts.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "content_set_hash": content_set_hash,
        "pois_jsonl_sha256": digest.hexdigest(),
        "coverage": catalogs["coverage"],
        "attribution": "© OpenStreetMap contributors",
        "license": "ODbL 1.0",
        **({
            "osm_only": True,
            "missing_properties": "unknown_no_assertion",
            "rating_policy": "osm_observed_or_unknown_no_synthetic_reviews",
            "display_name_policy": "osm_name_or_explicit_generated_category_label",
            "unnamed_poi_count": sum(record.name is None for record in records),
            "reader_omissions": dict(sorted(reader_stats.items())) if reader_stats is not None else {},
            "signature_registry_sha256": file_sha256(signature_registry_path) if signature_registry_path else None,
        } if osm_only else {}),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "category_coverage.json").write_text(json.dumps({"category_counts": manifest["category_counts"], "rule_counts": manifest["mapping_rule_counts"], "coverage": manifest["coverage"]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    unmapped_examples.sort(key=lambda row: (str(row.get("type", "")), int(row.get("id", 0))))
    (output / "unmapped.json").write_text(json.dumps({"tag_signature_counts": dict(sorted(unmapped_signatures.items())), "examples": unmapped_examples}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "rejected.json").write_text(json.dumps({"counts": dict(sorted(rejected.items()))}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if osm_only:
        signature_report = [
            {"signature": key, "count": signature_counts[key],
             "dispositions": dict(sorted(signature_dispositions.get(key, {}).items()))}
            for key in sorted(signature_counts)
        ]
        (output / "candidate_signatures.json").write_text(
            json.dumps({"signatures": signature_report,
                        "reader_omissions": manifest["reader_omissions"]},
                       ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
    return manifest


def enrich_artifact(
    source_manifest_path: str | Path,
    *,
    output_dir: str | Path,
    taxonomy: Taxonomy,
    mapping_path: str | Path,
    property_path: str | Path,
    mock_schema_version: str = "poi-mock-v2",
    seed: int | None = None,
) -> dict[str, Any]:
    """Enrich saved OSM tags without re-reading the Vietnam PBF or changing POI geometry."""
    from .poi_repository import PoiRepository

    source_manifest_path = Path(source_manifest_path)
    source = load_json(source_manifest_path)
    source_rows = source_manifest_path.parent / "pois.jsonl"
    if file_sha256(source_rows) != source.get("pois_jsonl_sha256"):
        raise ValueError("source artifact checksum mismatch")
    if source.get("taxonomy_fingerprint") != taxonomy.fingerprint():
        raise ValueError("source artifact uses a different taxonomy")
    if source.get("mapping_sha256") != file_sha256(mapping_path):
        raise ValueError("source artifact uses a different OSM mapping catalog")

    catalogs = validate_catalogs(taxonomy, mapping_path, property_path)
    profiles: PropertyProfiles = catalogs["profiles"]
    if profiles.minimum_known < 5:
        raise ValueError("property catalog must require at least five known properties")
    seed = int(source.get("mock_seed", 17) if seed is None else seed)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite existing POI artifact: {output}")

    records: list[PoiRecord] = []
    coverage: dict[str, dict[str, Any]] = {}
    for record in PoiRepository.from_jsonl(source_rows).records:
        if record.category_id not in taxonomy.items or taxonomy.has_children(record.category_id):
            raise ValueError(f"invalid category in source artifact: {record.category_id}")
        applicable = profiles.applicable_ids(taxonomy, record.category_id)
        real = normalize_osm_properties(record.osm_tags, profiles.definitions)
        synthetic = _mock_properties(
            category_code=record.category_code,
            category_id=record.category_id,
            poi_id=record.poi_id,
            real_properties=real,
            property_catalog=profiles.definitions,
            seed=seed,
            mock_schema_version=mock_schema_version,
            applicable_ids=applicable,
        )
        effective = {**real, **synthetic}
        known = sum(effective.get(key, {}).get("status") == "known" for key in applicable)
        if known < profiles.minimum_known:
            raise ValueError(f"{record.poi_id} has {known} known applicable properties; need {profiles.minimum_known}")
        new_rating, new_reviews = record.rating, record.review_count
        new_rating_version, new_rating_seed = record.rating_mock_schema_version, record.rating_mock_seed
        if record.rating_source == "mock" and (
            record.rating_mock_schema_version != mock_schema_version or record.rating_mock_seed != seed
        ):
            new_rating, new_reviews = mock_rating(record.poi_id, record.category_id, seed, mock_schema_version)
            new_rating_version, new_rating_seed = mock_schema_version, seed
        new_record = replace(
            record,
            properties_osm=real,
            properties_mock=synthetic,
            effective_properties=effective,
            rating=new_rating,
            review_count=new_reviews,
            rating_mock_schema_version=new_rating_version,
            rating_mock_seed=new_rating_seed,
            content_hash="",
        )
        records.append(replace(new_record, content_hash=stable_hash(new_record.as_dict())))
        stats = coverage.setdefault(record.category_id, {"poi_count": 0, "minimum_known": None, "osm_known": 0, "mock_known": 0})
        stats["poi_count"] += 1
        stats["minimum_known"] = known if stats["minimum_known"] is None else min(stats["minimum_known"], known)
        stats["osm_known"] += sum(real.get(key, {}).get("status") == "known" for key in applicable)
        stats["mock_known"] += sum(synthetic.get(key, {}).get("status") == "known" for key in applicable)
    if len(records) != int(source.get("poi_count", -1)):
        raise ValueError("source artifact POI count mismatch")

    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "pois.jsonl"
    sha = hashlib.sha256()
    with rows_path.open("wb") as handle:
        for record in records:
            line = _canonical_line(record)
            handle.write(line)
            sha.update(line)
    property_payload = load_json(property_path)
    base = property_payload.get("base_catalog")
    base_hash = file_sha256(Path(property_path).parent / base) if base else None
    property_hash = file_sha256(property_path)
    content_set_hash = stable_hash([(record.poi_id, record.content_hash) for record in records])
    fingerprint = stable_hash({
        "parent_artifact_id": source["artifact_id"],
        "parent_content_set_hash": source["content_set_hash"],
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "mapping_sha256": source["mapping_sha256"],
        "property_taxonomy_sha256": property_hash,
        "property_base_sha256": base_hash,
        "mock_schema_version": mock_schema_version,
        "mock_seed": seed,
        "content_set_hash": content_set_hash,
    })
    manifest = {
        **source,
        "artifact_id": fingerprint[:24],
        "artifact_kind": "hcmc_osm_poi_enriched_v2",
        "input_fingerprint": fingerprint,
        "pipeline_version": "osm-poi-enrichment-v2",
        "parent_artifact_id": source["artifact_id"],
        "parent_content_set_hash": source["content_set_hash"],
        "property_taxonomy_version": profiles.version,
        "property_taxonomy_sha256": property_hash,
        "property_base_sha256": base_hash,
        "mock_schema_version": mock_schema_version,
        "mock_seed": seed,
        "content_set_hash": content_set_hash,
        "pois_jsonl_sha256": sha.hexdigest(),
        "property_coverage": dict(sorted(coverage.items(), key=lambda row: int(row[0]))),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "property_coverage.json").write_text(json.dumps(manifest["property_coverage"], ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def iter_pbf_elements(
    path: str | Path, *, reader_stats: Counter[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream candidate POIs and assembled areas; count reader omissions when requested.

    Pyosmium may fail to assemble some relations *before* iteration; those
    upstream losses cannot be measured here and are explicitly reported.
    """
    try:
        import osmium  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install the optional 'poi' dependencies to read .osm.pbf files") from exc

    try:
        from shapely import wkb
        from shapely.geometry import mapping as geometry_mapping
    except ImportError as exc:
        raise RuntimeError("install the optional 'poi' dependencies to assemble OSM areas") from exc

    area_factory = osmium.geom.WKBFactory()
    processor = osmium.FileProcessor(str(path)).with_locations(storage="sparse_file_array").with_areas()
    for element in processor:
        class_name = type(element).__name__.casefold()
        tags = dict(element.tags)
        if reader_stats is not None:
            reader_stats[f"observed_{class_name}"] += 1
        if not tags or not candidate_tags(tags):
            if reader_stats is not None and tags:
                reader_stats["excluded_non_poi_tags"] += 1
                if "highway" in tags and _norm_tag(tags["highway"]) not in POI_QUALIFIED_VALUES["highway"]:
                    reader_stats["excluded_road_or_non_poi_highway"] += 1
                if "aeroway" in tags and _norm_tag(tags["aeroway"]) not in POI_QUALIFIED_VALUES["aeroway"]:
                    reader_stats["excluded_airfield_path_or_non_poi_aeroway"] += 1
                if "barrier" in tags and not POI_PRIMARY_KEYS.intersection(tags):
                    reader_stats["excluded_barrier_without_poi_tag"] += 1
            continue
        if reader_stats is not None:
            reader_stats["tagged_candidate_before_geometry"] += 1
        version = getattr(element, "version", None)
        timestamp = str(getattr(element, "timestamp", "")) or None
        if class_name == "node":
            try:
                if not element.location.valid():
                    if reader_stats is not None:
                        reader_stats["omitted_invalid_node_location"] += 1
                    continue
                latitude = float(element.location.lat)
                longitude = float(element.location.lon)
            except (AttributeError, RuntimeError, ValueError):
                if reader_stats is not None:
                    reader_stats["omitted_invalid_node_location"] += 1
                continue
            if reader_stats is not None:
                reader_stats["emitted_node"] += 1
            yield {
                "type": "node", "id": int(element.id), "version": version,
                "timestamp": timestamp, "lat": latitude, "lon": longitude, "tags": tags,
            }
        elif class_name == "area":
            try:
                geometry = wkb.loads(area_factory.create_multipolygon(element))
            except Exception:
                if reader_stats is not None:
                    reader_stats["omitted_unassembled_or_invalid_area_geometry"] += 1
                continue
            from_way = bool(element.from_way()) if callable(getattr(element, "from_way", None)) else False
            source_type = "way" if from_way else "relation"
            source_id = int(element.orig_id()) if callable(getattr(element, "orig_id", None)) else int(element.id)
            if reader_stats is not None:
                reader_stats[f"emitted_{source_type}_area"] += 1
            yield {
                "type": source_type, "id": source_id, "version": version,
                "timestamp": timestamp, "geometry": geometry_mapping(geometry), "tags": tags,
            }
        elif class_name == "way":
            # Closed tagged ways are represented by assembled Area objects.
            try:
                if element.is_closed():
                    if reader_stats is not None:
                        reader_stats["skipped_closed_way_replaced_by_area"] += 1
                    continue
            except AttributeError:
                pass
            geometry: list[dict[str, float]] = []
            for node in element.nodes:
                try:
                    if node.location.valid():
                        geometry.append({"lat": float(node.location.lat), "lon": float(node.location.lon)})
                except (AttributeError, RuntimeError, ValueError):
                    continue
            if len(geometry) >= 2:
                if reader_stats is not None:
                    reader_stats["emitted_open_way"] += 1
                yield {
                    "type": "way", "id": int(element.id), "version": version,
                    "timestamp": timestamp, "geometry": geometry, "tags": tags,
                }
            elif reader_stats is not None:
                reader_stats["omitted_way_missing_node_locations"] += 1
        elif reader_stats is not None:
            reader_stats["omitted_unsupported_osm_element_type"] += 1


def write_source_manifest(source_manifest_path: str | Path, *, pbf_path: str | Path) -> dict[str, Any]:
    """Validate source checksum metadata and report the downloaded snapshot."""
    manifest = load_json(source_manifest_path)
    actual_hash = file_sha256(pbf_path)
    expected = manifest.get("expected_sha256")
    if expected and expected != actual_hash:
        raise ValueError(f"source PBF hash mismatch: expected {expected}, got {actual_hash}")
    manifest["observed_sha256"] = actual_hash
    manifest["observed_bytes"] = Path(pbf_path).stat().st_size
    manifest["local_filename"] = Path(pbf_path).name
    return manifest
