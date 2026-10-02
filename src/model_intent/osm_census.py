from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .osm_ingest import Boundary, _element_point, candidate_signature, candidate_tags, match_mapping, normalize_element
from .poi_properties import PropertyProfiles
from .taxonomy import Taxonomy

_PRIMARY_TAGS = {
    "amenity", "shop", "tourism", "leisure", "office", "healthcare", "craft",
    "public_transport", "historic", "sport", "government", "emergency", "club",
}
_RAILWAY_POIS = {"station", "halt", "tram_stop", "subway_entrance"}


def census_hcmc_candidates(
    elements: Iterable[Mapping[str, Any]],
    *,
    boundary: Boundary,
    taxonomy: Taxonomy,
    mapping_catalog: Mapping[str, Any],
    property_profiles: PropertyProfiles,
    sample_limit: int = 30,
    osm_only: bool = False,
    reader_stats: Counter[str] | None = None,
    complete_signatures: bool = False,
) -> dict[str, Any]:
    """Count HCMC tagged candidate objects *before* category/name rejection.

    The opt-in complete-signatures census accounts for every in-boundary
    candidate signature and each disposition; legacy census stays unchanged.
    """
    if complete_signatures:
        return _complete_signature_census(
            elements, boundary=boundary, taxonomy=taxonomy,
            mapping_catalog=mapping_catalog, property_profiles=property_profiles,
            sample_limit=sample_limit, osm_only=osm_only, reader_stats=reader_stats,
        )
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    unknown_tags: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for element in elements:
        counts["reader_emitted"] += 1
        tags = element.get("tags") or {}
        poi_like = bool(_PRIMARY_TAGS.intersection(tags)) or tags.get("highway") == "bus_stop" or tags.get("railway") in _RAILWAY_POIS
        if not poi_like:
            continue
        counts["poi_like_tagged_candidates_nationwide"] += 1
        point = _element_point(element, boundary)
        if point is None:
            counts["geometry_unavailable_or_area_outside_boundary"] += 1
            continue
        longitude, latitude = point
        if not boundary.covers(longitude, latitude):
            counts["outside_hcmc"] += 1
            continue
        counts["inside_hcmc_tagged_candidates"] += 1
        source_id = f"osm:{element.get('type')}:{element.get('id')}"
        if source_id in seen_ids:
            counts["duplicate_osm_identity"] += 1
            continue
        seen_ids.add(source_id)
        rule = match_mapping(tags, mapping_catalog)
        if rule is None:
            counts["inside_hcmc_unmapped"] += 1
            primary = ";".join(f"{key}={tags[key]}" for key in sorted(_PRIMARY_TAGS) if key in tags)
            unknown_tags[primary or f"railway={tags.get('railway', tags.get('highway', 'unknown'))}"] += 1
            if len(examples) < sample_limit:
                examples.append({"osm_id": source_id, "tags": dict(tags)})
            continue
        counts["inside_hcmc_mapped_rule"] += 1
        name = tags.get("name:vi") or tags.get("name") or tags.get("official_name") or tags.get("name:en")
        if not name and not rule.get("allow_unnamed", False):
            counts["mapped_missing_name"] += 1
            continue
        if str(element.get("type")) == "way" and isinstance(element.get("geometry"), list):
            way = element.get("geometry") or []
            if len(way) < 4 or way[0] != way[-1]:
                counts["mapped_unsupported_open_way"] += 1
                continue
        record, reason = normalize_element(
            element, taxonomy=taxonomy, mapping_catalog=mapping_catalog,
            property_catalog=property_profiles.definitions, property_profiles=property_profiles,
            boundary=boundary, mock_schema_version="poi-mock-v2",
        )
        if record is None:
            counts[f"mapped_rejected_{reason}"] += 1
        else:
            counts["inside_hcmc_importable"] += 1
            categories[record.category_id] += 1
    return {
        "kind": "hcmc_osm_tagged_candidate_census_v1",
        "counts": dict(sorted(counts.items())),
        "mapped_category_counts": dict(sorted(categories.items(), key=lambda item: int(item[0]))),
        "unmapped_tag_counts": dict(unknown_tags.most_common(200)),
        "sample_unmapped": examples,
        "limitations": [
            "The reader preselects tagged OSM objects and can skip invalid source geometry.",
            "Tagged candidate objects include some non-business facilities; they are not all real-world POIs.",
            "Objects with different OSM IDs can represent the same physical place; no cross-identity deduplication is asserted.",
            "This census cannot validate an external 400,000-POI estimate without its dated source and definition.",
        ],
    }


def _complete_signature_census(
    elements: Iterable[Mapping[str, Any]], *, boundary: Boundary,
    taxonomy: Taxonomy, mapping_catalog: Mapping[str, Any],
    property_profiles: PropertyProfiles, sample_limit: int,
    osm_only: bool, reader_stats: Counter[str] | None,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    signatures: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    excluded = mapping_catalog.get("intentionally_excluded_signatures", {})
    for element in elements:
        counts["reader_emitted"] += 1
        tags = element.get("tags") or {}
        if not candidate_tags(tags):
            counts["not_poi_like"] += 1
            continue
        counts["poi_like_tagged_candidates_nationwide"] += 1
        point = _element_point(element, boundary)
        if point is None:
            counts["geometry_unavailable_or_area_outside_boundary"] += 1
            continue
        if not boundary.covers(*point):
            counts["outside_hcmc"] += 1
            continue
        counts["inside_hcmc_tagged_candidates"] += 1
        signature = candidate_signature(tags)
        row = signatures.setdefault(signature, {
            "signature": signature, "count": 0, "dispositions": Counter(),
            "mapping_rules": Counter(), "category_ids": Counter(), "examples": [],
        })
        row["count"] += 1
        identity = f"osm:{element.get('type')}:{element.get('id')}"
        if len(row["examples"]) < sample_limit:
            row["examples"].append(identity)
        if identity in seen:
            counts["duplicate_osm_identity"] += 1
            row["dispositions"]["duplicate_osm_identity"] += 1
            continue
        seen.add(identity)
        rule = match_mapping(tags, mapping_catalog)
        if rule is None:
            disposition = excluded[signature]["disposition"] if signature in excluded else "unmapped_needs_review"
            counts[f"inside_hcmc_{disposition}"] += 1
            row["dispositions"][disposition] += 1
            continue
        row["mapping_rules"][str(rule["id"])] += 1
        row["category_ids"][str(rule["category_id"])] += 1
        counts["inside_hcmc_mapped_rule"] += 1
        record, reason = normalize_element(
            element, taxonomy=taxonomy, mapping_catalog=mapping_catalog,
            property_catalog=property_profiles.definitions, boundary=boundary,
            property_profiles=property_profiles, osm_only=osm_only,
        )
        disposition = ("imported_unnamed" if record.name is None else "imported") if record else str(reason)
        row["dispositions"][disposition] += 1
        counts[f"inside_hcmc_{disposition}"] += 1
        if record:
            categories[record.category_id] += 1
    serialized = []
    for key in sorted(signatures):
        row = signatures[key]
        serialized.append({
            **row, "dispositions": dict(sorted(row["dispositions"].items())),
            "mapping_rules": dict(sorted(row["mapping_rules"].items())),
            "category_ids": dict(sorted(row["category_ids"].items(), key=lambda pair: int(pair[0]))),
        })
    return {
        "kind": "hcmc_osm_candidate_signatures_v3", "counts": dict(sorted(counts.items())),
        "mapped_category_counts": dict(sorted(categories.items(), key=lambda pair: int(pair[0]))),
        "signatures": serialized,
        "reader_omissions": dict(sorted(reader_stats.items())) if reader_stats is not None else {},
        "limitations": [
            "Candidate object count is not a count of distinct real-world businesses or places.",
            "Pyosmium may omit malformed OSM areas before yielding them; upstream losses cannot be measured.",
            "Invalid source geometries, open ways and duplicate identities have separate reported dispositions.",
            "Objects with different OSM identities can represent the same real-world place.",
        ],
    }
