from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .taxonomy import Taxonomy


@dataclass(frozen=True)
class PropertyProfiles:
    definitions: dict[str, dict[str, Any]]
    version: str
    minimum_known: int
    families: Mapping[str, tuple[str, ...]]
    categories: Mapping[str, tuple[str, ...]]

    def applicable_ids(self, taxonomy: Taxonomy, category_id: str) -> tuple[str, ...]:
        category_id = str(category_id)
        if category_id in self.categories:
            return self.categories[category_id]
        for ancestor_id in reversed(taxonomy.path_ids(category_id)[:-1]):
            if ancestor_id in self.families:
                return self.families[ancestor_id]
        return ()

    def validate_coverage(self, taxonomy: Taxonomy, category_ids: set[str]) -> dict[str, tuple[str, ...]]:
        missing: list[str] = []
        resolved: dict[str, tuple[str, ...]] = {}
        for category_id in sorted(category_ids, key=int):
            ids = self.applicable_ids(taxonomy, category_id)
            mockable = [key for key in ids if self.definitions[key]["value_type"] in {"boolean", "enum", "ordinal"}]
            if len(mockable) < self.minimum_known:
                missing.append(category_id)
            resolved[category_id] = ids
        if missing:
            raise ValueError(f"fewer than {self.minimum_known} suitable properties for mapped categories: {', '.join(missing)}")
        return resolved


def load_property_definitions(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("property catalog must be an object")
    base = payload.get("base_catalog")
    definitions: dict[str, dict[str, Any]] = {}
    if base:
        base_path = (path.parent / str(base)).resolve()
        if base_path == path.resolve():
            raise ValueError("property catalog cannot include itself")
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        for row in base_payload.get("properties", []):
            definitions[str(row["id"])] = dict(row)
    for row in payload.get("properties", []):
        key = str(row.get("id", ""))
        if not key or key in definitions:
            raise ValueError(f"duplicate or missing property id: {key}")
        definitions[key] = dict(row)
    for key, overrides in (payload.get("property_overrides") or {}).items():
        if key not in definitions or not isinstance(overrides, dict):
            raise ValueError(f"invalid property override: {key}")
        definitions[key] = {**definitions[key], **overrides}
    for key, row in definitions.items():
        if row.get("value_type") not in {"boolean", "enum", "ordinal", "string"}:
            raise ValueError(f"invalid value_type for {key}")
        if row.get("value_type") in {"enum", "ordinal"} and len(row.get("values") or []) < 2:
            raise ValueError(f"property {key} needs at least two possible values")
        if not isinstance(row.get("aliases"), dict):
            raise ValueError(f"invalid aliases for {key}")
        for rate in (row.get("mock_true_rate_by_code") or {}).values():
            if not 0 <= float(rate) <= 1:
                raise ValueError(f"invalid mock rate for {key}")
    return definitions


def load_property_profiles(path: str | Path, taxonomy: Taxonomy) -> PropertyProfiles:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    definitions = load_property_definitions(path)

    def read_profiles(rows: Any, id_field: str) -> dict[str, tuple[str, ...]]:
        profiles: dict[str, tuple[str, ...]] = {}
        for row in rows or []:
            item_id = str(row.get(id_field, ""))
            if item_id not in taxonomy.items or item_id in profiles:
                raise ValueError(f"invalid or duplicate {id_field}: {item_id}")
            ids = tuple(str(key) for key in row.get("property_ids", []))
            if len(ids) != len(set(ids)) or any(key not in definitions for key in ids):
                raise ValueError(f"unknown or repeated property in {id_field} {item_id}")
            profiles[item_id] = ids
        return profiles

    families = read_profiles(payload.get("family_profiles"), "ancestor_id")
    categories = read_profiles(payload.get("category_profiles"), "category_id")
    if any(taxonomy.has_children(category_id) for category_id in categories):
        raise ValueError("category profiles must target leaves")
    minimum = int(payload.get("minimum_known_properties", 0))
    if minimum < 0:
        raise ValueError("minimum_known_properties must not be negative")
    return PropertyProfiles(
        definitions=definitions,
        version=str(payload.get("property_taxonomy_version", "unknown")),
        minimum_known=minimum,
        families=families,
        categories=categories,
    )
