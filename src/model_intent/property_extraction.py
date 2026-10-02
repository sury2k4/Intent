from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .normalization import fold_vietnamese, normalize_text
from .osm_ingest import load_property_catalog
from .poi_properties import PropertyProfiles
from .poi_schemas import ExtractedProperty, PropertyExtraction
from .taxonomy import Taxonomy

EXTRACTOR_VERSION = "poi-property-extractor-v2"

# Search aliases only: these do not change the immutable catalog/artifact hash.
_QUERY_ALIASES = {
    "seating": ("ghế ngồi", "ghe ngoi", "ghế đá", "ghe da", "có ghế", "benches", "park bench"),
    "shade": ("bóng mát", "bong mat", "có bóng mát", "shady"),
    "shelter": ("nhà chờ xe buýt", "nha cho xe buyt", "bus shelter"),
    "delivery": ("giao đến nhà", "giao den nha"),
}
_CLAUSE_BREAK = re.compile(r"[,;.!?\n]|\b(?:nhưng|nhung|but|và|va|and)\b", re.IGNORECASE)
_NOT_NEEDED = re.compile(r"(?:^|\s)(?:khong can|dont need|do not need|not need)\s*$")
_EXCLUDED = re.compile(r"(?:^|\s)(?:khong co|khong muon|khong ngoi|without|no)\s*$")
_PREFERRED = re.compile(r"(?:^|\s)(?:uu tien(?: co)?|neu co|tot nhat|prefer|ideally|if possible)\s*$")


def _fold_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Fold accents while retaining offsets into the original user input."""
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, char in enumerate(text):
        if char in "Đđ":
            mapped = "d"
        else:
            mapped = unicodedata.normalize("NFKD", char.casefold())
        for part in mapped:
            if unicodedata.combining(part):
                if ends:
                    ends[-1] = index + 1
                continue
            chars.append(part)
            starts.append(index)
            ends.append(index + 1)
    return "".join(chars), starts, ends


class PropertyExtractor:
    def __init__(self, property_catalog: Mapping[str, Mapping[str, Any]], *, schema_version: str = EXTRACTOR_VERSION):
        self.property_catalog = dict(property_catalog)
        self.schema_version = schema_version
        aliases: list[tuple[str, str]] = []
        for property_id, definition in self.property_catalog.items():
            if definition.get("queryable", True) is False:
                continue
            for values in (definition.get("aliases") or {}).values():
                for alias in values or []:
                    if str(alias).strip():
                        aliases.append((property_id, str(alias).strip()))
            aliases.extend((property_id, alias) for alias in _QUERY_ALIASES.get(property_id, ()))
        self.aliases = tuple(sorted(set(aliases), key=lambda row: (-len(row[1]), row[0], row[1])))

    @classmethod
    def from_path(cls, path: str | Path) -> "PropertyExtractor":
        return cls(load_property_catalog(path))

    def extract(self, query: str) -> PropertyExtraction:
        text = str(query)
        if not text.strip():
            return PropertyExtraction(text, self.schema_version)
        try:
            folded, starts, ends = _fold_with_offsets(text)
            candidates: list[tuple[int, int, str, str, int]] = []
            for property_id, alias in self.aliases:
                needle = fold_vietnamese(alias).casefold()
                for match in re.finditer(rf"(?<!\w){re.escape(needle)}(?!\w)", folded):
                    start, end = match.span()
                    candidates.append((starts[start], ends[end - 1], property_id, alias, end - start))
            # Match the longest phrase across overlaps, then restore query order.
            selected: list[tuple[int, int, str, str, int]] = []
            for candidate in sorted(candidates, key=lambda row: (-row[4], row[0], row[2], row[3])):
                start, end = candidate[:2]
                if not any(start < previous[1] and end > previous[0] for previous in selected):
                    selected.append(candidate)
            selected.sort(key=lambda row: (row[0], row[1], row[2]))
            constraints: list[ExtractedProperty] = []
            previous_end = 0
            for candidate in selected:
                constraints.append(self._constraint(text, candidate, previous_end))
                previous_end = candidate[1]
            constraints = self._mark_conflicts(constraints)
            constraints.extend(self._extract_numeric_constraints(text))
            return PropertyExtraction(text, self.schema_version, tuple(constraints))
        except Exception as exc:  # invalid catalogs fail at startup; a query fails open
            return PropertyExtraction(text, self.schema_version, (), (f"property extraction failed open: {type(exc).__name__}: {exc}",), True)

    def _constraint(self, text: str, item: tuple[int, int, str, str, int], previous_end: int) -> ExtractedProperty:
        start, end, property_id, alias, _length = item
        prefix = fold_vietnamese(text[previous_end:start]).casefold()
        breaks = list(_CLAUSE_BREAK.finditer(prefix))
        if breaks:
            prefix = prefix[breaks[-1].end():]
        prefix = normalize_text(prefix)
        # "có" can belong to the alias: "không có mái che", "nếu có ghế ngồi".
        folded_alias = normalize_text(fold_vietnamese(text[start:end]))
        context = f"{prefix} {folded_alias.split()[0]}".strip()
        not_needed = _NOT_NEEDED.search(prefix) or _NOT_NEEDED.search(context)
        excluded = _EXCLUDED.search(prefix) or _EXCLUDED.search(context)
        preferred = _PREFERRED.search(prefix) or _PREFERRED.search(context)
        if not_needed:
            mode, applied, ignored_reason, cue = "preferred", False, "not_requested", not_needed.group(0).strip()
        elif excluded:
            mode, applied, ignored_reason, cue = "excluded", True, None, excluded.group(0).strip()
        elif preferred:
            mode, applied, ignored_reason, cue = "preferred", True, None, preferred.group(0).strip()
        else:
            mode, applied, ignored_reason, cue = "required", True, None, "explicit_property"
        value_type = str(self.property_catalog[property_id].get("value_type", "boolean"))
        if value_type == "ordinal":
            value: Any = "high"
            operator = "gte"
        elif value_type == "enum":
            value, operator = "yes", "eq"
        elif value_type == "string":
            value, operator, applied, ignored_reason = None, "eq", False, "value_unspecified"
        else:
            value, operator = True, "eq"
        return ExtractedProperty(
            property_id=property_id, mode=mode, operator=operator, value=value,
            evidence_text=text[start:end], start_offset=start, end_offset=end,
            matched_alias=alias, cue=cue, confidence=0.98 if applied else 0.9,
            applied=applied, ignored_reason=ignored_reason,
        )

    @staticmethod
    def _mark_conflicts(constraints: Iterable[ExtractedProperty]) -> list[ExtractedProperty]:
        rows = list(constraints)
        grouped: dict[str, set[tuple[str, str, str]]] = {}
        for row in rows:
            grouped.setdefault(row.property_id, set()).add((row.mode, row.operator, repr(row.value)))
        if any(len(grouped[row.property_id]) > 1 for row in rows):
            return [
                replace(row, applied=False, ignored_reason="conflicting_constraints")
                if row.applied and len(grouped[row.property_id]) > 1 else row
                for row in rows
            ]
        seen: set[tuple[str, str, str, str, bool, str | None]] = set()
        distinct: list[ExtractedProperty] = []
        for row in rows:
            key = (row.property_id, row.mode, row.operator, repr(row.value), row.applied, row.ignored_reason)
            if key not in seen:
                distinct.append(row)
                seen.add(key)
        return distinct

    @staticmethod
    def _extract_numeric_constraints(text: str) -> list[ExtractedProperty]:
        output: list[ExtractedProperty] = []
        rating_pattern = re.compile(
            r"(?P<cue>rating|điểm|d[aá]nh gi[aá]|stars?)\s*"
            r"(?P<op>từ|trên|ít nhất|at least|above|over|>=|>)?\s*"
            r"(?P<value>[0-5](?:[\.,]\d+)?)(?!\d)", re.IGNORECASE,
        )
        for match in rating_pattern.finditer(text):
            value = float(match.group("value").replace(",", "."))
            invalid = not 1.0 <= value <= 5.0
            op = normalize_text(fold_vietnamese(match.group("op") or ""))
            operator = "gt" if op in {"tren", "above", "over"} or match.group("op") == ">" else "gte"
            output.append(ExtractedProperty(
                property_id="rating", mode="required", operator=operator, value=value,
                evidence_text=match.group(0), start_offset=match.start(), end_offset=match.end(),
                matched_alias=match.group("cue"), cue=match.group("op") or "rating",
                confidence=0.97, applied=not invalid,
                ignored_reason="rating_out_of_range" if invalid else None,
            ))
        radius_pattern = re.compile(
            r"(?:trong|within|under)\s+(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>km|kilometers?|m|meters?)\b",
            re.IGNORECASE,
        )
        for match in radius_pattern.finditer(text):
            value = float(match.group("value").replace(",", "."))
            if match.group("unit").lower().startswith("m"):
                value /= 1000.0
            output.append(ExtractedProperty(
                property_id="radius_km", mode="required", operator="lte", value=value,
                evidence_text=match.group(0), start_offset=match.start(), end_offset=match.end(),
                matched_alias=match.group("unit"), cue="radius", confidence=0.95,
            ))
        return output


def reconcile_for_category(
    extraction: PropertyExtraction,
    *,
    category_code: str,
    property_catalog: Mapping[str, Mapping[str, Any]],
    category_id: str | None = None,
    taxonomy: Taxonomy | None = None,
    profiles: PropertyProfiles | None = None,
) -> PropertyExtraction:
    """Mark category-inapplicable constraints as diagnostics, not filters."""
    output: list[ExtractedProperty] = []
    warnings = list(extraction.warnings)
    resolved_ids = (
        set(profiles.applicable_ids(taxonomy, category_id))
        if profiles is not None and taxonomy is not None and category_id is not None
        else None
    )
    for row in extraction.properties:
        definition = property_catalog.get(row.property_id)
        legacy_codes = set(str(value) for value in (definition or {}).get("mock_applicable_codes", []))
        applicable = row.property_id in resolved_ids if resolved_ids is not None else not legacy_codes or category_code in legacy_codes
        if row.applied and row.property_id not in {"rating", "radius_km"} and not applicable:
            output.append(replace(row, applied=False, ignored_reason="category_inapplicable"))
            warnings.append(f"{row.property_id} is not configured for category {category_code}")
        else:
            output.append(row)
    return PropertyExtraction(extraction.query, extraction.schema_version, tuple(output), tuple(dict.fromkeys(warnings)), extraction.failed_open)


def active_constraints(extraction: PropertyExtraction) -> tuple[ExtractedProperty, ...]:
    return tuple(row for row in extraction.properties if row.applied and row.ignored_reason is None)
