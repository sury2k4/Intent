from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .category_resolution import ExplicitCategory, explicit_categories
from .normalization import fold_vietnamese, normalize_text
from .poi_schemas import ExtractedProperty, PropertyExtraction
from .taxonomy import Taxonomy

_CUE_SUFFIX = re.compile(
    r"(?i)(?:\s*[,;]\s*|\s*\b(?:có|co|không có|khong co|không cần|khong can|"
    r"ưu tiên|uu tien|nếu có|neu co|with|without|no|and|và|va|nhưng|nhung|but)\b\s*)+$"
)
_CONSTRAINT_CUE = re.compile(r"(?:\b(?:co|khong co|khong can|uu tien|neu co|with|without|no)\s*)$")
_WEAK_WORDS = {"co", "khong", "uu", "tien", "neu", "va", "nhung", "with", "without", "no", "and", "but"}


@dataclass(frozen=True)
class InterpretedQuery:
    original: str
    category_text: str
    extraction: PropertyExtraction
    category_matches: tuple[ExplicitCategory, ...]
    property_roles: tuple[Mapping[str, Any], ...]
    focused_is_usable: bool
    focused_reason: str

    @property
    def variants(self) -> tuple[str, ...]:
        if self.category_text != self.original and self.focused_is_usable:
            return (self.category_text, self.original)
        return (self.original,)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": "joint-query-interpretation-v1",
            "raw_query": self.original,
            "category_text": self.category_text,
            "semantic_variants": list(self.variants),
            "category_matches": [
                {"item_id": match.item_id, "evidence": match.evidence,
                 "start_offset": match.start_offset, "end_offset": match.end_offset}
                for match in self.category_matches
            ],
            "property_roles": [dict(row) for row in self.property_roles],
            "focused_is_usable": self.focused_is_usable,
            "focused_reason": self.focused_reason,
        }


def _prefix_heads(taxonomy: Taxonomy, enrichment: Mapping[str, Mapping[str, Any]]) -> set[str]:
    heads: set[str] = set()
    for item_id, item in taxonomy.items.items():
        if taxonomy.has_children(item_id):
            continue
        data = enrichment.get(item_id) or {}
        for phrase in (*item.names.values(), *(data.get("aliases_vi") or ()), *(data.get("aliases_en") or ())):
            words = normalize_text(fold_vietnamese(phrase)).split()
            if len(words) >= 2 and len(words[0]) >= 3:
                heads.add(words[0])
    return heads


def _category_modifier(text: str, start: int, heads: set[str]) -> bool:
    prefix = normalize_text(fold_vietnamese(text[:start]))
    words = prefix.split()
    return (
        bool(words)
        and len(words) <= 3
        and words[-1] in heads
        and not _CONSTRAINT_CUE.search(prefix)
    )


def _constraint_removal(text: str, start: int, end: int) -> tuple[int, int]:
    preceding = text[:start]
    suffix = _CUE_SUFFIX.search(preceding)
    if suffix is not None:
        start = suffix.start()
    return start, end


def interpret_query(
    text: str,
    extraction: PropertyExtraction,
    taxonomy: Taxonomy,
    enrichment: Mapping[str, Mapping[str, Any]],
) -> InterpretedQuery:
    """Resolve category/attribute overlap before constructing E5 query variants."""
    all_matches = explicit_categories(text, taxonomy, enrichment)
    heads = _prefix_heads(taxonomy, enrichment)
    chosen: list[ExplicitCategory] = list(all_matches)
    roles: list[dict[str, Any]] = []
    resolved_properties: list[ExtractedProperty] = []
    removed_spans: list[tuple[int, int]] = []
    for row in extraction.properties:
        start, end = row.start_offset, row.end_offset
        if start is None or end is None:
            resolved_properties.append(row)
            continue
        overlaps = [
            match for match in all_matches
            if match.start_offset is not None and start < match.end_offset and end > match.start_offset
        ]
        separate_category = any(
            match.start_offset is not None and match.end_offset <= start and match not in overlaps
            for match in chosen
        )
        if overlaps and separate_category and _CONSTRAINT_CUE.search(
            normalize_text(fold_vietnamese(text[max(0, overlaps[0].start_offset - 20):overlaps[0].start_offset]))
        ):
            end = max(end, *(match.end_offset for match in overlaps))
            chosen = [match for match in chosen if match not in overlaps]
            overlaps = []
        if overlaps:
            role = "category_component"
            resolved_properties.append(replace(row, applied=False, ignored_reason=role))
        elif not all_matches and _category_modifier(text, start, heads):
            role = "dual_role_category_modifier"
            resolved_properties.append(row)
        else:
            role = "constraint"
            resolved_properties.append(row)
            removed_spans.append(_constraint_removal(text, start, end))
        roles.append({
            "property_id": row.property_id, "evidence": row.evidence_text,
            "start_offset": start, "end_offset": end, "role": role,
        })

    merged: list[tuple[int, int]] = []
    for start, end in sorted(removed_spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    offset = 0
    for start, end in merged:
        pieces.append(text[offset:start])
        offset = end
    pieces.append(text[offset:])
    focused = re.sub(r"\s+", " ", "".join(pieces)).strip(" ,.;")
    if not focused:
        focused = text
    words = [word for word in normalize_text(fold_vietnamese(focused)).split() if word not in _WEAK_WORDS]
    usable = bool(chosen) or len(words) >= 2
    if focused == text:
        reason = "no_safe_constraints_to_remove"
    elif not usable:
        reason = "focused_text_has_no_meaningful_place_target"
    else:
        reason = "category_evidence_preserved"
    return InterpretedQuery(
        text, focused,
        PropertyExtraction(extraction.query, extraction.schema_version, tuple(resolved_properties), extraction.warnings, extraction.failed_open),
        tuple(chosen), tuple(roles), usable, reason,
    )
