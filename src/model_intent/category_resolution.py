from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .normalization import fold_vietnamese, normalize_text
from .property_extraction import _fold_with_offsets
from .taxonomy import Taxonomy

_SHORT_PLACE_NAMES = {"atm", "gym"}
_CONTEXT_IGNORED = re.compile(
    r"(?:\b(?:gan|near|around|khong phai|not a|not an|khong can|khong tim|khong muon|"
    r"dung dua|dung cho|not looking for|not the|instead of)\s*)$", re.IGNORECASE,
)
_NEGATED_CLAUSE = re.compile(
    r"^(?:(?:tui|toi|minh|i)\s+)?(?:khong phai|khong (?:can )?(?:tim|kiem|muon)|"
    r"not (?:a |an |looking for|want|need))\b"
)
_HOMONYM_CONTEXT = {
    "dinh": re.compile(r"\bgia\s*$", re.IGNORECASE),
    "dinh co": re.compile(r"\bgia\s*$", re.IGNORECASE),
}


@dataclass(frozen=True)
class ExplicitCategory:
    item_id: str
    evidence: str
    start_offset: int | None = None
    end_offset: int | None = None


def unique_leaf_name(
    category_text: str,
    taxonomy: Taxonomy,
    matches: tuple[ExplicitCategory, ...],
) -> ExplicitCategory | None:
    """Match an entire leaf name, not a category mentioned within another request."""
    if len(matches) != 1:
        return None
    normalized = normalize_text(category_text)
    for prefix in ("tôi muốn tìm ", "cho tôi tìm ", "tìm ", "toi muon tim ", "cho toi tim ", "tim ", "find ", "search for "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    if not normalized:
        return None
    named_ids = {
        item_id for item_id, item in taxonomy.items.items()
        if not taxonomy.has_children(item_id)
        and any(normalize_text(name) == normalized for name in item.names.values())
    }
    return matches[0] if named_ids == {matches[0].item_id} else None


def explicit_categories(
    query: str,
    taxonomy: Taxonomy,
    enrichment: Mapping[str, Mapping[str, Any]],
    *,
    property_spans: tuple[tuple[int, int], ...] = (),
) -> tuple[ExplicitCategory, ...]:
    """Find explicitly named leaf place types, not inferred amenities.

    This is a fallback for the POC's fake embedder when property-rich text
    distracts the semantic category decision. Multiple distinct place types
    remain ambiguous rather than silently selecting the longest unrelated one.
    """
    folded, starts, ends = _fold_with_offsets(query)
    matches: list[tuple[int, int, str, str]] = []
    for item_id, item in taxonomy.items.items():
        if taxonomy.has_children(item_id):
            continue
        aliases = enrichment.get(item_id) or {}
        phrases = [*item.names.values(), *(aliases.get("aliases_vi") or ()), *(aliases.get("aliases_en") or ())]
        for phrase in phrases:
            normalized = normalize_text(fold_vietnamese(str(phrase)))
            if not normalized or (len(normalized) < 4 and normalized not in _SHORT_PLACE_NAMES):
                continue
            pattern = r"[\s\-]+".join(re.escape(part) for part in normalized.split())
            for match in re.finditer(rf"(?<!\w){pattern}(?!\w)", folded):
                start, end = starts[match.start()], ends[match.end() - 1]
                if any(start < prop_end and end > prop_start for prop_start, prop_end in property_spans):
                    continue
                preceding = normalize_text(fold_vietnamese(query[max(0, start - 24):start]))
                clause_start = max(
                    query.rfind(",", 0, start), query.rfind(";", 0, start), query.rfind(".", 0, start),
                ) + 1
                clause_prefix = normalize_text(fold_vietnamese(query[clause_start:start]))
                if _NEGATED_CLAUSE.match(clause_prefix) and not any(
                    cue in clause_prefix for cue in ("ma la", "nhung", "but", "instead")
                ):
                    continue
                if _CONTEXT_IGNORED.search(preceding) or (
                    (homonym := _HOMONYM_CONTEXT.get(normalized)) is not None
                    and homonym.search(preceding)
                ):
                    continue
                matches.append((start, end, item_id, query[start:end]))

    # Eliminate nested generic names such as "restaurant" inside "Japanese
    # restaurant", but retain distinct categories at the same position.
    longest = sorted(set(matches), key=lambda row: (-(row[1] - row[0]), row[0], row[2]))
    selected: list[tuple[int, int, str, str]] = []
    for match in longest:
        if any(match[0] < end and match[1] > start and (match[0], match[1]) != (start, end)
               for start, end, _, _ in selected):
            continue
        selected.append(match)
    by_id: dict[str, tuple[str, int, int]] = {}
    for start, end, item_id, phrase in selected:
        by_id.setdefault(item_id, (phrase, start, end))
    return tuple(
        ExplicitCategory(item_id, evidence, start, end)
        for item_id, (evidence, start, end) in sorted(by_id.items())
    )
