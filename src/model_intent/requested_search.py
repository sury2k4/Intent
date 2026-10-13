"""Extract requested POI filters and ordering without executing a POI search."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from .normalization import fold_vietnamese, normalize_text
from .property_extraction import _fold_with_offsets
from .property_schemas import ExtractedProperty
from .query_scope import non_place_reason

_NEAREST = re.compile(r"(?<!\w)(?:gan\s+nhat|nearest|closest)(?!\w)")
_NEAR = re.compile(r"(?<!\w)(?:gan\s+day|quanh\s+day|nearby|near\s+me)(?!\w)")
_QUALITY = re.compile(r"(?<!\w)(?:ngon\s+mieng|ngon|xin|tasty|delicious|high\s+quality)(?!\w)")
_STARS = re.compile(r"(?<![\w.,])(?P<value>[1-5](?:[.,]\d+)?)\s*(?:sao|stars?)(?!\w)")
_REVIEW_CUE = re.compile(r"\b(?:danh\s+gia|diem\s+danh\s+gia|rating|review|rated|reviews)\b")
_HOTEL_CONTEXT = re.compile(r"\b(?:khach\s+san|hotel|resort)\b")
_TEMPORAL_NEAR = re.compile(r"\b(?:dao|thoi\s+gian|moi\s+mo|moi\s+khai\s+truong|vua\s+mo)\s*$")
_FOR_PLACE = re.compile(r"^(?:cho|for)\s+(?:quán|quan|nhà|nha|tiệm|tiem|cửa hàng|cua hang|hotel|restaurant|cafe|shop|store)\b")
_VENUE_PREFIX = re.compile(r"\b(?:quán|quan|nhà hàng|nha hang|tiệm|tiem|cửa hàng|cua hang|restaurant|cafe|shop|store)\s*$")
_NEGATED = re.compile(r"\b(?:khong(?:\s+(?:can|co|muon|thich))?|not(?:\s+(?:looking\s+for|want|needed))?|dont\s+want)\s*$")


def _candidate(kind: str, status: str, requires: list[str], text: str, start: int, end: int, **extra: Any) -> dict[str, Any]:
    return {
        "kind": kind, "status": status, "requires": requires,
        "evidence_text": text[start:end], "start_offset": start, "end_offset": end,
        **extra,
    }


def extract_requested_search(
    text: str, *, category_matches: Iterable[Any] = (), properties: Iterable[ExtractedProperty] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[tuple[int, int], ...]]:
    """Return pending property requests, sort requests and removable original-text spans."""
    requested: dict[str, list[dict[str, Any]]] = {"filter_candidates": [], "sort_candidates": []}
    if non_place_reason(text):
        return [], [], ()
    folded, starts, ends = _fold_with_offsets(text)
    matches = tuple(category_matches)
    ratings = tuple(row for row in properties if row.property_id == "rating" and row.start_offset is not None)
    spans: list[tuple[int, int]] = []
    covered: list[tuple[int, int]] = []

    def protected(start: int, end: int) -> bool:
        return any(
            match.start_offset is not None and match.end_offset is not None
            and start < match.end_offset and end > match.start_offset
            for match in matches
        )

    def append_filter(kind: str, status: str, requires: list[str], start: int, end: int, **extra: Any) -> None:
        requested["filter_candidates"].append(_candidate(kind, status, requires, text, start, end, **extra))

    def append_sort(kind: str, status: str, requires: list[str], start: int, end: int, **extra: Any) -> None:
        requested["sort_candidates"].append(_candidate(kind, status, requires, text, start, end, **extra))

    for pattern, tag in ((_NEAREST, "nearest"), (_NEAR, "near"), (_QUALITY, "quality"), (_STARS, "stars")):
        for match in pattern.finditer(folded):
            start, end = starts[match.start()], ends[match.end() - 1]
            if protected(start, end) or any(start < right and end > left for left, right in covered):
                continue
            prefix = normalize_text(fold_vietnamese(text[max(0, start - 35):start]))
            if _NEGATED.search(prefix):
                continue
            if tag in {"near", "nearest"} and _TEMPORAL_NEAR.search(prefix):
                continue
            if tag == "quality" and re.search(r"\b(?:khong|khong co|not|without|no)\s+(?:duoc\s+)?$", prefix):
                continue
            if tag == "stars":
                clause_prefix = normalize_text(fold_vietnamese(text[max(0, start - 65):start])).split(",")[-1]
                if re.search(r"\b(?:khong\s+can|khong\s+muon|not\s+required|without)\s+(?:(?:khach\s+san|hotel|rating|danh\s+gia)\s+)?$", clause_prefix):
                    continue
            if tag == "stars" and not 1 <= float(match.group("value").replace(",", ".")) <= 5:
                continue
            if tag == "quality" and normalize_text(folded[match.start():match.end()]) == "xin" and unicodedata.normalize("NFC", text[start:end]).casefold() != "xịn":
                continue
            if tag == "quality" and _FOR_PLACE.search(normalize_text(fold_vietnamese(text[end:]))):
                continue
            if tag == "quality" and text[start:end][:1].isupper() and _VENUE_PREFIX.search(prefix):
                continue
            covered.append((start, end))
            spans.append((start, end))
            if tag in {"nearest", "near"}:
                required = ["user_location", "poi_coordinates"]
                append_filter("near", "pending_location", required, start, end)
                if tag == "nearest":
                    append_sort("distance_asc", "pending_location", required, start, end,
                                sort_by="distance", direction="asc", priority=1)
            elif tag == "quality":
                required = ["verified_reviews"]
                append_filter("quality_by_reviews", "pending_review_data", required, start, end)
                append_sort("review_score_desc", "pending_review_data", required, start, end,
                            sort_by="review_score", direction="desc", priority=2)
            else:
                value = float(match.group("value").replace(",", "."))
                clause_prefix = normalize_text(fold_vietnamese(text[max(0, start - 65):start])).split(",")[-1]
                clause_suffix = normalize_text(fold_vietnamese(text[end:end + 30]))
                numeric = next((row for row in ratings if row.start_offset <= start < row.end_offset
                                or start <= row.start_offset < end), None)
                if _REVIEW_CUE.search(clause_prefix):
                    append_filter("review_rating", "pending_review_data", ["verified_reviews"], start, end,
                                  operator=numeric.operator if numeric else "gte", value=value)
                elif _HOTEL_CONTEXT.search(clause_prefix) or re.match(r"^(?:\s*[- ]*\s*)(?:hotel|khach san)\b", clause_suffix):
                    append_filter("accommodation_star_class", "pending_classification_data",
                                  ["verified_accommodation_class"], start, end,
                                  operator="eq", value=value)
                else:
                    append_filter("ambiguous_stars", "needs_clarification", ["star_meaning"], start, end,
                                  value=value)

    # Existing explicit numeric ratings (even without "sao") are also pending review requests.
    for row in ratings:
        if row.ignored_reason == "rating_out_of_range" or row.end_offset is None:
            continue
        if any(row.start_offset < right and row.end_offset > left for left, right in covered):
            continue
        if protected(row.start_offset, row.end_offset):
            continue
        append_filter("review_rating", "pending_review_data", ["verified_reviews"],
                      row.start_offset, row.end_offset, operator=row.operator, value=row.value)
        spans.append((row.start_offset, row.end_offset))

    requested["filter_candidates"].sort(key=lambda row: (row["start_offset"], row["end_offset"]))
    requested["sort_candidates"].sort(key=lambda row: (row["priority"], row["start_offset"]))
    return requested["filter_candidates"], requested["sort_candidates"], tuple(sorted(spans))
