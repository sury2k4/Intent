from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from .category_resolution import ExplicitCategory, explicit_categories
from .normalization import fold_vietnamese, normalize_text
from .property_extraction import _fold_with_offsets
from .property_schemas import ExtractedProperty, PropertyExtraction
from .requested_search import extract_requested_search
from .query_scope import non_place_reason
from .taxonomy import Taxonomy

_CUE_SUFFIX = re.compile(
    r"(?i)(?:\s*[,;]\s*|\s*\b(?:có|co|không có|khong co|không cần|khong can|"
    r"ưu tiên|uu tien|nếu có|neu co|without(?: a| an)?|with|đừng(?: có)?|dung(?: co)?|không|khong|no(?: a| an)?|and|và|va|nhưng|nhung|but|"
    r"phục vụ|phuc vu|chuyên về|chuyen ve|hỗ trợ|ho tro|yêu cầu|yeu cau)\b\s*)+$"
)
_CONSTRAINT_CUE = re.compile(
    r"(?:\b(?:co|khong co|khong can|uu tien|neu co|with|without(?: a| an)?|no(?: a| an)?|dung co|"
    r"phuc vu|cho|chuyen ve|ho tro|yeu cau)\s*)$"
)
_WEAK_WORDS = {"co", "khong", "uu", "tien", "neu", "va", "nhung", "with", "without", "no", "and", "but"}
_PROPERTY_CUE = re.compile(
    r"(?i)(?<!\w)(?:không\s+(?:có|cần)|khong\s+(?:co|can)|"
    r"đừng(?:\s+có)?|dung(?:\s+co)?|ưu\s+tiên|uu\s+tien|"
    r"bắt\s+buộc(?:\s+có)?|bat\s+buoc(?:\s+co)?|phải\s+có|phai\s+co|"
    r"cần(?:\s+có)?|can(?:\s+co)?|nếu\s+có|neu\s+co|không|khong|"
    r"phục\s+vụ|phuc\s+vu|"
    r"cho\s+(?:thanh\s+toán|chuyển\s+khoản|quét|quẹt|phép|khách|gửi|đậu|sạc|xạc)|"
    r"cho\s+(?:thanh\s+toan|chuyen\s+khoan|quet|phep|khach|gui|dau|sac|xac)|"
    r"chuyên\s+(?:về|phục\s+vụ|bán)|chuyen\s+(?:ve|phuc\s+vu|ban)|"
    r"hỗ\s+trợ|ho\s+tro|yêu\s+cầu|yeu\s+cau|"
    r"must\s+have|do\s+not\s+need|not\s+required|with|without|need|prefer|có|co)(?!\w)"
)
_LOCATION_ONLY = re.compile(r"^(?:ở|o|tại|tai|gần|gan|trong|near|around|in|at)\b", re.IGNORECASE)
_PLACE_EXISTENCE = re.compile(r"^(?:có|co)\s+(?:tiệm|tiem|quán|quan|cửa hàng|cua hang|nhà|nha|trạm|tram|chỗ|cho|địa điểm|dia diem)\s+(?:nào|nao|không|khong)\b", re.IGNORECASE)
_QUESTION_TAIL = re.compile(r"(?:\b(?:không|khong|ko|nào|nao|gì|gi|how|whether)\b\s*\??|\?)\s*$", re.IGNORECASE)
_QUESTION_TARGET = re.compile(r"^(?:nào|nao|gì|gi|ở đâu|o dau|where|which)\b", re.IGNORECASE)
_POLITE_TAIL = re.compile(r"^(?:(?:chỉ|chi|giúp|giup|cho)\s+(?:giùm|gium|giúp|giup|dễ|de|mình|minh|tui|tôi|toi|với|voi)\b|cho\s+khách\b)", re.IGNORECASE)
_ALTERNATIVE = re.compile(r"\b(?:hoac|hay|or)\s+(?:co\s+)?\w", re.IGNORECASE)
_UNRESOLVED_SKIP = re.compile(
    r"^(?:if possible|would be nice|ở|o|gần|gan|tại|tai|near|around|in|at)\b",
    re.IGNORECASE,
)
_UNRESOLVED_CONNECTOR = re.compile(r"[,;]|(?<!\w)(?:và|va|nhưng|nhung|and|but)(?!\w)", re.IGNORECASE)
_SEARCH_ONLY = re.compile(r"^(?:tìm|tim|find(?: a| an)?|search for(?: a| an)?)$", re.IGNORECASE)

_CONVERSATIONAL_PREFIXES = re.compile(
    r"(?i)^(?:(?:cho\s+(?:mình|minh|tôi|toi|em)\s+hỏi(?:\s+là)?|"
    r"cho\s+hỏi(?:\s+là)?|"
    r"chỉ\s+(?:tôi|toi|mình|minh|em|giúp|giup)|"
    r"gợi\s+ý(?:\s+(?:cho\s+)?(?:tôi|toi|mình|minh|em|giúp))?|"
    r"tìm\s+(?:giúp|giup)(?:\s+(?:mình|minh|tôi|toi|em))?|"
    r"cần\s+tìm|can\s+tim|"
    r"muốn\s+(?:tìm|tim|ghé|ghe|đi|di|đến|den)|"
    r"ở\s+đâu(?:\s+có)?|o\s+dau(?:\s+co)?|"
    r"nơi\s+nào(?:\s+có)?|noi\s+nao(?:\s+co)?|"
    r"chỗ\s+nào(?:\s+có)?|cho\s+nao(?:\s+co)?|"
    r"quanh\s+đây(?:\s+có)?|quanh\s+day(?:\s+co)?|"
    r"gần\s+đây(?:\s+có)?|gan\s+day(?:\s+co)?)\s*)+(?:có|co)?\s*",
)

_SPATIAL_AND_RADIUS = re.compile(
    r"(?i)\b(?:"
    r"trong\s+bán\s+kính\s+\d+(?:[.,]\d+)?\s*(?:km|m)?|"
    r"bán\s+kính\s+\d+(?:[.,]\d+)?\s*(?:km|m)?|"
    r"cách\s+(?:đây\s+)?\d+(?:[.,]\d+)?\s*(?:km|m)?|"
    r"gần\s+đây\s+\d+(?:[.,]\d+)?\s*(?:km|m)?|"
    r"quận\s+(?:\d+|bình\s+thạnh|thủ\s+đức|gò\s+vấp|tân\s+bình|phú\s+nhuận|tân\s+phú|bình\s+tân)|"
    r"(?:(?:ở|tại|trong)\s+)?phường\s+(?:\d+[a-zA-Z]?|[^\W\d_]+(?:\s+[^\W\d_]+){0,2})|"
    r"gần\s+(?:chợ\s+bến\s+thành|nhà\s+thờ\s+đức\s+bà|phố\s+đi\s+bộ|landmark\s*81|sân\s+bay(?:\s+tân\s+sơn\s+nhất)?|hồ\s+con\s+rùa)"
    r")\b"
)

_TEENCODE_MAP = {
    "fong": "phòng",
    "fòg": "phòng",
    "phòg": "phòng",
    "hag": "hàng",
    "hàg": "hàng",
    "vjp": "vip",
    "trog": "trong",
    "ks": "khách sạn",
    "bv": "bệnh viện",
    "tttm": "trung tâm thương mại",
    "nt": "nhà thuốc",
}

_GENERIC_PLACEHOLDERS = {
    "chỗ nào", "nơi nào", "ở đâu", "địa điểm nào", "quanh đây",
    "cho nao", "noi nao", "o dau", "dia diem nao", "quanh day"
}

_INVERTED_PURPOSE = re.compile(
    r"(?i)\b(?:để|de|nhằm|nham|cho\s+việc|cho\s+viec)\s+(?:ngồi|ngoi|uống|uong|đi|di|ăn|an|tập|tap|vào|vao|ghé|ghe)?\s*([a-z0-9\s\u00C0-\u1EF9]+)$"
)


@dataclass(frozen=True)
class InterpretedQuery:
    original: str
    category_text: str
    extraction: PropertyExtraction
    category_matches: tuple[ExplicitCategory, ...]
    property_roles: tuple[Mapping[str, Any], ...]
    focused_is_usable: bool
    focused_reason: str
    category_boundary: Mapping[str, Any] | None = None
    unresolved_property_clauses: tuple[Mapping[str, Any], ...] = ()
    requested_search: Mapping[str, Any] | None = None

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
            "category_boundary": dict(self.category_boundary) if self.category_boundary else None,
            "unresolved_property_clauses": [dict(row) for row in self.unresolved_property_clauses],
            "requested_search": {
                "sort_candidates": [dict(row) for row in (self.requested_search or {}).get("sort_candidates", ())],
            },
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
    chars: list[str] = []
    offsets: list[int] = []
    index = 0
    while index < len(preceding):
        original_start = index
        index += 1
        while index < len(preceding) and unicodedata.combining(preceding[index]):
            index += 1
        for char in unicodedata.normalize("NFC", preceding[original_start:index]):
            chars.append(char)
            offsets.append(original_start)
    suffix = _CUE_SUFFIX.search("".join(chars))
    if suffix is not None:
        start = offsets[suffix.start()]
    return start, end


def _first_property_boundary(
    text: str, matches: list[ExplicitCategory], roles: list[dict[str, Any]], heads: set[str],
) -> tuple[int, int] | None:
    folded, starts, ends = _fold_with_offsets(text)
    for cue in _PROPERTY_CUE.finditer(folded):
        cue_start, cue_end = starts[cue.start()], ends[cue.end() - 1]
        prefix = text[:cue_start].strip(" ,;.!?")
        suffix = text[cue_end:].strip(" ,;.!?")
        if not prefix or not suffix:
            continue
        folded_suffix = normalize_text(fold_vietnamese(text[cue_end:]))
        if _ALTERNATIVE.search(folded_suffix):
            alternatives = list(_ALTERNATIVE.finditer(folded_suffix))
            if not all(
                folded_suffix[alternative.start():].split(" ", 1)[-1] in {"khong", "khong co", "not"}
                or any(
                    row["role"] == "constraint"
                    and re.sub(r"^(?:co|khong co)\s+", "", folded_suffix[alternative.start():].split(" ", 1)[-1]).startswith(
                        normalize_text(fold_vietnamese(row["evidence"]))
                    )
                    for row in roles
                ) for alternative in alternatives
            ):
                continue
        preceding = [match for match in matches if match.end_offset is not None and match.end_offset <= cue_start]
        if not preceding and any(match.start_offset is not None and match.start_offset >= cue_end for match in matches):
            continue
        recognized = any(row["role"] == "constraint" and row["start_offset"] >= cue_end for row in roles)
        venue_heads = heads | {"quan", "tiem", "nha", "cho", "dia", "tram", "resort", "hotel", "cf", "phong", "khach", "san"}
        if not preceding:
            words = normalize_text(fold_vietnamese(prefix)).split()
            if not recognized or len(words) < 2 or not any(word in venue_heads for word in words[-3:]):
                continue
        if not recognized and (
            _LOCATION_ONLY.match(suffix) or _PLACE_EXISTENCE.match(text[cue_start:])
            or _QUESTION_TARGET.match(suffix)
        ):
            continue
        return cue_start, cue_end
    return None


def _unresolved_property_clauses(
    text: str, start: int, removed_spans: list[tuple[int, int]],
    short_property_candidate: Callable[[str], tuple[str, str] | None] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    spans = sorted((max(start, left), min(len(text), right)) for left, right in removed_spans if right > start)
    gaps: list[tuple[int, int]] = []
    position = start
    for left, right in spans:
        if left > position:
            gaps.append((position, left))
        position = max(position, right)
    if position < len(text):
        gaps.append((position, len(text)))
    unresolved: list[Mapping[str, Any]] = []
    weak = _WEAK_WORDS | {"can", "phai", "bat", "buoc", "do", "not", "need", "must", "have", "prefer", "hoac", "hay", "or", "nhung", "dung"}

    def meaningful(left: int, right: int) -> bool:
        words = [word for word in normalize_text(fold_vietnamese(text[left:right])).split() if word not in weak]
        return len(words) >= 2

    for left, right in gaps:
        sections = [(left, right)]
        for connector in _UNRESOLVED_CONNECTOR.finditer(text, left, right):
            for position, (part_start, part_end) in enumerate(sections):
                if not part_start <= connector.start() < part_end:
                    continue
                before = (part_start, connector.start())
                after = (connector.end(), part_end)
                if meaningful(*before) and meaningful(*after):
                    sections[position:position + 1] = [before, after]
                elif meaningful(*before) and not text[connector.end():part_end].strip(" ,;.!?"):
                    sections[position] = before
                elif not meaningful(*before) and meaningful(*after):
                    sections[position] = after
                break
        for left, right in sections:
            while left < right and (text[left].isspace() or text[left] in ",;.!?"):
                left += 1
            while right > left and (text[right - 1].isspace() or text[right - 1] in ",;.!?"):
                right -= 1
            if left < right and (_POLITE_TAIL.match(text[left:right]) or _UNRESOLVED_SKIP.match(text[left:right])):
                continue
            words = [word for word in normalize_text(fold_vietnamese(text[left:right])).split() if word not in weak]
            short_candidate = (
                short_property_candidate(words[0]) if short_property_candidate and len(words) == 1
                else None
            )
            if left < right and (meaningful(left, right) or short_candidate is not None):
                unresolved.append({
                    "text": text[left:right], "start_offset": left, "end_offset": right,
                    "reason": "unrecognized_property_clause",
                })
    return tuple(unresolved)


def _merge_pending_requests(
    text: str, properties: list[ExtractedProperty], requests: list[dict[str, Any]],
) -> list[ExtractedProperty]:
    merged = list(properties)
    for request in requests:
        start, end = request["start_offset"], request["end_offset"]
        status = str(request["status"])
        required_data = tuple(str(value) for value in request["requires"])
        overlapping_rating = next((
            index for index, row in enumerate(merged)
            if row.property_id == "rating" and row.start_offset is not None and row.end_offset is not None
            and start < row.end_offset and end > row.start_offset
        ), None)
        if overlapping_rating is not None:
            row = merged[overlapping_rating]
            if row.ignored_reason == "rating_out_of_range":
                continue
            if request["kind"] == "review_rating":
                merged[overlapping_rating] = replace(
                    row, status=status, requires=required_data,
                    applied=False, ignored_reason="rating_unverified",
                )
                continue
            merged.pop(overlapping_rating)
        if any(
            row.start_offset is not None and row.end_offset is not None
            and start < row.end_offset and end > row.start_offset
            for row in merged
        ):
            continue
        kind = str(request["kind"])
        merged.append(ExtractedProperty(
            property_id="rating" if kind == "review_rating" else kind,
            mode="preferred" if kind == "quality_by_reviews" else "required",
            operator=str(request.get("operator", "near" if kind == "near" else "rank" if kind == "quality_by_reviews" else "eq")),
            value=request.get("value"), evidence_text=text[start:end],
            start_offset=start, end_offset=end, matched_alias=text[start:end],
            cue="requested_search", confidence=0.0, applied=False,
            ignored_reason="rating_unverified" if kind == "review_rating" else status,
            status=status, requires=required_data,
        ))
    return sorted(merged, key=lambda row: (
        row.start_offset if row.start_offset is not None else len(text),
        row.end_offset if row.end_offset is not None else len(text),
    ))


def interpret_query(
    text: str,
    extraction: PropertyExtraction,
    taxonomy: Taxonomy,
    enrichment: Mapping[str, Mapping[str, Any]],
    *,
    additional_matches: tuple[ExplicitCategory, ...] = (),
    short_property_candidate: Callable[[str], tuple[str, str] | None] | None = None,
) -> InterpretedQuery:
    """Resolve category/attribute overlap before constructing E5 query variants."""
    all_matches = (*explicit_categories(text, taxonomy, enrichment), *additional_matches)
    heads = _prefix_heads(taxonomy, enrichment)
    category_codes = {item.code for item_id, item in taxonomy.items.items() if not taxonomy.has_children(item_id)}
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
        elif (
            not all_matches and row.property_id in category_codes
            and _SEARCH_ONLY.match(normalize_text(fold_vietnamese(text[:start])))
            and not text[end:].strip(" ,;.!?")
        ):
            role = "possible_category_target"
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

    pending_filters, sort_requests, requested_spans = extract_requested_search(
        text, category_matches=chosen, properties=resolved_properties,
    )
    resolved_properties = _merge_pending_requests(text, resolved_properties, pending_filters)
    for start, end in requested_spans:
        left, _ = _constraint_removal(text, start, end)
        if left == start:
            prefix = re.search(r"(?i)\b(?:ở|tại|in|at)\s*$", text[:start])
            if prefix:
                left = prefix.start()
        removed_spans.append((left, end))

    cue = _first_property_boundary(text, chosen, roles, heads) if not extraction.failed_open else None
    if cue is None and not chosen and not extraction.failed_open and not non_place_reason(text):
        folded, starts, ends = _fold_with_offsets(text)
        leading_space = len(folded) - len(folded.lstrip())
        leading = _PROPERTY_CUE.match(folded, leading_space)
        if leading and not _PLACE_EXISTENCE.match(text[starts[leading.start()]:]):
            leading_end = ends[leading.end() - 1]
            tail = text[leading_end:].strip(" ,;.!?\n")
            words = normalize_text(fold_vietnamese(tail)).split()
            if (
                (2 <= len(words) <= 6 or (
                    len(words) == 1 and short_property_candidate is not None
                    and short_property_candidate(words[0]) is not None
                ))
                and not any(char in text for char in ",;?!")
                and not set(words) & {"nao", "khong", "gan", "tim", "kiem", "find", "where", "near", "instead"}
            ):
                cue = (0, leading_end)
    unresolved: tuple[Mapping[str, Any], ...] = ()
    boundary: Mapping[str, Any] | None = None
    if cue is not None:
        cue_start, cue_end = cue
        preceding_cats = [m for m in chosen if m.start_offset is not None and m.end_offset <= cue_start]
        if preceding_cats:
            chosen = preceding_cats
        unresolved = _unresolved_property_clauses(
            text, cue_start, removed_spans, short_property_candidate,
        )
        boundary = {"cue": text[cue_start:cue_end], "start_offset": cue_start, "end_offset": cue_end}
        removed_spans.append((cue_start, len(text)))

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
    if focused:
        cleaned_prefix = _CONVERSATIONAL_PREFIXES.sub("", focused).strip(" ,.;!?")
        if cleaned_prefix and len(cleaned_prefix.split()) >= 1:
            focused = cleaned_prefix
        # Ward names can contain several words, but must not swallow a following
        # recognized category (e.g. "phường Bến Thành quán cà phê").
        protected_starts = [
            found.start() for match in chosen if match.evidence
            if (found := re.search(re.escape(match.evidence), focused, re.IGNORECASE))
        ]

        def strip_spatial(match: re.Match[str]) -> str:
            if re.search(r"\bphường\b", match.group(), re.IGNORECASE):
                protected = [start for start in protected_starts if match.start() < start < match.end()]
                if protected:
                    return focused[min(protected):match.end()]
            return ""

        cleaned_geo = _SPATIAL_AND_RADIUS.sub(strip_spatial, focused).strip(" ,.;!?")
        if cleaned_geo and len(cleaned_geo.split()) >= 1:
            focused = cleaned_geo
        raw_words = focused.split()
        norm_words = [_TEENCODE_MAP.get(w.lower(), w) for w in raw_words]
        focused = " ".join(norm_words)
        folded_focused = normalize_text(fold_vietnamese(focused))
        if folded_focused in _GENERIC_PLACEHOLDERS:
            m = _INVERTED_PURPOSE.search(text)
            if m and m.group(1).strip():
                cand_target = m.group(1).strip()
                if len(cand_target.split()) >= 1:
                    focused = cand_target

    category_text_missing = not focused
    if category_text_missing:
        focused = text
    words = [word for word in normalize_text(fold_vietnamese(focused)).split() if word not in _WEAK_WORDS]
    usable = not category_text_missing and (bool(chosen) or len(words) >= 2)
    if category_text_missing:
        reason = "no_category_text_after_property_removal"
    elif focused == text:
        reason = "no_safe_constraints_to_remove"
    elif not usable:
        reason = "focused_text_has_no_meaningful_place_target"
    elif cue is not None:
        reason = "property_clause_boundary"
    else:
        reason = "category_evidence_preserved"
    return InterpretedQuery(
        text, focused,
        PropertyExtraction(extraction.query, extraction.schema_version, tuple(resolved_properties), extraction.warnings, extraction.failed_open),
        tuple(chosen), tuple(roles), usable, reason, boundary, unresolved,
        requested_search={"sort_candidates": sort_requests},
    )
