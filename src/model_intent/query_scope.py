from __future__ import annotations

import re

from .normalization import fold_vietnamese, normalize_text

_INFORMATIONAL = re.compile(r"^(?:cach (?:mo|dang ky|su dung)|how (?:do i reset|to (?:open|register|use))|tai sao|why)\b")
_EXPLICIT_INFORMATION_REQUEST = re.compile(
    r"\b(?:muon hoi cach|chi cach|(?:chi|xin|muon) huong dan cach|cho (?:toi|tui|minh) biet .*?thuoc .*?(?:uong sao|dung sao|lieu dung)|"
    r"muon biet gia .*?(?:ngay mai|tang|giam)|xem giup xet nghiem .*?co bi benh|"
    r"(?:chi hoi|muon hoi|muon biet|can tu van|xin tu van) .*?thuoc .*?(?:uong|dung|lieu|sao))\b"
)
_NEGATED_ONLY = re.compile(r"^(?:(?:tui|toi|minh|i) )?(?:khong phai|khong (?:can )?(?:tim|kiem|muon)|(?:do )?not (?:a |an |looking for|want|need))\b")
_SEARCH_VERB = re.compile(r"\b(?:tim|find|search|looking for)\b")
_LEADING_SEARCH = re.compile(r"^(?:(?:tim|find|search|looking for)(?: a| an)? )?")
_CORRECTION_CUE = re.compile(r"[,;:]|\b(?:ma la|nhung|but|instead)\b")
_PROPERTY_ONLY = re.compile(r"^(?:co|with|without|uu tien|neu co|wifi|internet|parking|bai do xe|cho do xe)\b")
_DIGITAL_PRODUCT = re.compile(r"\b(?:phan mem|software|ung dung|application|app)\b")
_PHYSICAL_PLACE = re.compile(
    r"\b(?:cua hang|tiem|quan (?:ca phe|cafe|an|tra|nuoc)|nha hang|nha thuoc|benh vien|phong kham|"
    r"tram|cho ban|shop|store|restaurant|cafe|pharmacy|hospital|clinic|station)\b"
)
_PLACE_SEEKING_PREFIX = re.compile(r"\b(?:tim|kiem|find|looking for|muon|can|need|want|nen di|cho minh|cho tui)\b")
_NEGATED_SEARCH_PREFIX = re.compile(r"\b(?:khong (?:can )?(?:tim|kiem|muon)|not (?:looking for|want|need))(?: (?:a|an))?\s*$")
_LOCATION_CUE = re.compile(r"\b(?:o|gan|quanh|dia chi|cho nao|where|near|nearby|around|close to)\b")


def has_place_seeking_anchor(query: str, start_offset: int | None) -> bool:
    if start_offset is None:
        return False
    prefix = normalize_text(fold_vietnamese(query[:start_offset]))
    return bool(_PLACE_SEEKING_PREFIX.search(prefix)) and not bool(_NEGATED_SEARCH_PREFIX.search(prefix))


def has_location_request(query: str, end_offset: int | None) -> bool:
    if end_offset is None:
        return False
    suffix = normalize_text(fold_vietnamese(query[end_offset:]))
    return bool(_LOCATION_CUE.search(suffix))


def non_place_reason(query: str) -> str | None:
    """Veto unmistakably informational/digital requests, not place queries."""
    normalized = normalize_text(fold_vietnamese(query))
    explicit_information = _EXPLICIT_INFORMATION_REQUEST.search(normalized)
    if explicit_information and not _SEARCH_VERB.search(normalized[explicit_information.end():]):
        return "informational_not_place_search"
    if _INFORMATIONAL.search(normalized) and not _SEARCH_VERB.search(normalized):
        return "informational_not_place_search"
    without_search = _LEADING_SEARCH.sub("", normalized, count=1)
    if _NEGATED_ONLY.search(without_search):
        clauses = [normalize_text(part) for part in _CORRECTION_CUE.split(fold_vietnamese(query).casefold())]
        clauses = [part for part in clauses if part]
        if len(clauses) <= 1 or all(
            _NEGATED_ONLY.search(_LEADING_SEARCH.sub("", part, count=1)) or _PROPERTY_ONLY.search(part)
            for part in clauses[1:]
        ):
            return "negated_place_only"
    digital = _DIGITAL_PRODUCT.search(normalized)
    digital_store = re.search(r"\b(?:software|phan mem)\s+(?:store|shop|cua hang)\b", normalized)
    if digital and not (_PHYSICAL_PLACE.search(normalized[:digital.start()]) or digital_store):
        return "digital_product_not_place_search"
    return None
