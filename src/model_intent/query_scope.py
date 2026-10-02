from __future__ import annotations

import re

from .normalization import fold_vietnamese, normalize_text

_INFORMATIONAL = re.compile(r"^(?:cach (?:mo|dang ky|su dung)|how (?:do i reset|to (?:open|register|use))|tai sao|why)\b")
_SEARCH_VERB = re.compile(r"\b(?:tim|find|search|looking for)\b")
_DIGITAL_PRODUCT = re.compile(r"\b(?:phan mem|software|ung dung|application|app)\b")
_PHYSICAL_PLACE = re.compile(
    r"\b(?:cua hang|tiem|quan (?:ca phe|cafe|an|tra|nuoc)|nha hang|tram|cho ban|shop|store|restaurant|cafe|pharmacy|station)\b"
)


def non_place_reason(query: str) -> str | None:
    """Veto unmistakably informational/digital requests, not place queries."""
    normalized = normalize_text(fold_vietnamese(query))
    if _INFORMATIONAL.search(normalized) and not _SEARCH_VERB.search(normalized):
        return "informational_not_place_search"
    if _DIGITAL_PRODUCT.search(normalized) and not _PHYSICAL_PLACE.search(normalized):
        return "digital_product_not_place_search"
    return None
