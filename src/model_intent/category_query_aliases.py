from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .category_resolution import ExplicitCategory
from .normalization import fold_vietnamese, normalize_text
from .property_extraction import _fold_with_offsets
from .taxonomy import Taxonomy

_PREFIXES = (
    "toi muon tim ", "cho toi tim ", "hay tim ", "tim ",
    "find a ", "find an ", "find ", "looking for a ", "looking for an ", "looking for ",
    "search for a ", "search for an ", "search for ", "search ",
)
_SUFFIXES = (" gan day", " gan toi", " o dau", " near me", " nearby")
_ALTERNATIVES = re.compile(r"\s+(?:hoac|hay|or|va|and)\s+")
_SHORT_NAMES = {"atm"}
_PROPERTY_CUES = re.compile(r"^(?:co|khong co|khong can|uu tien|neu co|with|without|prefer)\b")


@dataclass(frozen=True)
class AliasCandidate:
    category_id: str
    canonical_name: str
    phrase: str


class CategoryQueryAliases:
    """Globally unique whole-phrase names for an E5 canonical-query probe."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        enrichment: Mapping[str, Mapping[str, Any]],
        *,
        overrides_path: str | Path | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        owners: dict[str, set[str]] = defaultdict(set)
        overrides: dict[str, str] = {}
        for item_id, item in taxonomy.items.items():
            if taxonomy.has_children(item_id):
                continue
            for phrase in (
                *item.names.values(),
                *(enrichment.get(item_id) or {}).get("aliases_vi", ()),
                *(enrichment.get(item_id) or {}).get("aliases_en", ()),
            ):
                key = normalize_text(fold_vietnamese(str(phrase)))
                if len(key) >= 4 or key in _SHORT_NAMES:
                    owners[key].add(item_id)
        if overrides_path is not None:
            payload = json.loads(Path(overrides_path).read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1 or payload.get("taxonomy_version") != taxonomy.document.taxonomy_version:
                raise ValueError("category query aliases do not match the active taxonomy")
            rows = payload.get("aliases")
            if not isinstance(rows, list):
                raise ValueError("category query aliases must be a list")
            for row in rows:
                if not isinstance(row, dict):
                    raise TypeError("category query alias must be an object")
                category_id = str(row.get("category_id", ""))
                phrase = row.get("phrase")
                language = row.get("language")
                evidence = row.get("evidence")
                if (
                    category_id not in taxonomy.items or taxonomy.has_children(category_id)
                    or language not in {"vi", "en"}
                    or not isinstance(phrase, str) or not phrase.strip()
                    or not isinstance(evidence, str) or not evidence.strip()
                ):
                    raise ValueError(f"invalid curated category query alias for {category_id}")
                key = normalize_text(fold_vietnamese(phrase))
                if len(key) < 4 or key in overrides:
                    raise ValueError(f"short or duplicated category query alias: {phrase}")
                if owners.get(key) and owners[key] != {category_id}:
                    raise ValueError(f"category query alias collides with another category: {phrase}")
                overrides[key] = category_id
                owners[key].add(category_id)
        self.owners = {key: next(iter(ids)) for key, ids in owners.items() if len(ids) == 1}
        self.overrides = overrides

    def is_curated(self, phrase: str, category_id: str) -> bool:
        key = normalize_text(fold_vietnamese(phrase))
        return self.overrides.get(key) == category_id

    def is_canonical(self, phrase: str, category_id: str) -> bool:
        key = normalize_text(fold_vietnamese(phrase))
        return any(
            normalize_text(fold_vietnamese(name)) == key
            for name in self.taxonomy.items[category_id].names.values()
        )

    def leading_curated_match(self, text: str) -> ExplicitCategory | None:
        """Protect a reviewed place name before a property cue from category overlap."""
        folded, starts, ends = _fold_with_offsets(text)
        allowed_prefixes = {"", *(prefix.strip() for prefix in _PREFIXES)}
        for phrase, item_id in sorted(self.overrides.items(), key=lambda row: -len(row[0])):
            pattern = r"(?<!\w)" + r"[\s\-]+".join(re.escape(word) for word in phrase.split()) + r"(?!\w)"
            match = re.search(pattern, folded)
            if match is None:
                continue
            prefix = normalize_text(folded[:match.start()])
            suffix = normalize_text(folded[match.end():])
            if prefix in allowed_prefixes and suffix and _PROPERTY_CUES.match(suffix):
                start, end = starts[match.start()], ends[match.end() - 1]
                return ExplicitCategory(item_id, text[start:end], start, end)
        return None

    def match_correction(self, text: str, *, language: str = "vi") -> AliasCandidate | None:
        folded = fold_vietnamese(text).casefold()
        segments = re.split(r"[,;]|\b(?:ma la|nhung|but|instead)\b", folded)
        if len(segments) != 2:
            return None
        first = normalize_text(segments[0])
        if not re.match(r"^(?:(?:tim|find|search|looking for) )?(?:khong phai|not (?:a |an )?)\b", first):
            return None
        second = normalize_text(segments[1])
        if re.match(r"^(?:khong phai|not (?:a |an )?)\b", second):
            return None
        return self.match_whole_phrase(second, language=language)

    def match_alternatives(self, text: str, *, language: str = "vi") -> tuple[AliasCandidate, ...]:
        parts = _ALTERNATIVES.split(normalize_text(fold_vietnamese(text)))
        if not 2 <= len(parts) <= 4:
            return ()
        matches = [self.match_whole_phrase(part, language=language, include_canonical=True) for part in parts]
        if any(match is None for match in matches):
            return ()
        return tuple(matches) if len({match.category_id for match in matches}) >= 2 else ()

    def match_whole_phrase(
        self, text: str, *, language: str = "vi", include_canonical: bool = False,
    ) -> AliasCandidate | None:
        normalized = normalize_text(fold_vietnamese(text))
        for prefix in _PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        for suffix in _SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        category_id = self.owners.get(normalized)
        if category_id is None:
            return None
        item = self.taxonomy.items[category_id]
        canonical = item.names.get(language) or item.names.get("vi") or item.names.get("en")
        if not canonical or (not include_canonical and normalized == normalize_text(fold_vietnamese(canonical))):
            return None
        return AliasCandidate(category_id, canonical, normalized)
