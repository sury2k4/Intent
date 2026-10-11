from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .normalization import normalize_text
from .serving_config import E5_MODEL_ID, E5_MODEL_REVISION
from .property_extraction import semantic_property_cue
from .poi_properties import PropertyProfiles
from .taxonomy import Taxonomy

_PROFILE_RENDER_VERSION = "property-semantic-e5-v1"
_NEGATIVE_EMBEDDING = re.compile(
    r"(?i)\b(?:không\s+(?:có|cần|tự|đồng\s+nghĩa)|chẳng|tránh|thiếu|"
    r"without|avoid|exclude|lacks?|never|not|no|rather\s+than|distinct\s+from)\b"
)


def stable_property_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticPropertyProfile:
    property_id: str
    language: str
    profile_key: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class SemanticPropertyCorpus:
    rows: tuple[SemanticPropertyProfile, ...]
    catalog_hash: str
    source_hash: str
    taxonomy_fingerprint: str
    renderer_version: str
    content_set_hash: str
    property_count: int
    version: str

    @property
    def profile_count(self) -> int:
        return len(self.rows)


def load_property_semantic_corpus(
    path: str | Path, definitions: Mapping[str, Mapping[str, Any]], taxonomy: Taxonomy,
    embedder_metadata: Mapping[str, Any], *, token_counter: Callable[[str], int] | None = None,
) -> SemanticPropertyCorpus:
    """Validate a versioned positive-intent corpus and render stable E5 passages."""
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if (
        payload.get("schema_version") != 2
        or payload.get("version") != "property-semantic-poc-v2"
        or payload.get("catalog_version") != "poi-properties-poc-v8-osm-only"
    ):
        raise ValueError("unsupported property semantic corpus or catalog version")
    if (
        embedder_metadata.get("model_id") != E5_MODEL_ID
        or embedder_metadata.get("revision") != E5_MODEL_REVISION
        or embedder_metadata.get("query_prefix") != "query: "
        or embedder_metadata.get("document_prefix") != "passage: "
        or int(embedder_metadata.get("dimension", 0)) != 768
    ):
        raise ValueError("property profiles require the pinned E5-base 768D model and prefixes")
    queryable = {
        key for key, definition in definitions.items()
        if definition.get("queryable", True) is not False
        and definition.get("value_type") in {"boolean", "enum", "ordinal"}
    }
    records = payload.get("profiles")
    if not isinstance(records, list) or len(records) != len(queryable):
        raise ValueError("property corpus must cover exactly the queryable catalog IDs")
    ids: set[str] = set()
    rows: list[SemanticPropertyProfile] = []
    for entry in records:
        if not isinstance(entry, dict):
            raise ValueError("each property semantic profile must be an object")
        property_id = str(entry.get("property_id", ""))
        if property_id in ids or property_id not in queryable:
            raise ValueError(f"repeated or unknown queryable property: {property_id}")
        ids.add(property_id)
        definition = definitions[property_id]
        if entry.get("facet") != definition.get("facet") or entry.get("value_type") != definition.get("value_type"):
            raise ValueError(f"property metadata does not match catalog: {property_id}")
        related = entry.get("confusable_with")
        if not isinstance(related, list) or len(related) != len(set(related)) or any(
            other not in queryable or other == property_id for other in related
        ):
            raise ValueError(f"invalid confusable IDs: {property_id}")
        for language in ("vi", "en"):
            label = (entry.get("labels") or {}).get(language)
            description = (entry.get("embedding_definitions") or {}).get(language)
            phrases = (entry.get("positive_phrases") or {}).get(language)
            if not isinstance(label, str) or not label.strip() or not isinstance(description, str) or not description.strip():
                raise ValueError(f"missing bilingual label or positive embedding definition: {property_id}/{language}")
            if _NEGATIVE_EMBEDDING.search(description):
                raise ValueError(f"negative contrast in positive property profile: {property_id}/{language}")
            if not isinstance(phrases, list) or not phrases or any(not isinstance(value, str) or not value.strip() for value in phrases):
                raise ValueError(f"missing positive phrases: {property_id}/{language}")
            normalized = [normalize_text(phrase) for phrase in phrases]
            if len(set(normalized)) != len(phrases):
                raise ValueError(f"duplicate positive phrases: {property_id}/{language}")
            if any(_NEGATIVE_EMBEDDING.search(phrase) for phrase in phrases):
                raise ValueError(f"negative request in positive property phrases: {property_id}/{language}")
            if language == "vi":
                text = f"Thuộc tính: {label.strip()}\nMô tả: {description.strip()}\nCách diễn đạt: {'; '.join(value.strip() for value in phrases)}"
            else:
                text = f"Property: {label.strip()}\nDescription: {description.strip()}\nWays to ask: {'; '.join(value.strip() for value in phrases)}"
            if len(text) > 1100 or len(text.split()) > 120:
                raise ValueError(f"property profile is too long for E5: {property_id}/{language}")
            if token_counter is not None and token_counter(f"passage: {text}") > int(embedder_metadata.get("max_seq_length", 192)):
                raise ValueError(f"property profile exceeds E5 max_seq_length: {property_id}/{language}")
            profile_key = "primary"
            content_hash = stable_property_hash({
                "property_id": property_id, "language": language,
                "profile_key": profile_key, "text": text,
            })
            rows.append(SemanticPropertyProfile(property_id, language, profile_key, text, content_hash))
    if ids != queryable:
        raise ValueError(f"missing queryable property IDs: {sorted(queryable - ids)}")
    rows.sort(key=lambda row: (row.property_id, row.language, row.profile_key))
    return SemanticPropertyCorpus(
        rows=tuple(rows), catalog_hash=stable_property_hash(definitions),
        source_hash=hashlib.sha256(raw).hexdigest(), taxonomy_fingerprint=taxonomy.fingerprint(),
        renderer_version=_PROFILE_RENDER_VERSION,
        content_set_hash=stable_property_hash([(row.property_id, row.language, row.profile_key, row.content_hash) for row in rows]),
        property_count=len(ids), version=payload["version"],
    )


def prepare_property_queries(
    spans: Sequence[Mapping[str, Any]], typo_alias: Callable[[str], tuple[str, str] | None],
    *, query_text: Callable[[str], str],
) -> tuple[list[str], list[tuple[int, str]], dict[int, dict[str, str]]]:
    prepared = [query_text(str(span["text"])) for span in spans]
    valid: list[tuple[int, str]] = []
    probes: dict[int, dict[str, str]] = {}
    for index, value in enumerate(prepared):
        if len(value.split()) >= 2:
            valid.append((index, value))
        elif len(value.split()) == 1:
            alias = typo_alias(value)
            if alias is not None:
                property_id, canonical = alias
                prepared[index] = canonical
                valid.append((index, canonical))
                probes[index] = {"source": "one_edit_alias_e5_probe", "original": value,
                                 "canonical": canonical, "candidate_property_id": property_id}
    return prepared, valid, probes


def rank_property_scores(
    spans: Sequence[Mapping[str, Any]], prepared: Sequence[str], valid: Sequence[tuple[int, str]],
    scores: np.ndarray, ids: Sequence[str], definitions: Mapping[str, Mapping[str, Any]],
    *, category_id: str | None = None, taxonomy: Taxonomy | None = None,
    profiles: PropertyProfiles | None = None, top_k: int = 3, minimum_cosine: float = 0.80,
    typo_probes: Mapping[int, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if scores.shape != (len(valid), len(ids)) or not np.all(np.isfinite(scores)):
        raise ValueError("semantic property scores do not match the profile corpus")
    output: list[dict[str, Any]] = [
        {"span": dict(span), "query_text": prepared[index], "candidates": [], "status": "no_candidate"}
        for index, span in enumerate(spans)
    ]
    applicable = (
        set(profiles.applicable_ids(taxonomy, category_id))
        if category_id and taxonomy is not None and profiles is not None else None
    )
    for (index, _), row in zip(valid, scores):
        by_id: dict[str, float] = {}
        for property_id, score in zip(ids, row):
            by_id[property_id] = max(by_id.get(property_id, -math.inf), float(score))
        ranked = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))
        output[index]["unrestricted_top"] = [
            {"property_id": property_id, "cosine": score} for property_id, score in ranked[:3]
        ]
        probe = (typo_probes or {}).get(index)
        if probe is not None:
            output[index]["resolution"] = dict(probe)
            confirmed = (
                len(ranked) >= 2 and ranked[0][0] == probe["candidate_property_id"]
                and ranked[0][1] >= minimum_cosine
                and ranked[0][1] - ranked[1][1] >= 0.02
            )
            output[index]["resolution"]["corroborated"] = bool(confirmed)
            if not confirmed:
                continue
        candidates = []
        for property_id, score in ranked:
            if len(candidates) >= top_k or score < minimum_cosine:
                break
            if probe is not None and property_id != probe["candidate_property_id"]:
                continue
            if applicable is not None and property_id not in applicable:
                continue
            aliases = definitions[property_id].get("aliases") or {}
            label = (aliases.get("vi") or aliases.get("en") or [property_id])[0]
            candidates.append({
                "property_id": property_id, "label": label, "cosine": score,
                "applicable": None if applicable is None else True,
            })
        output[index]["candidates"] = candidates
        cue_mode, _ = semantic_property_cue(str(spans[index]["text"]))
        if cue_mode == "not_requested":
            output[index]["status"] = "not_requested"
        elif candidates:
            output[index]["status"] = "needs_confirmation"
    return output
