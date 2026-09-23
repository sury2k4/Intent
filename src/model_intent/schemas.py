from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaxonomyItem:
    id: str
    parent_id: str | None
    code: str
    names: Mapping[str, str]


@dataclass(frozen=True)
class TaxonomyDocument:
    schema_version: int
    taxonomy_version: str
    status: str
    source: str
    languages: tuple[str, ...]
    items: tuple[TaxonomyItem, ...]


@dataclass(frozen=True)
class CategorySearchProfile:
    """One aggregate localized text document sent to the embedding model."""

    item_id: str
    language: str
    text: str


@dataclass(frozen=True)
class CategoryVariant:
    """Legacy focused document retained for the pre-profile rollback path."""

    item_id: str
    language: str
    text: str
    path_ids: tuple[str, ...]
    variant_kind: str = "path"
    source_text: str = ""

    @property
    def profile_kind(self) -> str:
        return self.variant_kind


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    track: str
    split: str
    query: str
    language: str | None = None
    gold_ids: tuple[str, ...] = ()
    acceptable_ids: tuple[str, ...] = ()
    difficulty: str = "normal"
    is_ood: bool = False
    notes: str = ""
    expected_action: str | None = None
    clarification_ids: tuple[str, ...] = ()
    near_group: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_id: str
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32
    max_seq_length: int = 128
    expected_dimension: int | None = None
    revision: str | None = None
    enabled: bool = True
    stretch: bool = False


@dataclass(frozen=True)
class RetrievalHit:
    item_id: str
    code: str
    score: float
    rank: int
    language: str
    matched_text: str
    path: tuple[str, ...] = ()
    profile_kind: str = "primary"
    query_variant_kind: str = "original"

    @property
    def document_variant_kind(self) -> str:
        """Compatibility name for legacy UI/report payloads."""
        return self.profile_kind


@dataclass(frozen=True)
class Prediction:
    query_id: str
    query: str
    hits: tuple[RetrievalHit, ...]
    top1_score: float | None
    margin: float | None
    model_key: str
    query_variants: tuple[str, ...] = ()
    variant_top_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClarificationOption:
    item_id: str
    code: str
    label: str
    hint: str
    score: float
    relation: str


@dataclass(frozen=True)
class IntentDecision:
    action: str
    reason: str
    language: str
    accepted_item_id: str | None = None
    accepted_code: str | None = None
    question: str | None = None
    options: tuple[ClarificationOption, ...] = ()
    top1_score: float | None = None
    margin: float | None = None
    calibrated_confidence: float | None = None
    in_domain_confidence: float | None = None
    ambiguity_score: float | None = None
    confidence_features: Mapping[str, float] = field(default_factory=dict)
    calibration_version: str = "none"
    policy_version: str = "unknown"


@dataclass
class BenchmarkResult:
    model_key: str
    model_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    resource: dict[str, Any] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
