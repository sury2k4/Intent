from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractedProperty:
    """A property constraint extracted from a user query."""

    property_id: str
    mode: str  # required, preferred, excluded
    operator: str = "eq"
    value: Any = True
    evidence_text: str = ""
    start_offset: int | None = None
    end_offset: int | None = None
    matched_alias: str = ""
    cue: str = ""
    confidence: float = 1.0
    applied: bool = True
    ignored_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "mode": self.mode,
            "operator": self.operator,
            "value": self.value,
            "evidence_text": self.evidence_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "matched_alias": self.matched_alias,
            "cue": self.cue,
            "confidence": self.confidence,
            "applied": self.applied,
            "ignored_reason": self.ignored_reason,
        }


@dataclass(frozen=True)
class PropertyExtraction:
    query: str
    schema_version: str
    properties: tuple[ExtractedProperty, ...] = ()
    warnings: tuple[str, ...] = ()
    failed_open: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "schema_version": self.schema_version,
            "properties": [item.as_dict() for item in self.properties],
            "warnings": list(self.warnings),
            "failed_open": self.failed_open,
        }
