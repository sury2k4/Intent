from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import TaxonomyDocument, TaxonomyItem


class TaxonomyError(ValueError):
    pass


class Taxonomy:
    def __init__(self, document: TaxonomyDocument):
        self.document = document
        self.items = {item.id: item for item in document.items}
        self.by_code = {item.code: item for item in document.items if item.code}
        self.children: dict[str | None, list[str]] = defaultdict(list)
        for item in document.items:
            self.children[item.parent_id].append(item.id)
        for child_ids in self.children.values():
            child_ids.sort()

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Taxonomy":
        rows = payload.get("items")
        if not isinstance(rows, list):
            raise TaxonomyError("taxonomy.items must be a list")
        items: list[TaxonomyItem] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise TaxonomyError("every taxonomy item must be an object")
            item_id = str(row.get("id", "")).strip()
            code = str(row.get("code", "")).strip()
            names = row.get("names") or {}
            if not item_id or not code or not isinstance(names, dict) or not names:
                raise TaxonomyError(f"invalid taxonomy item: {row!r}")
            if item_id in seen:
                raise TaxonomyError(f"duplicate taxonomy id: {item_id}")
            seen.add(item_id)
            parent_raw = row.get("parent_id")
            parent_id = None if parent_raw in (None, "", 0, "0") else str(parent_raw).strip()
            clean_names = {str(k): str(v).strip() for k, v in names.items() if str(v).strip()}
            if not clean_names:
                raise TaxonomyError(f"taxonomy item {item_id} has no non-empty names")
            items.append(TaxonomyItem(item_id, parent_id, code, clean_names))
        document = TaxonomyDocument(
            schema_version=int(payload.get("schema_version", 1)),
            taxonomy_version=str(payload.get("taxonomy_version", "unknown")),
            status=str(payload.get("status", "unknown")),
            source=str(payload.get("source", "unknown")),
            languages=tuple(str(x) for x in payload.get("languages", ())),
            items=tuple(items),
        )
        taxonomy = cls(document)
        taxonomy.validate()
        return taxonomy

    def validate(self) -> list[str]:
        warnings: list[str] = []
        normalized_codes: dict[str, list[str]] = defaultdict(list)
        for item in self.document.items:
            normalized = re.sub(r"[-\s]+", "_", item.code.strip().lower())
            normalized_codes[normalized].append(item.id)
            if item.parent_id is not None and item.parent_id not in self.items:
                warnings.append(f"missing parent {item.parent_id} for {item.id}")
        for normalized, ids in normalized_codes.items():
            if len(ids) > 1:
                warnings.append(f"normalized code collision {normalized}: {', '.join(sorted(ids))}")
        for item in self.document.items:
            self.path_ids(item.id)  # detects cycles
        return warnings

    def has_children(self, item_id: str) -> bool:
        return bool(self.children.get(item_id))

    def ancestors(self, item_id: str) -> tuple[str, ...]:
        path = self.path_ids(item_id)
        return path[:-1]

    def descendants(self, item_id: str, leaves_only: bool = False) -> tuple[str, ...]:
        if item_id not in self.items:
            raise TaxonomyError(f"unknown taxonomy id: {item_id}")
        output: list[str] = []
        stack = list(reversed(self.children.get(item_id, [])))
        while stack:
            current = stack.pop()
            child_ids = self.children.get(current, [])
            if not leaves_only or not child_ids:
                output.append(current)
            stack.extend(reversed(child_ids))
        return tuple(output)

    def siblings(self, item_id: str, include_self: bool = False) -> tuple[str, ...]:
        if item_id not in self.items:
            raise TaxonomyError(f"unknown taxonomy id: {item_id}")
        parent_id = self.items[item_id].parent_id
        return tuple(x for x in self.children.get(parent_id, []) if include_self or x != item_id)

    def is_ancestor(self, ancestor_id: str, item_id: str) -> bool:
        return ancestor_id in self.ancestors(item_id)

    def nearest_common_ancestor(self, left_id: str, right_id: str) -> str | None:
        left_path = self.path_ids(left_id)
        right_path = self.path_ids(right_id)
        common = None
        for left, right in zip(left_path, right_path):
            if left != right:
                break
            common = left
        return common

    def relation(self, left_id: str, right_id: str) -> str:
        if left_id == right_id:
            return "same"
        if self.is_ancestor(left_id, right_id):
            return "ancestor"
        if self.is_ancestor(right_id, left_id):
            return "descendant"
        if right_id in self.siblings(left_id):
            return "sibling"
        if self.nearest_common_ancestor(left_id, right_id) is not None:
            return "same_branch"
        return "unrelated"

    def path_ids(self, item_id: str) -> tuple[str, ...]:
        if item_id not in self.items:
            raise TaxonomyError(f"unknown taxonomy id: {item_id}")
        path: list[str] = []
        current: str | None = item_id
        visited: set[str] = set()
        while current is not None and current in self.items:
            if current in visited:
                raise TaxonomyError(f"parent cycle involving {current}")
            visited.add(current)
            path.append(current)
            current = self.items[current].parent_id
        path.reverse()
        return tuple(path)

    def path_names(self, item_id: str, language: str) -> tuple[str, ...]:
        result: list[str] = []
        for current in self.path_ids(item_id):
            item = self.items[current]
            name = item.names.get(language) or item.names.get("en") or next(iter(item.names.values()))
            result.append(name)
        return tuple(result)

    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.document.schema_version,
            "taxonomy_version": self.document.taxonomy_version,
            "items": [
                {"id": i.id, "parent_id": i.parent_id, "code": i.code, "names": dict(sorted(i.names.items()))}
                for i in sorted(self.document.items, key=lambda x: x.id)
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def inspect(self, item_id: str) -> dict[str, Any]:
        item = self.items[item_id]
        return {
            "id": item.id,
            "parent_id": item.parent_id,
            "code": item.code,
            "names": dict(item.names),
            "path_ids": list(self.path_ids(item_id)),
            "path_names_vi": list(self.path_names(item_id, "vi")),
            "path_names_en": list(self.path_names(item_id, "en")),
        }
