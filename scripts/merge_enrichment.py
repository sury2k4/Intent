from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_intent.rendering import validate_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "data/taxonomies/place_types.poc.v1.json"
TARGET_PATH = ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json"
OVERRIDE_PATH = ROOT / "data/taxonomies/rich_description_overrides.poc.v1.json"
FRAGMENT_DIR = ROOT / "data/taxonomies/fragments"
RICH_FIELDS = ("aliases",)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _complete_profile(item: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for language in ("vi", "en"):
        name = str((item.get("names") or {}).get(language) or next(iter(item["names"].values()))).strip()
        description_key = f"description_{language}"
        aliases_key = f"aliases_{language}"
        base_description = str(profile.get(description_key) or "").strip()
        if not base_description:
            base_description = (
                f"Địa điểm hoặc dịch vụ thuộc nhóm {name}."
                if language == "vi"
                else f"A place or service in the {name} category."
            )
        rich_parts = [base_description]
        purposes = _dedupe(list(profile.get(f"purposes_{language}") or []))
        amenities = _dedupe(list(profile.get(f"amenities_{language}") or []))
        intents = _dedupe(list(profile.get(f"common_intents_{language}") or []))
        if purposes:
            rich_parts.append(("Mục tiêu sử dụng: " if language == "vi" else "Typical uses: ") + "; ".join(purposes))
        if amenities:
            rich_parts.append(("Tiện ích thường gặp: " if language == "vi" else "Typical facilities: ") + "; ".join(amenities))
        if intents and intents != purposes:
            rich_parts.append(("Nhu cầu tìm kiếm: " if language == "vi" else "Search needs: ") + "; ".join(intents))
        output[description_key] = ". ".join(part.rstrip(".") for part in rich_parts) + "."
        output[aliases_key] = _dedupe(list(profile.get(aliases_key) or [name]))
    return output


def build_payload(taxonomy: Taxonomy) -> tuple[dict[str, Any], list[str]]:
    merged: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for path in sorted(FRAGMENT_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items") or {}
        if not isinstance(items, dict):
            raise ValueError(f"{path.name}: items must be an object")
        for key, value in items.items():
            key = str(key)
            if key in merged:
                raise ValueError(f"duplicate enrichment category {key} across fragments")
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}: enrichment {key} must be an object")
            merged[key] = value
        sources.append(path.name)

    leaves = {
        item_id for item_id in taxonomy.items
        if not taxonomy.has_children(item_id)
    }
    # A category without a reviewed fragment still receives a deterministic
    # minimal profile so every leaf can participate in the same index. It is
    # visible in the generated file and can be replaced by curated content.
    for item_id in sorted(leaves - set(merged), key=lambda value: int(value) if value.isdigit() else value):
        merged[item_id] = {}

    overrides = {}
    if OVERRIDE_PATH.exists():
        override_payload = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
        overrides = {str(key): value for key, value in (override_payload.get("items") or {}).items()}

    completed: dict[str, dict[str, Any]] = {}
    for item_id, profile in merged.items():
        item = taxonomy.items[item_id]
        item_payload = {
            "id": item.id,
            "code": item.code,
            "names": dict(item.names),
        }
        combined = dict(profile)
        combined.update(overrides.get(item_id, {}))
        completed[item_id] = _complete_profile(item_payload, combined)
    ordered = dict(sorted(completed.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else pair[0]))
    payload = {
        "schema_version": 5,
        "source": "merged_reviewed_fragments_with_leaf_defaults",
        "sources": sources,
        "items": ordered,
    }
    return payload, sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the committed generated file without writing")
    parser.add_argument("--output", type=Path, default=TARGET_PATH, help="generated JSON path")
    args = parser.parse_args(argv)
    taxonomy = Taxonomy.load(TAXONOMY_PATH)
    payload, sources = build_payload(taxonomy)
    warnings = validate_enrichment(taxonomy, payload["items"], require_rich_fields=True)
    leaves = {item_id for item_id in taxonomy.items if not taxonomy.has_children(item_id)}
    enriched_leaves = leaves & set(payload["items"])
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    output_path = args.output.resolve()
    if args.check:
        current = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if current != serialized:
            raise SystemExit(f"generated enrichment differs from {output_path}; run scripts/merge_enrichment.py")
    else:
        output_path.write_text(serialized, encoding="utf-8")
    print(json.dumps({
        "taxonomy_items": len(taxonomy.items),
        "leaf_items": len(leaves),
        "enriched_items": len(payload["items"]),
        "enriched_leaves": len(enriched_leaves),
        "profiles": len(enriched_leaves) * 2,
        "leaf_coverage": len(enriched_leaves) / len(leaves) if leaves else 0,
        "warnings": warnings,
        "sources": sources,
        "check": args.check,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
