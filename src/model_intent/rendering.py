from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .normalization import fold_vietnamese, normalize_text
from .schemas import CategorySearchProfile, CategoryVariant
from .taxonomy import Taxonomy

LEGACY_RENDERER_VERSION = "focused-variants-v3"
PROFILE_RENDERER_VERSION = "category-profile-v3"
RENDERER_VERSION = PROFILE_RENDERER_VERSION
PROFILE_SCHEMA_VERSION = 3
PROFILE_LANGUAGES = ("vi", "en")


def load_enrichment(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("items") or {}
    if not isinstance(items, dict):
        raise ValueError("enrichment.items must be an object")
    return {str(key): value for key, value in items.items()}


def _validate_string_list(value: Any, field: str, item_id: str) -> None:
    if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
        raise ValueError(f"{field} for {item_id} must be a list of non-empty strings")
    if len(value) > 32:
        raise ValueError(f"{field} for {item_id} has too many entries")
    if any(len(x.strip()) > 500 for x in value):
        raise ValueError(f"{field} for {item_id} contains an entry longer than 500 characters")
    normalized = [normalize_text(x) for x in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} for {item_id} contains duplicate entries")


def validate_enrichment(
    taxonomy: Taxonomy,
    enrichment: dict[str, dict[str, Any]],
    *,
    require_rich_fields: bool = False,
) -> list[str]:
    """Validate the source fields used to build aggregate profile text.

    Examples remain tolerated in legacy source fixtures, but are not required,
    rendered, or copied into the production profile artifact.
    """
    unknown = sorted(set(enrichment) - set(taxonomy.items))
    if unknown:
        raise ValueError(f"unknown enrichment IDs: {unknown}")
    warnings: list[str] = []
    leaves = {item_id for item_id in taxonomy.items if not taxonomy.has_children(item_id)}
    if require_rich_fields:
        missing = sorted(
            leaves - set(enrichment),
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        )
        extra = sorted(
            set(enrichment) - leaves,
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        )
        if missing:
            raise ValueError(f"missing enrichment for leaf categories: {missing}")
        if extra:
            raise ValueError(f"rich enrichment may only contain leaf categories: {extra}")
    allowed_fields = {
        f"{field}_{language}"
        for field in ("description", "aliases", "purposes", "amenities", "common_intents", "examples")
        for language in PROFILE_LANGUAGES
    }
    for item_id, profile in enrichment.items():
        if not isinstance(profile, dict):
            raise ValueError(f"enrichment {item_id} must be an object")
        unknown_fields = sorted(set(profile) - allowed_fields)
        if unknown_fields and require_rich_fields:
            raise ValueError(f"unknown enrichment fields for {item_id}: {unknown_fields}")
        if taxonomy.has_children(item_id):
            warnings.append(f"enrichment {item_id} is a non-leaf category")
        for language in PROFILE_LANGUAGES:
            description = profile.get(f"description_{language}")
            if description is not None and (not isinstance(description, str) or not description.strip()):
                raise ValueError(f"description_{language} for {item_id} must be non-empty text")
            if description and len(description.strip()) > 2000:
                raise ValueError(f"description_{language} for {item_id} is too long")
            if not description:
                warnings.append(f"missing description_{language} for {item_id}")
            for field in ("aliases",):
                key = f"{field}_{language}"
                if key not in profile:
                    if require_rich_fields:
                        raise ValueError(f"missing {key} for {item_id}")
                    continue
                _validate_string_list(profile[key], key, item_id)
            # Other legacy semantic fields are tolerated but ignored by v3.
            for field in ("purposes", "amenities", "common_intents", "examples"):
                key = f"{field}_{language}"
                if key in profile:
                    _validate_string_list(profile[key], key, item_id)
    return warnings


def _append_variant(
    output: list[CategoryVariant],
    seen: set[tuple[str, str]],
    item_id: str,
    language: str,
    text: str,
    path_ids: tuple[str, ...],
    kind: str,
    source_text: str,
) -> None:
    text = str(text).strip()
    key = (language, normalize_text(text))
    if not key[1] or key in seen:
        return
    seen.add(key)
    output.append(CategoryVariant(item_id, language, text, path_ids, kind, source_text))


def render_variants(
    taxonomy: Taxonomy,
    languages: tuple[str, ...] = ("vi", "en"),
    enrichment: dict[str, dict[str, Any]] | None = None,
) -> list[CategoryVariant]:
    """Render the legacy focused corpus for rollback and compatibility tests."""
    enrichment = enrichment or {}
    validate_enrichment(taxonomy, enrichment)
    variants: list[CategoryVariant] = []
    for item in sorted(taxonomy.document.items, key=lambda x: x.id):
        profile = enrichment.get(item.id, {})
        path_ids = taxonomy.path_ids(item.id)
        seen: set[tuple[str, str]] = set()
        for language in languages:
            path = " > ".join(taxonomy.path_names(item.id, language))
            _append_variant(variants, seen, item.id, language, path, path_ids, "path", path)
            description = profile.get(f"description_{language}")
            if description:
                _append_variant(variants, seen, item.id, language, f"{path}. {description}", path_ids, "description", description)
            for alias in profile.get(f"aliases_{language}", []):
                _append_variant(variants, seen, item.id, language, f"{path}. {alias}", path_ids, "alias", alias)
            for example in profile.get(f"examples_{language}", []):
                _append_variant(variants, seen, item.id, language, str(example), path_ids, "example", example)
            if language == "vi":
                existing = list(variants)
                for variant in existing:
                    if variant.item_id != item.id or variant.language != "vi":
                        continue
                    folded = fold_vietnamese(variant.text)
                    if normalize_text(folded) != normalize_text(variant.text):
                        _append_variant(
                            variants,
                            seen,
                            item.id,
                            "vi",
                            folded,
                            path_ids,
                            f"{variant.profile_kind}_accent_folded",
                            variant.source_text or variant.text,
                        )
    return variants


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = normalize_text(text)
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def _path_text(taxonomy: Taxonomy, item_id: str, language: str) -> str:
    path_ids = taxonomy.path_ids(item_id)
    names: list[str] = []
    for path_id in path_ids:
        item = taxonomy.items[path_id]
        if language not in item.names or not item.names[language].strip():
            raise ValueError(f"category path {path_id} has no explicit {language} name")
        names.append(item.names[language].strip())
    if not names:
        raise ValueError(f"category {item_id} has no complete {language} hierarchy path")
    return " > ".join(names)


def _profile_values(
    item_id: str,
    language: str,
    taxonomy: Taxonomy,
    profile: dict[str, Any],
) -> dict[str, list[str] | str]:
    leaf_name = taxonomy.path_names(item_id, language)[-1]
    description = str(profile.get(f"description_{language}") or "").strip()
    if not description:
        description = (
            f"Địa điểm hoặc dịch vụ thuộc nhóm {leaf_name}."
            if language == "vi"
            else f"A place or service in the {leaf_name} category."
        )
    return {
        "description": description,
        "aliases": _dedupe(profile.get(f"aliases_{language}", []) or [leaf_name]),
    }


def _section(label: str, values: Iterable[str]) -> str | None:
    values = list(values)
    if not values:
        return None
    return f"{label}: " + "; ".join(values)


def render_category_profiles(
    taxonomy: Taxonomy,
    languages: tuple[str, ...] = PROFILE_LANGUAGES,
    enrichment: dict[str, dict[str, Any]] | None = None,
    *,
    require_rich_fields: bool = False,
) -> list[CategorySearchProfile]:
    """Render one aggregate hierarchy/description/alias text per leaf/language."""
    enrichment = enrichment or {}
    validate_enrichment(taxonomy, enrichment, require_rich_fields=require_rich_fields)
    unknown_languages = set(languages) - set(PROFILE_LANGUAGES)
    if unknown_languages:
        raise ValueError(f"unsupported profile languages: {sorted(unknown_languages)}")
    profiles: list[CategorySearchProfile] = []
    for item in sorted(
        taxonomy.document.items,
        key=lambda value: (not value.id.isdigit(), int(value.id) if value.id.isdigit() else value.id),
    ):
        if taxonomy.has_children(item.id):
            continue
        source = enrichment.get(item.id, {})
        vi_path = _path_text(taxonomy, item.id, "vi")
        en_path = _path_text(taxonomy, item.id, "en")
        for language in languages:
            values = _profile_values(item.id, language, taxonomy, source)
            if language == "vi":
                lines = [
                    f"Danh mục: {vi_path} ({en_path})",
                    f"Mô tả: {values['description']}",
                    _section("Tên gọi khác", values["aliases"]),
                ]
            else:
                lines = [
                    f"Category: {en_path} ({vi_path})",
                    f"Description: {values['description']}",
                    _section("Alternative names", values["aliases"]),
                ]
            text = "\n".join(line for line in lines if line)
            profiles.append(CategorySearchProfile(item_id=item.id, language=language, text=text))
    return profiles
