import pytest

from model_intent.rendering import render_category_profiles, render_variants
from model_intent.taxonomy import Taxonomy


def taxonomy():
    return Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "food", "names": {"vi": "Ăn uống", "en": "Food"}},
        {"id": "2", "parent_id": "1", "code": "coffee", "names": {"vi": "Cà phê", "en": "Coffee"}},
    ]})


def test_localized_paths_and_folded_variants_are_deterministic():
    variants = render_variants(taxonomy(), ("vi", "en"))
    paths = [(v.item_id, v.language, v.text, v.variant_kind) for v in variants]
    assert ("1", "vi", "Ăn uống", "path") in paths
    assert ("1", "vi", "An uong", "path_accent_folded") in paths
    assert ("1", "en", "Food", "path") in paths
    assert ("2", "vi", "Ăn uống > Cà phê", "path") in paths
    assert ("2", "en", "Food > Coffee", "path") in paths


def test_enrichment_creates_focused_semantic_variants():
    variants = render_variants(taxonomy(), ("vi",), {"2": {
        "description_vi": "Nơi uống cà phê",
        "aliases_vi": ["quán cafe"],
        "examples_vi": ["cần một nơi yên tĩnh để mở laptop"],
    }})
    coffee = [v for v in variants if v.item_id == "2"]
    assert any(v.variant_kind == "description" and "Nơi uống cà phê" in v.text for v in coffee)
    assert any(v.variant_kind == "alias" and "quán cafe" in v.text for v in coffee)
    assert any(v.variant_kind == "example" and "mở laptop" in v.text for v in coffee)
    assert all(v.path_ids == ("1", "2") for v in coffee)


def test_missing_language_falls_back_to_available_name():
    tax = Taxonomy.from_payload({"items": [{"id": "1", "parent_id": "0", "code": "x", "names": {"vi": "X"}}]})
    assert render_variants(tax, ("en",))[0].text == "X"


def test_category_profiles_are_leaf_only_and_include_bilingual_hierarchy():
    profiles = render_category_profiles(taxonomy(), ("vi", "en"), {"2": {
        "description_vi": "Nơi uống cà phê",
        "description_en": "A place for coffee",
        "aliases_vi": ["quán cafe"],
        "aliases_en": ["coffee shop"],
        "purposes_vi": ["ngồi làm việc"],
        "purposes_en": ["work remotely"],
        "amenities_vi": ["wifi"],
        "amenities_en": ["wifi"],
        "common_intents_vi": ["tìm quán cà phê"],
        "common_intents_en": ["find a coffee shop"],
    }})
    assert len(profiles) == 2
    assert {profile.item_id for profile in profiles} == {"2"}
    vi = next(profile for profile in profiles if profile.language == "vi")
    en = next(profile for profile in profiles if profile.language == "en")
    assert "Danh mục: Ăn uống > Cà phê (Food > Coffee)" in vi.text
    assert "Mục đích sử dụng:" not in vi.text
    assert "Tiện ích thường gặp:" not in vi.text
    assert "Ý định phổ biến:" not in vi.text
    assert "Category: Food > Coffee (Ăn uống > Cà phê)" in en.text
    assert "Common uses:" not in en.text
    assert "Typical amenities:" not in en.text
    assert "Common intents:" not in en.text


def test_strict_profile_validation_rejects_missing_leaf_fields():
    tax = taxonomy()
    with pytest.raises(ValueError, match="missing aliases"):
        render_category_profiles(
            tax,
            ("vi", "en"),
            {"2": {
                "description_vi": "mô tả",
                "description_en": "description",
            }},
            require_rich_fields=True,
        )


    profiles = render_category_profiles(taxonomy(), ("vi",), {})
    assert len(profiles) == 1
    assert "Mô tả:" in profiles[0].text
    assert "Tên gọi khác:" in profiles[0].text
    assert "Mục đích sử dụng:" not in profiles[0].text
    assert "Tiện ích thường gặp:" not in profiles[0].text
    assert "Ý định phổ biến:" not in profiles[0].text
    assert "_accent_folded" not in profiles[0].text
    assert "Ví dụ tìm kiếm:" not in profiles[0].text
