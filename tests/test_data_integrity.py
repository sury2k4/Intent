from pathlib import Path

from model_intent.decision import load_decision_policy, validate_decision_policy
from model_intent.poi_properties import load_property_profiles
from model_intent.rendering import load_enrichment, render_category_profiles, validate_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]
TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
ENRICHMENT = ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json"


def test_active_category_resources_and_property_catalog_are_consistent():
    taxonomy = Taxonomy.load(TAXONOMY)
    enrichment = load_enrichment(ENRICHMENT)
    policy = load_decision_policy(ROOT / "data/policies/place_type_decision_policy.poc.v1.json")
    properties = load_property_profiles(ROOT / "data/properties/poi_properties.poc.v8.json", taxonomy)
    assert validate_enrichment(taxonomy, enrichment, require_rich_fields=True) == []
    assert validate_decision_policy(policy, taxonomy) == []
    leaves = {item_id for item_id in taxonomy.items if not taxonomy.has_children(item_id)}
    assert len(taxonomy.items) == 400
    assert len(leaves) == 384
    assert set(enrichment) == leaves
    assert properties.definitions
    assert all(key in properties.definitions for ids in properties.categories.values() for key in ids)


def test_active_category_profiles_are_bounded_and_bilingual():
    taxonomy = Taxonomy.load(TAXONOMY)
    enrichment = load_enrichment(ENRICHMENT)
    profiles = render_category_profiles(taxonomy, ("vi", "en"), enrichment, require_rich_fields=True)
    leaves = {item_id for item_id in taxonomy.items if not taxonomy.has_children(item_id)}
    assert len(profiles) == 768
    assert {(profile.item_id, profile.language) for profile in profiles} == {
        (item_id, language) for item_id in leaves for language in ("vi", "en")
    }
    assert all(("Danh mục:" in profile.text or "Category:" in profile.text) for profile in profiles)
    assert all("Ví dụ tìm kiếm:" not in profile.text and "Search examples:" not in profile.text for profile in profiles)
    assert all(not key.startswith("examples_") for profile in enrichment.values() for key in profile)
