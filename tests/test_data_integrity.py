import json
from pathlib import Path

from model_intent.decision import load_decision_policy, validate_decision_policy
from model_intent.evaluation import load_cases, validate_cases
from model_intent.rendering import load_enrichment, render_category_profiles, validate_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def test_full_data_references_are_valid_and_leaf_coverage_is_high():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    policy = load_decision_policy(ROOT / "data/policies/place_type_decision_policy.poc.v1.json")
    cases = load_cases(ROOT / "data/eval/place_type_queries.poc.v1.jsonl")
    assert validate_enrichment(taxonomy, enrichment) == []
    assert validate_decision_policy(policy, taxonomy) == []
    assert validate_cases(cases, set(taxonomy.items)) == []
    leaves = {item_id for item_id in taxonomy.items if not taxonomy.has_children(item_id)}
    assert len(taxonomy.items) == 238
    assert len(leaves) == 222
    assert len(enrichment) == 222
    assert leaves - set(enrichment) == set()
    assert all(
        set(profile) == {"description_vi", "description_en", "aliases_vi", "aliases_en"}
        for profile in enrichment.values()
    )


def test_rendered_profile_corpus_is_bounded_and_bilingual():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    profiles = render_category_profiles(taxonomy, ("vi", "en"), enrichment)
    leaves = {item_id for item_id in taxonomy.items if not taxonomy.has_children(item_id)}
    assert len(profiles) == len(leaves) * 2
    assert {profile.item_id for profile in profiles} == leaves
    assert {(profile.item_id, profile.language) for profile in profiles} == {
        (item_id, language) for item_id in leaves for language in ("vi", "en")
    }
    assert all(("Danh mục:" in profile.text or "Category:" in profile.text) for profile in profiles)
    assert all("Ví dụ tìm kiếm:" not in profile.text for profile in profiles)
    assert all("Search examples:" not in profile.text for profile in profiles)


def test_production_enrichment_contains_no_examples():
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    assert all(
        not key.startswith("examples_")
        for profile in enrichment.values()
        for key in profile
    )
