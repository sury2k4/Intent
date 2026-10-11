import hashlib
import json
from types import SimpleNamespace

import pytest

from model_intent.category_confidence_eval import (
    evaluate_reviewed_cases,
    load_reviewed_cases,
    validate_review_manifest,
)
from model_intent.taxonomy import Taxonomy


def _row(case_id, query, split, *, is_ood=False, gold_ids=None):
    return {
        "id": case_id,
        "query": query,
        "language": "vi",
        "split": split,
        "gold_ids": [] if is_ood else gold_ids or ["23"],
        "acceptable_ids": [],
        "clarification_ids": [],
        "is_ood": is_ood,
        "expected_action": "reject" if is_ood else "accept",
        "provenance": {
            "label_origin": "independent_human_adjudication",
            "source_group": f"test-session-{case_id}",
            "family_id": f"test-family-{case_id}",
            "reviewer_count": 2,
            "adjudicated": True,
            "privacy_reviewed": True,
        },
    }


def _cases(tmp_path, rows):
    path = tmp_path / "review.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    taxonomy = Taxonomy.load("data/taxonomies/place_types.poc.v3.json")
    return path, taxonomy


def _split_rows():
    # Schema fixtures only; these are not independent human labels or a quality benchmark.
    return [
        _row("a", "tìm quán cà phê khu vực phía bắc", "train"),
        _row("b", "gần tôi có địa điểm nghỉ ngơi không", "validation", is_ood=True),
        _row("c", "quán cà phê thích hợp để gặp bạn", "test"),
    ]


def test_review_dataset_requires_independent_adjudication_and_three_grouped_splits(tmp_path):
    path, taxonomy = _cases(tmp_path, _split_rows())
    cases = load_reviewed_cases(path, taxonomy)
    assert len(cases) == 3
    assert {case.split for case in cases} == {"train", "validation", "test"}
    rows = _split_rows()
    rows[1]["provenance"]["label_origin"] = "taxonomy_alias_proxy"
    path, _ = _cases(tmp_path, rows)
    with pytest.raises(ValueError, match="independently human adjudicated"):
        load_reviewed_cases(path, taxonomy)


@pytest.mark.parametrize("change,error", [
    (lambda rows: rows[1]["provenance"].update(source_group=rows[0]["provenance"]["source_group"]), "source group crosses"),
    (lambda rows: rows[1]["provenance"].update(family_id=rows[0]["provenance"]["family_id"]), "family group crosses"),
    (lambda rows: rows[1].update(query=rows[0]["query"] + "!"), "duplicate reviewed query"),
    (lambda rows: rows[2].update(query="tìm quán cà phê khu vực phía bắc á"), "near-duplicate query crosses"),
    (lambda rows: rows[1]["provenance"].update(privacy_reviewed=False), "privacy review"),
    (lambda rows: rows[1].update(gold_ids=["23"]), "OOD requires no category labels"),
])
def test_review_data_rejects_leakage_and_unsupported_provenance(tmp_path, change, error):
    rows = _split_rows()
    change(rows)
    path, taxonomy = _cases(tmp_path, rows)
    with pytest.raises(ValueError, match=error):
        load_reviewed_cases(path, taxonomy)


def test_manifest_requires_frozen_cases_and_matching_serving_identity(tmp_path):
    path, _taxonomy = _cases(tmp_path, _split_rows())
    identity = {"model_revision": "test-revision", "category_generation_id": 6}
    manifest = {
        "dataset_version": "reviewed-v1",
        "annotation_guidelines_version": "guide-v1",
        "label_origin": "independent_human_adjudication",
        "cases_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **identity,
    }
    validate_review_manifest(manifest, identity, path)
    manifest["model_revision"] = "outdated"
    with pytest.raises(ValueError, match="incompatible with serving model_revision"):
        validate_review_manifest(manifest, identity, path)
    manifest["model_revision"] = "test-revision"
    manifest["cases_sha256"] = "bad"
    with pytest.raises(ValueError, match="checksum differs"):
        validate_review_manifest(manifest, identity, path)


def test_evaluator_separates_raw_top_probability_from_final_category_action(tmp_path):
    path, taxonomy = _cases(tmp_path, _split_rows())
    cases = load_reviewed_cases(path, taxonomy)
    called = []

    def analyze(query, *, language):
        called.append((query, language))
        return {
            "intent": {
                "decision": {"action": "accept", "accepted_item_id": "23", "calibrated_confidence": 0.8,
                             "in_domain_confidence": 0.9},
                "confidence_assessment": {"semantic_top_item_id": "34", "final_decision_source": "literal_leaf_name"},
            }
        }

    report = evaluate_reviewed_cases(SimpleNamespace(analyze_intent=analyze), cases, split="test")
    assert len(called) == 1
    assert report["summary"]["n"] == 1
    assert report["summary"]["accepted_correct_n"] == 1
    assert report["summary"]["top1_reliability"]["brier"] == pytest.approx(0.8 ** 2)
    assert report["summary"]["conclusive"] is False
    assert report["by_decision_source"]["literal_leaf_name"]["accepted_n"] == 1
    assert "insufficient independent coverage" in report["interpretation"]
