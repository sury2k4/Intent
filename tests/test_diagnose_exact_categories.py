from __future__ import annotations

from types import SimpleNamespace

from model_intent.runtime import RuntimeMetadata
from model_intent.search_runtime import SearchRuntime
from model_intent.taxonomy import Taxonomy
from scripts.diagnose_exact_categories import Case, build_cases, diagnose, main


def _taxonomy():
    return Taxonomy.from_payload({
        "items": [
            {"id": "1", "code": "cafe", "parent_id": None, "names": {"vi": "Quán cà phê", "en": "Cafe"}},
            {"id": "2", "code": "pharmacy", "parent_id": None, "names": {"vi": "Nhà thuốc", "en": "Pharmacy"}},
            {"id": "3", "code": "parent", "parent_id": None, "names": {"vi": "Cửa hàng", "en": "Shop"}},
            {"id": "4", "code": "child", "parent_id": "3", "names": {"vi": "Tiệm", "en": "Store"}},
        ],
    })


def test_canonical_names_and_prefixes_cover_both_languages_but_only_leaves():
    cases, collisions = build_cases(_taxonomy(), {}, include_negatives=False)
    assert not collisions
    assert {(case.query, case.language, case.expected_item_id) for case in cases} >= {
        ("Quán cà phê", "vi", "1"), ("tìm Quán cà phê", "vi", "1"),
        ("Cafe", "en", "1"), ("find Cafe", "en", "1"),
    }
    assert len(cases) == 12  # three leaves x two languages x bare/prefixed
    assert all(case.expected_item_id != "3" for case in cases)


def test_aliases_are_opt_in_and_colliding_phrases_are_not_scored():
    enrichment = {
        "1": {"aliases_vi": ["Cà phê", "Hiệu thuốc"], "aliases_en": ["coffee", "Pharmacy"]},
        "2": {"aliases_vi": ["Hieu thuoc"], "aliases_en": []},
    }
    cases, collisions = build_cases(_taxonomy(), enrichment, include_aliases=True, include_negatives=False)
    assert {("Cà phê", "1"), ("coffee", "1")} <= {
        (case.query, case.expected_item_id) for case in cases
    }
    assert "Hiệu thuốc" not in {case.query for case in cases}
    assert "Pharmacy" not in {case.query for case in cases}
    assert {("1", "Hiệu thuốc"), ("2", "Hieu thuoc"), ("1", "Pharmacy")} <= {
        (row["item_id"], row["phrase"]) for row in collisions
    }
    assert all(row["colliding_item_ids"] in (["1", "2"],) for row in collisions)
    default, _ = build_cases(_taxonomy(), enrichment, include_negatives=False)
    assert "coffee" not in {case.query for case in default}


def _metadata():
    return RuntimeMetadata(
        model_key="e5-base-multilingual", model_id="intfloat/multilingual-e5-base",
        candidate_count=6, taxonomy_items=3, taxonomy_fingerprint="test",
        cache_hit=True, model_load_seconds=0.0, candidate_load_seconds=0.0,
        candidate_encode_seconds=None, cache_path="postgresql://category_search_profiles_v3",
        backend="pgvector", retrieval_mode="pg_exact", generation_id=42,
    )


def test_summary_separates_retrieval_miss_from_calibration_reject():
    cases = [
        Case("a", "vi", "name_bare", "1"),
        Case("b", "vi", "name_bare", "1"),
        Case("c", "en", "negative", None),
    ]

    def query(text, *, top_k, language):
        assert top_k == 2
        if text == "a":
            hits = [{"item_id": "2", "score": 0.80, "rank": 1}, {"item_id": "1", "score": 0.78, "rank": 2}]
        else:
            hits = [{"item_id": "1", "score": 0.82, "rank": 1}]
        return {
            "prediction": {"hits": hits, "top1_score": hits[0]["score"], "margin": 0.02},
            "decision": {"action": "reject", "reason": "low_calibrated_confidence", "accepted_item_id": None,
                         "calibrated_confidence": 0.4},
            "runtime": {"generation_id": 42, "retrieval_mode": "pg_exact"},
        }

    runtime = SimpleNamespace(
        query=query, metadata=_metadata(), taxonomy=_taxonomy(), enrichment={},
        confidence_calibration={"calibration_version": "test"},
    )
    result = diagnose(cases, runtime, top_k=2)
    assert result["summary"]["retrieval_miss"] == 1
    assert result["summary"]["calibration_reject"] == 1
    assert result["summary"]["negative_false_accept"] == 0
    assert result["cases"][0]["expected_rank"] == 2
    assert result["cases"][0]["expected_cosine"] == 0.78
    assert result["cases"][1]["semantic_decision"]["calibrated_confidence"] == 0.4
    assert result["runtime_metadata"]["generation_id"] == 42


def test_search_runtime_reports_semantic_reject_and_explicit_rescue():
    runtime = object.__new__(SearchRuntime)
    runtime.category_runtime = SimpleNamespace(
        metadata=_metadata(), confidence_calibration={"calibration_version": "test"},
        query=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("semantic result is already present")),
    )
    runtime.analyze_intent = lambda query, **kwargs: {"intent": {
        "prediction": {"hits": [{"item_id": "1", "score": 0.75, "rank": 1}], "top1_score": 0.75},
        "decision": {"action": "accept", "reason": "explicit_leaf_name_evidence", "accepted_item_id": "1",
                     "calibrated_confidence": 0.3},
        "category_resolution": {
            "source": "literal_leaf_name", "explicit_matches": [{"item_id": "1", "evidence": "Cafe"}],
            "semantic_decision": {"action": "reject", "reason": "low_calibrated_confidence"},
        },
        "runtime": {"generation_id": 42},
    }}
    result = diagnose([Case("Cafe", "en", "name_bare", "1")], runtime, top_k=5)
    assert result["summary"]["calibration_reject_rescued"] == 1
    assert result["summary"]["correct_final_accept"] == 1
    assert result["cases"][0]["semantic_decision"]["reason"] == "low_calibrated_confidence"
    assert result["cases"][0]["category_resolution"]["source"] == "literal_leaf_name"
    assert "poi_generation_id" not in result["runtime_metadata"]


def test_intent_runtime_diagnostic_needs_no_poi_generation():
    runtime = object.__new__(SearchRuntime)
    runtime.category_runtime = SimpleNamespace(
        metadata=_metadata(), confidence_calibration={"calibration_version": "test"},
        query=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("semantic result already captured")),
    )
    runtime.analyze_intent = lambda query, **kwargs: {"intent": {
        "prediction": {"hits": [{"item_id": "1", "score": 0.86, "rank": 1}], "top1_score": 0.86},
        "decision": {"action": "accept", "reason": "calibrated_high_confidence", "accepted_item_id": "1"},
        "category_resolution": {"source": "alias_canonical_probe", "semantic_query": "Cafe", "original_semantic_decision": {
            "action": "reject", "reason": "low_calibrated_confidence",
        }},
        "runtime": {"generation_id": 42},
    }}
    result = diagnose([Case("coffee", "en", "alias_bare", "1")], runtime, top_k=5)
    assert result["summary"]["calibration_reject_rescued"] == 1
    assert result["summary"]["correct_final_accept"] == 1
    assert "poi_generation_id" not in result["runtime_metadata"]


def test_fake_model_is_rejected_before_cache_creation(capsys):
    assert main(["--model", "fake", "--runtime", "matcher", "--item-id", "1"]) == 2
    assert "write a NumPy cache" in capsys.readouterr().err


def test_missing_database_url_fails_before_constructing_runtime(monkeypatch, capsys):
    from scripts import diagnose_exact_categories as diagnostic

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(diagnostic.SearchRuntime, "from_intent_config", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("runtime must not be constructed")
    ))
    assert main(["--item-id", "1"]) == 2
    assert "DATABASE_URL" in capsys.readouterr().err
