from collections import Counter

import pytest

from scripts.report_category_stress import summarize
from scripts.stress_category_intent import _safe_error, generate_cases


def test_synthetic_stress_corpus_covers_all_active_categories_and_traps():
    cases = generate_cases()
    assert 3000 <= len(cases) <= 5000
    assert len({case["expected_category_id"] for case in cases if case["expected_category_id"]}) == 384
    assert len({case["id"] for case in cases}) == len(cases)
    counts = Counter(case["kind"] for case in cases)
    assert counts["enrichment_alias_bare"] >= 200
    assert counts["property_required"] >= 300
    assert counts["property_preferred"] >= 300
    assert counts["ood_negated_only"] >= 50
    assert counts["ood_informational"] >= 50
    assert any(case["query"] == "nhà thương có bãi đỗ xe" for case in cases)
    assert any(case["query"] == "not a hospital but with wifi" for case in cases)
    assert all(case["gold_provenance"] != "independent_human_adjudication" for case in cases)


def test_database_connection_errors_redact_credentials(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@127.0.0.1:5433/model_intent_unified")
    message = _safe_error(RuntimeError("secret at postgresql://user:secret@127.0.0.1:5433/model_intent_unified"))
    assert "secret" not in message
    assert "DATABASE_URL redacted" in message


def test_report_rejects_missing_or_non_pgvector_rows():
    case = {"id": "test-1", "query": "nhà thương", "language": "vi", "kind": "curated_alias_bare",
            "expected_category_id": "42", "expected_action": "accept", "expected_properties": [],
            "expected_options": []}
    row = {**case, "action": "accept", "accepted_category_id": "42", "backend": "pgvector",
           "retrieval_mode": "pg_exact", "category_generation_id": 6,
           "top_hits": [{"id": "42", "score": 0.86}], "properties": [], "timing_total_ms": 1.0}
    assert summarize([case], [row], shards=1)["overall"]["correct_accept"] == 1
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        summarize([case], [], shards=1)
    with pytest.raises(ValueError, match="exact pgvector"):
        summarize([case], [{**row, "backend": "numpy"}], shards=1)
    with pytest.raises(ValueError, match="mixed category generations"):
        summarize([case], [{**row, "category_generation_id": None}], shards=1)
