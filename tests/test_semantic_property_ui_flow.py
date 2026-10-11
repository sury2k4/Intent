from pathlib import Path
from types import SimpleNamespace

import pytest

from model_intent import ui
from model_intent.search_runtime import E5_POI_MODEL_ID, E5_POI_MODEL_KEY, E5_POI_REVISION, SearchRuntime

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("scenario", ["suggestions", "unavailable", "not_requested", "no_candidates"])
def test_unresolved_properties_are_display_only(monkeypatch, scenario):
    app_test = pytest.importorskip("streamlit.testing.v1")
    query = "quán cà phê hoặc nhà thuốc có mát rượi"
    start = query.index("có mát rượi")
    span = {"text": "có mát rượi", "start_offset": start, "end_offset": len(query)}
    calls = []
    metadata = SimpleNamespace(
        model_key=E5_POI_MODEL_KEY, model_id=E5_POI_MODEL_ID,
        model_revision=E5_POI_REVISION, backend="pgvector", retrieval_mode="pg_exact", generation_id=6,
    )

    def analyze(text, **kwargs):
        calls.append(text)
        assert text == query
        candidates = [
            {"property_id": "air_conditioning", "label": "máy lạnh", "cosine": .82},
            {"property_id": "outdoor_seating", "label": "chỗ ngồi ngoài trời", "cosine": .81},
        ] if scenario == "suggestions" else []
        suggestion = [{"span": span, "status": "not_requested" if scenario == "not_requested" else "needs_confirmation",
                       "candidates": candidates}] if scenario != "unavailable" else []
        return {
            "query": text,
            "query_parts": {
                "category_text": "quán cà phê hoặc nhà thuốc",
                "filter_candidates": [], "preference_candidates": [],
                "unresolved_property_clauses": [{**span, "reason": "unrecognized_property_clause"}],
                "semantic_property_suggestions": suggestion,
                "semantic_property_metadata": {"status": "unavailable" if scenario == "unavailable" else "suggestions_only"},
            },
            "intent": {"decision": {"action": "clarify", "options": [
                {"item_id": "23", "label": "Quán cà phê"},
                {"item_id": "34", "label": "Nhà thuốc"},
            ]}, "prediction": {"hits": [{"item_id": "23"}, {"item_id": "34"}]}},
            "property_extraction": {"properties": []},
        }

    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(
            metadata=metadata, taxonomy=SimpleNamespace(path_names=lambda item_id, *_: [
                {"23": "Quán cà phê", "34": "Nhà thuốc"}[item_id],
            ]),
        ),
        property_profiles=SimpleNamespace(version="test"), property_catalog={}, analyze_intent=analyze,
    )
    monkeypatch.setattr(ui, "_analysis_config", lambda: (
        ROOT / "configs/benchmark.poi.pg.v3.yaml", f"semantic-flow-{scenario}", "taxonomy", "properties", "database",
    ))
    monkeypatch.setattr(SearchRuntime, "from_intent_config", lambda **kwargs: runtime)

    def app():
        from model_intent import ui

        ui.main()

    at = app_test.AppTest.from_function(app, default_timeout=30).run()
    at.text_area[0].set_value(query)
    at.button(key="FormSubmitter:intent_analysis_form-Phân tích query").click().run()
    assert not at.exception
    assert calls == [query]
    assert at.session_state["intent_result"]["query_parts"]["filter_candidates"] == []
    assert at.session_state["intent_result"]["intent"]["decision"]["action"] == "clarify"
    assert not any(button.key and button.key.startswith(("intent-clarify-", "intent-property-confirm-"))
                   for button in at.button)
    assert not at.selectbox
    if scenario == "suggestions":
        assert any("máy lạnh" in str(row.value) for row in at.markdown)
        assert any("chỗ ngồi ngoài trời" in str(row.value) for row in at.markdown)
    elif scenario == "unavailable":
        assert any("Chưa thể gợi ý" in str(row.value) for row in at.error)
    elif scenario == "not_requested":
        assert any("không yêu cầu" in str(row.value) for row in at.caption)
    else:
        assert any("Chưa có gợi ý phù hợp" in str(row.value) for row in at.caption)
