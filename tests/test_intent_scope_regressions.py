"""Regression checks for intent-only UI and query/property interpretation."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_intent import ui
from model_intent.poi_properties import load_property_definitions as load_property_catalog
from model_intent.property_extraction import PropertyExtractor
from model_intent.query_interpretation import interpret_query
from model_intent.rendering import load_enrichment
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def interpreter_parts():
    taxonomy = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v1.json")
    enrichment = load_enrichment(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    catalog = load_property_catalog(ROOT / "data/properties/poi_properties.poc.v5.json")
    return taxonomy, enrichment, PropertyExtractor(catalog)


@pytest.mark.parametrize("query", [
    "ở phường 1 quán cà phê",
    "phường Bến Thành quán cà phê có wifi",
    "quán cà phê ở phường Bến Thành",
    "quán cà phê phường 1",
])
def test_phuong_scope_does_not_consume_category(query, interpreter_parts):
    taxonomy, enrichment, extractor = interpreter_parts
    result = interpret_query(query, extractor.extract(query), taxonomy, enrichment)
    assert result.category_text == "quán cà phê"
    assert result.focused_is_usable
    assert any(match.item_id == "23" for match in result.category_matches)


@pytest.mark.parametrize("query, property_id", [
    ("quán cà phê ưu tiên không có wifi", "internet_access"),
    ("quán cà phê ưu tiên không có máy lạnh", "air_conditioning"),
    ("nhà nghỉ ưu tiên không có bãi đỗ ô tô", "parking"),
])
def test_preferred_absence_is_soft_negative_constraint(query, property_id, interpreter_parts):
    taxonomy, enrichment, extractor = interpreter_parts
    parsed = extractor.extract(query)
    row = next(item for item in parsed.properties if item.property_id == property_id)
    assert (row.mode, row.value, row.cue, row.applied) == ("preferred", False, "prefer_absence", True)
    assert query[row.start_offset:row.end_offset] == row.evidence_text
    interpreted = interpret_query(query, parsed, taxonomy, enrichment)
    assert interpreted.category_text in {"quán cà phê", "nhà nghỉ"}


class RecordingUI:
    def __init__(self):
        self.session_state = {}
        self.markdowns = []
        self.errors = []
        self.warnings = []
        self.tables = []
        self.metrics = []

    def markdown(self, value, **kwargs):
        self.markdowns.append((value, kwargs))

    def error(self, value):
        self.errors.append(value)

    def warning(self, value):
        self.warnings.append(value)

    def table(self, rows):
        self.tables.append(rows)

    def metric(self, label, value):
        self.metrics.append((label, value))

    def columns(self, count, **kwargs):
        return [self] * count

    def expander(self, *args, **kwargs):
        return nullcontext()

    def selectbox(self, _label, choices, **kwargs):
        return choices[kwargs.get("index", 0)]

    def button(self, *args, **kwargs):
        return False

    def __getattr__(self, name):
        if name in {"caption", "subheader", "success", "info", "write", "json"}:
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def _runtime(catalog=None):
    return SimpleNamespace(
        category_runtime=SimpleNamespace(taxonomy=SimpleNamespace(path_names=lambda *_: ["Quán cà phê"])),
        property_catalog=catalog or {},
    )


def test_intent_property_card_escapes_dynamic_html_and_class_name():
    st = RecordingUI()
    attack = '<img src=x onerror="alert(1)">'
    result = {
        "query_parts": {"filter_candidates": [], "preference_candidates": []},
        "property_extraction": {"properties": [{
            "property_id": attack, "mode": f"required{attack}", "operator": attack,
            "value": attack, "evidence_text": attack, "ignored_reason": attack,
            "applied": False,
        }]},
        "intent": {"decision": {"action": "reject"}},
    }
    runtime = _runtime({attack: {"aliases": {"vi": [attack]}}})
    ui._render_intent_result(st, result, runtime, "vi")
    card = next(value for value, options in st.markdowns if 'class="prop-item-card"' in value)
    assert attack not in card
    assert card.count("&lt;img") >= 3
    assert 'class="prop-pill"' in card
    assert all(options.get("unsafe_allow_html") is True for value, options in st.markdowns if '<div class="prop-item-card"' in value)


def test_semantic_suggestion_card_escapes_catalog_label_and_id():
    st = RecordingUI()
    attack = '<svg onload="alert(1)">'
    result = {
        "query": "quán cà phê có gì lạ",
        "query_parts": {"semantic_property_suggestions": [{
            "span": {"text": "có gì lạ", "start_offset": 12, "end_offset": 21},
            "query_text": "gì lạ", "status": "needs_confirmation",
            "candidates": [{"property_id": attack, "label": attack, "cosine": 0.83}],
        }]},
    }
    ui._render_semantic_property_suggestions(st, result)
    card = next(value for value, _ in st.markdowns if "prop-suggestion-card" in value)
    assert attack not in card
    assert card.count("&lt;svg") == 1


def test_semantic_backend_unavailable_is_visible_without_auto_filter():
    st = RecordingUI()
    result = {
        "query_parts": {
            "filter_candidates": [], "preference_candidates": [],
            "unresolved_property_clauses": [{"text": "có mát rượi"}],
            "semantic_property_suggestions": [],
            "semantic_property_metadata": {"status": "unavailable", "error_type": "RuntimeError"},
        },
        "property_extraction": {"properties": []},
        "intent": {"decision": {"action": "reject"}},
    }
    ui._render_intent_result(st, result, _runtime(), "vi")
    assert len(st.errors) == 1
    assert "Chưa thể gợi ý" in st.errors[0]
    assert "không tự thêm điều kiện lọc" in st.errors[0]
    assert any("chưa tạo điều kiện lọc" in warning for warning in st.warnings)
    assert result["query_parts"]["filter_candidates"] == []
    assert result["query_parts"]["semantic_property_suggestions"] == []


def test_intent_timing_breakdown_remains_visible():
    st = RecordingUI()
    ui._render_timing_breakdown(st, {
        "timing": {"version": 1, "path": "intent_clarification_selection", "total_ms": 8.3,
                   "stages": [{"name": "property_semantic_suggestions", "duration_ms": 2.5}]},
    })
    assert st.metrics == [("Tổng thời gian trên server", "8.30 ms")]
    assert any("Gợi ý thuộc tính E5" in row["Bước thực hiện"] for row in st.tables[-1])
