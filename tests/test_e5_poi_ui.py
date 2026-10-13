from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import sys

import pytest
import yaml

from model_intent import ui
from model_intent.search_runtime import (
    E5_POI_DIMENSION,
    E5_POI_MODEL_ID,
    E5_POI_MODEL_KEY,
    E5_POI_REVISION,
    SearchRuntime,
)

ROOT = Path(__file__).parents[1]


VALID_DATABASE_URL = "postgresql://test_user:secret-one@127.0.0.1:5433/model_intent_unified"


def test_ui_defaults_to_unified_category_db_and_pinned_e5(monkeypatch):
    monkeypatch.delenv("MODEL_INTENT_CONFIG", raising=False)
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    config_path, config_hash, taxonomy_hash, resources_hash, database_hash = ui._analysis_config()
    assert config_path == ROOT / "configs/benchmark.poi.pg.v3.yaml"
    assert ui.DEFAULT_TAXONOMY == ROOT / "data/taxonomies/place_types.poc.v3.json"
    assert all(len(value) == 64 for value in (config_hash, taxonomy_hash, resources_hash, database_hash))
    assert "secret-one" not in str((config_path, config_hash, taxonomy_hash, resources_hash, database_hash))


def test_ui_analysis_does_not_require_poi_serving(monkeypatch):
    from model_intent import serving_config

    config = yaml.safe_load((ROOT / "configs/benchmark.poi.pg.v3.yaml").read_text(encoding="utf-8"))
    config["legacy_poi"] = {"backend": "artifact", "mapping_path": "missing-mapping.json"}
    monkeypatch.delenv("MODEL_INTENT_CONFIG", raising=False)
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    monkeypatch.setattr(serving_config, "load_yaml", lambda path: config)
    assert len(ui._analysis_config()) == 5


def test_ui_redacts_database_credentials_from_runtime_errors(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    message = ui._safe_error_message(RuntimeError(f"Failed for {VALID_DATABASE_URL}: secret-one"))
    assert "secret-one" not in message
    assert VALID_DATABASE_URL not in message
    assert "DATABASE_URL redacted" in message


@pytest.mark.parametrize("database_url", [None, "postgresql://user:wrong@localhost:5432/model_intent",
                                             "postgresql://user:wrong@localhost:5433/other_db"])
def test_ui_rejects_missing_or_wrong_database_url(monkeypatch, database_url):
    monkeypatch.delenv("MODEL_INTENT_CONFIG", raising=False)
    if database_url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", database_url)
    with pytest.raises(RuntimeError, match="DATABASE_URL") as error:
        ui._analysis_config()
    assert "wrong" not in str(error.value)


@pytest.mark.parametrize("field,value", [
    ("retrieval_backend", "numpy"), ("retrieval_mode", "pg_hnsw"),
    ("require_explicit_database_url", False),
    ("models.0.model_id", "BAAI/bge-m3"), ("models.0.expected_dimension", 1024),
])
def test_ui_refuses_unpinned_category_config(monkeypatch, tmp_path, field, value):
    config = yaml.safe_load((ROOT / "configs/benchmark.poi.pg.v3.yaml").read_text(encoding="utf-8"))
    target = config
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdecimal() else target[part]
    target[parts[-1]] = value
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("MODEL_INTENT_CONFIG", str(path))
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    with pytest.raises(RuntimeError, match="pgvector|pinned E5-base"):
        ui._analysis_config()


class _FakeStreamlit:
    def __init__(self, mode):
        self.mode = mode
        self.sidebar = nullcontext()
        self.session_state = {}
        self.resources = {}
        self.cache_keys = []
        self.errors = []

    def cache_resource(self, **kwargs):
        def decorate(func):
            def cached(*args):
                key = (func.__name__, args)
                self.cache_keys.append(key)
                if key not in self.resources:
                    self.resources[key] = func(*args)
                return self.resources[key]
            return cached
        return decorate

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def form(self, *args, **kwargs):
        return nullcontext()

    def set_page_config(self, **kwargs):
        pass

    def title(self, *args, **kwargs):
        pass

    def markdown(self, *args, **kwargs):
        pass

    def caption(self, *args, **kwargs):
        pass

    def subheader(self, *args, **kwargs):
        pass

    def header(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass

    def metric(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def error(self, message):
        self.errors.append(message)

    def stop(self):
        raise RuntimeError("streamlit stopped")

    def radio(self, label, options, **kwargs):
        return self.mode if label == "Demo" else options[0]

    def slider(self, label, *args, **kwargs):
        return kwargs.get("value", args[-1] if args else None)

    def toggle(self, *args, **kwargs):
        return kwargs.get("value", False)

    def checkbox(self, *args, **kwargs):
        return kwargs.get("value", False)

    def button(self, *args, **kwargs):
        return False

    def form_submit_button(self, *args, **kwargs):
        return False

    def text_area(self, *args, **kwargs):
        return kwargs.get("value", "")

    def text_input(self, *args, **kwargs):
        return kwargs.get("value", "")

    def columns(self, count, **kwargs):
        return [self] * count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def expander(self, *args, **kwargs):
        return nullcontext()

    def selectbox(self, label, options, **kwargs):
        assert label != "Embedding model"  # BGE must not be selectable for serving.
        return options[0]


def test_ui_caches_analysis_by_database_config_and_properties(monkeypatch, tmp_path):
    config = yaml.safe_load((ROOT / "configs/benchmark.poi.pg.v3.yaml").read_text(encoding="utf-8"))
    for key in ("taxonomy_path", "enrichment_path", "policy_path", "confidence_calibration_path", "category_query_aliases_path"):
        config[key] = str(ROOT / config[key])
    alias_path = tmp_path / "category_aliases.json"
    alias_path.write_bytes(Path(config["category_query_aliases_path"]).read_bytes())
    config["category_query_aliases_path"] = str(alias_path)
    property_path = tmp_path / "properties.json"
    property_path.write_bytes((ROOT / config["property_intent"]["property_path"]).read_bytes())
    config["property_intent"]["property_path"] = str(property_path)
    semantic_path = tmp_path / "semantic_properties.json"
    semantic_path.write_bytes((ROOT / config["property_intent"]["property_semantic_profiles_path"]).read_bytes())
    config["property_intent"]["property_semantic_profiles_path"] = str(semantic_path)
    config_path = tmp_path / "configs/benchmark.poi.pg.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("MODEL_INTENT_CONFIG", str(config_path))
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    st = _FakeStreamlit("analysis")
    monkeypatch.setitem(sys.modules, "streamlit", st)
    calls = []
    metadata = SimpleNamespace(
        model_key=E5_POI_MODEL_KEY, model_id=E5_POI_MODEL_ID, model_revision=E5_POI_REVISION,
        backend="pgvector", retrieval_mode="pg_exact", generation_id=6,
    )
    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(
            metadata=metadata, confidence_calibration={"calibration_version": "v3"}, taxonomy=SimpleNamespace(),
        ),
        property_profiles=SimpleNamespace(version="v5"),
    )

    def load_analysis(**kwargs):
        calls.append(kwargs)
        return runtime

    monkeypatch.setattr(SearchRuntime, "from_intent_config", load_analysis)
    ui.main()
    st.session_state["intent_result"] = None
    ui.main()
    assert len(calls) == 1
    assert "intent_result" in st.session_state

    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL.replace("secret-one", "secret-two"))
    ui.main()
    assert len(calls) == 2
    assert "intent_result" not in st.session_state
    st.session_state["intent_result"] = None

    property_path.write_text(property_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ui.main()
    assert len(calls) == 3
    assert "intent_result" not in st.session_state
    st.session_state["intent_result"] = None

    alias_path.write_text(alias_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ui.main()
    assert len(calls) == 4
    assert "intent_result" not in st.session_state
    st.session_state["intent_result"] = None

    semantic_path.write_text(semantic_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ui.main()
    assert len(calls) == 5  # pg_exact_v1 pins this corpus into the runtime fingerprint
    assert "intent_result" not in st.session_state
    st.session_state["intent_result"] = None

    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    ui.main()
    assert len(calls) == 6
    assert "intent_result" not in st.session_state
    assert all(call["config_path"] == str(config_path) for call in calls)
    assert all(call["taxonomy_path"] == ui._configured_taxonomy(config_path) for call in calls)
    assert all(call["model_key"] == E5_POI_MODEL_KEY for call in calls)
    assert "secret-one" not in str(st.cache_keys)
    assert "secret-two" not in str(st.cache_keys)
    assert not st.errors


def test_ui_fails_before_loading_runtime_if_database_url_is_missing(monkeypatch):
    monkeypatch.delenv("MODEL_INTENT_CONFIG", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    st = _FakeStreamlit("analysis")
    monkeypatch.setitem(sys.modules, "streamlit", st)
    ui.main()
    assert any("DATABASE_URL" in error for error in st.errors)
    assert not st.cache_keys


def test_ui_submits_updated_query_on_consecutive_analyses(monkeypatch):
    app_test = pytest.importorskip("streamlit.testing.v1")
    submitted_queries = []
    metadata = SimpleNamespace(
        model_key=E5_POI_MODEL_KEY, model_id=E5_POI_MODEL_ID, model_revision=E5_POI_REVISION,
        backend="pgvector", retrieval_mode="pg_exact", generation_id=6,
    )
    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(
            metadata=metadata, confidence_calibration={"calibration_version": "test"},
            taxonomy=SimpleNamespace(path_names=lambda item_id, language: [
                {"23": "Quán cà phê", "34": "Nhà thuốc"}[item_id],
            ]),
        ),
        property_catalog={"air_conditioning": {"aliases": {"vi": ["máy lạnh"]}}},
        property_profiles=SimpleNamespace(version="test"),
    )

    def analyze(query, **kwargs):
        submitted_queries.append(query)
        decision = (
            {"action": "clarify", "reason": "near_tied_category_candidates", "question": "Chọn category",
             "calibrated_confidence": 0.7317, "options": [
                {"item_id": "23", "label": "Quán cà phê", "code": "coffee", "hint": "Ăn uống > Quán cà phê", "score": 0.873},
                {"item_id": "34", "label": "Nhà thuốc", "code": "pharmacy", "hint": "Y tế > Nhà thuốc", "score": 0.869},
            ]}
            if "hoặc" in query else {"action": "reject", "reason": "test"}
        )
        return {
            "query": query,
            "query_parts": {
                "category_text": query.replace(" có máy lạnh", ""),
                "pgvector_queries": [query.replace(" có máy lạnh", "")],
                "vector_query_mode": "focused",
                "filter_candidates": [{"property_id": "air_conditioning", "mode": "required", "value": True}],
            },
            "intent": {"decision": decision, "prediction": {"hits": [
                {"item_id": "23", "code": "coffee", "score": 0.873},
                {"item_id": "34", "code": "pharmacy", "score": 0.869},
            ]}, "category_resolution": {
                "action_thresholds": {"accept_min_top_confidence": 0.805},
                "near_tie": {"observed_gap": 0.0021},
            }},
            "property_extraction": {"properties": [
                {"property_id": "air_conditioning", "mode": "required", "operator": "eq",
                 "value": True, "evidence_text": "máy lạnh", "applied": True},
            ]},
        }

    runtime.analyze_intent = analyze
    monkeypatch.setattr(ui, "_analysis_config", lambda: (
        ROOT / "configs/benchmark.poi.pg.v3.yaml", "config", "taxonomy", "properties", "database",
    ))
    monkeypatch.setattr(SearchRuntime, "from_intent_config", lambda **kwargs: runtime)

    def app():
        from model_intent import ui

        ui.main()

    at = app_test.AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    for query in ("nhà thuốc có máy lạnh", "quán cà phê", "quán cà phê hoặc nhà thuốc có máy lạnh"):
        at.text_area[0].set_value(query)
        at.button(key="FormSubmitter:intent_analysis_form-Phân tích query").click().run()
        assert not at.exception
        assert submitted_queries[-1] == query
        assert at.session_state["intent_query"] == query
        assert "poi_search" not in at.session_state["intent_result"]
        assert at.session_state["intent_result"]["query_parts"]["pgvector_queries"] == [
            query.replace(" có máy lạnh", ""),
        ]
        assert any("Properties · Yêu cầu thêm" in str(row.value) for row in at.subheader)
        assert any("máy lạnh" in str(row.value) for row in at.markdown)
    assert at.session_state["intent_result"]["intent"]["decision"]["action"] == "clarify"
    assert any("Quán cà phê" in str(row.value) and "1." in str(row.value) for row in at.markdown)
    assert any("Nhà thuốc" in str(row.value) and "2." in str(row.value) for row in at.markdown)
    assert not any(button.key and button.key.startswith("intent-clarify-") for button in at.button)
    assert "poi_search" not in at.session_state["intent_result"]


def test_ui_shows_all_property_suggestions_without_confirmation():
    class RecordingUI:
        def __init__(self):
            self.messages = []
            self.cards = []

        def markdown(self, value, **kwargs):
            self.cards.append(value)

        def caption(self, value):
            self.messages.append(value)

        def write(self, value):
            self.messages.append(value)

        def selectbox(self, *args, **kwargs):
            pytest.fail("Property confirmation controls must not be rendered")

        def button(self, *args, **kwargs):
            pytest.fail("Property confirmation controls must not be rendered")

    st = RecordingUI()
    result = {
        "query": "quán cà phê có mát rượi",
        "query_parts": {"semantic_property_suggestions": [{
            "span": {"text": "có mát rượi", "start_offset": 12, "end_offset": 23},
            "status": "needs_confirmation",
            "unrestricted_top": [{"property_id": "shade", "cosine": .838}],
            "candidates": [
                {"property_id": "air_conditioning", "label": "máy lạnh", "cosine": .823},
                {"property_id": "outdoor_seating", "label": "chỗ ngồi ngoài trời", "cosine": .811},
            ],
        }]},
    }
    ui._render_semantic_property_suggestions(st, result)
    assert "máy lạnh" in st.cards[1]
    assert "chỗ ngồi ngoài trời" in st.cards[2]
    assert not any("shade" in card for card in st.cards)
    assert any("chưa áp dụng" in line for line in st.messages)
    assert result["query_parts"]["semantic_property_suggestions"][0]["status"] == "needs_confirmation"


def test_ui_keeps_accepted_category_distinct_from_semantic_top_hit():
    from model_intent import ui

    class RecordingUI:
        def __init__(self):
            self.messages = []

        def __getattr__(self, name):
            if name in {"subheader", "success", "warning", "caption", "info", "write"}:
                return lambda text, **kwargs: self.messages.append((name, text))
            if name == "columns":
                return lambda *args, **kwargs: (self, self)
            if name == "expander":
                return lambda *args, **kwargs: nullcontext()
            if name == "json":
                return lambda payload: self.messages.append(("json", payload))
            raise AttributeError(name)

    st = RecordingUI()
    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(taxonomy=SimpleNamespace(path_names=lambda item_id, *_: [
            {"23": "Quán cà phê", "34": "Nhà thuốc"}[item_id],
        ])),
        property_catalog={},
    )
    result = {
        "intent": {
            "decision": {"action": "accept", "accepted_item_id": "23", "calibrated_confidence": 0.4},
            "prediction": {"hits": [{"item_id": "34", "score": 0.8}]},
            "confidence_assessment": {"semantic_top_item_id": "34", "semantic_top_cosine": 0.8},
        },
        "property_extraction": {"properties": []},
    }
    ui._render_intent_result(st, result, runtime, "vi")
    summary = " ".join(str(text) for kind, text in st.messages if kind != "json")
    assert "Quán cà phê" in summary
    assert "Nhà thuốc" not in summary
    assert "0.4000" not in summary and "0.8000" not in summary
    assert next(text for kind, text in st.messages if kind == "json")["prediction"] == result["intent"]["prediction"]


def test_ui_displays_requested_filters_and_sorts_as_pending_only():
    class RecordingUI:
        def __init__(self):
            self.messages = []

        def __getattr__(self, method):
            if method in {"subheader", "success", "warning", "caption", "info", "write", "markdown"}:
                return lambda text, **kwargs: self.messages.append((method, text))
            if method == "columns":
                return lambda *args, **kwargs: (self, self)
            if method == "expander":
                return lambda *args, **kwargs: nullcontext()
            if method == "json":
                return lambda data: self.messages.append(("json", data))
            if method in {"selectbox", "button"}:
                pytest.fail("Pending search request must not be applied by UI")
            raise AttributeError(method)

    st = RecordingUI()
    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(taxonomy=SimpleNamespace(path_names=lambda *_: ["Quán ăn"])),
        property_catalog={},
    )
    result = {
        "query": "quán ăn ngon gần nhất",
        "intent": {"decision": {"action": "accept", "accepted_item_id": "20"}},
        "property_extraction": {"properties": [
            {"property_id": "quality_by_reviews", "mode": "preferred", "status": "pending_review_data",
             "applied": False, "evidence_text": "ngon", "value": None},
            {"property_id": "near", "mode": "required", "status": "pending_location",
             "applied": False, "evidence_text": "gần nhất", "value": None},
        ]},
        "query_parts": {
            "filter_candidates": [], "preference_candidates": [],
            "requested_search": {
                "sort_candidates": [
                    {"kind": "distance_asc", "status": "pending_location", "evidence_text": "gần nhất"},
                    {"kind": "review_score_desc", "status": "pending_review_data", "evidence_text": "ngon"},
                ],
            },
        },
    }
    ui._render_intent_result(st, result, runtime, "vi")
    shown = " ".join(str(value) for method, value in st.messages if method != "json")
    cards = [value for method, value in st.messages if method == "markdown" and 'class="prop-item-card"' in value]
    assert len(cards) == 2
    assert "Ưu tiên" in cards[0] and "Ưu tiên nơi được đánh giá tốt" in cards[0]
    assert "Bắt buộc" in cards[1] and "Gần vị trí người dùng" in cards[1]
    assert "Dự kiến, chưa áp dụng trên POI" in cards[0] and "Dự kiến, chưa áp dụng trên POI" in cards[1]
    assert "Cần vị trí người dùng" in shown and "Cần dữ liệu đánh giá" in shown
    assert "gần nhất trước" in shown and "điểm đánh giá cao trước" in shown
    assert "Bộ lọc và xếp hạng dự kiến" not in shown
    assert result["query_parts"]["filter_candidates"] == []
    payload = next(value for method, value in st.messages if method == "json")
    assert [row["property_id"] for row in payload["property_extraction"]["properties"]] == ["quality_by_reviews", "near"]
    assert "filter_candidates" not in payload["query_parts"]["requested_search"]


def test_ui_shows_folded_category_suggestion_and_unresolved_property_clause():
    class RecordingUI:
        def __init__(self):
            self.session_state = {"intent_query": "nhà nghĩ có dịch vụ chưa xác định"}
            self.messages = []
            self.buttons = []

        def __getattr__(self, name):
            if name in {"subheader", "success", "warning", "caption", "info", "write"}:
                return lambda text, **kwargs: self.messages.append((name, text))
            if name == "button":
                return lambda text, **kwargs: self.buttons.append(kwargs["key"]) or False
            if name == "columns":
                return lambda *args, **kwargs: (self, self)
            if name == "expander":
                return lambda *args, **kwargs: nullcontext()
            if name == "json":
                return lambda payload: None
            raise AttributeError(name)

    st = RecordingUI()
    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(taxonomy=SimpleNamespace(path_names=lambda *_: ["Nhà nghỉ"])),
        property_catalog={},
    )
    result = {
        "query_parts": {
            "category_text": "nhà nghĩ", "pgvector_queries": ["nhà nghĩ"],
            "category_boundary": {"cue": "có"},
            "unresolved_property_clauses": [{"text": "có dịch vụ chưa xác định"}],
            "filter_candidates": [], "preference_candidates": [],
        },
        "intent": {
            "decision": {"action": "clarify", "reason": "folded_category_name_confirmation",
                         "options": [{"item_id": "137", "code": "motel", "label": "Nhà nghỉ", "score": None}],
                         "calibrated_confidence": 0.263},
            "confidence_assessment": {"semantic_top_cosine": 0.8078,
                                      "final_decision_source": "folded_name_confirmation"},
        },
        "property_extraction": {"properties": []},
    }
    ui._render_intent_result(st, result, runtime, "vi")
    shown = " ".join(str(text) for _, text in st.messages)
    assert "Nhà nghỉ" in shown
    assert "có dịch vụ chưa xác định" in shown
    assert "chưa tạo điều kiện lọc" in shown
    assert st.buttons == []


def test_quick_example_runs_once_and_shows_request_latency(monkeypatch):
    app_test = pytest.importorskip("streamlit.testing.v1")
    submitted = []
    metadata = SimpleNamespace(
        model_key=E5_POI_MODEL_KEY, model_id=E5_POI_MODEL_ID, model_revision=E5_POI_REVISION,
        backend="pgvector", retrieval_mode="pg_exact", generation_id=6,
    )

    def analyze(query, **kwargs):
        submitted.append((query, kwargs["language"]))
        return {
            "query": query,
            "intent": {"decision": {"action": "accept", "accepted_item_id": "23"}},
            "query_parts": {"filter_candidates": [], "preference_candidates": []},
            "property_extraction": {"properties": []},
            "timing": {"version": 1, "path": "intent_analysis", "total_ms": 12.4, "stages": [
                {"name": "category_variant", "parent": None, "status": "ok", "duration_ms": 7.0},
            ]},
        }

    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(
            metadata=metadata, taxonomy=SimpleNamespace(path_names=lambda *_: ["Quán cà phê"]),
        ),
        property_profiles=SimpleNamespace(version="test"), property_catalog={}, analyze_intent=analyze,
    )
    monkeypatch.setattr(ui, "_analysis_config", lambda: (
        ROOT / "configs/benchmark.poi.pg.v3.yaml", "quick-example-config", "taxonomy", "properties", "database",
    ))
    monkeypatch.setattr(SearchRuntime, "from_intent_config", lambda **kwargs: runtime)

    def app():
        from model_intent import ui

        ui.main()

    at = app_test.AppTest.from_function(app, default_timeout=30).run()
    at.button(key="intent-example-vi-0-0").click().run()
    assert not at.exception
    assert not at.error, [str(row.value) for row in at.error]
    assert submitted == [("quán cà phê có máy lạnh", "vi")]
    assert at.text_area[0].value == submitted[0][0]
    assert at.metric[0].value == "12.40 ms"
    assert any("Quán cà phê" in str(row.value) for row in at.success)
    assert not any(button.key and button.key.startswith("intent-clarify-") for button in at.button)
    assert any("Chưa xử lý trọn vẹn" in str(row.value) for row in at.caption)
    assert any(button.key and button.key.startswith("intent-example-vi-1-") for button in at.button)
    assert not any(button.key and button.key.startswith("intent-limitation-") for button in at.button)
    at.button(key="intent-example-vi-0-4").click().run()
    assert not at.exception
    assert submitted[-1] == ("nhà nghỉ có bãi đỗ ô tô không có chỗ cho xe lăn", "vi")
    assert len(submitted) == 2
    assert at.text_area[0].value == submitted[-1][0]
    at.button(key="intent-example-vi-0-8").click().run()
    assert not at.exception
    assert submitted[-1] == ("bảo tàng có thuyết minh tự động", "vi")
    assert len(submitted) == 3
    assert at.text_area[0].value == submitted[-1][0]
    at.button(key="intent-example-vi-1-0").click().run()
    assert not at.exception
    assert submitted[-1] == ("quán cà phê hoặc nhà thuốc có máy lạnh", "vi")
    assert len(submitted) == 4
    assert at.text_area[0].value == submitted[-1][0]
    at.button(key="intent-example-vi-1-3").click().run()
    assert not at.exception
    assert submitted[-1] == ("bệnh viện có nhà vệ sinh xe lăn vào được", "vi")
    assert len(submitted) == 5
    assert at.text_area[0].value == submitted[-1][0]
    at.radio[0].set_value("en").run()
    assert not at.exception
    at.button(key="intent-example-en-0-8").click().run()
    assert not at.exception
    assert submitted[-1] == ("museum with audio guide", "en")
    assert len(submitted) == 6
    assert at.text_area[0].value == submitted[-1][0]
    at.button(key="intent-example-en-1-2").click().run()
    assert not at.exception
    assert submitted[-1] == ("motel with a place to leave my car", "en")
    assert len(submitted) == 7
    assert at.text_area[0].value == submitted[-1][0]


def test_quick_examples_offer_two_matching_clickable_groups():
    for language, cases in ui._QUICK_CASES.items():
        assert len(cases) == 9
        assert len({label for label, _ in cases}) == 9, language
        assert len({query.casefold() for _, query in cases}) == 9, language
        assert len({label.split(" · ")[0] for label, _ in cases}) >= 8, language
        limitations = ui._LIMITATION_CASES[language]
        assert len(limitations) == 4
        assert all(label and example for label, example in limitations)
        assert len({example.casefold() for _, example in limitations}) == 4
        assert not ({example.casefold() for _, example in limitations}
                    & {query.casefold() for _, query in cases})
    assert ui._QUICK_CASES["vi"][0][1] == "quán cà phê có máy lạnh"
    assert ui._LIMITATION_CASES["vi"][0][1] == "quán cà phê hoặc nhà thuốc có máy lạnh"
    assert ui._LIMITATION_CASES["vi"][3] == ("Yêu cầu kết hợp", "bệnh viện có nhà vệ sinh xe lăn vào được")
    assert not any("ship đồ" in query for _, query in ui._LIMITATION_CASES["vi"])


def test_quick_examples_have_directly_recognized_properties():
    from model_intent.poi_properties import load_property_definitions
    from model_intent.property_extraction import PropertyExtractor

    extractor = PropertyExtractor(load_property_definitions(ROOT / "data/properties/poi_properties.poc.v8.json"))
    for language, cases in ui._QUICK_CASES.items():
        for label, query in cases:
            assert any(row.applied for row in extractor.extract(query).properties), (language, label)


def test_result_columns_keep_latency_and_technical_details_full_width():
    events = []

    class RecordingColumn:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, method):
            if method in {"subheader", "success", "info", "warning", "caption", "write", "markdown", "error"}:
                return lambda *args, **kwargs: events.append((self.name, method, args[0]))
            raise AttributeError(method)

    class RecordingPage(RecordingColumn):
        def __init__(self):
            super().__init__("full-width")

        def columns(self, count, **kwargs):
            events.append(("layout", "columns", (count, kwargs)))
            return RecordingColumn("category"), RecordingColumn("properties")

        def expander(self, label, **kwargs):
            events.append(("full-width", "expander", label))
            return nullcontext()

        def metric(self, label, value):
            events.append(("full-width", "metric", (label, value)))

        def table(self, rows):
            events.append(("full-width", "table", rows))

        def json(self, payload):
            events.append(("full-width", "json", payload))

    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(taxonomy=SimpleNamespace(path_names=lambda *_: ["Quán cà phê"])),
        property_catalog={"air_conditioning": {"aliases": {"vi": ["máy lạnh"]}}},
    )
    result = {
        "query": "quán cà phê có máy lạnh, mát rượi",
        "intent": {"decision": {"action": "accept", "accepted_item_id": "23"}},
        "property_extraction": {"properties": [{
            "property_id": "air_conditioning", "mode": "required", "applied": True,
            "evidence_text": "máy lạnh",
        }]},
        "query_parts": {
            "filter_candidates": [{"property_id": "air_conditioning"}],
            "preference_candidates": [],
            "semantic_property_suggestions": [{
                "span": {"text": "mát rượi", "start_offset": 26, "end_offset": 34},
                "candidates": [{"property_id": "shade", "label": "bóng mát"}],
            }],
        },
        "timing": {"version": 1, "path": "intent_analysis", "total_ms": 12.0,
                   "stages": [{"name": "category_variant", "parent": None, "status": "ok", "duration_ms": 8.0}]},
    }
    ui._render_intent_result(RecordingPage(), result, runtime, "vi")
    assert ("layout", "columns", (2, {"gap": "large"})) in events
    assert ("category", "subheader", "Category · Loại địa điểm") in events
    assert ("properties", "subheader", "Properties · Yêu cầu thêm") in events
    assert any(location == "properties" and method == "markdown" and "máy lạnh" in value
               for location, method, value in events)
    assert any(location == "properties" and method == "markdown" and "bóng mát" in value
               for location, method, value in events)
    assert ("full-width", "metric", ("Tổng thời gian trên server", "12.00 ms")) in events
    assert any(location == "full-width" and method == "json" for location, method, _ in events)
    assert not any(location == "full-width" and method == "subheader" and "Category" in value
                   for location, method, value in events)


def test_unresolved_category_shows_ranked_candidates_only_when_place_seeking():
    class RecordingUI:
        def __init__(self):
            self.messages = []

        def __getattr__(self, name):
            if name in {"subheader", "success", "warning", "caption", "info", "write", "markdown"}:
                return lambda text, **kwargs: self.messages.append((name, text))
            if name == "columns":
                return lambda *args, **kwargs: (self, self)
            if name == "expander":
                return lambda *args, **kwargs: nullcontext()
            if name == "json":
                return lambda payload: None
            if name == "button":
                pytest.fail("Category clarification controls must not be rendered")
            raise AttributeError(name)

    runtime = SimpleNamespace(
        category_runtime=SimpleNamespace(taxonomy=SimpleNamespace(path_names=lambda item_id, *_: [f"Loại {item_id}"])),
        property_catalog={},
    )
    result = {
        "query": "tìm quán cà phê hoặc nhà thuốc",
        "query_parts": {"filter_candidates": [], "preference_candidates": []},
        "intent": {"decision": {"action": "clarify", "options": [
            {"item_id": "23", "label": "Quán cà phê"}, {"item_id": "34", "label": "Nhà thuốc"},
        ]}, "prediction": {"hits": [
            {"item_id": str(i), "score": 1 - i / 100} for i in (23, 34, 40, 41, 42, 43)
        ]}},
        "property_extraction": {"properties": []},
    }
    st = RecordingUI()
    ui._render_intent_result(st, result, runtime, "vi")
    shown = " ".join(str(text) for kind, text in st.messages if kind != "json")
    assert "Loại 23" in shown and "Loại 34" in shown
    assert "Loại 42" in shown and "Loại 43" not in shown
    assert "chỉ để tham khảo" in shown
    assert result["intent"]["decision"]["action"] == "clarify"

    result["intent"]["decision"] = {"action": "reject", "reason": "informational_not_place_search"}
    result["intent"]["category_resolution"] = {"source": "informational_veto"}
    st = RecordingUI()
    ui._render_intent_result(st, result, runtime, "vi")
    shown = " ".join(str(text) for kind, text in st.messages)
    assert "không yêu cầu tìm địa điểm" in shown
    assert "Loại 23" not in shown

    result["intent"]["decision"] = {"action": "reject", "reason": "no_category_phrase"}
    result["intent"]["prediction"] = {"hits": []}
    result["query_parts"]["vector_query_mode"] = "no_category_phrase"
    st = RecordingUI()
    ui._render_intent_result(st, result, runtime, "vi")
    assert not any("Loại 23" in str(text) for kind, text in st.messages if kind != "json")
