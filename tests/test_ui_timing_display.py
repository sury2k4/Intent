from contextlib import nullcontext

import pytest

from model_intent import ui


class RecordingUI:
    def __init__(self):
        self.captions = []
        self.metrics = []
        self.tables = []
        self.expanders = []
        self.subheaders = []

    def subheader(self, text):
        self.subheaders.append(text)

    def metric(self, label, value):
        self.metrics.append((label, value))

    def caption(self, text):
        self.captions.append(text)

    def table(self, rows):
        self.tables.append(rows)

    def expander(self, label, **kwargs):
        self.expanders.append(label)
        return nullcontext()


def test_summary_counts_only_root_spans_and_keeps_detail_order():
    result = {"timing": {"version": 1, "path": "intent_analysis", "total_ms": 27.0, "stages": [
        {"name": "property_extraction", "parent": None, "status": "ok", "duration_ms": 2.0},
        {"name": "query_interpretation", "parent": None, "status": "ok", "duration_ms": 1.0},
        {"name": "category_variant", "parent": None, "status": "ok", "duration_ms": 7.0, "variant_index": 0},
        {"name": "category_retrieval", "parent": 2, "status": "ok", "duration_ms": 6.0},
        {"name": "category_db_variant", "parent": 3, "status": "ok", "duration_ms": 4.0, "row_count": 17},
        {"name": "category_variant", "parent": None, "status": "ok", "duration_ms": 5.0, "variant_index": 1},
        {"name": "category_resolution", "parent": None, "status": "ok", "duration_ms": 1.0},
        {"name": "property_reconcile", "parent": None, "status": "ok", "duration_ms": 0.8},
        {"name": "property_semantic_suggestions", "parent": None, "status": "ok", "duration_ms": 6.0},
        {"name": "property_db_exact", "parent": 8, "status": "ok", "duration_ms": 2.0, "span_count": 1},
        {"name": "response_serialize", "parent": None, "status": "ok", "duration_ms": 1.0},
    ]}}
    view = ui._timing_view_model(result)
    assert view["total_ms"] == 27
    assert [(row["Giai đoạn"], row["Thời gian"]) for row in view["groups"]] == [
        ("Tách câu hỏi", "3.00 ms"), ("Loại địa điểm", "13.00 ms"),
        ("Yêu cầu thêm", "6.80 ms"), ("Hoàn tất", "1.00 ms"),
    ]
    assert [row["Cấp"] for row in view["details"]] == [0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0]
    assert [row["#"] for row in view["details"]] == list(range(1, 12))
    assert "biến thể 1" in view["details"][2]["Bước thực hiện"]
    assert "17 hàng" not in view["details"][4]["Bước thực hiện"]
    assert "hàng 17" in view["details"][4]["Bước thực hiện"]
    assert "Tìm thuộc tính bằng pgvector" in view["details"][9]["Bước thực hiện"]

    st = RecordingUI()
    ui._render_timing_breakdown(st, result)
    assert st.metrics == [("Tổng thời gian trên server", "27.00 ms")]
    assert st.tables == [view["groups"], view["details"]]
    assert st.expanders == ["Xem thời gian từng bước"]
    assert "không cập nhật trực tiếp" in st.captions[0]
    assert "bước con đã nằm trong thời gian bước cha" in st.captions[0]


def test_selection_explains_skipped_vector_and_confirmation_preserves_old_timing():
    result = {
        "timing": {"version": 1, "path": "intent_clarification_selection", "total_ms": 8.3, "stages": [
            {"name": "category_selection", "parent": None, "status": "ok", "duration_ms": 1.0},
            {"name": "category_retrieval", "parent": None, "status": "skipped", "duration_ms": None},
            {"name": "category_decision", "parent": None, "status": "skipped", "duration_ms": None},
            {"name": "property_semantic_suggestions", "parent": None, "status": "ok", "duration_ms": 2.5},
        ]},
        "query_parts": {"confirmed_semantic_choices": [{"property_id": "parking"}]},
    }
    view = ui._timing_view_model(result)
    assert view["request_label"] == "chọn category sau hỏi lại"
    assert view["groups"][1] == {
        "Giai đoạn": "Loại địa điểm", "Thời gian": "1.00 ms",
        "Trạng thái": "Chọn thủ công · không tìm vector",
    }
    assert view["details"][1]["Thời gian"] == "—"
    assert view["details"][1]["Trạng thái"] == "Bỏ qua"
    st = RecordingUI()
    ui._render_timing_breakdown(st, result)
    assert any("Xác nhận thuộc tính" in caption for caption in st.captions)


def test_unavailable_semantic_backend_is_not_shown_as_success_or_zero():
    result = {
        "timing": {"version": 1, "path": "intent_analysis", "total_ms": 9.0, "stages": [
            {"name": "property_semantic_suggestions", "parent": None, "status": "ok", "duration_ms": 6.0},
        ]},
        "query_parts": {"semantic_property_metadata": {"status": "unavailable", "error_type": "RuntimeError"}},
    }
    view = ui._timing_view_model(result)
    assert view["groups"][2] == {
        "Giai đoạn": "Yêu cầu thêm", "Thời gian": "6.00 ms", "Trạng thái": "Không khả dụng",
    }
    assert view["details"][0]["Trạng thái"] == "Không khả dụng"
    assert ui._timing_view_model({"timing": {"version": 1, "path": "intent_analysis", "total_ms": 2.0,
        "stages": []}, "query_parts": {"semantic_property_metadata": {"status": "skipped"}}})["groups"][2] == {
        "Giai đoạn": "Yêu cầu thêm", "Thời gian": "—", "Trạng thái": "Bỏ qua",
    }
    st = RecordingUI()
    ui._render_timing_breakdown(st, {"timing": {"version": 1, "path": "intent_analysis", "total_ms": 2.0,
        "stages": []}, "query_parts": {"semantic_property_metadata": {"status": "skipped"}}})
    assert any("không cần tìm vector thuộc tính" in caption for caption in st.captions)


def test_missing_invalid_or_unknown_timing_never_fabricates_duration():
    st = RecordingUI()
    ui._render_timing_breakdown(st, {})
    assert st.metrics == [] and st.tables == []
    assert "Chưa có số đo" in st.captions[0]
    assert ui._timing_view_model({"timing": {"version": 2, "total_ms": 2.0}}) is None
    assert ui._timing_view_model({"timing": {"version": 1, "total_ms": float("nan")}}) is None
    view = ui._timing_view_model({"timing": {"version": 1, "path": "unrecognized", "total_ms": 3.0,
        "stages": [
            {"name": "property_query_encode", "parent": 42, "status": "error", "duration_ms": 1.0},
            {"name": "response_serialize", "parent": None, "status": "ok", "duration_ms": None},
        ]}})
    assert view["unknown_parents"] == 1
    assert view["details"][0]["Cấp"] == "?"
    assert view["details"][0]["Trạng thái"] == "Lỗi"
    assert view["details"][1]["Thời gian"] == "—"
    assert view["groups"][-1]["Thời gian"] == "—"
    assert view["request_label"] == "request không rõ loại"


def test_streamlit_renders_summary_and_separate_detail_table():
    app_test = pytest.importorskip("streamlit.testing.v1")

    def app():
        import streamlit as st
        from model_intent import ui

        ui._render_timing_breakdown(st, {"timing": {
            "version": 1, "path": "intent_analysis", "total_ms": 13.2,
            "stages": [
                {"name": "category_variant", "parent": None, "status": "ok", "duration_ms": 8.0},
                {"name": "category_retrieval", "parent": 0, "status": "ok", "duration_ms": 7.5},
                {"name": "property_semantic_suggestions", "parent": None, "status": "skipped", "duration_ms": None},
            ],
        }})

    at = app_test.AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    assert at.metric[0].label == "Tổng thời gian trên server"
    assert at.metric[0].value == "13.20 ms"
    assert len(at.table) == 2
    assert at.expander[0].label == "Xem thời gian từng bước"
