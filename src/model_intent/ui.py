import html
import json
import math
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/benchmark.poi.pg.v3.yaml"
DEFAULT_TAXONOMY = PROJECT_ROOT / "data/taxonomies/place_types.poc.v3.json"


def _clean_html(html_str: str) -> str:
    """Format HTML/CSS for Streamlit markdown: strip indentation, blank lines, and comments so CommonMark never treats HTML as an indented code block."""
    lines = []
    for raw in html_str.splitlines():
        line = raw.strip()
        if not line or line.startswith("<!--"):
            continue
        lines.append(line)
    return " ".join(lines)


def _load_env_database_url() -> bool:
    """Read DATABASE_URL from .env file in project root and set os.environ."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return False
    try:
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
        if values.get("DATABASE_URL"):
            os.environ["DATABASE_URL"] = values["DATABASE_URL"]
            return True
    except Exception:
        pass
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "DATABASE_URL":
                    os.environ["DATABASE_URL"] = v.strip().strip("\"'")
                    return True
    except Exception:
        pass
    return False


def _auto_load_dotenv_if_needed() -> None:
    """Auto-load DATABASE_URL from .env if running outside of test runner."""
    if os.getenv("DATABASE_URL"):
        return
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        return
    _load_env_database_url()


def _configured_taxonomy(config_path: Path, config: dict[str, Any] | None = None) -> Path:
    """Resolve the serving taxonomy relative to the project root."""
    if config is None:
        from model_intent.serving_config import load_yaml

        config = load_yaml(config_path)
    selected = config.get("taxonomy_path")
    if not selected:
        return DEFAULT_TAXONOMY
    path = Path(selected)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _pinned_category_config() -> tuple[Path, dict[str, Any], str]:
    _auto_load_dotenv_if_needed()
    from model_intent.serving_config import load_yaml
    from model_intent.db import validate_database_target
    from model_intent.search_runtime import E5_POI_DIMENSION, E5_POI_MODEL_ID, E5_POI_MODEL_KEY, E5_POI_REVISION

    config_path = Path(os.getenv("MODEL_INTENT_CONFIG", str(DEFAULT_CONFIG))).resolve()
    config = load_yaml(config_path)
    if (
        config.get("retrieval_backend") != "pgvector"
        or config.get("retrieval_mode") != "pg_exact"
        or config.get("require_explicit_database_url") is not True
    ):
        raise RuntimeError("UI requires exact pgvector category retrieval and an explicit DATABASE_URL")
    profiles = config.get("models") or []
    if len(profiles) != 1 or any(
        profile.get("key") != E5_POI_MODEL_KEY
        or profile.get("model_id") != E5_POI_MODEL_ID
        or profile.get("revision") != E5_POI_REVISION
        or profile.get("expected_dimension") != E5_POI_DIMENSION
        for profile in profiles
    ):
        raise RuntimeError("UI requires the pinned E5-base 768D model profile")
    return config_path, config, validate_database_target(config)


def _analysis_config() -> tuple[Path, str, str, str, str]:
    """Fingerprint category and property resources without storing credentials."""
    config_path, config, database_url = _pinned_category_config()
    property_config = config.get("property_intent") or {}
    property_name = property_config.get("property_path")
    if not property_name:
        raise RuntimeError("UI requires a versioned property catalog")
    project_root = config_path.parent.parent
    property_path = Path(property_name)
    if not property_path.is_absolute():
        property_path = (project_root / property_path).resolve()
    property_payload = json.loads(property_path.read_text(encoding="utf-8"))
    resources = [property_path]
    semantic_path = property_config.get("property_semantic_profiles_path")
    if semantic_path:
        semantic_path = Path(semantic_path)
        resources.append(semantic_path if semantic_path.is_absolute() else (project_root / semantic_path).resolve())
    if property_payload.get("base_catalog"):
        resources.append((property_path.parent / property_payload["base_catalog"]).resolve())
    for key in ("enrichment_path", "policy_path", "confidence_calibration_path", "category_query_aliases_path"):
        if config.get(key):
            path = Path(config[key])
            resources.append(path if path.is_absolute() else (project_root / path).resolve())
    digest = sha256()
    for path in resources:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return (
        config_path,
        sha256(config_path.read_bytes()).hexdigest(),
        sha256(_configured_taxonomy(config_path, config).read_bytes()).hexdigest(),
        digest.hexdigest(),
        sha256(database_url.encode("utf-8")).hexdigest(),
    )


def _safe_error_message(exc: Exception) -> str:
    message = str(exc)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        message = message.replace(database_url, "[DATABASE_URL redacted]")
        try:
            password = urlsplit(database_url).password
        except ValueError:
            password = None
        if password:
            for value in {password, unquote(password)}:
                message = message.replace(value, "[redacted]")
    return message


_STAGE_LABELS = {
    "validate": "Kiểm tra tham số", "property_extraction": "Trích xuất thuộc tính",
    "query_interpretation": "Tách category và thuộc tính", "category_variant": "Truy vấn category",
    "category_retrieval": "Tìm category bằng vector", "category_query_variants": "Tạo biến thể truy vấn",
    "category_encode_normalize": "E5 encode và chuẩn hóa",
    "category_db_variant": "Truy vấn pgvector theo biến thể",
    "category_pool_wait": "Chờ kết nối category", "category_db_execute_fetch": "SQL category và lấy dữ liệu",
    "category_hydrate_merge": "Gộp hồ sơ category", "category_sort_hits": "Sắp xếp category",
    "category_decision": "Quyết định category", "category_calibration": "Đánh giá bằng chứng",
    "category_serialize": "Đóng gói category", "category_resolution": "Giải quyết category",
    "near_tie_clarification": "Kiểm tra category gần điểm", "category_selection": "Chọn category",
    "property_reconcile": "Kiểm tra thuộc tính phù hợp",
    "property_semantic_suggestions": "Gợi ý thuộc tính E5",
    "property_query_encode": "E5 encode cụm thuộc tính", "property_db_exact": "Tìm thuộc tính bằng pgvector",
    "property_cosine_rank": "Sắp xếp gợi ý thuộc tính", "response_serialize": "Đóng gói kết quả",
}
_TIMING_GROUPS = (
    ("Tách câu hỏi", frozenset({"validate", "property_extraction", "query_interpretation"})),
    ("Loại địa điểm", frozenset({
        "category_variant", "category_retrieval", "category_decision", "category_resolution",
        "near_tie_clarification", "category_selection", "category_serialize",
    })),
    ("Yêu cầu thêm", frozenset({"property_reconcile", "property_semantic_suggestions"})),
    ("Hoàn tất", frozenset({"response_serialize"})),
)
_TIMING_PATHS = {
    "intent_analysis": "phân tích câu hỏi",
    "intent_clarification_selection": "chọn category sau hỏi lại",
}
_TIMING_COUNTS = {
    "variant_index": "biến thể", "span_count": "cụm", "row_count": "hàng",
    "candidate_count": "ứng viên",
}
_QUICK_CASES = {
    "vi": (
        ("Cà phê · máy lạnh", "quán cà phê có máy lạnh"),
        ("Công viên · bóng mát", "công viên có bóng mát và ghế ngồi"),
        ("Xe buýt · mái che", "điểm dừng xe buýt có mái che và ghế ngồi"),
        ("Bệnh viện · xe lăn", "bệnh viện có lối tiếp cận xe lăn"),
        ("Nhà nghỉ · đỗ ô tô", "nhà nghỉ có bãi đỗ ô tô không có chỗ cho xe lăn"),
        ("ATM · rút tiền", "ATM rút tiền có mái che"),
        ("Cây xăng · dầu diesel", "cây xăng không có dầu diesel"),
        ("Phòng khám · đặt lịch", "phòng khám đặt lịch khám trước"),
        ("Bảo tàng · thuyết minh", "bảo tàng có thuyết minh tự động"),
    ),
    "en": (
        ("Cafe · air conditioning", "cafe with air conditioning"),
        ("Park · shade", "park with shade and seating"),
        ("Bus stop · shelter", "bus stop with shelter and seating"),
        ("Hospital · accessibility", "hospital with wheelchair access"),
        ("Motel · parking", "Motel with car parking, no wheelchair access"),
        ("Restaurant · takeaway", "restaurant with takeaway"),
        ("Library · air conditioning", "library with air conditioning"),
        ("Supermarket · parking", "supermarket with parking"),
        ("Museum · audio guide", "museum with audio guide"),
    ),
}
_LIMITATION_CASES = {
    "vi": (
        ("Nhiều loại địa điểm", "quán cà phê hoặc nhà thuốc có máy lạnh"),
        ("Thuộc tính mơ hồ", "quán cà phê có mát rượi"),
        ("Đồng nghĩa ngoài catalog", "nhà nghỉ có chỗ cất ô tô"),
        ("Yêu cầu kết hợp", "bệnh viện có nhà vệ sinh xe lăn vào được"),
    ),
    "en": (
        ("Multiple place types", "cafe or pharmacy with air conditioning"),
        ("Ambiguous property", "cafe with a cooling breeze"),
        ("Synonym outside the catalog", "motel with a place to leave my car"),
        ("Compound requirement", "bus stop with a sheltered live arrival display"),
    ),
}


def _render_quick_case_buttons(st, cases, language: str, group_index: int) -> str | None:
    selected = None
    columns = st.columns(min(3, len(cases)))
    for index, (label, example) in enumerate(cases):
        with columns[index % len(columns)]:
            if st.button(
                label, key=f"intent-example-{language}-{group_index}-{index}",
                help=example, use_container_width=True,
            ):
                st.session_state["intent_query_input"] = example
                selected = example
    return selected


def _measured_ms(value: Any) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _format_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return "< 0.01 ms" if 0 < value < 0.01 else f"{value:.2f} ms"


def _timing_view_model(result: dict[str, Any]) -> dict[str, Any] | None:
    timing = result.get("timing") or {}
    total = _measured_ms(timing.get("total_ms"))
    if timing.get("version") != 1 or total is None:
        return None
    stages = timing.get("stages") or []
    if not isinstance(stages, list):
        stages = []
    property_meta = (result.get("query_parts") or {}).get("semantic_property_metadata") or {}
    roots: dict[str, list[tuple[dict[str, Any], float | None]]] = {
        title: [] for title, _ in _TIMING_GROUPS
    }
    depths: list[int | None] = []
    details: list[dict[str, str | int]] = []
    unknown_parents = 0
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            depths.append(None)
            unknown_parents += 1
            continue
        parent = stage.get("parent")
        if "parent" not in stage:
            depth = None
        elif parent is None:
            depth = 0
        elif type(parent) is int and 0 <= parent < index and depths[parent] is not None:
            depth = depths[parent] + 1
        else:
            depth = None
        depths.append(depth)
        if depth is None:
            unknown_parents += 1
        name = str(stage.get("name") or "unknown")
        duration = _measured_ms(stage.get("duration_ms"))
        status = stage.get("status")
        if name == "property_semantic_suggestions" and property_meta.get("status") == "unavailable":
            state = "Không khả dụng"
        elif status == "ok" and duration is None:
            state = "Chưa ghi nhận"
        else:
            state = {"ok": "Đã chạy", "skipped": "Bỏ qua", "error": "Lỗi"}.get(status, "Không rõ")
        counts = [
            f"{_TIMING_COUNTS.get(key, key.replace('_', ' '))} {value + 1 if key == 'variant_index' else value:,}"
            for key, value in stage.items()
            if key not in {"name", "parent", "status", "duration_ms"} and type(value) is int
        ]
        label = _STAGE_LABELS.get(name, name.replace("_", " "))
        details.append({
            "#": index + 1,
            "Cấp": depth if depth is not None else "?",
            "Bước thực hiện": label + (f" · {', '.join(counts)}" if counts else ""),
            "Thời gian": _format_ms(duration),
            "Trạng thái": state,
        })
        if depth == 0:
            group = next((title for title, names in _TIMING_GROUPS if name in names), "Khác")
            roots.setdefault(group, []).append((stage, duration))

    groups = []
    for title, rows in roots.items():
        if title == "Khác" and not rows:
            continue
        measured = [ms for row, ms in rows if ms is not None and row.get("status") != "skipped"]
        if title == "Yêu cầu thêm" and property_meta.get("status") == "unavailable":
            state = "Không khả dụng"
        elif any(row.get("status") == "error" for row, _ in rows):
            state = "Lỗi"
        elif title == "Loại địa điểm" and timing.get("path") == "intent_clarification_selection":
            state = "Chọn thủ công · không tìm vector"
        elif title == "Loại địa điểm" and (result.get("query_parts") or {}).get("vector_query_mode") == "no_category_phrase":
            state = "Không có category · không tìm vector"
        elif measured:
            state = "Đã chạy"
        elif rows or (title == "Yêu cầu thêm" and str(property_meta.get("status", "")).startswith("skipped")):
            state = "Bỏ qua"
        else:
            state = "Chưa ghi nhận"
        groups.append({
            "Giai đoạn": title,
            "Thời gian": _format_ms(sum(measured) if measured else None),
            "Trạng thái": state,
        })
    return {
        "total_ms": total,
        "request_label": _TIMING_PATHS.get(timing.get("path"), "request không rõ loại"),
        "groups": groups,
        "details": details,
        "unknown_parents": unknown_parents,
        "confirmed_after_request": bool((result.get("query_parts") or {}).get("confirmed_semantic_choices")),
        "property_semantic_status": property_meta.get("status"),
    }


def _render_timing_breakdown(st, result: dict[str, Any]) -> None:
    st.subheader("Thời gian xử lý từng bước")
    view = _timing_view_model(result)
    if view is None:
        st.caption("Chưa có số đo hợp lệ cho request này; không suy ra thời gian từ hoạt ảnh.")
        return
    st.metric("Tổng thời gian trên server", _format_ms(view["total_ms"]))
    st.caption("Số đo của lượt xử lý đã hoàn tất, không cập nhật trực tiếp. "
               "Chi tiết bước con đã nằm trong thời gian bước cha; tổng thời gian có thể gồm phần chưa đo riêng.")
    if view["confirmed_after_request"]:
        st.caption("Xác nhận thuộc tính chỉ cập nhật kết quả; thời gian hiển thị vẫn thuộc request trước đó.")
    if view["property_semantic_status"] == "skipped":
        st.caption("Không có cụm thuộc tính chưa rõ, nên không cần tìm vector thuộc tính.")
    elif view["property_semantic_status"] == "skipped_non_place":
        st.caption("Không cần tìm gợi ý thuộc tính vì câu hỏi không tìm địa điểm.")
    st.table(view["groups"])
    with st.expander("Xem thời gian từng bước"):
        st.caption("Cấp 0 là bước chính; cấp lớn hơn nằm trong thời gian bước cha. '—' là không đo/bỏ qua, không phải 0 ms.")
        if view["unknown_parents"]:
            st.caption(f"{view['unknown_parents']} bước thiếu quan hệ lồng nhau, không được gộp vào thời gian giai đoạn.")
        if view["details"]:
            st.table(view["details"])
        else:
            st.caption("Request này không ghi lại thời gian của từng bước.")


def _render_intent_analysis(st) -> None:
    from model_intent.search_runtime import E5_POI_MODEL_KEY, SearchRuntime

    st.title("Hiểu yêu cầu tìm địa điểm")
    st.caption("Tách loại địa điểm và các yêu cầu thêm trong câu hỏi. Chưa tìm hay kiểm tra địa điểm thực tế.")

    @st.cache_resource(max_entries=2, show_spinner=False)
    def load_intent_runtime(
        config_path: str, config_hash: str, taxonomy_hash: str,
        resources_hash: str, database_hash: str, taxonomy_path: Path,
    ):
        return SearchRuntime.from_intent_config(
            config_path=config_path,
            taxonomy_path=taxonomy_path,
            model_key=E5_POI_MODEL_KEY,
            local_files_only=True,
        )

    try:
        config_path, config_hash, taxonomy_hash, resources_hash, database_hash = _analysis_config()
        taxonomy_path = _configured_taxonomy(config_path)
        with st.spinner("Đang tải bộ phân tích…"):
            runtime = load_intent_runtime(
                str(config_path), config_hash, taxonomy_hash, resources_hash,
                database_hash, taxonomy_path,
            )
    except Exception as exc:
        st.error(f"Chưa thể mở bộ phân tích: {_safe_error_message(exc)}")
        st.info("Kiểm tra kết nối cơ sở dữ liệu và dữ liệu category đã sẵn sàng; không cần dữ liệu POI đang phục vụ.")
        return

    meta = runtime.category_runtime.metadata
    identity = (
        "intent-ui-read-only-v2", str(config_path), config_hash, taxonomy_hash, resources_hash, database_hash,
        meta.model_key, meta.model_revision, meta.backend, meta.generation_id,
        runtime.property_profiles.version if runtime.property_profiles else None,
    )
    if st.session_state.get("intent_runtime_identity") != identity:
        st.session_state.pop("intent_result", None)
        st.session_state.pop("intent_language", None)
        st.session_state["intent_runtime_identity"] = identity
    with st.sidebar:
        st.subheader("Cách xem")
        language = st.radio("Query language", ("vi", "en"), horizontal=True)
        st.caption("Ứng viên chưa chắc chắn chỉ để tham khảo, không tự tạo điều kiện lọc.")

    if "intent_query_input" not in st.session_state:
        st.session_state["intent_query_input"] = "quán cà phê có máy lạnh"
    st.subheader("Mẫu test nhanh" if language == "vi" else "Quick test examples")
    st.caption("Đã xử lý được" if language == "vi" else "Handled examples")
    quick_query = _render_quick_case_buttons(st, _QUICK_CASES[language], language, 0)
    st.caption("Chưa xử lý trọn vẹn" if language == "vi" else "Not fully resolved")
    limitation_query = _render_quick_case_buttons(st, _LIMITATION_CASES[language], language, 1)
    if limitation_query is not None:
        quick_query = limitation_query
    with st.form("intent_analysis_form"):
        query = st.text_area(
            "Bạn đang tìm nơi như thế nào?" if language == "vi" else "What place are you looking for?",
            key="intent_query_input",
            height=100,
            placeholder="Ví dụ: quán cà phê có máy lạnh",
        )
        submitted = st.form_submit_button("Phân tích query", type="primary", width="stretch")
    if quick_query is not None or submitted:
        try:
            selected_query = (quick_query if quick_query is not None else query).strip()
            st.session_state["intent_query"] = selected_query
            st.session_state["intent_result"] = runtime.analyze_intent(selected_query, language=language)
            st.session_state["intent_language"] = language
        except Exception as exc:
            st.session_state.pop("intent_result", None)
            st.error(f"Không thể phân tích câu hỏi: {_safe_error_message(exc)}")

    result = st.session_state.get("intent_result")
    if result is None:
        st.info("Nhập mô tả địa điểm và các yêu cầu để nhận category cùng thuộc tính được bóc tách.")
        return
    if st.session_state.get("intent_language") not in (None, language):
        st.info("Đã đổi ngôn ngữ. Phân tích lại để cập nhật kết quả.")
        return
    _render_intent_result(st, result, runtime, language)


def _category_name(taxonomy, item_id: Any, language: str, fallback: str) -> str:
    if item_id is not None:
        try:
            return " › ".join(taxonomy.path_names(str(item_id), language))
        except (KeyError, ValueError, AttributeError):
            pass
    return fallback


def _render_intent_result(st, result, runtime, language: str) -> None:
    taxonomy = runtime.category_runtime.taxonomy
    intent = result.get("intent") or {}
    decision = intent.get("decision") or {}
    action = str(decision.get("action", "unknown")).lower()
    accepted_id = decision.get("accepted_item_id")
    parts = result.get("query_parts") or {}
    hits = (intent.get("prediction") or {}).get("hits") or []
    category_text = parts.get("category_text")
    category_st, property_st = st.columns(2, gap="large")

    category_st.subheader("Category · Loại địa điểm")
    if action == "accept" and accepted_id:
        category_st.success(_category_name(taxonomy, accepted_id, language, str(accepted_id)))
        category_st.caption("Đã nhận diện loại địa điểm trong câu hỏi; chưa tìm địa điểm cụ thể.")
    elif parts.get("vector_query_mode") == "no_category_phrase" or decision.get("reason") == "no_category_phrase":
        category_st.info("Chưa có loại địa điểm trong câu hỏi. Bạn vẫn có thể xem các yêu cầu thêm bên dưới.")
    elif (intent.get("category_resolution") or {}).get("source") in {
        "non_place_veto", "informational_veto", "informational_query",
    } or decision.get("reason") in {
        "informational_not_place_search", "negated_place_only", "digital_product_not_place_search",
    }:
        category_st.info("Câu này không yêu cầu tìm địa điểm nên chưa chọn loại địa điểm.")
    else:
        candidates = []
        seen_ids = set()
        for hit in hits[:5]:
            item_id = hit.get("item_id")
            if item_id is not None and str(item_id) not in seen_ids:
                candidates.append(_category_name(taxonomy, item_id, language, str(hit.get("code") or item_id)))
                seen_ids.add(str(item_id))
        alternatives = []
        options = decision.get("options") if action == "clarify" else []
        for option in options or []:
            item_id = option.get("item_id")
            if item_id is not None and str(item_id) not in seen_ids:
                alternatives.append(_category_name(
                    taxonomy, item_id, language, str(option.get("label") or option.get("code") or item_id),
                ))
                seen_ids.add(str(item_id))
        if candidates:
            category_st.warning("Chưa phân định được loại địa điểm. Các khả năng gần nhất chỉ để tham khảo:")
            for index, label in enumerate(candidates, start=1):
                category_st.write(f"{index}. {label}")
        if alternatives:
            category_st.caption("Cách hiểu khác từ câu hỏi (chưa xác nhận):" if candidates else "Các cách hiểu có thể có (chưa xác nhận):")
            for label in alternatives:
                category_st.write(f"• {label}")
        if not candidates and not alternatives:
            category_st.info("Chưa xác định được loại địa điểm từ câu hỏi này.")
    if category_text and parts.get("vector_query_mode") != "no_category_phrase":
        category_st.caption(f"Phần câu nói về loại địa điểm: {category_text}")

    # Properties section
    extraction = result.get("property_extraction") or {}
    properties = extraction.get("properties") or []
    filter_candidates = parts.get("filter_candidates") or []
    preference_candidates = parts.get("preference_candidates") or []

    property_st.subheader("Properties · Yêu cầu thêm")
    property_st.caption(
        f"Đã hiểu {len(filter_candidates)} điều kiện bắt buộc/loại trừ và "
        f"{len(preference_candidates)} ưu tiên. Đây chỉ là cách hiểu câu hỏi, chưa lọc địa điểm nào."
    )
    if parts.get("applicability") == "pending_category":
        property_st.caption("Chưa xác định loại địa điểm, nên một số yêu cầu chưa thể kiểm tra độ phù hợp.")

    if not properties:
        property_st.info("Chưa nhận diện được yêu cầu thêm nào.")
    else:
        for row in properties:
            prop_id = str(row.get("property_id", ""))
            definition = runtime.property_catalog.get(prop_id, {})
            aliases = (definition.get("aliases") or {}).get(language) or []
            label = aliases[0] if aliases else prop_id
            mode_raw = row.get("mode", "required")
            mode_label = {"required": "Bắt buộc", "preferred": "Ưu tiên", "excluded": "Loại trừ"}.get(
                mode_raw, "Chưa rõ vai trò"
            )
            if not row.get("applied"):
                state_desc = "Chưa áp dụng" + (f": {row['ignored_reason']}" if row.get("ignored_reason") else "")
            elif prop_id == "radius_km":
                state_desc = "Chỉ ghi nhận khoảng cách, chưa lọc địa điểm"
            elif action == "accept" and accepted_id:
                state_desc = "Đã hiểu yêu cầu; chưa kiểm tra trên địa điểm thực tế"
            else:
                state_desc = "Đã hiểu yêu cầu; chưa kiểm tra độ phù hợp với loại địa điểm"
            evidence = row.get("evidence_text") or "không rõ"
            if hasattr(property_st, "markdown"):
                property_st.markdown(
                    f'<div class="prop-item-card">'
                    f'<span class="prop-pill">{html.escape(str(mode_label))}</span>'
                    f'<strong>{html.escape(str(label))}</strong>'
                    f'<div class="prop-evidence">Từ câu: “{html.escape(str(evidence))}” · '
                    f'{html.escape(state_desc)}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                property_st.write(f"{mode_label}: {label} — {state_desc} (từ câu: {evidence})")

    semantic_meta = parts.get("semantic_property_metadata") or {}
    if semantic_meta.get("status") == "unavailable":
        property_st.error("Chưa thể gợi ý cho các cụm chưa hiểu; không tự thêm điều kiện lọc.")
    suggestions = parts.get("semantic_property_suggestions") or []
    suggested_spans = {
        (span.get("start_offset"), span.get("end_offset"))
        for suggestion in suggestions
        if (span := suggestion.get("span"))
    }
    for clause in parts.get("unresolved_property_clauses") or []:
        if (clause.get("start_offset"), clause.get("end_offset")) not in suggested_spans:
            property_st.warning(f"Chưa hiểu yêu cầu “{clause['text']}”; chưa tạo điều kiện lọc.")
    if suggestions:
        _render_semantic_property_suggestions(property_st, result)

    _render_timing_breakdown(st, result)
    with st.expander("Chi tiết kỹ thuật"):
        st.caption("Điểm tương đồng chỉ dùng xếp gợi ý, không phải xác suất kết quả đúng. "
                   "Điểm đánh giá category hiện chỉ là chỉ báo tạm thời, chưa được kiểm chứng độc lập.")
        st.json({
            "query_parts": parts,
            "decision": decision,
            "prediction": intent.get("prediction"),
            "category_resolution": intent.get("category_resolution"),
            "confidence_assessment": intent.get("confidence_assessment"),
            "query_interpretation": intent.get("query_interpretation"),
            "property_extraction": extraction,
            "timing": result.get("timing"),
        })


def _render_semantic_property_suggestions(st, result) -> None:
    suggestions = result["query_parts"]["semantic_property_suggestions"]
    st.markdown("**Những yêu cầu chưa rõ**")
    st.caption("Các gợi ý dưới đây chưa được thêm vào điều kiện; không có địa điểm nào được lọc.")
    for suggestion in suggestions:
        span = suggestion.get("span") or {}
        st.write(f"Trong câu: “{span.get('text', '')}”")
        if suggestion.get("status") == "not_requested":
            st.caption("Cụm này không yêu cầu một thuộc tính; không tạo điều kiện lọc.")
            continue
        candidates = suggestion.get("candidates") or []
        if not candidates:
            st.caption("Chưa có gợi ý phù hợp; không tạo điều kiện lọc.")
            continue
        st.caption(f"{len(candidates)} khả năng gần nhất — chỉ để tham khảo, chưa áp dụng:")
        for index, candidate in enumerate(candidates, start=1):
            label = str(candidate.get("label") or candidate.get("property_id") or "Chưa rõ")
            st.markdown(
                f'<div class="prop-suggestion-card">{index}. '
                f'<strong>{html.escape(label)}</strong></div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Hiểu yêu cầu tìm địa điểm", page_icon="🔍", layout="wide")
    st.markdown(
        _clean_html("""
        <style>
        .block-container {padding-top: 1.5rem; max-width: 1080px;}
        .prop-item-card, .prop-suggestion-card {
            background: var(--secondary-background-color, #202637);
            color: var(--text-color, #f8fafc);
            border: 1px solid color-mix(in srgb, var(--text-color, #f8fafc) 20%, transparent);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .prop-pill {display: inline-block; border-radius: 5px; padding: 2px 8px;
            background: color-mix(in srgb, var(--text-color, #f8fafc) 12%, transparent);
            font-size: .82rem; margin-right: 8px;}
        .prop-evidence {opacity: .78; font-size: .86rem; margin-top: 5px;}
        </style>
        """),
        unsafe_allow_html=True,
    )

    _render_intent_analysis(st)


if __name__ == "__main__":
    main()
