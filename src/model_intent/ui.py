import os
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/benchmark.poi.pg.v3.yaml"
DEFAULT_TAXONOMY = PROJECT_ROOT / "data/taxonomies/place_types.poc.v3.json"


def _configured_taxonomy(config_path: Path, config: dict[str, Any] | None = None) -> Path:
    """Resolve the configured taxonomy relative to the project root, preserving the v2 default."""
    if config is None:
        from model_intent.benchmark import load_yaml

        config = load_yaml(config_path)
    selected = config.get("taxonomy_path")
    if not selected:
        return DEFAULT_TAXONOMY
    path = Path(selected)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _serving_config() -> tuple[Path, str, str, str]:
    """Validate the shared Postgres target and fingerprint cache inputs without storing the URL."""
    from model_intent.benchmark import load_yaml
    from model_intent.db import validate_database_target
    from model_intent.search_runtime import E5_POI_DIMENSION, E5_POI_MODEL_ID, E5_POI_MODEL_KEY, E5_POI_REVISION

    config_path = Path(os.getenv("MODEL_INTENT_POI_CONFIG", str(DEFAULT_CONFIG))).resolve()
    config = load_yaml(config_path)
    if (
        config.get("retrieval_backend") != "pgvector"
        or config.get("retrieval_mode") != "pg_exact"
        or (config.get("poi_search") or {}).get("backend") != "postgres"
        or config.get("require_explicit_database_url") is not True
    ):
        raise RuntimeError("UI requires exact pgvector category retrieval and Postgres POI serving with an explicit DATABASE_URL; artifact/NumPy fallback is disabled")
    profiles = config.get("models") or []
    if len(profiles) != 1 or any(
        profile.get("key") != E5_POI_MODEL_KEY
        or profile.get("model_id") != E5_POI_MODEL_ID
        or profile.get("revision") != E5_POI_REVISION
        or profile.get("expected_dimension") != E5_POI_DIMENSION
        for profile in profiles
    ):
        raise RuntimeError("UI requires the pinned E5-base 768D model profile")
    database_url = validate_database_target(config)
    return (
        config_path,
        sha256(config_path.read_bytes()).hexdigest(),
        sha256(_configured_taxonomy(config_path, config).read_bytes()).hexdigest(),
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


def result_view_model(result: dict[str, Any]) -> dict[str, Any]:
    decision = result.get("decision", {})
    prediction = result.get("prediction", {})
    action = decision.get("action", "reject")
    colors = {
        "accept": {"bg": "#dcfce7", "border": "#16a34a", "text": "#166534", "label": "ACCEPT"},
        "clarify": {"bg": "#fef3c7", "border": "#d97706", "text": "#92400e", "label": "CLARIFY"},
        "reject": {"bg": "#fee2e2", "border": "#dc2626", "text": "#991b1b", "label": "REJECT"},
    }
    return {
        "action": action, "style": colors.get(action, colors["reject"]),
        "reason": decision.get("reason", "unknown"), "question": decision.get("question"),
        "accepted_item_id": decision.get("accepted_item_id"), "accepted_code": decision.get("accepted_code"),
        "options": decision.get("options", []), "top1_score": prediction.get("top1_score"),
        "calibrated_confidence": decision.get("calibrated_confidence"),
        "in_domain_confidence": decision.get("in_domain_confidence"),
        "ambiguity_score": decision.get("ambiguity_score"),
        "margin": prediction.get("margin"), "policy_version": decision.get("policy_version"),
        "hits": prediction.get("hits", []), "query_variants": prediction.get("query_variants", []),
        "query_elapsed_ms": result.get("runtime", {}).get("query_elapsed_ms"),
    }


def diagnostic_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Rank": hit.get("rank"), "Category": " > ".join(hit.get("path") or []),
            "Code": hit.get("code"), "ID": hit.get("item_id"),
            "Cosine score": round(float(hit.get("score", 0.0)), 4),
            "Matched semantic text": hit.get("matched_text"), "Language": hit.get("language"),
            "Profile kind": hit.get("profile_kind", hit.get("document_variant_kind")),
            "Document variant": hit.get("profile_kind", hit.get("document_variant_kind")),
            "Query variant": hit.get("query_variant_kind"),
        }
        for hit in result.get("prediction", {}).get("hits", [])
    ]


def _render_poi_search(st) -> None:
    from model_intent.search_explanation import hit_steps, search_steps
    from model_intent.search_runtime import E5_POI_DIMENSION, E5_POI_MODEL_KEY, SearchRuntime

    st.title("📍 HCMC POI Search · E5-base 768D")

    @st.cache_resource(max_entries=2, show_spinner=False)
    def load_poi_runtime(config_path: str, config_hash: str, taxonomy_hash: str, database_hash: str, taxonomy_path: Path):
        return SearchRuntime.from_config(
            config_path=config_path,
            taxonomy_path=taxonomy_path,
            model_key=E5_POI_MODEL_KEY,
            local_files_only=True,
        )

    try:
        poi_config, config_hash, taxonomy_hash, database_hash = _serving_config()
        taxonomy_path = _configured_taxonomy(poi_config)
        with st.spinner("Loading pinned E5-base 768D category model and HCMC POI dataset…"):
            runtime = load_poi_runtime(str(poi_config), config_hash, taxonomy_hash, database_hash, taxonomy_path)
    except Exception as exc:
        st.error(f"E5 POI search is not ready: {_safe_error_message(exc)}")
        st.info("Use the unified Postgres config and set DATABASE_URL to its isolated database name and port. Ensure pgvector and both active generations are ready; no artifact fallback is used.")
        return

    osm_only = int(runtime.ranking_policy.get("schema_version", 1)) >= 3
    if osm_only:
        st.caption("E5-base category matching → OSM-evidenced property filters → POI ranking by preferences and optional distance. Missing OSM tags remain unknown; no ratings or reviews are imputed.")
    else:
        st.caption("E5-base semantic category matching → property filters → HCMC POI ranking (ratings/properties may be mock).")
    meta = runtime.category_runtime.metadata
    poi_count = runtime.poi_repository.count()
    identity = (
        str(poi_config), config_hash, taxonomy_hash, database_hash,
        meta.model_key, meta.model_revision, meta.backend,
        meta.generation_id, runtime.poi_repository.source, runtime.poi_repository.generation_id,
        runtime.ranking_policy.get("policy_version"),
        runtime.property_profiles.version if runtime.property_profiles else None,
        poi_count,
    )
    if st.session_state.get("poi_runtime_identity") != identity:
        st.session_state.pop("poi_result", None)
        st.session_state["poi_runtime_identity"] = identity
    taxonomy = runtime.category_runtime.taxonomy
    with st.sidebar:
        st.subheader("POI search")
        st.caption(f"Category model: {meta.model_id} · {E5_POI_DIMENSION}D")
        st.caption(f"Revision: {(meta.model_revision or '')[:12]} · {meta.backend}/{meta.retrieval_mode}")
        st.caption(f"Calibration: {runtime.category_runtime.confidence_calibration['calibration_version']}")
        st.caption(f"Category generation: {meta.generation_id}")
        st.caption(f"POI source: {runtime.poi_repository.source} · generation: {runtime.poi_repository.generation_id}")
        st.caption(f"Indexed HCMC POIs: {poi_count:,}")
        language = st.radio("Query language", ("vi", "en"), horizontal=True)
        top_k = st.slider("Results", min_value=3, max_value=20, value=10)
        verified_only = True if osm_only else st.toggle(
            "Only verified filter evidence", value=False,
            help="Require OSM evidence for hard filters. Ratings may still be synthetic when sorting.",
        )
        if osm_only:
            st.caption("OSM-only mode: hard filters and preferences require explicit OSM evidence.")
        explain = st.checkbox("Xem giải thích từng bước", value=False,
                              help="Hiện quá trình chọn category, tách yêu cầu, lọc và chấm điểm từng POI.")
        st.caption(f"Config: {poi_config.name}")

    with st.form("poi_search_form"):
        query = st.text_area(
            "What place are you looking for?",
            value=st.session_state.get("poi_query", "quán cà phê có máy lạnh, ưu tiên thoáng mát"),
            height=100,
            placeholder="Ví dụ: quán cà phê có máy lạnh, ưu tiên thoáng mát",
        )
        coordinates = st.columns(2)
        latitude = coordinates[0].text_input("Latitude (optional)", value="", placeholder="10.77")
        longitude = coordinates[1].text_input("Longitude (optional)", value="", placeholder="106.69")
        submitted = st.form_submit_button("Search POIs", type="primary", width="stretch")

    if submitted:
        try:
            if bool(latitude.strip()) != bool(longitude.strip()):
                raise ValueError("Provide both latitude and longitude, or leave both blank")
            origin = (float(latitude), float(longitude)) if latitude.strip() else None
            st.session_state["poi_query"] = query.strip()
            st.session_state["poi_result"] = runtime.search(
                query,
                poi_top_k=top_k,
                language=language,
                origin=origin,
                verified_only=verified_only,
            )
            st.session_state["poi_origin"] = origin
        except Exception as exc:
            st.session_state.pop("poi_result", None)
            st.error(f"POI search failed: {_safe_error_message(exc)}")

    result = st.session_state.get("poi_result")
    if not result:
        st.info("Describe the place and its requirements in one query; category and properties will be detected automatically.")
        return
    if result.get("poi_search", {}).get("verified_only", False) != verified_only:
        st.info("Search again to apply the selected verification mode.")
        return
    if explain:
        st.subheader("Vì sao ra kết quả này?")
        for step in search_steps(result):
            st.markdown(f"**{step['title']}**")
            for detail in step["details"]:
                st.write(f"• {detail}")

    decision = result.get("intent", {}).get("decision", {})
    if decision.get("action") == "clarify":
        st.warning(decision.get("question") or "Which type of place do you mean?")
        options = decision.get("options") or []
        allowed_ids = [str(option["item_id"]) for option in options]
        for option in options:
            if st.button(option.get("label") or option["code"], key=f"poi-clarify-{option['item_id']}"):
                try:
                    st.session_state["poi_result"] = runtime.resolve_selection_and_search(
                        query=st.session_state["poi_query"],
                        selected_id=str(option["item_id"]),
                        allowed_option_ids=allowed_ids,
                        poi_top_k=top_k,
                        language=language,
                        origin=st.session_state.get("poi_origin"),
                        verified_only=verified_only,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Category selection failed: {_safe_error_message(exc)}")
        return
    if decision.get("action") != "accept":
        status = result.get("poi_search", {}).get("status")
        if status == "unsupported_category":
            st.warning("Query này chưa có category chung trong taxonomy/dữ liệu POI. E5 chỉ thấy các category gắn thương hiệu; hệ thống không tự đổi ‘trạm sạc’ thành ‘VinFast’. Cần bổ sung category và OSM mapping chung trước khi có thể trả POI.")
        else:
            st.warning("Could not confidently identify a place category. Try naming the type of place in your query.")
        return

    selected = str(decision["accepted_item_id"])
    category_label = " > ".join(taxonomy.path_names(selected, language))
    search = result.get("poi_search", {})
    extraction = result.get("property_extraction", {})
    st.markdown(f"**Detected category:** {category_label} · ID `{selected}`")
    if not osm_only:
        st.warning("POC: thuộc tính và rating có nhãn mock là dữ liệu giả lập, không phải thông tin đã xác minh.")
    for warning in search.get("warnings", []):
        st.info(warning)

    constraints = extraction.get("properties", [])
    if constraints:
        st.markdown("**Interpreted requirements**")
        for row in constraints:
            value = row.get("value")
            if isinstance(value, bool):
                value = "yes" if value else "no"
            mode = str(row.get("mode", "required")).replace("_", " ").title()
            evidence = row.get("evidence_text") or ""
            state = "applied" if row.get("applied") else f"ignored: {row.get('ignored_reason') or 'not applied'}"
            st.markdown(f"- **{mode}** · {row.get('property_id')}: **{value}** · {state} · _{evidence}_")
    else:
        st.caption("No structured property constraints detected; POIs use the category and, if supplied, distance." if osm_only
                   else "No structured property constraints detected; ranking uses rating and review count.")

    counts = st.columns(3)
    counts[0].write(f"Category candidates: **{search.get('category_candidate_count', 0):,}**")
    counts[1].write(f"Pass hard filters: **{search.get('hard_filter_match_count', 0):,}**")
    counts[2].write(f"Returned: **{search.get('returned_count', 0):,}**")
    if search.get("verified_required_count") is not None:
        if osm_only:
            st.caption(f"Filter evidence: {search['verified_required_count']:,} verified · "
                       f"{search['mock_supported_count']:,} lacking verified evidence.")
        else:
            st.caption(f"Filter evidence: {search['verified_required_count']:,} verified · "
                       f"{search['mock_supported_count']:,} lacking verified evidence (mock or unknown). "
                       "Turn on verified-only to require OSM evidence.")

    if search.get("status") == "no_verified_rating_data":
        st.warning("No verified ratings are available for this category in the current dataset.")
        return
    if search.get("status") == "no_pois_for_category":
        st.warning("Category đã được nhận diện nhưng dataset HCM hiện chưa có POI được map vào category này.")
        return
    if search.get("status") == "no_matches":
        st.warning("No POIs match the active filters in this dataset. This does not prove the venue lacks the requested features.")
        return
    hits = search.get("hits", [])
    if not hits:
        st.info(f"POI search status: {search.get('status', 'no results')}")
        return

    st.markdown("### Ranked places")
    for index, hit in enumerate(hits, 1):
        poi = hit
        rating = poi.get("rating", {})
        score = poi.get("score", {})
        effective = (poi.get("properties") or {}).get("effective", {})
        with st.container(border=True):
            label = poi.get("name") or poi.get("display_name") or poi.get("poi_id")
            if poi.get("name") is None:
                label += " (generated label; OSM name unavailable)"
            st.markdown(f"**{index}. {label}**")
            st.caption(f"{poi.get('poi_id')} · score {score.get('final_score', 0):.3f}")
            if not osm_only:
                rating_source = "mock — synthetic POC data" if rating.get("source") == "mock" else rating.get("source", "unknown")
                st.write(f"⭐ {rating.get('value', 'n/a')} · {rating.get('review_count', 0):,} reviews · {rating_source}")
            applicable = (
                runtime.property_profiles.applicable_ids(taxonomy, str(poi.get("category_id") or selected))
                if runtime.property_profiles is not None
                else tuple(sorted(effective))
            )
            with st.expander(f"Category properties ({len(applicable)})", expanded=index == 1):
                for key in applicable:
                    assertion = effective.get(key)
                    definition = runtime.property_catalog.get(key, {})
                    aliases = (definition.get("aliases") or {}).get(language) or []
                    label = aliases[0] if aliases else key.replace("_", " ")
                    if not isinstance(assertion, dict) or assertion.get("status") != "known":
                        status = "inapplicable" if isinstance(assertion, dict) and assertion.get("status") == "inapplicable" else "unknown"
                        st.write(f"{label}: {status} (no confirmed OSM value)" if osm_only else f"{label}: {status}")
                        continue
                    value = assertion.get("value")
                    if isinstance(value, bool):
                        value = "yes" if value else "no"
                    source = "mock · chưa xác minh" if assertion.get("source") == "mock" else "OSM" if assertion.get("source") == "osm" else "unknown"
                    st.write(f"{label}: {value} ({source})")
            required_sources = poi.get("required_property_sources") or {}
            if required_sources:
                st.caption("Required matches: " + ", ".join(f"{key}={source}" for key, source in required_sources.items()))
            if osm_only:
                st.caption(f"OSM preferred-property component {score.get('preferred_property_score', 0):.2f} · "
                           f"distance component {score.get('distance_score', 0):.2f} "
                           "(included only with an origin). Ratings and reviews are not ranked.")
            else:
                st.caption(
                    f"Rating component {score.get('rating_score', 0):.2f} "
                    f"(source factor {score.get('rating_source_factor', 1):.2f}) · "
                    f"preferred properties {score.get('preferred_property_score', 0):.2f} · "
                    f"review volume {score.get('review_volume_score', 0):.2f}"
                )
            if explain:
                with st.expander(f"Vì sao POI này đứng thứ {index}?", expanded=index == 1):
                    for detail in hit_steps(poi, extraction, runtime.ranking_policy, verified_only=verified_only):
                        st.write(f"• {detail}")

    with st.expander("Search response JSON"):
        st.json({"property_extraction": extraction, "poi_search": search})


def main() -> None:
    import streamlit as st
    from model_intent.runtime import MatcherRuntime

    st.set_page_config(page_title="HCMC POI Search · E5-base 768D", page_icon="📍", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; max-width: 1280px;}
        .intent-banner {border-left: 6px solid; border-radius: 10px; padding: 1rem 1.2rem; margin: .5rem 0 1rem 0;}
        .option-card {border: 1px solid #dbe2ea; border-radius: 12px; padding: .8rem; min-height: 145px; background: rgba(255,255,255,.03);}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        demo_mode = st.radio(
            "Demo",
            ("HCMC POI Search · E5-base", "Category diagnostics · pgvector"),
            index=0,
        )
    if demo_mode == "HCMC POI Search · E5-base":
        _render_poi_search(st)
        return

    @st.cache_resource(max_entries=2, show_spinner=False)
    def load_runtime(config_path: str, config_hash: str, taxonomy_hash: str,
                     database_hash: str, offline: bool, taxonomy_path: Path) -> MatcherRuntime:
        return MatcherRuntime.from_config(
            config_path=config_path,
            taxonomy_path=taxonomy_path,
            model_key="e5-base-multilingual",
            local_files_only=offline,
        )

    st.title("🧭 Category Intent Search · pgvector")
    st.caption("Hierarchy-aware category profiles, PostgreSQL vector storage, and exact cosine retrieval.")

    with st.sidebar:
        st.header("Product runtime")
        st.caption("Embedding model: intfloat/multilingual-e5-base · 768D (pinned)")
        language = st.radio("Decision language", ("vi", "en"), horizontal=True)
        top_k = st.slider("Top-k diagnostics", 3, 15, 5)
        offline = st.toggle("Use local model files only", value=True)
        if st.button("Clear result", width="stretch"):
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_query", None)
            st.rerun()

    try:
        config_path, config_hash, taxonomy_hash, database_hash = _serving_config()
        taxonomy_path = _configured_taxonomy(config_path)
        with st.spinner("Loading pinned E5-base 768D model and active pgvector generation…"):
            runtime = load_runtime(str(config_path), config_hash, taxonomy_hash, database_hash, offline, taxonomy_path)
    except Exception as exc:
        st.error(f"Product runtime unavailable: {_safe_error_message(exc)}")
        st.info("Use the unified Postgres config and set DATABASE_URL to its isolated database name and port. Ensure pgvector and the active E5 generation are ready; no artifact fallback is used.")
        st.stop()

    meta = runtime.metadata
    identity = (str(config_path), config_hash, taxonomy_hash, database_hash,
                offline, meta.model_key, meta.model_revision, meta.generation_id)
    if st.session_state.get("category_runtime_identity") != identity:
        st.session_state.pop("last_result", None)
        st.session_state["category_runtime_identity"] = identity

    with st.sidebar:
        st.success("Runtime ready")
        st.metric("Stored profiles", f"{meta.profile_count or meta.candidate_count:,}")
        st.caption(f"Indexed categories: {meta.indexed_category_count or 'n/a'}")
        st.caption(f"Backend: {meta.backend} / {meta.retrieval_mode}")
        st.caption(f"Generation: {meta.generation_id}")
        st.caption(f"Revision: {(meta.model_revision or '')[:12]}…")
        st.caption(f"Taxonomy: {meta.taxonomy_items} records")

    examples = [
        "bị sốt và muốn tìm nơi bán thuốc",
        "đau đầu quá, cần chỗ chăm sóc sức khỏe",
        "chỗ ngồi làm việc có wifi",
        "mai trời có mưa không",
    ]
    example = st.selectbox("Quick examples", [""] + examples, format_func=lambda value: "Choose an example…" if not value else value)
    with st.form("query_form"):
        query = st.text_area(
            "Search intent", value=example or st.session_state.get("last_query", ""),
            height=95, placeholder="Nhập nhu cầu tìm địa điểm…",
        )
        submitted = st.form_submit_button("Run pgvector semantic search", type="primary", width="stretch")
    if submitted:
        try:
            st.session_state["last_query"] = query.strip()
            st.session_state["last_result"] = runtime.query(query, top_k=top_k, language=language)
        except Exception as exc:
            st.error(f"Query failed: {_safe_error_message(exc)}")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Submit a query to test the active pgvector embedding generation.")
        return

    view = result_view_model(result)
    style = view["style"]
    st.markdown(
        f"<div class='intent-banner' style='background:{style['bg']};border-color:{style['border']};color:{style['text']}'>"
        f"<strong>{style['label']}</strong> · {view['reason']}</div>", unsafe_allow_html=True,
    )
    metrics = st.columns(8)
    metrics[0].metric("Cosine similarity", f"{view['top1_score']:.4f}" if view["top1_score"] is not None else "n/a")
    metrics[1].metric("Top confidence", f"{100 * view['calibrated_confidence']:.1f}%" if view["calibrated_confidence"] is not None else "n/a")
    metrics[2].metric("In-domain", f"{100 * view['in_domain_confidence']:.1f}%" if view["in_domain_confidence"] is not None else "n/a")
    metrics[3].metric("Ambiguity", f"{100 * view['ambiguity_score']:.1f}%" if view["ambiguity_score"] is not None else "n/a")
    info = result.get("runtime", {})
    metrics[4].metric("Encode", f"{info.get('query_encode_ms', 0):.1f} ms")
    metrics[5].metric("DB search", f"{info.get('db_search_ms', 0):.1f} ms")
    metrics[6].metric("Decision", f"{info.get('decision_ms', 0):.1f} ms")
    metrics[7].metric("Total", f"{info.get('query_elapsed_ms', 0):.1f} ms")
    st.caption("Scores are raw cosine similarities from E5-base 768D; calibrated decision confidence is shown separately.")

    if view["action"] == "accept":
        item_id = str(view["accepted_item_id"])
        item = runtime.taxonomy.items.get(item_id)
        label = item.names.get(language) if item else view["accepted_code"]
        path = " > ".join(runtime.taxonomy.path_names(item_id, language)) if item else ""
        st.success(f"**{label}** · `{view['accepted_code']}` · ID `{item_id}`")
        if path: st.caption(path)
    elif view["action"] == "clarify":
        st.warning(view["question"] or "Please choose the intended place type.")
        options = view["options"]
        allowed_ids = [str(option["item_id"]) for option in options]
        columns = st.columns(min(4, len(options)))
        for index, option in enumerate(options):
            with columns[index % len(columns)]:
                st.markdown(
                    f"<div class='option-card'><strong>{option['label']}</strong><br><code>{option['code']}</code> · ID {option['item_id']}<br><br>"
                    f"{option.get('hint') or ''}<br><small>score {option['score']:.4f}</small></div>", unsafe_allow_html=True,
                )
                if st.button(f"Choose {option['label']}", key=f"choose-{option['item_id']}", width="stretch"):
                    selected = runtime.resolve_selection(
                        query=st.session_state.get("last_query", ""), selected_id=str(option["item_id"]),
                        allowed_option_ids=allowed_ids, language=language,
                    )
                    result["decision"] = selected["decision"]
                    st.session_state["last_result"] = result
                    st.rerun()
    else:
        st.error("No reliable place type was selected from the active semantic index.")

    with st.expander("pgvector request metadata"):
        st.json({
            "backend": info.get("backend"), "mode": info.get("mode"),
            "generation_id": info.get("generation_id"), "pool_wait_ms": info.get("pool_wait_ms"),
            "db_search_ms": info.get("db_search_ms"), "returned_count": info.get("returned_count"),
            "ef_search": info.get("ef_search"), "overfetch": info.get("overfetch"),
        })
    with st.expander("Ranked dense-embedding diagnostics", expanded=True):
        st.dataframe(diagnostic_rows(result), width="stretch", hide_index=True)
        if view["query_variants"]:
            st.caption("Embedded query variants: " + " · ".join(f"`{value}`" for value in view["query_variants"]))


if __name__ == "__main__":
    main()
