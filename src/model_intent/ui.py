import os
from pathlib import Path
from typing import Any

if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://model_intent:model_intent_dev@localhost:5432/model_intent"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/benchmark.cpu.yaml"
DEFAULT_TAXONOMY = PROJECT_ROOT / "data/taxonomies/place_types.poc.v1.json"


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


def main() -> None:
    import streamlit as st
    from model_intent.runtime import MatcherRuntime

    st.set_page_config(page_title="Category Search · pgvector", page_icon="🧭", layout="wide")
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

    @st.cache_resource(max_entries=2, show_spinner=False)
    def load_runtime(offline: bool, model_key: str = "bge-m3") -> MatcherRuntime:
        return MatcherRuntime.from_config(
            config_path=DEFAULT_CONFIG,
            taxonomy_path=DEFAULT_TAXONOMY,
            model_key=model_key,
            local_files_only=offline,
        )

    st.title("🧭 Category Intent Search · pgvector")
    st.caption("Hierarchy-aware category profiles, PostgreSQL vector storage, and exact cosine retrieval.")

    with st.sidebar:
        st.header("Product runtime")
        model_choice = st.selectbox(
            "Embedding model",
            ["intfloat/multilingual-e5-base (768D · Đã cập nhật)", "BAAI/bge-m3 (1024D)"],
            index=0,
        )
        selected_model_key = "e5-base-multilingual" if "e5-base" in model_choice else "bge-m3"
        language = st.radio("Decision language", ("vi", "en"), horizontal=True)
        top_k = st.slider("Top-k diagnostics", 3, 15, 5)
        offline = st.toggle("Use local model files only", value=True)
        if st.button("Clear result", width="stretch"):
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_query", None)
            st.rerun()

    try:
        with st.spinner(f"Loading pinned {selected_model_key} model and active pgvector generation…"):
            runtime = load_runtime(offline, selected_model_key)
    except Exception as exc:
        st.error(f"Product runtime unavailable: {exc}")
        st.info("Ensure Docker pgvector is healthy, DATABASE_URL is set, migrations are applied, and embeddings sync has an active ready generation.")
        st.stop()

    with st.sidebar:
        meta = runtime.metadata
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
            st.error(f"Query failed: {exc}")

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
