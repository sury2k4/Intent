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


def test_poi_config_pins_real_e5_and_keeps_artifact_backend():
    config = yaml.safe_load((ROOT / "configs/benchmark.poi.v2.yaml").read_text(encoding="utf-8"))
    profile = next(item for item in config["models"] if item["key"] == E5_POI_MODEL_KEY)
    assert config["retrieval_backend"] == "numpy"
    assert config["poi_search"]["backend"] == "artifact"
    assert profile["model_id"] == E5_POI_MODEL_ID
    assert profile["revision"] == E5_POI_REVISION
    assert profile["expected_dimension"] == E5_POI_DIMENSION
    assert profile["query_prefix"] == "query: "
    assert profile["document_prefix"] == "passage: "
    assert config["confidence_calibration_path"]


def test_unified_db_config_uses_one_e5_profile_and_postgres_poi_backend():
    config = yaml.safe_load((ROOT / "configs/benchmark.poi.pg.yaml").read_text(encoding="utf-8"))
    assert config["retrieval_backend"] == "pgvector"
    assert config["retrieval_mode"] == "pg_exact"
    assert config["poi_search"]["backend"] == "postgres"
    assert {profile["key"] for profile in config["models"]} == {E5_POI_MODEL_KEY}
    assert config["models"][0]["expected_dimension"] == 768


def test_poi_runtime_rejects_fake_model_even_if_loaded_under_e5_key(monkeypatch):
    import model_intent.search_runtime as module

    class FakeCategory:
        metadata = SimpleNamespace(
            model_id="fake", model_revision=E5_POI_REVISION,
            backend="numpy", retrieval_mode="numpy_exact",
        )
        embedder = SimpleNamespace(metadata=lambda: {"dimension": 768})
        index = SimpleNamespace(vectors=SimpleNamespace(shape=(444, 768)))
        confidence_calibration = {"calibration_version": "e5-base-confidence-v1"}

    monkeypatch.setattr(module.MatcherRuntime, "from_config", lambda **kwargs: FakeCategory())
    with pytest.raises(RuntimeError, match="pinned E5-base 768D"):
        module.SearchRuntime.from_config(
            config_path=ROOT / "configs/benchmark.poi.v2.yaml",
            taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v2.json",
            model_key=E5_POI_MODEL_KEY,
            local_files_only=True,
        )


@pytest.mark.real_model
def test_pinned_e5_category_to_artifact_poi_flow():
    required = os.getenv("MODEL_INTENT_REQUIRE_REAL_MODEL") == "1"
    if importlib.util.find_spec("sentence_transformers") is None:
        if required:
            pytest.fail("E5 integration requires sentence-transformers")
        pytest.skip("optional model packages are not installed")
    artifact = ROOT / "artifacts/osm/hcmc-poc-v4/pois.jsonl"
    if not artifact.exists():
        if required:
            pytest.fail("build the pinned HCMC POI artifact before E5 integration")
        pytest.skip("large HCMC artifact is not part of a fresh checkout")
    runtime = SearchRuntime.from_config(
        config_path=ROOT / "configs/benchmark.poi.v2.yaml",
        taxonomy_path=ROOT / "data/taxonomies/place_types.poc.v2.json",
        model_key=E5_POI_MODEL_KEY,
        local_files_only=True,
    )
    try:
        metadata = runtime.category_runtime.metadata
        assert metadata.model_key == E5_POI_MODEL_KEY
        assert metadata.model_id == E5_POI_MODEL_ID
        assert metadata.model_revision == E5_POI_REVISION
        assert metadata.backend == "numpy"
        assert metadata.retrieval_mode == "numpy_exact"
        assert runtime.category_runtime.index.vectors.shape == (452, 768)
        assert runtime.category_runtime.confidence_calibration["calibration_version"] == "e5-base-confidence-v2"
        accepted = runtime.search("quán cà phê có máy lạnh, ưu tiên thoáng mát", poi_top_k=3)
        assert accepted["intent"]["decision"]["action"] == "accept"
        assert accepted["intent"]["decision"]["accepted_item_id"] == "23"
        assert accepted["intent"]["decision"]["reason"] == "calibrated_high_confidence"
        assert accepted["intent"]["category_resolution"]["source"] == "semantic_model"
        assert accepted["poi_search"]["returned_count"] == 3
        assert all(hit["category_id"] == "23" for hit in accepted["poi_search"]["hits"])
        charging = runtime.search("trạm sạc nhanh ,có nhà vệ sinh", poi_top_k=3)
        assert charging["intent"]["query_interpretation"]["category_text"] == "trạm sạc nhanh"
        assert [row["role"] for row in charging["intent"]["query_interpretation"]["property_roles"]] == [
            "dual_role_category_modifier", "constraint",
        ]
        assert charging["intent"]["category_resolution"]["semantic_variants"][0]["accepted_item_id"] == "242"
        assert charging["intent"]["decision"]["accepted_item_id"] == "242"
        assert charging["poi_search"]["status"] == "ok"
        assert charging["poi_search"]["category_candidate_count"] == 23
        assert {hit["category_id"] for hit in charging["poi_search"]["hits"]} <= {"242", "19"}
        bank = runtime.search("phòng giao dịch ngân hàng", poi_top_k=3)
        if bank["intent"]["decision"]["action"] == "accept":
            assert bank["intent"]["decision"]["accepted_item_id"] == "243"
            assert bank["poi_search"]["category_candidate_count"] > 0
        else:
            assert bank["poi_search"]["status"].startswith("skipped_category_")
        parking = runtime.search("bãi đỗ xe", poi_top_k=3)
        assert parking["intent"]["decision"]["accepted_item_id"] == "86"
        assert parking["property_extraction"]["properties"][0]["ignored_reason"] == "category_component"
        assert parking["poi_search"]["returned_count"] == 3
        rejected = runtime.search("mai trời có mưa không")
        assert rejected["intent"]["decision"]["action"] == "reject"
        assert rejected["poi_search"]["status"] == "skipped_category_reject"
    finally:
        runtime.close()


VALID_DATABASE_URL = "postgresql://test_user:secret-one@127.0.0.1:5433/model_intent_unified"


def test_ui_defaults_to_unified_postgres_and_pinned_e5(monkeypatch):
    monkeypatch.delenv("MODEL_INTENT_POI_CONFIG", raising=False)
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    config_path, config_hash, taxonomy_hash, database_hash = ui._serving_config()
    assert config_path == ROOT / "configs/benchmark.poi.pg.v3.yaml"
    assert ui.DEFAULT_TAXONOMY == ROOT / "data/taxonomies/place_types.poc.v3.json"
    assert all(len(value) == 64 for value in (config_hash, taxonomy_hash, database_hash))
    assert "secret-one" not in str((config_path, config_hash, taxonomy_hash, database_hash))


def test_ui_redacts_database_credentials_from_runtime_errors(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    message = ui._safe_error_message(RuntimeError(f"Failed for {VALID_DATABASE_URL}: secret-one"))
    assert "secret-one" not in message
    assert VALID_DATABASE_URL not in message
    assert "DATABASE_URL redacted" in message


@pytest.mark.parametrize("database_url", [None, "postgresql://user:wrong@localhost:5432/model_intent",
                                             "postgresql://user:wrong@localhost:5433/other_db"])
def test_ui_rejects_missing_or_wrong_database_url(monkeypatch, database_url):
    monkeypatch.delenv("MODEL_INTENT_POI_CONFIG", raising=False)
    if database_url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", database_url)
    with pytest.raises(RuntimeError, match="DATABASE_URL") as error:
        ui._serving_config()
    assert "wrong" not in str(error.value)


@pytest.mark.parametrize("field,value", [
    ("retrieval_backend", "numpy"), ("retrieval_mode", "pg_hnsw"),
    ("poi_search.backend", "artifact"), ("require_explicit_database_url", False),
    ("models.0.model_id", "BAAI/bge-m3"), ("models.0.expected_dimension", 1024),
])
def test_ui_refuses_artifact_or_unpinned_config(monkeypatch, tmp_path, field, value):
    config = yaml.safe_load((ROOT / "configs/benchmark.poi.pg.yaml").read_text(encoding="utf-8"))
    target = config
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdecimal() else target[part]
    target[parts[-1]] = value
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("MODEL_INTENT_POI_CONFIG", str(path))
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    with pytest.raises(RuntimeError, match="pgvector|pinned E5-base"):
        ui._serving_config()


def test_configured_taxonomy_path_drives_ui_cache_fingerprint(monkeypatch, tmp_path):
    config_path = tmp_path / "configs/benchmark.poi.pg.yaml"
    config_path.parent.mkdir(parents=True)
    config = yaml.safe_load((ROOT / "configs/benchmark.poi.pg.yaml").read_text(encoding="utf-8"))
    config["taxonomy_path"] = "data/taxonomies/place_types.poc.v3.json"
    taxonomy_path = tmp_path / config["taxonomy_path"]
    taxonomy_path.parent.mkdir(parents=True)
    taxonomy_path.write_text("one", encoding="utf-8")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("MODEL_INTENT_POI_CONFIG", str(config_path))
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    assert ui._configured_taxonomy(config_path) == taxonomy_path
    first = ui._serving_config()
    taxonomy_path.write_text("two", encoding="utf-8")
    second = ui._serving_config()
    assert first[0] == second[0] == config_path
    assert first[1] == second[1]
    assert first[2] != second[2]


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

    def columns(self, count):
        return [self] * count

    def selectbox(self, label, options, **kwargs):
        assert label != "Embedding model"  # BGE must not be selectable for serving.
        return options[0]


@pytest.mark.parametrize("mode", ["HCMC POI Search · E5-base", "Category diagnostics · pgvector"])
def test_ui_caches_by_database_and_config_without_leaking_credentials(monkeypatch, tmp_path, mode):
    import model_intent.runtime as category_module
    import model_intent.search_runtime as poi_module

    config_path = tmp_path / "benchmark.poi.pg.yaml"
    config_path.write_bytes((ROOT / "configs/benchmark.poi.pg.yaml").read_bytes())
    monkeypatch.setenv("MODEL_INTENT_POI_CONFIG", str(config_path))
    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL)
    st = _FakeStreamlit(mode)
    monkeypatch.setitem(sys.modules, "streamlit", st)
    category_calls, poi_calls = [], []
    metadata = SimpleNamespace(
        model_key=E5_POI_MODEL_KEY, model_id=E5_POI_MODEL_ID, model_revision=E5_POI_REVISION,
        backend="pgvector", retrieval_mode="pg_exact", generation_id=4,
        profile_count=452, candidate_count=452, indexed_category_count=243, taxonomy_items=243,
    )
    category_runtime = SimpleNamespace(
        metadata=metadata, confidence_calibration={"calibration_version": "v2"}, taxonomy=SimpleNamespace(),
    )
    poi_runtime = SimpleNamespace(
        category_runtime=category_runtime,
        poi_repository=SimpleNamespace(count=lambda: 10, source="postgres", generation_id=5),
        ranking_policy={"policy_version": "v2"}, property_profiles=None,
    )

    def load_category(**kwargs):
        category_calls.append(kwargs)
        return category_runtime

    def load_poi(**kwargs):
        poi_calls.append(kwargs)
        return poi_runtime

    monkeypatch.setattr(category_module.MatcherRuntime, "from_config", load_category)
    monkeypatch.setattr(poi_module.SearchRuntime, "from_config", load_poi)
    active_calls = poi_calls if mode.startswith("HCMC") else category_calls
    result_key = "poi_result" if mode.startswith("HCMC") else "last_result"

    ui.main()
    st.session_state[result_key] = []
    ui.main()
    assert len(active_calls) == 1
    assert result_key in st.session_state and st.session_state[result_key] == []

    monkeypatch.setenv("DATABASE_URL", VALID_DATABASE_URL.replace("secret-one", "secret-two"))
    ui.main()
    assert len(active_calls) == 2
    assert result_key not in st.session_state
    st.session_state[result_key] = []

    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    ui.main()
    assert len(active_calls) == 3
    assert result_key not in st.session_state
    assert all(call["config_path"] == str(config_path) for call in active_calls)
    assert all(call["taxonomy_path"] == ui._configured_taxonomy(config_path) for call in active_calls)
    assert all(call["model_key"] == E5_POI_MODEL_KEY for call in active_calls)
    assert "secret-one" not in str(st.cache_keys)
    assert "secret-two" not in str(st.cache_keys)
    assert not st.errors


@pytest.mark.parametrize("mode", ["HCMC POI Search · E5-base", "Category diagnostics · pgvector"])
def test_ui_fails_before_loading_runtime_if_database_url_is_missing(monkeypatch, mode):
    monkeypatch.delenv("MODEL_INTENT_POI_CONFIG", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    st = _FakeStreamlit(mode)
    monkeypatch.setitem(sys.modules, "streamlit", st)
    if mode.startswith("HCMC"):
        ui.main()
    else:
        with pytest.raises(RuntimeError, match="streamlit stopped"):
            ui.main()
    assert any("DATABASE_URL" in error for error in st.errors)
    assert not st.cache_keys
