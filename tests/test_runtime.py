from pathlib import Path

import pytest
import yaml

from model_intent.embedding_sync import (
    profile_content_hash, profile_source_fingerprint, semantic_model_metadata, stable_hash,
)
from model_intent.rendering import (
    PROFILE_RENDERER_VERSION, PROFILE_SCHEMA_VERSION, load_enrichment, render_category_profiles,
)
from model_intent.runtime import MatcherRuntime
from model_intent.schemas import ModelProfile, Prediction, RetrievalHit
from model_intent.search_timing import SearchTiming
from model_intent.serving_config import make_embedder
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/benchmark.poi.pg.v3.yaml"
TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
ENRICHMENT = ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json"
E5_KEY = "e5-base-multilingual"
E5_ID = "intfloat/multilingual-e5-base"


def write_config(tmp_path, **changes):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for key in ("enrichment_path", "policy_path", "confidence_calibration_path"):
        config[key] = str(ROOT / config[key])
    config.update(changes)
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def generation():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    taxonomy = Taxonomy.load(TAXONOMY)
    enrichment = load_enrichment(ENRICHMENT)
    profile = MatcherRuntime._resolve_profile(config, E5_KEY)
    profiles = render_category_profiles(taxonomy, tuple(config["languages"]), enrichment, require_rich_fields=True)
    return {
        "generation_id": 17,
        "candidate_count": len(profiles),
        "model_id": profile.model_id,
        "revision": profile.revision,
        "config_hash": stable_hash(semantic_model_metadata(profile)),
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "renderer_version": PROFILE_RENDERER_VERSION,
        "profile_set_id": 9,
        "content_set_hash": stable_hash(sorted(
            (item.item_id, item.language, profile_content_hash(item)) for item in profiles
        )),
        "dimension": 768,
        "source_fingerprint": profile_source_fingerprint(enrichment),
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "indexed_category_count": len({item.item_id for item in profiles}),
        "profile_count_from_set": len(profiles),
        "artifact_kind": "category_profile_v3",
    }


@pytest.fixture
def pg_mocks(monkeypatch, generation):
    state = {"active": dict(generation), "databases": [], "indexes": [], "hits": [("34", 0.2), ("144", 0.19)]}
    revision = generation["revision"]

    class StubEmbedder:
        def metadata(self):
            return {"model_id": E5_ID, "revision": revision, "dimension": 768}

        def encode_documents(self, _):
            pytest.fail("serving must never encode candidate documents")

    class StubDatabase:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.opened = False
            self.closed = False
            state["databases"].append(self)

        def open(self):
            self.opened = True

        def close(self):
            self.closed = True

    class StubPgVectorIndex:
        def __init__(self, taxonomy, database, *, model_key, mode, generation_id=None, dimension=768):
            self.taxonomy = taxonomy
            self.database = database
            self.model_key = model_key
            self.mode = mode
            self.generation_id = generation_id
            self.dimension = dimension
            self.last_search_metadata = {}
            self.search_calls = []
            self.probe_calls = []
            state["indexes"].append(self)

        def active_generation(self):
            self.probe_calls.append("active")
            return dict(state["active"])

        def generation_metadata(self, generation_id):
            self.probe_calls.append(("generation", generation_id))
            return dict(state["active"])

        def search(self, text, embedder, *, top_k, timing):
            assert self.generation_id == generation["generation_id"]
            assert self.database.opened and not self.database.closed
            self.search_calls.append((text, top_k, timing))
            hits = tuple(
                RetrievalHit(
                    item_id=item_id, code=self.taxonomy.items[item_id].code, score=score,
                    rank=rank, language="vi", matched_text=f"profile {item_id}",
                )
                for rank, (item_id, score) in enumerate(state["hits"][:top_k], 1)
            )
            self.last_search_metadata = {"mode": "pg_exact", "generation_id": self.generation_id,
                                         "db_search_ms": 1.0, "dimension": self.dimension}
            return Prediction("query", text, hits, hits[0].score if hits else None,
                              hits[0].score - hits[1].score if len(hits) > 1 else None,
                              self.model_key, (text,), (hits[0].item_id,) if hits else ())

    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost@127.0.0.1:5433/model_intent_unified")
    monkeypatch.setattr("model_intent.runtime.configure_runtime", lambda *_: None)
    monkeypatch.setattr("model_intent.runtime.make_embedder", lambda *_: StubEmbedder())
    monkeypatch.setattr("model_intent.runtime.Database", StubDatabase)
    monkeypatch.setattr("model_intent.runtime.PgVectorIndex", StubPgVectorIndex)
    return state


def test_runtime_pins_active_e5_generation_and_uses_existing_profiles(tmp_path, pg_mocks, generation):
    runtime = MatcherRuntime.from_config(
        config_path=write_config(tmp_path), taxonomy_path=TAXONOMY,
        model_key=E5_KEY, local_files_only=True,
    )
    assert pg_mocks["indexes"][0].probe_calls == ["active"]
    assert pg_mocks["indexes"][1].generation_id == generation["generation_id"]
    assert pg_mocks["indexes"][1].mode == "pg_exact"
    assert runtime.database is pg_mocks["databases"][0]
    assert runtime.metadata.backend == "pgvector"
    assert runtime.metadata.retrieval_mode == "pg_exact"
    assert runtime.metadata.candidate_count == generation["candidate_count"]
    assert runtime.metadata.profile_count == generation["profile_count_from_set"]
    assert runtime.metadata.indexed_category_count == generation["indexed_category_count"]
    assert runtime.metadata.model_revision == generation["revision"]
    assert runtime.metadata.profile_set_id == generation["profile_set_id"]
    assert runtime.metadata.candidate_encode_seconds is None
    assert runtime.metadata.cache_path == "postgresql://category_search_profiles_v3"
    assert runtime.confidence_calibration is not None


def test_runtime_pins_requested_ready_generation(tmp_path, pg_mocks, generation):
    runtime = MatcherRuntime.from_config(
        config_path=write_config(tmp_path), taxonomy_path=TAXONOMY,
        model_key=E5_KEY, generation_id=generation["generation_id"],
    )
    assert pg_mocks["indexes"][0].probe_calls == [("generation", generation["generation_id"])]
    assert runtime.metadata.generation_id == generation["generation_id"]


@pytest.mark.parametrize("field,value", [
    ("taxonomy_fingerprint", "stale"), ("config_hash", "stale"),
    ("model_id", "other"), ("revision", "old"),
    ("renderer_version", "old"), ("artifact_kind", "category_profile_bge_m3_v1"),
    ("dimension", 1024), ("source_fingerprint", "stale"),
    ("profile_schema_version", 2), ("content_set_hash", "stale"),
    ("candidate_count", 1), ("profile_count_from_set", 1),
    ("indexed_category_count", 1),
])
def test_runtime_rejects_incompatible_generation_and_closes_db(tmp_path, pg_mocks, field, value):
    pg_mocks["active"][field] = value
    with pytest.raises(RuntimeError, match="active pgvector generation is incompatible"):
        MatcherRuntime.from_config(
            config_path=write_config(tmp_path), taxonomy_path=TAXONOMY, model_key=E5_KEY,
        )
    assert pg_mocks["databases"][0].closed
    assert len(pg_mocks["indexes"]) == 1


def test_runtime_rejects_invalid_pinned_calibration_before_opening_db(tmp_path, pg_mocks, monkeypatch):
    def invalid_calibration(*args, **kwargs):
        raise ValueError("stale model revision")

    monkeypatch.setattr("model_intent.runtime.load_confidence_calibration", invalid_calibration)
    with pytest.raises(RuntimeError, match="invalid pinned category confidence calibration"):
        MatcherRuntime.from_config(
            config_path=write_config(tmp_path), taxonomy_path=TAXONOMY, model_key=E5_KEY,
        )
    assert pg_mocks["databases"] == []


def test_runtime_rejects_wrong_embedder_dimension_before_opening_db(tmp_path, pg_mocks, monkeypatch):
    class WrongDimensionEmbedder:
        def metadata(self):
            return {"model_id": E5_ID, "revision": pg_mocks["active"]["revision"], "dimension": 384}

    monkeypatch.setattr("model_intent.runtime.make_embedder", lambda *_: WrongDimensionEmbedder())
    with pytest.raises(ValueError, match="768D dimension"):
        MatcherRuntime.from_config(
            config_path=write_config(tmp_path, confidence_calibration_path=None),
            taxonomy_path=TAXONOMY, model_key=E5_KEY,
        )
    assert pg_mocks["databases"] == []


@pytest.mark.parametrize("model_key", ["fake", "bge-m3"])
def test_runtime_rejects_other_models_before_loading_or_opening_db(tmp_path, pg_mocks, model_key):
    with pytest.raises(ValueError, match="only e5-base-multilingual"):
        MatcherRuntime.from_config(
            config_path=write_config(tmp_path), taxonomy_path=TAXONOMY, model_key=model_key,
        )
    assert pg_mocks["databases"] == []


@pytest.mark.parametrize("backend,mode", [("numpy", "numpy_exact"), ("pgvector", "pg_hnsw")])
def test_runtime_rejects_non_exact_pgvector_backend(tmp_path, pg_mocks, backend, mode):
    with pytest.raises(ValueError, match="exact pgvector"):
        MatcherRuntime.from_config(
            config_path=write_config(tmp_path, retrieval_backend=backend, retrieval_mode=mode),
            taxonomy_path=TAXONOMY, model_key=E5_KEY,
        )
    assert pg_mocks["databases"] == []


@pytest.mark.parametrize("changes", [
    {"model_id": "not-e5"}, {"revision": None}, {"expected_dimension": 384},
    {"query_prefix": "search: "}, {"document_prefix": "document: "},
])
def test_runtime_requires_pinned_e5_config(tmp_path, pg_mocks, changes):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["models"][0].update(changes)
    with pytest.raises(ValueError, match="pinned intfloat/multilingual-e5-base"):
        MatcherRuntime.from_config(
            config_path=write_config(tmp_path, models=config["models"]),
            taxonomy_path=TAXONOMY, model_key=E5_KEY,
        )
    assert pg_mocks["databases"] == []


def test_serving_embedder_rejects_fake_bge_and_unpinned_models(monkeypatch):
    monkeypatch.setattr("model_intent.serving_config.SentenceTransformerEmbedder",
                        lambda *args, **kwargs: pytest.fail("invalid profile loaded"))
    for profile in (
        ModelProfile("fake", "fake", revision="r", expected_dimension=768),
        ModelProfile("bge-m3", "BAAI/bge-m3", revision="r", expected_dimension=768),
        ModelProfile(E5_KEY, E5_ID, expected_dimension=768),
        ModelProfile(E5_KEY, E5_ID, revision="r", expected_dimension=384),
    ):
        with pytest.raises(ValueError):
            make_embedder(profile, local_files_only=True)


def test_available_models_advertises_only_pinned_e5(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["models"].extend([
        {"key": "fake", "model_id": "fake"},
        {"key": "bge-m3", "model_id": "BAAI/bge-m3", "revision": "r", "expected_dimension": 1024},
    ])
    path = write_config(tmp_path, models=config["models"])
    assert MatcherRuntime.available_models(path) == [
        {"key": E5_KEY, "model_id": E5_ID, "stretch": False},
    ]
    assert MatcherRuntime.available_models(write_config(tmp_path, retrieval_backend="numpy")) == []


def test_runtime_query_slices_display_but_keeps_calibrated_decision_and_timing(tmp_path, pg_mocks):
    runtime = MatcherRuntime.from_config(
        config_path=write_config(tmp_path), taxonomy_path=TAXONOMY, model_key=E5_KEY,
    )
    pg_mocks["hits"] = [("34", 0.2), ("144", 0.19), ("42", 0.18), ("77", 0.17)]
    timing = SearchTiming("intent_analysis")
    result = runtime.query("  mai trời có mưa không  ", top_k=1, language="vi", timing=timing)
    assert len(result["prediction"]["hits"]) == 1
    assert result["prediction"]["hits"][0]["item_id"] == "34"
    assert result["prediction"]["query"] == "mai trời có mưa không"
    assert result["decision"]["action"] == "reject"
    assert result["decision"]["calibration_version"] != "none"
    assert result["runtime"]["generation_id"] == runtime.metadata.generation_id
    assert result["runtime"]["db_search_ms"] == 1.0
    assert result["runtime"]["query_elapsed_ms"] >= 0
    assert result["runtime"]["decision_ms"] >= 0
    assert pg_mocks["indexes"][1].search_calls == [("mai trời có mưa không", 20, timing)]
    names = [stage["name"] for stage in timing.stages]
    assert names[0] == "category_retrieval"
    assert "category_decision" in names and "category_serialize" in names
    assert names.index("category_decision") < names.index("category_serialize")


def test_runtime_category_decision_uses_hits_beyond_display_limit(tmp_path, pg_mocks):
    runtime = MatcherRuntime.from_config(
        config_path=write_config(tmp_path, confidence_calibration_path=None),
        taxonomy_path=TAXONOMY, model_key=E5_KEY,
    )
    pg_mocks["hits"] = [("34", 0.9), ("144", 0.89), ("42", 0.6)]
    result = runtime.query("tìm chăm sóc sức khỏe", top_k=1)
    assert len(result["prediction"]["hits"]) == 1
    assert result["decision"]["action"] == "clarify"
    assert result["decision"]["reason"] == "near_group_competition"
    assert {option["item_id"] for option in result["decision"]["options"]} == {"34", "144"}
    assert result["runtime"]["candidate_encode_seconds"] is None


def test_runtime_selection_is_restricted_to_offered_leaf_ids(tmp_path, pg_mocks):
    runtime = MatcherRuntime.from_config(
        config_path=write_config(tmp_path), taxonomy_path=TAXONOMY, model_key=E5_KEY,
    )
    selected = runtime.resolve_selection(
        query="đau đầu", selected_id="34", allowed_option_ids=["34", "144"], language="vi",
    )
    assert selected["decision"]["action"] == "accept"
    assert selected["decision"]["accepted_code"] == "pharmacy"
    assert selected["runtime"]["generation_id"] == runtime.metadata.generation_id
    assert pg_mocks["indexes"][1].search_calls == []
    with pytest.raises(ValueError, match="not offered"):
        runtime.resolve_selection(query="q", selected_id="85", allowed_option_ids=["34"])
    with pytest.raises(ValueError, match="leaf"):
        runtime.resolve_selection(query="q", selected_id="41", allowed_option_ids=["41"])


@pytest.mark.parametrize("text,top_k,language", [(" ", 5, "vi"), ("place", 0, "vi"), ("place", 1, "fr")])
def test_runtime_query_rejects_invalid_inputs_before_retrieval(tmp_path, pg_mocks, text, top_k, language):
    runtime = MatcherRuntime.from_config(
        config_path=write_config(tmp_path), taxonomy_path=TAXONOMY, model_key=E5_KEY,
    )
    with pytest.raises(ValueError):
        runtime.query(text, top_k=top_k, language=language)
    assert pg_mocks["indexes"][1].search_calls == []
