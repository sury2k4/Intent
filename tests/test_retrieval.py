import numpy as np

from model_intent.embedders.fake import FakeEmbedder
from model_intent.rendering import render_category_profiles, render_variants
from model_intent.retrieval import SemanticIndex, accept_prediction
from model_intent.taxonomy import Taxonomy


def test_profile_retrieval_collapses_bilingual_profiles_to_one_category():
    taxonomy = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "food", "names": {"vi": "Ăn uống", "en": "Food"}},
        {"id": "2", "parent_id": "1", "code": "coffee", "names": {"vi": "Cà phê", "en": "Coffee"}},
        {"id": "3", "parent_id": "1", "code": "tea", "names": {"vi": "Trà", "en": "Tea"}},
    ]})
    embedder = FakeEmbedder()
    profiles = render_category_profiles(taxonomy)
    prediction = SemanticIndex.build(taxonomy, profiles, embedder).search("cà phê", embedder, top_k=3)
    assert len(prediction.hits) == 2
    assert len({hit.item_id for hit in prediction.hits}) == 2
    assert prediction.hits[0].item_id == "2"
    assert prediction.hits[0].profile_kind == "primary"


def test_retrieval_returns_one_hit_per_category():
    taxonomy = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "medicine", "names": {"vi": "Thuốc", "en": "Medicine"}},
        {"id": "2", "parent_id": "0", "code": "coffee", "names": {"vi": "Cà phê", "en": "Coffee"}},
    ]})
    embedder = FakeEmbedder()
    variants = render_variants(taxonomy)
    index = SemanticIndex.build(taxonomy, variants, embedder)
    prediction = index.search("thuốc", embedder, top_k=2)
    assert len(prediction.hits) == 2
    assert prediction.hits[0].item_id == "1"
    assert prediction.hits[0].rank == 1
    assert prediction.margin is not None


def test_zero_vector_does_not_crash():
    taxonomy = Taxonomy.from_payload({"items": [{"id": "1", "parent_id": "0", "code": "x", "names": {"vi": "X"}}]})
    variants = render_variants(taxonomy)
    index = SemanticIndex(taxonomy, variants, np.zeros((len(variants), 4), dtype=np.float32), "test")
    class Zero:
        model_key = "test"
        def encode_queries(self, texts): return np.zeros((len(texts), 4), dtype=np.float32)
    prediction = index.search("anything", Zero(), top_k=1)
    assert prediction.hits[0].score == 0.0


def test_confidence_gate_abstains_on_low_margin():
    taxonomy = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "a", "names": {"vi": "A"}},
        {"id": "2", "parent_id": "0", "code": "b", "names": {"vi": "B"}},
    ]})
    embedder = FakeEmbedder()
    index = SemanticIndex.build(taxonomy, render_variants(taxonomy, ("vi",)), embedder)
    prediction = index.search("không liên quan", embedder, top_k=2)
    assert accept_prediction(prediction, min_score=0.99, min_margin=0.0) is None
