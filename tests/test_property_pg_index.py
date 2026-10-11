from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from model_intent.poi_properties import load_property_profiles
from model_intent.property_pg_index import PropertyPgIndex
from model_intent.property_semantic import load_property_semantic_corpus
from model_intent.embedding_sync import stable_hash
from model_intent.taxonomy import Taxonomy


class FakeE5:
    def metadata(self):
        return {
            "model_key": "e5-base-multilingual", "model_id": "intfloat/multilingual-e5-base",
            "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
            "query_prefix": "query: ", "document_prefix": "passage: ",
            "dimension": 768, "max_seq_length": 192, "normalize_embeddings": True,
            "adapter": "sentence-transformer-v1",
        }

    def encode_documents(self, texts):
        output = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=768).astype(np.float32)
            output.append(vector / np.linalg.norm(vector))
        return np.array(output)

    def encode_queries(self, texts):
        return self.encode_documents(texts)


class Result:
    def __init__(self, row=None, rows=()):
        self.row, self.rows = row, rows

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, corpus, vectors, model):
        self.corpus, self.vectors, self.model = corpus, vectors, model
        self.state = "ready"
        self.catalog_hash = corpus.catalog_hash
        self.mismatch = False

    def execute(self, statement, parameters):
        assert statement.lstrip().startswith("SELECT"), "property DB index must only read"
        if "FROM property_embedding_generations g" in statement:
            meta = self.model.metadata()
            return Result(row=(
                self.state, len(self.corpus.rows), len(self.corpus.rows), self.corpus.content_set_hash,
                self.catalog_hash, self.corpus.source_hash, self.corpus.taxonomy_fingerprint,
                self.corpus.renderer_version, self.corpus.content_set_hash,
                self.corpus.property_count, self.corpus.profile_count,
                meta["model_key"], meta["model_id"], meta["revision"], 768,
                meta["query_prefix"], meta["document_prefix"], True, meta["max_seq_length"],
                "sentence-transformer-v1", stable_hash({
                    "adapter": "sentence-transformer-v1", "model_key": meta["model_key"],
                    "model_id": meta["model_id"], "revision": meta["revision"],
                    "query_prefix": meta["query_prefix"], "document_prefix": meta["document_prefix"],
                    "normalize_embeddings": True, "dimension": 768,
                    "max_seq_length": meta["max_seq_length"],
                }), "active",
            ))
        if "vector_dims(embedding)" in statement:
            rows = tuple((
                row.property_id, row.language, row.profile_key,
                row.text if not self.mismatch else "tampered", row.content_hash, 768, 1.0,
            ) for row in self.corpus.rows)
            return Result(rows=rows)
        if "p.embedding <=>" in statement:
            if self.state != "ready":
                return Result(rows=())
            vector = parameters[0]
            rows = tuple((
                row.property_id, row.language, row.profile_key,
                float(1 - (1 - float(np.dot(vector, source_vector)))),
            ) for row, source_vector in zip(self.corpus.rows, self.vectors))
            return Result(rows=rows)
        raise AssertionError(statement)


class FakeDatabase:
    def __init__(self, connection):
        self.connection_impl = connection

    @contextmanager
    def connection(self):
        yield self.connection_impl, None


@pytest.fixture(scope="module")
def corpus_and_vectors():
    taxonomy = Taxonomy.load("data/taxonomies/place_types.poc.v3.json")
    profiles = load_property_profiles("data/properties/poi_properties.poc.v8.json", taxonomy)
    model = FakeE5()
    corpus = load_property_semantic_corpus(
        "data/properties/property_semantic_profiles.poc.v2.json", profiles.definitions,
        taxonomy, model.metadata(),
    )
    vectors = model.encode_documents([row.text for row in corpus.rows])
    return taxonomy, profiles, model, corpus, vectors


def test_pg_exact_matches_pinned_document_cosines_without_in_memory_index(corpus_and_vectors):
    taxonomy, profiles, model, corpus, vectors = corpus_and_vectors
    connection = FakeConnection(corpus, vectors, model)
    index = PropertyPgIndex(
        FakeDatabase(connection), corpus, profiles.definitions,
        model, threading.RLock(), generation_id=12,
    )
    spans = [
        {"text": "có chỗ cất ô tô", "start_offset": 9, "end_offset": 24},
        {"text": "mát rượi", "start_offset": 25, "end_offset": 33},
    ]
    postgres = index.suggest(spans, taxonomy=taxonomy, profiles=profiles, category_id="137", minimum_cosine=-1)
    for span, row in zip(spans, postgres):
        query_text = PropertyPgIndex._query_text(span["text"])
        assert row["query_text"] == query_text
        query_vector = model.encode_queries([query_text])[0]
        by_id = {}
        for profile, vector in zip(corpus.rows, vectors):
            by_id[profile.property_id] = max(by_id.get(profile.property_id, -1.0), float(np.dot(query_vector, vector)))
        expected = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))[:3]
        assert [candidate["property_id"] for candidate in row["unrestricted_top"]] == [key for key, _ in expected]
        assert [candidate["cosine"] for candidate in row["unrestricted_top"]] == pytest.approx(
            [score for _, score in expected], abs=1e-6,
        )
        assert row["status"] != "confirmed" and "applied" not in row
    connection.state = "retired"
    with pytest.raises(RuntimeError, match="changed during retrieval"):
        index.suggest(spans, taxonomy=taxonomy, profiles=profiles, category_id="137")


def test_property_pg_index_rejects_invalid_query_vectors(corpus_and_vectors):
    _, profiles, model, corpus, vectors = corpus_and_vectors
    index = PropertyPgIndex(
        FakeDatabase(FakeConnection(corpus, vectors, model)), corpus, profiles.definitions,
        model, threading.RLock(), generation_id=12,
    )
    span = [{"text": "có bóng mát"}]
    original_encode = model.encode_queries
    try:
        model.encode_queries = lambda texts: np.zeros((len(texts), 768), dtype=np.float32)
        with pytest.raises(ValueError, match="zero vector"):
            index.suggest(span)
        model.encode_queries = lambda texts: np.full((len(texts), 768), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="nonfinite"):
            index.suggest(span)
        model.encode_queries = lambda texts: np.zeros((len(texts), 767), dtype=np.float32)
        with pytest.raises(ValueError, match="invalid shape"):
            index.suggest(span)
    finally:
        model.encode_queries = original_encode


def test_property_pg_index_rejects_wrong_model_or_generation(corpus_and_vectors):
    _, profiles, model, corpus, vectors = corpus_and_vectors
    database = FakeDatabase(FakeConnection(corpus, vectors, model))
    for generation in (0, -1, True):
        with pytest.raises(ValueError, match="positive pinned"):
            PropertyPgIndex(database, corpus, profiles.definitions, model, threading.RLock(), generation_id=generation)
    original_metadata = model.metadata
    try:
        model.metadata = lambda: {**original_metadata(), "query_prefix": ""}
        with pytest.raises(ValueError, match="pinned E5"):
            PropertyPgIndex(database, corpus, profiles.definitions, model, threading.RLock(), generation_id=12)
    finally:
        model.metadata = original_metadata


def test_property_pg_index_rejects_wrong_generation_or_tampered_rows(corpus_and_vectors):

    taxonomy, profiles, model, corpus, vectors = corpus_and_vectors
    connection = FakeConnection(corpus, vectors, model)
    connection.state = "building"
    with pytest.raises(RuntimeError, match="not ready"):
        PropertyPgIndex(FakeDatabase(connection), corpus, profiles.definitions, model, threading.RLock(), generation_id=12)
    connection.state = "ready"
    connection.mismatch = True
    with pytest.raises(RuntimeError, match="row content"):
        PropertyPgIndex(FakeDatabase(connection), corpus, profiles.definitions, model, threading.RLock(), generation_id=12)
    connection.mismatch = False
    connection.catalog_hash = "f" * 64
    with pytest.raises(RuntimeError, match="incompatible provenance"):
        PropertyPgIndex(FakeDatabase(connection), corpus, profiles.definitions, model, threading.RLock(), generation_id=12)
