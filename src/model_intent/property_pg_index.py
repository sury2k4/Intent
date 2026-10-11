"""Read-only exact pgvector property suggestions from a pinned ready generation."""
from __future__ import annotations

from contextlib import nullcontext
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .db import Database
from .embedding_sync import stable_hash
from .normalization import fold_vietnamese, normalize_text
from .poi_properties import PropertyProfiles
from .property_extraction import PropertyExtractor, _fold_with_offsets
from .property_semantic import (
    SemanticPropertyCorpus, prepare_property_queries, rank_property_scores, stable_property_hash,
)
from .search_timing import SearchTiming
from .serving_config import E5_MODEL_ID, E5_MODEL_REVISION
from .taxonomy import Taxonomy

_CUE_PREFIX = re.compile(
    r"^(?:(?:va|and|nhung|but|hoac|or)\s+)*"
    r"(?:(?:khong co|khong can|khong muon|khong cho|khong ngoi|khong quan trong|"
    r"dung co|dung|uu tien|neu co|co|can co|can|phai co|"
    r"do not want|dont want|do not need|dont need|not required|"
    r"with(?: a| an)?|without(?: a| an)?|must have|prefer|need|no)\s+)*"
)
_CUE_SUFFIX = re.compile(r"\s+(?:neu co|if possible|would be nice|would be great)$")


class PropertyPgIndex:
    def __init__(
        self, database: Database, corpus: SemanticPropertyCorpus,
        definitions: Mapping[str, Mapping[str, Any]], embedder, encode_lock,
        *, generation_id: int,
    ) -> None:
        if type(generation_id) is not int or generation_id < 1:
            raise ValueError("a positive pinned property generation_id is required")
        self.database = database
        self.corpus = corpus
        self.definitions = definitions
        self._typo_extractor = PropertyExtractor(definitions)
        self.embedder = embedder
        self.encode_lock = encode_lock
        self.generation_id = generation_id
        self.profile_count = corpus.profile_count
        self.fingerprint = corpus.content_set_hash
        metadata = embedder.metadata()
        if (
            metadata.get("model_id") != E5_MODEL_ID
            or metadata.get("revision") != E5_MODEL_REVISION
            or metadata.get("query_prefix") != "query: "
            or metadata.get("document_prefix") != "passage: "
            or metadata.get("adapter") != "sentence-transformer-v1"
            or int(metadata.get("dimension", 0)) != 768
        ):
            raise ValueError("property pgvector requires the pinned E5-base 768D model and prefixes")
        if corpus.catalog_hash != stable_property_hash(definitions):
            raise ValueError("property DB generation and serving catalog do not match")
        with database.connection() as (connection, _):
            row = connection.execute(
                """
                SELECT g.state, g.expected_count, g.completed_count, g.content_set_hash,
                       s.catalog_hash, s.source_hash, s.taxonomy_fingerprint,
                       s.renderer_version, s.content_set_hash, s.property_count, s.profile_count,
                       m.model_key, m.model_id, m.revision, m.dimension,
                       m.query_prefix, m.document_prefix, m.normalize_embeddings,
                       m.max_seq_length, m.adapter_version, m.config_hash, m.status
                FROM property_embedding_generations g
                JOIN property_profile_sets s ON s.id=g.profile_set_id
                JOIN embedding_model_configs m ON m.id=g.model_config_id
                WHERE g.id=%s
                """,
                (generation_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("property embedding generation does not exist")
            (
                state, expected, completed, generation_hash, catalog_hash, source_hash,
                taxonomy_hash, renderer, set_hash, property_count, profile_count,
                model_key, model_id, revision, dimension, query_prefix, document_prefix,
                normalize_embeddings, max_seq_length, adapter_version, model_config_hash, model_status,
            ) = row
            if (
                state != "ready" or int(expected) != self.profile_count or int(completed) != self.profile_count
                or generation_hash != corpus.content_set_hash or set_hash != corpus.content_set_hash
                or catalog_hash != corpus.catalog_hash or source_hash != corpus.source_hash
                or taxonomy_hash != corpus.taxonomy_fingerprint or renderer != corpus.renderer_version
                or int(property_count) != corpus.property_count or int(profile_count) != self.profile_count
                or model_key != metadata.get("model_key") or model_id != metadata.get("model_id")
                or revision != metadata.get("revision") or int(dimension) != 768
                or query_prefix != "query: " or document_prefix != "passage: "
                or not normalize_embeddings or int(max_seq_length) != int(metadata.get("max_seq_length", 0))
                or adapter_version != "sentence-transformer-v1" or model_status != "active"
                or model_config_hash != stable_hash({
                    "adapter": "sentence-transformer-v1", "model_key": metadata.get("model_key"),
                    "model_id": metadata.get("model_id"), "revision": metadata.get("revision"),
                    "query_prefix": metadata.get("query_prefix"), "document_prefix": metadata.get("document_prefix"),
                    "normalize_embeddings": metadata.get("normalize_embeddings"),
                    "dimension": metadata.get("dimension"), "max_seq_length": metadata.get("max_seq_length"),
                })
            ):
                raise RuntimeError("property embedding generation is not ready or has incompatible provenance")
            persisted = connection.execute(
                """
                SELECT property_id, language, profile_key, search_text, content_hash,
                       vector_dims(embedding), vector_norm(embedding)
                FROM property_semantic_profiles
                WHERE generation_id=%s
                ORDER BY property_id COLLATE "C", language COLLATE "C", profile_key COLLATE "C"
                """,
                (generation_id,),
            ).fetchall()
        if len(persisted) != self.profile_count:
            raise RuntimeError("property embedding generation has incomplete profile rows")
        for db_row, source in zip(persisted, corpus.rows):
            if (
                tuple(db_row[:5]) != (
                    source.property_id, source.language, source.profile_key,
                    source.text, source.content_hash,
                )
                or int(db_row[5]) != 768 or not 0.99 < float(db_row[6]) < 1.01
            ):
                raise RuntimeError("property embedding row content or vector norm differs from the pinned corpus")
        self._ids = tuple(row.property_id for row in corpus.rows)

    @staticmethod
    def _query_text(text: str) -> str:
        folded, _, ends = _fold_with_offsets(text)
        prefix = _CUE_PREFIX.match(folded)
        start = ends[prefix.end() - 1] if prefix and prefix.end() else 0
        source = text[start:].strip(" ,;.!?\n")
        folded_source = normalize_text(fold_vietnamese(source))
        suffix = _CUE_SUFFIX.search(folded_source)
        if suffix:
            words = source.split()
            suffix_words = len(folded_source[suffix.start():].split())
            source = " ".join(words[:-suffix_words]).strip(" ,;.!?\n")
        return source.strip()

    def suggest(
        self, spans: Sequence[Mapping[str, Any]], *, category_id: str | None = None,
        taxonomy: Taxonomy | None = None, profiles: PropertyProfiles | None = None,
        top_k: int = 3, minimum_cosine: float = 0.80, timing: SearchTiming | None = None,
    ) -> list[dict[str, Any]]:
        if not spans:
            return []
        prepared, valid, probes = prepare_property_queries(
            spans, self._typo_extractor.unique_typo_alias, query_text=self._query_text,
        )
        if not valid:
            return rank_property_scores(
                spans, prepared, valid, np.empty((0, self.profile_count), dtype=np.float32),
                self._ids, self.definitions, category_id=category_id, taxonomy=taxonomy,
                profiles=profiles, top_k=top_k, minimum_cosine=minimum_cosine,
                typo_probes=probes,
            )
        with (timing.measure("property_query_encode", span_count=len(valid)) if timing else nullcontext()):
            with self.encode_lock:
                vectors = np.asarray(self.embedder.encode_queries([value for _, value in valid]), dtype=np.float32)
        if vectors.shape != (len(valid), 768) or not np.all(np.isfinite(vectors)):
            raise ValueError("property query E5 embeddings have invalid shape or nonfinite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ValueError("property query E5 embeddings include a zero vector")
        vectors = np.asarray(vectors / norms, dtype=np.float32)
        matrix = np.empty((len(valid), self.profile_count), dtype=np.float64)
        with (timing.measure("property_db_exact", span_count=len(valid)) if timing else nullcontext()):
            with self.database.connection() as (connection, _):
                for index, vector in enumerate(vectors):
                    rows = connection.execute(
                        """
                        SELECT p.property_id, p.language, p.profile_key,
                               1 - (p.embedding <=> %s) AS cosine
                        FROM property_semantic_profiles p
                        JOIN property_embedding_generations g ON g.id=p.generation_id AND g.state='ready'
                        WHERE p.generation_id=%s
                        ORDER BY p.property_id COLLATE "C", p.language COLLATE "C", p.profile_key COLLATE "C"
                        """,
                        (vector, self.generation_id),
                    ).fetchall()
                    if len(rows) != self.profile_count or tuple(str(row[0]) for row in rows) != self._ids:
                        raise RuntimeError("pinned property profile generation changed during retrieval")
                    for row_index, (row, expected) in enumerate(zip(rows, self.corpus.rows)):
                        if (str(row[0]), str(row[1]), str(row[2])) != (
                            expected.property_id, expected.language, expected.profile_key,
                        ):
                            raise RuntimeError("pinned property profile row order changed during retrieval")
                        matrix[index, row_index] = float(row[3])
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("pgvector property cosine scores contain nonfinite values")
        with (timing.measure("property_cosine_rank", span_count=len(valid)) if timing else nullcontext()):
            return rank_property_scores(
                spans, prepared, valid, matrix, self._ids, self.definitions,
                category_id=category_id, taxonomy=taxonomy,
                profiles=profiles, top_k=top_k, minimum_cosine=minimum_cosine,
                typo_probes=probes,
            )
