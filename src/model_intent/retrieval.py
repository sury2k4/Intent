from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .normalization import text_variants
from .schemas import CategorySearchProfile, Prediction, RetrievalHit
from .taxonomy import Taxonomy


class SemanticIndex:
    """Dense embedding index using cosine similarity only."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        profiles: Sequence[CategorySearchProfile],
        vectors: np.ndarray,
        model_key: str,
    ) -> None:
        if len(profiles) != len(vectors):
            raise ValueError("profiles and vectors must have the same length")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("candidate vectors must be a finite 2D matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(norms, 1e-12), dtype=np.float32)
        self.taxonomy = taxonomy
        self.variants = list(profiles)
        self.vectors = matrix
        self.model_key = model_key

    @classmethod
    def build(cls, taxonomy: Taxonomy, profiles: Sequence[CategorySearchProfile], embedder) -> "SemanticIndex":
        vectors = embedder.encode_documents([profile.text for profile in profiles])
        return cls(taxonomy, profiles, vectors, embedder.model_key)

    def search(self, query: str, embedder, top_k: int = 5, query_id: str = "query") -> Prediction:
        # Accent-folding is an embedding preprocessing variant. No lexical,
        # substring, edit-distance, or dictionary score enters the ranking.
        query_entries = text_variants(query)
        query_texts = [value for _, value in query_entries]
        query_matrix = np.asarray(embedder.encode_queries(query_texts), dtype=np.float32)
        if query_matrix.ndim != 2 or not np.isfinite(query_matrix).all():
            raise ValueError("query vectors must be a finite 2D matrix")
        norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        query_matrix = np.divide(query_matrix, np.maximum(norms, 1e-12), dtype=np.float32)

        all_scores = query_matrix @ self.vectors.T
        variant_top_ids = tuple(
            self.variants[int(np.argmax(row))].item_id
            for row in all_scores
        )
        winning_query_indexes = np.argmax(all_scores, axis=0)
        scores = np.max(all_scores, axis=0)
        order = np.argsort(-scores, kind="stable")
        by_item: dict[str, RetrievalHit] = {}
        for index in order:
            variant = self.variants[int(index)]
            item = self.taxonomy.items[variant.item_id]
            query_kind = query_entries[int(winning_query_indexes[int(index)])][0]
            hit = RetrievalHit(
                item_id=item.id,
                code=item.code,
                score=float(scores[int(index)]),
                rank=0,
                language=variant.language,
                matched_text=variant.text,
                path=tuple(self.taxonomy.path_names(item.id, variant.language)),
                profile_kind=getattr(variant, "profile_kind", "primary"),
                query_variant_kind=query_kind,
            )
            previous = by_item.get(item.id)
            if previous is None or hit.score > previous.score:
                by_item[item.id] = hit
        hits = sorted(by_item.values(), key=lambda hit: (-hit.score, hit.item_id))[:top_k]
        ranked = tuple(
            RetrievalHit(
                hit.item_id,
                hit.code,
                hit.score,
                rank + 1,
                hit.language,
                hit.matched_text,
                hit.path,
                hit.profile_kind,
                hit.query_variant_kind,
            )
            for rank, hit in enumerate(hits)
        )
        top1 = ranked[0].score if ranked else None
        margin = top1 - ranked[1].score if len(ranked) > 1 else None
        return Prediction(
            query_id, query, ranked, top1, margin, self.model_key,
            tuple(query_texts), variant_top_ids,
        )


def accept_prediction(prediction: Prediction, min_score: float, min_margin: float) -> RetrievalHit | None:
    """Compatibility helper; new callers should use the decision policy."""
    if not prediction.hits or prediction.top1_score is None or prediction.margin is None:
        return None
    if prediction.top1_score < min_score or prediction.margin < min_margin:
        return None
    return prediction.hits[0]


def prediction_to_dict(prediction: Prediction) -> dict[str, object]:
    data = asdict(prediction)
    data["hits"] = [asdict(hit) for hit in prediction.hits]
    return data
