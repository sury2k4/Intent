from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ..schemas import ModelProfile


class SentenceTransformerEmbedder:
    def __init__(self, profile: ModelProfile, local_files_only: bool = False, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the optional 'models' dependencies to use real embedders") from exc
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        self.profile = profile
        self.model_key = profile.key
        self.model_id = profile.model_id
        kwargs = {"device": device, "trust_remote_code": False}
        if profile.revision:
            kwargs["revision"] = profile.revision
        if local_files_only:
            kwargs["local_files_only"] = True
        try:
            self.model = SentenceTransformer(profile.model_id, **kwargs)
        except AttributeError as exc:
            if local_files_only:
                raise RuntimeError(
                    f"local model snapshot for {profile.model_id!r} is missing or incomplete; "
                    "run once without --local-files-only to download it"
                ) from exc
            raise
        self.model.max_seq_length = profile.max_seq_length
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if profile.expected_dimension is not None and self.dimension != profile.expected_dimension:
            raise ValueError(
                f"model {profile.model_id} produced {self.dimension} dimensions; "
                f"expected {profile.expected_dimension}"
            )

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        prepared = [prefix + str(text) for text in texts]
        result = self.model.encode(
            prepared,
            batch_size=self.profile.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )
        return np.asarray(result, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.profile.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.profile.document_prefix)

    def metadata(self) -> dict[str, object]:
        return {
            "adapter": "sentence-transformer-v1",
            "model_key": self.model_key,
            "model_id": self.model_id,
            "revision": self.profile.revision,
            "query_prefix": self.profile.query_prefix,
            "document_prefix": self.profile.document_prefix,
            "normalize_embeddings": True,
            "dimension": self.dimension,
            "max_seq_length": self.profile.max_seq_length,
            "batch_size": self.profile.batch_size,
            "device": "cpu",
        }
