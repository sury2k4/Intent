from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np


class FakeEmbedder:
    """Deterministic token/character hash embedder for network-free tests."""

    model_key = "fake"
    model_id = "fake"
    dimension = 128

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def metadata(self) -> dict[str, object]:
        return {"model_key": self.model_key, "model_id": self.model_id, "dimension": self.dimension}
