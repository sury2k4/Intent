from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    model_key: str
    model_id: str
    dimension: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def metadata(self) -> dict[str, object]: ...
