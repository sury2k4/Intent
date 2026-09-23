from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def cache_key(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def save_matrix(path: str | Path, matrix: np.ndarray, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    np.savez_compressed(temp, matrix=np.asarray(matrix, dtype=np.float32))
    # np.savez appends .npz when the temporary name does not end in .npz.
    generated = Path(str(temp) + ".npz") if not str(temp).endswith(".npz") else temp
    os.replace(generated, target)
    target.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_matrix(path: str | Path, expected_metadata: dict[str, Any] | None = None) -> np.ndarray | None:
    target = Path(path)
    meta_path = target.with_suffix(".json")
    if not target.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_metadata is not None and metadata != expected_metadata:
        return None
    with np.load(target) as data:
        return np.asarray(data["matrix"], dtype=np.float32)
