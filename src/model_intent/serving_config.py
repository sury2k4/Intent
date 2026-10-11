from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .embedders import SentenceTransformerEmbedder
from .schemas import ModelProfile

E5_MODEL_KEY = "e5-base-multilingual"
E5_MODEL_ID = "intfloat/multilingual-e5-base"
E5_MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
E5_DIMENSION = 768


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to load configuration") from exc
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def profiles_from_config(config: dict[str, Any], include_stretch: bool = False) -> list[ModelProfile]:
    profiles = []
    for raw in config.get("models", []):
        if not raw.get("enabled", True) or (raw.get("stretch", False) and not include_stretch):
            continue
        profiles.append(ModelProfile(
            key=str(raw["key"]), model_id=str(raw["model_id"]),
            query_prefix=str(raw.get("query_prefix", "")), document_prefix=str(raw.get("document_prefix", "")),
            batch_size=int(raw.get("batch_size", 32)), max_seq_length=int(raw.get("max_seq_length", 128)),
            expected_dimension=(int(raw["expected_dimension"]) if raw.get("expected_dimension") is not None else None),
            revision=raw.get("revision"), enabled=True, stretch=bool(raw.get("stretch", False)),
        ))
    return profiles


def make_embedder(profile: ModelProfile, local_files_only: bool = False):
    if profile.key != E5_MODEL_KEY or profile.model_id != E5_MODEL_ID:
        raise ValueError("category serving requires intfloat/multilingual-e5-base")
    if profile.revision != E5_MODEL_REVISION:
        raise ValueError("category serving requires the pinned E5-base model revision")
    if profile.expected_dimension != E5_DIMENSION:
        raise ValueError("category serving requires an explicitly configured 768D E5 model")
    if profile.query_prefix != "query: " or profile.document_prefix != "passage: ":
        raise ValueError("category serving requires the pinned E5 query/passage prefixes")
    return SentenceTransformerEmbedder(profile, local_files_only=local_files_only)


def configure_runtime(config: dict[str, Any], local_files_only: bool) -> None:
    threads = max(1, int(config.get("threads", 1)))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import random
        random.seed(int(config.get("seed", 17)))
        np.random.seed(int(config.get("seed", 17)))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(int(config.get("seed", 17)))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except (ImportError, RuntimeError):
        pass
