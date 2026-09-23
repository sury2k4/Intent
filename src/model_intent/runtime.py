from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .benchmark import _configure_runtime, load_yaml, make_embedder, profiles_from_config
from .cache import cache_key, load_matrix, save_matrix
from .confidence import load_confidence_calibration
from .decision import (
    decide_intent,
    decision_to_dict,
    load_decision_policy,
    validate_decision_policy,
)
from .db import Database
from .embedding_sync import (
    profile_content_hash,
    profile_hash,
    profile_source_fingerprint,
    semantic_model_metadata,
    stable_hash,
)
from .embedders import FakeEmbedder
from .pgvector_index import PgVectorIndex
from .rendering import (
    PROFILE_RENDERER_VERSION,
    PROFILE_SCHEMA_VERSION,
    RENDERER_VERSION,
    load_enrichment,
    render_category_profiles,
    validate_enrichment,
)
from .retrieval import SemanticIndex, prediction_to_dict
from .schemas import IntentDecision, ModelProfile
from .taxonomy import Taxonomy


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RuntimeMetadata:
    model_key: str
    model_id: str
    candidate_count: int
    taxonomy_items: int
    taxonomy_fingerprint: str
    cache_hit: bool
    model_load_seconds: float
    candidate_load_seconds: float
    candidate_encode_seconds: float | None
    cache_path: str
    backend: str = "numpy"
    retrieval_mode: str = "numpy_exact"
    generation_id: int | None = None
    model_revision: str | None = None
    profile_count: int | None = None
    indexed_category_count: int | None = None
    profile_set_id: int | None = None
    artifact_kind: str = "category_profile_v3"
    profile_schema_version: int = 3


class MatcherRuntime:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        taxonomy: Taxonomy,
        enrichment: dict[str, dict[str, Any]],
        policy: dict[str, Any],
        model_key: str,
        embedder: Any,
        index: Any,
        metadata: RuntimeMetadata,
        confidence_calibration: dict[str, Any] | None = None,
        database: Database | None = None,
    ) -> None:
        self.config = config
        self.taxonomy = taxonomy
        self.enrichment = enrichment
        self.policy = policy
        self.model_key = model_key
        self.embedder = embedder
        self.index = index
        self.metadata = metadata
        self.confidence_calibration = confidence_calibration
        self.database = database
        self._lock = threading.RLock()

    @classmethod
    def from_config(
        cls,
        *,
        config_path: str | Path,
        taxonomy_path: str | Path,
        model_key: str,
        local_files_only: bool = False,
        generation_id: int | None = None,
    ) -> "MatcherRuntime":
        config_path = Path(config_path).resolve()
        taxonomy_path = Path(taxonomy_path).resolve()
        project_root = config_path.parent.parent
        config = load_yaml(config_path)
        _configure_runtime(config, local_files_only)
        taxonomy = Taxonomy.load(taxonomy_path)

        def resolve_config_path(value: str | Path | None) -> Path | None:
            if value is None:
                return None
            path = Path(value)
            return path if path.is_absolute() else (project_root / path).resolve()

        enrichment_path = resolve_config_path(config.get("enrichment_path"))
        policy_path = resolve_config_path(config.get("policy_path"))
        if policy_path is None:
            raise ValueError("config requires policy_path")
        enrichment = load_enrichment(enrichment_path)
        validate_enrichment(taxonomy, enrichment)
        policy = load_decision_policy(policy_path)
        validate_decision_policy(policy, taxonomy)

        profile = cls._resolve_profile(config, model_key)
        model_started = time.perf_counter()
        embedder = FakeEmbedder() if model_key == "fake" else make_embedder(profile, local_files_only)
        model_load_seconds = time.perf_counter() - model_started

        embedder_metadata = embedder.metadata()
        calibration_path = resolve_config_path(config.get("confidence_calibration_path"))
        confidence_calibration = None
        if calibration_path is not None and model_key != "fake":
            try:
                confidence_calibration = load_confidence_calibration(
                    calibration_path,
                    model_key=model_key,
                    model_revision=str(embedder_metadata.get("revision") or ""),
                    renderer_version=PROFILE_RENDERER_VERSION,
                )
            except Exception:
                confidence_calibration = None
        actual_dimension = int(embedder_metadata.get("dimension", getattr(embedder, "dimension", 0)))
        if profile.expected_dimension is not None and actual_dimension != profile.expected_dimension:
            raise ValueError(
                f"model {model_key} produced {actual_dimension} dimensions; "
                f"expected {profile.expected_dimension}"
            )
        backend = str(config.get("retrieval_backend", "numpy"))
        profiles = render_category_profiles(
            taxonomy,
            tuple(config.get("languages", ["vi", "en"])),
            enrichment,
            require_rich_fields=True,
        )
        expected_source_fingerprint = profile_source_fingerprint(enrichment)
        expected_content_set_hash = stable_hash(sorted(
            (profile.item_id, profile.language, profile_content_hash(profile))
            for profile in profiles
        ))
        database: Database | None = None
        if backend == "pgvector" and model_key != "fake":
            if model_key not in ("e5-base-multilingual", "bge-m3"):
                raise ValueError("pgvector serving supports e5-base-multilingual and bge-m3")
            database = Database(
                min_size=int(config.get("db_pool_min_size", 1)),
                max_size=int(config.get("db_pool_max_size", 8)),
                timeout=float(config.get("db_pool_timeout_seconds", 10)),
                statement_timeout_ms=int(config.get("db_statement_timeout_ms", 5000)),
            )
            database.open()
            mode = str(config.get("retrieval_mode", "pg_exact"))
            try:
                load_started = time.perf_counter()
                probe = PgVectorIndex(
                    taxonomy,
                    database,
                    model_key=model_key,
                    mode=mode,
                    ef_search=int(config.get("hnsw_ef_search", 100)),
                    overfetch=int(config.get("hnsw_overfetch", 200)),
                    dimension=actual_dimension,
                )
                active = (
                    probe.generation_metadata(generation_id)
                    if generation_id is not None
                    else probe.active_generation()
                )
                candidate_load_seconds = time.perf_counter() - load_started
                expected_config_hash = stable_hash(semantic_model_metadata(profile))
                expected_artifact_kind = "category_profile_bge_m3_v1" if model_key == "bge-m3" else "category_profile_v3"
                if (
                    active["taxonomy_fingerprint"] != taxonomy.fingerprint()
                    or (model_key == "e5-base-multilingual" and active["config_hash"] != expected_config_hash)
                    or active["renderer_version"] != PROFILE_RENDERER_VERSION
                    or active["artifact_kind"] != expected_artifact_kind
                    or active["dimension"] != actual_dimension
                    or active["source_fingerprint"] != expected_source_fingerprint
                    or active["profile_schema_version"] != PROFILE_SCHEMA_VERSION
                    or (model_key == "e5-base-multilingual" and active["content_set_hash"] != expected_content_set_hash)
                    or active["profile_count_from_set"] != len(profiles)
                    or active["indexed_category_count"] != len({item.item_id for item in profiles})
                ):
                    raise RuntimeError(
                        "active pgvector generation is incompatible with the current model/taxonomy/renderer; "
                        "run embeddings sync"
                    )
                index = PgVectorIndex(
                    taxonomy,
                    database,
                    model_key=model_key,
                    mode=mode,
                    ef_search=int(config.get("hnsw_ef_search", 100)),
                    overfetch=int(config.get("hnsw_overfetch", 200)),
                    generation_id=int(active["generation_id"]),
                    dimension=actual_dimension,
                )
            except Exception:
                database.close()
                raise
            metadata = RuntimeMetadata(
                model_key=model_key,
                model_id=str(active["model_id"]),
                candidate_count=int(active["candidate_count"]),
                taxonomy_items=len(taxonomy.items),
                taxonomy_fingerprint=taxonomy.fingerprint(),
                cache_hit=True,
                model_load_seconds=model_load_seconds,
                candidate_load_seconds=candidate_load_seconds,
                candidate_encode_seconds=None,
                cache_path="postgresql://category_search_profiles_v3",
                backend="pgvector",
                retrieval_mode=mode,
                generation_id=int(active["generation_id"]),
                model_revision=str(active["revision"]),
                profile_count=int(active["profile_count_from_set"]),
                indexed_category_count=int(active["indexed_category_count"]),
                profile_set_id=int(active["profile_set_id"]),
                artifact_kind=str(active["artifact_kind"]),
                profile_schema_version=int(active["profile_schema_version"]),
            )
        else:
            key = cache_key(
                taxonomy.fingerprint(),
                RENDERER_VERSION,
                embedder_metadata,
                [(p.item_id, p.language, p.text, profile_hash(p)) for p in profiles],
            )
            cache_root_value = Path(config.get("cache_dir", ".cache/model-intent"))
            cache_root = cache_root_value if cache_root_value.is_absolute() else project_root / cache_root_value
            matrix_path = cache_root / f"{model_key}-{key}.npz"
            cache_metadata = {
                "key": key,
                "model": embedder_metadata,
                "taxonomy": taxonomy.fingerprint(),
                "renderer": RENDERER_VERSION,
                "rows": len(profiles),
                "dtype": "float32",
            }
            load_started = time.perf_counter()
            vectors = load_matrix(matrix_path, cache_metadata)
            candidate_load_seconds = time.perf_counter() - load_started
            cache_hit = vectors is not None
            candidate_encode_seconds: float | None = None
            if vectors is None:
                encode_started = time.perf_counter()
                vectors = embedder.encode_documents([profile.text for profile in profiles])
                candidate_encode_seconds = time.perf_counter() - encode_started
                save_matrix(matrix_path, vectors, cache_metadata)
            index = SemanticIndex(taxonomy, profiles, vectors, model_key)
            metadata = RuntimeMetadata(
                model_key=model_key,
                model_id=embedder.model_id,
                candidate_count=len(profiles),
                taxonomy_items=len(taxonomy.items),
                taxonomy_fingerprint=taxonomy.fingerprint(),
                cache_hit=cache_hit,
                model_load_seconds=model_load_seconds,
                candidate_load_seconds=candidate_load_seconds,
                candidate_encode_seconds=candidate_encode_seconds,
                cache_path=str(matrix_path),
                backend="numpy",
                retrieval_mode="numpy_exact",
                model_revision=str(embedder_metadata.get("revision") or ""),
                profile_count=len(profiles),
                indexed_category_count=len({item.item_id for item in profiles}),
                artifact_kind="category_profile_v3",
                profile_schema_version=PROFILE_SCHEMA_VERSION,
            )
        return cls(
            config=config,
            taxonomy=taxonomy,
            enrichment=enrichment,
            policy=policy,
            model_key=model_key,
            embedder=embedder,
            index=index,
            metadata=metadata,
            confidence_calibration=confidence_calibration,
            database=database,
        )

    @staticmethod
    def _resolve_profile(config: dict[str, Any], model_key: str) -> ModelProfile:
        if model_key == "fake":
            return ModelProfile(key="fake", model_id="fake")
        profile = next(
            (item for item in profiles_from_config(config, include_stretch=True) if item.key == model_key),
            None,
        )
        if profile is None:
            raise ValueError(f"unknown model profile: {model_key}")
        return profile

    @staticmethod
    def available_models(config_path: str | Path) -> list[dict[str, Any]]:
        config = load_yaml(config_path)
        models = [{"key": "fake", "model_id": "fake", "stretch": False}]
        models.extend(
            {"key": profile.key, "model_id": profile.model_id, "stretch": profile.stretch}
            for profile in profiles_from_config(config, include_stretch=True)
        )
        return models

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        language: str = "vi",
        overrides: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        text = str(text).strip()
        if not text:
            raise ValueError("query must not be empty")
        if language not in {"vi", "en"}:
            raise ValueError("language must be 'vi' or 'en'")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        if isinstance(self.index, PgVectorIndex):
            prediction = self.index.search(text, self.embedder, top_k=max(top_k, 20))
        else:
            with self._lock:
                prediction = self.index.search(text, self.embedder, top_k=max(top_k, 20))
        decision_started = time.perf_counter()
        decision = decide_intent(
            prediction,
            self.taxonomy,
            self.policy,
            self.enrichment,
            language,
            overrides,
            self.confidence_calibration,
        )
        decision_ms = (time.perf_counter() - decision_started) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000
        prediction_payload = prediction_to_dict(prediction)
        prediction_payload["hits"] = prediction_payload["hits"][:top_k]
        runtime_payload = {**asdict(self.metadata), "query_elapsed_ms": elapsed_ms, "decision_ms": decision_ms}
        runtime_payload.update(getattr(self.index, "last_search_metadata", {}))
        return {
            "prediction": prediction_payload,
            "decision": decision_to_dict(decision),
            "runtime": runtime_payload,
        }

    def resolve_selection(
        self,
        *,
        query: str,
        selected_id: str,
        allowed_option_ids: Iterable[str],
        language: str = "vi",
    ) -> dict[str, Any]:
        selected_id = str(selected_id)
        allowed = {str(value) for value in allowed_option_ids}
        if selected_id not in allowed:
            raise ValueError("selected item was not offered by the clarification decision")
        if selected_id not in self.taxonomy.items or self.taxonomy.has_children(selected_id):
            raise ValueError("selected item must be a known leaf taxonomy item")
        item = self.taxonomy.items[selected_id]
        decision = IntentDecision(
            action="accept",
            reason="explicit_user_selection",
            language=language,
            accepted_item_id=item.id,
            accepted_code=item.code,
            policy_version=str(self.policy.get("policy_version", "unknown")),
        )
        return {
            "query": query,
            "decision": decision_to_dict(decision),
            "runtime": asdict(self.metadata),
        }
