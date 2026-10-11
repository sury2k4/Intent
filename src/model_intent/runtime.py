from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .serving_config import (
    E5_DIMENSION,
    E5_MODEL_ID,
    E5_MODEL_KEY,
    E5_MODEL_REVISION,
    configure_runtime,
    load_yaml,
    make_embedder,
    profiles_from_config,
)
from .confidence import load_confidence_calibration
from .decision import (
    decide_intent,
    decision_to_dict,
    load_decision_policy,
    validate_decision_policy,
)
from .db import Database, validate_database_target
from .embedding_sync import (
    profile_content_hash,
    profile_source_fingerprint,
    semantic_model_metadata,
    stable_hash,
)
from .pgvector_index import PgVectorIndex
from .rendering import (
    PROFILE_RENDERER_VERSION,
    PROFILE_SCHEMA_VERSION,
    load_enrichment,
    render_category_profiles,
    validate_enrichment,
)
from .schemas import IntentDecision, ModelProfile, Prediction
from .search_timing import SearchTiming
from .taxonomy import Taxonomy


def prediction_to_dict(prediction: Prediction) -> dict[str, object]:
    data = asdict(prediction)
    data["hits"] = [asdict(hit) for hit in prediction.hits]
    return data


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
    backend: str = "pgvector"
    retrieval_mode: str = "pg_exact"
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
        if model_key != E5_MODEL_KEY:
            raise ValueError(f"category serving supports only {E5_MODEL_KEY}")
        if config.get("retrieval_backend") != "pgvector" or config.get("retrieval_mode", "pg_exact") != "pg_exact":
            raise ValueError("category serving requires exact pgvector retrieval")
        profile = cls._resolve_profile(config, model_key)
        validate_database_target(config)
        configure_runtime(config, local_files_only)
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

        model_started = time.perf_counter()
        embedder = make_embedder(profile, local_files_only)
        model_load_seconds = time.perf_counter() - model_started

        embedder_metadata = embedder.metadata()
        calibration_path = resolve_config_path(config.get("confidence_calibration_path"))
        confidence_calibration = None
        if calibration_path is not None:
            try:
                confidence_calibration = load_confidence_calibration(
                    calibration_path,
                    model_key=model_key,
                    model_id=str(embedder_metadata.get("model_id") or profile.model_id),
                    model_revision=str(embedder_metadata.get("revision") or ""),
                    renderer_version=PROFILE_RENDERER_VERSION,
                    taxonomy_fingerprint=taxonomy.fingerprint(),
                )
            except Exception as exc:
                raise RuntimeError(f"invalid pinned category confidence calibration: {exc}") from exc
        actual_dimension = int(embedder_metadata.get("dimension", getattr(embedder, "dimension", 0)))
        if (actual_dimension != E5_DIMENSION or embedder_metadata.get("model_id") != profile.model_id
                or embedder_metadata.get("revision") != profile.revision):
            raise ValueError("loaded E5 embedder does not match the pinned model, revision, or 768D dimension")
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
        database = Database(
            min_size=int(config.get("db_pool_min_size", 1)),
            max_size=int(config.get("db_pool_max_size", 8)),
            timeout=float(config.get("db_pool_timeout_seconds", 10)),
            statement_timeout_ms=int(config.get("db_statement_timeout_ms", 5000)),
        )
        database.open()
        try:
            load_started = time.perf_counter()
            probe = PgVectorIndex(
                taxonomy, database, model_key=model_key, mode="pg_exact", dimension=actual_dimension,
            )
            active = (
                probe.generation_metadata(generation_id)
                if generation_id is not None
                else probe.active_generation()
            )
            candidate_load_seconds = time.perf_counter() - load_started
            expected_config_hash = stable_hash(semantic_model_metadata(profile))
            if (
                active["taxonomy_fingerprint"] != taxonomy.fingerprint()
                or active["config_hash"] != expected_config_hash
                or active["model_id"] != profile.model_id
                or active["revision"] != profile.revision
                or active["renderer_version"] != PROFILE_RENDERER_VERSION
                or active["artifact_kind"] != "category_profile_v3"
                or active["dimension"] != E5_DIMENSION
                or active["source_fingerprint"] != expected_source_fingerprint
                or active["profile_schema_version"] != PROFILE_SCHEMA_VERSION
                or active["content_set_hash"] != expected_content_set_hash
                or active["candidate_count"] != len(profiles)
                or active["profile_count_from_set"] != len(profiles)
                or active["indexed_category_count"] != len({item.item_id for item in profiles})
            ):
                raise RuntimeError(
                    "active pgvector generation is incompatible with the current model/taxonomy/renderer; "
                    "run embeddings sync"
                )
            index = PgVectorIndex(
                taxonomy, database, model_key=model_key, mode="pg_exact",
                generation_id=int(active["generation_id"]), dimension=actual_dimension,
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
            retrieval_mode="pg_exact",
            generation_id=int(active["generation_id"]),
            model_revision=str(active["revision"]),
            profile_count=int(active["profile_count_from_set"]),
            indexed_category_count=int(active["indexed_category_count"]),
            profile_set_id=int(active["profile_set_id"]),
            artifact_kind=str(active["artifact_kind"]),
            profile_schema_version=int(active["profile_schema_version"]),
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
        if model_key != E5_MODEL_KEY:
            raise ValueError(f"category serving supports only {E5_MODEL_KEY}")
        profile = next(
            (item for item in profiles_from_config(config, include_stretch=True) if item.key == model_key),
            None,
        )
        if profile is None:
            raise ValueError(f"unknown model profile: {model_key}")
        if (profile.model_id != E5_MODEL_ID or profile.revision != E5_MODEL_REVISION
                or profile.expected_dimension != E5_DIMENSION
                or profile.query_prefix != "query: " or profile.document_prefix != "passage: "):
            raise ValueError("category serving requires pinned intfloat/multilingual-e5-base with E5 prefixes at 768D")
        return profile

    @staticmethod
    def available_models(config_path: str | Path) -> list[dict[str, Any]]:
        config = load_yaml(config_path)
        if config.get("retrieval_backend") != "pgvector" or config.get("retrieval_mode", "pg_exact") != "pg_exact":
            return []
        return [
            {"key": profile.key, "model_id": profile.model_id, "stretch": profile.stretch}
            for profile in profiles_from_config(config, include_stretch=True)
            if profile.key == E5_MODEL_KEY and profile.model_id == E5_MODEL_ID
            and profile.revision and profile.expected_dimension == E5_DIMENSION
        ]

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        language: str = "vi",
        overrides: dict[str, float] | None = None,
        timing: SearchTiming | None = None,
    ) -> dict[str, Any]:
        text = str(text).strip()
        if not text:
            raise ValueError("query must not be empty")
        if language not in {"vi", "en"}:
            raise ValueError("language must be 'vi' or 'en'")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        with (timing.measure("category_retrieval") if timing else nullcontext()):
            prediction = self.index.search(text, self.embedder, top_k=max(top_k, 20), timing=timing)
        decision_started = time.perf_counter()
        with (timing.measure("category_decision") if timing else nullcontext()):
            decision = decide_intent(
                prediction,
                self.taxonomy,
                self.policy,
                self.enrichment,
                language,
                overrides,
                self.confidence_calibration,
                timing=timing,
            )
        decision_ms = (time.perf_counter() - decision_started) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000
        with (timing.measure("category_serialize") if timing else nullcontext()):
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
