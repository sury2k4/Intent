from __future__ import annotations

from pathlib import Path
import copy
import math
import threading
from typing import Any, Mapping

from .normalization import fold_vietnamese, normalize_text

from .serving_config import E5_DIMENSION, E5_MODEL_ID, E5_MODEL_KEY, E5_MODEL_REVISION, load_yaml
from .category_resolution import ExplicitCategory, explicit_categories, unique_leaf_name
from .category_query_aliases import CategoryQueryAliases
from .db import Database, validate_database_target
from .poi_properties import PropertyProfiles, load_property_profiles
from .property_schemas import ExtractedProperty, PropertyExtraction
from .query_interpretation import InterpretedQuery, interpret_query
from .query_scope import has_location_request, has_place_seeking_anchor, non_place_reason
from .property_extraction import PropertyExtractor, reconcile_for_category, semantic_property_cue
from .property_semantic import load_property_semantic_corpus
from .property_pg_index import PropertyPgIndex
from .runtime import MatcherRuntime
from .search_timing import SearchTiming

E5_POI_MODEL_KEY = E5_MODEL_KEY
E5_POI_MODEL_ID = E5_MODEL_ID
E5_POI_REVISION = E5_MODEL_REVISION
E5_POI_DIMENSION = E5_DIMENSION

_GENERIC_CATEGORY_WORDS = frozenset({
    "cua", "hang", "nha", "diem", "tram", "cho", "noi", "khu", "dich", "vu", "phong",
    "shop", "store", "station", "place", "facility", "center", "centre", "service",
})


class SearchRuntime:
    """Compose category and property interpretation with pinned pgvector profiles."""

    def __init__(
        self,
        *,
        category_runtime: MatcherRuntime,
        property_extractor: PropertyExtractor,
        property_catalog: Mapping[str, Mapping[str, Any]],
        property_profiles: PropertyProfiles | None = None,
        query_aliases: CategoryQueryAliases | None = None,
        property_semantic_path: Path | None = None,
        property_semantic_backend: str = "pg_exact_v1",
        property_generation_id: int | None = None,
    ) -> None:
        self.category_runtime = category_runtime
        self.property_extractor = property_extractor
        self.property_catalog = dict(property_catalog)
        self.property_profiles = property_profiles
        self.query_aliases = query_aliases
        if property_semantic_backend != "pg_exact_v1":
            raise ValueError("intent analysis requires the pinned pgvector property backend")
        if property_semantic_path is None or property_generation_id is None:
            raise ValueError("pgvector property backend requires a corpus path and pinned generation ID")
        self.property_semantic_path = property_semantic_path
        self.property_semantic_backend = property_semantic_backend
        self.property_generation_id = property_generation_id
        self._semantic_property_index: PropertyPgIndex | None = None
        self._semantic_property_lock = threading.Lock()

    @staticmethod
    def _validate_e5_category_runtime(category_runtime: MatcherRuntime, model_key: str) -> None:
        if model_key != E5_POI_MODEL_KEY:
            return
        metadata = category_runtime.metadata
        dimension = int(category_runtime.embedder.metadata().get("dimension", 0))
        backend_ready = (
            metadata.backend == "pgvector"
            and metadata.retrieval_mode == "pg_exact"
            and getattr(category_runtime.index, "dimension", None) == E5_POI_DIMENSION
            and metadata.generation_id is not None
            and category_runtime.database is not None
        )
        if (
            metadata.model_id != E5_POI_MODEL_ID
            or metadata.model_revision != E5_POI_REVISION
            or dimension != E5_POI_DIMENSION
            or not backend_ready
        ):
            raise RuntimeError("category runtime requires pinned E5-base 768D exact pgvector retrieval")
        if category_runtime.confidence_calibration is None:
            raise RuntimeError("E5 category confidence calibration is missing or incompatible")

    @staticmethod
    def _load_query_aliases(
        config: Mapping[str, Any], config_path: Path, category_runtime: MatcherRuntime,
    ) -> CategoryQueryAliases | None:
        selected = config.get("category_query_aliases_path")
        if not selected:
            return None
        path = Path(selected)
        if not path.is_absolute():
            path = (config_path.parent.parent / path).resolve()
        return CategoryQueryAliases(category_runtime.taxonomy, category_runtime.enrichment, overrides_path=path)

    @classmethod
    def from_intent_config(
        cls,
        *,
        config_path: str | Path,
        taxonomy_path: str | Path,
        model_key: str,
        local_files_only: bool = False,
        generation_id: int | None = None,
    ) -> "SearchRuntime":
        config_path = Path(config_path).resolve()
        config = load_yaml(config_path)
        validate_database_target(config)
        if model_key != E5_POI_MODEL_KEY:
            raise RuntimeError("intent analysis requires pinned E5-base 768D")
        category_runtime = MatcherRuntime.from_config(
            config_path=config_path,
            taxonomy_path=taxonomy_path,
            model_key=model_key,
            local_files_only=local_files_only,
            generation_id=generation_id,
        )
        try:
            cls._validate_e5_category_runtime(category_runtime, model_key)
            property_config = config.get("property_intent") or {}
            property_name = property_config.get("property_path")
            if not property_name:
                raise ValueError("config property_intent.property_path is required")
            property_path = Path(property_name)
            if not property_path.is_absolute():
                property_path = (config_path.parent.parent / property_path).resolve()
            if not property_path.exists():
                raise FileNotFoundError(f"Property catalog not found: {property_path}")
            profiles = load_property_profiles(property_path, category_runtime.taxonomy)
            backend = str(property_config.get("property_semantic_backend", ""))
            corpus_path = property_config.get("property_semantic_profiles_path")
            if corpus_path:
                corpus_path = Path(corpus_path)
                if not corpus_path.is_absolute():
                    corpus_path = (config_path.parent.parent / corpus_path).resolve()
            return cls(
                category_runtime=category_runtime,
                property_extractor=PropertyExtractor(profiles.definitions),
                property_catalog=profiles.definitions,
                property_profiles=profiles,
                query_aliases=cls._load_query_aliases(config, config_path, category_runtime),
                property_semantic_path=corpus_path,
                property_semantic_backend=backend,
                property_generation_id=property_config.get("property_generation_id"),
            )
        except Exception:
            database = getattr(category_runtime, "database", None)
            if database is not None:
                database.close()
            raise

    def _category_query(
        self, candidate: str, *, top_k: int, language: str,
        overrides: dict[str, float] | None, timing: SearchTiming, variant_index: int,
    ) -> dict[str, Any]:
        with timing.measure("category_variant", variant_index=variant_index):
            kwargs = {"top_k": top_k, "language": language, "overrides": overrides}
            if isinstance(self.category_runtime, MatcherRuntime):
                kwargs["timing"] = timing
            return self.category_runtime.query(candidate, **kwargs)

    def _interpret_category(
        self, text: str, *, category_top_k: int, language: str,
        overrides: dict[str, float] | None, timing: SearchTiming,
        focused_only: bool = False,
    ) -> tuple[dict[str, Any], PropertyExtraction]:
        with timing.measure("property_extraction"):
            extraction = self.property_extractor.extract(text)
        with timing.measure("query_interpretation"):
            leading_alias = self.query_aliases.leading_curated_match(text) if self.query_aliases else None
            interpretation = interpret_query(
                text, extraction, self.category_runtime.taxonomy, self.category_runtime.enrichment,
                additional_matches=(leading_alias,) if leading_alias else (),
                short_property_candidate=self.property_extractor.unique_typo_alias,
            )
            extraction = interpretation.extraction
        real_model = getattr(self.category_runtime, "model_key", "fake") != "fake"
        candidates = interpretation.variants
        if real_model and focused_only:
            if interpretation.focused_reason == "no_category_text_after_property_removal":
                candidates = ()
            elif interpretation.focused_is_usable:
                candidates = (interpretation.category_text,)
        if real_model and not candidates:
            category_payload = {
                "prediction": {"query": None, "hits": []},
                "decision": {"action": "reject", "reason": "no_category_phrase", "accepted_item_id": None,
                             "accepted_code": None, "question": None, "options": []},
                "category_resolution": {"source": "no_category_phrase", "semantic_variants": []},
            }
        elif real_model:
            outcomes = [
                self._category_query(
                    candidate, top_k=category_top_k, language=language, overrides=overrides,
                    timing=timing, variant_index=index,
                )
                for index, candidate in enumerate(candidates)
            ]
            with timing.measure("category_resolution"):
                category_payload = self._choose_real_category(
                    text, outcomes, interpretation, language, candidates=candidates,
                )
            category_payload = self._probe_unique_alias(
                text, category_payload, interpretation, language,
                category_top_k=category_top_k, overrides=overrides, timing=timing,
                variant_index=len(outcomes),
            )
            if focused_only:
                with timing.measure("near_tie_clarification"):
                    category_payload = self._clarify_near_tied_intent(
                        text, category_payload, interpretation, language, overrides,
                    )
                category_payload = self._confirm_folded_category_name(
                    text, category_payload, interpretation, language,
                )
        else:
            category_payload = self._category_query(
                text, top_k=category_top_k, language=language, overrides=overrides,
                timing=timing, variant_index=0,
            )
            with timing.measure("category_resolution"):
                category_payload = self._resolve_explicit_category(text, category_payload, language, extraction)
        interpretation_payload = interpretation.as_dict()
        if real_model:
            probe = (category_payload.get("category_resolution") or {}).get("alias_probe") or {}
            canonical_query = probe.get("canonical_query")
            interpretation_payload["semantic_variants"] = [
                *candidates, *([canonical_query] if canonical_query else []),
            ]
            interpretation_payload["pgvector_queries"] = list(interpretation_payload["semantic_variants"])
            if not candidates:
                mode = "no_category_phrase"
            elif focused_only and not interpretation.focused_is_usable and interpretation.category_text != text:
                mode = "original_fallback"
            elif focused_only and interpretation.category_text != text:
                mode = "focused"
            else:
                mode = "original"
            interpretation_payload["vector_query_mode"] = mode
        category_payload["query_interpretation"] = interpretation_payload
        self._attach_confidence_provenance(category_payload)
        return category_payload, extraction

    def _probe_unique_alias(
        self, text: str, payload: dict[str, Any], interpretation: InterpretedQuery,
        language: str, *, category_top_k: int, overrides: dict[str, float] | None,
        timing: SearchTiming, variant_index: int,
    ) -> dict[str, Any]:
        if self.query_aliases is None or interpretation.extraction.failed_open or non_place_reason(text):
            return payload
        alternatives = self.query_aliases.match_alternatives(interpretation.category_text, language=language)
        if alternatives and all(
            self.query_aliases.is_curated(alias.phrase, alias.category_id)
            or self.query_aliases.is_canonical(alias.phrase, alias.category_id)
            for alias in alternatives
        ):
            taxonomy = self.category_runtime.taxonomy
            option_ids = sorted({candidate.category_id for candidate in alternatives}, key=int)
            result = dict(payload)
            result["category_resolution"] = {
                **payload["category_resolution"], "source": "curated_alias_alternatives",
                "original_semantic_decision": dict(payload.get("decision") or {}),
                "alias_options": [{"phrase": alias.phrase, "item_id": alias.category_id} for alias in alternatives],
            }
            result["decision"] = {
                **(payload.get("decision") or {}),
                "action": "clarify", "reason": "multiple_explicit_place_types",
                "accepted_item_id": None, "accepted_code": None,
                "question": "Bạn muốn tìm loại địa điểm nào?" if language == "vi" else "Which type of place are you looking for?",
                "options": [
                    {"item_id": item_id, "code": taxonomy.items[item_id].code,
                     "label": taxonomy.items[item_id].names.get(language) or taxonomy.items[item_id].names.get("vi") or taxonomy.items[item_id].code,
                     "hint": "curated category phrase", "score": 0.0, "relation": "explicit_alternative"}
                    for item_id in option_ids
                ],
            }
            return result
        alias = self.query_aliases.match_whole_phrase(interpretation.category_text, language=language)
        corrected = False
        if alias is None:
            alias = self.query_aliases.match_correction(interpretation.category_text, language=language)
            corrected = alias is not None
        if alias is None:
            return payload
        mentioned_ids = {match.item_id for match in interpretation.category_matches}
        if mentioned_ids and mentioned_ids != {alias.category_id}:
            return payload
        decision = payload.get("decision") or {}
        original_id = str(decision["accepted_item_id"]) if decision.get("accepted_item_id") is not None else None
        curated = self.query_aliases.is_curated(alias.phrase, alias.category_id)
        if corrected and not curated:
            return payload
        if original_id == alias.category_id or decision.get("action") == "clarify":
            return payload
        if decision.get("action") == "reject" and decision.get("reason") not in {
            "low_calibrated_confidence", "insufficient_calibrated_evidence",
        }:
            return payload
        probe = self._category_query(
            alias.canonical_name, top_k=category_top_k, language=language,
            overrides=overrides, timing=timing, variant_index=variant_index,
        )
        probe = self._guard_real_category(
            text, probe, language, interpretation.extraction, alias.canonical_name,
            matches=interpretation.category_matches, interpretation=interpretation,
        )
        hits = (probe.get("prediction") or {}).get("hits") or []
        probe_decision = probe.get("decision") or {}
        confirmed = (
            probe_decision.get("action") == "accept"
            and str(probe_decision.get("accepted_item_id")) == alias.category_id
            and hits and str(hits[0].get("item_id")) == alias.category_id
        )
        alias_trace = {
            "phrase": alias.phrase, "candidate_id": alias.category_id,
            "canonical_query": alias.canonical_name,
            "semantic_top_item_id": str(hits[0].get("item_id")) if hits else None,
            "semantic_action": probe_decision.get("action"),
            "corrected_request": corrected,
            "corroborated": bool(confirmed),
        }
        if not confirmed:
            if curated and original_id is not None and original_id != alias.category_id:
                result = dict(payload)
                result["decision"] = {
                    **decision, "action": "reject", "reason": "alias_unconfirmed_conflict",
                    "accepted_item_id": None, "accepted_code": None, "question": None, "options": [],
                }
                result["category_resolution"] = {
                    **payload["category_resolution"], "source": "alias_unconfirmed_conflict", "alias_probe": alias_trace,
                    "original_semantic_decision": dict(decision),
                }
                return result
            payload["category_resolution"]["alias_probe"] = alias_trace
            return payload
        original_hits = (payload.get("prediction") or {}).get("hits") or []
        original_top_id = str(original_hits[0].get("item_id")) if original_hits else None
        if original_id is None and original_top_id != alias.category_id and not curated:
            # Existing enrichment can describe adjacent services (e.g. an emergency
            # department is not necessarily a hospital). Never override a mismatched
            # raw vector hit without an individually curated equivalent place name.
            alias_trace["acceptance_blocked"] = "uncurated_alias_disagrees_with_original_e5"
            payload["category_resolution"]["alias_probe"] = alias_trace
            return payload
        if original_id is not None and original_id != alias.category_id:
            result = dict(payload)
            result["category_resolution"] = {
                **payload["category_resolution"], "source": "alias_semantic_conflict", "alias_probe": alias_trace,
            }
            taxonomy = self.category_runtime.taxonomy
            candidates = (original_id, alias.category_id)
            result["decision"] = {
                **decision, "action": "clarify", "reason": "alias_semantic_conflict",
                "accepted_item_id": None, "accepted_code": None,
                "question": "Bạn muốn tìm loại địa điểm nào?" if language == "vi" else "Which type of place are you looking for?",
                "options": [
                    {"item_id": item_id, "code": taxonomy.items[item_id].code,
                     "label": taxonomy.items[item_id].names.get(language) or taxonomy.items[item_id].names.get("vi") or taxonomy.items[item_id].code,
                     "hint": "original E5 prediction" if item_id == original_id else "curated alias and canonical E5 probe",
                     "score": 0.0, "relation": "alias_semantic_conflict"}
                    for item_id in candidates
                ],
            }
            return result
        result = dict(probe)
        result["category_resolution"] = {
            **probe["category_resolution"], "source": "alias_canonical_probe",
            "alias_probe": alias_trace,
            "original_semantic_decision": dict(decision),
            "original_semantic_top_id": (
                str(payload["prediction"]["hits"][0]["item_id"])
                if (payload.get("prediction") or {}).get("hits") else None
            ),
            "original_semantic_variants": payload["category_resolution"].get("semantic_variants", []),
        }
        return result

    def _clarify_near_tied_intent(
        self, text: str, payload: dict[str, Any], interpretation: InterpretedQuery,
        language: str, overrides: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        decision = payload.get("decision") or {}
        if (
            decision.get("action") != "reject"
            or decision.get("reason") != "insufficient_calibrated_evidence"
            or interpretation.extraction.failed_open
            or interpretation.category_matches
            or non_place_reason(text)
        ):
            return payload
        calibration = getattr(self.category_runtime, "confidence_calibration", None) or {}
        action_thresholds = calibration.get("action_thresholds") or {}
        policy = getattr(self.category_runtime, "policy", None) or {}
        model_key = getattr(self.category_runtime, "model_key", "")
        limits = {
            **policy.get("thresholds", {}).get("default", {}),
            **policy.get("thresholds", {}).get(model_key, {}),
            **(overrides or {}),
        }
        try:
            min_domain = float(action_thresholds["clarify_min_in_domain_confidence"])
            min_top = float(action_thresholds["reject_min_top_confidence"])
            min_cosine = float(limits["clarify_min_score"])
            max_gap = min(float(limits["coherent_cluster_delta"]), float(limits["ambiguity_delta"]))
            domain = float(decision["in_domain_confidence"])
            top_confidence = float(decision["calibrated_confidence"])
            max_options = int(policy.get("max_options", 4))
        except (KeyError, TypeError, ValueError):
            return payload
        if (
            not all(math.isfinite(value) for value in (min_domain, min_top, min_cosine, max_gap, domain, top_confidence))
            or max_gap <= 0 or max_options < 2
            or domain < min_domain or top_confidence < min_top
        ):
            return payload
        hits = (payload.get("prediction") or {}).get("hits") or []
        if len(hits) < 2:
            return payload
        taxonomy = self.category_runtime.taxonomy
        first_id = str(hits[0].get("item_id"))
        if first_id not in taxonomy.items or taxonomy.has_children(first_id):
            return payload
        try:
            first_score = float(hits[0]["score"])
        except (KeyError, TypeError, ValueError):
            return payload
        if not math.isfinite(first_score) or first_score < min_cosine:
            return payload

        def type_words(item_id: str) -> set[str]:
            words: set[str] = set()
            for name in taxonomy.items[item_id].names.values():
                words.update(normalize_text(fold_vietnamese(name)).split())
            return {word for word in words if len(word) >= 3 and word not in _GENERIC_CATEGORY_WORDS}

        query_words = set(normalize_text(fold_vietnamese(interpretation.category_text)).split())
        first_words = type_words(first_id)
        close: list[tuple[dict[str, Any], float]] = []
        seen: set[str] = set()
        for hit in hits:
            item_id = str(hit.get("item_id"))
            if item_id in seen or item_id not in taxonomy.items or taxonomy.has_children(item_id):
                continue
            try:
                score = float(hit["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(score) or score < min_cosine or not 0 <= first_score - score <= max_gap:
                continue
            # A small vector gap alone does not make two place types valid alternatives.
            if (
                item_id != first_id
                and taxonomy.relation(first_id, item_id) != "sibling"
                and not (query_words & first_words & type_words(item_id))
            ):
                continue
            seen.add(item_id)
            close.append((hit, score))
            if len(close) == max_options:
                break
        if len(close) < 2 or str(close[0][0]["item_id"]) != first_id:
            return payload
        options = []
        for hit, score in close:
            item_id = str(hit["item_id"])
            item = taxonomy.items[item_id]
            options.append({
                "item_id": item_id, "code": item.code,
                "label": item.names.get(language) or item.names.get("vi") or item.code,
                "hint": " > ".join(taxonomy.path_names(item_id, language)),
                "score": score, "relation": taxonomy.relation(first_id, item_id),
            })
        result = dict(payload)
        result["decision"] = {
            **decision, "action": "clarify", "reason": "near_tied_category_candidates",
            "accepted_item_id": None, "accepted_code": None,
            "question": (
                "Các loại địa điểm có điểm E5 gần nhau. Bạn muốn tìm loại nào?"
                if language == "vi" else "These place types have similar E5 scores. Which one do you mean?"
            ),
            "options": options,
        }
        result["category_resolution"] = {
            **(payload.get("category_resolution") or {}),
            "source": "near_tie_clarification",
            "original_semantic_decision": dict(decision),
            "near_tie": {"max_gap": max_gap, "observed_gap": first_score - close[1][1]},
        }
        return result

    def _confirm_folded_category_name(
        self, text: str, payload: dict[str, Any], interpretation: InterpretedQuery, language: str,
    ) -> dict[str, Any]:
        decision = payload.get("decision") or {}
        if (
            decision.get("action") != "reject"
            or decision.get("reason") not in {"low_calibrated_confidence", "insufficient_calibrated_evidence"}
            or len(interpretation.category_matches) != 1
            or interpretation.extraction.failed_open
            or non_place_reason(text)
        ):
            return payload
        match = interpretation.category_matches[0]
        taxonomy = self.category_runtime.taxonomy
        item = taxonomy.items.get(match.item_id)
        if item is None or taxonomy.has_children(item.id):
            return payload
        evidence = normalize_text(match.evidence)
        folded_evidence = normalize_text(fold_vietnamese(match.evidence))
        if not any(
            evidence != normalize_text(name)
            and folded_evidence == normalize_text(fold_vietnamese(name))
            for name in item.names.values()
        ):
            return payload
        name = item.names.get(language) or item.names.get("vi") or item.code
        result = dict(payload)
        result["decision"] = {
            **decision, "action": "clarify", "reason": "folded_category_name_confirmation",
            "accepted_item_id": None, "accepted_code": None,
            "question": (
                f"Bạn muốn tìm '{name}' phải không? Cách viết trong query chưa được E5 xác nhận."
                if language == "vi" else f"Did you mean '{name}'? E5 did not confirm the spelling in your query."
            ),
            "options": [{"item_id": item.id, "code": item.code, "label": name,
                         "hint": "accent-folded name match, not E5 confirmation", "score": None,
                         "relation": "folded_name_candidate"}],
        }
        result["category_resolution"] = {
            **(payload.get("category_resolution") or {}),
            "source": "folded_name_confirmation",
            "original_semantic_decision": dict(decision),
            "folded_name_candidate": {"evidence": match.evidence, "item_id": item.id, "canonical_name": name},
        }
        return result

    @staticmethod
    def _attach_confidence_provenance(
        category_payload: dict[str, Any], *, decision_source: str | None = None,
    ) -> None:
        decision = category_payload.get("decision") or {}
        prediction = category_payload.get("prediction") or {}
        hits = prediction.get("hits") or []
        top_id = str(hits[0].get("item_id")) if hits and hits[0].get("item_id") is not None else None
        resolution = category_payload.get("category_resolution") or {}
        selected_id = decision.get("accepted_item_id")
        source = decision_source or resolution.get("source") or (
            "explicit_user_selection" if decision.get("reason") == "explicit_user_selection" else "semantic_model"
        )
        if source == "semantic_model" and selected_id is not None and top_id is not None and str(selected_id) != top_id:
            source = "policy_parent_to_leaf"
        category_payload["confidence_assessment"] = {
            "target": decision.get("confidence_target", "none"),
            "quality": decision.get("confidence_quality", "none"),
            "semantic_top_item_id": top_id,
            "semantic_top_cosine": hits[0].get("score") if hits else None,
            "semantic_variant_query": resolution.get("semantic_query") or prediction.get("query"),
            "final_decision_source": source,
            "final_action": decision.get("action"),
            "final_accepted_item_id": str(selected_id) if selected_id is not None else None,
            "is_probability_of_final_category": False,
        }

    def analyze_intent(
        self, text: str, *, category_top_k: int = 5, language: str = "vi",
        overrides: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        timing = SearchTiming("intent_analysis")
        category_payload, extraction = self._interpret_category(
            text, category_top_k=category_top_k, language=language,
            overrides=overrides, timing=timing, focused_only=True,
        )
        return self._analysis_from_category(text, category_payload, extraction, timing)

    def _semantic_property_suggestions(
        self, interpretation: Mapping[str, Any], decision: Mapping[str, Any], timing: SearchTiming,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        spans = interpretation.get("unresolved_property_clauses") or []
        if not spans or not isinstance(self.category_runtime, MatcherRuntime):
            return [], {"status": "skipped"}
        if non_place_reason(str(interpretation.get("raw_query", ""))):
            return [], {"status": "skipped_non_place"}
        with timing.measure("property_semantic_suggestions", span_count=len(spans)):
            try:
                if self._semantic_property_index is None:
                    with self._semantic_property_lock:
                        if self._semantic_property_index is None:
                            category = self.category_runtime
                            encode_lock = getattr(category.index, "_embedding_lock", category._lock)
                            tokenizer = getattr(getattr(category.embedder, "model", None), "tokenizer", None)
                            count_tokens = (
                                (lambda value: len(tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"]))
                                if tokenizer is not None else None
                            )
                            corpus = load_property_semantic_corpus(
                                self.property_semantic_path, self.property_catalog,
                                category.taxonomy, category.embedder.metadata(), token_counter=count_tokens,
                            )
                            if category.database is None:
                                raise RuntimeError("property pgvector backend needs the unified DB")
                            index = PropertyPgIndex(
                                category.database, corpus, self.property_catalog, category.embedder,
                                encode_lock, generation_id=int(self.property_generation_id),
                            )
                            self._semantic_property_index = index
                index = self._semantic_property_index
                accepted_id = (
                    str(decision["accepted_item_id"])
                    if decision.get("action") == "accept" and decision.get("accepted_item_id") else None
                )
                suggestions = index.suggest(
                    spans, category_id=accepted_id,
                    taxonomy=self.category_runtime.taxonomy, profiles=self.property_profiles,
                    timing=timing,
                )
            except Exception as exc:
                return [], {"status": "unavailable", "error_type": type(exc).__name__}
        return suggestions, {
            "status": "suggestions_only", "model_key": self.category_runtime.model_key,
            "backend": self.property_semantic_backend,
            "generation_id": index.generation_id,
            "profile_count": index.profile_count, "profile_fingerprint": index.fingerprint,
            "cosine_floor_for_display": 0.80,
            "is_calibrated_probability": False, "auto_applied": False,
        }

    @staticmethod
    def _refresh_property_candidates(result: dict[str, Any]) -> None:
        properties = (result.get("property_extraction") or {}).get("properties") or []
        parts = result["query_parts"]
        parts["filter_candidates"] = [
            row for row in properties
            if row["applied"] and row["mode"] in {"required", "excluded"}
            and row["property_id"] != "radius_km"
        ]
        parts["preference_candidates"] = [
            row for row in properties
            if row["applied"] and row["mode"] == "preferred"
            and row["property_id"] != "radius_km"
        ]

    def _analysis_from_category(
        self, text: str, category_payload: Mapping[str, Any],
        extraction: PropertyExtraction, timing: SearchTiming,
    ) -> dict[str, Any]:
        decision = category_payload.get("decision") or {}
        if decision.get("action") == "accept" and decision.get("accepted_item_id"):
            item = self.category_runtime.taxonomy.items[str(decision["accepted_item_id"])]
            with timing.measure("property_reconcile"):
                extraction = reconcile_for_category(
                    extraction,
                    category_code=item.code,
                    property_catalog=self.property_catalog,
                    category_id=item.id,
                    taxonomy=self.category_runtime.taxonomy,
                    profiles=self.property_profiles,
                )
        interpretation = category_payload.get("query_interpretation") or {}
        suggestions, semantic_meta = self._semantic_property_suggestions(interpretation, decision, timing)
        with timing.measure("response_serialize"):
            property_data = extraction.as_dict()
            result = {
                "query": text,
                "intent": dict(category_payload),
                "property_extraction": property_data,
                "query_parts": {
                    "category_text": None if interpretation.get("vector_query_mode") == "no_category_phrase"
                    else interpretation.get("category_text", text),
                    "category_mentions": interpretation.get("category_matches", []),
                    "category_boundary": interpretation.get("category_boundary"),
                    "unresolved_property_clauses": interpretation.get("unresolved_property_clauses", []),
                    "semantic_property_suggestions": suggestions,
                    "semantic_property_metadata": semantic_meta,
                    "pgvector_queries": interpretation.get("pgvector_queries", []),
                    "vector_query_mode": interpretation.get("vector_query_mode", "unknown"),
                    "applicability": "checked" if decision.get("action") == "accept" else "pending_category",
                },
            }
        self._refresh_property_candidates(result)
        result["timing"] = timing.as_dict()
        return result

    def confirm_semantic_property(
        self, result: Mapping[str, Any], *, span_index: int, property_id: str, mode: str,
    ) -> dict[str, Any]:
        """Accept a displayed suggestion only after an explicit property and mode choice."""
        if mode not in {"required", "preferred", "excluded"}:
            raise ValueError("choose required, preferred, or excluded for the property")
        parts = result.get("query_parts") or {}
        suggestions = parts.get("semantic_property_suggestions") or []
        if not isinstance(span_index, int) or not 0 <= span_index < len(suggestions):
            raise ValueError("property span was not offered")
        suggestion = suggestions[span_index]
        if suggestion.get("status") != "needs_confirmation" or property_id not in {
            row["property_id"] for row in suggestion.get("candidates") or [] if row.get("applicable") is not False
        }:
            raise ValueError("property candidate was not offered or is inapplicable")
        definition = self.property_catalog.get(property_id)
        if not definition or definition.get("queryable", True) is False or definition.get("value_type") not in {
            "boolean", "enum", "ordinal",
        }:
            raise ValueError("property is not a supported filter candidate")
        current = (result.get("property_extraction") or {}).get("properties") or []
        if any(row.get("property_id") == property_id for row in current):
            raise ValueError("property already exists in the extracted constraints")
        span = suggestion["span"]
        start, end = int(span["start_offset"]), int(span["end_offset"])
        query = str(result.get("query", ""))
        if not 0 <= start < end <= len(query) or query[start:end] != span["text"]:
            raise ValueError("property evidence does not match the original query")
        accepted_id = (result.get("intent") or {}).get("decision", {}).get("accepted_item_id")
        if accepted_id and self.property_profiles and property_id not in self.property_profiles.applicable_ids(
            self.category_runtime.taxonomy, str(accepted_id),
        ):
            raise ValueError("property does not apply to the selected category")
        value_type = definition["value_type"]
        operator = "gte" if value_type == "ordinal" else "eq"
        value: Any = "high" if value_type == "ordinal" else "yes" if value_type == "enum" else True
        cue_mode, negative_preference = semantic_property_cue(span["text"])
        if cue_mode == "not_requested":
            raise ValueError("the query says this property is not requested")
        if mode == "preferred" and negative_preference:
            if value_type != "boolean":
                raise ValueError("negative preference cannot be represented for this property")
            value = False
        extracted = ExtractedProperty(
            property_id=property_id, mode=mode, operator=operator, value=value,
            evidence_text=span["text"], start_offset=start, end_offset=end,
            cue="semantic_user_confirmed", confidence=0.0,
        ).as_dict()
        updated = copy.deepcopy(result)
        updated["property_extraction"]["properties"].append(extracted)
        updated_parts = updated["query_parts"]
        self._refresh_property_candidates(updated)
        updated_parts["semantic_property_suggestions"][span_index]["status"] = "confirmed"
        updated_parts.setdefault("confirmed_semantic_choices", []).append({
            "start_offset": start, "end_offset": end, "property_id": property_id, "mode": mode,
        })
        return updated

    def _choose_real_category(
        self,
        text: str,
        outcomes: list[dict[str, Any]],
        interpretation: InterpretedQuery,
        language: str,
        *,
        candidates: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        candidates = candidates or interpretation.variants
        chosen = outcomes[0]
        conflicting_ids: tuple[str, str] | None = None
        if len(outcomes) > 1:
            focused, raw = outcomes
            focused_decision = focused["decision"]
            raw_decision = raw["decision"]
            focused_id = focused_decision.get("accepted_item_id") if focused_decision["action"] == "accept" else None
            raw_id = raw_decision.get("accepted_item_id") if raw_decision["action"] == "accept" else None
            anchored_ids = {match.item_id for match in interpretation.category_matches}
            if focused_id and raw_id and focused_id != raw_id:
                if anchored_ids != {str(focused_id)}:
                    conflicting_ids = str(focused_id), str(raw_id)
            elif not focused_id and raw_id:
                chosen = raw
            elif not focused_id and focused_decision["action"] == "reject" and raw_decision["action"] == "clarify":
                chosen = raw
        focused_text = str(chosen.get("prediction", {}).get("query") or candidates[0])
        payload = self._guard_real_category(
            text, chosen, language, interpretation.extraction, focused_text,
            matches=interpretation.category_matches,
            interpretation=interpretation,
        )
        resolution = payload["category_resolution"]
        resolution["semantic_variants"] = [
            {"query": variant, "action": outcome["decision"]["action"],
             "reason": outcome["decision"].get("reason"),
             "accepted_item_id": outcome["decision"].get("accepted_item_id"),
             "top1_score": outcome.get("prediction", {}).get("top1_score"),
             "margin": outcome.get("prediction", {}).get("margin"),
             "calibrated_confidence": outcome["decision"].get("calibrated_confidence"),
             "in_domain_confidence": outcome["decision"].get("in_domain_confidence"),
             "hits": [
                 {"item_id": hit.get("item_id"), "code": hit.get("code"), "score": hit.get("score")}
                 for hit in (outcome.get("prediction", {}).get("hits") or [])[:3]
             ]}
            for variant, outcome in zip(candidates, outcomes)
        ]
        calibration = getattr(self.category_runtime, "confidence_calibration", None) or {}
        if calibration.get("action_thresholds"):
            resolution["action_thresholds"] = dict(calibration["action_thresholds"])
        if conflicting_ids and payload["decision"]["action"] == "accept":
            taxonomy = self.category_runtime.taxonomy
            options = []
            for item_id in sorted(conflicting_ids, key=int):
                item = taxonomy.items[item_id]
                options.append({
                    "item_id": item_id, "code": item.code,
                    "label": item.names.get(language) or item.names.get("vi") or item.code,
                    "hint": "E5 category variant", "score": 0.0, "relation": "semantic_disagreement",
                })
            payload["decision"] = {
                **payload["decision"], "action": "clarify", "reason": "semantic_variants_disagree",
                "accepted_item_id": None, "accepted_code": None,
                "question": "Bạn muốn tìm loại địa điểm nào?" if language == "vi" else "Which type of place are you looking for?",
                "options": options,
            }
            resolution["source"] = "semantic_variants_disagree"
        if resolution.get("source") == "literal_leaf_name" and payload["decision"]["action"] == "accept":
            variant_top_ids = {
                str(outcome["prediction"]["hits"][0]["item_id"])
                for outcome in outcomes if (outcome.get("prediction") or {}).get("hits")
            }
            if len(variant_top_ids) > 1:
                taxonomy = self.category_runtime.taxonomy
                options = [
                    {"item_id": item_id, "code": taxonomy.items[item_id].code,
                     "label": taxonomy.items[item_id].names.get(language) or taxonomy.items[item_id].names.get("vi") or taxonomy.items[item_id].code,
                     "hint": "E5 top candidate across query variants", "score": 0.0, "relation": "semantic_disagreement"}
                    for item_id in sorted(variant_top_ids, key=int)
                    if item_id in taxonomy.items and not taxonomy.has_children(item_id)
                ][:4]
                payload["decision"] = {
                    **payload["decision"], "action": "clarify" if len(options) >= 2 else "reject",
                    "reason": "semantic_variants_disagree", "accepted_item_id": None, "accepted_code": None,
                    "question": ("Bạn muốn tìm loại địa điểm nào?" if language == "vi" else "Which type of place are you looking for?") if len(options) >= 2 else None,
                    "options": options if len(options) >= 2 else [],
                }
                resolution["source"] = "semantic_variants_disagree"
        return payload

    def _guard_real_category(
        self,
        text: str,
        category_payload: Mapping[str, Any],
        language: str,
        extraction: PropertyExtraction,
        semantic_query: str,
        *,
        matches: tuple[ExplicitCategory, ...] | None = None,
        interpretation: InterpretedQuery | None = None,
    ) -> dict[str, Any]:
        """Guard semantic decisions and admit unique literal names with matching top-1."""
        taxonomy = self.category_runtime.taxonomy
        if matches is None:
            spans = tuple(
                (row.start_offset, row.end_offset)
                for row in extraction.properties
                if row.start_offset is not None and row.end_offset is not None
                and row.ignored_reason != "category_component"
            )
            matches = explicit_categories(text, taxonomy, self.category_runtime.enrichment, property_spans=spans)
        payload = dict(category_payload)
        decision = dict(payload.get("decision") or {})
        payload["category_resolution"] = {
            "source": "semantic_model",
            "semantic_query": semantic_query,
            "explicit_matches": [{"item_id": match.item_id, "evidence": match.evidence} for match in matches],
        }
        normalized = normalize_text(fold_vietnamese(text))
        scope_reason = non_place_reason(text)
        if scope_reason == "digital_product_not_place_search" and len(matches) == 1:
            target = matches[0]
            if (
                has_place_seeking_anchor(text, target.start_offset)
                and non_place_reason(text[:target.start_offset]) != "digital_product_not_place_search"
            ):
                # "find a pharmacy with an app" still targets the venue;
                # "find software for a pharmacy" targets software.
                scope_reason = None
        if scope_reason:
            decision.update({
                "action": "reject", "reason": scope_reason,
                "accepted_item_id": None, "accepted_code": None, "question": None, "options": [],
            })
            payload["decision"] = decision
            payload["category_resolution"]["source"] = "informational_veto" if scope_reason == "informational_not_place_search" else "non_place_veto"
            return payload
        if decision.get("action") == "reject" and interpretation is not None and not extraction.failed_open:
            literal = unique_leaf_name(interpretation.category_text, taxonomy, matches)
            hits = (payload.get("prediction") or {}).get("hits") or []
            if (
                literal is not None
                and interpretation.focused_is_usable
                and decision.get("reason") in {"low_calibrated_confidence", "insufficient_calibrated_evidence"}
                and hits
            ):
                item = taxonomy.items[literal.item_id]
                top_id = str(hits[0].get("item_id"))
                if top_id == literal.item_id:
                    payload["category_resolution"].update({
                        "source": "literal_leaf_name",
                        "semantic_decision": {"action": decision["action"], "reason": decision["reason"]},
                        "literal_evidence": literal.evidence,
                    })
                    decision.update({
                        "action": "accept", "reason": "explicit_leaf_name_evidence",
                        "accepted_item_id": item.id, "accepted_code": item.code,
                        "question": None, "options": [],
                    })
                    payload["decision"] = decision
                elif (
                    top_id in taxonomy.items and not taxonomy.has_children(top_id)
                    and taxonomy.items[top_id].parent_id == item.parent_id
                    and any(str(hit.get("item_id")) == item.id for hit in hits)
                ):
                    semantic_reason = decision["reason"]
                    top_item = taxonomy.items[top_id]
                    decision.update({
                        "action": "clarify", "reason": "literal_semantic_conflict",
                        "accepted_item_id": None, "accepted_code": None,
                        "question": "Bạn muốn tìm loại địa điểm nào?" if language == "vi" else "Which type of place are you looking for?",
                        "options": [
                            {"item_id": candidate.id, "code": candidate.code,
                             "label": candidate.names.get(language) or candidate.names.get("vi") or candidate.code,
                             "hint": "explicit leaf name" if candidate.id == item.id else "E5 top candidate",
                             "score": next((hit.get("score", 0.0) for hit in hits if str(hit.get("item_id")) == candidate.id), 0.0),
                             "relation": "explicit_mention" if candidate.id == item.id else "semantic_prediction"}
                            for candidate in (item, top_item)
                        ],
                    })
                    payload["decision"] = decision
                    payload["category_resolution"].update({
                        "source": "literal_semantic_conflict",
                        "semantic_decision": {"action": "reject", "reason": semantic_reason},
                    })
        if (
            decision.get("action") == "reject" and len(matches) == 1 and not extraction.failed_open
            and decision.get("reason") in {"low_calibrated_confidence", "insufficient_calibrated_evidence"}
        ):
            match = matches[0]
            hits = (payload.get("prediction") or {}).get("hits") or []
            # Use the existing category cluster separation as a conservative
            # floor; a near-tie must never be promoted by a literal substring.
            min_margin = float(
                (getattr(self.category_runtime, "policy", {}) or {}).get("thresholds", {})
                .get("e5-base-multilingual", {}).get("coherent_cluster_delta", 0.012)
            )
            margin = float(hits[0]["score"]) - float(hits[1]["score"]) if len(hits) >= 2 else float("inf")
            if (
                hits and margin >= min_margin and str(hits[0].get("item_id")) == match.item_id
                and has_place_seeking_anchor(text, match.start_offset)
                and has_location_request(text, match.end_offset)
                and match.item_id in taxonomy.items and not taxonomy.has_children(match.item_id)
            ):
                item = taxonomy.items[match.item_id]
                payload["category_resolution"].update({
                    "source": "anchored_place_name", "literal_evidence": match.evidence,
                    "semantic_decision": {"action": decision["action"], "reason": decision["reason"]},
                })
                decision.update({
                    "action": "accept", "reason": "anchored_place_name_evidence",
                    "accepted_item_id": item.id, "accepted_code": item.code,
                    "question": None, "options": [],
                })
                payload["decision"] = decision
        if decision.get("action") != "accept":
            return payload
        accepted_id = str(decision.get("accepted_item_id"))
        normalized_tokens = set(normalized.split())
        brand_ancestors = [
            taxonomy.items[ancestor_id].code for ancestor_id in taxonomy.ancestors(accepted_id)
            if taxonomy.items[ancestor_id].code in {"vinfast", "vingroup"}
        ]
        category_code = taxonomy.items[accepted_id].code
        named_branded_leaf = category_code.startswith("vin") and normalize_text(category_code) in normalized_tokens
        if brand_ancestors and not named_branded_leaf and any(brand not in normalized_tokens for brand in brand_ancestors):
            decision.update({
                "action": "reject", "reason": "unsupported_branded_category",
                "accepted_item_id": None, "accepted_code": None,
                "question": None, "options": [],
            })
            payload["decision"] = decision
            payload["category_resolution"].update({
                "source": "unsupported_category",
                "unavailable_category_id": accepted_id,
                "missing_brand": brand_ancestors,
            })
            return payload
        match_ids = {match.item_id for match in matches}
        if len(match_ids) <= 1 and (not match_ids or accepted_id in match_ids):
            return payload
        option_ids = sorted(match_ids | {accepted_id}, key=int)
        decision.update({
            "action": "clarify", "reason": "semantic_explicit_conflict" if len(match_ids) == 1 else "multiple_explicit_place_types",
            "accepted_item_id": None, "accepted_code": None,
            "question": "Bạn muốn tìm loại địa điểm nào?" if language == "vi" else "Which type of place are you looking for?",
            "options": [
                {
                    "item_id": item_id,
                    "code": taxonomy.items[item_id].code,
                    "label": taxonomy.items[item_id].names.get(language) or taxonomy.items[item_id].names.get("vi") or taxonomy.items[item_id].code,
                    "hint": next((match.evidence for match in matches if match.item_id == item_id), "semantic prediction"),
                    "score": next((hit.get("score", 0.0) for hit in (payload.get("prediction") or {}).get("hits", []) if hit.get("item_id") == item_id), 0.0),
                    "relation": "explicit_mention" if item_id in match_ids else "semantic_prediction",
                }
                for item_id in option_ids[:4]
            ],
        })
        payload["decision"] = decision
        payload["category_resolution"]["source"] = "semantic_guard_clarify"
        return payload

    def _resolve_explicit_category(
        self,
        text: str,
        category_payload: Mapping[str, Any],
        language: str,
        extraction: PropertyExtraction,
    ) -> dict[str, Any]:
        taxonomy = self.category_runtime.taxonomy
        property_spans = tuple(
            (item.start_offset, item.end_offset)
            for item in extraction.properties
            if item.start_offset is not None and item.end_offset is not None
            and item.ignored_reason != "category_component"
        )
        matches = explicit_categories(text, taxonomy, self.category_runtime.enrichment, property_spans=property_spans)
        payload = dict(category_payload)
        decision = dict(payload.get("decision") or {})
        scope_reason = non_place_reason(text)
        if scope_reason:
            decision.update({
                "action": "reject", "reason": scope_reason,
                "accepted_item_id": None, "accepted_code": None,
                "question": None, "options": [],
            })
            payload["decision"] = decision
            payload["category_resolution"] = {"source": "informational_query" if scope_reason == "informational_not_place_search" else "non_place_veto", "explicit_matches": []}
            return payload
        if not matches:
            payload["category_resolution"] = {"source": "semantic", "explicit_matches": []}
            return payload

        payload["category_resolution"] = {
            "source": "explicit_phrase" if len(matches) == 1 else "ambiguous_explicit_phrases",
            "explicit_matches": [
                {"item_id": match.item_id, "evidence": match.evidence}
                for match in matches
            ],
        }
        decision = dict(payload.get("decision") or {})
        if len(matches) == 1:
            item_id = matches[0].item_id
            if decision.get("action") == "accept":
                accepted = str(decision.get("accepted_item_id"))
                if accepted == item_id:
                    payload["category_resolution"]["source"] = "semantic_accept"
                    return payload
                matches = (
                    matches[0],
                    ExplicitCategory(accepted, "semantic prediction"),
                )
                payload["category_resolution"]["source"] = "semantic_explicit_conflict"
            else:
                item = taxonomy.items[item_id]
                decision.update({
                    "action": "accept",
                    "reason": "explicit_place_type_phrase",
                    "accepted_item_id": item_id,
                    "accepted_code": item.code,
                    "question": None,
                    "options": [],
                })
                payload["decision"] = decision
                return payload

        options = []
        for match in matches[:4]:
            item = taxonomy.items[match.item_id]
            options.append({
                "item_id": item.id,
                "code": item.code,
                "label": item.names.get(language) or item.names.get("vi") or item.code,
                "hint": match.evidence,
                "score": 1.0,
                "relation": "explicit_mention",
            })
        decision.update({
            "action": "clarify",
            "reason": "multiple_explicit_place_types",
            "accepted_item_id": None,
            "accepted_code": None,
            "question": "Bạn muốn tìm loại địa điểm nào?" if language == "vi" else "Which type of place are you looking for?",
            "options": options,
        })
        payload["decision"] = decision
        return payload

    def _selected_category(
        self, *, query: str, selected_id: str,
        allowed_option_ids: list[str] | tuple[str, ...], language: str,
        timing: SearchTiming,
    ) -> tuple[dict[str, Any], PropertyExtraction]:
        with timing.measure("category_selection"):
            category_payload = self.category_runtime.resolve_selection(
                query=query,
                selected_id=selected_id,
                allowed_option_ids=allowed_option_ids,
                language=language,
            )
        with timing.measure("property_extraction"):
            extraction = self.property_extractor.extract(query)
        with timing.measure("query_interpretation"):
            leading_alias = self.query_aliases.leading_curated_match(query) if self.query_aliases else None
            interpretation = interpret_query(
                query, extraction, self.category_runtime.taxonomy, self.category_runtime.enrichment,
                additional_matches=(leading_alias,) if leading_alias else (),
                short_property_candidate=self.property_extractor.unique_typo_alias,
            )
        timing.skip("category_variant", "category_retrieval", "category_decision")
        interpretation_payload = interpretation.as_dict()
        interpretation_payload["vector_query_mode"] = "user_selection"
        interpretation_payload["pgvector_queries"] = []
        category_payload["query_interpretation"] = interpretation_payload
        self._attach_confidence_provenance(category_payload, decision_source="explicit_user_selection")
        return category_payload, interpretation.extraction

    def resolve_selection_and_analyze_intent(
        self, *, query: str, selected_id: str,
        allowed_option_ids: list[str] | tuple[str, ...], language: str = "vi",
    ) -> dict[str, Any]:
        timing = SearchTiming("intent_clarification_selection")
        category_payload, extraction = self._selected_category(
            query=query, selected_id=selected_id, allowed_option_ids=allowed_option_ids,
            language=language, timing=timing,
        )
        return self._analysis_from_category(query, category_payload, extraction, timing)

    def close(self) -> None:
        if self.category_runtime.database is not None:
            self.category_runtime.database.close()
