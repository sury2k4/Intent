from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Mapping

from .normalization import fold_vietnamese, normalize_text

from .benchmark import load_yaml
from .category_resolution import ExplicitCategory, explicit_categories
from .db import Database, validate_database_target
from .osm_ingest import file_sha256, validate_catalogs
from .poi_ranking import load_ranking_policy, rank_pois
from .poi_properties import PropertyProfiles
from .poi_repository import PoiRepository
from .poi_schemas import PoiSearchResult, PropertyExtraction
from .query_interpretation import InterpretedQuery, interpret_query
from .query_scope import non_place_reason
from .property_extraction import PropertyExtractor, active_constraints, reconcile_for_category
from .runtime import MatcherRuntime

E5_POI_MODEL_KEY = "e5-base-multilingual"
E5_POI_MODEL_ID = "intfloat/multilingual-e5-base"
E5_POI_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
E5_POI_DIMENSION = 768


class SearchRuntime:
    """Composition layer: existing category intent followed by POI search."""

    def __init__(
        self,
        *,
        category_runtime: MatcherRuntime,
        property_extractor: PropertyExtractor,
        poi_repository: PoiRepository,
        ranking_policy: Mapping[str, Any],
        property_catalog: Mapping[str, Mapping[str, Any]],
        property_profiles: PropertyProfiles | None = None,
    ) -> None:
        self.category_runtime = category_runtime
        self.property_extractor = property_extractor
        self.poi_repository = poi_repository
        self.ranking_policy = dict(ranking_policy)
        self.property_catalog = dict(property_catalog)
        self.property_profiles = property_profiles

    @classmethod
    def from_config(
        cls,
        *,
        config_path: str | Path,
        taxonomy_path: str | Path,
        model_key: str,
        local_files_only: bool = False,
        generation_id: int | None = None,
        poi_generation_id: int | None = None,
    ) -> "SearchRuntime":
        config_path = Path(config_path).resolve()
        project_root = config_path.parent.parent
        config = load_yaml(config_path)
        validate_database_target(config)
        if (config.get("poi_search") or {}).get("backend") == "postgres" and model_key != E5_POI_MODEL_KEY:
            raise RuntimeError("unified POI serving supports only pinned E5-base 768D")
        category_runtime = MatcherRuntime.from_config(
            config_path=config_path,
            taxonomy_path=taxonomy_path,
            model_key=model_key,
            local_files_only=local_files_only,
            generation_id=generation_id,
        )
        if model_key == E5_POI_MODEL_KEY:
            metadata = category_runtime.metadata
            dimension = int(category_runtime.embedder.metadata().get("dimension", 0))
            backend_ready = (
                metadata.backend == "numpy"
                and metadata.retrieval_mode == "numpy_exact"
                and getattr(category_runtime.index, "vectors", None) is not None
                and category_runtime.index.vectors.shape[1] == E5_POI_DIMENSION
            ) or (
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
                raise RuntimeError("POI category runtime requires pinned E5-base 768D on NumPy or exact pgvector retrieval")
            if category_runtime.confidence_calibration is None:
                raise RuntimeError("E5 category confidence calibration is missing or incompatible")
        poi_config = dict(config.get("poi_search") or {})

        def resolve(value: str | Path | None, default: str | Path | None = None) -> Path | None:
            selected = value if value is not None else default
            if selected is None:
                return None
            path = Path(selected)
            return path if path.is_absolute() else (project_root / path).resolve()

        property_path = resolve(poi_config.get("property_path"), "data/properties/poi_properties.poc.v1.json")
        mapping_path = resolve(poi_config.get("mapping_path"), "data/osm/mappings/place_types.poc.v1.json")
        ranking_path = resolve(poi_config.get("ranking_policy_path"), "data/policies/poi_ranking.poc.v1.json")
        artifact_path = resolve(poi_config.get("artifact_path"))
        backend = str(poi_config.get("backend", "artifact"))
        if property_path is None or not property_path.exists():
            raise FileNotFoundError(f"POI property taxonomy not found: {property_path}")
        if mapping_path is None or not mapping_path.exists():
            raise FileNotFoundError(f"POI mapping catalog not found: {mapping_path}")
        catalogs = validate_catalogs(
            category_runtime.taxonomy,
            mapping_path,
            property_path,
            require_complete=True,
        )
        property_catalog = catalogs["properties"]
        extractor = PropertyExtractor(property_catalog)
        poi_database: Database | None = None
        owns_database = False
        if backend == "postgres":
            poi_database = category_runtime.database
            if poi_database is None:
                poi_database = Database(
                    min_size=int(config.get("db_pool_min_size", 1)),
                    max_size=int(config.get("db_pool_max_size", 8)),
                    timeout=float(config.get("db_pool_timeout_seconds", 10)),
                    statement_timeout_ms=int(config.get("db_statement_timeout_ms", 5000)),
                )
                poi_database.open()
                owns_database = True
            try:
                repository = PoiRepository.from_database(
                    poi_database,
                    dataset_key=str(poi_config.get("dataset_key", "hcmc-osm")),
                    generation_id=poi_generation_id,
                    taxonomy_fingerprint=category_runtime.taxonomy.fingerprint(),
                    mapping_sha256=file_sha256(mapping_path),
                    property_taxonomy_sha256=file_sha256(property_path),
                    taxonomy=category_runtime.taxonomy,
                    owns_database=owns_database,
                )
            except Exception:
                if owns_database:
                    poi_database.close()
                if category_runtime.database is not None:
                    category_runtime.database.close()
                raise
        elif backend == "artifact":
            if artifact_path is None:
                raise ValueError("poi_search.artifact_path is required for artifact-backed search")
            if not artifact_path.exists():
                raise FileNotFoundError(f"POI artifact not found: {artifact_path}")
            manifest_path = artifact_path.parent / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"POI artifact manifest not found: {manifest_path}")
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("taxonomy_fingerprint") != category_runtime.taxonomy.fingerprint():
                raise RuntimeError("POI artifact taxonomy fingerprint does not match the category taxonomy")
            expected_artifact_hash = str(manifest.get("pois_jsonl_sha256", ""))
            if len(expected_artifact_hash) != 64 or file_sha256(artifact_path) != expected_artifact_hash:
                raise RuntimeError("POI artifact hash is missing or does not match its manifest")
            if manifest.get("mapping_sha256") != file_sha256(mapping_path):
                raise RuntimeError("POI artifact mapping catalog hash does not match current configuration")
            if manifest.get("property_taxonomy_sha256") != file_sha256(property_path):
                raise RuntimeError("POI artifact property taxonomy hash does not match current configuration")
            base_catalog = json.loads(property_path.read_text(encoding="utf-8")).get("base_catalog")
            if base_catalog and manifest.get("property_base_sha256") != file_sha256(property_path.parent / base_catalog):
                raise RuntimeError("POI artifact base property catalog hash does not match current configuration")
            repository = PoiRepository.from_jsonl(artifact_path, generation_id=poi_generation_id)
        else:
            raise ValueError(f"unsupported POI search backend: {backend}")
        policy = load_ranking_policy(ranking_path)
        return cls(
            category_runtime=category_runtime,
            property_extractor=extractor,
            poi_repository=repository,
            ranking_policy=policy,
            property_catalog=property_catalog,
            property_profiles=catalogs["profiles"],
        )

    @staticmethod
    def _validate_search_args(
        poi_top_k: int,
        origin: tuple[float, float] | None,
        radius_km: float | None,
    ) -> None:
        if poi_top_k < 1:
            raise ValueError("poi_top_k must be positive")
        if radius_km is not None and (not math.isfinite(radius_km) or radius_km < 0):
            raise ValueError("radius_km must be a finite non-negative number")
        if origin is not None:
            latitude, longitude = origin
            if not math.isfinite(latitude) or not -90 <= latitude <= 90:
                raise ValueError("origin latitude must be between -90 and 90")
            if not math.isfinite(longitude) or not -180 <= longitude <= 180:
                raise ValueError("origin longitude must be between -180 and 180")

    def search(
        self,
        text: str,
        *,
        category_top_k: int = 5,
        poi_top_k: int = 20,
        language: str = "vi",
        overrides: dict[str, float] | None = None,
        origin: tuple[float, float] | None = None,
        radius_km: float | None = None,
        verified_only: bool = False,
    ) -> dict[str, Any]:
        self._validate_search_args(poi_top_k, origin, radius_km)
        verified_only = verified_only or int(self.ranking_policy.get("schema_version", 1)) >= 3
        extraction = self.property_extractor.extract(text)
        interpretation = interpret_query(
            text, extraction, self.category_runtime.taxonomy, self.category_runtime.enrichment,
        )
        extraction = interpretation.extraction
        real_model = getattr(self.category_runtime, "model_key", "fake") != "fake"
        if real_model:
            outcomes = [
                self.category_runtime.query(
                    candidate, top_k=category_top_k, language=language, overrides=overrides,
                )
                for candidate in interpretation.variants
            ]
            category_payload = self._choose_real_category(text, outcomes, interpretation, language)
        else:
            category_payload = self.category_runtime.query(
                text, top_k=category_top_k, language=language, overrides=overrides,
            )
            category_payload = self._resolve_explicit_category(text, category_payload, language, extraction)
        category_payload["query_interpretation"] = interpretation.as_dict()
        return self._search_from_category(
            text,
            category_payload,
            poi_top_k=poi_top_k,
            language=language,
            origin=origin,
            radius_km=radius_km,
            verified_only=verified_only,
            extraction=extraction,
        )

    def _choose_real_category(
        self,
        text: str,
        outcomes: list[dict[str, Any]],
        interpretation: InterpretedQuery,
        language: str,
    ) -> dict[str, Any]:
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
        focused_text = str(chosen.get("prediction", {}).get("query") or interpretation.variants[0])
        payload = self._guard_real_category(
            text, chosen, language, interpretation.extraction, focused_text,
            matches=interpretation.category_matches,
        )
        resolution = payload["category_resolution"]
        resolution["semantic_variants"] = [
            {"query": variant, "action": outcome["decision"]["action"],
             "accepted_item_id": outcome["decision"].get("accepted_item_id"),
             "top1_score": outcome.get("prediction", {}).get("top1_score")}
            for variant, outcome in zip(interpretation.variants, outcomes)
        ]
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
    ) -> dict[str, Any]:
        """E5 alone may accept a leaf; lexical checks can only veto or clarify."""
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
        if scope_reason:
            decision.update({
                "action": "reject", "reason": scope_reason,
                "accepted_item_id": None, "accepted_code": None, "question": None, "options": [],
            })
            payload["decision"] = decision
            payload["category_resolution"]["source"] = "informational_veto" if scope_reason == "informational_not_place_search" else "non_place_veto"
            return payload
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

    def resolve_selection_and_search(
        self,
        *,
        query: str,
        selected_id: str,
        allowed_option_ids: list[str] | tuple[str, ...],
        poi_top_k: int = 20,
        language: str = "vi",
        origin: tuple[float, float] | None = None,
        radius_km: float | None = None,
        verified_only: bool = False,
    ) -> dict[str, Any]:
        self._validate_search_args(poi_top_k, origin, radius_km)
        verified_only = verified_only or int(self.ranking_policy.get("schema_version", 1)) >= 3
        category_payload = self.category_runtime.resolve_selection(
            query=query,
            selected_id=selected_id,
            allowed_option_ids=allowed_option_ids,
            language=language,
        )
        interpretation = interpret_query(
            query, self.property_extractor.extract(query),
            self.category_runtime.taxonomy, self.category_runtime.enrichment,
        )
        category_payload["query_interpretation"] = interpretation.as_dict()
        return self._search_from_category(
            query,
            category_payload,
            poi_top_k=poi_top_k,
            language=language,
            origin=origin,
            radius_km=radius_km,
            verified_only=verified_only,
            extraction=interpretation.extraction,
        )

    def _search_from_category(
        self,
        text: str,
        category_payload: Mapping[str, Any],
        *,
        poi_top_k: int,
        language: str,
        origin: tuple[float, float] | None,
        radius_km: float | None,
        verified_only: bool,
        extraction: PropertyExtraction | None = None,
    ) -> dict[str, Any]:
        verified_only = verified_only or int(self.ranking_policy.get("schema_version", 1)) >= 3
        extraction = extraction or self.property_extractor.extract(text)
        decision = category_payload.get("decision") or {}
        action = str(decision.get("action", "reject"))
        accepted_id = decision.get("accepted_item_id")
        if action != "accept" or not accepted_id:
            reason = str(decision.get("reason", ""))
            status = (
                "unsupported_category" if reason == "unsupported_branded_category"
                else "skipped_category_clarify" if action == "clarify"
                else "skipped_category_reject"
            )
            warning = (
                "The current taxonomy only offers branded subcategories for this request; "
                "a generic place type and OSM mapping are not available yet."
                if status == "unsupported_category"
                else "POI retrieval is gated on an accepted leaf category"
            )
            result = PoiSearchResult(
                status=status,
                category_id=str(accepted_id) if accepted_id else None,
                ranking_policy_version=str(self.ranking_policy.get("policy_version", "unknown")),
                ranking_schema_version=3 if int(self.ranking_policy.get("schema_version", 1)) >= 3 else None,
                verified_only=verified_only,
                warnings=(warning,),
            )
            return {
                "query": text,
                "intent": dict(category_payload),
                "property_extraction": extraction.as_dict(),
                "poi_search": result.as_dict(),
            }
        item = self.category_runtime.taxonomy.items[str(accepted_id)]
        extraction = reconcile_for_category(
            extraction,
            category_code=item.code,
            property_catalog=self.property_catalog,
            category_id=item.id,
            taxonomy=self.category_runtime.taxonomy,
            profiles=self.property_profiles,
        )
        constraints = active_constraints(extraction)
        radius_constraints = [row for row in constraints if row.property_id == "radius_km" and row.operator == "lte"]
        if radius_km is None and radius_constraints:
            radius_km = min(float(row.value) for row in radius_constraints)
        elif radius_km is not None and radius_constraints and any(float(row.value) != radius_km for row in radius_constraints):
            from dataclasses import replace

            extraction = PropertyExtraction(
                extraction.query,
                extraction.schema_version,
                tuple(
                    replace(row, applied=False, ignored_reason="overridden_by_explicit_radius")
                    if row.property_id == "radius_km" and row.applied else row
                    for row in extraction.properties
                ),
                tuple(dict.fromkeys((*extraction.warnings, "query radius was replaced by the explicit radius input"))),
                extraction.failed_open,
            )
        if radius_km is not None and origin is None:
            from dataclasses import replace

            extraction = PropertyExtraction(
                extraction.query,
                extraction.schema_version,
                tuple(
                    replace(row, applied=False, ignored_reason="missing_origin")
                    if row.property_id == "radius_km" and row.applied else row
                    for row in extraction.properties
                ),
                tuple(dict.fromkeys((*extraction.warnings, "radius constraint was not applied because no origin coordinates were supplied"))),
                extraction.failed_open,
            )
            radius_km = None
        rank_constraints = tuple(row for row in active_constraints(extraction) if row.property_id != "radius_km")
        taxonomy = self.category_runtime.taxonomy
        restaurant_ids = (
            "20",
            *sorted(
                (leaf.id for leaf in taxonomy.items.values()
                 if leaf.parent_id == "4" and (
                     leaf.code.endswith("_restaurant") or leaf.code == "steak_house"
                 )),
                key=int,
            ),
        )
        if item.id == "20":
            # Cuisine-specific leaves are siblings of generic restaurant.
            category_records = self.poi_repository.for_categories(restaurant_ids)
        elif item.id in restaurant_ids and item.code.endswith("_restaurant"):
            cuisine = item.code.removesuffix("_restaurant")
            accepted_cuisines = {cuisine}
            if cuisine == "vegan":
                accepted_cuisines.add("vegetarian")
            elif cuisine == "vietnamese":
                accepted_cuisines.add("vietnam")
            category_records = self.poi_repository.for_categories(
                restaurant_ids,
                cuisine_filter=accepted_cuisines,
                always_include_category_id=item.id,
            )
        elif item.id == "242":
            # Generic charging includes operator-branded car chargers; it does not
            # include motorcycle-only charging or battery-swap cabinets.
            category_records = self.poi_repository.for_categories(("242", "19"))
        elif item.id == "244":
            # Unspecified schools also include explicitly tagged school grades.
            category_records = self.poi_repository.for_categories(("244", "44", "45", "46", "165"))
        else:
            category_records = self.poi_repository.for_category(item.id)
        hits, counts = rank_pois(
            category_records,
            rank_constraints,
            policy=self.ranking_policy,
            origin=origin,
            radius_km=radius_km,
            top_k=poi_top_k,
            verified_only=verified_only,
        )
        status = "ok" if hits else "no_pois_for_category" if not category_records else "no_matches"
        warnings = list(extraction.warnings)
        if verified_only and category_records and not hits and any(
            row.property_id == "rating" and row.mode == "required" for row in rank_constraints
        ) and not any(record.rating_source == "osm" and record.rating is not None for record in category_records):
            status = "no_verified_rating_data"
            warnings.append("no POI in this category has a verified rating in the current dataset")
        elif not verified_only and any(row.mode == "excluded" for row in rank_constraints) and counts.get("mock_supported_count"):
            warnings.append("demo exclusions may include POIs without verified negative evidence")
        result = PoiSearchResult(
            status=status,
            category_id=str(accepted_id),
            generation_id=self.poi_repository.generation_id,
            category_candidate_count=len(category_records),
            hard_filter_match_count=len(category_records) - int(counts.get("required_rejected", 0)) - int(counts.get("excluded_rejected", 0)) - int(counts.get("distance_rejected", 0)),
            verified_required_count=counts.get("verified_required_count"),
            mock_supported_count=counts.get("mock_supported_count"),
            verified_only=verified_only,
            effective_radius_km=radius_km,
            returned_count=len(hits),
            ranking_policy_version=str(self.ranking_policy.get("policy_version", "unknown")),
            ranking_schema_version=3 if int(self.ranking_policy.get("schema_version", 1)) >= 3 else None,
            hits=hits,
            warnings=tuple(warnings),
        )
        return {
            "query": text,
            "intent": dict(category_payload),
            "property_extraction": extraction.as_dict(),
            "poi_search": result.as_dict(),
        }

    def close(self) -> None:
        self.poi_repository.close()
        if self.category_runtime.database is not None:
            self.category_runtime.database.close()
