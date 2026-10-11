from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .confidence import ConfidenceEvidence, evaluate_confidence
from .schemas import ClarificationOption, IntentDecision, Prediction, RetrievalHit
from .search_timing import SearchTiming
from .taxonomy import Taxonomy


class DecisionPolicyError(ValueError):
    pass


def load_decision_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("thresholds"), dict) or not isinstance(payload.get("groups"), list):
        raise DecisionPolicyError("decision policy requires thresholds and groups")
    return payload


def validate_decision_policy(policy: dict[str, Any], taxonomy: Taxonomy) -> list[str]:
    warnings: list[str] = []
    seen_groups: set[str] = set()
    for group in policy.get("groups", []):
        group_id = str(group.get("id", "")).strip()
        members = tuple(str(x) for x in group.get("members", []))
        if not group_id or group_id in seen_groups:
            raise DecisionPolicyError(f"invalid or duplicate group id: {group_id!r}")
        seen_groups.add(group_id)
        if len(set(members)) != len(members):
            raise DecisionPolicyError(f"group {group_id} contains duplicate members")
        if len(members) < 2:
            raise DecisionPolicyError(f"group {group_id} needs at least two unique members")
        unknown = sorted(set(members) - set(taxonomy.items))
        if unknown:
            raise DecisionPolicyError(f"group {group_id} has unknown IDs: {unknown}")
        parent_members = [item_id for item_id in members if taxonomy.has_children(item_id)]
        if parent_members:
            if policy.get("selectability", "leaves_only") == "leaves_only":
                raise DecisionPolicyError(f"group {group_id} includes non-leaf IDs: {parent_members}")
            warnings.append(f"group {group_id} includes non-leaf IDs: {parent_members}")
    return warnings


def _thresholds(policy: dict[str, Any], model_key: str, overrides: dict[str, float] | None) -> dict[str, float]:
    base = dict(policy.get("thresholds", {}).get("default", {}))
    base.update(policy.get("thresholds", {}).get(model_key, {}))
    if overrides:
        base.update({key: float(value) for key, value in overrides.items() if value is not None})
    required = ("reject_min_score", "clarify_min_score", "accept_min_score", "accept_min_margin", "ambiguity_delta")
    missing = [key for key in required if key not in base]
    if missing:
        raise DecisionPolicyError(f"missing thresholds: {missing}")
    values = {key: float(base[key]) for key in required}
    values["group_accept_min_margin"] = float(
        base.get("group_accept_min_margin", max(values["accept_min_margin"] * 2.0, values["ambiguity_delta"]))
    )
    if not all(math.isfinite(value) for value in values.values()):
        raise DecisionPolicyError("decision thresholds must be finite")
    if not values["reject_min_score"] <= values["clarify_min_score"] <= values["accept_min_score"]:
        raise DecisionPolicyError("thresholds must satisfy reject <= clarify <= accept")
    if values["accept_min_margin"] < 0 or values["ambiguity_delta"] < 0 or values["group_accept_min_margin"] < 0:
        raise DecisionPolicyError("decision margins and deltas must be non-negative")
    return values


def _legacy_ambiguity_delta(top_score: float, values: Mapping[str, float]) -> float:
    """Preserve adaptive behavior for uncalibrated legacy/test models."""
    clarify_span = max(0.0, (top_score - values["clarify_min_score"]) * 0.8)
    return max(values["ambiguity_delta"], values["group_accept_min_margin"], clarify_span)


def _language_name(taxonomy: Taxonomy, item_id: str, language: str) -> str:
    item = taxonomy.items[item_id]
    return item.names.get(language) or item.names.get("vi") or item.names.get("en") or next(iter(item.names.values()))


def _hint(enrichment: dict[str, dict[str, Any]], item_id: str, language: str) -> str:
    profile = enrichment.get(item_id, {})
    description = str(profile.get(f"description_{language}") or profile.get("description_vi") or profile.get("description_en") or "")
    return description.strip()


def _group_lookup(policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_member: dict[str, list[dict[str, Any]]] = {}
    for group in policy.get("groups", []):
        for member in group.get("members", []):
            by_member.setdefault(str(member), []).append(group)
    return by_member


def _evidence_kwargs(evidence: ConfidenceEvidence | None) -> dict[str, Any]:
    if evidence is None:
        return {}
    return {
        "calibrated_confidence": evidence.top_correct_confidence,
        "in_domain_confidence": evidence.in_domain_confidence,
        "ambiguity_score": evidence.ambiguity_score,
        "confidence_features": dict(evidence.features),
        "calibration_version": evidence.calibration_version,
        "confidence_quality": evidence.quality,
        "confidence_target": evidence.target,
    }


def _clarification(
    hits: list[RetrievalHit],
    taxonomy: Taxonomy,
    policy: dict[str, Any],
    enrichment: dict[str, dict[str, Any]],
    language: str,
    reason: str,
    group: dict[str, Any] | None,
    prediction: Prediction,
    evidence: ConfidenceEvidence | None,
    margin: float | None = None,
) -> IntentDecision:
    max_options = int(policy.get("max_options", 4))
    options = tuple(
        ClarificationOption(
            item_id=hit.item_id,
            code=hit.code,
            label=_language_name(taxonomy, hit.item_id, language),
            hint=_hint(enrichment, hit.item_id, language),
            score=hit.score,
            relation=(str(group["id"]) if group else taxonomy.relation(hits[0].item_id, hit.item_id)),
        )
        for hit in hits[:max_options]
    )
    questions = policy.get("questions", {}).get(language) or policy.get("questions", {}).get("vi", {})
    question = None
    if group:
        question = group.get(f"question_{language}") or group.get("question_vi") or group.get("question_en")
    if not question:
        question = questions.get("parent" if reason == "nonselectable_parent" else "generic")
    return IntentDecision(
        action="clarify",
        reason=reason,
        language=language,
        question=question,
        options=options,
        top1_score=prediction.top1_score,
        margin=margin if margin is not None else prediction.margin,
        policy_version=str(policy.get("policy_version", "unknown")),
        **_evidence_kwargs(evidence),
    )


def _anchored_group_candidates(
    top: RetrievalHit,
    leaf_hits: list[RetrievalHit],
    by_member: dict[str, list[dict[str, Any]]],
    ambiguity_delta: float,
) -> list[tuple[float, int, dict[str, Any], list[RetrievalHit]]]:
    buckets: list[tuple[float, int, dict[str, Any], list[RetrievalHit]]] = []
    for group_order, group in enumerate(by_member.get(top.item_id, [])):
        members = {str(member) for member in group.get("members", [])}
        candidates = [
            hit for hit in leaf_hits
            if hit.item_id in members and hit.score >= top.score - ambiguity_delta
        ]
        candidates.sort(key=lambda hit: (-hit.score, hit.item_id))
        if len(candidates) >= 2 and candidates[0].item_id == top.item_id:
            buckets.append((candidates[1].score, -group_order, group, candidates))
    return buckets


def decide_intent(
    prediction: Prediction,
    taxonomy: Taxonomy,
    policy: dict[str, Any],
    enrichment: dict[str, dict[str, Any]] | None = None,
    language: str | None = None,
    overrides: dict[str, float] | None = None,
    confidence_calibration: Mapping[str, Any] | None = None,
    *,
    policy_prevalidated: bool = False,
    timing: SearchTiming | None = None,
) -> IntentDecision:
    if not policy_prevalidated:
        validate_decision_policy(policy, taxonomy)
    enrichment = enrichment or {}
    language = language or str(policy.get("fallback_language", "vi"))
    values = _thresholds(policy, prediction.model_key, overrides)
    policy_version = str(policy.get("policy_version", "unknown"))
    with (timing.measure("category_calibration") if timing and confidence_calibration else nullcontext()):
        evidence = evaluate_confidence(prediction, confidence_calibration) if confidence_calibration else None

    if not prediction.hits or prediction.top1_score is None:
        return IntentDecision(
            "reject", "no_hits", language, policy_version=policy_version,
            **_evidence_kwargs(evidence),
        )
    top = prediction.hits[0]
    leaf_hits = [hit for hit in prediction.hits if not taxonomy.has_children(hit.item_id)]
    by_member = _group_lookup(policy)

    if taxonomy.has_children(top.item_id):
        descendant_ids = set(taxonomy.descendants(top.item_id, leaves_only=True))
        delta = values["ambiguity_delta"] if evidence is not None else _legacy_ambiguity_delta(top.score, values)
        descendants = [
            hit for hit in leaf_hits
            if hit.item_id in descendant_ids and hit.score >= top.score - delta
        ]
        if len(descendants) >= 2:
            return _clarification(
                descendants, taxonomy, policy, enrichment, language,
                "nonselectable_parent", None, prediction, evidence,
                margin=(descendants[0].score - descendants[1].score),
            )
        if len(descendants) == 1:
            top = descendants[0]
        else:
            return IntentDecision(
                "reject", "nonselectable_parent_without_options", language,
                top1_score=prediction.top1_score, margin=prediction.margin,
                policy_version=policy_version, **_evidence_kwargs(evidence),
            )

    second_leaf = next((hit for hit in leaf_hits if hit.item_id != top.item_id), None)
    leaf_margin = top.score - second_leaf.score if second_leaf else float("inf")

    if evidence is not None:
        action_thresholds = dict(confidence_calibration.get("action_thresholds", {}))
        reject_domain = float(action_thresholds["reject_min_in_domain_confidence"])
        reject_top = float(action_thresholds["reject_min_top_confidence"])
        if evidence.in_domain_confidence < reject_domain or evidence.top_correct_confidence < reject_top:
            return IntentDecision(
                "reject", "low_calibrated_confidence", language,
                top1_score=top.score, margin=leaf_margin, policy_version=policy_version,
                **_evidence_kwargs(evidence),
            )
        ambiguity_delta = values["ambiguity_delta"]
        groups = _anchored_group_candidates(top, leaf_hits, by_member, ambiguity_delta)
        if (
            groups
            and evidence.in_domain_confidence >= float(action_thresholds["clarify_min_in_domain_confidence"])
            and evidence.ambiguity_score >= float(action_thresholds["ambiguity_min_score"])
        ):
            _, _, selected_group, candidates = max(groups, key=lambda item: (item[0], item[1]))
            return _clarification(
                candidates, taxonomy, policy, enrichment, language,
                "near_group_competition", selected_group, prediction, evidence,
                margin=leaf_margin,
            )
        if (
            evidence.in_domain_confidence >= float(action_thresholds["accept_min_in_domain_confidence"])
            and evidence.top_correct_confidence >= float(action_thresholds["accept_min_top_confidence"])
        ):
            return IntentDecision(
                "accept", "calibrated_high_confidence", language,
                accepted_item_id=top.item_id, accepted_code=top.code,
                top1_score=top.score, margin=leaf_margin, policy_version=policy_version,
                **_evidence_kwargs(evidence),
            )
        return IntentDecision(
            "reject", "insufficient_calibrated_evidence", language,
            top1_score=top.score, margin=leaf_margin, policy_version=policy_version,
            **_evidence_kwargs(evidence),
        )

    # Legacy fallback for fake/benchmark configurations without a calibration artifact.
    if top.score < values["reject_min_score"]:
        return IntentDecision(
            "reject", "low_score", language, top1_score=top.score,
            margin=leaf_margin, policy_version=policy_version,
        )
    ambiguity_delta = _legacy_ambiguity_delta(top.score, values)
    groups = _anchored_group_candidates(top, leaf_hits, by_member, ambiguity_delta)
    if groups and top.score >= values["clarify_min_score"]:
        _, _, selected_group, candidates = max(groups, key=lambda item: (item[0], item[1]))
        return _clarification(
            candidates, taxonomy, policy, enrichment, language,
            "near_group_competition", selected_group, prediction, None,
            margin=leaf_margin,
        )
    if top.score >= values["accept_min_score"] and leaf_margin >= values["accept_min_margin"]:
        return IntentDecision(
            "accept", "high_confidence_leaf", language,
            accepted_item_id=top.item_id, accepted_code=top.code,
            top1_score=top.score, margin=leaf_margin, policy_version=policy_version,
        )
    return IntentDecision(
        "reject", "uncertain_unrelated_candidates", language,
        top1_score=top.score, margin=leaf_margin, policy_version=policy_version,
    )


def decision_to_dict(decision: IntentDecision) -> dict[str, Any]:
    return asdict(decision)
