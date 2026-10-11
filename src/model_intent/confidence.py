from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import Prediction

EPSILON = 1e-8
FEATURE_NAMES = (
    "top1_similarity",
    "absolute_margin",
    "relative_margin",
    "z_gap",
    "softmax_p1",
    "topk_entropy",
    "variant_agreement",
)
CONFIDENCE_TARGET = "retrieved_top1_acceptable_leaf_not_ood"
LEGACY_QUALITY = "legacy_unverified"
ACTION_THRESHOLDS = (
    "reject_min_in_domain_confidence",
    "reject_min_top_confidence",
    "clarify_min_in_domain_confidence",
    "accept_min_in_domain_confidence",
    "accept_min_top_confidence",
    "ambiguity_min_score",
)


class ConfidenceCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class ConfidenceEvidence:
    top_correct_confidence: float
    in_domain_confidence: float
    ambiguity_score: float
    features: Mapping[str, float]
    calibration_version: str
    quality: str = LEGACY_QUALITY
    target: str = CONFIDENCE_TARGET


def load_confidence_calibration(
    path: str | Path,
    *,
    model_key: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    renderer_version: str | None = None,
    taxonomy_fingerprint: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "logistic-evidence-v1":
        raise ConfidenceCalibrationError("unsupported confidence calibration kind")
    if model_key is not None and payload.get("model_key") != model_key:
        raise ConfidenceCalibrationError(
            f"confidence calibration model mismatch: {payload.get('model_key')} != {model_key}"
        )
    if model_id is not None and payload.get("model_id") != model_id:
        raise ConfidenceCalibrationError("confidence calibration model ID mismatch")
    expected_revision = str(payload.get("model_revision") or "")
    if model_revision is not None and expected_revision != str(model_revision or ""):
        raise ConfidenceCalibrationError("confidence calibration revision mismatch")
    if renderer_version is not None and payload.get("renderer_version") != renderer_version:
        raise ConfidenceCalibrationError("confidence calibration renderer mismatch")
    pinned_taxonomy = payload.get("taxonomy_fingerprint")
    if taxonomy_fingerprint is not None and pinned_taxonomy is not None and pinned_taxonomy != taxonomy_fingerprint:
        raise ConfidenceCalibrationError("confidence calibration taxonomy mismatch")
    quality = payload.get("quality", LEGACY_QUALITY)
    if not isinstance(quality, str) or quality not in {
        LEGACY_QUALITY, "legacy_poc", "provisional_synthetic_proxy_not_production_validated",
    }:
        raise ConfidenceCalibrationError("confidence calibration quality requires independent validation support")
    payload["quality"] = quality
    for name in ("top_correct", "in_domain", "ambiguity"):
        model = payload.get(name)
        if not isinstance(model, dict):
            raise ConfidenceCalibrationError(f"missing confidence model: {name}")
        raw_features = model.get("features")
        if not isinstance(raw_features, list) or not all(isinstance(feature, str) for feature in raw_features):
            raise ConfidenceCalibrationError(f"invalid feature list for {name}")
        feature_names = tuple(raw_features)
        if not feature_names or len(set(feature_names)) != len(feature_names) or any(feature not in FEATURE_NAMES for feature in feature_names):
            raise ConfidenceCalibrationError(f"invalid feature list for {name}")
        for field in ("means", "scales", "weights"):
            values = model.get(field)
            if not isinstance(values, list) or len(values) != len(feature_names):
                raise ConfidenceCalibrationError(f"invalid {field} for {name}")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise ConfidenceCalibrationError(f"non-finite {field} for {name}")
            if field == "scales" and any(value <= 0 for value in values):
                raise ConfidenceCalibrationError(f"non-positive scales for {name}")
        bias = model.get("bias")
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
            raise ConfidenceCalibrationError(f"invalid bias for {name}")
    thresholds = payload.get("action_thresholds")
    if not isinstance(thresholds, dict) or any(key not in thresholds for key in ACTION_THRESHOLDS):
        raise ConfidenceCalibrationError("missing confidence action thresholds")
    if any(
        isinstance(thresholds[key], bool) or not isinstance(thresholds[key], (int, float))
        or not math.isfinite(thresholds[key]) or not 0 <= thresholds[key] <= 1
        for key in ACTION_THRESHOLDS
    ):
        raise ConfidenceCalibrationError("confidence action thresholds must be finite values in [0, 1]")
    if (
        thresholds["reject_min_top_confidence"] > thresholds["accept_min_top_confidence"]
        or thresholds["reject_min_in_domain_confidence"] > thresholds["accept_min_in_domain_confidence"]
        or thresholds["clarify_min_in_domain_confidence"] < thresholds["reject_min_in_domain_confidence"]
    ):
        raise ConfidenceCalibrationError("confidence action thresholds are inconsistent")
    temperature = payload.get("temperature", 0.05)
    top_k = payload.get("top_k", 5)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ConfidenceCalibrationError("confidence temperature must be positive and finite")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 2:
        raise ConfidenceCalibrationError("confidence top_k must be at least 2")
    return payload


def extract_confidence_features(
    prediction: Prediction,
    *,
    temperature: float = 0.05,
    top_k: int = 5,
) -> dict[str, float]:
    scores = np.asarray([hit.score for hit in prediction.hits[:top_k]], dtype=np.float64)
    if scores.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}
    if not np.isfinite(scores).all():
        raise ConfidenceCalibrationError("prediction contains non-finite similarities")
    s1 = float(scores[0])
    s2 = float(scores[1]) if scores.size > 1 else 0.0
    absolute_margin = max(0.0, s1 - s2)
    relative_margin = absolute_margin / max(1.0 - s2, EPSILON)
    topk_mean = float(np.mean(scores))
    topk_std = float(np.std(scores))
    z_gap = (s1 - topk_mean) / max(topk_std, EPSILON)
    safe_temperature = max(float(temperature), 1e-4)
    logits = (scores - np.max(scores)) / safe_temperature
    probabilities = np.exp(logits)
    probabilities /= max(float(np.sum(probabilities)), EPSILON)
    softmax_p1 = float(probabilities[0])
    if probabilities.size <= 1:
        entropy = 0.0
    else:
        entropy = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, EPSILON)))
            / math.log(probabilities.size)
        )
    variant_ids = tuple(item_id for item_id in prediction.variant_top_ids if item_id)
    variant_agreement = 1.0 if len(set(variant_ids)) <= 1 else 0.0
    return {
        "top1_similarity": s1,
        "absolute_margin": absolute_margin,
        "relative_margin": relative_margin,
        "z_gap": z_gap,
        "softmax_p1": softmax_p1,
        "topk_entropy": entropy,
        "variant_agreement": variant_agreement,
    }


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _predict_logistic(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    names = tuple(model["features"])
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.maximum(np.asarray(model["scales"], dtype=np.float64), EPSILON)
    weights = np.asarray(model["weights"], dtype=np.float64)
    values = np.asarray([features[name] for name in names], dtype=np.float64)
    standardized = (values - means) / scales
    return _sigmoid(float(model.get("bias", 0.0)) + float(standardized @ weights))


def evaluate_confidence(
    prediction: Prediction,
    calibration: Mapping[str, Any],
) -> ConfidenceEvidence:
    features = extract_confidence_features(
        prediction,
        temperature=float(calibration.get("temperature", 0.05)),
        top_k=int(calibration.get("top_k", 5)),
    )
    top_correct = _predict_logistic(calibration["top_correct"], features)
    in_domain = _predict_logistic(calibration["in_domain"], features)
    ambiguity = _predict_logistic(calibration["ambiguity"], features)
    return ConfidenceEvidence(
        top_correct_confidence=top_correct,
        in_domain_confidence=in_domain,
        ambiguity_score=ambiguity,
        features=features,
        calibration_version=str(calibration.get("calibration_version", "unknown")),
        quality=str(calibration.get("quality", LEGACY_QUALITY)),
        target=CONFIDENCE_TARGET,
    )
