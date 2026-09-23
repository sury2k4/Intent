import math

import pytest

from model_intent.confidence import (
    ConfidenceCalibrationError,
    evaluate_confidence,
    extract_confidence_features,
    load_confidence_calibration,
)
from model_intent.schemas import Prediction, RetrievalHit


def hit(item_id: str, score: float, rank: int) -> RetrievalHit:
    return RetrievalHit(item_id, item_id, score, rank, "vi", item_id)


def prediction(scores, variant_top_ids=("1", "1")):
    hits = tuple(hit(str(index + 1), score, index + 1) for index, score in enumerate(scores))
    return Prediction("q", "query", hits, scores[0], scores[0] - scores[1], "e5-base-multilingual", (), variant_top_ids)


def simple_calibration():
    features = [
        "top1_similarity", "absolute_margin", "relative_margin", "z_gap",
        "softmax_p1", "topk_entropy", "variant_agreement",
    ]
    model = {
        "features": features,
        "means": [0.0] * len(features),
        "scales": [1.0] * len(features),
        "weights": [1.0, 2.0, 1.0, 0.2, 1.0, -1.0, 0.5],
        "bias": -0.5,
    }
    return {
        "calibration_version": "test",
        "temperature": 0.05,
        "top_k": 5,
        "top_correct": model,
        "in_domain": model,
        "ambiguity": model,
    }


def test_feature_extraction_is_finite_and_bounded():
    features = extract_confidence_features(prediction([0.85, 0.78, 0.72, 0.70, 0.68]))
    assert all(math.isfinite(value) for value in features.values())
    assert features["absolute_margin"] == pytest.approx(0.07)
    assert 0 <= features["softmax_p1"] <= 1
    assert 0 <= features["topk_entropy"] <= 1
    assert features["variant_agreement"] == 1.0


def test_stronger_margin_increases_calibrated_evidence():
    calibration = simple_calibration()
    weak = evaluate_confidence(prediction([0.82, 0.81, 0.79, 0.77, 0.75]), calibration)
    strong = evaluate_confidence(prediction([0.86, 0.70, 0.68, 0.66, 0.64]), calibration)
    assert strong.top_correct_confidence > weak.top_correct_confidence
    assert strong.in_domain_confidence > weak.in_domain_confidence


def test_variant_disagreement_lowers_evidence():
    calibration = simple_calibration()
    agree = evaluate_confidence(prediction([0.84, 0.78, 0.74], ("1", "1")), calibration)
    disagree = evaluate_confidence(prediction([0.84, 0.78, 0.74], ("1", "2")), calibration)
    assert agree.top_correct_confidence > disagree.top_correct_confidence


def test_calibration_model_mismatch_fails_closed(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"kind":"logistic-evidence-v1","model_key":"x","model_revision":"r",'
        '"top_correct":{"features":["top1_similarity"],"means":[0],"scales":[1],"weights":[1]},'
        '"in_domain":{"features":["top1_similarity"],"means":[0],"scales":[1],"weights":[1]}}',
        encoding="utf-8",
    )
    with pytest.raises(ConfidenceCalibrationError, match="model mismatch"):
        load_confidence_calibration(path, model_key="y", model_revision="r")
