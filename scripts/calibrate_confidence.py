from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model_intent.benchmark import load_yaml, make_embedder, profiles_from_config
from model_intent.confidence import FEATURE_NAMES, extract_confidence_features
from model_intent.evaluation import load_cases
from model_intent.rendering import load_enrichment, render_category_profiles
from model_intent.query_scope import non_place_reason
from model_intent.retrieval import SemanticIndex
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


def fit_logistic(matrix: np.ndarray, labels: np.ndarray, *, l2: float = 0.4) -> dict:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-6] = 1.0
    values = (matrix - means) / scales
    weights = np.zeros(values.shape[1], dtype=np.float64)
    prevalence = float(np.clip(labels.mean(), 1e-4, 1 - 1e-4))
    bias = float(np.log(prevalence / (1.0 - prevalence)))
    learning_rate = 0.08
    for _ in range(5000):
        logits = np.clip(values @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        grad_weights = values.T @ error / len(labels) + l2 * weights / len(labels)
        grad_bias = float(error.mean())
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
    return {
        "features": list(FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "training_n": int(len(labels)),
        "positive_n": int(labels.sum()),
    }


def predict(model: dict, matrix: np.ndarray) -> np.ndarray:
    means = np.asarray(model["means"])
    scales = np.asarray(model["scales"])
    weights = np.asarray(model["weights"])
    logits = (matrix - means) / scales @ weights + float(model["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.1, 0.9, 161):
        predicted = probabilities >= threshold
        positives = labels == 1
        negatives = ~positives
        tpr = float((predicted & positives).sum() / max(1, positives.sum()))
        tnr = float(((~predicted) & negatives).sum() / max(1, negatives.sum()))
        score = (tpr + tnr) / 2.0
        if score > best[0] or (score == best[0] and threshold > best[1]):
            best = (score, float(threshold))
    return best[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/benchmark.cpu.yaml"))
    parser.add_argument("--taxonomy", default=str(ROOT / "data/taxonomies/place_types.poc.v1.json"))
    parser.add_argument("--evaluation", default=str(ROOT / "data/eval/place_type_queries.poc.v1.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/policies/e5_base_confidence_calibration.v1.json"))
    parser.add_argument("--calibration-version", default="e5-base-confidence-v1")
    parser.add_argument("--provisional", action="store_true", help="Record synthetic-label holdout diagnostics; not production calibration")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    profile = next(
        item for item in profiles_from_config(config, include_stretch=True)
        if item.key == "e5-base-multilingual"
    )
    taxonomy = Taxonomy.load(args.taxonomy)
    enrichment_path = Path(config["enrichment_path"])
    if not enrichment_path.is_absolute():
        enrichment_path = ROOT / enrichment_path
    enrichment = load_enrichment(enrichment_path)
    embedder = make_embedder(profile, local_files_only=args.local_files_only)
    documents = render_category_profiles(taxonomy, tuple(config.get("languages", ["vi", "en"])), enrichment)
    index = SemanticIndex.build(taxonomy, documents, embedder)

    feature_rows = []
    top_labels = []
    domain_labels = []
    ambiguity_labels = []
    case_ids = []
    holdout_rows = []
    for case in load_cases(args.evaluation):
        if case.split not in {"calibration", "test"}:
            continue
        prediction = index.search(case.query, embedder, top_k=20, query_id=case.id)
        features = extract_confidence_features(prediction, temperature=0.05, top_k=5)
        valid = set(case.gold_ids + case.acceptable_ids + case.clarification_ids)
        top_id = prediction.hits[0].item_id if prediction.hits else None
        correct = float(not case.is_ood and top_id in valid)
        in_domain = float(not case.is_ood)
        ambiguous = float((case.expected_action or ("reject" if case.is_ood else "accept")) == "clarify")
        row = [features[name] for name in FEATURE_NAMES]
        if case.split == "test":
            if args.provisional:
                holdout_rows.append((case.id, case.query, row, correct, in_domain, ambiguous))
            continue
        feature_rows.append(row)
        top_labels.append(correct)
        domain_labels.append(in_domain)
        ambiguity_labels.append(ambiguous)
        case_ids.append(case.id)

    if args.provisional and (
        len(feature_rows) < 20 or len(holdout_rows) < 20
        or sum(label == 0 for label in domain_labels) < 3
        or sum(in_domain == 0 for _, _, _, _, in_domain, _ in holdout_rows) < 3
    ):
        raise ValueError("provisional calibration requires separate training/holdout cases, including OOD proxy examples")
    matrix = np.asarray(feature_rows, dtype=np.float64)
    top_labels_array = np.asarray(top_labels, dtype=np.float64)
    domain_labels_array = np.asarray(domain_labels, dtype=np.float64)
    ambiguity_labels_array = np.asarray(ambiguity_labels, dtype=np.float64)
    top_model = fit_logistic(matrix, top_labels_array)
    domain_model = fit_logistic(matrix, domain_labels_array)
    ambiguity_model = fit_logistic(matrix, ambiguity_labels_array)
    top_probabilities = predict(top_model, matrix)
    domain_probabilities = predict(domain_model, matrix)
    ambiguity_probabilities = predict(ambiguity_model, matrix)
    top_threshold = best_threshold(top_probabilities, top_labels_array)
    domain_threshold = best_threshold(domain_probabilities, domain_labels_array)
    ambiguity_threshold = best_threshold(ambiguity_probabilities, ambiguity_labels_array)

    holdout = None
    if args.provisional:
        held = np.asarray([row for _, _, row, *_ in holdout_rows], dtype=np.float64)
        held_top = np.asarray([correct for _, _, _, correct, _, _ in holdout_rows], dtype=np.float64)
        held_domain = np.asarray([label for _, _, _, _, label, _ in holdout_rows], dtype=np.float64)
        held_ambiguity = np.asarray([label for _, _, _, _, _, label in holdout_rows], dtype=np.float64)
        top_estimates = predict(top_model, held)
        domain_estimates = predict(domain_model, held)
        ambiguity_estimates = predict(ambiguity_model, held)
        raw_accepted = (top_estimates >= max(top_threshold, 0.55)) & (domain_estimates >= max(domain_threshold, 0.55))
        scope_vetoed = np.asarray([non_place_reason(query) is not None for _, query, *_ in holdout_rows])
        accepted = raw_accepted & ~scope_vetoed
        accepted_correct = int((accepted & (held_top == 1)).sum())
        holdout = {
            "kind": "synthetic_alias_and_authored_proxy_not_independent_gold",
            "n": len(holdout_rows),
            "ood_n": int((held_domain == 0).sum()),
            "accepted_n": int(accepted.sum()),
            "accepted_correct_proxy_n": accepted_correct,
            "accepted_precision_proxy": accepted_correct / int(accepted.sum()) if accepted.any() else None,
            "raw_ood_false_accept_n": int((raw_accepted & (held_domain == 0)).sum()),
            "scope_vetoed_n": int(scope_vetoed.sum()),
            "ood_false_accept_n": int((accepted & (held_domain == 0)).sum()),
            "ood_false_accept_ids": [
                case_id for (case_id, _, *_), false_accept in zip(holdout_rows, accepted & (held_domain == 0))
                if false_accept
            ],
            "top_correct_proxy_n": int(held_top.sum()),
            "ambiguity_proxy_n": int(held_ambiguity.sum()),
            "ambiguity_detected_n": int(((ambiguity_estimates >= ambiguity_threshold) & (held_ambiguity == 1)).sum()),
        }
        if holdout["accepted_n"] == 0 or holdout["ood_false_accept_n"] > 0:
            raise ValueError(f"provisional calibration did not pass basic held-out proxy gates: {holdout}")

    payload = {
        "schema_version": 1,
        "calibration_version": args.calibration_version,
        "quality": "provisional_synthetic_proxy_not_production_validated" if args.provisional else "legacy_poc",
        "holdout_proxy": holdout,
        "kind": "logistic-evidence-v1",
        "taxonomy_fingerprint": taxonomy.fingerprint(),
        "model_key": profile.key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "renderer_version": "category-profile-v3",
        "temperature": 0.05,
        "top_k": 5,
        "top_correct": top_model,
        "in_domain": domain_model,
        "ambiguity": ambiguity_model,
        "action_thresholds": {
            "reject_min_in_domain_confidence": domain_threshold,
            "reject_min_top_confidence": max(0.2, top_threshold - 0.2),
            "clarify_min_in_domain_confidence": domain_threshold,
            "accept_min_in_domain_confidence": max(domain_threshold, 0.55),
            "accept_min_top_confidence": max(top_threshold, 0.55),
            "ambiguity_min_score": ambiguity_threshold,
        },
        "training": {
            "case_ids": case_ids,
            "n": len(case_ids),
            "top_correct_positive_n": int(top_labels_array.sum()),
            "in_domain_positive_n": int(domain_labels_array.sum()),
            "ambiguity_positive_n": int(ambiguity_labels_array.sum()),
        },
    }
    target = Path(args.output)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.provisional and target.exists() and target.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite different provisional calibration: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(json.dumps(payload["action_thresholds"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
