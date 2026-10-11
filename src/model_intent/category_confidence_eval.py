from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .serving_config import load_yaml
from .embedding_sync import profile_source_fingerprint
from .evaluation import probability_metrics
from .normalization import fold_vietnamese, normalize_text
from .rendering import PROFILE_RENDERER_VERSION
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class ReviewedCategoryCase:
    id: str
    query: str
    language: str
    split: str
    source_group: str
    family_id: str
    gold_ids: tuple[str, ...]
    acceptable_ids: tuple[str, ...]
    clarification_ids: tuple[str, ...]
    is_ood: bool
    expected_action: str


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serving_identity(runtime: Any, config_path: str | Path) -> dict[str, Any]:
    """The category resources that must match a reviewed dataset manifest."""
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    metadata = runtime.category_runtime.metadata
    project_root = config_path.parent.parent

    def resource_hash(field: str) -> str:
        selected = Path(config[field])
        return _file_sha256(selected if selected.is_absolute() else project_root / selected)

    property_name = (config.get("property_intent") or {}).get("property_path")
    if not property_name:
        raise ValueError("serving config requires a property catalog")
    property_path = Path(property_name)
    if not property_path.is_absolute():
        property_path = project_root / property_path
    property_payload = json.loads(property_path.read_text(encoding="utf-8"))
    property_digest = hashlib.sha256(property_path.read_bytes())
    if property_payload.get("base_catalog"):
        property_digest.update((property_path.parent / property_payload["base_catalog"]).read_bytes())
    source_dir = Path(__file__).resolve().parent
    code_digest = hashlib.sha256()
    for name in (
        "search_runtime.py", "query_interpretation.py", "category_resolution.py",
        "category_query_aliases.py", "property_extraction.py", "query_scope.py",
        "decision.py", "confidence.py",
    ):
        code_digest.update((source_dir / name).read_bytes())

    return {
        "model_key": metadata.model_key,
        "model_id": metadata.model_id,
        "model_revision": metadata.model_revision,
        "model_max_seq_length": runtime.category_runtime.embedder.metadata().get("max_seq_length"),
        "category_generation_id": metadata.generation_id,
        "taxonomy_fingerprint": metadata.taxonomy_fingerprint,
        "renderer_version": PROFILE_RENDERER_VERSION,
        "profile_source_fingerprint": profile_source_fingerprint(runtime.category_runtime.enrichment),
        "config_sha256": _file_sha256(config_path),
        "policy_sha256": resource_hash("policy_path"),
        "calibration_sha256": resource_hash("confidence_calibration_path"),
        "query_aliases_sha256": resource_hash("category_query_aliases_path") if config.get("category_query_aliases_path") else None,
        "property_catalog_sha256": property_digest.hexdigest(),
        "analysis_code_sha256": code_digest.hexdigest(),
    }


def validate_review_manifest(
    manifest: Mapping[str, Any], identity: Mapping[str, Any], cases_path: str | Path,
) -> None:
    if not isinstance(manifest, Mapping):
        raise TypeError("review manifest must be an object")
    for key in ("dataset_version", "annotation_guidelines_version"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"review manifest requires {key}")
    if manifest.get("label_origin") != "independent_human_adjudication":
        raise ValueError("review manifest requires independent human adjudication")
    if manifest.get("cases_sha256") != _file_sha256(Path(cases_path)):
        raise ValueError("review cases checksum differs from the frozen manifest")
    for key, expected in identity.items():
        if expected is None or manifest.get(key) != expected:
            raise ValueError(f"review manifest incompatible with serving {key}")


def load_reviewed_cases(path: str | Path, taxonomy: Taxonomy) -> list[ReviewedCategoryCase]:
    """Accept only adjudicated, privacy-reviewed real queries; never proxy gold."""
    cases: list[ReviewedCategoryCase] = []
    seen_ids: set[str] = set()
    seen_queries: set[tuple[str, str]] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("case must be an object")
            case_id = row.get("id")
            query = row.get("query")
            split = row.get("split")
            language = row.get("language")
            provenance = row.get("provenance")
            if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
                raise ValueError("duplicate or missing case ID")
            if not isinstance(query, str) or not normalize_text(query):
                raise ValueError("query is empty")
            if split not in {"train", "validation", "test"} or language not in {"vi", "en"}:
                raise ValueError("invalid split or language")
            query_key = (language, normalize_text(fold_vietnamese(query)))
            if query_key in seen_queries:
                raise ValueError("duplicate reviewed query")
            if not isinstance(provenance, dict) or provenance.get("label_origin") != "independent_human_adjudication":
                raise ValueError("labels must be independently human adjudicated")
            if provenance.get("privacy_reviewed") is not True or provenance.get("adjudicated") is not True:
                raise ValueError("case requires privacy review and adjudication")
            reviewers = provenance.get("reviewer_count")
            if isinstance(reviewers, bool) or not isinstance(reviewers, int) or reviewers < 2:
                raise ValueError("case requires two independent reviewers")
            source_group = provenance.get("source_group")
            family_id = provenance.get("family_id")
            if not all(isinstance(value, str) and value.strip() for value in (source_group, family_id)):
                raise ValueError("case requires source_group and paraphrase family_id")
            if any(key in row for key in ("predicted_category", "model_output", "model_score")):
                raise ValueError("gold labels must not contain model predictions")
            fields = ("gold_ids", "acceptable_ids", "clarification_ids")
            if any(not isinstance(row.get(name), list) or any(not isinstance(value, str) for value in row[name]) for name in fields):
                raise ValueError("category ID sets must be string lists")
            gold, acceptable, clarification = (tuple(row[name]) for name in fields)
            labeled_ids = {*gold, *acceptable, *clarification}
            if any(item_id not in taxonomy.items or taxonomy.has_children(item_id) for item_id in labeled_ids):
                raise ValueError("category labels must be known taxonomy leaves")
            if row.get("is_ood") is not True and row.get("is_ood") is not False:
                raise ValueError("is_ood must be a boolean")
            is_ood = row["is_ood"]
            if is_ood and labeled_ids or not is_ood and not gold:
                raise ValueError("OOD requires no category labels; in-domain requires a gold leaf")
            expected_action = row.get("expected_action")
            if expected_action not in {"accept", "clarify", "reject"}:
                raise ValueError("invalid expected_action")
            if is_ood and expected_action != "reject":
                raise ValueError("OOD cases must expect rejection")
            if expected_action == "clarify" and len(set(clarification)) < 2:
                raise ValueError("clarification needs at least two leaf options")
            cases.append(ReviewedCategoryCase(
                case_id, query, language, split, source_group, family_id,
                gold, acceptable, clarification, is_ood, expected_action,
            ))
            seen_ids.add(case_id)
            seen_queries.add(query_key)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid reviewed case on line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("reviewed dataset is empty")
    _check_split_leakage(cases)
    return cases


def _check_split_leakage(cases: list[ReviewedCategoryCase]) -> None:
    if {case.split for case in cases} != {"train", "validation", "test"}:
        raise ValueError("reviewed dataset requires train, validation and untouched test splits")
    seen_groups: dict[str, str] = {}
    by_split: dict[str, list[tuple[str, str]]] = {}
    for case in cases:
        for kind, group in (("source", case.source_group), ("family", case.family_id)):
            key = f"{kind}:{group}"
            if key in seen_groups and seen_groups[key] != case.split:
                raise ValueError(f"{kind} group crosses reviewed data splits")
            seen_groups[key] = case.split
        normalized = normalize_text(fold_vietnamese(case.query))
        by_split.setdefault(case.split, []).append((case.id, normalized))
    splits = tuple(by_split)
    for index, left_split in enumerate(splits):
        for right_split in splits[index + 1:]:
            for left_id, left in by_split[left_split]:
                for right_id, right in by_split[right_split]:
                    if left == right:
                        raise ValueError(f"identical query crosses splits: {left_id}, {right_id}")
                    if (
                        min(len(left), len(right)) >= 20
                        and abs(len(left) - len(right)) <= max(3, int(max(len(left), len(right)) * 0.08))
                        and SequenceMatcher(None, left, right).ratio() >= 0.93
                    ):
                        raise ValueError(f"near-duplicate query crosses splits: {left_id}, {right_id}")


def _wilson_interval(positives: int, count: int, z: float = 1.96) -> tuple[float, float] | None:
    if not count:
        return None
    share = positives / count
    adjustment = z * z / count
    center = (share + adjustment / 2) / (1 + adjustment)
    radius = z * math.sqrt(share * (1 - share) / count + adjustment / (4 * count)) / (1 + adjustment)
    return max(0.0, center - radius), min(1.0, center + radius)


def _summary(rows: list[tuple[ReviewedCategoryCase, dict[str, Any]]]) -> dict[str, Any]:
    accepted = [(case, result) for case, result in rows if result["intent"]["decision"]["action"] == "accept"]
    accepted_correct = sum(
        not case.is_ood and str(result["intent"]["decision"].get("accepted_item_id")) in {*case.gold_ids, *case.acceptable_ids}
        for case, result in accepted
    )
    ood = [(case, result) for case, result in rows if case.is_ood]
    ood_false_accept = sum(result["intent"]["decision"]["action"] == "accept" for _, result in ood)
    clarifies = [(case, result) for case, result in rows if result["intent"]["decision"]["action"] == "clarify"]
    correct_clarify = sum(case.expected_action == "clarify" for case, _ in clarifies)
    option_recall: list[float] = []
    for case, result in clarifies:
        if case.expected_action != "clarify":
            continue
        expected = set(case.clarification_ids)
        offered = {str(option["item_id"]) for option in result["intent"]["decision"].get("options") or []}
        option_recall.append(len(expected & offered) / len(expected))
    top_probabilities: list[float] = []
    top_labels: list[float] = []
    domain_probabilities: list[float] = []
    domain_labels: list[float] = []
    for case, result in rows:
        intent = result["intent"]
        decision = intent["decision"]
        raw_top_id = (intent.get("confidence_assessment") or {}).get("semantic_top_item_id")
        top_correct = not case.is_ood and raw_top_id in {*case.gold_ids, *case.acceptable_ids, *case.clarification_ids}
        if decision.get("calibrated_confidence") is not None:
            top_probabilities.append(float(decision["calibrated_confidence"]))
            top_labels.append(float(top_correct))
        if decision.get("in_domain_confidence") is not None:
            domain_probabilities.append(float(decision["in_domain_confidence"]))
            domain_labels.append(float(not case.is_ood))
    return {
        "n": len(rows),
        "action_counts": dict(Counter(result["intent"]["decision"]["action"] for _, result in rows)),
        "accepted_n": len(accepted),
        "accepted_correct_n": accepted_correct,
        "accepted_precision": accepted_correct / len(accepted) if accepted else None,
        "accepted_precision_wilson_95": _wilson_interval(accepted_correct, len(accepted)),
        "accept_coverage": len(accepted) / len(rows) if rows else None,
        "ood_n": len(ood),
        "ood_false_accept_n": ood_false_accept,
        "ood_false_accept_rate": ood_false_accept / len(ood) if ood else None,
        "ood_false_accept_wilson_95": _wilson_interval(ood_false_accept, len(ood)),
        "clarify_n": len(clarifies),
        "correct_clarify_n": correct_clarify,
        "clarification_precision": correct_clarify / len(clarifies) if clarifies else None,
        "clarification_option_recall": sum(option_recall) / len(option_recall) if option_recall else None,
        "top1_reliability": probability_metrics(top_probabilities, top_labels),
        "in_domain_reliability": probability_metrics(domain_probabilities, domain_labels),
        "conclusive": (
            len(rows) >= 300 and len(ood) >= 300 and len(accepted) >= 100
            and sum(top_labels) >= 30 and len(top_labels) - sum(top_labels) >= 30
        ),
    }


def evaluate_reviewed_cases(
    runtime: Any, cases: Iterable[ReviewedCategoryCase], *, split: str = "test",
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("invalid evaluated split")
    rows: list[tuple[ReviewedCategoryCase, dict[str, Any]]] = []
    by_language: dict[str, list[tuple[ReviewedCategoryCase, dict[str, Any]]]] = {}
    by_source: dict[str, list[tuple[ReviewedCategoryCase, dict[str, Any]]]] = {}
    for case in cases:
        if case.split != split:
            continue
        result = runtime.analyze_intent(case.query, language=case.language)
        rows.append((case, result))
        by_language.setdefault(case.language, []).append((case, result))
        source = (result["intent"].get("confidence_assessment") or {}).get("final_decision_source") or "unknown"
        by_source.setdefault(source, []).append((case, result))
    if not rows:
        raise ValueError(f"no reviewed cases for split {split}")
    summary = _summary(rows)
    return {
        "split": split,
        "summary": summary,
        "by_language": {key: _summary(value) for key, value in sorted(by_language.items())},
        "by_decision_source": {key: _summary(value) for key, value in sorted(by_source.items())},
        "interpretation": (
            "descriptive review only; insufficient independent coverage for a calibrated release"
            if not summary["conclusive"] else
            "cohort sizes passed minimum checks; human review and predeclared precision/OOD gates are still required"
        ),
    }
