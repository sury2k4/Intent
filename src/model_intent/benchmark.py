from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cache import cache_key, load_matrix, save_matrix
from .confidence import load_confidence_calibration
from .decision import decide_intent, decision_to_dict, load_decision_policy
from .embedders import FakeEmbedder, SentenceTransformerEmbedder
from .evaluation import evaluate, load_cases, validate_cases
from .rendering import RENDERER_VERSION, load_enrichment, render_category_profiles
from .reporting import write_comparison, write_json, write_model_report
from .retrieval import SemanticIndex, prediction_to_dict
from .schemas import ModelProfile
from .taxonomy import Taxonomy


class _PeakRSSSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = 0.0
        self.thread: threading.Thread | None = None

    @staticmethod
    def current_mb() -> float | None:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except ImportError:
            return None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            current = self.current_mb()
            if current is not None:
                self.peak = max(self.peak, current)
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.peak = self.current_mb() or 0.0
        self.thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self.thread.start()

    def stop(self) -> float | None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        current = self.current_mb()
        if current is not None:
            self.peak = max(self.peak, current)
        return self.peak or None


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to load benchmark configuration") from exc
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
    if profile.key == "fake":
        return FakeEmbedder()
    return SentenceTransformerEmbedder(profile, local_files_only=local_files_only)


def _configure_runtime(config: dict[str, Any], local_files_only: bool) -> None:
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


def run_model(
    profile: ModelProfile,
    taxonomy: Taxonomy,
    cases,
    config: dict[str, Any],
    local_files_only: bool = False,
) -> dict[str, Any]:
    _configure_runtime(config, local_files_only)
    start = time.perf_counter()
    sampler = _PeakRSSSampler()
    sampler.start()
    result: dict[str, Any] = {
        "model_key": profile.key, "model_id": profile.model_id, "metrics": {},
        "resource": {}, "predictions": [], "decisions": [], "error": None,
    }
    try:
        embedder = make_embedder(profile, local_files_only=local_files_only)
        load_seconds = time.perf_counter() - start
        enrichment_path = config.get("enrichment_path")
        enrichment = load_enrichment(enrichment_path) if enrichment_path else {}
        policy_path = config.get("policy_path")
        if not policy_path:
            raise ValueError("config requires policy_path")
        decision_policy = load_decision_policy(policy_path)
        calibration_path = config.get("confidence_calibration_path")
        confidence_calibration = None
        if calibration_path and profile.key != "fake":
            confidence_calibration = load_confidence_calibration(
                calibration_path,
                model_key=profile.key,
                model_revision=str(embedder.metadata().get("revision") or ""),
                renderer_version=RENDERER_VERSION,
            )
        profiles = render_category_profiles(
            taxonomy,
            tuple(config.get("languages", ["vi", "en"])),
            enrichment,
            require_rich_fields=False,
        )
        embedder_metadata = embedder.metadata()
        key = cache_key(
            taxonomy.fingerprint(), RENDERER_VERSION,
            embedder_metadata,
            [(p.item_id, p.language, p.text) for p in profiles],
        )
        cache_root = Path(config.get("cache_dir", ".cache/model-intent"))
        matrix_path = cache_root / f"{profile.key}-{key}.npz"
        metadata = {
            "key": key, "model": embedder_metadata,
            "taxonomy": taxonomy.fingerprint(), "renderer": RENDERER_VERSION,
            "rows": len(profiles), "dtype": "float32",
        }
        cache_start = time.perf_counter()
        vectors = load_matrix(matrix_path, metadata)
        cache_load_seconds = time.perf_counter() - cache_start
        cache_hit = vectors is not None
        candidate_encode_seconds: float | None = None
        if vectors is None:
            encoding_start = time.perf_counter()
            vectors = embedder.encode_documents([p.text for p in profiles])
            candidate_encode_seconds = time.perf_counter() - encoding_start
            save_matrix(matrix_path, vectors, metadata)
        index = SemanticIndex(taxonomy, profiles, vectors, profile.key)
        result["resource"].update({
            "model_load_seconds": load_seconds,
            "cache_load_seconds": cache_load_seconds,
            "candidate_encode_seconds": candidate_encode_seconds,
            "cache_hit": cache_hit,
            "candidate_count": len(profiles),
            **embedder_metadata,
            "threads": int(config.get("threads", 1)),
        })
        predictions = {}
        decisions = {}
        # Decision logic needs a deeper candidate list than the displayed top-k metrics.
        top_k = max(20, int(config.get("top_k", 5)))
        query_cases = list(cases)
        for case in query_cases:
            prediction = index.search(case.query, embedder, top_k=top_k, query_id=case.id)
            predictions[case.id] = prediction
            decisions[case.id] = decide_intent(
                prediction, taxonomy, decision_policy, enrichment,
                case.language if case.language in {"vi", "en"} else None,
                confidence_calibration=confidence_calibration,
            )
        result["predictions"] = [prediction_to_dict(predictions[case.id]) for case in query_cases]
        result["decisions"] = [decision_to_dict(decisions[case.id]) | {"query_id": case.id, "query": case.query} for case in query_cases]
        result["metrics"] = evaluate(cases, predictions, (1, 3, 5), seed=int(config.get("seed", 17)), decisions=decisions)

        warmup = max(0, int(config.get("warmup_queries", 2)))
        latency_repeats = max(1, int(config.get("latency_repeats", 3)))
        latency_cases = query_cases[: max(1, min(len(query_cases), 20))]
        for case in latency_cases[:warmup]:
            index.search(case.query, embedder, top_k=top_k)
        timings_ms = []
        for _ in range(latency_repeats):
            for case in latency_cases:
                tick = time.perf_counter()
                index.search(case.query, embedder, top_k=top_k)
                timings_ms.append((time.perf_counter() - tick) * 1000)
        timings_ms.sort()
        if timings_ms:
            result["resource"].update({
                "query_p50_ms": float(np.percentile(timings_ms, 50)),
                "query_p95_ms": float(np.percentile(timings_ms, 95)),
                "query_mean_ms": float(statistics.fmean(timings_ms)),
                "timed_queries": len(timings_ms),
                "workload_ids": [case.id for case in latency_cases],
            })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["resource"]["peak_rss_mb"] = sampler.stop()
        if "embedder" in locals():
            del embedder
        gc.collect()
    return result


def _worker_command(
    taxonomy_path: str | Path,
    evaluation_path: str | Path,
    config_path: str | Path,
    model_key: str,
    result_path: str | Path,
    local_files_only: bool = False,
    include_stretch: bool = True,
) -> int:
    config = load_yaml(config_path)
    profiles = [p for p in profiles_from_config(config, include_stretch=include_stretch) if p.key == model_key]
    if not profiles:
        write_json(result_path, {"model_key": model_key, "model_id": "", "error": "unknown model profile"})
        return 2
    taxonomy = Taxonomy.load(taxonomy_path)
    cases = load_cases(evaluation_path)
    validate_cases(cases, set(taxonomy.items))
    result = run_model(profiles[0], taxonomy, cases, config, local_files_only=local_files_only)
    write_json(result_path, result)
    return 0 if result.get("error") is None else 2


def run_benchmark(
    taxonomy_path: str | Path,
    evaluation_path: str | Path,
    config_path: str | Path,
    include_stretch: bool = False,
    local_files_only: bool = False,
    only_model: str | None = None,
) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    profiles = profiles_from_config(config, include_stretch=include_stretch)
    if only_model:
        profiles = [p for p in profiles if p.key == only_model]
    if not profiles:
        raise ValueError("no enabled model profiles selected")
    output_root = Path(config.get("output_dir", "artifacts/benchmark"))
    worker_root = output_root / ".workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    results = []
    for profile in profiles:
        worker_result = worker_root / f"{profile.key}.json"
        command = [
            sys.executable, "-m", "model_intent", "benchmark", "worker",
            "--taxonomy", str(taxonomy_path), "--evaluation", str(evaluation_path),
            "--config", str(config_path), "--model", profile.key, "--result", str(worker_result),
        ]
        if local_files_only:
            command.append("--local-files-only")
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if worker_result.exists():
            result = json.loads(worker_result.read_text(encoding="utf-8"))
        else:
            result = {"model_key": profile.key, "model_id": profile.model_id, "metrics": {}, "resource": {}, "predictions": [], "error": completed.stderr[-4000:] or f"worker exit {completed.returncode}"}
        result.setdefault("resource", {})["worker_returncode"] = completed.returncode
        results.append(result)
        write_model_report(output_root / profile.key, result)
    write_comparison(output_root, results)
    return results
