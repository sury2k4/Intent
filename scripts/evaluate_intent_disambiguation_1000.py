# -*- coding: utf-8 -*-
"""Run intent disambiguation evaluation benchmark on 1000 ambiguous queries."""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_QUERIES = ROOT / "data/eval/ambiguous_1000_queries.json"
CONFIG_PATH = ROOT / "configs/benchmark.cpu.yaml"
TAXONOMY_PATH = ROOT / "data/taxonomies/place_types.poc.v1.json"
OUTPUT_REPORT = ROOT / "artifacts/benchmark/intent_disambiguation_1000_results.json"
OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


def main():
    from model_intent.runtime import MatcherRuntime

    print(f"Loading 1000 ambiguous queries from {BENCHMARK_QUERIES}...")
    queries = json.loads(BENCHMARK_QUERIES.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} queries.")

    print("Initializing MatcherRuntime (intfloat/multilingual-e5-base on pgvector)...")
    init_start = time.perf_counter()
    runtime = MatcherRuntime.from_config(
        config_path=CONFIG_PATH,
        taxonomy_path=TAXONOMY_PATH,
        model_key="e5-base-multilingual",
        local_files_only=True,
    )
    print(f"Runtime initialized in {time.perf_counter() - init_start:.2f}s.")

    results = []
    action_counts = Counter()
    domain_stats = defaultdict(lambda: Counter())
    latencies = []
    clarify_recall_hits = 0
    total_clarify = 0
    accept_precision_hits = 0
    total_accept = 0

    print("Running 1000 queries through the disambiguation engine...")
    eval_start = time.perf_counter()

    for idx, item in enumerate(queries, 1):
        q = item["query"]
        candidates = set(str(c) for c in item.get("candidate_ids", []))
        domain = item.get("domain", "Unknown")

        t0 = time.perf_counter()
        resp = runtime.query(q, top_k=5, language="vi")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        decision = resp.get("decision", {})
        prediction = resp.get("prediction", {})
        action = decision.get("action", "reject")
        reason = decision.get("reason", "unknown")
        accepted_id = str(decision.get("accepted_item_id") or "")
        options = decision.get("options", [])
        option_ids = [str(opt.get("item_id")) for opt in options]

        action_counts[action] += 1
        domain_stats[domain][action] += 1

        is_hit = False
        if action == "accept":
            total_accept += 1
            if accepted_id in candidates:
                accept_precision_hits += 1
                is_hit = True
        elif action == "clarify":
            total_clarify += 1
            # Check if any candidate is offered in clarification options
            if candidates.intersection(set(option_ids)):
                clarify_recall_hits += 1
                is_hit = True

        results.append({
            "id": idx,
            "query": q,
            "domain": domain,
            "candidates": list(candidates),
            "action": action,
            "reason": reason,
            "accepted_id": accepted_id,
            "accepted_code": decision.get("accepted_code"),
            "question": decision.get("question"),
            "options": option_ids,
            "top1_score": prediction.get("top1_score"),
            "margin": prediction.get("margin"),
            "is_valid_resolution": is_hit,
            "elapsed_ms": round(elapsed_ms, 2),
        })

        if idx % 100 == 0:
            print(f"Processed {idx:4d}/1000 queries... [Accept: {action_counts['accept']}, Clarify: {action_counts['clarify']}, Reject: {action_counts['reject']}]")

    total_time = time.perf_counter() - eval_start
    avg_latency = sum(latencies) / len(latencies)
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(len(latencies) * 0.50)]
    p90 = sorted_latencies[int(len(latencies) * 0.90)]
    p99 = sorted_latencies[int(len(latencies) * 0.99)]

    accept_rate = action_counts["accept"] / len(queries)
    clarify_rate = action_counts["clarify"] / len(queries)
    reject_rate = action_counts["reject"] / len(queries)

    accept_accuracy = (accept_precision_hits / total_accept) if total_accept else 0.0
    clarify_recall = (clarify_recall_hits / total_clarify) if total_clarify else 0.0
    overall_resolution_rate = (accept_precision_hits + clarify_recall_hits) / len(queries)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY: 1000 AMBIGUOUS QUERIES EVALUATION")
    print("=" * 80)
    print(f"Total Queries Evaluated:    {len(queries)}")
    print(f"Total Execution Time:       {total_time:.2f}s ({len(queries) / total_time:.1f} queries/sec)")
    print(f"Latency P50 / P90 / P99:    {p50:.1f}ms / {p90:.1f}ms / {p99:.1f}ms (Avg: {avg_latency:.1f}ms)")
    print("-" * 80)
    print("ACTION DISTRIBUTION:")
    print(f"  - ACCEPT:                 {action_counts['accept']:4d} ({accept_rate * 100:.1f}%)")
    print(f"  - CLARIFY (Disambiguate): {action_counts['clarify']:4d} ({clarify_rate * 100:.1f}%)")
    print(f"  - REJECT:                 {action_counts['reject']:4d} ({reject_rate * 100:.1f}%)")
    print("-" * 80)
    print("DISAMBIGUATION EFFECTIVENESS:")
    print(f"  - Accept Accuracy:        {accept_accuracy * 100:.1f}% ({accept_precision_hits}/{total_accept})")
    print(f"  - Clarify Recall:         {clarify_recall * 100:.1f}% ({clarify_recall_hits}/{total_clarify})")
    print(f"  - Overall Sound Routing:  {overall_resolution_rate * 100:.1f}%")
    print("-" * 80)
    print("DOMAIN-LEVEL BREAKDOWN:")
    for dom, counts in sorted(domain_stats.items()):
        total_dom = sum(counts.values())
        print(f"  * {dom:<18}: Total {total_dom:3d} | Accept: {counts['accept']:3d} ({counts['accept']/total_dom*100:4.1f}%) | Clarify: {counts['clarify']:3d} ({counts['clarify']/total_dom*100:4.1f}%) | Reject: {counts['reject']:3d}")

    # Sample Cases of Clarification
    print("-" * 80)
    print("SAMPLE CLARIFICATION DISAMBIGUATIONS:")
    clarify_samples = [r for r in results if r["action"] == "clarify"][:5]
    for s in clarify_samples:
        print(f"\n[Query]: \"{s['query']}\"")
        print(f"  -> Disambiguation Question: {s['question']}")
        print(f"  -> Options Offered (IDs):   {s['options']}")
        print(f"  -> Target Candidates:      {s['candidates']}")
        print(f"  -> Top 1 Score: {s['top1_score']:.4f}, Margin: {s['margin']:.4f}")

    # Sample Cases of Clear Direct Acceptance
    print("-" * 80)
    print("SAMPLE DIRECT ACCEPTANCES (Subtle Nuance Detected):")
    accept_samples = [r for r in results if r["action"] == "accept"][:5]
    for s in accept_samples:
        print(f"\n[Query]: \"{s['query']}\"")
        print(f"  -> Accepted Code:           {s['accepted_code']} (ID {s['accepted_id']})")
        print(f"  -> Reason:                  {s['reason']}")
        print(f"  -> Top 1 Score:             {s['top1_score']:.4f}, Margin: {s['margin']:.4f}")

    # Save artifact json
    summary_report = {
        "benchmark_metadata": {
            "model": "intfloat/multilingual-e5-base",
            "dimension": 768,
            "vector_backend": "pgvector",
            "total_queries": len(queries),
            "total_time_seconds": round(total_time, 2),
            "throughput_qps": round(len(queries) / total_time, 1),
            "latency": {
                "avg_ms": round(avg_latency, 2),
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "p99_ms": round(p99, 2),
            }
        },
        "metrics": {
            "accept_count": action_counts["accept"],
            "accept_rate": round(accept_rate, 4),
            "accept_accuracy": round(accept_accuracy, 4),
            "clarify_count": action_counts["clarify"],
            "clarify_rate": round(clarify_rate, 4),
            "clarify_recall": round(clarify_recall, 4),
            "reject_count": action_counts["reject"],
            "reject_rate": round(reject_rate, 4),
            "overall_resolution_rate": round(overall_resolution_rate, 4),
        },
        "domain_breakdown": {
            dom: dict(counts) for dom, counts in domain_stats.items()
        },
        "sample_clarifications": clarify_samples,
        "sample_acceptances": accept_samples,
    }
    OUTPUT_REPORT.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed benchmark metrics saved to {OUTPUT_REPORT.resolve()}")


if __name__ == "__main__":
    main()
