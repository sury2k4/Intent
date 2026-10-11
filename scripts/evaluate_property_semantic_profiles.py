"""Evaluate authored property probes without treating cosine as calibrated accuracy."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from model_intent.serving_config import configure_runtime, load_yaml, make_embedder
from model_intent.db import Database, validate_database_target
from model_intent.embedding_sync import model_profile
from model_intent.poi_properties import load_property_profiles
from model_intent.property_pg_index import PropertyPgIndex
from model_intent.property_semantic import load_property_semantic_corpus
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "data/eval/property_semantic_probes.v1.jsonl"))
    parser.add_argument("--config", default=str(ROOT / "configs/benchmark.poi.pg.v3.yaml"))
    parser.add_argument("--taxonomy", default=str(ROOT / "data/taxonomies/place_types.poc.v3.json"))
    parser.add_argument("--semantic", default=str(ROOT / "data/properties/property_semantic_profiles.poc.v2.json"))
    parser.add_argument("--backend", choices=("pg_exact_v1",), default="pg_exact_v1")
    parser.add_argument("--generation-id", type=int, required=True, help="Pinned ready DB property generation")
    parser.add_argument("--output", help="New JSONL output; never overwrite an existing report")
    args = parser.parse_args()
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("property probe IDs must be unique")
    config = load_yaml(args.config)
    validate_database_target(config)
    os.environ["PGOPTIONS"] = (os.getenv("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()
    taxonomy = Taxonomy.load(args.taxonomy)
    path = config["property_intent"]["property_path"]
    property_path = Path(path)
    if not property_path.is_absolute():
        property_path = Path(args.config).resolve().parent.parent / property_path
    profiles = load_property_profiles(property_path, taxonomy)
    profile = model_profile(config)
    configure_runtime(config, True)
    embedder = make_embedder(profile, local_files_only=True)
    import threading
    index_lock = threading.RLock()
    corpus = load_property_semantic_corpus(args.semantic, profiles.definitions, taxonomy, embedder.metadata())
    database = Database(statement_timeout_ms=5000)
    try:
        database.open()
        with database.connection() as (connection, _):
            if connection.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] != "on":
                raise RuntimeError("property evaluation must use a read-only DB connection")
        index = PropertyPgIndex(
            database, corpus, profiles.definitions, embedder, index_lock,
            generation_id=args.generation_id,
        )
        counts = Counter()
        rows = []
        for case in cases:
            span = {"text": case["span"], "start_offset": 0, "end_offset": len(case["span"])}
            suggestion = index.suggest(
                [span], category_id=case.get("category_id"),
                taxonomy=taxonomy, profiles=profiles,
            )[0]
            candidates = [item["property_id"] for item in suggestion["candidates"]]
            expected = case["expected_property_id"]
            counts["cases"] += 1
            if expected is not None:
                counts["labeled_positive"] += 1
                counts["top1_correct"] += candidates[:1] == [expected]
                counts["top3_contains"] += expected in candidates
                if expected not in profiles.applicable_ids(taxonomy, str(case["category_id"])):
                    counts["labeled_inapplicable"] += 1
            else:
                counts["labeled_ambiguous_or_ood"] += 1
                counts["ambiguous_or_ood_with_candidates"] += bool(candidates)
            rows.append({
                **case, "backend": args.backend, "property_generation_id": args.generation_id,
                "profile_source_sha256": corpus.source_hash,
                "profile_content_set_hash": corpus.content_set_hash,
                "suggestion": suggestion,
            })
        payload = {"counts": dict(counts), "backend": args.backend, "corpus_sha256": corpus.source_hash,
                   "content_set_hash": corpus.content_set_hash,
                   "note": "authored review probes, not independent user gold or calibrated precision"}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
