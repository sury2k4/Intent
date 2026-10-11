from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from model_intent.serving_config import load_yaml
from model_intent.category_query_aliases import CategoryQueryAliases
from model_intent.db import validate_database_target
from model_intent.normalization import fold_vietnamese, normalize_text
from model_intent.poi_properties import load_property_profiles
from model_intent.rendering import load_enrichment
from model_intent.search_runtime import SearchRuntime
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark.poi.pg.v3.yaml"
TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
ENRICHMENT = ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json"
ALIASES = ROOT / "data/taxonomies/category_query_aliases.poc.v3.json"
PROPERTIES = ROOT / "data/properties/poi_properties.poc.v8.json"


def _key(text: str) -> str:
    return normalize_text(fold_vietnamese(text))


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        message = message.replace(database_url, "[DATABASE_URL redacted]")
        try:
            password = urlsplit(database_url).password
        except ValueError:
            password = None
        if password:
            for value in (password, unquote(password)):
                message = message.replace(value, "[redacted]")
    return f"{type(exc).__name__}: {message}"


def generate_cases() -> list[dict[str, Any]]:
    taxonomy = Taxonomy.load(TAXONOMY)
    enrichment = load_enrichment(ENRICHMENT)
    alias_index = CategoryQueryAliases(taxonomy, enrichment, overrides_path=ALIASES)
    profiles = load_property_profiles(PROPERTIES, taxonomy)
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(query: str, language: str, kind: str, category: str | None, **expectations: Any) -> None:
        query = query.strip()
        signature = (language, _key(query))
        if signature in seen:
            return
        seen.add(signature)
        cases.append({
            "id": f"stress-{len(cases) + 1:05d}", "query": query, "language": language,
            "kind": kind, "expected_category_id": category,
            "expected_action": expectations.pop("expected_action", "accept" if category else "reject"),
            "expected_properties": expectations.pop("expected_properties", []),
            "expected_options": expectations.pop("expected_options", []),
            "gold_provenance": expectations.pop("gold_provenance", "taxonomy_or_template_proxy_not_human_gold"),
            **expectations,
        })

    leaves = [(item_id, item) for item_id, item in taxonomy.items.items() if not taxonomy.has_children(item_id)]
    leaves.sort(key=lambda row: int(row[0]))
    short_names: dict[str, set[str]] = defaultdict(set)
    for item_id, item in leaves:
        for name in item.names.values():
            if len(_key(name)) < 4:
                short_names[_key(name)].add(item_id)
    for item_id, item in leaves:
        for language, prefix in (("vi", "tìm "), ("en", "find ")):
            canonical = item.names.get(language)
            if not canonical:
                continue
            key = _key(canonical)
            if alias_index.owners.get(key) != item_id and (len(key) >= 4 or short_names[key] != {item_id}):
                continue
            add(canonical, language, "canonical_bare", item_id)
            add(prefix + canonical, language, "canonical_prefixed", item_id)
            alternatives = [
                alias for alias in (enrichment.get(item_id) or {}).get(f"aliases_{language}", ())
                if _key(alias) != _key(canonical)
                and alias_index.owners.get(_key(alias)) == item_id
            ]
            if alternatives:
                alias = alternatives[0]
                add(alias, language, "enrichment_alias_bare", item_id,
                    gold_provenance="unreviewed_enrichment_alias_proxy")
                add(prefix + alias, language, "enrichment_alias_prefixed", item_id,
                    gold_provenance="unreviewed_enrichment_alias_proxy")

    curated = json.loads(ALIASES.read_text(encoding="utf-8"))["aliases"]
    for row in curated:
        language = row["language"]
        prefix = "tìm " if language == "vi" else "find a "
        add(row["phrase"], language, "curated_alias_bare", str(row["category_id"]),
            gold_provenance="authored_curated_alias_proxy")
        add(prefix + row["phrase"], language, "curated_alias_prefixed", str(row["category_id"]),
            gold_provenance="authored_curated_alias_proxy")

    preferred_ids = (
        "internet_access", "air_conditioning", "parking", "toilets", "wheelchair",
        "outdoor_seating", "open_24h", "quiet", "shade", "delivery", "takeaway",
    )
    for ordinal, (item_id, item) in enumerate(leaves):
        applicable = set(profiles.applicable_ids(taxonomy, item_id))
        candidates = [key for key in preferred_ids if key in applicable and key in profiles.definitions]
        if not candidates:
            continue
        name = item.names.get("vi")
        if not name:
            continue
        property_id = candidates[ordinal % len(candidates)]
        aliases = (profiles.definitions[property_id].get("aliases") or {}).get("vi") or []
        phrase = next((alias for alias in aliases if _key(alias) not in _key(name)), None)
        if phrase is None:
            continue
        required_phrase = phrase if _key(phrase).startswith("co ") else "có " + phrase
        preferred_phrase = "ưu tiên " + required_phrase
        add(f"{name}, {required_phrase}", "vi", "property_required", item_id,
            expected_properties=[{"property_id": property_id, "mode": "required"}],
            gold_provenance="taxonomy_plus_property_template_proxy")
        add(f"{name}, {preferred_phrase}", "vi", "property_preferred", item_id,
            expected_properties=[{"property_id": property_id, "mode": "preferred"}],
            gold_provenance="taxonomy_plus_property_template_proxy")

    trap_ids = ["23", "34", "42", "7", "243", "1", "40", "46", "52", "20", "114", "86", "74"]
    for item_id in trap_ids:
        name = taxonomy.items[item_id].names["vi"]
        for phrase, property_id, mode, ignored in (
            ("không có wifi", "internet_access", "excluded", None),
            ("không cần wifi", "internet_access", "preferred", "not_requested"),
            ("ưu tiên yên tĩnh", "quiet", "preferred", None),
            ("có máy lạnh", "air_conditioning", "required", None),
            ("có bãi đỗ xe", "parking", "required", None),
        ):
            if item_id == "86" and property_id == "parking":
                continue
            add(f"{name}, {phrase}", "vi", "property_trap", item_id,
                expected_properties=[{"property_id": property_id, "mode": mode, "ignored_reason": ignored}],
                gold_provenance="authored_property_trap_expectation")

    for item_id, item in leaves[:32]:
        vi = item.names.get("vi", "")
        en = item.names.get("en", "")
        if vi:
            add(f"cách mở {vi}", "vi", "ood_informational", None)
            add(f"không phải {vi}", "vi", "ood_negated_only", None)
        if en:
            add(f"how to open a {en}", "en", "ood_informational", None)
            add(f"not a {en}", "en", "ood_negated_only", None)

    authored = (
        ("nhà thương có bãi đỗ xe", "vi", "42", "accept", [{"property_id": "parking", "mode": "required"}], []),
        ("nhà thương có máy lạnh", "vi", "42", "accept", [{"property_id": "air_conditioning", "mode": "required"}], []),
        ("nhà thương hoặc nhà thuốc", "vi", None, "clarify", [], ["34", "42"]),
        ("ATM hoặc nhà băng", "vi", None, "clarify", [], ["52", "243"]),
        ("không phải nhà thương, nhà băng gần đây", "vi", "243", "accept", [], []),
        ("not a hospital, a pharmacy near me", "en", "34", "accept", [], []),
        ("not a hospital but with wifi", "en", None, "reject", [], []),
        ("không phải nhà thương, có bãi đỗ xe", "vi", None, "reject", [], []),
        ("nhà thương mại", "vi", None, "reject", [], []),
        ("tìm phần mềm quản lý bệnh viện", "vi", None, "reject", [], []),
        ("thời tiết ngày mai như thế nào", "vi", None, "reject", [], []),
    )
    for query, language, category, action, properties, options in authored:
        add(query, language, "authored_adversarial", category,
            expected_action=action, expected_properties=properties, expected_options=options,
            gold_provenance="authored_adversarial_expectation_not_independent_gold")
    return cases


def _iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _generate(path: Path) -> None:
    cases = generate_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case["kind"]] += 1
    print(json.dumps({"cases": len(cases), "kinds": counts, "sha256": _fingerprint(path)}, ensure_ascii=False))


def _run(
    cases_path: Path, output_path: Path, shard: int, shards: int,
    *, config_path: Path = CONFIG, taxonomy_path: Path = TAXONOMY,
) -> None:
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("invalid shard index")
    config = load_yaml(config_path)
    if config.get("retrieval_backend") != "pgvector" or config.get("retrieval_mode") != "pg_exact":
        raise RuntimeError("stress test requires exact pgvector serving")
    validate_database_target(config)
    os.environ["PGOPTIONS"] = (os.getenv("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()
    cases = list(_iter_jsonl(cases_path))
    selected = [case for case in cases if int(hashlib.sha256(case["id"].encode()).hexdigest(), 16) % shards == shard]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = SearchRuntime.from_intent_config(
        config_path=config_path, taxonomy_path=taxonomy_path,
        model_key="e5-base-multilingual", local_files_only=True,
    )
    try:
        meta = runtime.category_runtime.metadata
        if meta.backend != "pgvector" or meta.retrieval_mode != "pg_exact" or meta.generation_id is None:
            raise RuntimeError("stress test did not acquire an active exact pgvector category generation")
        with runtime.category_runtime.database.connection() as (connection, _):
            if connection.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] != "on":
                raise RuntimeError("DB connection is not read-only")
        with output_path.open("x", encoding="utf-8", buffering=1) as output:
            for index, case in enumerate(selected, 1):
                row = {
                    "id": case["id"], "query": case["query"], "language": case["language"],
                    "kind": case["kind"], "expected_category_id": case["expected_category_id"],
                    "expected_action": case["expected_action"],
                    "expected_properties": case["expected_properties"],
                    "expected_options": case["expected_options"],
                    "gold_provenance": case["gold_provenance"],
                    "category_generation_id": meta.generation_id,
                    "backend": meta.backend, "retrieval_mode": meta.retrieval_mode,
                }
                try:
                    result = runtime.analyze_intent(case["query"], language=case["language"])
                    intent = result["intent"]
                    decision = intent["decision"]
                    row.update({
                        "action": decision.get("action"), "reason": decision.get("reason"),
                        "accepted_category_id": decision.get("accepted_item_id"),
                        "clarification_ids": [str(option["item_id"]) for option in decision.get("options") or []],
                        "top_hits": [
                            {"id": str(hit["item_id"]), "score": hit["score"]}
                            for hit in (intent.get("prediction") or {}).get("hits") or []
                        ],
                        "confidence": decision.get("calibrated_confidence"),
                        "confidence_quality": decision.get("confidence_quality"),
                        "decision_source": (intent.get("confidence_assessment") or {}).get("final_decision_source"),
                        "category_text": (intent.get("query_interpretation") or {}).get("category_text"),
                        "property_roles": (intent.get("query_interpretation") or {}).get("property_roles"),
                        "properties": result["property_extraction"]["properties"],
                        "extraction_warnings": result["property_extraction"].get("warnings"),
                        "timing_total_ms": (result.get("timing") or {}).get("total_ms"),
                    })
                except Exception as exc:  # noqa: BLE001 - record every failed case, then continue the shard
                    row["error"] = _safe_error(exc)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                if index % 100 == 0:
                    print(f"shard {shard}/{shards}: {index}/{len(selected)}", file=sys.stderr, flush=True)
    finally:
        runtime.close()
    print(json.dumps({"shard": shard, "shards": shards, "written": len(selected),
                      "generation_id": meta.generation_id, "output": str(output_path)}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic category/property stress test against read-only exact pgvector")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--config", type=Path, default=CONFIG)
    run.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--shards", type=int, required=True)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    if args.command == "generate":
        _generate(args.output)
    else:
        _run(
            args.cases, args.output, args.shard, args.shards,
            config_path=args.config, taxonomy_path=args.taxonomy,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
