from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                yield json.loads(line)


def _evaluation_row(*, case_id: str, split: str, query: str, track: str,
                    gold_ids=(), clarification_ids=(), expected_action: str = "accept",
                    language: str = "vi", notes: str = "", is_ood: bool = False) -> dict:
    return {
        "id": case_id, "track": track, "split": split, "query": query,
        "language": language, "gold_ids": list(gold_ids),
        "clarification_ids": list(clarification_ids),
        "expected_action": expected_action, "is_ood": is_ood, "notes": notes,
    }


def make_cases(taxonomy_path: Path, enrichment_path: Path, legacy_path: Path, poi_path: Path) -> list[dict]:
    taxonomy = Taxonomy.load(taxonomy_path)
    old = Taxonomy.load(ROOT / "data/taxonomies/place_types.poc.v2.json")
    enrichment = json.loads(enrichment_path.read_text(encoding="utf-8"))["items"]
    cases: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(row: dict) -> None:
        query = str(row["query"]).strip()
        key = (str(row.get("language", "vi")), " ".join(query.casefold().split()))
        if query and key not in seen:
            seen.add(key)
            cases.append(row)

    for row in _load_jsonl(legacy_path):
        copied = dict(row)
        copied["id"] = f"legacy-{row['id']}"
        copied["notes"] = f"Existing authored case; {row.get('notes', '')}".strip()
        add(copied)

    for index, row in enumerate(_load_jsonl(poi_path)):
        action = str(row.get("expected_action", "accept"))
        expected_id = row.get("expected_category_id")
        options = row.get("expected_options") or []
        if action == "accept" and not expected_id:
            continue
        if action == "clarify" and len(options) < 2:
            continue
        add(_evaluation_row(
            case_id=f"poi-{row['id']}", split="calibration" if index % 4 == 0 else "test",
            query=str(row["query"]), language=str(row.get("language", "vi")),
            track="authored_poi", gold_ids=[str(expected_id)] if expected_id else (),
            clarification_ids=[str(value) for value in options], expected_action=action,
            notes="Existing authored POI regression example; not independent population gold",
            is_ood=action == "reject",
        ))

    new_leaf_ids = sorted(
        (item_id for item_id in taxonomy.items if item_id not in old.items and not taxonomy.has_children(item_id)),
        key=int,
    )
    for item_id in new_leaf_ids:
        item = taxonomy.items[item_id]
        profile = enrichment[item_id]
        for language in ("vi", "en"):
            name = item.names.get(language, "").strip()
            aliases = [str(alias).strip() for alias in profile[f"aliases_{language}"] if str(alias).strip()]
            train = list(dict.fromkeys([name, *aliases[:2]]))[:3]
            validation = list(dict.fromkeys([*aliases[2:4], name]))[:3]
            for split, values in (("calibration", train), ("test", validation)):
                for index, value in enumerate(values):
                    query = f"{'tìm' if language == 'vi' else 'find'} {value}" if split == "calibration" else (
                        f"{'địa điểm' if language == 'vi' else 'place'} {value} {'ở Sài Gòn' if language == 'vi' else 'in HCMC'}"
                    )
                    add(_evaluation_row(
                        case_id=f"osm-{item_id}-{language}-{split}-{index}", split=split,
                        query=query, language=language, track="synthetic_category_alias",
                        gold_ids=[item_id],
                        notes="Synthetic from taxonomy alias; provisional proxy, NOT adjudicated category gold",
                    ))

    for index, query in enumerate((
        "mai trời có mưa không", "how do I reset my password", "tại sao đơn hàng bị hủy",
        "hãy viết một bài thơ", "thời tiết tuần sau thế nào", "what time is the football match",
        "cách lập tài khoản mới", "why did my account get locked",
    )):
        add(_evaluation_row(
            case_id=f"synthetic-ood-{index}", split="calibration" if index % 2 == 0 else "test",
            query=query, track="synthetic_ood", expected_action="reject", is_ood=True,
            language="en" if index in (1, 5, 7) else "vi",
            notes="Synthetic non-place query; provisional, NOT independent OOD gold",
        ))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Build explicitly provisional category eval cases for an expanded OSM taxonomy")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "data/taxonomies/place_types.poc.v3.json")
    parser.add_argument("--enrichment", type=Path, default=ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json")
    parser.add_argument("--legacy", type=Path, default=ROOT / "data/eval/place_type_queries.poc.v1.jsonl")
    parser.add_argument("--poi", type=Path, default=ROOT / "data/eval/hcmc_poi_queries.v3.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval/hcmc_category_provisional.v3.jsonl")
    args = parser.parse_args()
    rows = make_cases(args.taxonomy, args.enrichment, args.legacy, args.poi)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different evaluation cases: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    splits = {split: sum(row["split"] == split for row in rows) for split in ("calibration", "test")}
    print(json.dumps({"output": str(args.output), "cases": len(rows), "new_categories": len({row["gold_ids"][0] for row in rows if row["track"] == "synthetic_category_alias"}), "splits": splits, "warning": "synthetic aliases are provisional proxies, not independent gold"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
