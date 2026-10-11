from __future__ import annotations

# ruff: noqa: ISC004
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
CONFIG = ROOT / "configs/benchmark.poi.pg.v3.yaml"
ALIASES = ROOT / "data/taxonomies/category_query_aliases.poc.v3.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator * 100:.1f}%" if denominator else "n/a"


def _ms(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "'")


def _quantile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _status(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "error"
    if row["expected_action"] == "accept":
        if row.get("action") == "accept":
            return "correct_accept" if str(row.get("accepted_category_id")) == str(row["expected_category_id"]) else "wrong_accept"
        return row.get("action") or "no_action"
    if row["expected_action"] == "clarify":
        actual = set(row.get("clarification_ids") or ())
        expected = set(row.get("expected_options") or ())
        return "correct_clarify" if row.get("action") == "clarify" and actual == expected else "clarify_mismatch"
    return "correct_reject" if row.get("action") == "reject" else "ood_false_accept" if row.get("action") == "accept" else "ood_other"


def _property_status(row: dict[str, Any]) -> list[dict[str, Any]]:
    found = row.get("properties") or []
    results = []
    for expected in row["expected_properties"]:
        candidates = [item for item in found if item.get("property_id") == expected["property_id"]]
        matching = [item for item in candidates if item.get("mode") == expected["mode"]]
        if not candidates:
            outcome = "missing"
        elif not matching:
            outcome = "wrong_mode"
        elif expected.get("ignored_reason") and not any(
            item.get("ignored_reason") == expected["ignored_reason"] for item in matching
        ):
            outcome = "wrong_ignored_reason"
        else:
            outcome = "matched"
        if not matching:
            application = "not_extracted"
        elif expected.get("ignored_reason") and any(
            item.get("ignored_reason") == expected["ignored_reason"] for item in matching
        ):
            application = "expected_ignored"
        elif any(item.get("applied") and not item.get("ignored_reason") for item in matching):
            application = "applied"
        else:
            application = str(matching[0].get("ignored_reason") or "not_applied_unknown")
        results.append({"property_id": expected["property_id"], "outcome": outcome, "application": application})
    return results


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(_status(row) for row in rows)
    positives = sum(row["expected_action"] == "accept" for row in rows)
    errors = counts["error"]
    top1 = sum(
        row["expected_category_id"] is not None and row.get("top_hits")
        and str(row["top_hits"][0]["id"]) == str(row["expected_category_id"])
        for row in rows if not row.get("error")
    )
    property_checks = [item for row in rows for item in _property_status(row)]
    props = Counter(item["outcome"] for item in property_checks)
    applications = Counter(item["application"] for item in property_checks)
    latency = [float(row["timing_total_ms"]) for row in rows if row.get("timing_total_ms") is not None]
    return {
        "n": len(rows), "errors": errors, "expected_accept": positives,
        "correct_accept": counts["correct_accept"], "wrong_accept": counts["wrong_accept"],
        "reject_when_expected_accept": counts["reject"], "clarify_when_expected_accept": counts["clarify"],
        "top1_expected_id": top1, "expected_reject": sum(row["expected_action"] == "reject" for row in rows),
        "correct_reject": counts["correct_reject"], "ood_false_accept": counts["ood_false_accept"],
        "expected_clarify": sum(row["expected_action"] == "clarify" for row in rows),
        "correct_clarify": counts["correct_clarify"], "property_checks": len(property_checks),
        "property_outcomes": dict(props), "property_application": dict(applications),
        "latency_p50_ms": _quantile(latency, 0.5),
        "latency_p95_ms": _quantile(latency, 0.95),
        "status_counts": dict(counts),
    }


def summarize(cases: list[dict[str, Any]], rows: list[dict[str, Any]], *, shards: int) -> dict[str, Any]:
    by_id = {row["id"]: row for row in rows}
    case_ids = {case["id"] for case in cases}
    if len(by_id) != len(rows) or set(by_id) != case_ids:
        raise ValueError(f"shards incomplete or duplicate: expected {len(cases)}, received {len(rows)} rows/{len(by_id)} IDs")
    if any(row.get("backend") != "pgvector" or row.get("retrieval_mode") != "pg_exact" for row in rows):
        raise ValueError("some results did not use exact pgvector")
    generation_ids = {row.get("category_generation_id") for row in rows}
    if len(generation_ids) != 1 or None in generation_ids:
        raise ValueError(f"results mixed category generations: {generation_ids}")
    ordered = [{**case, **by_id[case["id"]]} for case in cases]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_origin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_property: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_kind[row["kind"]].append(row)
        by_language[row["language"]].append(row)
        by_author[str(row.get("author") or "taxonomy_generator")].append(row)
        by_origin["paraphrase" if row.get("derived_from") else "agent_seed"].append(row)
        if row["expected_category_id"] is not None:
            by_category[str(row["expected_category_id"])].append(row)
        for item in row["expected_properties"]:
            by_property[item["property_id"]].append(row)
    return {
        "cases_sha256": None, "shards": shards, "category_generation_id": next(iter(generation_ids)),
        "overall": _group_summary(ordered),
        "by_kind": {name: _group_summary(part) for name, part in sorted(by_kind.items())},
        "by_language": {name: _group_summary(part) for name, part in sorted(by_language.items())},
        "by_author": {name: _group_summary(part) for name, part in sorted(by_author.items())},
        "by_origin": {name: _group_summary(part) for name, part in sorted(by_origin.items())},
        "by_category": {name: _group_summary(part) for name, part in sorted(by_category.items(), key=lambda x: int(x[0]))},
        "by_property": {name: _group_summary(part) for name, part in sorted(by_property.items())},
        "rows": ordered,
    }


def _write_results(
    folder: Path, report_path: Path, summary: dict[str, Any], taxonomy: Taxonomy,
    *, corpus_kind: str = "taxonomy_proxy",
) -> None:
    rows = summary.pop("rows")
    with (folder / "all_results.jsonl").open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    failures = [row for row in rows if _status(row) not in {"correct_accept", "correct_reject", "correct_clarify"}
                or any(item["outcome"] != "matched" or item["application"] not in {"applied", "expected_ignored"}
                       for item in _property_status(row))]
    with (folder / "failures.jsonl").open("x", encoding="utf-8") as output:
        for row in failures:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (folder / "per_category.csv").open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("id", "name_vi", "name_en", "cases", "correct_accept", "wrong_accept", "reject", "clarify", "top1_expected_id", "property_checks", "property_matched"))
        writer.writeheader()
        for category_id, data in summary["by_category"].items():
            item = taxonomy.items[category_id]
            writer.writerow({
                "id": category_id, "name_vi": item.names.get("vi"), "name_en": item.names.get("en"),
                "cases": data["n"], "correct_accept": data["correct_accept"], "wrong_accept": data["wrong_accept"],
                "reject": data["reject_when_expected_accept"], "clarify": data["clarify_when_expected_accept"],
                "top1_expected_id": data["top1_expected_id"], "property_checks": data["property_checks"],
                "property_matched": data["property_outcomes"].get("matched", 0),
            })
    with (folder / "summary.json").open("x", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)

    overall = summary["overall"]
    is_natural = corpus_kind == "agent_authored_natural"
    lines = [
        "# Kiểm thử query mô phỏng người dùng trên PostgreSQL/pgvector thật"
        if is_natural else "# Kiểm thử category + properties trên PostgreSQL/pgvector thật",
        "",
        "**Quan trọng:** Câu hỏi do agent mô phỏng cách nói của người dùng và tự gắn nhãn kỳ vọng; "
        "đây KHÔNG PHẢI log người dùng thật hoặc gold độc lập. Các tỷ lệ chỉ đánh giá corpus mô phỏng, "
        "không đại diện chất lượng ngoài đời."
        if is_natural else
        "**Quan trọng:** Các nhãn kỳ vọng dưới đây được tạo từ taxonomy, alias và template (proxy), "
        "không phải nhãn độc lập do người dùng gán. Các tỷ lệ không đại diện độ chính xác ngoài thực tế.",
        "",
        "## Môi trường và phạm vi",
        "",
        f"- Backend: **pgvector pg_exact** trên unified DB; E5-base 768D, category generation **{summary['category_generation_id']}**.",
        f"- Số câu: **{overall['n']:,}** trên **{len(summary['by_category'])}/384** leaf; "
        f"{summary['shards']} nhánh agent độc lập. Kết quả JSONL: `{folder.as_posix()}/all_results.jsonl`.",
        "- Nguồn truy vấn: agent viết câu hội thoại Việt/Anh cho category, tên gọi thay thế và bẫy properties/OOD; "
        "một số câu accept được tạo thêm biến thể hội thoại/vị trí từ seed. Đây không phải ghi nhận hành vi người dùng thật."
        if is_natural else
        "- Nguồn truy vấn: tên chuẩn, alias Việt/Anh có sẵn, alias được biên soạn, thuộc tính bắt buộc/ưu tiên, "
        "phủ định, OOD có từ category, và bẫy nhiều category. Truy vấn vùng miền chỉ gồm các cách gọi đã có trong danh mục; "
        "không suy diễn bao phủ mọi phương ngữ.",
        f"- SHA-256 corpus: `{summary['cases_sha256']}`. Config: `{summary['config_sha256']}`; "
        f"taxonomy: `{summary['taxonomy_sha256']}`; alias: `{summary['aliases_sha256']}`.",
        "- Kết nối được ép `default_transaction_read_only=on` và kiểm tra trước khi chạy; không migrate/sync/promote POI hay embedding.",
        "",
        "## Kết quả tổng hợp (proxy)",
        "",
        f"- Accept đúng leaf theo nhãn proxy: **{overall['correct_accept']:,}/{overall['expected_accept']:,} "
        f"({_percent(overall['correct_accept'], overall['expected_accept'])})**; "
        f"accept sai: **{overall['wrong_accept']}**; không accept (reject/clarify): "
        f"**{overall['reject_when_expected_accept']}/{overall['clarify_when_expected_accept']}**.",
        "- Top-1 E5 trùng leaf kỳ vọng (trong các ca kỳ vọng category): "
        f"**{overall['top1_expected_id']:,}/{overall['expected_accept']:,}** — phép đo này có thể là "
        "tên chuẩn sau alias probe, không nhất thiết là score nguyên văn query.",
        f"- OOD/truy vấn cần từ chối: **{overall['correct_reject']}/{overall['expected_reject']}** reject, "
        f"**{overall['ood_false_accept']}** accept sai (nếu có phải xem từng câu trong `failures.jsonl`).",
        f"- Câu cần làm rõ: **{overall['correct_clarify']}/{overall['expected_clarify']}** có đúng tập lựa chọn theo template.",
        "- Property ID/mode được bóc tách theo template: "
        f"**{overall['property_outcomes'].get('matched',0)}/{overall['property_checks']}**; "
        f"missing={overall['property_outcomes'].get('missing',0)}, "
        f"wrong_mode={overall['property_outcomes'].get('wrong_mode',0)}, "
        f"sai lý do bỏ qua={overall['property_outcomes'].get('wrong_ignored_reason',0)}.",
        f"- Trong {overall['property_checks']} property đã kiểm tra: "
        f"applied={overall['property_application'].get('applied',0)}, "
        f"bỏ qua đúng yêu cầu={overall['property_application'].get('expected_ignored',0)}, "
        f"category_inapplicable={overall['property_application'].get('category_inapplicable',0)}, "
        f"category_component={overall['property_application'].get('category_component',0)}, "
        f"conflicting_constraints={overall['property_application'].get('conflicting_constraints',0)}. "
        "Bóc tách đúng ID/mode **không có nghĩa** điều kiện đã được áp dụng hoặc có bằng chứng POI.",
        f"- Lỗi thực thi: **{overall['errors']}**. Timing mỗi request trong các nhánh chạy đồng thời: "
        f"p50={_ms(overall['latency_p50_ms'])} ms, p95={_ms(overall['latency_p95_ms'])} ms; "
        "không phải latency của UI/browser hay tải production.",
        "",
        "## Theo loại query",
        "",
        "| Track | Số câu | Accept đúng/kỳ vọng | Accept sai | Reject / Clarify thay accept | OOD accept sai | Property khớp/kỳ vọng |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in summary["by_kind"].items():
        lines.append(
            f"| `{name}` | {data['n']} | {data['correct_accept']}/{data['expected_accept']} | "
            f"{data['wrong_accept']} | {data['reject_when_expected_accept']} / {data['clarify_when_expected_accept']} | "
            f"{data['ood_false_accept']} | {data['property_outcomes'].get('matched',0)}/{data['property_checks']} |"
        )
    lines.extend(["", "## Theo ngôn ngữ", "", "| Ngôn ngữ | Số câu | Accept đúng/kỳ vọng | OOD accept sai |", "|---|---:|---:|---:|"])
    for name, data in summary["by_language"].items():
        lines.append(f"| `{name}` | {data['n']} | {data['correct_accept']}/{data['expected_accept']} | {data['ood_false_accept']} |")
    lines.extend(["", "## Câu gốc do agent viết so với biến thể mở rộng", "", "| Nguồn | Số câu | Accept đúng/kỳ vọng proxy | OOD accept sai | Property applied/kỳ vọng |", "|---|---:|---:|---:|---:|"])
    for name, data in summary["by_origin"].items():
        lines.append(
            f"| `{name}` | {data['n']} | {data['correct_accept']}/{data['expected_accept']} | "
            f"{data['ood_false_accept']} | {data['property_application'].get('applied',0)}/{data['property_checks']} |"
        )
    lines.extend(["", "## Theo agent tác giả", "", "| Agent | Số câu | Accept đúng/kỳ vọng proxy | OOD accept sai | Property applied/kỳ vọng |", "|---|---:|---:|---:|---:|"])
    for name, data in summary["by_author"].items():
        lines.append(
            f"| `{name}` | {data['n']} | {data['correct_accept']}/{data['expected_accept']} | "
            f"{data['ood_false_accept']} | {data['property_application'].get('applied',0)}/{data['property_checks']} |"
        )
    lines.extend(["", "## Theo thuộc tính", "", "| Property | Số case | ID/mode đúng | Áp dụng | Bỏ qua (kể cả đúng theo yêu cầu) | Thiếu/sai mode |", "|---|---:|---:|---:|---:|---:|"])
    for name, data in summary["by_property"].items():
        outcomes = data["property_outcomes"]
        applications = data["property_application"]
        ignored = data["property_checks"] - applications.get("applied", 0)
        lines.append(f"| `{name}` | {data['n']} | {outcomes.get('matched',0)}/{data['property_checks']} | {applications.get('applied',0)} | {ignored} | {outcomes.get('missing',0) + outcomes.get('wrong_mode',0)} |")
    ordered_categories = sorted(summary["by_category"].items(), key=lambda item: (item[1]["correct_accept"] / max(1, item[1]["expected_accept"]), -item[1]["n"]))
    lines.extend(["", "## Category có tỷ lệ accept đúng thấp nhất (proxy)", "", "| ID | Category | Đúng / kỳ vọng | Sai category | Reject | Clarify |", "|---:|---|---:|---:|---:|---:|"])
    for category_id, data in ordered_categories[:25]:
        lines.append(f"| {category_id} | {_markdown(taxonomy.items[category_id].names.get('vi'))} | {data['correct_accept']}/{data['expected_accept']} | {data['wrong_accept']} | {data['reject_when_expected_accept']} | {data['clarify_when_expected_accept']} |")
    lines.extend(["", f"Bảng đủ **{len(ordered_categories)} leaf**: `{folder.as_posix()}/per_category.csv`.", "", "## Lỗi đáng xem trước", ""])
    sorted_failures = sorted(failures, key=lambda row: (
        0 if _status(row) in {"wrong_accept", "ood_false_accept"} else 1,
        0 if row["kind"].startswith("authored") or row["kind"] == "property_trap" else 1,
        row["id"],
    ))
    lines.extend(["| ID | Track | Query | Kỳ vọng → Kết quả | Top E5 | Property miss |", "|---|---|---|---|---|---|"])
    for row in sorted_failures[:45]:
        hit = (row.get("top_hits") or [{}])[0]
        missed = ",".join(
            f"{part['property_id']}:{part['outcome'] if part['outcome'] != 'matched' else part['application']}"
            for part in _property_status(row)
            if part["outcome"] != "matched" or part["application"] not in {"applied", "expected_ignored"}
        )
        lines.append(
            f"| {row['id']} | `{row['kind']}` | {_markdown(row['query'])} | "
            f"{row['expected_action']}:{row['expected_category_id']} → {row.get('action')}:{row.get('accepted_category_id')} "
            f"({row.get('reason','error')}) | {hit.get('id', 'n/a')} | {missed or '-'} |"
        )
    lines.extend([
        "", f"Tất cả **{len(failures)} ca không đạt kỳ vọng** (gồm property sai dù category đúng): "
        f"`{folder.as_posix()}/failures.jsonl`. Mỗi dòng kết quả gốc gồm query, E5 top-5, action, "
        "category, properties/ignored_reason và timing, đối chiếu được bằng ID với `cases.jsonl`.",
        "", "## Giới hạn diễn giải và việc tiếp theo", "",
        "1. Nhãn kỳ vọng của query tự nhiên do agent tạo, không được reviewer độc lập xác nhận; "
        "các biến thể câu có cùng giả định với câu gốc và vì thế không phải quan sát độc lập. "
        "Case sai cần con người thẩm định trước khi sửa taxonomy/policy."
        if is_natural else
        "1. Nhãn tên/alias và tính phù hợp thuộc tính là **proxy** từ chính taxonomy/profiles; alias OSM có thể không phải "
        "từ đồng nghĩa thực. Một label sai tạo ra 'lỗi' giả; nhận đúng alias đã có cũng không chứng minh hiểu cách nói mới.",
        "2. Property matching chỉ kiểm tra các ID/mode tác giả *có ghi*, không tính false-positive ngoài nhãn; "
        "`applied` không đồng nghĩa có POI thỏa yêu cầu. OOD mô phỏng quá ít và có thể bị gán gold sai; "
        "không suy ra tỷ lệ reject của traffic thực."
        if is_natural else
        "2. Case ngoài hệ thống chỉ là những mẫu OOD tổng hợp; tỷ lệ reject không đại diện lưu lượng thực. "
        "Các property mẫu được tạo bằng template; đặc biệt 'không', phủ định category, overlap và nhu cầu ngầm "
        "cần người thẩm định. Không coi `calibrated_confidence` là xác suất đúng trên thực tế.",
        "3. Query thực do người dùng cung cấp cần được khử thông tin cá nhân và gán nhãn độc lập trước khi "
        "điều chỉnh model/calibration. Những case ngoài category chưa được gán gold chỉ có thể dùng làm stress probe.",
        "4. Kết quả ở đây chạy **category/profile pgvector thật**, không tìm hoặc ranking POI; UI intent-only cũng dùng "
        "chính đường phân tích này. Không promote generation DB hay cập nhật embedding.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "n": overall["n"], "failed_cases": len(failures), "errors": overall["errors"]}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize complete multi-agent exact pgvector category stress results")
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, help="destination for the reviewed revision; defaults to --folder")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--corpus-kind", choices=("taxonomy_proxy", "agent_authored_natural"), default="taxonomy_proxy")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    cases_path = args.folder / "cases.jsonl"
    cases = _read_jsonl(cases_path)
    parts: list[dict[str, Any]] = []
    for index in range(args.shards):
        path = args.folder / f"part-{index}.jsonl"
        shard = _read_jsonl(path)
        for row in shard:
            if int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % args.shards != index:
                raise ValueError(f"case {row['id']} in wrong shard {index}")
        parts.extend(shard)
    summary = summarize(cases, parts, shards=args.shards)
    destination = args.output_folder or args.folder
    if args.report.exists():
        raise FileExistsError(f"report already exists: {args.report}")
    for name in ("cases.jsonl", "all_results.jsonl", "failures.jsonl", "per_category.csv", "summary.json"):
        if (destination / name).exists() and (destination != args.folder or name != "cases.jsonl"):
            raise FileExistsError(f"summary artifact already exists: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    if destination != args.folder:
        shutil.copyfile(cases_path, destination / "cases.jsonl")
    summary["cases_sha256"] = _digest(cases_path)
    summary["config_sha256"] = _digest(CONFIG)
    summary["taxonomy_sha256"] = _digest(TAXONOMY)
    summary["aliases_sha256"] = _digest(ALIASES)
    summary["corpus_kind"] = args.corpus_kind
    _write_results(
        destination, args.report, summary, Taxonomy.load(TAXONOMY), corpus_kind=args.corpus_kind,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
