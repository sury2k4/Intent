from pathlib import Path

from scripts.prepare_poi_v3_evaluation import make_cases

ROOT = Path(__file__).parents[1]


def test_provisional_evaluation_preserves_authored_cases_and_marks_proxy_labels():
    rows = make_cases(
        ROOT / "data/taxonomies/place_types.poc.v2.json",
        ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v2.json",
        ROOT / "data/eval/place_type_queries.poc.v1.jsonl",
        ROOT / "data/eval/hcmc_poi_queries.v3.jsonl",
    )
    assert len(rows) >= 100
    assert len({row["id"] for row in rows}) == len(rows)
    assert any(row["track"] == "authored_poi" for row in rows)
    assert any(row["track"] == "synthetic_ood" for row in rows)
    assert all(row["expected_action"] == "reject" for row in rows if row.get("is_ood") and row["track"] == "synthetic_ood")
    assert all(row["split"] in {"test", "calibration"} for row in rows)
