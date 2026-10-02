from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_intent.osm_census import census_hcmc_candidates
from model_intent.osm_ingest import Boundary, file_sha256, iter_pbf_elements, validate_catalogs
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Count HCMC tagged OSM candidates before category filtering")
    parser.add_argument("--pbf", type=Path, default=ROOT / "data/osm/raw/vietnam-latest.osm.pbf")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "data/taxonomies/place_types.poc.v2.json")
    parser.add_argument("--mapping", type=Path, default=ROOT / "data/osm/mappings/place_types.poc.v2.json")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/properties/poi_properties.poc.v4.json")
    parser.add_argument("--boundary", type=Path, default=ROOT / "data/osm/boundaries/hcmc.admin.v1.geojson")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements-jsonl", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing census: {args.output}")
    taxonomy = Taxonomy.load(args.taxonomy)
    catalogs = validate_catalogs(taxonomy, args.mapping, args.properties)
    if args.elements_jsonl:
        elements = (json.loads(line) for line in args.elements_jsonl.open(encoding="utf-8") if line.strip())
    else:
        source = json.loads((ROOT / "data/osm/sources/hcmc.poc.v1.json").read_text(encoding="utf-8"))
        actual = file_sha256(args.pbf)
        if actual != source["expected_sha256"]:
            raise ValueError("source PBF checksum differs from the pinned Vietnam snapshot")
        elements = iter_pbf_elements(args.pbf)
    report = census_hcmc_candidates(
        elements, boundary=Boundary.from_geojson(args.boundary), taxonomy=taxonomy,
        mapping_catalog=catalogs["mapping"], property_profiles=catalogs["profiles"],
    )
    report["taxonomy_fingerprint"] = taxonomy.fingerprint()
    report["mapping_sha256"] = file_sha256(args.mapping)
    report["boundary_sha256"] = file_sha256(args.boundary)
    if not args.elements_jsonl:
        report["pbf_sha256"] = actual
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": report["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
