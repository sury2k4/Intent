from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_new(path: str, payload: dict) -> None:
    target = ROOT / path
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different versioned resource: {target}")
        return
    target.write_text(content, encoding="utf-8")


def main() -> None:
    taxonomy = load("data/taxonomies/place_types.poc.v1.json")
    taxonomy["taxonomy_version"] = "place-types-poc-v2"
    taxonomy["items"].extend([
        {"id": "242", "parent_id": "80", "code": "ev_charging", "names": {"en": "Electric vehicle charging station", "vi": "Trạm sạc xe điện"}},
        {"id": "243", "parent_id": "8", "code": "bank", "names": {"en": "Bank branch", "vi": "Ngân hàng"}},
        {"id": "244", "parent_id": "2", "code": "school", "names": {"en": "School", "vi": "Trường học"}},
        {"id": "245", "parent_id": "80", "code": "motorcycle_repair", "names": {"en": "Motorcycle repair", "vi": "Sửa xe máy"}},
    ])
    write_new("data/taxonomies/place_types.poc.v2.json", taxonomy)

    enrichment = load("data/taxonomies/place_type_enrichment.rich.poc.v1.json")
    enrichment["source"] = "rich_poc_v1_plus_reviewed_generic_pois_v2"
    enrichment["items"].update({
        "242": {
            "description_vi": "Điểm nạp năng lượng cho xe điện của nhiều nhà vận hành, không giới hạn một thương hiệu. Phân biệt với trạm chỉ dành cho VinFast hoặc tủ đổi pin.",
            "description_en": "A public charging point for electric vehicles regardless of operator or brand, distinct from a branded-only charging station or battery-swap cabinet.",
            "aliases_vi": ["trạm sạc điện", "trạm sạc xe điện", "điểm sạc xe điện", "trạm sạc nhanh cho xe điện"],
            "aliases_en": ["EV charging station", "electric vehicle charger", "public charging point"],
        },
        "243": {
            "description_vi": "Chi nhánh hoặc phòng giao dịch ngân hàng có nhân viên cung cấp dịch vụ tài khoản và tín dụng; khác với máy ATM tự phục vụ.",
            "description_en": "A staffed bank branch offering account and financial services, separate from a self-service ATM.",
            "aliases_vi": ["chi nhánh ngân hàng", "phòng giao dịch ngân hàng", "ngân hàng"],
            "aliases_en": ["bank branch", "banking office", "bank"],
        },
        "244": {
            "description_vi": "Cơ sở giảng dạy phổ thông khi chưa rõ cấp học; không tự gán thành trường tiểu học, THCS hoặc THPT nếu dữ liệu không ghi rõ.",
            "description_en": "A general school whose grade level is unspecified; do not infer primary, secondary or high school without explicit evidence.",
            "aliases_vi": ["trường học", "trường phổ thông", "ngôi trường"],
            "aliases_en": ["school", "general school"],
        },
        "245": {
            "description_vi": "Cửa hàng hoặc xưởng chuyên bảo trì, sửa chữa xe máy; không mặc định là sửa ô tô hay cửa hàng bán lốp xe.",
            "description_en": "A workshop servicing and repairing motorcycles, distinct from car repair and tire retail shops.",
            "aliases_vi": ["tiệm sửa xe máy", "xưởng sửa xe máy", "sửa xe máy"],
            "aliases_en": ["motorcycle repair", "motorbike service", "motorcycle workshop"],
        },
    })
    write_new("data/taxonomies/place_type_enrichment.rich.poc.v2.json", enrichment)

    mapping = load("data/osm/mappings/place_types.poc.v1.json")
    mapping["mapping_version"] = "hcmc-osm-place-types-poc-v2"
    mapping["taxonomy_version"] = taxonomy["taxonomy_version"]
    mapping["rules"].extend([
        {"id": "generic-ev-charging", "priority": 997, "category_id": "242", "match": {"tag": "amenity", "op": "eq", "value": "charging_station"}, "allow_unnamed": True},
        {"id": "amenity-bank", "priority": 649, "category_id": "243", "match": {"tag": "amenity", "op": "eq", "value": "bank"}},
        {"id": "amenity-school-unspecified", "priority": 643, "category_id": "244", "match": {"tag": "amenity", "op": "eq", "value": "school"}},
        {"id": "shop-car-repair", "priority": 585, "category_id": "83", "match": {"tag": "shop", "op": "eq", "value": "car_repair"}},
        {"id": "shop-motorcycle-repair", "priority": 584, "category_id": "245", "match": {"tag": "shop", "op": "eq", "value": "motorcycle_repair"}},
    ])
    write_new("data/osm/mappings/place_types.poc.v2.json", mapping)

    properties = load("data/properties/poi_properties.poc.v2.json")
    properties["schema_version"] = 3
    properties["property_taxonomy_version"] = "poi-properties-poc-v3"
    properties["category_profiles"].extend([
        {"category_id": "242", "replace": True, "property_ids": ["fast_charging", "parking", "covered_parking", "wheelchair", "toilets", "waiting_area", "lighting", "open_24h", "accepts_cards"]},
        {"category_id": "243", "replace": True, "property_ids": ["wheelchair", "waiting_area", "cash_withdrawal", "cash_deposit", "accepts_cards", "parking", "air_conditioning", "toilets", "open_late"]},
        {"category_id": "245", "replace": True, "property_ids": ["parking", "accepts_cards", "covered_area", "toilets", "air_compressor", "waiting_area", "open_late", "lighting"]},
    ])
    write_new("data/properties/poi_properties.poc.v3.json", properties)


if __name__ == "__main__":
    main()
