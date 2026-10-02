from __future__ import annotations

"""Build immutable, pinned HCMC OSM-only v3 resources and optionally an artifact.

No network, SQL, or changes to pre-existing resource/artifact directories.
Use --survey to inspect the candidate signatures before committing new mappings.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from model_intent.osm_census import census_hcmc_candidates
from model_intent.osm_ingest import (
    Boundary, POI_PRIMARY_KEYS, POI_QUALIFIED_VALUES, build_artifact, file_sha256,
    iter_pbf_elements, load_mapping_catalog, stable_hash, validate_catalogs,
)
from model_intent.poi_properties import load_property_definitions, load_property_profiles
from model_intent.normalization import fold_vietnamese, normalize_text
from model_intent.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/osm/sources/hcmc.poc.v1.json"
PBF = ROOT / "data/osm/raw/vietnam-latest.osm.pbf"
BOUNDARY = ROOT / "data/osm/boundaries/hcmc.admin.v1.geojson"
TAXONOMY = ROOT / "data/taxonomies/place_types.poc.v3.json"
ENRICHMENT = ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v3.json"
MAPPING = ROOT / "data/osm/mappings/place_types.poc.v3.json"
PROPERTIES = ROOT / "data/properties/poi_properties.poc.v5.json"
REGISTRY = ROOT / "data/osm/mappings/osm_signature_registry.poc.v3.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/osm/hcmc-osm-v3"

# Parent / bilingual names / explicit semantically bounded meaning. Numeric IDs
# are append-only after the v2 maximum (245), never derived from PBF frequency.
NEW_CATEGORIES: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("246", "9", "drinking_water_point", "Drinking water point", "Điểm nước uống công cộng", "A place explicitly tagged as potable water; the tag does not establish free entry or operating hours."),
    ("247", "9", "public_bench", "Public bench", "Ghế ngồi công cộng", "A public bench or seating point, not an indoor seating promise for another POI."),
    ("248", "6", "vending_machine", "Vending machine", "Máy bán hàng tự động", "A self-service vending machine; goods offered are unspecified without a vending tag."),
    ("249", "79", "taxi_stand", "Taxi stand", "Điểm đón taxi", "A designated taxi boarding or waiting location, not a taxi operator office."),
    ("250", "79", "bicycle_parking_facility", "Bicycle parking facility", "Bãi đỗ xe đạp", "A bicycle parking location; access, capacity and security remain unknown without tags."),
    ("251", "115", "sports_pitch", "Sports pitch", "Sân thể thao", "A marked outdoor sports pitch; its sport is unspecified without a sport tag."),
    ("252", "115", "outdoor_fitness_station", "Outdoor fitness station", "Điểm tập thể dục ngoài trời", "Public outdoor exercise equipment, not an indoor gym or staffed training centre."),
    ("253", "9", "viewpoint", "Viewpoint", "Điểm ngắm cảnh", "A designated viewpoint, with no claim about accessibility, views or opening hours."),
    ("254", "131", "visitor_information", "Visitor information point", "Điểm thông tin du khách", "A tourist information point; it may be a staffed office or an information board, and staffing must not be inferred."),
    ("255", "131", "public_artwork", "Public artwork", "Tác phẩm nghệ thuật công cộng", "A displayed public artwork, not necessarily a monument or museum."),
    ("256", "9", "picnic_site", "Picnic site", "Khu dã ngoại", "An area designated for picnics; no seating, facilities or permission to barbecue is implied."),
    ("257", "79", "metro_entrance", "Metro entrance", "Lối vào ga tàu điện", "An entrance to an urban rail station, not the entire station or its services."),
    ("258", "79", "dock", "Dock", "Bến cập tàu", "A landing or docking location; no passenger ferry schedule is implied."),
    ("259", "9", "beach", "Beach", "Bãi biển", "A beach landform, not a claim that swimming is safe or permitted."),
    ("260", "8", "accountant", "Accounting office", "Văn phòng kế toán", "An accounting-service office, separate from a bank branch or law office."),
    ("261", "41", "unspecified_clinic", "Clinic (unspecified specialty)", "Phòng khám chưa rõ chuyên khoa", "A medical clinic without evidence that it is general, dental or a specific specialty."),
    ("262", "8", "shoe_repair", "Shoe repair", "Tiệm sửa giày", "A workshop repairing footwear, distinct from a shoe retailer."),
    ("263", "110", "homestay", "Homestay", "Lưu trú homestay", "An overnight homestay, distinct from a hotel; amenities are not inferred."),
    ("264", "9", "dog_park", "Dog park", "Công viên cho chó", "A designated dog-exercise area, not a pet shop or veterinary practice."),
    ("265", "80", "motorcycle_parking_facility", "Motorcycle parking", "Bãi đỗ xe máy", "A parking location specifically for motorcycles; security and cost are unknown."),
    ("266", "9", "public_fountain", "Fountain", "Đài phun nước", "An ornamental fountain; its water must not be assumed safe to drink."),
    ("267", "9", "public_play_area", "Indoor play centre", "Khu vui chơi trong nhà", "A dedicated indoor play centre; supervision and fee are unknown."),
    ("268", "79", "ferry_boarding_point", "Ferry boarding point", "Điểm lên phà", "A passenger-ferry boarding point, not necessarily a standalone terminal."),
    ("269", "9", "public_shelter", "Public shelter", "Nhà chờ công cộng", "A standalone shelter from weather; it is not necessarily a bus stop or overnight accommodation."),
    ("270", "79", "transit_stop_position", "Transit stop position", "Vị trí dừng phương tiện công cộng", "An OSM transit vehicle stopping position; the transport mode and passenger facilities may be unspecified."),
    ("271", "8", "commercial_building", "Commercial building", "Tòa nhà thương mại", "A building marked commercial without evidence that it is a specific shop, company office or shopping mall."),
    ("272", "6", "retail_building", "Retail building", "Tòa nhà bán lẻ", "A building tagged for retail use, not evidence of a named retailer or a complete shopping mall."),
    ("273", "4", "food_court", "Food court", "Khu ẩm thực", "A shared dining area with multiple vendors; individual restaurants are not inferred."),
    ("274", "110", "short_stay_apartment", "Short-stay apartment", "Căn hộ lưu trú ngắn hạn", "A tourism apartment offered for short stays; not necessarily a residential apartment building."),
    ("275", "139", "worship_unspecified", "Place of worship (unspecified)", "Nơi thờ tự chưa rõ loại", "A worship facility without enough tags to determine religion or building type."),
    ("276", "9", "public_garden", "Garden", "Khu vườn", "A landscaped garden, distinct from an explicitly tagged public park; public access is not implied."),
    ("277", "6", "variety_store", "Variety store", "Cửa hàng hàng tổng hợp", "A general-merchandise variety retailer; not necessarily a food grocery or supermarket."),
    ("278", "6", "beverage_retail", "Beverage retailer", "Cửa hàng đồ uống", "A shop selling packaged beverages; not necessarily a café serving prepared drinks."),
    ("279", "6", "stationery_store", "Stationery store", "Cửa hàng văn phòng phẩm", "A retail shop for paper and stationery, distinct from a print service or bookshop."),
    ("280", "79", "transit_platform", "Transit platform", "Sân ga hoặc điểm chờ phương tiện", "A passenger platform with unspecified transport mode, not necessarily a bus or rail station."),
    ("281", "79", "helipad", "Helipad", "Bãi đáp trực thăng", "An aircraft landing pad for helicopters; it is not a passenger airport."),
    ("282", "9", "recreation_ground", "Recreation ground", "Khu vui chơi ngoài trời", "An open area intended for recreation without evidence that it is a managed public park."),
    ("283", "79", "transit_station_unspecified", "Transit station (mode unspecified)", "Ga giao thông công cộng chưa rõ loại", "A public transport station whose mode is not established; do not assume it is a bus or rail station."),
)

# Provisional descriptions state precise OSM meaning in each language, without
# inventing services, verified quality or an actual business name.
VI_MEANINGS: dict[str, str] = {
    "246": "Vị trí được gắn thẻ nước uống; chưa xác minh chất lượng nước hoặc miễn phí.",
    "247": "Ghế công cộng độc lập; không suy ra chỗ ngồi trong một cửa hàng gần đó.",
    "248": "Máy bán hàng tự phục vụ; chưa rõ loại hàng nếu không có thẻ vending.",
    "249": "Điểm đón hoặc chờ taxi được chỉ định, không phải văn phòng hãng taxi.",
    "250": "Nơi đỗ xe đạp; khả năng tiếp cận, an ninh và sức chứa chưa rõ.",
    "251": "Sân thể thao có đánh dấu; môn thể thao chỉ xác định khi OSM ghi rõ.",
    "252": "Thiết bị tập luyện ngoài trời; không mặc định là phòng gym có nhân viên.",
    "253": "Vị trí được đánh dấu để ngắm cảnh; không đảm bảo khả năng tiếp cận.",
    "254": "Điểm cung cấp thông tin du khách; có thể là bảng tự phục vụ hoặc văn phòng, chưa rõ nhân viên.",
    "255": "Tác phẩm nghệ thuật được trưng bày; không mặc định là di tích hoặc bảo tàng.",
    "256": "Nơi được chỉ định dã ngoại; không suy ra có bàn ghế hoặc cho phép nướng BBQ.",
    "257": "Cửa vào ga tàu điện đô thị, không phải toàn bộ nhà ga hoặc dịch vụ nhà ga.",
    "258": "Nơi tàu cập bến; không khẳng định có dịch vụ phà chở khách.",
    "259": "Địa hình bãi biển; không khẳng định an toàn hoặc được phép tắm biển.",
    "260": "Văn phòng cung cấp dịch vụ kế toán; không phải chi nhánh ngân hàng.",
    "261": "Phòng khám chưa rõ chuyên khoa; không suy ra là đa khoa hoặc nha khoa.",
    "262": "Nơi sửa chữa giày dép; khác cửa hàng bán giày.",
    "263": "Lưu trú homestay qua đêm, khác khách sạn; chưa xác minh tiện nghi.",
    "264": "Khu được chỉ định cho chó vận động; không phải cửa hàng thú cưng.",
    "265": "Điểm đỗ xe máy; phí và mức độ an ninh chưa được xác minh.",
    "266": "Đài phun nước trang trí; không suy ra nước uống được.",
    "267": "Khu vui chơi trong nhà; chưa rõ mức phí hoặc giám sát trẻ em.",
    "268": "Điểm lên phà của hành khách; không mặc định là nhà ga phà riêng.",
    "269": "Nhà che mưa nắng độc lập; không tự suy ra là trạm xe buýt hoặc nơi ngủ qua đêm.",
    "270": "Vị trí phương tiện công cộng dừng đỗ; phương thức vận tải chưa chắc được ghi rõ.",
    "271": "Tòa nhà dùng cho thương mại; chưa xác định cửa hàng, văn phòng hay trung tâm mua sắm.",
    "272": "Tòa nhà có mục đích bán lẻ; không đồng nghĩa một cửa hàng có tên hoặc trung tâm thương mại.",
    "273": "Khu ăn uống chung nhiều quầy; không suy ra từng nhà hàng cụ thể.",
    "274": "Căn hộ lưu trú du lịch ngắn hạn, không mặc định là chung cư cư trú lâu dài.",
    "275": "Cơ sở thờ cúng chưa rõ tôn giáo hoặc kiểu công trình; không tự suy ra chùa hay nhà thờ.",
    "276": "Khu vườn cây cảnh; không mặc định là công viên mở cửa công cộng.",
    "277": "Cửa hàng bán nhiều nhóm hàng tổng hợp; không nhất thiết bán thực phẩm như tạp hóa.",
    "278": "Cửa hàng bán đồ uống đóng chai; không mặc định là quán pha chế hoặc cà phê.",
    "279": "Cửa hàng bán giấy và đồ văn phòng phẩm, khác dịch vụ in ấn hoặc hiệu sách.",
    "280": "Nền chờ hành khách chưa rõ phương tiện; không tự coi là ga tàu hoặc bến xe.",
    "281": "Bãi đáp trực thăng, không phải sân bay thương mại phục vụ hành khách.",
    "282": "Không gian ngoài trời dành cho giải trí, không khẳng định là công viên được quản lý.",
    "283": "Nhà ga giao thông công cộng chưa rõ loại phương tiện; không tự nhận là bến xe hay ga tàu.",
}

# Reviewed OSM key/value signatures. Existing leaves are preferred whenever
# their meaning fits; a new leaf is used only when no suitable v2 leaf exists.
# More specific conditions have higher priority; v2 rules are reused verbatim.
EXTRA_RULES: tuple[tuple[str, str, str], ...] = (
    ("shop", "hairdresser", "40"), ("shop", "beauty", "39"), ("shop", "optician", "219"),
    ("shop", "furniture", "175"), ("shop", "grocery", "176"), ("shop", "greengrocer", "205"),
    ("shop", "seafood", "204"), ("shop", "fruit", "207"), ("shop", "beer", "182"),
    ("shop", "car_wash", "84"), ("shop", "motorcycle", "35"), ("shop", "vending_machine", "248"),
    ("shop", "general", "277"),
    ("shop", "wine", "127"), ("shop", "massage", "170"),
    ("shop", "travel_agency", "107"), ("shop", "laundry", "100"),
    ("shop", "copyshop", "177"), ("shop", "stationery", "279"),
    ("shop", "variety_store", "277"), ("shop", "beverages", "278"),
    ("shop", "lottery", "206"), ("shop", "tailor", "105"),
    ("shop", "pastry", "21"),
    ("shop", "bed", "175"), ("shop", "interior_decoration", "125"),
    ("shop", "dry_cleaning", "100"),
    ("amenity", "drinking_water", "246"), ("amenity", "bench", "247"),
    ("amenity", "shelter", "269"), ("amenity", "food_court", "273"),
    ("amenity", "place_of_worship", "275"),
    ("amenity", "vending_machine", "248"), ("amenity", "taxi", "249"),
    ("amenity", "bicycle_parking", "250"), ("amenity", "motorcycle_parking", "265"),
    ("amenity", "fountain", "266"), ("amenity", "ferry_terminal", "113"),
    ("amenity", "car_wash", "84"), ("amenity", "social_facility", "191"),
    ("amenity", "grave_yard", "184"), ("amenity", "car_rental", "82"),
    ("amenity", "karaoke_box", "188"),
    ("amenity", "theatre", "134"), ("amenity", "arts_centre", "163"),
    ("amenity", "driving_school", "153"), ("amenity", "language_school", "152"),
    ("amenity", "college", "48"), ("amenity", "biergarten", "26"),
    ("amenity", "events_venue", "90"),
    ("amenity", "conference_centre", "215"),
    ("amenity", "nightclub", "94"),
    ("tourism", "guest_house", "212"),
    ("tourism", "viewpoint", "253"), ("tourism", "information", "254"),
    ("tourism", "apartment", "274"),
    ("tourism", "artwork", "255"), ("tourism", "picnic_site", "256"),
    ("tourism", "zoo", "97"), ("tourism", "camp_site", "239"),
    ("tourism", "theme_park", "89"), ("tourism", "chalet", "212"),
    ("tourism", "homestay", "263"),
    ("leisure", "pitch", "251"), ("leisure", "fitness_station", "252"),
    ("leisure", "garden", "276"), ("leisure", "resort", "135"),
    ("leisure", "sports_hall", "202"),
    ("leisure", "dog_park", "264"),
    ("leisure", "amusement_arcade", "88"),
    ("leisure", "water_park", "199"), ("leisure", "miniature_golf", "117"),
    ("leisure", "indoor_play", "267"),
    ("office", "lawyer", "101"), ("office", "travel_agent", "107"),
    ("office", "estate_agent", "104"), ("office", "accountant", "260"),
    ("office", "government", "173"), ("office", "company", "192"),
    ("office", "consulting", "98"),
    ("healthcare", "clinic", "261"), ("healthcare", "doctor", "77"),
    ("healthcare", "dentist", "76"), ("healthcare", "pharmacy", "34"),
    ("healthcare", "physiotherapist", "149"),
    ("craft", "tailor", "105"), ("craft", "photographer", "183"),
    ("craft", "shoemaker", "262"), ("craft", "printer", "177"),
    ("historic", "memorial", "194"), ("historic", "monument", "194"),
    ("historic", "wayside_shrine", "159"), ("historic", "archaeological_site", "91"),
    ("historic", "ruins", "91"), ("historic", "castle", "91"),
    ("public_transport", "stop_position", "270"),
    ("public_transport", "platform", "280"),
    ("highway", "rest_area", "241"), ("highway", "services", "87"),
    ("railway", "subway_entrance", "257"), ("railway", "tram_stop", "114"),
    ("railway", "halt", "111"),
    ("public_transport", "station", "283"),
    ("emergency", "fire_station", "71"), ("emergency", "police", "68"),
    ("landuse", "cemetery", "184"), ("landuse", "recreation_ground", "282"),
    ("waterway", "dock", "258"),
    ("natural", "beach", "259"), ("man_made", "pier", "258"),
    ("building", "commercial", "271"), ("building", "retail", "272"),
    ("building", "train_station", "111"), ("building", "hospital", "42"),
    ("building", "school", "244"), ("aeroway", "helipad", "281"),
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace immutable v3 resource: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def verify_pinned_source() -> dict[str, Any]:
    source = read(SOURCE)
    if PBF.stat().st_size != source["expected_bytes"] or file_sha256(PBF) != source["expected_sha256"]:
        raise ValueError("Vietnam PBF is not the pinned snapshot; cannot generate v3 resources")
    if file_sha256(BOUNDARY) != source["boundary_sha256"]:
        raise ValueError("HCMC boundary differs from the pinned source manifest")
    return source


def resources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    taxonomy = read(ROOT / "data/taxonomies/place_types.poc.v2.json")
    v2_leaves = {item["id"] for item in taxonomy["items"] if item["id"] not in {
        old["parent_id"] for old in taxonomy["items"]
    }}
    if any(parent_id in v2_leaves for _id, parent_id, *_ in NEW_CATEGORIES):
        raise ValueError("new v3 categories cannot turn a v2 leaf into a parent")
    taxonomy["taxonomy_version"] = "place-types-poc-v3"
    taxonomy["status"] = "partial_poc_osm_expansion_provisional"
    enrichment = read(ROOT / "data/taxonomies/place_type_enrichment.rich.poc.v2.json")
    enrichment["source"] = "reviewed_v2_plus_explicit_provisional_osm_v3"
    for category_id, parent_id, code, en, vi, meaning in NEW_CATEGORIES:
        taxonomy["items"].append({
            "id": category_id, "parent_id": parent_id, "code": code, "names": {"en": en, "vi": vi},
        })
        enrichment["items"][category_id] = {
            "description_en": f"Provisional OSM-derived category. {meaning} Classification reflects explicit OSM tags only; it does not verify amenities or quality.",
            "description_vi": f"Loại tạm thời theo thẻ OSM. {VI_MEANINGS[category_id]} Chỉ phân loại theo thẻ được ghi rõ; chưa xác minh chất lượng hoặc giờ hoạt động.",
            "aliases_en": list({normalize_text(alias): alias for alias in reversed((en.lower(), code.replace("_", " ")))}.values()),
            "aliases_vi": [vi.lower()],
        }
    mapping = read(ROOT / "data/osm/mappings/place_types.poc.v2.json")
    mapping["mapping_version"] = "hcmc-osm-place-types-poc-v3"
    mapping["taxonomy_version"] = taxonomy["taxonomy_version"]
    # Previously unconstrained office=present and religion=other could swallow
    # novel, semantically unrelated signatures. Replace with exact rules only.
    mapping["rules"] = [
        rule for rule in mapping["rules"]
        if rule["id"] not in {"office-company", "amenity-other-worship"}
    ]
    existing_ids = {rule["id"] for rule in mapping["rules"]}
    existing_priorities = {rule["priority"] for rule in mapping["rules"]}
    # Reused v2 specific rules outrank v3 new fallback signatures; among new
    # signatures deterministic key/value priority prevents arbitrary ties.
    priority = 567
    for key, value, category_id in EXTRA_RULES:
        rule_id = f"v3-{key}-{value.replace('_', '-') }"
        if rule_id in existing_ids:
            raise ValueError(f"duplicate new mapping ID: {rule_id}")
        while priority in existing_priorities:
            priority -= 1
        match: dict[str, Any] = {"tag": key, "op": "eq", "value": value}
        if key == "public_transport":
            # A station/platform/stop_position alone never implies a bus. More
            # precise railway, ferry, pier or building tags must win instead.
            match = {"all": [match, {"none": [
                {"tag": specific, "op": "present"}
                for specific in ("amenity", "railway", "man_made", "building", "highway")
            ]}]}
        mapping["rules"].append({
            "id": rule_id, "priority": priority, "category_id": category_id,
            "match": match,
        })
        existing_ids.add(rule_id)
        existing_priorities.add(priority)
        priority -= 1
    leaves = {item["id"] for item in taxonomy["items"] if item["id"] not in {i["parent_id"] for i in taxonomy["items"]}}
    mapped = {rule["category_id"] for rule in mapping["rules"]}
    mapping["intentionally_unmapped"] = {
        category_id: "No semantically justified explicit OSM signature in reviewed mapping; do not infer this leaf from another category."
        for category_id in sorted(leaves - mapped, key=int)
    }
    properties = read(ROOT / "data/properties/poi_properties.poc.v4.json")
    properties.pop("base_catalog", None)
    properties.pop("property_overrides", None)
    properties["properties"] = list(load_property_definitions(
        ROOT / "data/properties/poi_properties.poc.v4.json",
    ).values())
    properties["schema_version"] = 5
    properties["property_taxonomy_version"] = "poi-properties-poc-v5-osm-only"
    properties["minimum_known_properties"] = 0
    properties["evidence_policy"] = "osm_only; missing is unknown; no synthetic properties, ratings, or reviews"
    return taxonomy, enrichment, mapping, properties


# Hand-reviewed bilingual labels for high-volume observed types that have no
# semantically equivalent v2 leaf. Rarer labels retain the exact OSM key/value
# in both languages and an explicit provisional caveat instead of guessing.
OBSERVED_LABELS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("shop", "tyres"): ("Tyre shop", "Cửa hàng lốp xe", "6"),
    ("shop", "car_parts"): ("Car parts store", "Cửa hàng phụ tùng ô tô", "80"),
    ("shop", "coffee"): ("Coffee retail shop", "Cửa hàng cà phê đóng gói", "6"),
    ("shop", "baby_goods"): ("Baby goods store", "Cửa hàng đồ dùng trẻ em", "6"),
    ("shop", "toys"): ("Toy store", "Cửa hàng đồ chơi", "6"),
    ("shop", "ticket"): ("Ticket shop", "Cửa hàng bán vé", "8"),
    ("shop", "confectionery"): ("Confectionery shop", "Cửa hàng bánh kẹo", "6"),
    ("shop", "medical_supply"): ("Medical supply shop", "Cửa hàng vật tư y tế", "6"),
    ("shop", "photo"): ("Photography supply shop", "Cửa hàng thiết bị ảnh", "6"),
    ("shop", "tattoo"): ("Tattoo studio", "Tiệm xăm", "8"),
    ("shop", "outdoor"): ("Outdoor goods shop", "Cửa hàng đồ dã ngoại", "6"),
    ("amenity", "waste_basket"): ("Public waste bin", "Thùng rác công cộng", "9"),
    ("amenity", "parking_space"): ("Parking space", "Chỗ đậu xe đơn lẻ", "80"),
    ("amenity", "parking_entrance"): ("Parking entrance", "Lối vào bãi đậu xe", "80"),
    ("amenity", "bicycle_rental"): ("Bicycle rental point", "Điểm thuê xe đạp", "79"),
    ("amenity", "research_institute"): ("Research institute", "Viện nghiên cứu", "2"),
    ("amenity", "casino"): ("Casino", "Sòng bạc", "9"),
    ("amenity", "childcare"): ("Childcare centre", "Trung tâm chăm sóc trẻ", "2"),
    ("amenity", "post_box"): ("Post box", "Hòm thư công cộng", "79"),
    ("amenity", "parcel_locker"): ("Parcel locker", "Tủ nhận bưu kiện", "79"),
    ("leisure", "track"): ("Sports track", "Đường chạy thể thao", "115"),
    ("leisure", "marina"): ("Marina", "Bến du thuyền", "79"),
    ("man_made", "tower"): ("Tower", "Tháp", "9"),
    ("man_made", "water_tower"): ("Water tower", "Tháp nước", "9"),
    ("natural", "peak"): ("Mountain summit", "Đỉnh núi", "9"),
    ("aeroway", "terminal"): ("Airport terminal", "Nhà ga sân bay", "79"),
}
PARENT_FOR_KEY: dict[str, str] = {
    "shop": "6", "amenity": "8", "tourism": "131", "leisure": "9",
    "office": "185", "healthcare": "41", "craft": "8", "historic": "131",
    "club": "9", "government": "67", "emergency": "67", "public_transport": "79",
    "railway": "79", "highway": "79", "aeroway": "79", "building": "8",
    "landuse": "9", "man_made": "9", "natural": "9", "waterway": "79",
}
MAX_OBSERVED_LEAVES = 120
MIN_OBSERVED_COUNT = 2


def add_observed_types(
    report: dict[str, Any], taxonomy: dict[str, Any],
    enrichment: dict[str, Any], mapping: dict[str, Any],
) -> dict[str, Any]:
    """Bounded explicit per-tag categories for clear types missed by reviewed rules.

    Deterministic on the pinned census: count-ranked selection, sorted stable IDs
    for this version. Unselected/ambiguous signatures remain visible in registry.
    """
    observed: Counter[tuple[str, str]] = Counter()
    for row in report["signatures"]:
        unresolved = row["dispositions"].get("unmapped_needs_review", 0)
        if not unresolved:
            continue
        parts = [part.split("=", 1) for part in row["signature"].split("|")]
        primary = [(key, value) for key, value in parts if key in POI_PRIMARY_KEYS or key in POI_QUALIFIED_VALUES]
        if len(primary) != 1:
            continue  # Competing primary tags require a human precedence decision.
        key, value = primary[0]
        if key not in PARENT_FOR_KEY or key == "sport" or value.casefold() in {
            "no", "yes", "other", "unknown", "unspecified", "fixme", "?",
        }:
            continue
        observed[(key, value)] += unresolved
    eligible = [pair for pair, count in observed.items() if count >= MIN_OBSERVED_COUNT]
    selected = set(sorted(eligible, key=lambda pair: (-observed[pair], pair))[:MAX_OBSERVED_LEAVES])
    used_codes = {row["code"] for row in taxonomy["items"]}
    rules = mapping["rules"]
    next_priority = min(int(rule["priority"]) for rule in rules) - 1
    start_id = max(int(item["id"]) for item in taxonomy["items"]) + 1
    existing_parents = {item["parent_id"] for item in taxonomy["items"]}
    old_leaf_ids = {item["id"] for item in taxonomy["items"] if item["id"] not in existing_parents}
    for offset, (key, value) in enumerate(sorted(selected)):
        id_value = str(start_id + offset)
        en, vi, parent_id = OBSERVED_LABELS.get(
            (key, value),
            (f"OSM {key}={value} location", f"Địa điểm OSM {key}={value}", PARENT_FOR_KEY[key]),
        )
        if parent_id in old_leaf_ids or parent_id not in {item["id"] for item in taxonomy["items"]}:
            raise ValueError(f"v3 observed leaf cannot turn existing leaf into a parent: {parent_id}")
        slug = re.sub(r"[^a-z0-9]+", "_", fold_vietnamese(value).lower()).strip("_")[:36] or "value"
        code = f"osm_{key}_{slug}_{stable_hash([key, value])[:8]}"
        if code in used_codes:
            raise ValueError(f"observed OSM category code collision: {code}")
        used_codes.add(code)
        taxonomy["items"].append({
            "id": id_value, "parent_id": parent_id, "code": code,
            "names": {"en": en, "vi": vi},
        })
        signature = f"{key}={value}"
        enrichment["items"][id_value] = {
            "description_en": (
                f"Provisional category for explicitly recorded OSM {signature}. "
                f"This is a location tagged as {en.lower()}; the OSM tag does not verify a business name, "
                "public access, current operation, opening hours or facilities. "
                "Review local meaning before treating the tag as a stronger claim."
            ),
            "description_vi": (
                f"Loại tạm thời theo đúng thẻ OSM {signature}. Đây là điểm được gắn thẻ {vi.lower()}; "
                "thẻ không xác minh tên cơ sở, quyền vào, hoạt động hiện tại, giờ mở hoặc tiện ích. "
                "Cần duyệt nghĩa địa phương trước khi diễn giải chắc chắn hơn."
            ),
            "aliases_en": [en.lower()], "aliases_vi": [vi.lower()],
        }
        rules.append({
            "id": f"v3-observed-{key}-{slug}-{stable_hash([key, value])[:8]}",
            "priority": next_priority - offset, "category_id": id_value,
            "match": {"tag": key, "op": "eq", "value": value},
        })
    leaves = {item["id"] for item in taxonomy["items"] if item["id"] not in {i["parent_id"] for i in taxonomy["items"]}}
    mapped = {rule["category_id"] for rule in rules}
    mapping["intentionally_unmapped"] = {
        category_id: "No semantically justified explicit OSM signature; do not infer this leaf."
        for category_id in sorted(leaves - mapped, key=int)
    }
    return {
        "added_leaf_count": len(selected),
        "selected_candidate_count": sum(observed[pair] for pair in selected),
        "deferred_clear_type_count": len(set(observed) - selected),
        "deferred_clear_candidate_count": sum(observed[pair] for pair in set(observed) - selected),
        "threshold": MIN_OBSERVED_COUNT, "maximum_added_leaves": MAX_OBSERVED_LEAVES,
    }


def census(
    taxonomy_path: Path, properties_path: Path, *, survey: bool = False,
    mapping_override: dict[str, Any] | None = None,
    taxonomy_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    taxonomy = Taxonomy.from_payload(taxonomy_override) if taxonomy_override else Taxonomy.load(taxonomy_path)
    mapping_catalog = (
        load_mapping_catalog(ROOT / "data/osm/mappings/place_types.poc.v2.json", taxonomy)
        if survey else {**mapping_override, "rules": sorted(mapping_override["rules"], key=lambda rule: -rule["priority"])}
    )
    profiles = load_property_profiles(properties_path, taxonomy)
    stats: Counter[str] = Counter()
    report = census_hcmc_candidates(
        iter_pbf_elements(PBF, reader_stats=stats), boundary=Boundary.from_geojson(BOUNDARY),
        taxonomy=taxonomy, mapping_catalog=mapping_catalog, property_profiles=profiles,
        osm_only=True, complete_signatures=True, reader_stats=stats, sample_limit=2,
    )
    return report


def generate(*, survey: bool = False, artifact_output: Path | None = None) -> dict[str, Any]:
    source = verify_pinned_source()
    if survey:
        report = census(ROOT / "data/taxonomies/place_types.poc.v2.json",
                        ROOT / "data/properties/poi_properties.poc.v4.json", survey=True)
        print(json.dumps({
            "counts": report["counts"], "reader_omissions": report["reader_omissions"],
            "signature_count": len(report["signatures"]),
            "unmapped_top": [
                {"signature": row["signature"], "count": row["dispositions"]["unmapped_needs_review"]}
                for row in sorted(report["signatures"], key=lambda r: -r["dispositions"].get("unmapped_needs_review", 0))
                if row["dispositions"].get("unmapped_needs_review")
            ][:180],
        }, ensure_ascii=True, indent=2))
        return report
    taxonomy, enrichment, mapping, properties = resources()
    # First pass discovers candidate types missing all reviewed rules. It writes
    # nothing; category IDs are allocated deterministically before resources.
    discovery = census(
        TAXONOMY, ROOT / "data/properties/poi_properties.poc.v4.json",
        mapping_override=mapping, taxonomy_override=taxonomy,
    )
    observed_summary = add_observed_types(discovery, taxonomy, enrichment, mapping)
    for path, payload in ((TAXONOMY, taxonomy), (ENRICHMENT, enrichment), (PROPERTIES, properties)):
        write_new(path, payload)
    report = census(TAXONOMY, PROPERTIES, mapping_override=mapping)
    # Every unreconciled signature stays separately visible. Ambiguous tags
    # are not conflated with clear OSM types lacking a reviewed taxonomy leaf.
    ambiguous = {"yes", "no", "other", "unknown", "unspecified", "fixme", "repair"}
    exclusions: dict[str, dict[str, str]] = {}
    for row in report["signatures"]:
        if not row["dispositions"].get("unmapped_needs_review"):
            continue
        parts = [part.split("=", 1) for part in row["signature"].split("|")]
        primary = [(key, value) for key, value in parts if key in POI_PRIMARY_KEYS or key in POI_QUALIFIED_VALUES]
        unclear = len(primary) > 1 or any(value.casefold() in ambiguous for _key, value in parts)
        disposition = "intentionally_excluded_ambiguous" if unclear else "intentionally_excluded_unmodelled"
        exclusions[row["signature"]] = {
            "disposition": disposition,
            "reason": (
                "OSM type is ambiguous or lacks the detail needed for a semantically valid leaf; do not infer one."
                if unclear else "Observed POI-like type without a reviewed semantically valid v3 leaf/rule; retained for explicit future review."
            ),
        }
        unresolved = row["dispositions"].pop("unmapped_needs_review")
        row["dispositions"][disposition] = unresolved
    mapping["intentionally_excluded_signatures"] = exclusions
    write_new(MAPPING, mapping)
    validate_catalogs(Taxonomy.load(TAXONOMY), MAPPING, PROPERTIES)
    report["counts"].pop("inside_hcmc_unmapped_needs_review", None)
    for row in report["signatures"]:
        for disposition in ("intentionally_excluded_ambiguous", "intentionally_excluded_unmodelled"):
            report["counts"][f"inside_hcmc_{disposition}"] = (
                report["counts"].get(f"inside_hcmc_{disposition}", 0)
                + row["dispositions"].get(disposition, 0)
            )
    registry = {
        "schema_version": 1, "registry_version": "hcmc-osm-signatures-poc-v3",
        "mapping_version": mapping["mapping_version"], "taxonomy_version": taxonomy["taxonomy_version"],
        "pbf_sha256": source["expected_sha256"], "boundary_sha256": source["boundary_sha256"],
        "counts": report["counts"], "reader_omissions": report["reader_omissions"],
        "observed_type_expansion": observed_summary,
        "limitations": report["limitations"], "signatures": report["signatures"],
        "audit_sha256": stable_hash(report["signatures"]),
    }
    write_new(REGISTRY, registry)
    if artifact_output is not None:
        stats: Counter[str] = Counter()
        manifest = build_artifact(
            iter_pbf_elements(PBF, reader_stats=stats), taxonomy=Taxonomy.load(TAXONOMY),
            mapping_path=MAPPING, property_path=PROPERTIES, boundary_path=BOUNDARY,
            output_dir=artifact_output, source_manifest_path=SOURCE, source_pbf_path=PBF,
            osm_only=True, signature_registry_path=REGISTRY, reader_stats=stats,
        )
        print(json.dumps({"artifact": str(artifact_output), "poi_count": manifest["poi_count"],
                          "registry": str(REGISTRY), "signature_count": len(report["signatures"])}, ensure_ascii=False))
    else:
        print(json.dumps({"registry": str(REGISTRY), "signature_count": len(report["signatures"]),
                          "counts": report["counts"]}, ensure_ascii=False))
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", action="store_true", help="print candidate signature survey without writing files")
    parser.add_argument("--build-artifact", action="store_true", help="also create fresh v3 OSM-only POI artifact")
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    generate(survey=args.survey, artifact_output=args.artifact_output if args.build_artifact else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
