"""Data-only checks for the bilingual property intent profile catalog."""

import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/properties/poi_properties.poc.v8.json"
PROFILES_PATH = ROOT / "data/properties/property_semantic_profiles.poc.v2.json"
LANGUAGES = {"vi", "en"}
PROFILE_FIELDS = {
    "property_id", "facet", "value_type", "labels", "definitions",
    "embedding_definitions", "positive_phrases", "confusable_with",
}


def load_data():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    return catalog, profiles


def test_profile_schema_and_exact_catalog_coverage():
    catalog, data = load_data()
    assert set(data) == {"schema_version", "version", "catalog_version", "profiles"}
    assert data["schema_version"] == 2
    assert data["version"] == "property-semantic-poc-v2"
    assert data["catalog_version"] == catalog["property_taxonomy_version"] == "poi-properties-poc-v8-osm-only"

    expected = {
        item["id"]: item
        for item in catalog["properties"]
        if item.get("queryable", True) and item["value_type"] != "string"
    }
    assert len(catalog["properties"]) == 107
    assert len(expected) == 106
    assert isinstance(data["profiles"], list)
    ids = [item["property_id"] for item in data["profiles"]]
    assert len(ids) == len(set(ids)) == len(expected)
    assert set(ids) == set(expected)
    assert "cuisine" not in ids
    assert all(value not in ids for value in ("rating", "radius", "radius_km"))

    for profile in data["profiles"]:
        assert set(profile) == PROFILE_FIELDS, profile["property_id"]
        canonical = expected[profile["property_id"]]
        assert profile["facet"] == canonical["facet"]
        assert profile["value_type"] == canonical["value_type"]


def test_bilingual_positive_descriptions_and_no_negative_requests():
    _, data = load_data()
    # Only embedding_definitions and positive_phrases may be placed in positive retrieval passages.
    # The ordinary definitions may contrast related concepts and remain metadata only.
    excluded_requests = re.compile(
        r"\b(?:không(?!\s+(?:gian|chạm)\b)|chẳng|chưa|đừng|tránh|thiếu|vắng|"
        r"without|avoid|exclude|lacks?|not|no|never|neither|"
        r"rather\s+than|distinct\s+from|instead\s+of|unlike|"
        r"doesn['’]?t|isn['’]?t|aren['’]?t|can['’]?t)\b",
        flags=re.IGNORECASE,
    )
    for profile in data["profiles"]:
        for field in ("labels", "definitions", "embedding_definitions", "positive_phrases"):
            assert set(profile[field]) == LANGUAGES, (profile["property_id"], field)
        for language in LANGUAGES:
            assert isinstance(profile["labels"][language], str)
            assert profile["labels"][language].strip() == profile["labels"][language]
            assert len(profile["labels"][language]) > 2
            assert isinstance(profile["definitions"][language], str)
            assert len(profile["definitions"][language].strip()) >= 35
            passage = profile["embedding_definitions"][language]
            assert isinstance(passage, str) and len(passage.strip()) >= 35
            assert passage == passage.strip()
            assert not excluded_requests.search(passage), (profile["property_id"], language, passage)
            phrases = profile["positive_phrases"][language]
            assert isinstance(phrases, list) and len(phrases) >= 3
            assert len(phrases) == len({phrase.casefold() for phrase in phrases})
            for phrase in phrases:
                assert isinstance(phrase, str) and len(phrase.strip()) >= 6
                assert phrase == phrase.strip()
                assert not excluded_requests.search(phrase), (profile["property_id"], language, phrase)


def test_confusables_are_known_distinct_queryable_properties():
    _, data = load_data()
    profiles = {item["property_id"]: item for item in data["profiles"]}
    for property_id, profile in profiles.items():
        confusables = profile["confusable_with"]
        assert isinstance(confusables, list)
        assert len(confusables) == len(set(confusables))
        assert property_id not in confusables
        assert set(confusables) <= profiles.keys(), property_id


@pytest.mark.parametrize("left,right", [
    ("air_conditioning", "airy"), ("air_conditioning", "shade"),
    ("parking", "motorbike_parking"), ("parking", "covered_parking"),
    ("parking", "disabled_parking"), ("open_late", "open_24h"),
    ("open_24h", "reception_24h"), ("covered_area", "shade"),
    ("covered_area", "canopy"), ("covered_area", "shelter"),
    ("delivery", "home_delivery"), ("delivery", "takeaway"),
    ("delivery", "online_ordering"), ("wheelchair", "accessible_toilet"),
    ("toilets", "accessible_toilet"), ("toilets", "gender_neutral"),
    ("indoor_seating", "outdoor_seating"), ("seating", "outdoor_seating"),
    ("accepts_cards", "contactless"), ("accepts_cards", "cashless"),
    ("contactless", "cashless"), ("route_information", "realtime_arrivals"),
    ("ticketing", "queue_ticket"), ("reservation", "appointment"),
    ("breakfast", "cafeteria"),
])
def test_critical_intent_boundaries_are_explicit_in_both_directions(left, right):
    _, data = load_data()
    profiles = {item["property_id"]: item for item in data["profiles"]}
    assert right in profiles[left]["confusable_with"]
    assert left in profiles[right]["confusable_with"]


@pytest.mark.parametrize("property_id,language,phrase", [
    ("air_conditioning", "vi", "phòng chạy điều hòa"),
    ("airy", "vi", "quán thoáng mát"),
    ("shade", "vi", "nhiều bóng cây"),
    ("parking", "vi", "có chỗ đỗ ô tô"),
    ("motorbike_parking", "vi", "chỗ gửi xe máy"),
    ("open_late", "en", "open late at night"),
    ("open_24h", "vi", "mở cửa 24/24"),
    ("reception_24h", "en", "24-hour front desk"),
    ("home_delivery", "vi", "giao hàng tận nhà"),
    ("accessible_toilet", "en", "wheelchair-accessible toilet"),
    ("contactless", "en", "tap to pay"),
    ("cashless", "vi", "thanh toán phi tiền mặt"),
])
def test_vetted_intent_examples_stay_attached_to_the_right_property(property_id, language, phrase):
    _, data = load_data()
    profiles = {item["property_id"]: item for item in data["profiles"]}
    assert phrase in profiles[property_id]["positive_phrases"][language]
