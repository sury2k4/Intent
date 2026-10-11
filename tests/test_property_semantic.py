from __future__ import annotations

import numpy as np
import pytest

from model_intent.property_extraction import PropertyExtractor
from model_intent.property_pg_index import PropertyPgIndex
from model_intent.property_semantic import prepare_property_queries, rank_property_scores


def definitions():
    return {
        "air_conditioning": {
            "queryable": True, "value_type": "boolean",
            "aliases": {"vi": ["máy lạnh", "điều hòa không khí"], "en": ["air conditioning"]},
        },
        "shade": {
            "queryable": True, "value_type": "boolean",
            "aliases": {"vi": ["bóng mát"], "en": ["shaded"]},
        },
        "rating": {
            "queryable": False, "value_type": "ordinal", "aliases": {"vi": ["đánh giá"]},
        },
    }


def test_semantic_helpers_score_suggestions_but_never_apply_them():
    spans = [
        {"text": "có mát rượi", "start_offset": 10, "end_offset": 21},
        {"text": "không cần chỗ cất ô tô", "start_offset": 22, "end_offset": 44},
    ]
    prepared, valid, probes = prepare_property_queries(
        spans, PropertyExtractor(definitions()).unique_typo_alias,
        query_text=PropertyPgIndex._query_text,
    )
    assert prepared == ["mát rượi", "chỗ cất ô tô"]
    assert valid == list(enumerate(prepared)) and not probes
    suggestions = rank_property_scores(
        spans, prepared, valid, np.array([[.90, .82], [.87, .80]]),
        ("shade", "air_conditioning"), definitions(), typo_probes=probes,
    )
    assert suggestions[0]["candidates"][0]["property_id"] == "shade"
    assert "rating" not in {candidate["property_id"] for row in suggestions for candidate in row["candidates"]}
    assert suggestions[0]["status"] == "needs_confirmation"
    assert suggestions[1]["status"] == "not_requested"
    assert all("applied" not in row for row in suggestions)


def test_guarded_short_typo_uses_unique_alias_and_requires_correlated_top_hit():
    catalog = {"internet_access": {"value_type": "boolean", "queryable": True,
                                   "aliases": {"vi": ["wifi"], "en": ["wireless internet"]}},
               "shade": {"value_type": "boolean", "queryable": True,
                         "aliases": {"vi": ["bóng mát"], "en": ["shade"]}}}
    extractor = PropertyExtractor(catalog)
    assert extractor.unique_typo_alias("wifa") == ("internet_access", "wifi")
    assert extractor.unique_typo_alias("wifi") is None
    assert extractor.unique_typo_alias("giông") is None
    spans = [{"text": "không có wifa", "start_offset": 12, "end_offset": 25}]
    prepared, valid, probes = prepare_property_queries(
        spans, extractor.unique_typo_alias, query_text=PropertyPgIndex._query_text,
    )
    assert prepared == ["wifi"] and valid == [(0, "wifi")]
    assert probes[0]["candidate_property_id"] == "internet_access"

    def suggestions(first, second):
        return rank_property_scores(
            spans, prepared, valid, np.array([[first, second]]),
            ("internet_access", "shade"), catalog, typo_probes=probes,
        )[0]

    confirmed = suggestions(.86, .80)
    assert confirmed["resolution"]["corroborated"] is True
    assert [row["property_id"] for row in confirmed["candidates"]] == ["internet_access"]
    assert confirmed["status"] == "needs_confirmation"
    for first, second in ((.85, .84), (.75, .70), (.82, .90)):
        unconfirmed = suggestions(first, second)
        assert unconfirmed["resolution"]["corroborated"] is False
        assert unconfirmed["candidates"] == []
        assert unconfirmed["status"] == "no_candidate"


def test_property_query_text_retains_accents_and_discards_cues():
    query = PropertyPgIndex._query_text
    assert query("có mát rượi") == "mát rượi"
    assert query("không có chỗ cất ô tô") == "chỗ cất ô tô"
    assert query("ưu tiên mát rượi, nếu có") == "mát rượi"
    assert query("và máy lạnh") == "máy lạnh"
    assert query("không muốn chỗ che mưa") == "chỗ che mưa"
    assert query("do not want overhead cover") == "overhead cover"


def test_unsupported_short_queries_and_invalid_score_matrices():
    spans = [{"text": "có wifi"}, {"text": "có mát rượi"}]
    prepared, valid, probes = prepare_property_queries(
        spans, PropertyExtractor(definitions()).unique_typo_alias,
        query_text=PropertyPgIndex._query_text,
    )
    assert prepared == ["wifi", "mát rượi"] and valid == [(1, "mát rượi")] and not probes
    output = rank_property_scores(spans, prepared, valid, np.array([[.86]]), ("shade",), definitions())
    assert output[0]["status"] == "no_candidate" and output[0]["candidates"] == []
    with pytest.raises(ValueError, match="scores do not match"):
        rank_property_scores(spans, prepared, valid, np.array([[np.nan]]), ("shade",), definitions())
    with pytest.raises(ValueError, match="scores do not match"):
        rank_property_scores(spans, prepared, valid, np.empty((0, 1)), ("shade",), definitions())
