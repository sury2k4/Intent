from pathlib import Path

import pytest

from model_intent.taxonomy import Taxonomy, TaxonomyError

TAXONOMY = Path(__file__).parents[1] / "data/taxonomies/place_types.poc.v1.json"


def test_supplied_taxonomy_loads_and_has_expected_core_types():
    taxonomy = Taxonomy.load(TAXONOMY)
    assert len(taxonomy.items) >= 200
    assert taxonomy.by_code["pharmacy"].id == "34"
    assert taxonomy.by_code["gas_station"].id == "85"
    assert taxonomy.path_names("34", "vi") == ("Sức khỏe", "Nhà thuốc")


def test_missing_parent_is_warning_in_partial_taxonomy():
    taxonomy = Taxonomy.from_payload({
        "status": "partial_poc", "items": [
            {"id": "1", "parent_id": "999", "code": "leaf", "names": {"vi": "Lá"}},
        ],
    })
    assert any("missing parent" in warning for warning in taxonomy.validate())
    assert taxonomy.path_ids("1") == ("1",)


def test_parent_cycle_is_rejected():
    with pytest.raises(TaxonomyError, match="cycle"):
        Taxonomy.from_payload({"items": [
            {"id": "1", "parent_id": "2", "code": "a", "names": {"en": "A"}},
            {"id": "2", "parent_id": "1", "code": "b", "names": {"en": "B"}},
        ]})
