import pytest

from model_intent.query_scope import non_place_reason


@pytest.mark.parametrize("text,reason", [
    ("tìm phần mềm quản lý bán hàng cho laptop", "digital_product_not_place_search"),
    ("find accounting software for my laptop", "digital_product_not_place_search"),
    ("how do I reset my pharmacy password", "informational_not_place_search"),
    ("cách mở quán cà phê", "informational_not_place_search"),
    ("tìm quán cà phê có wifi", None),
    ("tìm cửa hàng phần mềm", None),
    ("find a software store nearby", None),
])
def test_non_place_veto_preserves_physical_searches(text, reason):
    assert non_place_reason(text) == reason
