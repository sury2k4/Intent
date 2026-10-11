import pytest

from model_intent.query_scope import has_place_seeking_anchor, non_place_reason


@pytest.mark.parametrize("text,reason", [
    ("tìm phần mềm quản lý bán hàng cho laptop", "digital_product_not_place_search"),
    ("find accounting software for my laptop", "digital_product_not_place_search"),
    ("how do I reset my pharmacy password", "informational_not_place_search"),
    ("cách mở quán cà phê", "informational_not_place_search"),
    ("not a hospital", "negated_place_only"),
    ("find not a hospital", "negated_place_only"),
    ("not a hospital, not a pharmacy", "negated_place_only"),
    ("not a hospital but with wifi", "negated_place_only"),
    ("không phải nhà thương, có bãi đỗ xe", "negated_place_only"),
    ("không phải nhà thương", "negated_place_only"),
    ("Tui không tìm nhà thuốc", "negated_place_only"),
    ("Mình không muốn tìm nhà thuốc", "negated_place_only"),
    ("I do not want a hospital", "negated_place_only"),
    ("Tui không tìm nhà thuốc, mình muốn tìm quán cà phê", None),
    ("không phải nhà thương, nhà thuốc", None),
    ("not a hospital, a pharmacy near me", None),
    ("Not an ATM: I need an actual bank branch with staff.", None),
    ("not a hospital: with wifi", "negated_place_only"),
    ("Bảo tàng có hướng dẫn tham quan thì hay.", None),
    ("không phải nhà thương, tìm nhà thuốc", None),
    ("tìm quán cà phê có wifi", None),
    ("Tui không tìm quán trà sữa; muốn hỏi cách tự pha trà sữa ở nhà.", "informational_not_place_search"),
    ("Không cần chỉ nhà thuốc, cho tui biết thuốc ho uống sao mới đúng.", "informational_not_place_search"),
    ("Mình không muốn tìm nhà thuốc, chỉ hỏi thuốc ho uống sao?", "informational_not_place_search"),
    ("Tui không ra cây xăng, muốn biết giá xăng ngày mai tăng hay giảm.", "informational_not_place_search"),
    ("Bạn xem giúp xét nghiệm máu của tôi có bị bệnh gì không?", "informational_not_place_search"),
    ("không phải nhà thuốc nhưng tôi muốn tìm quán cà phê", None),
    ("không phải nhà thuốc, tìm quán trà sữa gần đây", None),
    ("mình hỏi nhà thuốc ở đâu để mua thuốc ho", None),
    ("tìm cửa hàng phần mềm", None),
    ("bệnh viện có ứng dụng đặt lịch", None),
    ("hospital with an app for appointments", None),
    ("tìm phần mềm quản lý bệnh viện", "digital_product_not_place_search"),
    ("find an application for hospitals", "digital_product_not_place_search"),
    ("find a software store nearby", None),
])
def test_non_place_veto_preserves_physical_searches(text, reason):
    assert non_place_reason(text) == reason


@pytest.mark.parametrize("query,term,allowed", [
    ("Tui kiếm trường đại học ở Thủ Đức", "trường đại học", True),
    ("Mình muốn tìm nhà thuốc gần đây", "nhà thuốc", True),
    ("find a pharmacy near me", "pharmacy", True),
    ("không tìm quán trà sữa", "quán trà sữa", False),
    ("not looking for a hospital", "hospital", False),
    ("nhà thuốc nào bán thuốc ho", "nhà thuốc", False),
])
def test_only_positive_place_seeking_prefix_can_anchor_category(query, term, allowed):
    assert has_place_seeking_anchor(query, query.index(term)) is allowed
