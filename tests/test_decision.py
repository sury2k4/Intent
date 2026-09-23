from model_intent.decision import decide_intent
from model_intent.schemas import Prediction, RetrievalHit
from model_intent.taxonomy import Taxonomy


def taxonomy():
    return Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe", "en": "Health"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc", "en": "Pharmacy"}},
        {"id": "3", "parent_id": "1", "code": "clinic", "names": {"vi": "Phòng khám", "en": "Clinic"}},
        {"id": "4", "parent_id": "0", "code": "park", "names": {"vi": "Công viên", "en": "Park"}},
    ]})


def policy():
    return {
        "policy_version": "test", "fallback_language": "vi", "max_options": 3,
        "thresholds": {"default": {
            "reject_min_score": 0.4, "clarify_min_score": 0.6,
            "accept_min_score": 0.7, "accept_min_margin": 0.08, "ambiguity_delta": 0.08,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?", "parent": "Bạn cần loại cụ thể nào?"}},
        "groups": [{"id": "care", "members": ["2", "3"], "question_vi": "Mua thuốc hay đi khám?"}],
    }


def hit(item_id, code, score, rank):
    return RetrievalHit(item_id, code, score, rank, "vi", code)


def test_high_confidence_leaf_is_accepted():
    prediction = Prediction("q", "mua thuốc", (hit("2", "pharmacy", .9, 1), hit("4", "park", .6, 2)), .9, .3, "fake")
    decision = decide_intent(prediction, taxonomy(), policy())
    assert decision.action == "accept"
    assert decision.accepted_item_id == "2"


def test_close_related_leaves_trigger_clarification():
    prediction = Prediction("q", "đau đầu", (hit("2", "pharmacy", .84, 1), hit("3", "clinic", .81, 2)), .84, .03, "fake")
    decision = decide_intent(prediction, taxonomy(), policy())
    assert decision.action == "clarify"
    assert {option.item_id for option in decision.options} == {"2", "3"}
    assert decision.question == "Mua thuốc hay đi khám?"


def test_low_score_or_unrelated_close_hits_reject():
    low = Prediction("q", "weather", (hit("4", "park", .3, 1),), .3, None, "fake")
    assert decide_intent(low, taxonomy(), policy()).action == "reject"
    unrelated = Prediction("q", "unclear", (hit("2", "pharmacy", .72, 1), hit("4", "park", .70, 2)), .72, .02, "fake")
    decision = decide_intent(unrelated, taxonomy(), policy())
    assert decision.action == "reject"
    assert not decision.options


def test_close_unrelated_leaf_does_not_bypass_semantic_margin():
    first = RetrievalHit("2", "pharmacy", .82, 1, "vi", "mua thuốc")
    second = RetrievalHit("4", "park", .81, 2, "vi", "công viên")
    prediction = Prediction("q", "mua thuoc", (first, second), .82, .01, "fake")
    decision = decide_intent(prediction, taxonomy(), policy())
    assert decision.action == "reject"


def test_related_embedding_candidates_trigger_clarification():
    first = RetrievalHit("2", "pharmacy", .9, 1, "vi", "mua thuốc")
    second = RetrievalHit("3", "clinic", .84, 2, "vi", "phòng khám")
    prediction = Prediction("q", "chưa biết mua thuốc hay đi khám", (first, second), .9, .06, "fake")
    decision = decide_intent(prediction, taxonomy(), policy())
    assert decision.action == "clarify"




def test_parent_cannot_be_final_result():
    prediction = Prediction("q", "sức khỏe", (
        hit("1", "health", .85, 1), hit("2", "pharmacy", .82, 2), hit("3", "clinic", .81, 3),
    ), .85, .03, "fake")
    decision = decide_intent(prediction, taxonomy(), policy())
    assert decision.action == "clarify"
    assert all(option.item_id != "1" for option in decision.options)


def extended_taxonomy():
    return Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe", "en": "Health"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc", "en": "Pharmacy"}},
        {"id": "3", "parent_id": "1", "code": "clinic", "names": {"vi": "Phòng khám", "en": "Clinic"}},
        {"id": "4", "parent_id": "0", "code": "beverage", "names": {"vi": "Quán giải khát", "en": "Beverage"}},
        {"id": "5", "parent_id": "0", "code": "tea", "names": {"vi": "Trà sữa", "en": "Tea"}},
        {"id": "6", "parent_id": "0", "code": "coffee", "names": {"vi": "Cà phê", "en": "Coffee"}},
        {"id": "7", "parent_id": "0", "code": "restaurant", "names": {"vi": "Nhà hàng", "en": "Restaurant"}},
        {"id": "8", "parent_id": "0", "code": "library", "names": {"vi": "Thư viện", "en": "Library"}},
    ]})


def extended_policy():
    return {
        "policy_version": "test-v2", "selectability": "leaves_only", "fallback_language": "vi", "max_options": 4,
        "thresholds": {"default": {
            "reject_min_score": .4, "clarify_min_score": .6,
            "accept_min_score": .75, "accept_min_margin": .08, "ambiguity_delta": .08,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?", "parent": "Bạn cần loại cụ thể nào?"}},
        "groups": [
            {"id": "care", "members": ["2", "3"], "question_vi": "Mua thuốc hay đi khám?"},
            {"id": "food", "members": ["6", "7", "4", "5"], "question_vi": "Bạn muốn loại địa điểm ăn uống nào?"},
            {"id": "work", "members": ["6", "8"], "question_vi": "Bạn muốn uống cà phê hay làm việc ở thư viện?"},
        ],
    }


def test_unanchored_lower_food_cluster_does_not_clarify_medical_top():
    prediction = Prediction("q", "táo bón", (
        hit("2", "pharmacy", .68, 1),
        hit("4", "beverage", .67, 2),
        hit("5", "tea", .66, 3),
    ), .68, .01, "bge-m3")
    decision = decide_intent(prediction, extended_taxonomy(), extended_policy())
    assert decision.action == "reject"
    assert decision.reason == "uncertain_unrelated_candidates"
    assert not decision.options


def test_medical_top_clarifies_only_healthcare_members():
    prediction = Prediction("q", "mua thuốc hay đi khám", (
        hit("2", "pharmacy", .68, 1),
        hit("3", "clinic", .66, 2),
        hit("4", "beverage", .65, 3),
    ), .68, .02, "bge-m3")
    decision = decide_intent(prediction, extended_taxonomy(), extended_policy())
    assert decision.action == "clarify"
    assert {option.item_id for option in decision.options} == {"2", "3"}
    assert all(option.relation == "care" for option in decision.options)


def test_overlapping_groups_do_not_mix_options():
    prediction = Prediction("q", "cà phê làm việc", (
        hit("6", "coffee", .68, 1),
        hit("7", "restaurant", .66, 2),
        hit("8", "library", .65, 3),
    ), .68, .02, "bge-m3")
    decision = decide_intent(prediction, extended_taxonomy(), extended_policy())
    assert decision.action == "clarify"
    assert {option.item_id for option in decision.options} == {"6", "7"}
    assert all(option.relation == "food" for option in decision.options)


# ---- Cluster rescue tests ----

def rescue_taxonomy():
    """Taxonomy with a noisy 'milk-shop' leaf outside any health group."""
    return Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe", "en": "Health"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc", "en": "Pharmacy"}},
        {"id": "3", "parent_id": "1", "code": "clinic", "names": {"vi": "Phòng khám", "en": "Clinic"}},
        {"id": "9", "parent_id": "1", "code": "hospital", "names": {"vi": "Bệnh viện", "en": "Hospital"}},
        {"id": "10", "parent_id": "0", "code": "milk_shop", "names": {"vi": "Cửa hàng sữa", "en": "Milk shop"}},
        {"id": "11", "parent_id": "0", "code": "book_shop", "names": {"vi": "Nhà sách", "en": "Book shop"}},
        {"id": "12", "parent_id": "0", "code": "gym", "names": {"vi": "Phòng gym", "en": "Gym"}},
    ]})


def rescue_policy():
    return {
        "policy_version": "test-rescue", "selectability": "leaves_only",
        "fallback_language": "vi", "max_options": 4,
        "thresholds": {"default": {
            "reject_min_score": .4, "clarify_min_score": .6,
            "accept_min_score": .75, "accept_min_margin": .08,
            "ambiguity_delta": .08, "cluster_rescue_min_count": 2,
            "cluster_rescue_max_rank": 10,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?", "parent": "Bạn cần loại cụ thể nào?"}},
        "groups": [
            {"id": "care", "members": ["2", "3", "9"], "question_vi": "Mua thuốc, đi khám hay vào bệnh viện?"},
        ],
    }


def test_unanchored_cluster_rescue_is_disabled():
    """A lower healthcare cluster cannot override an unrelated rank-one hit."""
    prediction = Prediction("q", "đau bụng", (
        hit("10", "milk_shop", .89, 1),
        hit("11", "book_shop", .88, 2),
        hit("9", "hospital", .87, 3),
        hit("2", "pharmacy", .86, 4),
        hit("3", "clinic", .85, 5),
    ), .89, .01, "fake")
    decision = decide_intent(prediction, rescue_taxonomy(), rescue_policy())
    assert decision.action == "reject"
    assert decision.reason == "uncertain_unrelated_candidates"
    assert not decision.options


def test_cluster_rescue_does_not_fire_for_top1_own_group():
    """
    When top1 (pharmacy) belongs to group 'care' but no other care members
    appear close enough, and food members cluster below, the rescue must NOT
    pick the food cluster (which exists through a different group) — because
    top1 is already anchored to 'care' and the care check already failed.

    This is equivalent to the existing test_unanchored_lower_food_cluster_does_not_clarify_medical_top
    but explicitly validates that rescue skips groups that top1 belongs to.
    """
    prediction = Prediction("q", "táo bón", (
        hit("2", "pharmacy", .68, 1),
        hit("10", "milk_shop", .67, 2),
        hit("11", "book_shop", .66, 3),
    ), .68, .01, "fake")
    decision = decide_intent(prediction, rescue_taxonomy(), rescue_policy())
    assert decision.action == "reject"
    assert decision.reason == "uncertain_unrelated_candidates"


def test_cluster_rescue_respects_clarify_min_score():
    """
    Even if a cluster exists, rescue should NOT fire if the best group member
    score is below clarify_min_score.
    """
    prediction = Prediction("q", "test low", (
        hit("10", "milk_shop", .55, 1),
        hit("2", "pharmacy", .50, 2),
        hit("3", "clinic", .49, 3),
    ), .55, .05, "fake")
    decision = decide_intent(prediction, rescue_taxonomy(), rescue_policy())
    assert decision.action == "reject"


def test_cluster_rescue_requires_min_count_members():
    """
    Only 1 group member in top-k (below rescue_min_count=2) → no rescue.
    """
    prediction = Prediction("q", "test single", (
        hit("10", "milk_shop", .89, 1),
        hit("11", "book_shop", .88, 2),
        hit("9", "hospital", .87, 3),
        hit("12", "gym", .86, 4),
    ), .89, .01, "fake")
    decision = decide_intent(prediction, rescue_taxonomy(), rescue_policy())
    assert decision.action == "reject"


def test_unanchored_cluster_does_not_pick_largest_lower_group():
    """Lower-ranked group size never overrides an unrelated top result."""
    tax = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc"}},
        {"id": "3", "parent_id": "1", "code": "clinic", "names": {"vi": "Phòng khám"}},
        {"id": "9", "parent_id": "1", "code": "hospital", "names": {"vi": "Bệnh viện"}},
        {"id": "4", "parent_id": "0", "code": "beverage", "names": {"vi": "Giải khát"}},
        {"id": "5", "parent_id": "0", "code": "tea", "names": {"vi": "Trà sữa"}},
        {"id": "10", "parent_id": "0", "code": "milk_shop", "names": {"vi": "Sữa"}},
    ]})
    pol = {
        "policy_version": "test-multi", "selectability": "leaves_only",
        "fallback_language": "vi", "max_options": 4,
        "thresholds": {"default": {
            "reject_min_score": .4, "clarify_min_score": .6,
            "accept_min_score": .75, "accept_min_margin": .08,
            "ambiguity_delta": .08, "cluster_rescue_min_count": 2,
            "cluster_rescue_max_rank": 10,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?", "parent": "?"}},
        "groups": [
            {"id": "care", "members": ["2", "3", "9"], "question_vi": "Healthcare?"},
            {"id": "drink", "members": ["4", "5"], "question_vi": "Drink?"},
        ],
    }
    prediction = Prediction("q", "đau bụng", (
        hit("10", "milk_shop", .89, 1),
        hit("9", "hospital", .87, 2),
        hit("2", "pharmacy", .86, 3),
        hit("3", "clinic", .85, 4),
        hit("4", "beverage", .84, 5),
        hit("5", "tea", .83, 6),
    ), .89, .02, "fake")
    decision = decide_intent(prediction, tax, pol)
    assert decision.action == "reject"
    assert decision.reason == "uncertain_unrelated_candidates"
    assert not decision.options


# ---- Tests for Scoring Formula Enhancements ----

def test_clarification_reports_leaf_margin_consistently():
    """
    When rank 2 is a non-leaf parent node (e.g. Health 0.83), prediction.margin
    is 0.85 - 0.83 = 0.02. But the true leaf_margin is pharmacy (0.85) - clinic (0.82) = 0.03.
    Decision margin must report the leaf_margin (0.03), not prediction.margin.
    """
    tax = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc"}},
        {"id": "3", "parent_id": "1", "code": "clinic", "names": {"vi": "Phòng khám"}},
    ]})
    pol = {
        "policy_version": "test-margin",
        "fallback_language": "vi",
        "thresholds": {"default": {
            "reject_min_score": .4, "clarify_min_score": .6,
            "accept_min_score": .75, "accept_min_margin": .08,
            "ambiguity_delta": .08,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?"}},
        "groups": [{"id": "care", "members": ["2", "3"], "question_vi": "Care?"}],
    }
    # Hits: rank 1 is leaf pharmacy (0.85), rank 2 is parent health (0.83), rank 3 is leaf clinic (0.82)
    # prediction.margin = 0.85 - 0.83 = 0.02
    # leaf_margin = 0.85 - 0.82 = 0.03
    pred = Prediction("q", "đau bụng", (
        hit("2", "pharmacy", .85, 1),
        hit("1", "health", .83, 2),
        hit("3", "clinic", .82, 3),
    ), .85, .02, "fake")

    decision = decide_intent(pred, tax, pol)
    assert decision.action == "clarify"
    # Must report leaf_margin (0.03), not prediction.margin (0.02)
    assert round(decision.margin, 4) == 0.03


def test_runner_up_in_same_group_blocks_generic_accept_and_triggers_clarify():
    """
    Top1 is pharmacy (0.49), runner-up is hospital (0.46), both in healthcare_need.
    Generic accept_min_margin is 0.02. Leaf margin is 0.03 >= 0.02.
    Because runner-up shares the ambiguity group, it must NOT silently accept pharmacy.
    It must trigger clarification between the group members.
    """
    tax = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc"}},
        {"id": "3", "parent_id": "1", "code": "hospital", "names": {"vi": "Bệnh viện"}},
        {"id": "4", "parent_id": "0", "code": "park", "names": {"vi": "Công viên"}},
    ]})
    pol = {
        "policy_version": "test-group-accept",
        "fallback_language": "vi",
        "thresholds": {"bge-m3": {
            "reject_min_score": 0.45, "clarify_min_score": 0.46,
            "accept_min_score": 0.46, "accept_min_margin": 0.02,
            "ambiguity_delta": 0.02, "group_accept_min_margin": 0.05,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?"}},
        "groups": [{"id": "healthcare_need", "members": ["2", "3"], "question_vi": "Bạn cần mua thuốc hay bệnh viện?"}],
    }
    # Margin = 0.49 - 0.46 = 0.03 >= accept_min_margin (0.02), but < group_accept_min_margin (0.05)
    pred = Prediction("q", "đau bụng", (
        hit("2", "pharmacy", 0.49, 1),
        hit("3", "hospital", 0.46, 2),
    ), 0.49, 0.03, "bge-m3")

    decision = decide_intent(pred, tax, pol)
    assert decision.action == "clarify"
    assert decision.reason == "near_group_competition"
    assert {opt.item_id for opt in decision.options} == {"2", "3"}
    assert decision.question == "Bạn cần mua thuốc hay bệnh viện?"


def test_runner_up_in_same_group_accepts_when_margin_exceeds_group_threshold():
    """
    When the query is specific (e.g. 'mua thuốc panadol'), pharmacy (0.90) decisively
    beats hospital (0.60) by margin 0.30 >= group_accept_min_margin (0.05).
    In this case, accept is permitted.
    """
    tax = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc"}},
        {"id": "3", "parent_id": "1", "code": "hospital", "names": {"vi": "Bệnh viện"}},
    ]})
    pol = {
        "policy_version": "test-group-accept-decisive",
        "fallback_language": "vi",
        "thresholds": {"default": {
            "reject_min_score": 0.45, "clarify_min_score": 0.55,
            "accept_min_score": 0.60, "accept_min_margin": 0.035,
            "ambiguity_delta": 0.06, "group_accept_min_margin": 0.08,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?"}},
        "groups": [{"id": "healthcare_need", "members": ["2", "3"], "question_vi": "Thuốc hay viện?"}],
    }
    pred = Prediction("q", "mua thuốc panadol", (
        hit("2", "pharmacy", 0.90, 1),
        hit("3", "hospital", 0.60, 2),
    ), 0.90, 0.30, "fake")

    decision = decide_intent(pred, tax, pol)
    assert decision.action == "accept"
    assert decision.accepted_item_id == "2"


def test_adaptive_ambiguity_delta_scales_with_tight_model_score_range():
    """
    Simulates e5-small where top score is 0.92 and clarify_min is 0.82.
    With static delta = 0.035, a clinic at 0.86 (diff 0.06) would be dropped.
    With adaptive delta, span = (0.92 - 0.82) * 0.8 = 0.08, so clinic at 0.86 is included!
    """
    tax = Taxonomy.from_payload({"items": [
        {"id": "1", "parent_id": "0", "code": "health", "names": {"vi": "Sức khỏe"}},
        {"id": "2", "parent_id": "1", "code": "pharmacy", "names": {"vi": "Nhà thuốc"}},
        {"id": "3", "parent_id": "1", "code": "clinic", "names": {"vi": "Phòng khám"}},
    ]})
    pol = {
        "policy_version": "test-adaptive",
        "fallback_language": "vi",
        "thresholds": {"e5-small-multilingual": {
            "reject_min_score": 0.78, "clarify_min_score": 0.82,
            "accept_min_score": 0.84, "accept_min_margin": 0.018,
            "ambiguity_delta": 0.035,
        }},
        "questions": {"vi": {"generic": "Bạn cần loại nào?"}},
        "groups": [{"id": "care", "members": ["2", "3"], "question_vi": "Care?"}],
    }
    # Top is pharmacy at 0.92. Clinic is at 0.86.
    # diff = 0.06 > static 0.035.
    # Adaptive delta = max(0.035, 0.036, (0.92 - 0.82) * 0.8 = 0.08) = 0.08 >= 0.06.
    pred = Prediction("q", "y tế", (
        hit("2", "pharmacy", 0.92, 1),
        hit("3", "clinic", 0.86, 2),
    ), 0.92, 0.06, "e5-small-multilingual")

    decision = decide_intent(pred, tax, pol)
    assert decision.action == "clarify"
    assert {opt.item_id for opt in decision.options} == {"2", "3"}
