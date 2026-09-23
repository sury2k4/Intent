from model_intent.ui import diagnostic_rows, result_view_model


def test_ui_view_model_handles_clarification():
    result = {
        "prediction": {
            "top1_score": 0.9, "margin": 0.01,
            "query_variants": ["đau đầu", "dau dau"],
            "hits": [{
                "rank": 1, "item_id": "34", "code": "pharmacy", "score": 0.9,
                "path": ["Sức khỏe", "Nhà thuốc"], "matched_text": "mua thuốc",
                "language": "vi", "document_variant_kind": "example",
                "query_variant_kind": "original",
            }],
        },
        "decision": {
            "action": "clarify", "reason": "near_group_competition",
            "question": "Mua thuốc hay đi khám?", "options": [{"item_id": "34"}],
            "policy_version": "test",
        },
        "runtime": {"query_elapsed_ms": 12.5},
    }
    view = result_view_model(result)
    assert view["action"] == "clarify"
    assert view["style"]["label"] == "CLARIFY"
    assert view["question"] == "Mua thuốc hay đi khám?"
    rows = diagnostic_rows(result)
    assert rows[0]["Category"] == "Sức khỏe > Nhà thuốc"
    assert rows[0]["Cosine score"] == 0.9


def test_ui_view_model_handles_accept_and_reject():
    accepted = result_view_model({"decision": {"action": "accept", "accepted_item_id": "34"}, "prediction": {"hits": []}})
    rejected = result_view_model({"decision": {"action": "reject"}, "prediction": {"hits": []}})
    assert accepted["style"]["label"] == "ACCEPT"
    assert rejected["style"]["label"] == "REJECT"
