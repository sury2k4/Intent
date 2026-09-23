from model_intent.evaluation import evaluate
from model_intent.schemas import ClarificationOption, EvaluationCase, IntentDecision, Prediction, RetrievalHit


def pred(query_id, item_id="1", score=0.9):
    return Prediction(query_id, "q", (RetrievalHit(item_id, "x", score, 1, "vi", "X"),), score, None, "fake")


def test_evaluation_handles_strict_acceptable_and_ood():
    cases = [
        EvaluationCase("a", "semantic", "test", "a", gold_ids=("1",)),
        EvaluationCase("b", "semantic", "test", "b", gold_ids=("2",), acceptable_ids=("1",)),
        EvaluationCase("c", "semantic", "test", "c", is_ood=True),
    ]
    metrics = evaluate(cases, {"a": pred("a"), "b": pred("b"), "c": pred("c")})
    overall = metrics["overall"]
    assert overall["strict_hit@1"]["mean"] == 0.5
    assert overall["acceptable_hit@1"]["mean"] == 1.0
    assert overall["ood_n"] == 1




def test_clarification_option_metrics_measure_precision_and_recall():
    case = EvaluationCase(
        "c", "medical", "test", "uncertain", expected_action="clarify",
        clarification_ids=("1", "2", "3"),
    )
    decision = IntentDecision(
        action="clarify", reason="near_group_competition", language="vi",
        options=(
            ClarificationOption("1", "a", "A", "", .8, "care"),
            ClarificationOption("4", "d", "D", "", .7, "food"),
        ),
    )
    metrics = evaluate(
        [case],
        {"c": pred("c", "1", .8)},
        decisions={"c": decision},
    )["overall"]["decision"]
    assert metrics["option_recall"] == 1 / 3
    assert metrics["option_precision"] == 1 / 2
    assert metrics["option_exact_match"] == 0.0


def test_calibration_and_test_are_separate():
    cases = [EvaluationCase("a", "semantic", "calibration", "a", gold_ids=("1",))]
    metrics = evaluate(cases, {"a": pred("a")})
    assert "calibration" in metrics["tracks"]
    assert metrics["overall"]["n"] == 1
