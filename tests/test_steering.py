import pytest

from countersteer.steering import (
    CoefficientResult,
    balanced_contrastive_vector,
    degeneracy_metrics,
    evaluate_strategy,
    select_coefficient,
)


def test_causal_vector_filters_pairs_and_balances_answer_labels() -> None:
    records = [
        {
            "item_id": "a1",
            "correct_label": "A",
            "neutral_activation": [3.0, 1.0],
            "wrong_activation": [1.0, 1.0],
        },
        {
            "item_id": "a2",
            "correct_label": "A",
            "neutral_activation": [5.0, 1.0],
            "wrong_activation": [1.0, 1.0],
        },
        {
            "item_id": "b1",
            "correct_label": "B",
            "neutral_activation": [1.0, 5.0],
            "wrong_activation": [1.0, 1.0],
        },
    ]

    assert balanced_contrastive_vector(records, allowed_ids={"a1", "b1"}) == [1.0, 2.0]
    assert balanced_contrastive_vector(records) == [1.5, 2.0]


def test_coefficient_selection_enforces_accuracy_constraint() -> None:
    selected = select_coefficient(
        [
            CoefficientResult(0.5, 0.96, 0.4),
            CoefficientResult(1.0, 0.95, 0.2),
            CoefficientResult(2.0, 0.90, 0.0),
        ],
        baseline_neutral_accuracy=1.0,
    )
    assert selected.coefficient == 1.0


def _records(strategy_flips: bool = False) -> list[dict]:
    rows = []
    for item_id, correct_label in (("one", "A"), ("two", "B")):
        wrong_label = "B" if correct_label == "A" else "A"
        for condition in ("neutral", "correct_belief", "wrong_belief"):
            predicted = correct_label
            probability = 0.8
            if condition == "wrong_belief" and not strategy_flips:
                predicted = wrong_label
                probability = 0.2
            rows.append(
                {
                    "item_id": item_id,
                    "condition": condition,
                    "correct_label": correct_label,
                    "wrong_label": wrong_label,
                    "predicted_label": predicted,
                    "probability_correct": probability,
                    "generated_answer": predicted,
                }
            )
    return rows


def test_evaluation_matches_hand_calculated_improvement() -> None:
    baseline = _records(strategy_flips=False)
    steered = _records(strategy_flips=True)
    result = evaluate_strategy(steered, baseline, seed=42, bootstrap_replicates=200)

    assert result["neutral_accuracy"] == 1.0
    assert result["capitulation_rate"] == 0.0
    assert result["correct_belief_agreement"] == 1.0
    assert result["paired_changes_vs_no_steering"]["capitulation_reduction"]["mean"] == 1.0
    assert result["paired_changes_vs_no_steering"]["sensitivity_reduction"]["mean"] == pytest.approx(0.6)


def test_degeneracy_metrics_surface_failures() -> None:
    rows = _records(strategy_flips=True)
    for row in rows:
        row["predicted_label"] = "A"
        row["generated_answer"] = "Perhaps A"
    result = degeneracy_metrics(rows)

    assert result["label_a_rate"] == 1.0
    assert result["label_preference_imbalance"] == 0.5
    assert result["blanket_disagreement_rate"] == 0.5
    assert result["hedging_rate"] == 1.0
