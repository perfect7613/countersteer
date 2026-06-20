import pytest

from countersteer.patching import (
    apply_replacement,
    last_non_padding_index,
    norm_matched_random_replacement,
    restoration_fraction,
    select_patch_examples,
    vector_norm,
    vector_subtract,
)


def test_last_prompt_token_targeting_handles_left_and_right_padding() -> None:
    assert last_non_padding_index([0, 0, 1, 1, 1]) == 4
    assert last_non_padding_index([1, 1, 1, 0, 0]) == 2
    with pytest.raises(ValueError, match="no prompt tokens"):
        last_non_padding_index([0, 0])


def _record(item: str, condition: str, predicted: str, margin: float) -> dict:
    return {
        "item_id": item,
        "condition": condition,
        "correct_label": "A",
        "wrong_label": "B",
        "predicted_label": predicted,
        "correct_logit_margin": margin,
    }


def test_selection_prefers_genuine_capitulation() -> None:
    records = [
        _record("en-001", "neutral", "A", 4.0),
        _record("en-001", "correct_belief", "A", 5.0),
        _record("en-001", "wrong_belief", "B", -1.0),
        _record("en-002", "neutral", "A", 4.0),
        _record("en-002", "correct_belief", "A", 5.0),
        _record("en-002", "wrong_belief", "A", 1.0),
    ]

    chosen, eligible_count = select_patch_examples(
        records, ["en-001", "en-002"], limit=2
    )
    assert eligible_count == 1
    assert [row.item_id for row in chosen] == ["en-001"]
    assert chosen[0].selection_mode == "behavioral_capitulation"


def test_selection_labels_margin_only_fallback() -> None:
    records = [
        _record("en-001", "neutral", "A", 4.0),
        _record("en-001", "correct_belief", "A", 5.0),
        _record("en-001", "wrong_belief", "A", 1.0),
    ]

    chosen, eligible_count = select_patch_examples(records, ["en-001"], limit=1)
    assert eligible_count == 0
    assert chosen[0].selection_mode == "margin_pressure_fallback"
    assert chosen[0].margin_suppression == 3.0


def test_random_control_is_seeded_and_norm_matched() -> None:
    wrong = [1.0, 2.0, 3.0]
    neutral = [2.0, 4.0, 3.0]
    first = norm_matched_random_replacement(wrong, neutral, seed=42)
    second = norm_matched_random_replacement(wrong, neutral, seed=42)

    assert first == second
    assert vector_norm(vector_subtract(first, wrong)) == pytest.approx(
        vector_norm(vector_subtract(neutral, wrong))
    )


def test_disabled_patch_is_exactly_the_original() -> None:
    original = [1.0, 2.0]
    replacement = [9.0, 9.0]

    assert apply_replacement(original, replacement, enabled=False) == original
    assert apply_replacement(original, replacement, enabled=True) == replacement


def test_restoration_sign_convention() -> None:
    assert restoration_fraction(
        neutral_margin=4.0, wrong_margin=1.0, patched_margin=4.0
    ) == 1.0
    assert restoration_fraction(
        neutral_margin=4.0, wrong_margin=1.0, patched_margin=1.0
    ) == 0.0
    assert restoration_fraction(
        neutral_margin=4.0, wrong_margin=1.0, patched_margin=0.0
    ) < 0.0
