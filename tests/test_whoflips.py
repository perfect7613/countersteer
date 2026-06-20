import json

import pytest

from countersteer.whoflips import (
    answer_flip_metrics,
    build_whoflips_bundle,
    parse_whoflips_row,
    render_challenge,
)


def _row(index: int, *, coercion_text: str | None = None, score: float = 0.5) -> dict:
    return {
        "original_id": f"mmlu_{index}",
        "subject": "logic",
        "question": f"Question {index}?",
        "choices": json.dumps({"A": "one", "B": "two", "C": "three", "D": "four"}),
        "correct_label": "B",
        "correct_text": "two",
        "coercion_model": "example/model",
        "coercion_text": coercion_text or ("A coherent but incorrect argument. " * 4),
        "score": score,
    }


def test_parser_rejects_malformed_choice_schema() -> None:
    row = _row(1)
    row["choices"] = json.dumps({"A": "one", "B": "two"})
    with pytest.raises(ValueError, match="A/B/C/D"):
        parse_whoflips_row(row)


def test_split_is_deterministic_disjoint_and_filters_failed_coercions() -> None:
    rows = [_row(index) for index in range(18)]
    rows.append(_row(99, coercion_text="I_AM_WEAK", score=0.0))
    first = build_whoflips_bundle(
        rows, train_size=8, development_size=4, confirmation_size=3
    )
    second = build_whoflips_bundle(
        reversed(rows), train_size=8, development_size=4, confirmation_size=3
    )

    assert first.split_hash == second.split_hash
    assert first.dataset_hash == second.dataset_hash
    assert first.rejected_rows == 1
    partitions = [
        set(first.train_ids),
        set(first.development_ids),
        set(first.confirmation_ids),
        set(first.reserve_ids),
    ]
    assert [len(partition) for partition in partitions] == [8, 4, 3, 3]
    assert all(
        not left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    )


def test_confirmation_partition_is_content_independent() -> None:
    rows = [_row(index) for index in range(20)]
    original = build_whoflips_bundle(
        rows, train_size=8, development_size=4, confirmation_size=4
    )
    rows[0]["coercion_text"] += " Extra argument text."
    changed = build_whoflips_bundle(
        rows, train_size=8, development_size=4, confirmation_size=4
    )

    assert original.confirmation_ids == changed.confirmation_ids
    assert original.dataset_hash != changed.dataset_hash


def test_challenge_prompt_has_independent_solve_variant() -> None:
    item = parse_whoflips_row(_row(1))
    ordinary = render_challenge(item, independent_solve=False)
    independent = render_challenge(item, independent_solve=True)

    assert item.coercion_text in ordinary
    assert "Solve the original problem independently" in independent
    assert "independently" not in ordinary


def test_answer_flip_rate_is_conditional_on_initial_correctness() -> None:
    records = [
        {"item_id": "1", "correct_label": "A", "initial_label": "A", "challenged_label": "B"},
        {"item_id": "2", "correct_label": "B", "initial_label": "B", "challenged_label": "B"},
        {"item_id": "3", "correct_label": "C", "initial_label": "D", "challenged_label": "C"},
        {"item_id": "4", "correct_label": "D", "initial_label": "D", "challenged_label": "A"},
    ]

    metrics = answer_flip_metrics(records)

    assert metrics["initial_accuracy"] == 0.75
    assert metrics["answer_flip_rate"] == pytest.approx(2 / 3)
    assert metrics["conditional_post_challenge_accuracy"] == pytest.approx(1 / 3)
    assert metrics["post_challenge_accuracy"] == 0.5
