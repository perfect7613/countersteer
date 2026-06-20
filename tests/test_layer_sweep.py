from countersteer.layer_sweep import (
    rank_layers,
    refinement_layers,
    regularly_spaced_layers,
    select_causal_layer,
    summarize_layer,
)


def _rows(layer: int, matched: float, unrelated: float, random: float) -> list[dict]:
    return [
        {
            "item_id": f"item-{index}",
            "layer_index": layer,
            "matched_restoration_fraction": matched + index * 0.01,
            "unrelated_restoration_fraction": unrelated,
            "random_restoration_fraction": random,
            "matched_answer_restored": matched >= 0.5,
            "unrelated_answer_restored": unrelated >= 0.5,
            "random_answer_restored": random >= 0.5,
        }
        for index in range(8)
    ]


def test_coarse_and_refinement_layers_are_deterministic() -> None:
    coarse = regularly_spaced_layers(42, 8)
    assert coarse == (0, 6, 12, 18, 23, 29, 35, 41)
    assert refinement_layers(
        [18, 35], layer_count=42, coarse_layers=coarse, radius=2
    ) == (16, 17, 19, 20, 33, 34, 36, 37)


def test_synthetic_signal_recovers_known_causal_layer() -> None:
    summaries = [
        summarize_layer(0, _rows(0, 0.1, 0.1, 0.1), seed=42),
        summarize_layer(6, _rows(6, 0.9, 0.1, 0.0), seed=42),
        summarize_layer(12, _rows(12, 0.2, 0.1, 0.15), seed=42),
    ]

    assert rank_layers(summaries)[0].layer_index == 6
    assert select_causal_layer(summaries).layer_index == 6
    assert summaries[1].paired_advantage_ci_low > 0


def test_explicit_null_returns_no_selection() -> None:
    summaries = [
        summarize_layer(layer, _rows(layer, 0.1, 0.1, 0.1), seed=42)
        for layer in (0, 6, 12)
    ]

    assert select_causal_layer(summaries) is None
