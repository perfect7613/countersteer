"""Pure selection and control math for causal activation patching."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
import random
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class PatchSelection:
    item_id: str
    selection_mode: str
    neutral_margin: float
    wrong_belief_margin: float
    margin_suppression: float


def last_non_padding_index(attention_mask: Sequence[int]) -> int:
    """Return the final semantic prompt-token index for either padding side."""

    indices = [index for index, value in enumerate(attention_mask) if int(value) == 1]
    if not indices:
        raise ValueError("attention mask contains no prompt tokens")
    return indices[-1]


def select_patch_examples(
    records: Iterable[dict[str, Any]], train_ids: Iterable[str], limit: int
) -> tuple[list[PatchSelection], int]:
    """Prefer genuine flips; explicitly label a margin-only empirical fallback."""

    train = set(train_ids)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["item_id"] in train:
            grouped[record["item_id"]][record["condition"]] = record

    discrete: list[PatchSelection] = []
    continuous: list[PatchSelection] = []
    for item_id, conditions in grouped.items():
        if not {"neutral", "wrong_belief"}.issubset(conditions):
            continue
        neutral, wrong = conditions["neutral"], conditions["wrong_belief"]
        if neutral["predicted_label"] != neutral["correct_label"]:
            continue
        suppression = float(neutral["correct_logit_margin"]) - float(
            wrong["correct_logit_margin"]
        )
        selection = PatchSelection(
            item_id=item_id,
            selection_mode="behavioral_capitulation",
            neutral_margin=float(neutral["correct_logit_margin"]),
            wrong_belief_margin=float(wrong["correct_logit_margin"]),
            margin_suppression=suppression,
        )
        if wrong["predicted_label"] == wrong["wrong_label"]:
            discrete.append(selection)
        elif suppression > 0:
            continuous.append(
                PatchSelection(
                    **{
                        **selection.__dict__,
                        "selection_mode": "margin_pressure_fallback",
                    }
                )
            )
    discrete.sort(key=lambda row: (-row.margin_suppression, row.item_id))
    continuous.sort(key=lambda row: (-row.margin_suppression, row.item_id))
    chosen = discrete[:limit] if discrete else continuous[:limit]
    return chosen, len(discrete)


def vector_subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return [float(a) - float(b) for a, b in zip(left, right)]


def vector_norm(vector: Sequence[float]) -> float:
    return sqrt(sum(float(value) ** 2 for value in vector))


def norm_matched_random_replacement(
    wrong: Sequence[float], neutral: Sequence[float], seed: int
) -> list[float]:
    """Return wrong + random direction with norm ||neutral - wrong||."""

    delta = vector_subtract(neutral, wrong)
    target_norm = vector_norm(delta)
    generator = random.Random(seed)
    direction = [generator.gauss(0.0, 1.0) for _ in delta]
    direction_norm = vector_norm(direction)
    if direction_norm == 0:
        raise ValueError("sampled zero random direction")
    scale = target_norm / direction_norm
    return [float(base) + scale * value for base, value in zip(wrong, direction)]


def apply_replacement(
    original: Sequence[float], replacement: Sequence[float], enabled: bool
) -> list[float]:
    if len(original) != len(replacement):
        raise ValueError("vectors must have equal length")
    return list(replacement if enabled else original)


def restoration_fraction(
    *, neutral_margin: float, wrong_margin: float, patched_margin: float
) -> float | None:
    denominator = neutral_margin - wrong_margin
    if denominator <= 0:
        return None
    return (patched_margin - wrong_margin) / denominator
