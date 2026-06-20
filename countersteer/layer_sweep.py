"""Pure coarse-to-fine layer selection and paired uncertainty math."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
import random
from statistics import fmean
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class LayerSummary:
    layer_index: int
    examples: int
    matched_mean_restoration: float
    unrelated_mean_restoration: float
    random_mean_restoration: float
    matched_answer_restoration_rate: float
    unrelated_answer_restoration_rate: float
    random_answer_restoration_rate: float
    mean_paired_advantage: float
    paired_advantage_ci_low: float
    paired_advantage_ci_high: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def regularly_spaced_layers(layer_count: int, count: int) -> tuple[int, ...]:
    """Return deterministic endpoints-inclusive coarse layer indices."""

    if layer_count < 2:
        raise ValueError("layer_count must be at least two")
    if count < 2 or count > layer_count:
        raise ValueError("count must be between two and layer_count")
    last = layer_count - 1
    return tuple(sorted({round(index * last / (count - 1)) for index in range(count)}))


def refinement_layers(
    coarse_candidates: Sequence[int],
    *,
    layer_count: int,
    coarse_layers: Iterable[int],
    radius: int = 2,
) -> tuple[int, ...]:
    """Return unswept neighbors around the strongest coarse candidates."""

    if radius < 1:
        raise ValueError("radius must be positive")
    coarse = set(coarse_layers)
    refined: set[int] = set()
    for candidate in coarse_candidates:
        lower = max(0, candidate - radius)
        upper = min(layer_count - 1, candidate + radius)
        refined.update(range(lower, upper + 1))
    return tuple(sorted(refined - coarse))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_interval(
    values: Sequence[float], *, seed: int, replicates: int = 2000
) -> tuple[float, float]:
    """Percentile interval for a paired per-example statistic."""

    if not values:
        raise ValueError("paired bootstrap requires at least one value")
    if replicates < 100:
        raise ValueError("paired bootstrap requires at least 100 replicates")
    generator = random.Random(seed)
    size = len(values)
    means = [
        fmean(values[generator.randrange(size)] for _ in range(size))
        for _ in range(replicates)
    ]
    return _quantile(means, 0.025), _quantile(means, 0.975)


def summarize_layer(
    layer_index: int,
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int = 2000,
) -> LayerSummary:
    """Summarize matched restoration against paired conservative controls."""

    if not rows:
        raise ValueError("layer summary requires at least one example")
    metrics = ("matched", "unrelated", "random")
    restoration: dict[str, list[float]] = {name: [] for name in metrics}
    answers: dict[str, list[float]] = {name: [] for name in metrics}
    for row in rows:
        if int(row["layer_index"]) != layer_index:
            raise ValueError("row layer does not match requested layer")
        for name in metrics:
            value = float(row[f"{name}_restoration_fraction"])
            if not isfinite(value):
                raise ValueError("restoration fractions must be finite")
            restoration[name].append(value)
            answers[name].append(float(bool(row[f"{name}_answer_restored"])))

    advantages = [
        matched - max(unrelated, random_control)
        for matched, unrelated, random_control in zip(
            restoration["matched"], restoration["unrelated"], restoration["random"]
        )
    ]
    low, high = paired_bootstrap_interval(
        advantages, seed=seed + layer_index, replicates=bootstrap_replicates
    )
    return LayerSummary(
        layer_index=layer_index,
        examples=len(rows),
        matched_mean_restoration=fmean(restoration["matched"]),
        unrelated_mean_restoration=fmean(restoration["unrelated"]),
        random_mean_restoration=fmean(restoration["random"]),
        matched_answer_restoration_rate=fmean(answers["matched"]),
        unrelated_answer_restoration_rate=fmean(answers["unrelated"]),
        random_answer_restoration_rate=fmean(answers["random"]),
        mean_paired_advantage=fmean(advantages),
        paired_advantage_ci_low=low,
        paired_advantage_ci_high=high,
    )


def rank_layers(summaries: Iterable[LayerSummary]) -> list[LayerSummary]:
    return sorted(
        summaries,
        key=lambda row: (
            -row.mean_paired_advantage,
            -row.matched_answer_restoration_rate,
            -row.matched_mean_restoration,
            row.layer_index,
        ),
    )


def select_causal_layer(
    summaries: Iterable[LayerSummary], *, min_paired_advantage: float = 0.05
) -> LayerSummary | None:
    """Select a reproducible candidate only when matched patching beats controls."""

    for row in rank_layers(summaries):
        strongest_control_mean = max(
            row.unrelated_mean_restoration, row.random_mean_restoration
        )
        strongest_control_rate = max(
            row.unrelated_answer_restoration_rate,
            row.random_answer_restoration_rate,
        )
        if (
            row.mean_paired_advantage >= min_paired_advantage
            and row.matched_mean_restoration > strongest_control_mean
            and row.matched_answer_restoration_rate > strongest_control_rate
        ):
            return row
    return None
