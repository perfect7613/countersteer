"""Pure provenance and cost helpers shared by Modal and fixture-only tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class SmokeConfig:
    model_id: str
    model_revision: str
    system_prompt: str
    prompt: str
    seed: int
    max_new_tokens: int
    enable_thinking: bool
    gpu: str


@dataclass(frozen=True)
class CostRates:
    """Published per-second resource rates used for a transparent estimate."""

    gpu_per_second: float
    cpu_core_per_second: float
    memory_gib_per_second: float


def canonical_config_hash(config: SmokeConfig) -> str:
    """Hash a stable JSON representation, independent of dictionary ordering."""

    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def estimate_resource_cost_usd(
    *,
    compute_seconds: float,
    cpu_cores: float,
    memory_gib: float,
    rates: CostRates,
) -> float:
    """Estimate GPU + CPU + memory cost from measured remote compute time."""

    if compute_seconds < 0 or cpu_cores < 0 or memory_gib < 0:
        raise ValueError("resource measurements must be non-negative")
    per_second = (
        rates.gpu_per_second
        + cpu_cores * rates.cpu_core_per_second
        + memory_gib * rates.memory_gib_per_second
    )
    return round(compute_seconds * per_second, 6)


def build_manifest(
    *,
    config: SmokeConfig,
    hardware: Mapping[str, Any],
    timing: Mapping[str, float],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the public provenance manifest. Credentials are never accepted."""

    return {
        "schema_version": 1,
        "configuration_hash": canonical_config_hash(config),
        "model": {"id": config.model_id, "revision": config.model_revision},
        "seed": config.seed,
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": False,
            "enable_thinking": config.enable_thinking,
        },
        "hardware": dict(hardware),
        "timing_seconds": dict(timing),
        "cost": dict(cost),
    }
