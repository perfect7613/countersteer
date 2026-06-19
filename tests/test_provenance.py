from dataclasses import replace

import pytest

from countersteer.provenance import (
    CostRates,
    SmokeConfig,
    build_manifest,
    canonical_config_hash,
    estimate_resource_cost_usd,
)


@pytest.fixture
def config() -> SmokeConfig:
    return SmokeConfig(
        model_id="google/gemma-4-E4B-it",
        model_revision="fee6332c1abaafb77f6f9624236c63aa2f1d0187",
        system_prompt="fixture system prompt",
        prompt="fixture prompt",
        seed=42,
        max_new_tokens=8,
        enable_thinking=False,
        gpu="L4",
    )


def test_configuration_hash_is_stable_and_sensitive(config: SmokeConfig) -> None:
    assert canonical_config_hash(config) == canonical_config_hash(config)
    assert canonical_config_hash(config) != canonical_config_hash(
        replace(config, seed=43)
    )


def test_cost_uses_measured_seconds_and_all_resource_rates() -> None:
    cost = estimate_resource_cost_usd(
        compute_seconds=10,
        cpu_cores=2,
        memory_gib=32,
        rates=CostRates(0.1, 0.01, 0.001),
    )
    assert cost == 1.52


def test_negative_resource_measurement_is_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_resource_cost_usd(
            compute_seconds=-1,
            cpu_cores=2,
            memory_gib=32,
            rates=CostRates(0.1, 0.01, 0.001),
        )


def test_manifest_has_required_provenance_and_no_credential_field(
    config: SmokeConfig,
) -> None:
    manifest = build_manifest(
        config=config,
        hardware={"modal_gpu_request": "L4"},
        timing={"measured_compute": 3.5},
        cost={"estimated_resource_cost_usd": 0.001},
    )

    assert manifest["model"]["revision"] == config.model_revision
    assert manifest["seed"] == 42
    assert manifest["configuration_hash"] == canonical_config_hash(config)
    assert not _contains_sensitive_key(manifest)


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key.lower() in {"token", "secret", "password", "api_key"}
            or _contains_sensitive_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False
