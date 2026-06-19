import json

import pytest

from countersteer.artifacts import load_or_create_json_pair


def test_json_pair_is_write_once(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    original_manifest = {"configuration_hash": "first"}
    original_report = {"neutral_accuracy": 1.0}

    manifest, report, created = load_or_create_json_pair(
        manifest_path, original_manifest, report_path, original_report
    )
    assert created is True
    assert (manifest, report) == (original_manifest, original_report)

    manifest, report, created = load_or_create_json_pair(
        manifest_path,
        {"configuration_hash": "overwritten"},
        report_path,
        {"neutral_accuracy": 0.0},
    )
    assert created is False
    assert (manifest, report) == (original_manifest, original_report)
    assert json.loads(manifest_path.read_text()) == original_manifest


def test_json_pair_rejects_partial_state(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    with pytest.raises(RuntimeError, match="incomplete"):
        load_or_create_json_pair(
            manifest_path, {}, tmp_path / "report.json", {}
        )
