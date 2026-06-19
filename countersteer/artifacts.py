"""Small write-once helpers for immutable experiment evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_or_create_json_pair(
    first_path: Path,
    first_value: dict[str, Any],
    second_path: Path,
    second_value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Load an existing pair or exclusively create both without overwriting."""

    first_exists, second_exists = first_path.exists(), second_path.exists()
    if first_exists != second_exists:
        raise RuntimeError("write-once artifact pair is incomplete")
    if first_exists:
        return (
            json.loads(first_path.read_text("utf-8")),
            json.loads(second_path.read_text("utf-8")),
            False,
        )
    with first_path.open("x", encoding="utf-8") as file:
        json.dump(first_value, file, indent=2)
        file.write("\n")
    with second_path.open("x", encoding="utf-8") as file:
        json.dump(second_value, file, indent=2)
        file.write("\n")
    return first_value, second_value, True
