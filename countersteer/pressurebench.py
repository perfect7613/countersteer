"""Pinned PressureBench ingestion and controlled prompt construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Iterable

from countersteer.corpus import canonical_hash


DATASET_ID = "15juneee/pressure-bench-questions-v1"
DATASET_REVISION = "2eaca3b7600928284ecf5a68552d0754fad432eb"
DATASET_LICENSE = "CC-BY-4.0"
SPLIT_VERSION = "pressurebench-hash-split-v1"
RENDER_VERSION = "pressurebench-direct-expert-pressure-v1"


@dataclass(frozen=True)
class PressurePrompt:
    item_id: str
    domain: str
    condition: str
    prompt: str
    correct_label: str
    wrong_label: str
    option_a: str
    option_b: str


@dataclass(frozen=True)
class PressureBundle:
    prompts: tuple[PressurePrompt, ...]
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    dataset_hash: str
    split_hash: str


def _clean(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"PressureBench field {field} must be a non-empty string")
    return value.strip()


def render_pressure_row(row: dict[str, Any]) -> tuple[PressurePrompt, ...]:
    required = {
        "qid",
        "option_a",
        "option_b",
        "correct_option",
        "domain",
        "question",
        "question_direct",
        "question_syco",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"PressureBench row is missing fields: {sorted(missing)}")
    item_id = _clean(row["qid"], "qid")
    option_a = _clean(row["option_a"], "option_a")
    option_b = _clean(row["option_b"], "option_b")
    correct_label = _clean(row["correct_option"], "correct_option").upper()
    if correct_label not in {"A", "B"}:
        raise ValueError(f"{item_id} has invalid correct option {correct_label}")
    wrong_label = "B" if correct_label == "A" else "A"
    correct_answer = option_a if correct_label == "A" else option_b
    wrong_answer = option_b if correct_label == "A" else option_a
    direct = _clean(row.get("question_direct") or row["question"], "question")
    wrong_pressure = _clean(row["question_syco"], "question_syco")
    if wrong_answer not in wrong_pressure:
        raise ValueError(f"{item_id} pressure prompt does not contain the wrong answer")
    correct_pressure = wrong_pressure.replace(wrong_answer, correct_answer, 1)
    options = f"\n\nA. {option_a}\nB. {option_b}\nReply with only A or B."
    common = {
        "item_id": item_id,
        "domain": _clean(row["domain"], "domain"),
        "correct_label": correct_label,
        "wrong_label": wrong_label,
        "option_a": option_a,
        "option_b": option_b,
    }
    return (
        PressurePrompt(condition="neutral", prompt=direct + options, **common),
        PressurePrompt(
            condition="correct_belief", prompt=correct_pressure + options, **common
        ),
        PressurePrompt(
            condition="wrong_belief", prompt=wrong_pressure + options, **common
        ),
    )


def build_pressure_bundle(rows: Iterable[dict[str, Any]]) -> PressureBundle:
    prompts: list[PressurePrompt] = []
    seen: set[str] = set()
    for row in rows:
        rendered = render_pressure_row(row)
        if rendered[0].item_id in seen:
            raise ValueError(f"duplicate PressureBench qid: {rendered[0].item_id}")
        seen.add(rendered[0].item_id)
        prompts.extend(rendered)
    if not prompts:
        raise ValueError("PressureBench snapshot is empty")
    condition_order = {"neutral": 0, "correct_belief": 1, "wrong_belief": 2}
    prompts.sort(key=lambda prompt: (prompt.item_id, condition_order[prompt.condition]))

    item_ids = sorted(seen)
    ranked = sorted(
        item_ids,
        key=lambda item_id: sha256(
            f"{SPLIT_VERSION}:{item_id}".encode("utf-8")
        ).hexdigest(),
    )
    train_count = int(len(ranked) * 0.7)
    train_ids = tuple(sorted(ranked[:train_count]))
    test_ids = tuple(sorted(ranked[train_count:]))
    prompt_payload = [asdict(prompt) for prompt in prompts]
    dataset_hash = canonical_hash(
        {
            "source": DATASET_ID,
            "revision": DATASET_REVISION,
            "render_version": RENDER_VERSION,
            "prompts": prompt_payload,
        }
    )
    split_hash = canonical_hash(
        {
            "version": SPLIT_VERSION,
            "train": train_ids,
            "test": test_ids,
        }
    )
    return PressureBundle(
        prompts=tuple(prompts),
        train_ids=train_ids,
        test_ids=test_ids,
        dataset_hash=dataset_hash,
        split_hash=split_hash,
    )
