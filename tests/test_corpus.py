from collections import Counter
from dataclasses import asdict
import json

import pytest

from countersteer.corpus import (
    SourceItem,
    load_english_bundle,
    render_item,
    validate_source_items,
)


EXPECTED_DATASET_HASH = "b5b2366192f9e168bb5c74db884eba03cbab2eab17bd9d87d9c3ee45a72d970c"
EXPECTED_SPLIT_HASH = "a72800224e0ecb6855ca16b44b0151a94909d2eb7c52a79424cc9b59ff0d8fcc"


def test_frozen_corpus_counts_balance_and_hashes() -> None:
    bundle = load_english_bundle()
    neutral = [prompt for prompt in bundle.prompts if prompt.condition == "neutral"]

    assert (len(bundle.items), len(bundle.prompts)) == (60, 180)
    assert Counter(prompt.correct_label for prompt in neutral) == {"A": 30, "B": 30}
    assert bundle.dataset_hash == EXPECTED_DATASET_HASH
    assert bundle.split_hash == EXPECTED_SPLIT_HASH


def test_split_is_disjoint_complete_and_category_balanced() -> None:
    bundle = load_english_bundle()
    train, test = set(bundle.train_ids), set(bundle.test_ids)

    assert len(train) == 40
    assert len(test) == 20
    assert train.isdisjoint(test)
    assert train | test == {item.id for item in bundle.items}
    assert Counter(item.category for item in bundle.items if item.id in train) == {
        "mathematics": 20,
        "factual_knowledge": 20,
    }
    assert Counter(item.category for item in bundle.items if item.id in test) == {
        "mathematics": 10,
        "factual_knowledge": 10,
    }


def test_rendering_is_deterministic_and_only_belief_line_changes() -> None:
    item = SourceItem("en-001", "mathematics", "What is 1 + 1?", "2", "3")
    first = render_item(item)
    second = render_item(item)
    lines = [prompt.prompt.splitlines() for prompt in first]

    assert first == second
    assert [asdict(prompt) for prompt in first] == [asdict(prompt) for prompt in second]
    assert all(line_set[:3] == lines[0][:3] for line_set in lines)
    assert all(line_set[4:] == lines[0][4:] for line_set in lines)
    assert len({line_set[3] for line_set in lines}) == 3


def test_schema_rejects_duplicate_ids_and_identical_answers() -> None:
    valid = {
        "id": "en-001",
        "category": "mathematics",
        "question": "What is 1 + 1?",
        "correct_answer": "2",
        "incorrect_answer": "3",
    }
    with pytest.raises(ValueError, match="duplicate item id"):
        validate_source_items([valid, valid])
    with pytest.raises(ValueError, match="identical answer choices"):
        validate_source_items([{**valid, "incorrect_answer": "2"}])
    with pytest.raises(ValueError, match="exactly"):
        validate_source_items([{**valid, "unexpected": True}])


def test_source_json_is_canonical_data_not_generated_at_test_time() -> None:
    bundle = load_english_bundle()
    payload = bundle.remote_payload()

    assert json.loads(json.dumps(payload)) == payload

