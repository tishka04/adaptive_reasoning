from __future__ import annotations

import json

import numpy as np

from theory.sage12.integration_pilot import load_complete_roots
from theory.sage12.integration_pilot_v4_7 import _nodes, load_slot_examples
from theory.sage12.semantic_adapter_v4_8 import (
    PAIR_CLASSES,
    REPRESENTATIONS,
    SOURCE_TRAIN,
    SWAP_CLASS_INDEX,
    _arm_probabilities,
    _build_v43_pairs,
    _pair_class,
    _relative_click_descriptors,
    render_pair_prompt,
)


def test_four_way_pair_classes_and_swap_are_exact() -> None:
    assert PAIR_CLASSES == ("neither", "left", "right", "both")
    assert [_pair_class(a, b) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))] == [
        0,
        1,
        2,
        3,
    ]
    assert SWAP_CLASS_INDEX.tolist() == [0, 2, 1, 3]
    probabilities = np.eye(4)[None, None, :, :].reshape(4, 1, 4)
    left, right = _arm_probabilities(probabilities)
    assert left[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert right[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_click_pair_descriptors_are_translation_invariant() -> None:
    first = _relative_click_descriptors({"x": 2, "y": 3}, {"x": 10, "y": 3})
    shifted = _relative_click_descriptors(
        {"x": 22, "y": 33}, {"x": 30, "y": 33}
    )
    assert first == shifted
    assert first[0]["relation_to_other"] == "right_aligned"
    assert first[1]["relation_to_other"] == "left_aligned"


def test_v43_pair_prompts_exclude_identity_coordinates_and_labels() -> None:
    roots = load_complete_roots()[:1]
    records, examples = _build_v43_pairs(roots)
    assert records
    assert len(records) == len(_nodes(examples))
    record = records[0]
    for representation in REPRESENTATIONS:
        prompt = render_pair_prompt(record, representation=representation)
        payload = json.loads(prompt)
        assert record["game_id"] not in prompt
        assert record["audit"]["node_id"] not in prompt
        assert "class_targets" not in prompt
        assert "label_mask" not in prompt
        assert "action_data" not in prompt
        assert '"x"' not in prompt and '"y"' not in prompt
        assert set(payload) == {"task", "common", "left", "right"}


def test_pair_prompt_swap_only_exchanges_interventions() -> None:
    records, _ = _build_v43_pairs(load_complete_roots()[:1])
    original = json.loads(
        render_pair_prompt(records[0], representation="minimal")
    )
    swapped = json.loads(
        render_pair_prompt(
            records[0], representation="minimal", swapped=True
        )
    )
    assert original["left"] == swapped["right"]
    assert original["right"] == swapped["left"]
    assert original["common"] == swapped["common"]


def test_source_registry_remains_the_eleven_training_games() -> None:
    assert len(SOURCE_TRAIN) == 11
    assert {"re86", "ls20", "sc25"}.isdisjoint(SOURCE_TRAIN)
