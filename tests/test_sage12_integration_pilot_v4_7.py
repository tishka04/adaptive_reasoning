from __future__ import annotations

import json

import numpy as np
import pytest

from theory.sage12.compiler import (
    SLOT_EFFECTS,
    HypothesisCompiler,
    SemanticActionSlot,
    SlotAnnotation,
)
from theory.sage12.controller import Sage12Config, SemanticPlanningController
from theory.sage12.energy import PairwiseTrajectoryEBM
from theory.sage12.integration_pilot import load_complete_roots
from theory.sage12.integration_pilot_v4_7 import (
    _identity_probe,
    _masked_binary_probabilities,
    _nodes,
    freeze_manifest,
    load_manifest,
    load_slot_examples,
    render_slot_prompt,
)


def _probabilities(value: float = 0.0) -> dict[str, float]:
    return {effect: value for effect in SLOT_EFFECTS}


def test_candidate_complete_compiler_keeps_all_zero_slots() -> None:
    slots = (
        SemanticActionSlot(
            slot_id="left",
            action_name="ACTION6",
            action_data={"x": 1, "y": 2},
            semantic_signature={"kind": "free_slot", "direction": "left"},
        ),
        SemanticActionSlot(
            slot_id="right",
            action_name="ACTION6",
            action_data={"x": 7, "y": 8},
            semantic_signature={"kind": "free_slot", "direction": "right"},
        ),
    )
    annotations = tuple(
        SlotAnnotation(slot_id=slot.slot_id, effect_probabilities=_probabilities())
        for slot in slots
    )
    result = HypothesisCompiler().compile_slots(
        slots,
        annotations=annotations,
    )
    assert len(result.options) == len(slots)
    assert not result.rejected
    assert {option.action_key for option in result.options} == {
        slot.action_key for slot in slots
    }
    assert all(not option.asserted_effects for option in result.options)
    assert result.options[0].semantic_key != result.options[1].semantic_key


def test_slot_annotations_cannot_claim_support() -> None:
    with pytest.raises(ValueError, match="support=0"):
        SlotAnnotation(
            slot_id="bad",
            effect_probabilities=_probabilities(),
            support=1,
        )


def test_slot_signature_does_not_control_exact_execution_key() -> None:
    first = SemanticActionSlot(
        slot_id="a",
        action_name="ACTION6",
        action_data={"x": 1, "y": 2},
        semantic_signature={"kind": "free_slot"},
    )
    second = SemanticActionSlot(
        slot_id="b",
        action_name="ACTION6",
        action_data={"x": 3, "y": 4},
        semantic_signature={"kind": "free_slot"},
    )
    assert first.semantic_key == second.semantic_key
    assert first.action_key != second.action_key


def test_binary_logit_mask_normalizes_only_zero_and_one() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[100.0, -4.0, 2.0, 99.0]])
    probabilities = _masked_binary_probabilities(
        logits,
        torch.tensor([1, 2]),
    )
    assert probabilities.shape == (1, 2)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert float(probabilities[0, 1]) > 0.99


def test_ebm_accepts_frozen_eight_feature_contract() -> None:
    pytest.importorskip("torch")
    model = PairwiseTrajectoryEBM(input_width=8, hidden_width=4, seed=7)
    losses = model.fit_pairs(
        [[0.0] * 8],
        [[1.0] * 8],
        epochs=2,
    )
    assert len(losses) == 2
    assert len(model.energies([[0.5] * 8])) == 1
    with pytest.raises(ValueError, match="feature width"):
        model.energies([[0.0] * 6])


def test_real_prompts_exclude_identity_and_future_outcomes() -> None:
    roots = load_complete_roots()
    examples = load_slot_examples(roots[:1])
    node = _nodes(examples)[0]
    prompt = render_slot_prompt(node)
    payload = json.loads(prompt)
    assert node[0].game_id not in prompt
    assert node[0].root_key not in prompt
    assert "labels" not in payload["slots_0_then_1"][0]
    assert "utility" not in payload["slots_0_then_1"][0]
    assert "action_data" not in payload["slots_0_then_1"][0]
    assert len(payload["effect_order"]) == len(SLOT_EFFECTS)
    assert len(payload["recent_8"]) == 8
    shuffled = render_slot_prompt(node, relation_shuffle=True)
    assert shuffled != prompt
    assert render_slot_prompt(node, relation_shuffle=True) == shuffled


def test_manifest_freezes_every_complete_node(tmp_path) -> None:
    payload = freeze_manifest(output_dir=tmp_path)
    roots = load_complete_roots()
    examples = load_slot_examples(roots)
    assert payload["complete_roots"] == len(roots)
    assert payload["complete_nodes"] == len(_nodes(examples))
    assert payload["semantic_slots"] == len(examples)
    assert payload["qwen"]["relation_shuffle_node_ids"] == [
        node[0].node_id for node in _nodes(examples) if node[0].path == ""
    ]
    assert load_manifest(tmp_path)["manifest_checksum"] == payload[
        "manifest_checksum"
    ]


def test_controller_returns_only_a_legal_first_slot() -> None:
    slots = (
        SemanticActionSlot(
            slot_id="left",
            action_name="ACTION1",
            action_data={},
            semantic_signature={"kind": "targetless"},
        ),
        SemanticActionSlot(
            slot_id="right",
            action_name="ACTION2",
            action_data={},
            semantic_signature={"kind": "targetless"},
        ),
    )
    annotations = (
        SlotAnnotation(
            slot_id="left",
            effect_probabilities={
                **_probabilities(),
                "changed": 0.2,
                "level_complete": 0.1,
            },
        ),
        SlotAnnotation(
            slot_id="right",
            effect_probabilities={
                **_probabilities(),
                "changed": 0.9,
                "level_complete": 0.8,
            },
        ),
    )
    controller = SemanticPlanningController(
        config=Sage12Config(maximum_depth=3, beam_width=8)
    )
    selected = controller.select_slot_action(
        slots=slots,
        annotations=annotations,
        initial_state=frozenset(),
    )
    assert selected is not None
    assert selected.key in {slot.action_key for slot in slots}


def test_slot_effect_arrays_are_stable() -> None:
    assert SLOT_EFFECTS == (
        "changed",
        "moved",
        "target_created",
        "target_removed",
        "target_moved",
        "level_complete",
        "game_over",
    )
    assert np.asarray(list(_probabilities().values()), dtype=float).sum() == 0.0


def test_multiclass_identity_probe_uses_supported_sparse_solver() -> None:
    examples = load_slot_examples(load_complete_roots())
    games = sorted({item.game_id for item in examples})[:3]
    subset = tuple(
        item
        for game in games
        for item in [row for row in examples if row.game_id == game][:5]
    )
    annotations = {
        item.slot.slot_id: item.annotation(source="test") for item in subset
    }
    result = _identity_probe(subset, annotations)
    assert result["majority_accuracy"] == pytest.approx(1.0 / 3.0)
    assert 0.0 <= result["structured_accuracy"] <= 1.0
    assert 0.0 <= result["structured_plus_qwen_accuracy"] <= 1.0
