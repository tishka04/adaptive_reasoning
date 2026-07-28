from __future__ import annotations

import json

from theory.sage12.compiler import CompiledSemanticOption
from theory.sage12.integration_pilot import (
    DEFAULT_OUTPUT_DIR,
    load_complete_roots,
    load_manifest,
    repair_qwen_hypotheses,
    step_utility,
)
from theory.sage12.world_model import SemanticWorldModel


def _option(*, action_data: dict[str, int]) -> CompiledSemanticOption:
    return CompiledSemanticOption(
        option_id="option",
        hypothesis_id="hypothesis",
        action_name="ACTION6",
        action_data=action_data,
        bindings={},
        preconditions=(),
        asserted_effects=("level_complete|-|-|",),
        retracted_effects=(),
        confidence=0.0,
        source="test",
    )


def test_qwen_repair_accepts_only_emitted_legal_action_and_effect() -> None:
    raw = """```json
[
  {
    "action_id": "ACTION6",
    "hypotheses": [
      {"action_name": "assert", "effect": "level_complete"},
      {"action_name": "assert", "effect": "not_a_predicate"}
    ]
  }
]
```"""

    repaired = repair_qwen_hypotheses(
        raw,
        legal_actions=("ACTION3", "ACTION6"),
    )

    assert len(repaired) == 1
    assert repaired[0].action_name == "ACTION6"
    assert repaired[0].effects[0].predicate.name == "level_complete"
    assert repaired[0].support == 0


def test_qwen_repair_does_not_invent_a_legal_action() -> None:
    raw = json.dumps(
        [
            {
                "action_id": "ACTION7",
                "hypotheses": [
                    {"action_name": "ACTION7", "effect": "changed"}
                ],
            }
        ]
    )

    assert not repair_qwen_hypotheses(
        raw,
        legal_actions=("ACTION1", "ACTION2"),
    )


def test_name_keyed_world_model_transfers_across_grounded_arguments() -> None:
    trained = _option(action_data={"x": 1, "y": 2})
    unseen_argument = _option(action_data={"x": 40, "y": 50})
    name_model = SemanticWorldModel(action_key_mode="name")
    grounded_model = SemanticWorldModel(action_key_mode="grounded")
    for _ in range(8):
        name_model.observe(trained, {"level_complete|-|-|"})
        grounded_model.observe(trained, {"level_complete|-|-|"})

    assert name_model.effect_probability(
        unseen_argument, "level_complete|-|-|"
    ) > grounded_model.effect_probability(
        unseen_argument, "level_complete|-|-|"
    )
    assert name_model.summary()["action_key_mode"] == "name"


def test_frozen_manifest_binds_complete_source_only_trees() -> None:
    manifest = load_manifest(DEFAULT_OUTPUT_DIR)
    roots = load_complete_roots()

    assert manifest["source_only"] is True
    assert manifest["holdout_opened"] is False
    assert manifest["authority_promotion_allowed"] is False
    assert len(roots) == manifest["complete_roots"] == 340
    assert len(manifest["qwen_sample"]["root_keys"]) == 44
    assert set(manifest["qwen_sample"]["root_keys"]).issubset(
        {root.root_key for root in roots}
    )


def test_frozen_step_utility_rewards_productive_over_noop() -> None:
    trace_sets = (
        [
            arm.trace
            for pair in root.tree.values()
            for arm in (pair.left, pair.right)
        ]
        for root in load_complete_roots()
    )
    traces = next(
        items
        for items in trace_sets
        if any(trace.effects.noop for trace in items)
        and any(not trace.effects.noop for trace in items)
    )
    productive = next(trace for trace in traces if not trace.effects.noop)
    noop = next(trace for trace in traces if trace.effects.noop)

    assert step_utility(productive) > 0.0
    assert step_utility(productive) > step_utility(noop)
