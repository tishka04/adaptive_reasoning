from __future__ import annotations

import json

from theory.sage_t.contracts import ActionCandidate
from theory.sage_t.contracts import AbstractState
from theory.sage_t.goal_directed_v10_3_2 import GoalDirectedOption, OptionStep
from theory.sage_t.goal_directed_v10_3_3 import DYNAMIC_SUCCESSOR
from theory.sage_t.goal_directed_v10_3_6 import (
    BALANCED_CAUSAL_BINDING,
    FunctionalGoalDirectedSageTController,
    WITNESS_BINDING,
)
from theory.sage_t.progress_witness_v10 import CandidateMacro, GroundedAction
from theory.sage_t.t10_3_6_runtime import _synthetic_binding_cycle


def test_balanced_binding_changes_only_reset_local_grounding() -> None:
    left = _synthetic_binding_cycle(offset=0)
    right = _synthetic_binding_cycle(offset=1)

    assert left["selected"][0] != right["selected"][0]
    assert len({tuple(sorted(row.items())) for row in left["selected"]}) == 1
    assert len({tuple(sorted(row.items())) for row in right["selected"]}) == 1
    assert left["option_successes"] == right["option_successes"] == 1
    assert left["safe_registry"] and right["safe_registry"]


def test_visual_cycle_is_aborted_without_progress_credit() -> None:
    result = _synthetic_binding_cycle(offset=0, cycle=True)

    assert result["causal_cycle_aborts"] >= 1
    assert result["option_successes"] == 0
    assert result["posterior_events"] == 5


def test_balanced_option_serialization_has_no_ephemeral_binding() -> None:
    candidates = (
        ActionCandidate("ACTION6", {"x": 2, "y": 3}),
        ActionCandidate("ACTION6", {"x": 9, "y": 7}),
    )
    option = FunctionalGoalDirectedSageTController._balanced_repeat_option(
        candidates,
        horizon=5,
        witness=False,
        source="test",
    )

    assert option is not None
    assert all(step.binding_method == BALANCED_CAUSAL_BINDING for step in option.steps)
    text = json.dumps(option.safe_payload, sort_keys=True)
    assert '"x"' not in text and '"y"' not in text
    assert "entity_id" not in text and "game_id" not in text


def test_witness_binding_is_a_method_not_a_historical_action() -> None:
    candidates = (
        ActionCandidate("ACTION6", {"x": 1, "y": 1}),
        ActionCandidate("ACTION6", {"x": 5, "y": 5}),
    )
    option = FunctionalGoalDirectedSageTController._balanced_repeat_option(
        candidates,
        horizon=5,
        witness=True,
        source="t10_0b_structure_fresh_regrounding",
    )

    assert option is not None
    assert len(option.steps) == 5
    assert all(step.binding_method == WITNESS_BINDING for step in option.steps)
    assert "action_data" not in json.dumps(option.safe_payload)


def test_dynamic_successor_advances_in_freshly_recomputed_chain(monkeypatch) -> None:
    candidates = tuple(
        ActionCandidate("ACTION6", {"x": value, "y": value})
        for value in (1, 2, 3)
    )
    macro = CandidateMacro(
        schema="path_successor",
        relation="successor_toward_enclosure",
        actions=tuple(
            GroundedAction("ACTION6", (("x", value), ("y", value)))
            for value in (1, 2, 3)
        ),
    )
    monkeypatch.setattr(
        "theory.sage_t.goal_directed_v10_3_6.chain_successor_macro",
        lambda *args, **kwargs: macro,
    )
    controller = FunctionalGoalDirectedSageTController(phase="preflight")
    controller._active_option = GoalDirectedOption(
        schema="path_successor",
        steps=tuple(
            OptionStep("ACTION6", binding_method=DYNAMIC_SUCCESSOR)
            for _ in range(3)
        ),
        source="test",
    )
    state = AbstractState(entities=())

    selected = []
    for cursor in range(3):
        controller._active_cursor = cursor
        selected.append(
            dict(controller._continue_active_option(state, candidates).action_data)
        )

    assert selected == [
        {"x": 1, "y": 1},
        {"x": 2, "y": 2},
        {"x": 3, "y": 3},
    ]
    assert controller.summary()["successor_advances"] == 2


def test_sequence_mode_builds_a_mixed_balanced_automaton() -> None:
    controller = FunctionalGoalDirectedSageTController(
        phase="discovery",
        prefer_mixed=True,
    )
    candidates = (
        ActionCandidate("ACTION1", {}),
        ActionCandidate("ACTION6", {"x": 1, "y": 1}),
        ActionCandidate("ACTION6", {"x": 5, "y": 5}),
    )

    option = controller._mixed_option(candidates, goal_hypotheses=())

    assert option is not None and option.mixed
    assert option.schema == "mixed_automaton"
    assert {step.action_name for step in option.steps} == {"ACTION1", "ACTION6"}
    assert all(
        step.binding_method == BALANCED_CAUSAL_BINDING
        for step in option.steps
        if step.action_name == "ACTION6"
    )
