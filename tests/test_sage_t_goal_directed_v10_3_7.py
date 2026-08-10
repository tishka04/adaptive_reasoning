from __future__ import annotations

from theory.sage_t import goal_directed_v10_3_7 as module
from theory.sage_t.contracts import AbstractState, ActionCandidate
from theory.sage_t.goal_directed_v10_3_2 import GoalDirectedOption, OptionStep
from theory.sage_t.goal_directed_v10_3_3 import DYNAMIC_SUCCESSOR
from theory.sage_t.goal_directed_v10_3_7 import StableFreshPathSageTController
from theory.sage_t.progress_witness_v10 import CandidateMacro, GroundedAction


def _macro() -> CandidateMacro:
    return CandidateMacro(
        schema="path_successor",
        relation="successor_toward_enclosure",
        actions=tuple(
            GroundedAction("ACTION6", (("x", index), ("y", 20 - index)))
            for index in range(1, 11)
        ),
    )


def test_fresh_initial_plan_keeps_all_ten_waypoints(monkeypatch) -> None:
    macro = _macro()
    monkeypatch.setattr(module, "chain_successor_macro", lambda *args, **kwargs: macro)
    controller = StableFreshPathSageTController(phase="preflight")
    controller._active_option = GoalDirectedOption(
        schema="path_successor",
        steps=tuple(
            OptionStep("ACTION6", binding_method=DYNAMIC_SUCCESSOR)
            for _ in range(10)
        ),
        source="test",
    )
    candidates = tuple(
        ActionCandidate(item.action_name, dict(item.data)) for item in macro.actions
    )
    state = AbstractState(entities=())

    selected = []
    for cursor in range(10):
        controller._active_cursor = cursor
        selected.append(
            dict(controller._continue_active_option(state, tuple(reversed(candidates))).action_data)
        )

    assert selected == [dict(item.data) for item in macro.actions]
    assert controller.summary()["fresh_plan_reacquisitions"] == 10
    assert controller.summary()["fresh_successor_plan_persisted"] is False


def test_mid_option_recomputation_is_not_used(monkeypatch) -> None:
    macro = _macro()
    calls = 0

    def ground(*args, **kwargs):
        nonlocal calls
        calls += 1
        return macro if calls == 1 else CandidateMacro(
            schema="path_successor",
            relation="successor_toward_enclosure",
            actions=macro.actions[:-1],
        )

    monkeypatch.setattr(module, "chain_successor_macro", ground)
    controller = StableFreshPathSageTController(phase="preflight")
    controller._active_option = GoalDirectedOption(
        schema="path_successor",
        steps=tuple(OptionStep("ACTION6", binding_method=DYNAMIC_SUCCESSOR) for _ in range(10)),
        source="test",
    )
    candidates = tuple(ActionCandidate(item.action_name, dict(item.data)) for item in macro.actions)
    for cursor in range(10):
        controller._active_cursor = cursor
        assert controller._continue_active_option(AbstractState(entities=()), candidates) is not None

    assert calls == 1


def test_plan_is_cleared_when_option_finishes(monkeypatch) -> None:
    macro = _macro()
    monkeypatch.setattr(module, "chain_successor_macro", lambda *args, **kwargs: macro)
    controller = StableFreshPathSageTController(phase="preflight")
    controller._active_option = GoalDirectedOption(
        schema="path_successor",
        steps=(OptionStep("ACTION6", binding_method=DYNAMIC_SUCCESSOR),),
        source="test",
    )
    candidates = tuple(ActionCandidate(item.action_name, dict(item.data)) for item in macro.actions)
    assert controller._continue_active_option(AbstractState(entities=()), candidates) is not None
    assert controller._fresh_successor_plan

    controller._finish_active_option(progressed=False, reason="test")

    assert controller._fresh_successor_plan == ()
