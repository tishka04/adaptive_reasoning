from __future__ import annotations

from theory.sage_t.cross_game_transfer_v10_3_12c import (
    CrossGameFactorProgram,
    CrossGameFactorRegistry,
    GroundingDecision,
)
from theory.sage_t.executor_correspondence_v10_3_12d import (
    ARMS,
    PathExecutorController,
    compile_executor_registry,
)
from theory.sage_t.progress_witness_v10 import GroundedAction


def _parent_payload() -> dict:
    registry = CrossGameFactorRegistry()
    registry.register(
        CrossGameFactorProgram(
            arm="factorized_source",
            context="path_context",
            operator="parameterized_apply",
            role_binding="salient_end_prior_with_causal_verification",
            transition="successor_toward_goal_end",
            termination="stop_on_progress_or_ambiguity",
            safety_horizon=16,
            source_kind="test_parent",
        )
    )
    return registry.snapshot()


def _path() -> tuple[GroundedAction, ...]:
    return tuple(
        GroundedAction("ACTION6", (("x", index), ("y", 20 - index)))
        for index in range(1, 11)
    )


def _controller(arm: str, calls: list[int] | None = None) -> PathExecutorController:
    path = _path()

    def builder(state, candidates):
        del state, candidates
        if calls is not None:
            calls.append(1)
        return "path_context", path

    parent = CrossGameFactorRegistry(_parent_payload())
    executors = compile_executor_registry(_parent_payload())
    return PathExecutorController(
        arm=arm,
        registry=executors,
        parent_registry=parent,
        plan_builder=builder,
    )


def test_registry_contains_four_support_zero_executor_controls() -> None:
    registry = compile_executor_registry(_parent_payload())
    snapshot = registry.snapshot()
    assert len(snapshot["programs"]) == len(ARMS) == 4
    assert snapshot["local_support_total"] == 0
    assert registry.program_for("stable_source_cursor").continuation == "option_local_cursor"
    assert registry.program_for("stateless_source_replan").continuation == (
        "recompute_and_take_first"
    )
    assert registry.program_for("stable_reverse_orientation").orientation == (
        "reverse_source_end"
    )
    assert registry.program_for("stable_cursor_hold").continuation == (
        "hold_initial_waypoint"
    )


def test_stable_cursor_builds_once_and_reacquires_full_sequence() -> None:
    calls: list[int] = []
    controller = _controller("stable_source_cursor", calls)
    path = _path()
    candidates = tuple(item.candidate for item in reversed(path))
    selected = []
    for step in range(10):
        decision = controller.choose(
            state=None,
            candidates=candidates,
            shape=(32, 32),
            step_index=step,
        )
        selected.append(dict(decision.candidate.action_data))
    summary = controller.summary()
    assert selected == [item.data for item in path]
    assert len(calls) == 1
    assert summary["plan_builds"] == 1
    assert summary["replans"] == 0
    assert summary["reacquisitions"] == 10
    assert summary["path_plan_persisted"] is False


def test_reverse_and_cursor_hold_are_isolated_controls() -> None:
    path = _path()
    candidates = tuple(item.candidate for item in path)
    reverse = _controller("stable_reverse_orientation")
    reverse_selected = [
        dict(
            reverse.choose(
                state=None,
                candidates=candidates,
                shape=(32, 32),
                step_index=step,
            ).candidate.action_data
        )
        for step in range(10)
    ]
    hold = _controller("stable_cursor_hold")
    hold_selected = [
        dict(
            hold.choose(
                state=None,
                candidates=candidates,
                shape=(32, 32),
                step_index=step,
            ).candidate.action_data
        )
        for step in range(4)
    ]
    assert reverse_selected == [item.data for item in reversed(path)]
    assert hold_selected == [path[0].data] * 4


def test_stateless_control_reinvokes_grounder_on_every_decision() -> None:
    path = _path()
    candidates = tuple(item.candidate for item in path)
    calls: list[int] = []

    def grounder(*args, **kwargs):
        del args
        legal = tuple(kwargs["candidates"])
        index = len(calls) % 2
        calls.append(1)
        return GroundingDecision(
            candidate=legal[index],
            context="path_context",
            reason="test_stateless",
            inspections=len(legal),
            program_hash="test",
            ablated_factor=None,
        )

    parent = CrossGameFactorRegistry(_parent_payload())
    controller = PathExecutorController(
        arm="stateless_source_replan",
        registry=compile_executor_registry(_parent_payload()),
        parent_registry=parent,
        plan_builder=lambda state, legal: ("path_context", path),
        stateless_grounder=grounder,
    )
    selected = [
        controller.choose(
            state=None,
            candidates=candidates,
            shape=(32, 32),
            step_index=step,
        ).candidate
        for step in range(2)
    ]
    assert len(calls) == 2
    assert selected[0] != selected[1]
    assert controller.summary()["replans"] == 2


def test_missing_waypoint_and_non_path_context_abstain() -> None:
    path = _path()
    controller = _controller("stable_source_cursor")
    first = controller.choose(
        state=None,
        candidates=tuple(item.candidate for item in path),
        shape=(32, 32),
        step_index=0,
    )
    missing = controller.choose(
        state=None,
        candidates=(path[0].candidate,),
        shape=(32, 32),
        step_index=1,
    )
    parent = CrossGameFactorRegistry(_parent_payload())
    non_path = PathExecutorController(
        arm="stable_source_cursor",
        registry=compile_executor_registry(_parent_payload()),
        parent_registry=parent,
        plan_builder=lambda state, candidates: ("repeat_context", ()),
    ).choose(
        state=None,
        candidates=(path[0].candidate,),
        shape=(32, 32),
        step_index=0,
    )
    assert not first.abstained
    assert missing.abstained
    assert missing.reason == "stable_waypoint_reacquisition_miss"
    assert non_path.abstained
    assert non_path.reason == "non_path_context"
