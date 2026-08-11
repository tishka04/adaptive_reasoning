from __future__ import annotations

from theory.sage_t.closed_loop_successor_v10_3_12e import (
    ARMS,
    ClosedLoopSuccessorController,
    compile_closed_loop_registry,
)
from theory.sage_t.cross_game_transfer_v10_3_12c import (
    CrossGameFactorProgram,
    CrossGameFactorRegistry,
)
from theory.sage_t.executor_correspondence_v10_3_12d import compile_executor_registry
from theory.sage_t.progress_witness_v10 import GroundedAction


def _source_payload() -> dict:
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
            source_kind="test_source",
        )
    )
    return registry.snapshot()


def _registry():
    source = _source_payload()
    parent = compile_executor_registry(source).snapshot()
    return compile_closed_loop_registry(parent, source)


def _action(index: int) -> GroundedAction:
    return GroundedAction("ACTION6", (("x", index), ("y", 20 - index)))


def test_registry_has_four_support_zero_safe_controls() -> None:
    registry = _registry()
    snapshot = registry.snapshot()
    assert len(snapshot["programs"]) == len(ARMS) == 4
    assert snapshot["local_support_total"] == 0
    assert registry.program_for("anchored_goal_dynamic_successor").goal_anchor == (
        "source_salient_endpoint_role"
    )
    assert registry.program_for("frozen_grounded_cursor").successor_selection == (
        "fixed_initial_grounded_cursor"
    )
    assert registry.program_for("goal_end_swap").goal_anchor == (
        "anti_salient_endpoint_role"
    )
    forbidden = ("action_data", "entity_id", "game_id", '"x"', '"y"')
    encoded = str(snapshot)
    assert all(token not in encoded for token in forbidden)


def test_dynamic_successor_recomputes_and_skips_visited_frontier() -> None:
    first = _action(1)
    second = _action(2)
    third = _action(3)
    paths = [(first, second), (first, third)]
    calls: list[int] = []

    def builder(state, candidates):
        del state, candidates
        path = paths[len(calls)]
        calls.append(1)
        return "path_context", path

    controller = ClosedLoopSuccessorController(
        arm="anchored_goal_dynamic_successor",
        registry=_registry(),
        path_builder=builder,
    )
    candidates = tuple(item.candidate for item in (first, second, third))
    selected = [
        controller.choose(
            state=None,
            candidates=candidates,
            shape=(32, 32),
            step_index=step,
        ).candidate
        for step in range(2)
    ]
    summary = controller.summary()
    assert [dict(item.action_data) for item in selected] == [first.data, third.data]
    assert len(calls) == 2
    assert summary["anchor_builds"] == 1
    assert summary["relation_evaluations"] == 2
    assert summary["dynamic_regrounds"] == 2
    assert summary["frontier_advances"] == 2
    assert summary["repeat_proposals_rejected"] == 1
    assert summary["visited_action_keys_persisted"] is False


def test_stateless_repeats_first_while_goal_swap_reverses() -> None:
    path = tuple(_action(index) for index in range(1, 4))
    candidates = tuple(item.candidate for item in path)

    def builder(state, legal):
        del state, legal
        return "path_context", path

    stateless = ClosedLoopSuccessorController(
        arm="stateless_goal_and_successor",
        registry=_registry(),
        path_builder=builder,
    )
    repeated = [
        stateless.choose(
            state=None,
            candidates=candidates,
            shape=(32, 32),
            step_index=step,
        ).candidate
        for step in range(2)
    ]
    swapped = ClosedLoopSuccessorController(
        arm="goal_end_swap",
        registry=_registry(),
        path_builder=builder,
    ).choose(
        state=None,
        candidates=candidates,
        shape=(32, 32),
        step_index=0,
    )
    assert [dict(item.action_data) for item in repeated] == [path[0].data] * 2
    assert dict(swapped.candidate.action_data) == path[-1].data


def test_frozen_cursor_keeps_initial_path_and_fails_closed_on_missing_waypoint() -> None:
    path = tuple(_action(index) for index in range(1, 4))
    controller = ClosedLoopSuccessorController(
        arm="frozen_grounded_cursor",
        registry=_registry(),
        path_builder=lambda state, legal: ("path_context", path),
    )
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
    assert dict(first.candidate.action_data) == path[0].data
    assert missing.abstained
    assert missing.reason == "current_relational_successor_grounding_miss"
    assert controller.summary()["relation_evaluations"] == 1


def test_non_path_context_abstains_uniformly() -> None:
    candidate = _action(1).candidate
    for arm in ARMS:
        decision = ClosedLoopSuccessorController(
            arm=arm,
            registry=_registry(),
            path_builder=lambda state, legal: ("repeat_context", ()),
        ).choose(
            state=None,
            candidates=(candidate,),
            shape=(32, 32),
            step_index=0,
        )
        assert decision.abstained
        assert decision.reason == "non_path_context"
