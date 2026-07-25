"""SAGE.10e per-level route memory and shortening tests."""

from __future__ import annotations

from theory.online_level_route_memory import OnlineLevelRouteMemory


def _observe(
    memory: OnlineLevelRouteMemory,
    *,
    before: str,
    after: str,
    action: str,
    level: bool = False,
) -> dict:
    return memory.observe_transition(
        state_signature_before=before,
        state_signature_after=after,
        action_name=action,
        action_data=None,
        level_progressed=level,
        won=False,
        game_over=False,
    )


def test_completed_level_compiles_and_replays_exact_route():
    memory = OnlineLevelRouteMemory(enable_shortening=False)
    _observe(memory, before="s0", after="s1", action="ACTION1")
    _observe(
        memory,
        before="s1",
        after="level-1",
        action="ACTION2",
        level=True,
    )
    summary = memory.summary()
    assert summary["observed_routes"] == 1
    assert summary["confirmed_routes"] == 1

    memory.start_branch()
    first = memory.select(
        state_signature="s0",
        available_actions=("ACTION1", "ACTION2"),
    )
    assert first is not None
    assert first.action.action_name == "ACTION1"
    _observe(memory, before="s0", after="s1", action="ACTION1")
    second = memory.select(
        state_signature="s1",
        available_actions=("ACTION1", "ACTION2"),
    )
    assert second is not None
    assert second.action.action_name == "ACTION2"
    outcome = _observe(
        memory,
        before="s1",
        after="level-1",
        action="ACTION2",
        level=True,
    )

    assert outcome["route_confirmed"] is True
    assert memory.summary()["route_replay_actions"] == 2


def test_shorter_route_remains_candidate_until_terminal_replay():
    memory = OnlineLevelRouteMemory(enable_shortening=True)
    _observe(memory, before="s0", after="s1", action="ACTION1")
    _observe(memory, before="s1", after="s2", action="ACTION2")
    _observe(
        memory,
        before="s2",
        after="level-1",
        action="ACTION3",
        level=True,
    )
    candidates = [
        route for route in memory.routes() if route.shortening_candidate
    ]
    assert len(candidates) == 1
    assert candidates[0].candidate_only is True
    assert candidates[0].confirmations == 0

    memory.start_branch()
    actions = []
    state = "s0"
    while True:
        selection = memory.select(
            state_signature=state,
            available_actions=("ACTION1", "ACTION2", "ACTION3"),
        )
        if selection is None:
            break
        actions.append(selection.action.action_name)
        next_state = f"candidate-{len(actions)}"
        terminal = len(actions) == selection.action_limit
        outcome = _observe(
            memory,
            before=state,
            after="level-1" if terminal else next_state,
            action=selection.action.action_name,
            level=terminal,
        )
        state = next_state
        if terminal:
            assert outcome["route_confirmed"] is True
            break

    assert len(actions) < 3
    summary = memory.summary()
    assert summary["shortening_confirmations"] == 1
    assert summary["shortening_actions_saved"] >= 1


def test_failed_shortening_candidate_is_never_promoted():
    memory = OnlineLevelRouteMemory(enable_shortening=True)
    _observe(memory, before="s0", after="s1", action="ACTION1")
    _observe(
        memory,
        before="s1",
        after="level-1",
        action="ACTION2",
        level=True,
    )
    memory.start_branch()
    selection = memory.select(
        state_signature="s0",
        available_actions=("ACTION1", "ACTION2"),
    )
    assert selection is not None
    assert selection.shortening_candidate is True
    outcome = memory.observe_transition(
        state_signature_before="s0",
        state_signature_after="still-s0",
        action_name=selection.action.action_name,
        action_data=None,
        level_progressed=False,
        won=False,
        game_over=False,
    )

    assert outcome["route_refuted"] is True
    candidate = next(
        route for route in memory.routes()
        if route.route_id == selection.route_id
    )
    assert candidate.confirmations == 0
    assert candidate.status == "refuted"
