"""Tests for SAGE.9f terminal-negative frontier exploration."""

from theory.online_terminal_frontier import (
    OnlineTerminalFrontierExplorer,
    SuccessfulContinuation,
    TerminalFrontierAction,
)


def _action(name: str, **data: int) -> TerminalFrontierAction:
    return TerminalFrontierAction.from_parts(name, data)


def test_nonterminal_suffix_is_bounded_and_receives_no_credit():
    explorer = OnlineTerminalFrontierExplorer(max_suffix_actions=2)
    frontier_id = explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
        context_signature="branch-a",
    )

    first = explorer.select(
        state_signature="frontier-state",
        available_actions=["ACTION1"],
        proposed_actions=[_action("ACTION1")],
    )
    assert first is not None
    first_outcome = explorer.observe_transition(
        state_signature_before="frontier-state",
        state_signature_after="middle-state",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )
    second = explorer.select(
        state_signature="middle-state",
        available_actions=["ACTION2"],
        proposed_actions=[_action("ACTION2")],
    )
    assert second is not None
    second_outcome = explorer.observe_transition(
        state_signature_before="middle-state",
        state_signature_after="local-progress-only",
        action_name="ACTION2",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )

    assert first_outcome["credited"] is False
    assert second_outcome["credited"] is False
    summary = explorer.summary()
    assert summary["frontiers_captured"] == 1
    assert summary["nonterminal_suffixes"] == 1
    assert summary["terminal_credits"] == 0
    assert summary["successful_continuations"] == 0
    assert summary["active_frontier_id"] == ""
    assert explorer.frontiers()[0].frontier_id == frontier_id


def test_only_level_change_credits_suffix_and_enables_same_frontier_replay():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=3,
        max_trials_per_frontier=3,
    )
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    first = explorer.select(
        state_signature="frontier-state",
        available_actions=["ACTION1"],
        proposed_actions=[_action("ACTION1")],
    )
    assert first is not None
    explorer.observe_transition(
        state_signature_before="frontier-state",
        state_signature_after="middle-state",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )
    second = explorer.select(
        state_signature="middle-state",
        available_actions=["ACTION2"],
        proposed_actions=[_action("ACTION2")],
    )
    assert second is not None
    credited = explorer.observe_transition(
        state_signature_before="middle-state",
        state_signature_after="next-level",
        action_name="ACTION2",
        action_data={},
        level_progressed=True,
        won=False,
        game_over=False,
    )

    assert credited["credited"] is True
    assert explorer.summary()["terminal_credits"] == 1
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
        context_signature="branch-b",
    )
    replay_first = explorer.select(
        state_signature="frontier-state",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert replay_first is not None
    assert replay_first.action.action_name == "ACTION1"
    assert replay_first.replaying_successful_continuation is True
    explorer.observe_transition(
        state_signature_before="frontier-state",
        state_signature_after="middle-state",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )
    replay_second = explorer.select(
        state_signature="middle-state",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert replay_second is not None
    assert replay_second.action.action_name == "ACTION2"
    assert replay_second.replaying_successful_continuation is True
    confirmed = explorer.observe_transition(
        state_signature_before="middle-state",
        state_signature_after="next-level",
        action_name="ACTION2",
        action_data={},
        level_progressed=True,
        won=False,
        game_over=False,
    )

    assert confirmed["credited"] is True
    summary = explorer.summary()
    assert summary["terminal_credits"] == 2
    assert summary["successful_continuations"] == 1
    assert summary["successful_replays"] == 1
    record = summary["records"][0]
    assert record["successful_continuations"][0]["confirmations"] == 2


def test_game_over_suffix_is_unsafe_not_successful():
    explorer = OnlineTerminalFrontierExplorer(max_suffix_actions=3)
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    selected = explorer.select(
        state_signature="frontier-state",
        available_actions=["ACTION3"],
        proposed_actions=[_action("ACTION3")],
    )
    assert selected is not None
    outcome = explorer.observe_transition(
        state_signature_before="frontier-state",
        state_signature_after="dead-state",
        action_name="ACTION3",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=True,
    )

    assert outcome["credited"] is False
    summary = explorer.summary()
    assert summary["unsafe_suffixes"] == 1
    assert summary["terminal_credits"] == 0
    assert summary["successful_continuations"] == 0


def test_guarded_replay_action_is_not_counted_as_exact_confirmation():
    explorer = OnlineTerminalFrontierExplorer(max_suffix_actions=2)
    frontier_id = explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    frontier = explorer.frontiers()[0]
    credited_action = _action("ACTION1")
    frontier.successful_continuations[(credited_action.signature,)] = (
        SuccessfulContinuation(
            actions=(credited_action,),
            state_signatures=("frontier-state", "next-level"),
        )
    )
    explorer.start_branch()
    assert explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    ) == frontier_id
    selected = explorer.select(
        state_signature="frontier-state",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert selected is not None
    assert selected.replaying_successful_continuation is True

    outcome = explorer.observe_transition(
        state_signature_before="frontier-state",
        state_signature_after="next-level",
        action_name="ACTION2",
        action_data={},
        level_progressed=True,
        won=False,
        game_over=False,
    )

    assert outcome["credited"] is True
    assert outcome["replaying_successful_continuation"] is False
    summary = explorer.summary()
    assert summary["replay_divergences"] == 1
    assert summary["successful_replays"] == 0
