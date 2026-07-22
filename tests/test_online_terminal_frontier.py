"""Tests for SAGE.9f/9g terminal-negative frontier exploration."""

from theory.online_terminal_frontier import (
    OnlineTerminalFrontierExplorer,
    SuccessfulContinuation,
    TerminalFrontierAction,
)


def _action(name: str, **data: int) -> TerminalFrontierAction:
    return TerminalFrontierAction.from_parts(name, data)


def _run_suffix(
    explorer: OnlineTerminalFrontierExplorer,
    *,
    state: str,
    steps: int,
    terminal_step: int = 0,
):
    selections = []
    outcomes = []
    current = state
    for index in range(steps):
        selected = explorer.select(
            state_signature=current,
            available_actions=["ACTION1"],
            proposed_actions=[_action("ACTION1")],
        )
        assert selected is not None
        selections.append(selected)
        after = f"{state}::step-{index + 1}"
        outcomes.append(
            explorer.observe_transition(
                state_signature_before=current,
                state_signature_after=after,
                action_name="ACTION1",
                action_data={},
                level_progressed=index + 1 == terminal_step,
                won=False,
                game_over=False,
            )
        )
        current = after
    return selections, outcomes


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
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    selected_after_danger = explorer.select(
        state_signature="frontier-state",
        available_actions=["ACTION1"],
        proposed_actions=[_action("ACTION1")],
    )
    assert selected_after_danger is not None
    assert selected_after_danger.action_limit == 3
    assert explorer.summary()["adaptive_horizon_extensions"] == 0


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


def test_repeated_exhausted_frontier_receives_larger_terminal_only_horizon():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        max_trials_per_frontier=3,
        max_adaptive_suffix_actions=6,
        adaptive_horizon_increment=2,
    )
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    first_selections, first_outcomes = _run_suffix(
        explorer,
        state="frontier-state",
        steps=2,
    )
    assert {item.action_limit for item in first_selections} == {2}
    assert all(outcome["credited"] is False for outcome in first_outcomes)

    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    extended_selections, extended_outcomes = _run_suffix(
        explorer,
        state="frontier-state",
        steps=3,
        terminal_step=3,
    )

    assert {item.action_limit for item in extended_selections} == {4}
    assert extended_outcomes[-1]["credited"] is True
    assert extended_outcomes[-1]["adaptive_horizon"] is True
    summary = explorer.summary()
    assert summary["adaptive_horizon_extensions"] == 1
    assert summary["adaptive_horizon_actions_granted"] == 2
    assert summary["extended_suffix_actions"] == 1
    assert summary["terminal_credits"] == 1
    record = summary["records"][0]
    assert record["horizon_history"] == [2, 4]
    assert record["longest_suffix_actions"] == 3


def test_censored_frontier_does_not_earn_adaptive_horizon():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        max_adaptive_suffix_actions=6,
        adaptive_horizon_increment=2,
    )
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    _run_suffix(explorer, state="frontier-state", steps=1)
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    selected, _ = _run_suffix(
        explorer,
        state="frontier-state",
        steps=1,
    )

    assert selected[0].action_limit == 2
    summary = explorer.summary()
    assert summary["censored_suffixes"] == 1
    assert summary["adaptive_horizon_extensions"] == 0


def test_adaptive_horizon_ablation_keeps_original_bound():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        max_trials_per_frontier=2,
        enable_adaptive_horizon=False,
        max_adaptive_suffix_actions=6,
        adaptive_horizon_increment=2,
    )
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    _run_suffix(explorer, state="frontier-state", steps=2)
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    selected, _ = _run_suffix(
        explorer,
        state="frontier-state",
        steps=2,
    )

    assert {item.action_limit for item in selected} == {2}
    summary = explorer.summary()
    assert summary["adaptive_horizon_enabled"] is False
    assert summary["adaptive_horizon_extensions"] == 0
    assert summary["extended_suffix_actions"] == 0
