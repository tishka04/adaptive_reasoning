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


def _nominate_delayed_terminal_candidate(
    explorer: OnlineTerminalFrontierExplorer,
):
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    _run_suffix(explorer, state="frontier-state", steps=2)
    return explorer.observe_transition(
        state_signature_before="frontier-state::step-2",
        state_signature_after="next-level",
        action_name="ACTION1",
        action_data={},
        level_progressed=True,
        won=False,
        game_over=False,
    )


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


def test_delayed_terminal_lineage_is_nominated_without_credit():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        enable_adaptive_horizon=False,
        max_adaptive_suffix_actions=2,
        max_dormant_lineage_actions=8,
    )

    outcome = _nominate_delayed_terminal_candidate(explorer)

    assert outcome["terminal_success"] is True
    assert outcome["dormant_lineage_observation"] is True
    assert outcome["dormant_terminal_candidate_nominated"] is True
    assert outcome["credited"] is False
    summary = explorer.summary()
    assert summary["dormant_lineages_started"] == 1
    assert summary["dormant_lineage_actions"] == 1
    assert summary["dormant_lineage_terminal_candidates"] == 1
    assert summary["terminal_credits"] == 0
    candidate = summary["records"][0]["dormant_terminal_candidates"][0]
    assert len(candidate["actions"]) == 3
    assert candidate["status"] == "awaiting_replay"


def test_exact_delayed_terminal_replay_promotes_candidate():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        enable_adaptive_horizon=False,
        max_adaptive_suffix_actions=2,
        max_dormant_lineage_actions=8,
    )
    _nominate_delayed_terminal_candidate(explorer)
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )

    selections, outcomes = _run_suffix(
        explorer,
        state="frontier-state",
        steps=3,
        terminal_step=3,
    )

    assert all(
        item.replaying_dormant_terminal_candidate for item in selections
    )
    assert {item.action_limit for item in selections} == {3}
    assert outcomes[-1]["dormant_terminal_candidate_confirmed"] is True
    assert outcomes[-1]["credited"] is True
    summary = explorer.summary()
    assert summary["dormant_candidate_replay_attempts"] == 1
    assert summary["dormant_candidate_replay_actions"] == 3
    assert summary["dormant_candidate_confirmations"] == 1
    assert summary["terminal_credits"] == 1
    assert summary["successful_continuations"] == 1
    candidate = summary["records"][0]["dormant_terminal_candidates"][0]
    assert candidate["status"] == "terminal_confirmed"


def test_nonterminal_delayed_candidate_replay_is_refuted_not_credited():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        enable_adaptive_horizon=False,
        max_adaptive_suffix_actions=2,
        max_dormant_lineage_actions=8,
    )
    _nominate_delayed_terminal_candidate(explorer)
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )

    _run_suffix(explorer, state="frontier-state", steps=3)

    summary = explorer.summary()
    assert summary["dormant_candidate_refutations"] == 1
    assert summary["dormant_candidate_confirmations"] == 0
    assert summary["terminal_credits"] == 0
    candidate = summary["records"][0]["dormant_terminal_candidates"][0]
    assert candidate["status"] == "refuted"


def test_early_terminal_during_candidate_replay_is_not_exact_confirmation():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        enable_adaptive_horizon=False,
        max_adaptive_suffix_actions=2,
        max_dormant_lineage_actions=8,
    )
    _nominate_delayed_terminal_candidate(explorer)
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )

    _, outcomes = _run_suffix(
        explorer,
        state="frontier-state",
        steps=2,
        terminal_step=2,
    )

    assert outcomes[-1]["dormant_terminal_candidate_confirmed"] is False
    assert outcomes[-1]["credited"] is False
    summary = explorer.summary()
    assert summary["dormant_candidate_confirmations"] == 0
    assert summary["dormant_candidate_divergences"] == 1
    assert summary["terminal_credits"] == 0


def test_divergent_delayed_candidate_replay_is_inconclusive():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        enable_adaptive_horizon=False,
        max_adaptive_suffix_actions=2,
        max_dormant_lineage_actions=8,
    )
    _nominate_delayed_terminal_candidate(explorer)
    explorer.start_branch()
    explorer.capture(
        state_signature="frontier-state",
        objective_ids=["objective-a"],
    )
    selected = explorer.select(
        state_signature="frontier-state",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert selected is not None
    assert selected.replaying_dormant_terminal_candidate is True

    outcome = explorer.observe_transition(
        state_signature_before="frontier-state",
        state_signature_after="divergent-state",
        action_name="ACTION2",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )

    assert outcome["credited"] is False
    assert outcome["replaying_dormant_terminal_candidate"] is False
    summary = explorer.summary()
    assert summary["dormant_candidate_divergences"] == 1
    assert summary["dormant_candidate_refutations"] == 0
    assert summary["terminal_credits"] == 0
    candidate = summary["records"][0]["dormant_terminal_candidates"][0]
    assert candidate["status"] == "inconclusive_divergence"


def test_dormant_terminal_lineage_ablation_ignores_delayed_terminal():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        enable_adaptive_horizon=False,
        max_adaptive_suffix_actions=2,
        enable_dormant_terminal_lineage=False,
        max_dormant_lineage_actions=8,
    )

    outcome = _nominate_delayed_terminal_candidate(explorer)

    assert outcome["frontier_id"] == ""
    assert outcome["credited"] is False
    summary = explorer.summary()
    assert summary["dormant_terminal_lineage_enabled"] is False
    assert summary["dormant_lineages_started"] == 0
    assert summary["dormant_terminal_candidates"] == 0
    assert summary["terminal_credits"] == 0


def test_structural_terminal_requires_exact_replay_before_credit():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=4,
        enable_adaptive_horizon=False,
    )
    frontier_id = explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_motion",),
    )

    _, first_outcomes = _run_suffix(
        explorer,
        state="structural-state",
        steps=4,
        terminal_step=4,
    )

    assert first_outcomes[-1]["dormant_terminal_candidate_nominated"] is True
    assert first_outcomes[-1]["credited"] is False
    first_summary = explorer.summary()
    assert first_summary["structural_frontiers_captured"] == 1
    assert first_summary["structural_terminal_candidates"] == 1
    assert first_summary["terminal_credits"] == 0
    record = explorer.frontiers()[0]
    assert record.frontier_id == frontier_id
    assert record.frontier_kind == "structural_change"
    assert record.structural_trigger_families == {"entity_motion"}

    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-b",
        trigger_families=("relation_change",),
    )
    selections, replay_outcomes = _run_suffix(
        explorer,
        state="structural-state",
        steps=4,
        terminal_step=4,
    )

    assert all(
        selection.replaying_dormant_terminal_candidate
        for selection in selections
    )
    assert replay_outcomes[-1]["dormant_terminal_candidate_confirmed"] is True
    assert replay_outcomes[-1]["credited"] is True
    summary = explorer.summary()
    assert summary["structural_candidate_confirmations"] == 1
    assert summary["structural_terminal_credits"] == 1
    assert summary["causal_reduction_probes_compiled"] == 3


def test_structural_attribution_ablation_never_credits_first_terminal():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=2,
        enable_structural_terminal_attribution=False,
    )
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("structural_effect",),
    )

    _, outcomes = _run_suffix(
        explorer,
        state="structural-state",
        steps=1,
        terminal_step=1,
    )

    assert outcomes[-1]["credited"] is False
    summary = explorer.summary()
    assert summary["structural_terminal_candidates"] == 0
    assert summary["structural_terminal_credits"] == 0
    assert summary["structural_attribution_blocks"] == 1


def test_confirmed_structural_continuation_is_reduced_by_terminal_cut():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=4,
        enable_adaptive_horizon=False,
    )
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _run_suffix(
        explorer,
        state="structural-state",
        steps=4,
        terminal_step=4,
    )
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _run_suffix(
        explorer,
        state="structural-state",
        steps=4,
        terminal_step=4,
    )

    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    selections, outcomes = _run_suffix(
        explorer,
        state="structural-state",
        steps=3,
        terminal_step=3,
    )

    assert all(selection.testing_causal_reduction for selection in selections)
    assert outcomes[-1]["causal_reduction_confirmed"] is True
    assert outcomes[-1]["credited"] is True
    summary = explorer.summary()
    assert summary["causal_reduction_attempts"] == 1
    assert summary["causal_reduction_confirmations"] == 1
    assert summary["causal_reduction_terminal_credits"] == 1
    assert summary["minimum_confirmed_reduction_length"] == 3
    continuations = explorer.frontiers()[0].successful_continuations.values()
    assert any(item.causal_reduction for item in continuations)


def test_nonterminal_causal_cut_is_refuted_without_credit():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=4,
        enable_adaptive_horizon=False,
    )
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _run_suffix(
        explorer,
        state="structural-state",
        steps=4,
        terminal_step=4,
    )
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _run_suffix(
        explorer,
        state="structural-state",
        steps=4,
        terminal_step=4,
    )
    credits_before = explorer.summary()["terminal_credits"]

    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _, outcomes = _run_suffix(
        explorer,
        state="structural-state",
        steps=3,
    )

    assert outcomes[-1]["causal_reduction_refuted"] is True
    summary = explorer.summary()
    assert summary["causal_reduction_refutations"] == 1
    assert summary["terminal_credits"] == credits_before


def test_confirmed_frontier_is_reacquired_from_exact_reset_prefix():
    explorer = OnlineTerminalFrontierExplorer(max_suffix_actions=2)
    explorer.observe_transition(
        state_signature_before="reset-state",
        state_signature_after="structural-state",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )
    frontier_id = explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_motion",),
        equivalence_signature="equivalence-a",
    )
    frontier = explorer.frontiers()[0]
    terminal_action = _action("ACTION2")
    frontier.successful_continuations[(terminal_action.signature,)] = (
        SuccessfulContinuation(
            actions=(terminal_action,),
            state_signatures=("structural-state", "next-level"),
        )
    )

    explorer.start_branch()
    selected = explorer.select_reacquisition(
        state_signature="reset-state",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert selected is not None
    assert selected.frontier_id == frontier_id
    assert selected.action.action_name == "ACTION1"
    reacquired = explorer.observe_transition(
        state_signature_before="reset-state",
        state_signature_after="structural-state",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )

    assert reacquired["frontier_reacquisition_confirmed"] is True
    assert explorer.activate_reacquired_frontier(
        state_signature="structural-state"
    ) == frontier_id
    replay = explorer.select(
        state_signature="structural-state",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert replay is not None
    assert replay.replaying_successful_continuation is True
    summary = explorer.summary()
    assert summary["frontier_acquisition_paths"] == 1
    assert summary["frontier_reacquisition_attempts"] == 1
    assert summary["frontier_reacquisition_actions"] == 1
    assert summary["frontier_reacquisition_confirmations"] == 1


def test_reacquisition_ablation_records_no_reset_to_frontier_path():
    explorer = OnlineTerminalFrontierExplorer(
        enable_active_frontier_reacquisition=False,
    )
    explorer.observe_transition(
        state_signature_before="reset-state",
        state_signature_after="structural-state",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_motion",),
    )

    assert explorer.summary()["frontier_acquisition_paths"] == 0
    explorer.start_branch()
    assert explorer.active_reacquisition_available is False


def _confirm_eight_action_structural_continuation(
    explorer: OnlineTerminalFrontierExplorer,
) -> None:
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _run_suffix(
        explorer,
        state="structural-state",
        steps=8,
        terminal_step=8,
    )
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _run_suffix(
        explorer,
        state="structural-state",
        steps=8,
        terminal_step=8,
    )


def test_terminal_cut_confirmation_compiles_next_reduction_generation():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=8,
        enable_adaptive_horizon=False,
        max_causal_reduction_generations=3,
    )
    _confirm_eight_action_structural_continuation(explorer)
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    selections, outcomes = _run_suffix(
        explorer,
        state="structural-state",
        steps=6,
        terminal_step=6,
    )

    assert all(selection.testing_causal_reduction for selection in selections)
    assert outcomes[-1]["causal_reduction_confirmed"] is True
    summary = explorer.summary()
    assert summary["causal_reduction_probes_compiled"] == 6
    assert summary["recursive_reduction_probes_compiled"] == 3
    assert summary["maximum_reduction_generation"] == 2
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    recursive = explorer.select(
        state_signature="structural-state",
        available_actions=["ACTION1"],
    )
    assert recursive is not None
    assert recursive.testing_causal_reduction is True
    assert recursive.action_limit == 5


def test_recursive_reduction_ablation_keeps_first_generation_only():
    explorer = OnlineTerminalFrontierExplorer(
        max_suffix_actions=8,
        enable_adaptive_horizon=False,
        enable_recursive_terminal_causal_minimization=False,
    )
    _confirm_eight_action_structural_continuation(explorer)
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="structural-state",
        trigger_signature="trigger-a",
        trigger_families=("entity_transform",),
    )
    _run_suffix(
        explorer,
        state="structural-state",
        steps=6,
        terminal_step=6,
    )

    summary = explorer.summary()
    assert summary["causal_reduction_probes_compiled"] == 3
    assert summary["recursive_reduction_probes_compiled"] == 0
    assert summary["maximum_reduction_generation"] == 1


def test_structural_equivalence_nominates_terminal_only_transfer():
    explorer = OnlineTerminalFrontierExplorer(max_suffix_actions=3)
    explorer.capture_structural(
        state_signature="source-state",
        trigger_signature="trigger-source",
        trigger_families=("entity_motion",),
        equivalence_signature="equivalence-a",
    )
    source = explorer.frontiers()[0]
    actions = (_action("ACTION1"), _action("ACTION2"))
    source.successful_continuations[
        tuple(action.signature for action in actions)
    ] = SuccessfulContinuation(
        actions=actions,
        state_signatures=("source-state", "source-middle", "next-level"),
    )
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="target-state",
        trigger_signature="trigger-target",
        trigger_families=("entity_motion",),
        equivalence_signature="equivalence-a",
    )

    first = explorer.select(
        state_signature="target-state",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert first is not None
    assert first.testing_structural_transfer is True
    explorer.observe_transition(
        state_signature_before="target-state",
        state_signature_after="target-middle",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )
    second = explorer.select(
        state_signature="target-middle",
        available_actions=["ACTION1", "ACTION2"],
    )
    assert second is not None
    assert second.testing_structural_transfer is True
    terminal = explorer.observe_transition(
        state_signature_before="target-middle",
        state_signature_after="next-level",
        action_name="ACTION2",
        action_data={},
        level_progressed=True,
        won=False,
        game_over=False,
    )

    assert terminal["structural_transfer_confirmed"] is True
    assert terminal["credited"] is True
    summary = explorer.summary()
    assert summary["structural_transfer_probes_compiled"] == 1
    assert summary["structural_transfer_attempts"] == 1
    assert summary["structural_transfer_confirmations"] == 1
    assert summary["structural_transfer_terminal_credits"] == 1


def test_nonterminal_structural_transfer_is_refuted_without_credit():
    explorer = OnlineTerminalFrontierExplorer(max_suffix_actions=3)
    explorer.capture_structural(
        state_signature="source-state",
        trigger_signature="trigger-source",
        trigger_families=("entity_motion",),
        equivalence_signature="equivalence-a",
    )
    source = explorer.frontiers()[0]
    action = _action("ACTION1")
    source.successful_continuations[(action.signature,)] = (
        SuccessfulContinuation(
            actions=(action,),
            state_signatures=("source-state", "next-level"),
        )
    )
    explorer.start_branch()
    explorer.capture_structural(
        state_signature="target-state",
        trigger_signature="trigger-target",
        trigger_families=("entity_motion",),
        equivalence_signature="equivalence-a",
    )

    selected = explorer.select(
        state_signature="target-state",
        available_actions=["ACTION1"],
    )
    assert selected is not None
    assert selected.testing_structural_transfer is True
    outcome = explorer.observe_transition(
        state_signature_before="target-state",
        state_signature_after="still-target",
        action_name="ACTION1",
        action_data={},
        level_progressed=False,
        won=False,
        game_over=False,
    )

    assert outcome["structural_transfer_refuted"] is True
    assert outcome["credited"] is False
    summary = explorer.summary()
    assert summary["structural_transfer_refutations"] == 1
    assert summary["structural_transfer_terminal_credits"] == 0
