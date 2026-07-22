"""Hierarchical exploitation tests for confirmed causal dependencies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation, build_transition_record
from theory.online_causal_option import (
    CausalOptionTerminalStatus,
    OnlineCausalOptionStore,
)
from theory.online_causal_subgoal_graph import (
    CausalMechanicEvidence,
    CausalSubgoalEdge,
    CausalSubgoalEdgeStatus,
)
from theory.online_goal_hypothesis import GeneratedGoalHypothesis
from theory.online_mediated_exploitation import MediatedExploitationPrediction
from theory.online_terminal_objective import OnlineTerminalObjectiveStore
from theory.unified_cognitive_controller import UnifiedCognitiveController


def _edge(*, confirmed: bool = True) -> CausalSubgoalEdge:
    edge = CausalSubgoalEdge(
        edge_key="causal::source=>target",
        source_objective_id="source",
        target_objective_id="target",
        minimum_independent_support=2,
    )
    edge.intervention_evidence["ACTION1"] = CausalMechanicEvidence(
        signature="ACTION1",
        observations=3,
        source_progress_events=3,
        enablement_successes=2,
    )
    if confirmed:
        edge.support_events = 2
        edge.support_contexts.update({"context-0", "context-1"})
        edge.support_branches.update({0, 1})
    return edge


def _store() -> OnlineTerminalObjectiveStore:
    store = OnlineTerminalObjectiveStore()
    store.register_generated(GeneratedGoalHypothesis(
        objective_id="target",
        family="exhaust",
        source_color=3,
        target_color=None,
        predicate="object_count_equals_zero",
        supporting_rule_keys=(),
        supporting_actions=("ACTION1", "ACTION2"),
        generation_reason="synthetic_option_test",
        prior_priority=1.0,
    ))
    return store


def _grid(*, include_three: bool = True) -> np.ndarray:
    grid = np.zeros((6, 6), dtype=np.int32)
    if include_three:
        grid[2, 2] = 3
    grid[4, 4] = 8
    return grid


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=["ACTION1", "ACTION2"],
        infer_players=False,
    )


def _update(
    before: np.ndarray,
    after: np.ndarray,
    *,
    action: str = "ACTION2",
    terminal: bool = False,
    game_over: bool = False,
):
    record = build_transition_record(
        action=action,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1", "ACTION2"],
        levels_completed_before=0,
        levels_completed_after=1 if terminal else 0,
        game_state_after=(
            "WIN" if terminal else ("GAME_OVER" if game_over else "NOT_FINISHED")
        ),
        infer_players=False,
    )
    return SimpleNamespace(record=record, action=action)


def _compiled_store(*, max_actions: int = 3):
    edge = _edge()
    options = OnlineCausalOptionStore(
        max_downstream_actions=max_actions,
        terminal_credit_window=4,
    )
    options.sync_confirmed_edges([edge])
    option = options.options()[0]
    return edge, options, option


def test_only_mechanically_confirmed_edges_compile_into_options():
    candidate = _edge(confirmed=False)
    confirmed = _edge(confirmed=True)
    confirmed.edge_key = "causal::confirmed-source=>target"
    options = OnlineCausalOptionStore()

    compiled = options.sync_confirmed_edges([candidate, confirmed])

    assert candidate.status == CausalSubgoalEdgeStatus.CANDIDATE
    assert confirmed.status == CausalSubgoalEdgeStatus.CONFIRMED
    assert len(compiled) == 1
    assert compiled[0].edge_key == confirmed.edge_key
    assert compiled[0].productive_preparation_signatures == {"ACTION1"}


def test_opened_option_prefers_an_untried_nonpreparation_suffix_action():
    edge, options, option = _compiled_store()
    options.note_openings([edge.edge_key], context_signature="opened")

    selection = options.select_downstream(
        _observation(_grid()),
        safe_actions=["ACTION1", "ACTION2"],
    )

    assert selection is not None
    assert selection.option_id == option.option_id
    assert selection.action_name == "ACTION2"
    assert selection.intervention_signature == "ACTION2"
    assert selection.replaying_terminal_sequence is False


def test_successor_policy_owns_its_attempt_budget_after_state_change():
    edge, options, option = _compiled_store()
    options.note_openings([edge.edge_key], context_signature="opened")
    assert options._active is not None
    options._active.signature_attempts["ACTION2"] = (
        options.max_trials_per_signature
    )
    transferred = MediatedExploitationPrediction(
        policy_id="online-successor-policy",
        option_id=option.option_id,
        objective_id="target",
        action_signature="ACTION2",
        action_transfer_signature="ACTION2",
        compatible=True,
        same_latent_mode=True,
        state_id="successor-state",
        chain_depth=2,
        known_productive=True,
        structurally_transferred=True,
        structural_policy_id="successor-structural-policy:1",
        expected_gain=1.0,
        confidence=0.5,
    )

    selection = options.select_downstream(
        _observation(_grid()),
        safe_actions=["ACTION1", "ACTION2"],
        mediated_exploitation_predictions={"ACTION2": transferred},
    )

    assert selection is not None
    assert selection.action_name == "ACTION2"


def test_terminal_transition_credits_complete_option_not_target_goal_truth():
    edge, options, option = _compiled_store()
    store = _store()
    before = _grid()
    options.note_openings([edge.edge_key], context_signature="opened-0")
    selection = options.select_downstream(
        _observation(before),
        safe_actions=["ACTION1", "ACTION2"],
    )
    assert selection is not None

    outcome = options.observe_transition(
        _update(before, before, terminal=True),
        store=store,
        option_id=selection.option_id,
        causal_edge_key=selection.edge_key,
        intervention_signature=selection.intervention_signature,
        context_signature="terminal-0",
    )

    assert outcome["terminal_credited_option"] == option.option_id
    assert option.status == CausalOptionTerminalStatus.NEEDS_CONTRAST
    assert store.objective("target").terminal_support == 0
    assert edge.status == CausalSubgoalEdgeStatus.CONFIRMED


def test_terminal_success_from_unrelated_action_does_not_credit_open_option():
    edge, options, option = _compiled_store()
    store = _store()
    before = _grid()
    options.note_openings([edge.edge_key], context_signature="opened")

    outcome = options.observe_transition(
        _update(before, before, terminal=True),
        store=store,
        option_id="",
        causal_edge_key="",
        context_signature="unrelated-terminal",
    )

    assert outcome["terminal_credited_option"] == ""
    assert option.terminal_support == 0


def test_successful_suffix_replays_and_requires_an_independent_terminal_context():
    edge, options, option = _compiled_store()
    store = _store()
    before = _grid()
    for branch in range(2):
        options.note_openings(
            [edge.edge_key],
            context_signature=f"opened-{branch}",
        )
        selection = options.select_downstream(
            _observation(before),
            safe_actions=["ACTION1", "ACTION2"],
        )
        assert selection is not None
        if branch == 1:
            assert selection.action_name == "ACTION2"
            assert selection.replaying_terminal_sequence is True
        options.observe_transition(
            _update(before, before, terminal=True),
            store=store,
            option_id=selection.option_id,
            causal_edge_key=selection.edge_key,
            intervention_signature=selection.intervention_signature,
            context_signature=f"terminal-{branch}",
        )
        if branch == 0:
            assert option.status == CausalOptionTerminalStatus.NEEDS_CONTRAST
            options.start_branch()

    assert option.status == CausalOptionTerminalStatus.TERMINAL_SUPPORTED
    assert option.best_terminal_sequence == ("ACTION2",)


def test_two_nonterminal_target_completions_refute_option_value_not_mechanic():
    edge, options, option = _compiled_store(max_actions=1)
    store = _store()
    before = _grid()
    after = _grid(include_three=False)
    for branch in range(2):
        options.note_openings(
            [edge.edge_key],
            context_signature=f"opened-{branch}",
        )
        selection = options.select_downstream(
            _observation(before),
            safe_actions=["ACTION1", "ACTION2"],
        )
        assert selection is not None
        outcome = options.observe_transition(
            _update(before, after),
            store=store,
            option_id=selection.option_id,
            causal_edge_key=selection.edge_key,
            intervention_signature=selection.intervention_signature,
            context_signature=f"nonterminal-{branch}",
        )
        assert outcome["target_completed"] is True
        if branch == 0:
            options.start_branch()

    assert option.status == CausalOptionTerminalStatus.TERMINAL_REFUTED
    assert option.terminal_contradictions == 2
    assert edge.status == CausalSubgoalEdgeStatus.CONFIRMED
    assert store.objective("target").status.value == "candidate"


def test_two_target_completion_failures_in_one_branch_do_not_refute_option():
    edge, options, option = _compiled_store(max_actions=1)
    store = _store()
    before = _grid()
    after = _grid(include_three=False)
    for attempt in range(2):
        options.note_openings(
            [edge.edge_key],
            context_signature=f"same-branch-opening-{attempt}",
        )
        selection = options.select_downstream(
            _observation(before),
            safe_actions=["ACTION1", "ACTION2"],
        )
        assert selection is not None
        options.observe_transition(
            _update(before, after),
            store=store,
            option_id=selection.option_id,
            causal_edge_key=selection.edge_key,
            intervention_signature=selection.intervention_signature,
        )

    assert option.terminal_contradictions == 1
    assert option.status == CausalOptionTerminalStatus.CANDIDATE


def test_unspent_opening_is_censored_without_terminal_contradiction():
    edge, options, option = _compiled_store()
    options.note_openings([edge.edge_key], context_signature="opened")

    options.start_branch()

    assert option.terminal_contradictions == 0
    assert options.summary()["censored_openings"] == 1


def test_controller_prioritizes_active_confirmed_option_with_full_audit():
    controller = UnifiedCognitiveController(
        "synthetic-causal-option-controller",
        available_actions=["ACTION1", "ACTION2"],
    )
    controller.terminal_objectives.register_generated(GeneratedGoalHypothesis(
        objective_id="source",
        family="convert",
        source_color=3,
        target_color=4,
        predicate="source_target_color_transform",
        supporting_rule_keys=(),
        supporting_actions=("ACTION1",),
        generation_reason="synthetic_option_source",
        prior_priority=1.0,
    ))
    controller.terminal_objectives.register_generated(GeneratedGoalHypothesis(
        objective_id="target",
        family="exhaust",
        source_color=3,
        target_color=None,
        predicate="object_count_equals_zero",
        supporting_rule_keys=(),
        supporting_actions=("ACTION1", "ACTION2"),
        generation_reason="synthetic_option_target",
        prior_priority=1.0,
    ))
    observation = _observation(_grid())
    controller.causal_subgoals.note_blocked(
        "target",
        observation,
        controller.terminal_objectives,
    )
    edge = next(
        edge for edge in controller.causal_subgoals.edges()
        if edge.source_objective_id == "source"
        and edge.target_objective_id == "target"
    )
    edge.support_events = 2
    edge.support_contexts.update({"context-0", "context-1"})
    edge.support_branches.update({0, 1})
    edge.intervention_evidence["ACTION1"] = CausalMechanicEvidence(
        signature="ACTION1",
        observations=2,
        source_progress_events=2,
        enablement_successes=2,
    )
    controller.causal_options.sync_confirmed_edges([edge])
    controller.causal_options.note_openings(
        [edge.edge_key],
        context_signature="observed-opening",
    )

    decision = controller.select_action(
        current_grid=_grid(),
        available_actions=["ACTION1", "ACTION2"],
        legacy_action="ACTION1",
    )

    assert decision.source == "causal_option_downstream_probe"
    assert decision.action_name == "ACTION2"
    assert decision.causal_option_id
    assert decision.causal_option_edge_key == edge.edge_key
    assert decision.causal_option_terminal_status == "candidate"
    assert decision.causal_option_phase == "downstream_search"
    assert decision.causal_option_intervention_signature == "ACTION2"
    assert decision.causal_option_selection_utility is not None
