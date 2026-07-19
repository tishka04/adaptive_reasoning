"""Promotion, composition, and directed-progress tests for online options."""

from __future__ import annotations

import numpy as np

from theory.generic_discriminating_experiment_designer import (
    DiscriminatingPrediction,
)
from theory.live_transition_loop import build_observation
from theory.online_relational_option import OnlineRelationalOptionCompiler
from theory.promoted_relational_rule import PromotedRelationalRule
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


def _alignment_transition(
    source_position: tuple[int, int],
    target_position: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    before = np.zeros((12, 12), dtype=np.int32)
    before[source_position] = 3
    before[target_position] = 4
    after = before.copy()
    after[source_position] = 0
    after[source_position[0], target_position[1]] = 3
    return before, after


def _confirmed_rule(
    *,
    key: str,
    family: str,
    outcome: str,
    predicate: str = "aligned_with",
    source: int = 3,
    target: int = 4,
) -> PromotedRelationalRule:
    return PromotedRelationalRule(
        action="ACTION6",
        family=family,
        predicate=predicate,
        source_color=source,
        target_color=target,
        expected_outcome=outcome,
        source_prediction_key=key,
        support=2,
        independent_contexts={"context-a", "context-b"},
    )


def test_independent_live_confirmations_promote_and_execute_a_rule_option():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION6"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=8,
            enable_active_goal_hypotheses=False,
        ),
    )
    controller.register_predictions([
        DiscriminatingPrediction(
            key="relation::ACTION6::aligned_with::colors3_4::appears",
            action="ACTION6",
            source_color=3,
            target_color=4,
            family="relation",
            predicate="aligned_with",
            predicted_outcome="appears",
        ),
        DiscriminatingPrediction(
            key="relation::ACTION6::aligned_with::colors3_4::absent",
            action="ACTION6",
            source_color=3,
            target_color=4,
            family="relation",
            predicate="aligned_with",
            predicted_outcome="absent",
        ),
    ])

    for source, target in (((2, 2), (8, 8)), ((3, 3), (9, 9))):
        before, after = _alignment_transition(source, target)
        decision = controller.select_action(
            current_grid=before,
            available_actions=["ACTION6"],
            legacy_action="ACTION6",
        )
        assert decision.source == "relational_experiment"
        controller.observe_transition(
            action="ACTION6",
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION6"],
        )

    rules = controller.theory.promoted_relational_rules()
    assert len(rules) == 1
    assert rules[0].context_count == 2
    assert rules[0].goal_relevant is True

    before, after = _alignment_transition((4, 4), (10, 10))
    option = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    assert option.source == "terminal_objective_probe"
    assert option.action_data == {"x": 4, "y": 4}
    assert option.source_rule_key == rules[0].key
    assert option.objective_status == "candidate"
    assert option.objective_distance == 1.0
    controller.observe_transition(
        action="ACTION6",
        action_data=option.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION6"],
    )

    execution = controller.summary()["option_execution"]
    assert execution["expected_successes"] == 1
    assert execution["functional_successes"] == 1
    assert execution["visual_only_outcomes"] == 0


def test_repeated_same_context_confirms_candidate_but_does_not_promote_it():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION6"],
    )
    before = np.zeros((8, 8), dtype=np.int32)
    before[2, 2] = 3
    after = before.copy()
    after[2, 2] = 4

    for _ in range(2):
        decision = controller.select_action(
            current_grid=before,
            available_actions=["ACTION6"],
            legacy_action="ACTION6",
        )
        controller.observe_transition(
            action="ACTION6",
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION6"],
        )

    assert controller.summary()["active_promoted_relational_rules"] == 0


def test_compiler_uses_an_establishing_option_to_prepare_a_missing_relation():
    establish = _confirmed_rule(
        key="establish",
        family="relation",
        outcome="appears",
    )
    preserve = _confirmed_rule(
        key="preserve",
        family="relation",
        outcome="preserved",
    )
    compiler = OnlineRelationalOptionCompiler()
    options = compiler.compile([establish, preserve])
    by_rule = {option.rule_key: option for option in options}
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[2, 2] = 3
    grid[8, 8] = 4
    observation = build_observation(grid, available_actions=["ACTION6"])

    chain = compiler.preparation_chain(
        by_rule[preserve.key],
        options,
        observation,
    )

    assert [option.rule_key for option in chain] == [establish.key, preserve.key]
    assert compiler.assess(chain[0], observation).ready is True


def test_game_theory_ledger_keeps_promoted_rule_revision_state():
    controller = UnifiedCognitiveController("synthetic")
    rule = _confirmed_rule(
        key="ledger",
        family="color_transform",
        outcome="3->4",
        predicate="source_target_color_transform",
    )
    controller.theory.add_promoted_relational_rule(rule)

    record = next(
        row for row in controller.theory.to_ledger() if row.key == rule.key
    )
    summary = controller.theory.summary()

    assert record.support == 2
    assert rule.key in summary["promoted_relational_rules"]


def test_preservation_mechanic_is_not_invented_as_a_terminal_objective():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION6"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=0,
            enable_active_goal_hypotheses=False,
        ),
    )
    rule = _confirmed_rule(
        key="preserve-visual-only",
        family="relation",
        outcome="preserved",
    )
    controller.theory.add_promoted_relational_rule(rule)
    before = np.zeros((12, 12), dtype=np.int32)
    before[2, 2] = 3
    before[2, 8] = 4
    before[10, 10] = 5
    after = before.copy()
    after[10, 10] = 6

    decision = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    assert decision.source == "legacy_fallback"
    controller.observe_transition(
        action="ACTION6",
        action_data=decision.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION6"],
    )

    execution = controller.summary()["option_execution"]
    assert execution["executions"] == 0
    assert execution["expected_successes"] == 0
    assert execution["functional_successes"] == 0
    assert execution["visual_only_outcomes"] == 0
    assert controller.summary()["terminal_objectives"]["objectives"] == 0


def test_repeated_option_contradictions_remove_rule_from_active_compilation():
    rule = _confirmed_rule(
        key="revisable",
        family="relation",
        outcome="appears",
    )
    for index in range(4):
        rule.observe_application(
            expected_outcome_observed=False,
            functional_progress=False,
            level_progress=False,
            visual_change=False,
            context_signature=f"contradiction-{index}",
        )

    compiler = OnlineRelationalOptionCompiler()
    assert rule.status.value == "refuted"
    assert compiler.compile([rule]) == []


def test_confirmed_break_mechanic_compiles_as_a_directional_option():
    controller = UnifiedCognitiveController(
        "synthetic-break",
        available_actions=["ACTION6"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=0,
            enable_active_goal_hypotheses=False,
        ),
    )
    rule = _confirmed_rule(
        key="break-alignment",
        family="relation",
        outcome="broken",
    )
    controller.theory.add_promoted_relational_rule(rule)
    before = np.zeros((12, 12), dtype=np.int32)
    before[2, 2] = 3
    before[2, 8] = 4
    after = before.copy()
    after[2, 2] = 0
    after[5, 5] = 3

    decision = controller.select_action(
        current_grid=before,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )
    assert decision.source == "terminal_objective_probe"
    controller.observe_transition(
        action="ACTION6",
        action_data=decision.action_data,
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION6"],
    )

    outcome = controller.summary()["recent_option_outcomes"][-1]
    assert outcome["relation_broken"] is True
    assert outcome["functional_progress"] is True


def test_mechanically_true_preservation_never_consumes_goal_probe_budget():
    controller = UnifiedCognitiveController(
        "synthetic",
        available_actions=["ACTION6"],
        config=UnifiedCognitiveConfig(
            max_bootstrap_experiments=0,
            min_option_executions_before_quarantine=3,
            enable_active_goal_hypotheses=False,
        ),
    )
    rule = _confirmed_rule(
        key="sterile-preservation",
        family="relation",
        outcome="preserved",
    )
    controller.theory.add_promoted_relational_rule(rule)

    for index in range(3):
        before = np.zeros((12, 12), dtype=np.int32)
        before[2, 2] = 3
        before[2, 8] = 4
        before[10, 10] = 5 + index
        after = before.copy()
        after[10, 10] = 6 + index
        decision = controller.select_action(
            current_grid=before,
            available_actions=["ACTION6"],
            legacy_action="ACTION6",
        )
        assert decision.source == "legacy_fallback"
        controller.observe_transition(
            action="ACTION6",
            action_data=decision.action_data,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION6"],
        )

    next_grid = np.zeros((12, 12), dtype=np.int32)
    next_grid[3, 3] = 3
    next_grid[3, 9] = 4
    fallback = controller.select_action(
        current_grid=next_grid,
        available_actions=["ACTION6"],
        legacy_action="ACTION6",
    )

    assert not fallback.source.startswith("terminal_objective")
    assert rule.status.value == "confirmed"
    objective_summary = controller.summary()["terminal_objectives"]
    assert objective_summary["objectives"] == 0
    assert objective_summary["probe_actions"] == 0
