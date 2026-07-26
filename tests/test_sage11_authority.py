"""SAGE.11 authority staging and symbolic-safety tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from theory.sage11.authority import (
    NeuralActionCandidate,
    NeuralActionPrediction,
    NeuralAuthorityConfig,
    NeuralAuthorityMode,
    NeuroSymbolicRanker,
)
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


def _predict(
    _observation: object,
    candidates: Sequence[NeuralActionCandidate],
) -> Sequence[NeuralActionPrediction]:
    return tuple(
        NeuralActionPrediction(
            action_name=candidate.action_name,
            action_data=candidate.action_data,
            predicted_progress=(
                0.8 if candidate.action_name == "ACTION2" else 0.1
            ),
            predicted_effect=0.5,
            predicted_information_gain=0.5,
            predicted_risk=0.01,
            predicted_noop=0.01,
        )
        for candidate in candidates
    )


def _arbitrate(
    ranker: NeuroSymbolicRanker,
    *,
    protected: bool = False,
    danger_action: str = "",
):
    return ranker.arbitrate(
        symbolic_action_name="ACTION1",
        symbolic_action_data={},
        symbolic_source="legacy_fallback",
        observation=object(),
        candidates=(
            NeuralActionCandidate("ACTION1"),
            NeuralActionCandidate("ACTION2"),
        ),
        protected_competence_available=protected,
        context_signature="context-a",
        danger_veto=lambda action, _data: action == danger_action,
    )


def test_off_mode_does_not_invoke_predictor():
    def forbidden(*_args):
        raise AssertionError("off mode must not invoke neural predictor")

    ranker = NeuroSymbolicRanker(
        forbidden,
        config=NeuralAuthorityConfig(mode=NeuralAuthorityMode.OFF),
    )
    result = _arbitrate(ranker)
    assert not result.applied
    assert result.action_name == "ACTION1"
    assert ranker.summary()["evaluations"] == 0


def test_shadow_mode_is_action_identical_and_logs_counterfactual_ranking():
    ranker = NeuroSymbolicRanker(
        _predict,
        config=NeuralAuthorityConfig(mode=NeuralAuthorityMode.SHADOW),
    )
    result = _arbitrate(ranker)
    assert not result.applied
    assert result.action_name == "ACTION1"
    assert result.counterfactual_top_key
    ranker.observe_outcome(
        productive=True,
        unsafe=False,
        successful_route=True,
    )
    summary = ranker.summary()
    assert summary["action_identity_mismatches"] == 0
    assert summary["would_be_successful_route_preemptions"] == 1


def test_bounded_authority_requires_gate_and_yields_to_safety_and_competence():
    gated = NeuroSymbolicRanker(
        _predict,
        config=NeuralAuthorityConfig(
            mode=NeuralAuthorityMode.BOUNDED,
            bounded_gate_passed=True,
        ),
    )
    assert _arbitrate(gated).source == "neural_bounded_probe"
    protected = NeuroSymbolicRanker(
        _predict,
        config=NeuralAuthorityConfig(
            mode=NeuralAuthorityMode.BOUNDED,
            bounded_gate_passed=True,
        ),
    )
    assert not _arbitrate(protected, protected=True).applied
    danger = NeuroSymbolicRanker(
        _predict,
        config=NeuralAuthorityConfig(
            mode=NeuralAuthorityMode.BOUNDED,
            bounded_gate_passed=True,
        ),
    )
    assert not _arbitrate(danger, danger_action="ACTION2").applied
    assert danger.summary()["symbolic_danger_vetoes"] == 1


def test_two_nonproductive_bounded_probes_demote_until_rearmed():
    ranker = NeuroSymbolicRanker(
        _predict,
        config=NeuralAuthorityConfig(
            mode=NeuralAuthorityMode.BOUNDED,
            bounded_gate_passed=True,
            nonproductive_demotion_threshold=2,
        ),
    )
    for _ in range(2):
        assert _arbitrate(ranker).applied
        ranker.observe_outcome(
            productive=False,
            unsafe=False,
            successful_route=False,
        )
        ranker.start_branch()
    assert not _arbitrate(ranker).applied
    assert ranker.summary()["demotions"] == 1
    assert ranker.rearm(reason="new_effect") == 1
    ranker.start_branch()
    assert _arbitrate(ranker).applied


def test_controller_shadow_is_byte_identical_to_off_across_transitions():
    base_config = UnifiedCognitiveConfig(
        enable_relational_experiments=False,
        enable_operator_planning=False,
        enable_theory_planning=False,
        enable_promoted_options=False,
        enable_active_goal_hypotheses=False,
        enable_temporal_goal_composition=False,
        enable_causal_subgoal_induction=False,
        enable_frontier_oriented_exploration=False,
        enable_transferable_causal_schema_priors=False,
        enable_terminal_multiform_relational_induction=False,
    )
    off = UnifiedCognitiveController(
        "synthetic",
        config=base_config,
    )
    shadow = UnifiedCognitiveController(
        "synthetic",
        config=base_config,
        neural_ranker=NeuroSymbolicRanker(
            _predict,
            config=NeuralAuthorityConfig(
                mode=NeuralAuthorityMode.SHADOW,
            ),
        ),
    )
    grid = np.zeros((5, 5), dtype=np.int32)
    for _ in range(3):
        off_decision = off.select_action(
            current_grid=grid,
            available_actions=["ACTION1", "ACTION2"],
            legacy_action="ACTION1",
        )
        shadow_decision = shadow.select_action(
            current_grid=grid,
            available_actions=["ACTION1", "ACTION2"],
            legacy_action="ACTION1",
        )
        assert shadow_decision.to_dict() == off_decision.to_dict()
        for controller, decision in (
            (off, off_decision),
            (shadow, shadow_decision),
        ):
            controller.observe_transition(
                action=decision.action_name,
                action_data=decision.action_data,
                grid_before=grid,
                grid_after=grid.copy(),
                available_actions=["ACTION1", "ACTION2"],
            )
