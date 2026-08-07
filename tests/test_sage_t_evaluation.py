from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from theory.live_transition_loop import build_transition_record
from theory.sage_t.compiler import compile_transition_record
from theory.sage_t.evaluation import (
    ActivePairResult,
    CounterfactualPanel,
    SageTCounterfactualEvaluator,
    active_progress_gate,
    counterfactual_gate,
    panels_from_binding_pairs,
    panels_from_transitions,
)
from theory.sage_t.synthesis import (
    DeterministicFragmentProposer,
    ProgramAssembler,
)


def _transition(action: str, after: np.ndarray):
    before = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    return compile_transition_record(
        build_transition_record(
            action=action,
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION1", "ACTION2"],
        )
    )


def test_same_prestate_evaluation_hides_untouched_counterfactual_arms() -> None:
    moved = _transition(
        "ACTION1",
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.int64,
        ),
    )
    noop = _transition(
        "ACTION2",
        np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    transitions = (moved, noop)
    panels = panels_from_transitions(transitions, source_game="synthetic")
    proposal = DeterministicFragmentProposer().propose(
        available_actions=("ACTION1", "ACTION2"),
        transitions=transitions,
    )
    programs = ProgramAssembler().assemble(
        proposal.fragments,
        available_actions=("ACTION1", "ACTION2"),
    )

    result = SageTCounterfactualEvaluator().evaluate(
        programs,
        panels,
    )

    assert len(result.panels) == 1
    assert len(result.panels[0].revealed_actions) == 1
    assert math.isfinite(result.panels[0].held_out_log_likelihood)
    assert result.execution_errors == 0
    assert result.forbidden_fields == 0
    assert result.illegal_actions == 0


def test_counterfactual_gate_is_strictly_fail_closed() -> None:
    moved = _transition(
        "ACTION1",
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.int64,
        ),
    )
    noop = _transition(
        "ACTION2",
        np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    transitions = (moved, noop)
    panels = panels_from_transitions(transitions)
    proposal = DeterministicFragmentProposer().propose(
        available_actions=("ACTION1", "ACTION2"),
        transitions=transitions,
    )
    programs = ProgramAssembler().assemble(
        proposal.fragments,
        available_actions=("ACTION1", "ACTION2"),
    )
    evaluations = SageTCounterfactualEvaluator().evaluate_required_conditions(
        programs,
        panels,
    )

    report = counterfactual_gate(
        evaluations,
        baseline_log_likelihood=-100.0,
        paired_interval_lower=0.1,
        non_negative_games=2,
    )

    # Synthetic data need not pass every scientific comparison, but every gate
    # is explicit and a failed comparison cannot silently promote authority.
    assert report.passed == all(report.checks.values())
    assert set(report.checks) == {
        "better_than_baseline",
        "information_beats_random",
        "joint_beats_dynamics_only",
        "joint_beats_action_only",
        "two_of_three_games_non_negative",
        "zero_forbidden_fields",
        "zero_illegal_actions",
        "zero_execution_errors",
    }


def test_active_gate_requires_complete_safe_paired_improvement() -> None:
    pairs = tuple(
        ActivePairResult(
            game_id=game,
            seed=seed,
            sage_t_levels_completed_delta=2,
            baseline_levels_completed_delta=1,
            sage_t_actions=1000,
            baseline_actions=1000,
            sage_t_game_overs=0,
            baseline_game_overs=1,
        )
        for game in ("re86", "ls20", "sc25")
        for seed in range(5)
    )

    passed = active_progress_gate(
        pairs,
        configuration_frozen=True,
    )
    failed = active_progress_gate(
        pairs[:-1],
        configuration_frozen=True,
    )

    assert passed.passed is True
    assert passed.metrics["paired_interval_lower_95"] > 0.0
    assert failed.passed is False
    assert failed.checks["complete_paired_design"] is False


def test_counterfactual_evaluator_refuses_non_source_panels() -> None:
    moved = _transition(
        "ACTION1",
        np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64),
    )
    noop = _transition(
        "ACTION2",
        np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="source-only"):
        CounterfactualPanel(
            panel_id="closed_holdout",
            state=moved.state_before,
            arms=(moved, noop),
            split="holdout",
        )


def test_reset_compiles_as_non_evidential_branch_control() -> None:
    grid = np.array([[0, 1], [0, 0]], dtype=np.int64)
    reset = compile_transition_record(
        build_transition_record(
            action="RESET",
            grid_before=grid,
            grid_after=grid,
            available_actions=["ACTION1"],
        )
    )

    assert reset.reset is True
    assert reset.action.action_name == "RESET"
    assert reset.observation.known_channels == frozenset()


def test_existing_v43_binding_pairs_adapt_without_outcome_leakage() -> None:
    before = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int64)

    def arm(action: str, after: np.ndarray):
        return SimpleNamespace(
            trace=SimpleNamespace(
                selected_action_name=action,
                selected_action_data={},
                frame_before=before,
                frame_after=after,
                available_action_names=("ACTION1", "ACTION2"),
                game_state_before="NOT_FINISHED",
                game_state_after="NOT_FINISHED",
                levels_completed_before=0,
                levels_completed_after=0,
                step_index=3,
            )
        )

    pair = SimpleNamespace(
        source_split="source_validation",
        game_id="source_game",
        pair_digest="pair",
        left=arm(
            "ACTION1",
            np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int64),
        ),
        right=arm("ACTION2", before),
    )

    panels = panels_from_binding_pairs((pair,))

    assert len(panels) == 1
    assert panels[0].panel_id == "pair"
    assert {arm.action.action_name for arm in panels[0].arms} == {
        "ACTION1",
        "ACTION2",
    }
