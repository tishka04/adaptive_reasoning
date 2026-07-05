"""A6 semi-MDP theory option tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from theory.ar25_oracle import _DEFAULT_TASK_PROGRAM, build_ar25_oracle
from theory.correspondence_hypothesis import (
    CorrespondenceHypothesis,
    CorrespondenceObservation,
    correspondence_key,
)
from theory.live_transition_loop import LiveTransitionBeliefLoop
from theory.mechanic_hypothesis import GameTheory
from theory.precondition_hypothesis import (
    PreconditionHypothesis,
    PreconditionObservation,
    precondition_key,
)
from theory.theory_option import build_options_from_theory, run_option_trace_file


def test_builds_validate_correspondence_option_from_confirmed_theory():
    target_rule = correspondence_key("ACTION2", "validates", (10, 11))
    ready_key = precondition_key(
        target_rule,
        "ready_to_validate_correspondence",
    )
    theory = GameTheory("synthetic")
    theory.seed_actions(["ACTION2"])
    theory.add_semantic_hypotheses(
        correspondence=[
            CorrespondenceHypothesis(
                action="ACTION2",
                relation="validates",
                pair_colors=(10, 11),
            )
        ],
        preconditions=[
            PreconditionHypothesis(
                target_rule=target_rule,
                predicate="ready_to_validate_correspondence",
            )
        ],
    )
    for _ in range(2):
        theory.observe_correspondence(
            CorrespondenceObservation(
                action="ACTION2",
                pair_colors=(10, 11),
                level_complete=True,
            ),
            was_experiment=True,
        )
    for _ in range(3):
        theory.observe_precondition(
            PreconditionObservation(
                target_rule=target_rule,
                action="ACTION2",
                predicates_present={"ready_to_validate_correspondence"},
                succeeded=True,
            ),
            was_experiment=True,
        )

    options = build_options_from_theory(theory)

    assert len(options) == 1
    option = options[0]
    assert option.name == "validate_correspondence_colors10_11"
    assert option.target_rule == target_rule
    assert option.precondition_key == ready_key
    assert option.policy() == "ACTION2"
    assert option.can_initiate(
        theory,
        {"ready_to_validate_correspondence"},
        available_actions=["ACTION2"],
    )
    assert not option.can_initiate(theory, set(), available_actions=["ACTION2"])


def test_option_terminates_on_level_progress_or_contradiction():
    target_rule = correspondence_key("ACTION2", "validates", (10, 11))
    theory = GameTheory("synthetic")
    theory.seed_actions(["ACTION2"])
    theory.add_semantic_hypotheses(
        correspondence=[
            CorrespondenceHypothesis(
                action="ACTION2",
                relation="validates",
                pair_colors=(10, 11),
            )
        ],
        preconditions=[
            PreconditionHypothesis(
                target_rule=target_rule,
                predicate="ready_to_validate_correspondence",
                evidence_for=["a", "b", "c"],
            )
        ],
    )
    for _ in range(2):
        theory.observe_correspondence(
            CorrespondenceObservation(
                action="ACTION2",
                pair_colors=(10, 11),
                level_complete=True,
            ),
            was_experiment=True,
        )
    option = build_options_from_theory(theory)[0]

    before = np.zeros((6, 6), dtype=np.int32)
    before[1:3, 1:3] = 10
    before[1:3, 4:6] = 11
    loop = LiveTransitionBeliefLoop(
        "synthetic",
        available_actions=["ACTION2"],
        background_value=0,
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    success_update = loop.observe_grids(
        action="ACTION2",
        grid_before=before,
        grid_after=before.copy(),
        available_actions=["ACTION2"],
        levels_completed_before=0,
        levels_completed_after=1,
    )
    contradiction_update = loop.observe_grids(
        action="ACTION2",
        grid_before=before,
        grid_after=before.copy(),
        available_actions=["ACTION2"],
        levels_completed_before=1,
        levels_completed_after=1,
    )

    success = option.observe_termination(
        step=1,
        actual_action="ACTION2",
        predicates_present={"ready_to_validate_correspondence"},
        update=success_update,
    )
    contradiction = option.observe_termination(
        step=2,
        actual_action="ACTION2",
        predicates_present={"ready_to_validate_correspondence"},
        update=contradiction_update,
    )

    assert success is not None
    assert success.termination == "success"
    assert contradiction is not None
    assert contradiction.termination == "contradiction"


def test_ar25_option_trace_replay_no_longer_supplies_live_preparation():
    trace_path = Path("human_traces/ar25-e3c63847.20260420-163550.steps.jsonl")
    common_kwargs = dict(
        trace_path=trace_path,
        game_id="ar25-e3c63847",
        task_program_path=_DEFAULT_TASK_PROGRAM,
        max_steps=220,
        background_value=9,
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    option_run = run_option_trace_file(**common_kwargs)
    oracle = build_ar25_oracle()
    score = option_run.score(oracle)

    assert option_run.option_invocations == 0
    assert option_run.summary()["options_seen"] == []
    assert score.confirmation_precision >= 0.95
    assert score.wrong_confirmations == 0
    assert score.human_alignment == 1.0
