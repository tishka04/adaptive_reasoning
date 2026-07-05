"""A4 minimal theory-conditioned planner tests."""

from __future__ import annotations

from pathlib import Path

from theory.ar25_oracle import _DEFAULT_TASK_PROGRAM, build_ar25_oracle
from theory.correspondence_hypothesis import (
    CorrespondenceHypothesis,
    CorrespondenceObservation,
    correspondence_key,
)
from theory.mechanic_hypothesis import GameTheory
from theory.precondition_hypothesis import (
    PreconditionHypothesis,
    PreconditionObservation,
    precondition_key,
)
from theory.role_hypotheses import ActionRoleHypothesis
from theory.theory_conditioned_planner import (
    TheoryConditionedPlanner,
    run_planner_trace_file,
)


def test_planner_selects_validator_from_confirmed_correspondence_rule():
    theory = GameTheory("synthetic")
    theory.seed_actions(["ACTION2", "ACTION5"])
    theory.add_semantic_hypotheses(
        action_roles=[
            ActionRoleHypothesis(
                action="ACTION5",
                role="control_switch",
                evidence_for=["human_prior"],
                prior_confidence=0.7,
            )
        ],
        correspondence=[
            CorrespondenceHypothesis(
                action="ACTION2",
                relation="validates",
                pair_colors=(10, 11),
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

    planner = TheoryConditionedPlanner()
    plan = planner.plan(theory, ["ACTION1", "ACTION2", "ACTION5"])

    assert plan is not None
    assert plan.first_action == "ACTION2"
    assert plan.source_rule_key == correspondence_key(
        "ACTION2",
        "validates",
        (10, 11),
    )
    assert plan.roles["control_switch"] == ["ACTION5"]
    assert plan.roles["validates_correspondence"] == ["ACTION2"]
    assert "correspondence rule" in plan.planned_actions[0].reason


def test_precondition_hypothesis_gates_planner_when_required():
    target_key = correspondence_key("ACTION2", "validates", (10, 11))
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
                target_rule=target_key,
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

    planner = TheoryConditionedPlanner(require_preconditions=True)
    assert planner.plan(
        theory,
        ["ACTION2"],
        precondition_features={target_key: {"ready_to_validate_correspondence"}},
    ) is None

    for _ in range(3):
        theory.observe_precondition(
            PreconditionObservation(
                target_rule=target_key,
                action="ACTION2",
                predicates_present={"ready_to_validate_correspondence"},
                succeeded=True,
            ),
            was_experiment=True,
        )

    plan = planner.plan(
        theory,
        ["ACTION2"],
        precondition_features={target_key: {"ready_to_validate_correspondence"}},
    )
    assert plan is not None
    assert plan.first_action == "ACTION2"
    assert plan.preconditions == [
        precondition_key(target_key, "ready_to_validate_correspondence")
    ]
    assert planner.plan(theory, ["ACTION2"], precondition_features={target_key: set()}) is None


def test_ar25_planner_executes_correspondence_driven_action_without_bad_confirmations():
    run = run_planner_trace_file(
        Path("human_traces/ar25-e3c63847.20260420-163550.steps.jsonl"),
        game_id="ar25-e3c63847",
        task_program_path=_DEFAULT_TASK_PROGRAM,
        max_steps=220,
        background_value=9,
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    oracle = build_ar25_oracle()
    score = run.score(oracle)
    target_key = correspondence_key("ACTION2", "validates", (10, 11))

    assert any(event.rule_key == target_key for event in run.correspondence_driven_events)
    assert any(
        event.rule_key == target_key and event.executed
        for event in run.correspondence_driven_events
    )
    assert any(event.successful for event in run.correspondence_driven_events)
    assert score.confirmation_precision >= 0.95
    assert score.wrong_confirmations == 0
    assert score.human_alignment == 1.0
    assert target_key in run.summary()["correspondence_rules"]


def test_ar25_non_circular_ready_does_not_trace_align_opportunistically():
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
    ungated = run_planner_trace_file(**common_kwargs)
    gated = run_planner_trace_file(
        **common_kwargs,
        planner=TheoryConditionedPlanner(require_preconditions=True),
    )
    oracle = build_ar25_oracle()
    score = gated.score(oracle)
    target_key = correspondence_key("ACTION2", "validates", (10, 11))
    precondition = precondition_key(
        target_key,
        "ready_to_validate_correspondence",
    )

    ungated_executed = len(ungated.executed_events)
    gated_executed = len(gated.executed_events)

    assert gated_executed <= ungated_executed
    assert gated_executed == 0
    assert not any(event.precondition_key == precondition for event in gated.executed_events)
    assert precondition not in gated.summary()["preconditions_confirmed"]
    assert score.confirmation_precision >= 0.95
    assert score.wrong_confirmations == 0
    assert score.human_alignment == 1.0
