"""A22 generic relation option tests."""

from __future__ import annotations

from pathlib import Path

from theory.epistemic_metrics import HypothesisStatus
from theory.non_ar25_multi_family_experiment import MultiFamilyHypothesisRevision
from theory.relation_option import (
    AvoidRelationOption,
    PrepareRelationOption,
    run_relation_option_micro_run,
)


def test_prepare_relation_option_runs_active_relation_experiment():
    result = run_relation_option_micro_run(
        option=PrepareRelationOption("same_shape", desired_outcome="preserved"),
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
    )

    assert result.error == ""
    assert result.initiation_holds
    assert result.policy_action_chosen
    assert result.transition_count == 1
    assert result.env_actions == 1
    assert result.hypothesis_remains_unresolved_until_observation
    assert result.local_revision_after_observation
    assert result.termination_status == "success"
    assert result.termination_outcome == "preserved"
    assert result.wrong_confirmations == 0
    assert result.trace_support_counted_as_proof is False

    assert result.experiment is not None
    assert result.experiment.action.name == "ACTION6"
    assert result.experiment.prediction_families == ("relation", "relation")
    assert result.experiment.predicted_outcomes == ("preserved", "broken")
    assert result.experiment.has_divergent_predictions

    assert any(
        revision.key == "relation::ACTION6::same_shape::colors9_8::preserved"
        and revision.status_after == HypothesisStatus.CONFIRMED
        for revision in result.revisions
    )
    assert any(
        revision.key == "relation::ACTION6::same_shape::colors9_8::broken"
        and revision.status_after == HypothesisStatus.REFUTED
        for revision in result.revisions
    )


def test_avoid_relation_option_terminates_on_absent_relation():
    option = AvoidRelationOption("same_shape", desired_outcome="absent")
    revisions = [
        MultiFamilyHypothesisRevision(
            key="relation::ACTION3::same_shape::colors10_3::preserved",
            family="relation",
            predicate="same_shape",
            predicted_outcome="preserved",
            observed_outcome="absent",
            status_after=HypothesisStatus.REFUTED,
        )
    ]

    assert option.mode == "avoid"
    assert option.termination_outcomes == ("absent",)
    assert option.observe_termination(revisions) == "success"
