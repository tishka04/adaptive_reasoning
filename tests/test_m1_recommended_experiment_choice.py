from theory.m1.polymorphic_a25_adapter import PolymorphicMechanicExperiment
from theory.m1.recommended_experiment_choice import (
    mechanic_hypothesis_from_recommendation,
    mechanic_observation_from_experiment,
    polymorphic_choice_from_hypothesis,
    select_recommendation,
    summarize_recommended_choice,
)


def _recommendation(candidate_id, *, score, game_id="bp35", candidate_type="position_effect_candidate"):
    return {
        "candidate_id": candidate_id,
        "game_id": game_id,
        "candidate_type": candidate_type,
        "action": "ACTION6",
        "required_observation": "local_patch_before_after",
        "expected_state_change": "local_patch_before_after: estimated before/after delta",
        "expected_information_gain": score,
        "score": score,
        "status": "UNRESOLVED",
    }


def test_select_recommendation_picks_highest_score_with_filters():
    rows = [
        _recommendation("low", score=0.2),
        _recommendation("high", score=0.9),
        _recommendation("other", score=1.0, game_id="cd82"),
    ]

    selected = select_recommendation(rows, game_id="bp35")

    assert selected["candidate_id"] == "high"


def test_recommendation_becomes_unresolved_mechanic_choice():
    recommendation = _recommendation("best", score=0.8636)

    hypothesis = mechanic_hypothesis_from_recommendation(recommendation)
    choice = polymorphic_choice_from_hypothesis(hypothesis)

    assert hypothesis.mechanic_family == "position_effect_candidate"
    assert hypothesis.predicted_metric == "local_patch_before_after"
    assert hypothesis.status == "UNRESOLVED"
    assert hypothesis.trace_support_counted_as_proof is False
    assert choice.a25_choice_type == "polymorphic_mechanic_experiment"
    assert choice.competing_keys == ("best",)
    assert choice.status == "UNRESOLVED"


def test_executed_experiment_becomes_mechanic_observation_without_revision():
    recommendation = _recommendation("best", score=0.8636)
    hypothesis = mechanic_hypothesis_from_recommendation(recommendation)
    experiment = PolymorphicMechanicExperiment(
        candidate_id="best",
        game_id="bp35",
        candidate_type="position_effect_candidate",
        action="ACTION6",
        required_observation="local_patch_before_after",
        mechanic_experiment_generated=True,
        env_actions=1,
        measured_delta={"changed": True, "metric": "local_patch_before_after"},
        changed_pixels=12,
    )

    observation = mechanic_observation_from_experiment(hypothesis, experiment)
    summary = summarize_recommended_choice(
        choice=polymorphic_choice_from_hypothesis(hypothesis),
        observation=observation,
    )

    assert observation.status == "UNRESOLVED"
    assert observation.env_actions == 1
    assert observation.revision_performed is False
    assert observation.wrong_confirmations == 0
    assert summary["a25_polymorphic_choices"] == 1
    assert summary["observable_delta"] is True
    assert summary["wrong_confirmations"] == 0
