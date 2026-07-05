from theory.m1.mechanic_revision_candidate import (
    observed_outcome_from_delta,
    prediction_candidate_from_observation,
    revision_candidate_from_prediction,
    summarize_revision_candidates,
)


def _observation():
    return {
        "candidate_id": "m1e0015:bp35:ACTION6:position_effect_candidate",
        "game_id": "bp35",
        "mechanic_family": "position_effect_candidate",
        "action": "ACTION6",
        "predicted_metric": "local_patch_before_after",
        "expected_outcome": "local_patch_before_after: estimated before/after delta",
        "observed_delta": {
            "changed": True,
            "changed_pixels": 26,
            "changed_cell_ratio": 0.006348,
            "metric": "local_patch_before_after",
            "local_changed_pixels": 1,
            "patch_bbox": [0, 17, 1, 19],
        },
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def test_observed_outcome_from_delta_is_metric_specific():
    assert (
        observed_outcome_from_delta(
            "local_patch_before_after",
            {"local_changed_pixels": 1},
        )
        == "local_patch_changed"
    )
    assert (
        observed_outcome_from_delta(
            "object_counts_before_after",
            {"object_count_delta": -3},
        )
        == "object_count_decreased"
    )
    assert (
        observed_outcome_from_delta(
            "contact_graph_before_after",
            {"contact_pairs_added": [[0, 15]], "contact_pairs_removed": []},
        )
        == "contact_graph_changed"
    )


def test_prediction_candidate_from_observation_remains_unresolved():
    prediction = prediction_candidate_from_observation(_observation())
    payload = prediction.to_dict()

    assert prediction.observed_outcome == "local_patch_changed"
    assert prediction.status == "UNRESOLVED"
    assert prediction.controlled_test_required is True
    assert prediction.trace_support_counted_as_proof is False
    assert prediction.prior_counted_as_proof is False
    assert prediction.observation_counted_as_confirmation is False
    assert payload["observed_delta_summary"]["local_changed_pixels"] == 1


def test_revision_candidate_proposes_unresolved_a15_a31_record():
    prediction = prediction_candidate_from_observation(_observation())
    revision = revision_candidate_from_prediction(prediction)
    payload = revision.to_dict()
    proposal = payload["a15_a31_revision_proposal"]

    assert revision.status == "UNRESOLVED"
    assert revision.revision_performed is False
    assert revision.wrong_confirmations == 0
    assert proposal["proposed_status"] == "UNRESOLVED"
    assert proposal["support"] == 0
    assert proposal["contradictions"] == 0
    assert proposal["controlled_test_required"] is True
    assert proposal["observation_counted_as_confirmation"] is False


def test_summarize_revision_candidates_reports_guardrails():
    revision = revision_candidate_from_prediction(
        prediction_candidate_from_observation(_observation())
    )

    summary = summarize_revision_candidates([revision])

    assert summary["mechanic_predictions"] == 1
    assert summary["revision_candidates"] == 1
    assert summary["a15_a31_revision_proposals"] == 1
    assert summary["controlled_tests_required"] == 1
    assert summary["revision_performed"] is False
    assert summary["observation_counted_as_confirmation"] is False
    assert summary["wrong_confirmations"] == 0
