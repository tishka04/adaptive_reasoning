from theory.m1.experiment_value_estimator import (
    estimate_experiment_values,
    mark_recommended_estimates,
    summarize_experiment_value_estimates,
)


def _row(candidate_id, candidate_type, action="ACTION1", game_id="bp35"):
    return {
        "candidate_id": candidate_id,
        "game_id": game_id,
        "candidate_type": candidate_type,
        "action": action,
        "required_observation": "object_positions_before_after",
        "testability_status": "testable",
        "status": "UNRESOLVED",
        "available_live_affordance": {"live_non_background_object_count": 3},
    }


def test_observed_delta_drives_delta_score_and_keeps_unresolved():
    rows = [
        _row("m1e0000:bp35:ACTION1:object_motion_candidate", "object_motion_candidate"),
        _row(
            "m1e0001:bp35:ACTION2:object_lifecycle_candidate",
            "object_lifecycle_candidate",
            action="ACTION2",
        ),
    ]
    observed = {
        rows[0]["candidate_id"]: {
            "changed_pixels": 50,
            "measured_delta": {
                "changed_cell_ratio": 0.05,
                "metric": "object_positions_before_after",
            },
        },
        rows[1]["candidate_id"]: {
            "changed_pixels": 10,
            "measured_delta": {
                "changed_cell_ratio": 0.01,
                "metric": "object_counts_before_after",
            },
        },
    }

    estimates = estimate_experiment_values(rows, observed_by_id=observed)

    assert estimates[0].candidate_id == rows[0]["candidate_id"]
    assert estimates[0].delta_score > estimates[1].delta_score
    assert estimates[0].score_basis == "observed_delta"
    assert all(estimate.status == "UNRESOLVED" for estimate in estimates)
    assert all(estimate.trace_support_counted_as_proof is False for estimate in estimates)


def test_recommendations_prefer_diverse_types_when_scores_are_close():
    rows = [
        _row("m1e0000:bp35:ACTION1:object_motion_candidate", "object_motion_candidate"),
        _row("m1e0001:bp35:ACTION1:object_motion_candidate", "object_motion_candidate"),
        _row(
            "m1e0002:bp35:ACTION1:contact_change_candidate",
            "contact_change_candidate",
        ),
    ]
    support = {
        "m1e0000": {
            "support_rate": 1.0,
            "evidence": {"mean_motion_vectors": 5},
        },
        "m1e0001": {
            "support_rate": 0.95,
            "evidence": {"mean_motion_vectors": 5},
        },
        "m1e0002": {
            "support_rate": 1.0,
            "evidence": {"dominant_changed_contact_pairs": [{"count": 10}]},
        },
    }

    estimates = estimate_experiment_values(rows, support_by_id=support)
    marked = mark_recommended_estimates(estimates, max_recommended=2)
    recommended_types = {
        estimate.candidate_type for estimate in marked if estimate.recommended
    }

    assert recommended_types == {
        "object_motion_candidate",
        "contact_change_candidate",
    }


def test_summarize_experiment_value_estimates_reports_kpis():
    rows = [
        _row("m1e0000:bp35:ACTION1:object_motion_candidate", "object_motion_candidate"),
        _row(
            "m1e0001:bp35:ACTION2:object_lifecycle_candidate",
            "object_lifecycle_candidate",
            action="ACTION2",
        ),
    ]
    estimates = mark_recommended_estimates(
        estimate_experiment_values(rows),
        max_recommended=2,
    )

    summary = summarize_experiment_value_estimates(estimates)

    assert summary["candidates_total"] == 2
    assert summary["recommended_experiments"] == 2
    assert summary["mean_information_score"] > 0
    assert summary["type_diversity"] == 1.0
    assert summary["wrong_confirmations"] == 0
