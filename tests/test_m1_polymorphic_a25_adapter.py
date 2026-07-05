import numpy as np

from theory.m1.polymorphic_a25_adapter import (
    ConcretePolymorphicAction,
    PolymorphicMechanicExperiment,
    measure_required_observation,
    select_testable_candidates,
    summarize_polymorphic_adapter_results,
)


def _row(game_id, action, candidate_type, *, testable=True, candidate_id="cand"):
    return {
        "candidate_id": candidate_id,
        "game_id": game_id,
        "action": action,
        "candidate_type": candidate_type,
        "required_observation": "object_counts_before_after",
        "testability_status": "testable" if testable else "blocked",
        "status": "UNRESOLVED",
    }


def test_select_testable_candidates_preserves_explicit_spec_order():
    rows = [
        _row("bp35", "ACTION3", "object_motion_candidate", candidate_id="motion"),
        _row("bp35", "ACTION6", "object_lifecycle_candidate", candidate_id="life"),
        _row(
            "bp35",
            "ACTION6",
            "contact_change_candidate",
            testable=False,
            candidate_id="blocked",
        ),
        _row("cd82", "ACTION6", "object_motion_candidate", candidate_id="other"),
    ]

    selected = select_testable_candidates(
        rows,
        game_id="bp35",
        candidate_specs=(
            "ACTION6:object_lifecycle_candidate",
            "ACTION3:object_motion_candidate",
        ),
        max_candidates=3,
    )

    assert [row["candidate_id"] for row in selected] == ["life", "motion"]


def test_measure_object_lifecycle_delta_from_components():
    before = np.asarray(
        [
            [0, 0, 0],
            [0, 4, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    after = np.asarray(
        [
            [0, 5, 0],
            [0, 4, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    delta = measure_required_observation(
        before,
        after,
        required_observation="object_counts_before_after",
    )

    assert delta["metric"] == "object_counts_before_after"
    assert delta["object_count_before"] == 1
    assert delta["object_count_after"] == 2
    assert delta["object_count_delta"] == 1
    assert delta["object_count_delta_by_color"] == {"5": 1}
    assert delta["changed"] is True


def test_measure_contact_graph_delta():
    before = np.asarray(
        [
            [0, 0, 0],
            [0, 4, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    after = np.asarray(
        [
            [0, 0, 0],
            [0, 4, 5],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    delta = measure_required_observation(
        before,
        after,
        required_observation="contact_graph_before_after",
    )

    assert delta["metric"] == "contact_graph_before_after"
    assert delta["contact_pairs_added"] == [[4, 5]]
    assert delta["contact_graph_changed"] is True


def test_measure_object_motion_delta():
    before = np.asarray(
        [
            [0, 4, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    after = np.asarray(
        [
            [0, 0, 0],
            [0, 4, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    delta = measure_required_observation(
        before,
        after,
        required_observation="object_positions_before_after",
    )

    assert delta["metric"] == "object_positions_before_after"
    assert delta["matched_component_count"] == 1
    assert delta["moved_component_count"] == 1
    assert delta["motion_vectors"][0]["dy"] == 1.0


def test_summarize_polymorphic_adapter_results_keeps_unresolved_guardrail():
    experiments = [
        PolymorphicMechanicExperiment(
            candidate_id="life",
            game_id="bp35",
            candidate_type="object_lifecycle_candidate",
            action="ACTION6",
            required_observation="object_counts_before_after",
            concrete_action=ConcretePolymorphicAction(name="ACTION6"),
            mechanic_experiment_generated=True,
            env_actions=1,
            measured_delta={"changed": True},
            changed_pixels=3,
        ),
        PolymorphicMechanicExperiment(
            candidate_id="blocked",
            game_id="bp35",
            candidate_type="contact_change_candidate",
            action="ACTION6",
            required_observation="contact_graph_before_after",
            error="no_concrete_action_available",
        ),
    ]

    summary = summarize_polymorphic_adapter_results(experiments)

    assert summary["mechanic_candidates_consumed"] == 2
    assert summary["mechanic_experiments_generated"] == 1
    assert summary["env_actions"] == 1
    assert summary["observable_deltas"] == 1
    assert summary["wrong_confirmations"] == 0
    assert summary["revision_performed"] is False
