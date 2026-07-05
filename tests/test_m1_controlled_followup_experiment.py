from theory.m1.controlled_followup_experiment import (
    ControlledExperiment,
    controlled_delta,
    ledger_spec_from_entry,
    metric_signal,
    summarize_controlled_experiments,
    support_contradiction_from_delta,
    updated_ledger_entry_from_experiment,
)


def _entry():
    return {
        "key": "mechanic_prediction::bp35::ACTION6::position_effect_candidate::local_patch_before_after",
        "game_id": "bp35",
        "description": "ACTION6 position_effect_candidate via local_patch_before_after",
        "status": "unresolved",
        "controlled_test_required": True,
    }


def test_ledger_spec_from_entry_parses_mechanic_key():
    spec = ledger_spec_from_entry(_entry())

    assert spec == {
        "hypothesis_key": _entry()["key"],
        "game_id": "bp35",
        "target_action": "ACTION6",
        "mechanic_family": "position_effect_candidate",
        "predicted_metric": "local_patch_before_after",
    }


def test_metric_signal_is_metric_specific():
    assert metric_signal({"local_changed_pixels": 2}, "local_patch_before_after") == 2
    assert metric_signal({"object_count_delta": -3}, "object_counts_before_after") == 3
    assert (
        metric_signal(
            {"contact_pairs_added": [[1, 2]], "contact_pairs_removed": [[2, 3]]},
            "contact_graph_before_after",
        )
        == 2
    )
    assert metric_signal({"moved_component_count": 4}, "object_positions_before_after") == 4


def test_controlled_delta_produces_support_and_contradiction_events():
    support_delta = controlled_delta(
        {"local_changed_pixels": 0},
        {"local_changed_pixels": 1},
        predicted_metric="local_patch_before_after",
    )
    contradiction_delta = controlled_delta(
        {"local_changed_pixels": 2},
        {"local_changed_pixels": 1},
        predicted_metric="local_patch_before_after",
    )

    assert support_delta["effect_size"] == 1
    assert support_contradiction_from_delta(support_delta) == (1, 0)
    assert contradiction_delta["effect_size"] == -1
    assert support_contradiction_from_delta(contradiction_delta) == (0, 1)


def test_updated_ledger_entry_keeps_status_unresolved_with_event_counts():
    experiment = ControlledExperiment(
        hypothesis_key=_entry()["key"],
        game_id="bp35",
        mechanic_family="position_effect_candidate",
        target_action="ACTION6",
        control_action="ACTION3",
        baseline_sequence=("RESET", "ACTION3"),
        perturbation_sequence=("RESET", "ACTION6"),
        predicted_metric="local_patch_before_after",
        delta={"effect_size": 1},
        support_events=1,
        contradiction_events=0,
        env_actions=2,
        controlled_experiments_run=1,
    )

    updated = updated_ledger_entry_from_experiment(experiment)
    summary = summarize_controlled_experiments([experiment])

    assert updated["status"] == "UNRESOLVED"
    assert updated["support"] == 0
    assert updated["support_events"] == 1
    assert updated["controlled_experiments_run"] == 1
    assert updated["observation_counted_as_confirmation"] is False
    assert summary["controlled_experiments_run"] == 1
    assert summary["support_events"] == 1
    assert summary["wrong_confirmations"] == 0
