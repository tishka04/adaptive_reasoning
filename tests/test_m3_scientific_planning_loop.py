import json

import theory.m3.scientific_planning_loop as loop
from theory.m1.controlled_followup_experiment import ControlledExperiment


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


def _ledger_payload():
    return {
        "ledger_entries": [
            {
                "key": KEY,
                "game_id": "bp35-0a0ad940",
                "description": "ACTION6 position_effect_candidate via local_patch_before_after",
                "status": "unresolved",
                "support": 0,
                "contradictions": 0,
                "controlled_test_required": True,
            }
        ]
    }


def _prior_controlled_payload():
    return {
        "controlled_experiments": [
            {
                "hypothesis_key": KEY,
                "game_id": "bp35-0a0ad940",
                "mechanic_family": "position_effect_candidate",
                "target_action": "ACTION6",
                "control_action": "ACTION3",
                "predicted_metric": "local_patch_before_after",
                "support_events": 1,
                "contradiction_events": 0,
                "controlled_experiments_run": 1,
                "status": "UNRESOLVED",
                "support": 0,
            }
        ]
    }


def test_scientific_planning_loop_reaches_three_total_candidate_only_experiments(
    monkeypatch,
    tmp_path,
):
    ledger_path = tmp_path / "scientific_integration.json"
    prior_path = tmp_path / "controlled_results.json"
    ledger_path.write_text(json.dumps(_ledger_payload()), encoding="utf-8")
    prior_path.write_text(json.dumps(_prior_controlled_payload()), encoding="utf-8")

    monkeypatch.setattr(
        loop,
        "load_live_available_action_names",
        lambda game_id, environments_dir: ("ACTION3", "ACTION4", "ACTION6"),
    )

    def fake_execute_controlled_followup(
        ledger_entry,
        *,
        environments_dir,
        control_actions,
    ):
        control = tuple(control_actions)[0]
        return ControlledExperiment(
            hypothesis_key=KEY,
            game_id="bp35-0a0ad940",
            mechanic_family="position_effect_candidate",
            target_action="ACTION6",
            control_action=control,
            baseline_sequence=("RESET", control),
            perturbation_sequence=("RESET", "ACTION6"),
            predicted_metric="local_patch_before_after",
            delta={"effect_size": 1.0, "direction": "support"},
            support_events=1,
            contradiction_events=0,
            env_actions=2,
            controlled_experiments_run=1,
        )

    monkeypatch.setattr(
        loop,
        "execute_controlled_followup",
        fake_execute_controlled_followup,
    )

    payload = loop.run_scientific_planning_loop(
        scientific_integration_path=ledger_path,
        controlled_results_paths=(prior_path,),
        environments_dir=tmp_path,
        game_id="bp35-0a0ad940",
        budget=3,
    )

    summary = payload["summary"]
    updated = payload["updated_ledger_entries"][0]

    assert summary["controlled_experiments_run"] == 3
    assert summary["support_events"] == 3
    assert summary["independent_support_events"] == 2
    assert summary["reused_control_support_events"] == 1
    assert summary["contradiction_events"] == 0
    assert summary["propose_ready_for_A15_revision"] is True
    assert summary["revision_status"] == "CANDIDATE_ONLY"
    assert summary["support"] == 0
    assert payload["controlled_experiments"][-1]["control_reuse_reason"] == (
        "no_unused_distinct_control_available"
    )
    assert updated["status"] == "UNRESOLVED"
    assert updated["revision_status"] == "CANDIDATE_ONLY"
    assert updated["support"] == 0
    assert updated["controlled_test_required"] is True
    assert payload["wrong_confirmations"] == 0
