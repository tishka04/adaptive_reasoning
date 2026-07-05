import json

import theory.a34.confirmed_mechanic_usage_probe as probe


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


def _registry_entry():
    return {
        "key": KEY,
        "game_id": "bp35-0a0ad940",
        "action": "ACTION6",
        "mechanic_family": "position_effect_candidate",
        "predicted_metric": "local_patch_before_after",
        "confirmed_support_independent": 2,
        "experiments_spent": 3,
        "control_actions_used": ["ACTION3", "ACTION4"],
        "known_scope": "local_context",
        "status": "confirmed",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "support_reused_as_independent": False,
    }


def test_usage_probe_prioritizes_registry_action_and_reports_utility(monkeypatch, tmp_path):
    registry_path = tmp_path / "confirmed_mechanics_registry.json"
    registry_path.write_text(
        json.dumps({"confirmed_mechanics": [_registry_entry()]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        probe,
        "live_action_names",
        lambda game_id, environments_dir: ("ACTION3", "ACTION4", "ACTION6"),
    )

    def fake_execute(game_id, action_name, predicted_metric, *, environments_dir, metric_action_args=None):
        if action_name == "ACTION6":
            return {
                "metric_action_args": {"x": 1, "y": 2},
                "measurement": {
                    "metric": predicted_metric,
                    "changed": True,
                    "local_changed_pixels": 1,
                    "changed_pixels": 26,
                },
            }
        return {
            "metric_action_args": dict(metric_action_args or {}),
            "measurement": {
                "metric": predicted_metric,
                "changed": True,
                "local_changed_pixels": 0,
                "changed_pixels": 47,
            },
        }

    monkeypatch.setattr(probe, "execute_single_action_measurement", fake_execute)

    payload = probe.run_confirmed_mechanic_usage_probe(
        registry_path=registry_path,
        environments_dir=tmp_path,
    )
    result = payload["usage_probes"][0]

    assert payload["summary"]["mechanics_probed"] == 1
    assert payload["summary"]["useful"] == 1
    assert result["baseline_action"] == "ACTION3"
    assert result["treatment_action"] == "ACTION6"
    assert result["action_choice_changed"] is True
    assert result["action_prioritized_from_registry"] is True
    assert result["local_patch_before_after_observed"] is True
    assert result["useful_new_state"] is True
    assert result["functional_progress"] is True
    assert result["contradiction"] is False
    assert result["truth_status"] == "NOT_REEVALUATED_BY_A34"
    assert result["wrong_confirmations"] == 0


def test_usage_probe_can_report_not_useful_without_revising(monkeypatch, tmp_path):
    registry_path = tmp_path / "confirmed_mechanics_registry.json"
    registry_path.write_text(
        json.dumps({"confirmed_mechanics": [_registry_entry()]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        probe,
        "live_action_names",
        lambda game_id, environments_dir: ("ACTION3", "ACTION6"),
    )

    def fake_execute(game_id, action_name, predicted_metric, *, environments_dir, metric_action_args=None):
        return {
            "metric_action_args": {"x": 1, "y": 2},
            "measurement": {
                "metric": predicted_metric,
                "changed": False,
                "local_changed_pixels": 0,
                "changed_pixels": 0,
            },
        }

    monkeypatch.setattr(probe, "execute_single_action_measurement", fake_execute)

    payload = probe.run_confirmed_mechanic_usage_probe(
        registry_path=registry_path,
        environments_dir=tmp_path,
    )

    assert payload["summary"]["not_useful"] == 1
    assert payload["summary"]["truth_status"] == "NOT_REEVALUATED_BY_A34"
    assert payload["revision_performed"] is False
    assert payload["wrong_confirmations"] == 0


def test_choose_baseline_action_avoids_treatment_action():
    assert probe.choose_baseline_action(
        ("ACTION3", "ACTION4", "ACTION6"),
        treatment_action="ACTION6",
    ) == "ACTION3"
