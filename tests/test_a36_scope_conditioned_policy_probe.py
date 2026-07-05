import json

import theory.a36.scope_conditioned_policy_probe as policy


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
        "known_scope": "local_context",
        "status": "confirmed",
    }


def _scope_map(scope_assessment="CONTEXTUALLY_STABLE"):
    return {
        "key": KEY,
        "game_id": "bp35-0a0ad940",
        "mechanic": {
            "action": "ACTION6",
            "mechanic_family": "position_effect_candidate",
            "predicted_metric": "local_patch_before_after",
            "known_scope_from_a33": "local_context",
        },
        "scope_assessment": scope_assessment,
        "context_probes": [
            {
                "context_id": "reset_exact",
                "context_sequence": [],
                "baseline_action": "ACTION3",
                "functional_progress": True,
                "usage_contradiction": False,
            },
            {
                "context_id": "after_ACTION3_then_ACTION4",
                "context_sequence": ["ACTION3", "ACTION4"],
                "baseline_action": "ACTION3",
                "functional_progress": True,
                "usage_contradiction": False,
            },
        ],
        "truth_status": "NOT_REEVALUATED_BY_A35",
    }


def _write_inputs(tmp_path, *, scope_assessment="CONTEXTUALLY_STABLE"):
    registry_path = tmp_path / "confirmed_mechanics_registry.json"
    scope_path = tmp_path / "confirmed_mechanic_scope_map.json"
    registry_path.write_text(
        json.dumps({"confirmed_mechanics": [_registry_entry()]}),
        encoding="utf-8",
    )
    scope_path.write_text(
        json.dumps({"scope_maps": [_scope_map(scope_assessment)]}),
        encoding="utf-8",
    )
    return registry_path, scope_path


def test_policy_prioritizes_confirmed_mechanic_when_stable_scope_matches(monkeypatch, tmp_path):
    registry_path, scope_path = _write_inputs(tmp_path)

    def fake_execute(game_id, context_sequence, action_name, predicted_metric, *, environments_dir, metric_action_args=None):
        signal = 1 if action_name == "ACTION6" else 0
        return {
            "action": action_name,
            "context_sequence": list(context_sequence),
            "metric_action_args": {"x": 1, "y": 2},
            "measurement": {
                "metric": predicted_metric,
                "changed": bool(signal),
                "local_changed_pixels": signal,
            },
            "levels_before": 0,
            "levels_after": 0,
            "game_state_after": "NOT_FINISHED",
            "env_actions": len(context_sequence) + 1,
            "error": "",
        }

    monkeypatch.setattr(policy, "execute_contextual_action_measurement", fake_execute)

    payload = policy.run_scope_conditioned_policy_probe(
        registry_path=registry_path,
        scope_map_path=scope_path,
        live_context_sequence=("ACTION3", "ACTION4"),
        environments_dir=tmp_path,
    )
    result = payload["policy_probe"]

    assert result["policy_selected_action"] == "ACTION6"
    assert result["fallback_action"] == "ACTION3"
    assert result["selected_from_confirmed_mechanic"] is True
    assert result["scope_used"] == "CONTEXTUALLY_STABLE"
    assert result["context_match"] is True
    assert result["context_match_reason"] == "covered_context_exact"
    assert result["functional_progress"] is True
    assert result["useful_new_state"] is True
    assert result["truth_status"] == "NOT_REEVALUATED_BY_A36"
    assert result["wrong_confirmations"] == 0


def test_policy_falls_back_when_scope_is_not_stable(monkeypatch, tmp_path):
    registry_path, scope_path = _write_inputs(
        tmp_path,
        scope_assessment="PRECONDITION_DEPENDENT",
    )

    def fake_execute(game_id, context_sequence, action_name, predicted_metric, *, environments_dir, metric_action_args=None):
        return {
            "action": action_name,
            "context_sequence": list(context_sequence),
            "metric_action_args": {"x": 1, "y": 2},
            "measurement": {
                "metric": predicted_metric,
                "changed": False,
                "local_changed_pixels": 0,
            },
            "levels_before": 0,
            "levels_after": 0,
            "game_state_after": "NOT_FINISHED",
            "env_actions": len(context_sequence) + 1,
            "error": "",
        }

    monkeypatch.setattr(policy, "execute_contextual_action_measurement", fake_execute)

    payload = policy.run_scope_conditioned_policy_probe(
        registry_path=registry_path,
        scope_map_path=scope_path,
        live_context_sequence=("ACTION3", "ACTION4"),
        environments_dir=tmp_path,
    )
    result = payload["policy_probe"]

    assert result["policy_selected_action"] == "ACTION3"
    assert result["selected_from_confirmed_mechanic"] is False
    assert result["context_match"] is False
    assert result["functional_progress"] is False
    assert result["truth_status"] == "NOT_REEVALUATED_BY_A36"


def test_stable_scope_allows_short_neighbor_contexts():
    matched, reason = policy.context_matches_scope(
        _scope_map("CONTEXTUALLY_STABLE"),
        live_context_sequence=("ACTION3", "ACTION4", "ACTION3"),
    )

    assert matched is True
    assert reason == "neighbor_of_contextually_stable_scope"
