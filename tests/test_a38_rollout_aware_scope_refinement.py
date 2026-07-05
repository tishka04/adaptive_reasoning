import json

import theory.a38.rollout_aware_scope_refinement as refine


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


POSITIVE_BEFORE = [[5, 5, 5], [3, 5, 3]]
POSITIVE_AFTER = [[5, 5, 5], [3, 5, 10]]


def _registry_entry():
    return {
        "key": KEY,
        "game_id": "bp35-0a0ad940",
        "action": "ACTION6",
        "mechanic_family": "position_effect_candidate",
        "predicted_metric": "local_patch_before_after",
        "status": "confirmed",
    }


def _scope_map():
    return {
        "key": KEY,
        "scope_assessment": "CONTEXTUALLY_STABLE",
        "truth_status": "NOT_REEVALUATED_BY_A35",
    }


def _positive_step():
    return {
        "step": 0,
        "key": KEY,
        "context_id": "reset_exact",
        "context_signature": [],
        "policy_selected_action": "ACTION6",
        "selected_from_confirmed_mechanic": True,
        "selected_signal": 1.0,
        "functional_progress": True,
        "useful_new_state": True,
        "usage_contradiction": False,
        "measurement": {
            "local_patch_available": True,
            "local_changed_pixels": 1,
            "local_patch_before": POSITIVE_BEFORE,
            "local_patch_after": POSITIVE_AFTER,
            "patch_bbox": [0, 17, 1, 19],
        },
    }


def _fallback_step():
    return {
        "step": 1,
        "key": KEY,
        "context_id": "after_ACTION6",
        "context_signature": ["ACTION6"],
        "policy_selected_action": "ACTION3",
        "selected_from_confirmed_mechanic": False,
        "selected_signal": 0.0,
        "functional_progress": False,
        "useful_new_state": False,
        "usage_contradiction": False,
        "measurement": {"local_patch_available": False},
    }


def _negative_step():
    return {
        "step": 2,
        "key": KEY,
        "context_id": "after_ACTION3",
        "context_signature": ["ACTION3"],
        "policy_selected_action": "ACTION6",
        "selected_from_confirmed_mechanic": True,
        "selected_signal": 0.0,
        "functional_progress": False,
        "useful_new_state": False,
        "usage_contradiction": True,
        "measurement": {
            "local_patch_available": True,
            "local_changed_pixels": 0,
            "local_patch_before": POSITIVE_AFTER,
            "local_patch_after": POSITIVE_AFTER,
            "patch_bbox": [0, 17, 1, 19],
        },
    }


def test_refinement_extracts_saturation_precondition():
    refinement = refine.build_rollout_aware_scope_refinement(
        _registry_entry(),
        scope_map=_scope_map(),
        rollout_steps=(_positive_step(), _fallback_step(), _negative_step()),
    )

    assert refinement.refined_scope_assessment == (
        "CONTEXTUALLY_STABLE_WITH_PRECONDITIONS"
    )
    assert refinement.positive_usage_contexts[0]["context_id"] == "reset_exact"
    assert refinement.negative_usage_contexts[0]["context_id"] == "after_ACTION3"
    assert refinement.negative_usage_contexts[0]["target_patch_already_saturated"] is True
    assert refinement.blocked_contexts == ("after_ACTION3_live_after_ACTION6",)
    assert "target_patch_not_already_saturated=true" in refinement.usage_preconditions
    assert "effect_saturation_detected" in refinement.refinement_notes
    assert refinement.truth_status == "NOT_REEVALUATED_BY_A38"
    assert refinement.revision_performed is False
    assert refinement.wrong_confirmations == 0


def test_run_refinement_summarizes_without_revising(tmp_path):
    registry_path = tmp_path / "confirmed_mechanics_registry.json"
    scope_path = tmp_path / "confirmed_mechanic_scope_map.json"
    rollout_path = tmp_path / "scope_conditioned_policy_rollout.json"
    registry_path.write_text(
        json.dumps({"confirmed_mechanics": [_registry_entry()]}),
        encoding="utf-8",
    )
    scope_path.write_text(
        json.dumps({"scope_maps": [_scope_map()]}),
        encoding="utf-8",
    )
    rollout_path.write_text(
        json.dumps(
            {
                "rollout_steps": [
                    _positive_step(),
                    _fallback_step(),
                    _negative_step(),
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = refine.run_rollout_aware_scope_refinement(
        registry_path=registry_path,
        scope_map_path=scope_path,
        rollout_path=rollout_path,
    )

    assert payload["summary"]["mechanics_refined"] == 1
    assert payload["summary"]["refinements_with_preconditions"] == 1
    assert payload["summary"]["positive_usage_contexts"] == 1
    assert payload["summary"]["negative_usage_contexts"] == 1
    assert payload["summary"]["blocked_contexts"] == 1
    assert payload["summary"]["effect_saturation_detected"] is True
    assert payload["truth_status"] == "NOT_REEVALUATED_BY_A38"
    assert payload["revision_performed"] is False
    assert payload["wrong_confirmations"] == 0


def test_refinement_keeps_stable_scope_when_no_negative_usage():
    refinement = refine.build_rollout_aware_scope_refinement(
        _registry_entry(),
        scope_map=_scope_map(),
        rollout_steps=(_positive_step(),),
    )

    assert refinement.refined_scope_assessment == "CONTEXTUALLY_STABLE"
    assert refinement.blocked_contexts == ()
    assert "target_patch_not_already_saturated=true" not in refinement.usage_preconditions
