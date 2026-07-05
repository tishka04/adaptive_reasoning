import json

import theory.a40.frontier_handoff_requests as handoff


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


def _successful_step():
    return {
        "step": 0,
        "game_id": "bp35-0a0ad940",
        "key": KEY,
        "context_id": "reset_exact",
        "context_signature": [],
        "context_match": True,
        "selected_from_confirmed_mechanic": True,
        "policy_selected_action": "ACTION6",
        "functional_progress": True,
        "useful_new_state": True,
        "usage_contradiction": False,
        "state_signature_before": "state:0",
        "state_signature_after": "state:1",
        "selected_signal": 1.0,
    }


def _uncovered_fallback_step():
    return {
        "step": 1,
        "game_id": "bp35-0a0ad940",
        "key": KEY,
        "context_id": "after_ACTION6",
        "context_signature": ["ACTION6"],
        "context_match": False,
        "selected_from_confirmed_mechanic": False,
        "policy_selected_action": "ACTION3",
        "functional_progress": False,
        "useful_new_state": False,
        "state_signature_before": "state:1",
        "state_signature_after": "state:2",
        "selected_signal": 0.0,
    }


def _blocked_precondition_step():
    return {
        "step": 2,
        "game_id": "bp35-0a0ad940",
        "key": KEY,
        "context_id": "after_ACTION3",
        "context_signature": ["ACTION3"],
        "context_match": True,
        "selected_from_confirmed_mechanic": False,
        "blocked_confirmed_mechanic": True,
        "blocked_action": "ACTION6",
        "blocked_context_id": "after_ACTION3_live_after_ACTION6",
        "failed_precondition": "target_patch_not_already_saturated=true",
        "fallback_due_to_failed_precondition": True,
        "policy_selected_action": "ACTION4",
        "functional_progress": False,
        "useful_new_state": False,
        "state_signature_before": "state:2",
        "state_signature_after": "state:3",
        "selected_signal": 0.0,
    }


def test_frontier_requests_capture_uncovered_and_blocked_frontiers():
    requests = handoff.build_frontier_handoff_requests(
        (_successful_step(), _uncovered_fallback_step(), _blocked_precondition_step())
    )

    assert len(requests) == 2
    uncovered, blocked = requests
    assert uncovered.reason == "context_not_covered_by_scope"
    assert uncovered.frontier_context_id == "after_ACTION6"
    assert "uncovered_context" in uncovered.reason_categories
    assert "fallback_no_progress" in uncovered.reason_categories
    assert blocked.reason == "confirmed_skill_blocked_by_failed_precondition"
    assert blocked.frontier_context_id == "after_ACTION3_live_after_ACTION6"
    assert blocked.blocked_skill == "ACTION6"
    assert blocked.failed_precondition == "target_patch_not_already_saturated=true"
    assert blocked.recommended_next_scientific_action == (
        "generate_new_candidate_mechanics_from_current_state"
    )
    assert blocked.truth_status == "NOT_REEVALUATED_BY_A40"
    assert blocked.revision_performed is False
    assert blocked.wrong_confirmations == 0


def test_run_frontier_handoff_summarizes_kpis_without_revision(tmp_path):
    rollout_path = tmp_path / "precondition_aware_policy_rollout.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollout_steps": [
                    _successful_step(),
                    _uncovered_fallback_step(),
                    _blocked_precondition_step(),
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = handoff.run_frontier_handoff_requests(
        policy_rollout_path=rollout_path,
    )

    assert payload["summary"]["frontier_requests_created"] == 2
    assert payload["summary"]["blocked_skill_frontiers"] == 1
    assert payload["summary"]["uncovered_context_frontiers"] == 1
    assert payload["summary"]["fallback_no_progress_frontiers"] == 2
    assert payload["summary"]["ready_for_m1_or_m3"] is True
    assert payload["truth_status"] == "NOT_REEVALUATED_BY_A40"
    assert payload["revision_performed"] is False
    assert payload["wrong_confirmations"] == 0


def test_no_frontier_requests_when_rollout_has_only_successes():
    requests = handoff.build_frontier_handoff_requests((_successful_step(),))
    summary = handoff.summarize_frontier_requests(requests)

    assert requests == ()
    assert summary["frontier_requests_created"] == 0
    assert summary["ready_for_m1_or_m3"] is False
