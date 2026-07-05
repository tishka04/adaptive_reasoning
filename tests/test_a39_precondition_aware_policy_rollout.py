import json
from dataclasses import dataclass

import numpy as np

import theory.a39.precondition_aware_policy_rollout as policy


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)
SATURATED_PATCH = [[5, 5, 5], [3, 5, 10]]


@dataclass(frozen=True)
class _Action:
    name: str = "ACTION6"
    action_args: dict | None = None


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
        "context_probes": [
            {"context_sequence": [], "error": ""},
            {"context_sequence": ["ACTION3"], "error": ""},
            {"context_sequence": ["ACTION3", "ACTION4"], "error": ""},
        ],
    }


def _refinement():
    return {
        "key": KEY,
        "refined_scope_assessment": "CONTEXTUALLY_STABLE_WITH_PRECONDITIONS",
        "usage_preconditions": [
            "local_patch_available=true",
            "predicted_metric_signal_available=true",
            "selected_signal_expected=1",
            "target_patch_not_already_saturated=true",
        ],
        "positive_usage_contexts": [
            {
                "context_id": "reset_exact",
                "local_patch_after": SATURATED_PATCH,
            }
        ],
        "negative_usage_contexts": [
            {
                "blocked_context_id": "after_ACTION3_live_after_ACTION6",
                "context_id": "after_ACTION3",
                "target_patch_already_saturated": True,
                "local_patch_before": SATURATED_PATCH,
                "local_patch_after": SATURATED_PATCH,
            }
        ],
        "blocked_context_details": [
            {
                "blocked_context_id": "after_ACTION3_live_after_ACTION6",
                "context_id": "after_ACTION3",
                "reason": "target_patch_already_saturated",
            }
        ],
    }


def _decision():
    return policy.ScopeConditionedRolloutDecision(
        key=KEY,
        action="ACTION6",
        fallback_action="ACTION4",
        predicted_metric="local_patch_before_after",
        selected_from_confirmed_mechanic=True,
        scope_used="CONTEXTUALLY_STABLE",
        context_signature=("ACTION3",),
        context_match=True,
        context_match_reason="covered_context_exact",
        decision_reason="covered_scope_precondition_check_required",
    )


def test_precondition_check_blocks_saturated_patch():
    grid = np.asarray(SATURATED_PATCH, dtype=np.int32)
    action = _Action(action_args={"x": 1, "y": 0})

    check = policy.check_usage_preconditions(
        grid,
        action,
        decision=_decision(),
        refinement=_refinement(),
        action_history=("ACTION6", "ACTION3"),
    )

    assert check.satisfied is False
    assert check.failed_precondition == "target_patch_not_already_saturated=true"
    assert check.target_patch_already_saturated is True
    assert check.blocked_context_id == "after_ACTION3_live_after_ACTION6"


def test_precondition_check_allows_unsaturated_patch():
    grid = np.asarray([[5, 5, 5], [3, 5, 3]], dtype=np.int32)
    action = _Action(action_args={"x": 1, "y": 0})

    check = policy.check_usage_preconditions(
        grid,
        action,
        decision=_decision(),
        refinement=_refinement(),
        action_history=(),
    )

    assert check.satisfied is True
    assert check.target_patch_not_already_saturated is True
    assert check.failed_precondition == ""


def test_run_policy_rollout_summarizes_precondition_inhibition(monkeypatch, tmp_path):
    registry_path = tmp_path / "confirmed_mechanics_registry.json"
    scope_path = tmp_path / "confirmed_mechanic_scope_map.json"
    refinement_path = tmp_path / "rollout_aware_scope_refinement.json"
    registry_path.write_text(
        json.dumps({"confirmed_mechanics": [_registry_entry()]}),
        encoding="utf-8",
    )
    scope_path.write_text(
        json.dumps({"scope_maps": [_scope_map()]}),
        encoding="utf-8",
    )
    refinement_path.write_text(
        json.dumps({"scope_refinements": [_refinement()]}),
        encoding="utf-8",
    )
    fake_steps = (
        policy.PreconditionAwarePolicyRolloutStep(
            step=0,
            key=KEY,
            game_id="bp35-0a0ad940",
            context_signature=(),
            context_id="reset_exact",
            policy_selected_action="ACTION6",
            fallback_action="ACTION3",
            predicted_metric="local_patch_before_after",
            selected_from_confirmed_mechanic=True,
            blocked_confirmed_mechanic=False,
            blocked_action="",
            scope_used="CONTEXTUALLY_STABLE",
            refined_scope_used="CONTEXTUALLY_STABLE_WITH_PRECONDITIONS",
            context_match=True,
            context_match_reason="covered_context_exact",
            decision_reason="covered_scope_precondition_check_required",
            precondition_status="SATISFIED",
            selected_signal=1.0,
            functional_progress=True,
            useful_new_state=True,
        ),
        policy.PreconditionAwarePolicyRolloutStep(
            step=1,
            key=KEY,
            game_id="bp35-0a0ad940",
            context_signature=("ACTION6",),
            context_id="after_ACTION6",
            policy_selected_action="ACTION3",
            fallback_action="ACTION3",
            predicted_metric="local_patch_before_after",
            selected_from_confirmed_mechanic=False,
            blocked_confirmed_mechanic=False,
            blocked_action="",
            scope_used="CONTEXTUALLY_STABLE",
            refined_scope_used="CONTEXTUALLY_STABLE_WITH_PRECONDITIONS",
            context_match=False,
            context_match_reason="no_covered_rollout_context",
            decision_reason="fallback_neutral_exploration",
            precondition_status="NOT_APPLICABLE",
        ),
        policy.PreconditionAwarePolicyRolloutStep(
            step=2,
            key=KEY,
            game_id="bp35-0a0ad940",
            context_signature=("ACTION3",),
            context_id="after_ACTION3",
            policy_selected_action="ACTION4",
            fallback_action="ACTION4",
            predicted_metric="local_patch_before_after",
            selected_from_confirmed_mechanic=False,
            blocked_confirmed_mechanic=True,
            blocked_action="ACTION6",
            scope_used="CONTEXTUALLY_STABLE",
            refined_scope_used="CONTEXTUALLY_STABLE_WITH_PRECONDITIONS",
            context_match=True,
            context_match_reason="covered_context_exact",
            decision_reason="failed_precondition_fallback_executed",
            precondition_status="FAILED",
            failed_precondition="target_patch_not_already_saturated=true",
            blocked_context_id="after_ACTION3_live_after_ACTION6",
            avoided_saturated_reuse=True,
            fallback_due_to_failed_precondition=True,
        ),
    )

    monkeypatch.setattr(
        policy,
        "execute_precondition_aware_rollout",
        lambda registry_entries, *, scopes_by_key, refinements_by_key, environments_dir, budget, baseline_order: fake_steps,
    )

    payload = policy.run_precondition_aware_policy_rollout(
        registry_path=registry_path,
        scope_map_path=scope_path,
        refinement_path=refinement_path,
        environments_dir=tmp_path,
        budget=3,
    )

    assert payload["summary"]["policy_steps"] == 3
    assert payload["summary"]["policy_steps_from_confirmed_mechanic"] == 1
    assert payload["summary"]["fallback_due_to_failed_precondition"] == 1
    assert payload["summary"]["avoided_saturated_reuse"] == 1
    assert payload["summary"]["usage_contradictions"] == 0
    assert payload["truth_status"] == "NOT_REEVALUATED_BY_A39"
    assert payload["revision_performed"] is False
    assert payload["wrong_confirmations"] == 0
