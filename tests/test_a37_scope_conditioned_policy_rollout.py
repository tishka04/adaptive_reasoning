import json
from collections import Counter

import theory.a37.scope_conditioned_policy_rollout as rollout


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
            {"context_sequence": [], "baseline_action": "ACTION3", "error": ""},
            {"context_sequence": ["ACTION3"], "baseline_action": "ACTION3", "error": ""},
            {"context_sequence": ["ACTION4"], "baseline_action": "ACTION3", "error": ""},
            {
                "context_sequence": ["ACTION3", "ACTION4"],
                "baseline_action": "ACTION3",
                "error": "",
            },
        ],
        "truth_status": "NOT_REEVALUATED_BY_A35",
    }


def test_rollout_decision_uses_covered_suffix_after_fallback():
    decision = rollout.select_rollout_decision(
        [_registry_entry()],
        scopes_by_key={KEY: _scope_map()},
        action_history=("ACTION6", "ACTION3"),
        fallback_counts=Counter({"ACTION3": 1}),
    )

    assert decision.action == "ACTION6"
    assert decision.context_signature == ("ACTION3",)
    assert decision.selected_from_confirmed_mechanic is True
    assert decision.context_match is True
    assert decision.scope_used == "CONTEXTUALLY_STABLE"


def test_rollout_decision_falls_back_when_suffix_not_covered():
    decision = rollout.select_rollout_decision(
        [_registry_entry()],
        scopes_by_key={KEY: _scope_map()},
        action_history=("ACTION6",),
        fallback_counts=Counter(),
    )

    assert decision.action == "ACTION3"
    assert decision.selected_from_confirmed_mechanic is False
    assert decision.context_match is False
    assert decision.decision_reason == "fallback_neutral_exploration"


def test_run_rollout_summarizes_kpis_without_revising(monkeypatch, tmp_path):
    registry_path = tmp_path / "confirmed_mechanics_registry.json"
    scope_path = tmp_path / "confirmed_mechanic_scope_map.json"
    registry_path.write_text(
        json.dumps({"confirmed_mechanics": [_registry_entry()]}),
        encoding="utf-8",
    )
    scope_path.write_text(
        json.dumps({"scope_maps": [_scope_map()]}),
        encoding="utf-8",
    )

    fake_steps = (
        rollout.PolicyRolloutStep(
            step=0,
            key=KEY,
            game_id="bp35-0a0ad940",
            context_signature=(),
            context_id="reset_exact",
            policy_selected_action="ACTION6",
            fallback_action="ACTION3",
            predicted_metric="local_patch_before_after",
            selected_from_confirmed_mechanic=True,
            scope_used="CONTEXTUALLY_STABLE",
            context_match=True,
            context_match_reason="covered_context_exact",
            decision_reason="covered_scope_prioritize_confirmed_mechanic",
            selected_signal=1.0,
            functional_progress=True,
            useful_new_state=True,
        ),
        rollout.PolicyRolloutStep(
            step=1,
            key=KEY,
            game_id="bp35-0a0ad940",
            context_signature=("ACTION6",),
            context_id="after_ACTION6",
            policy_selected_action="ACTION3",
            fallback_action="ACTION3",
            predicted_metric="local_patch_before_after",
            selected_from_confirmed_mechanic=False,
            scope_used="CONTEXTUALLY_STABLE",
            context_match=False,
            context_match_reason="no_covered_rollout_context",
            decision_reason="fallback_neutral_exploration",
        ),
        rollout.PolicyRolloutStep(
            step=2,
            key=KEY,
            game_id="bp35-0a0ad940",
            context_signature=("ACTION3",),
            context_id="after_ACTION3",
            policy_selected_action="ACTION6",
            fallback_action="ACTION4",
            predicted_metric="local_patch_before_after",
            selected_from_confirmed_mechanic=True,
            scope_used="CONTEXTUALLY_STABLE",
            context_match=True,
            context_match_reason="covered_context_exact",
            decision_reason="covered_scope_prioritize_confirmed_mechanic",
            selected_signal=1.0,
            functional_progress=True,
            useful_new_state=True,
            repeated_usefulness=True,
        ),
    )

    monkeypatch.setattr(
        rollout,
        "execute_scope_conditioned_rollout",
        lambda registry_entries, *, scopes_by_key, environments_dir, budget, baseline_order: fake_steps,
    )

    payload = rollout.run_scope_conditioned_policy_rollout(
        registry_path=registry_path,
        scope_map_path=scope_path,
        environments_dir=tmp_path,
        budget=3,
    )

    assert payload["summary"]["policy_steps"] == 3
    assert payload["summary"]["policy_steps_from_confirmed_mechanic"] == 2
    assert payload["summary"]["functional_progress_steps"] == 2
    assert payload["summary"]["useful_new_states"] == 2
    assert payload["summary"]["fallback_steps"] == 1
    assert payload["summary"]["repeated_usefulness"] == 1
    assert payload["summary"]["usage_contradictions"] == 0
    assert payload["summary"]["truth_status"] == "NOT_REEVALUATED_BY_A37"
    assert payload["revision_performed"] is False
    assert payload["wrong_confirmations"] == 0
