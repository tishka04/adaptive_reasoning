from theory.m3.scientific_planner_state import (
    build_scientific_planning_state_from_payloads,
    updated_ledger_entries_from_state,
)


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


def _experiment(control, *, support=1, contradiction=0, reuse_reason=""):
    row = {
        "hypothesis_key": KEY,
        "game_id": "bp35-0a0ad940",
        "mechanic_family": "position_effect_candidate",
        "target_action": "ACTION6",
        "control_action": control,
        "predicted_metric": "local_patch_before_after",
        "support_events": support,
        "contradiction_events": contradiction,
        "controlled_experiments_run": 1,
        "status": "UNRESOLVED",
        "support": 0,
    }
    if reuse_reason:
        row["control_reuse_reason"] = reuse_reason
    return row


def test_state_uses_updated_ledger_entries_only_as_fallback():
    payload = {
        "controlled_experiments": [_experiment("ACTION3", support=1)],
        "updated_ledger_entries": [
            {
                "key": KEY,
                "game_id": "bp35-0a0ad940",
                "support_events": 1,
                "controlled_experiments_run": 1,
            }
        ],
    }

    state = build_scientific_planning_state_from_payloads(
        ledger_payload=_ledger_payload(),
        controlled_payloads=[payload],
        budget=3,
    )

    assert state.support_events_by_key[KEY] == 1
    assert state.independent_support_events_by_key[KEY] == 1
    assert state.controlled_experiments_by_key[KEY] == 1
    assert state.remaining_budget == 2


def test_state_splits_raw_independent_and_reused_support_events():
    state = build_scientific_planning_state_from_payloads(
        ledger_payload=_ledger_payload(),
        extra_controlled_experiments=[
            _experiment("ACTION3", support=1),
            _experiment("ACTION4", support=1),
            _experiment(
                "ACTION3",
                support=1,
                reuse_reason="no_unused_distinct_control_available",
            ),
        ],
        budget=3,
    )
    updated = updated_ledger_entries_from_state(state)[0]

    assert state.summary()["support_events_total"] == 3
    assert state.summary()["independent_support_events_total"] == 2
    assert state.summary()["reused_control_support_events_total"] == 1
    assert state.remaining_budget == 0
    assert updated["status"] == "UNRESOLVED"
    assert updated["revision_status"] == "CANDIDATE_ONLY"
    assert updated["support"] == 0
    assert updated["controlled_test_required"] is True
