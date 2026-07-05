from theory.m3.next_experiment_selector import (
    CONTROL_REUSE_REASON,
    available_controls_for_target,
    select_next_experiment,
)
from theory.m3.scientific_planner_state import (
    build_scientific_planning_state_from_payloads,
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


def _experiment(control, *, support=1):
    return {
        "hypothesis_key": KEY,
        "game_id": "bp35-0a0ad940",
        "target_action": "ACTION6",
        "control_action": control,
        "predicted_metric": "local_patch_before_after",
        "support_events": support,
        "contradiction_events": 0,
        "controlled_experiments_run": 1,
        "status": "UNRESOLVED",
    }


def test_available_controls_are_dynamic_with_preferred_sorting_then_other_actions():
    controls = available_controls_for_target(
        ["RESET", "ACTION6", "ACTION5", "ACTION4", "ACTION7"],
        target_action="ACTION6",
    )

    assert controls == ("ACTION4", "ACTION5", "ACTION7")


def test_selector_picks_next_unused_preferred_control_and_logs_skips():
    state = build_scientific_planning_state_from_payloads(
        ledger_payload=_ledger_payload(),
        extra_controlled_experiments=[_experiment("ACTION3")],
        budget=3,
    )

    plan = select_next_experiment(
        state,
        live_available_actions=("ACTION3", "ACTION4", "ACTION6"),
    )

    assert plan.control_action == "ACTION4"
    assert plan.control_reuse_reason == ""
    assert {row["action"] for row in plan.skipped_controls} == {"ACTION1", "ACTION2"}
    assert "insufficient_distinct_controls" not in plan.open_questions


def test_selector_marks_reused_control_after_distinct_controls_are_exhausted():
    state = build_scientific_planning_state_from_payloads(
        ledger_payload=_ledger_payload(),
        extra_controlled_experiments=[
            _experiment("ACTION3"),
            _experiment("ACTION4"),
        ],
        budget=3,
    )

    plan = select_next_experiment(
        state,
        live_available_actions=("ACTION3", "ACTION4", "ACTION6"),
    )

    assert plan.control_action == "ACTION3"
    assert plan.control_reuse_reason == CONTROL_REUSE_REASON
    assert "all_controls_support_hypothesis" in plan.open_questions
