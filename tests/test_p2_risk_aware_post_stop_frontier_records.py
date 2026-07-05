import json
from pathlib import Path

import pytest

from theory.p2.policy_frontier_records import TRUTH_STATUS
from theory.p2.risk_aware_post_stop_frontier_records import (
    OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION,
    P3G4_RISK_AWARE_STATUS,
    RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER,
    RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION,
    build_risk_aware_post_stop_frontier_record,
    run_risk_aware_post_stop_frontier_records,
    validate_risk_aware_post_stop_frontier_record,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _aggregate(condition: str, taps: float, terminal_rate: float = 0.0) -> dict:
    return {
        "condition": condition,
        "runs": 7,
        "mean_terminal_adjusted_progress": taps,
        "mean_delta_vs_hold": taps - 135.0,
        "terminal_rate": terminal_rate,
        "objective_completion_runs": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
    }


def _p3g4_payload(
    *,
    support: int = 0,
    verdict: bool = False,
    a32_write: bool = False,
    completion: bool = False,
) -> dict:
    objective_completion_runs = 1 if completion else 0
    status = (
        "RISK_TARGETED_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY"
        if completion
        else P3G4_RISK_AWARE_STATUS
    )
    summary = {
        "policy_utility_status": status,
        "adapter_relearned": False,
        "source_cells_rerun": True,
        "execution_performed": True,
        "selection_uses_risk_targeted_candidate_outcomes": False,
        "accepted_risk_targeted_safe_stops": 7,
        "selected_extension_count": 2,
        "selected_action6_only_count": 5,
        "selected_hold_count": 0,
        "terminal_rate": 0.0,
        "mean_terminal_adjusted_progress": 142.857143,
        "mean_delta_vs_hold": 7.857143,
        "mean_delta_vs_action6_only": 2.857143,
        "improvement_over_action6_only": True,
        "static_extension_terminal_options": 4,
        "static_extension_terminal_safe_stops": 2,
        "unsafe_extension_options_avoided": 4,
        "objective_completion_signal": completion,
        "objective_completion_runs": objective_completion_runs,
        "baseline_aggregates": {
            "hold_or_stop_state": _aggregate("hold_or_stop_state", 135.0),
            "ACTION6": _aggregate("ACTION6", 140.0),
            "ACTION6,ACTION3": _aggregate("ACTION6,ACTION3", 104.285714, 0.285714),
            "ACTION6,ACTION4": _aggregate("ACTION6,ACTION4", 104.285714, 0.285714),
        },
        "contextual_policy_aggregate": _aggregate(
            "contextual_post_stop_conversion_policy",
            142.857143,
        ),
        "risk_targeted_extension_risk_stats": {
            "action6_action3_terminal_rate": 0.285714,
            "action6_action4_terminal_rate": 0.285714,
            "static_extension_terminal_options": 4,
            "static_extension_terminal_safe_stops": 2,
            "unsafe_extension_options_avoided_by_selector": 4,
            "terminal_extension_records": [
                {
                    "safe_stop_id": "p3_g4::risk_oos_safe_stop::004",
                    "sampling_family": "base_tail_action6_action3",
                    "terminal_horizon_band": "horizon_mid_45_54",
                    "hold_baseline_band": "hold_high_ge_120",
                    "action_or_sequence": ["ACTION6", "ACTION3"],
                    "candidate_terminal_adjusted_progress_after_stop": 0.0,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
                }
            ],
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        },
        "policy_result_counted_as_scientific_verdict": verdict,
        "adapter_counted_as_mechanic_confirmation": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": a32_write,
        "a33_write_performed": False,
    }
    return {
        "summary": summary,
        "policy_utility_status": status,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_scientific_verdict": verdict,
        "adapter_counted_as_mechanic_confirmation": False,
        "a32_write_performed": a32_write,
        "a33_write_performed": False,
    }


def test_risk_aware_post_stop_frontier_records_builds_review_ready_record(
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "p3g4.json", _p3g4_payload())

    payload = run_risk_aware_post_stop_frontier_records(
        p3_risk_targeted_validation_path=source,
    )

    assert payload["summary"]["frontier_records"] == 1
    assert payload["summary"]["frontier_type"] == (
        RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER
    )
    assert payload["summary"]["frontier_reason"] == (
        RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION
    )
    assert payload["summary"]["blocked_capability"] == (
        OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
    )
    assert payload["summary"]["ready_for_risk_aware_objective_frontier_review"] is True
    assert payload["summary"]["ready_for_m2_or_m3"] is False
    record = payload["risk_aware_post_stop_frontier_records"][0]
    assert record["source_policy_utility_status"] == P3G4_RISK_AWARE_STATUS
    assert record["evidence"]["terminal_rate"] == 0.0
    assert record["evidence"]["static_extension_terminal_options"] == 4
    assert record["evidence"]["unsafe_extension_options_avoided"] == 4
    assert record["evidence"]["objective_completion_signal"] is False
    assert record["blocked_capability"] == (
        OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
    )
    assert "objective_readiness_detector_missing" in record[
        "blocked_capability_hypotheses"
    ]
    assert record["ready_for_m2_or_m3"] is False
    assert record["a33_ready"] is False
    assert record["support"] == 0
    assert record["truth_status"] == TRUTH_STATUS


def test_risk_aware_post_stop_frontier_records_noops_when_completion_exists(
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "p3g4.json", _p3g4_payload(completion=True))

    payload = run_risk_aware_post_stop_frontier_records(
        p3_risk_targeted_validation_path=source,
    )

    assert payload["risk_aware_post_stop_frontier_records"] == []
    assert payload["summary"]["frontier_records"] == 0
    assert payload["summary"]["ready_for_risk_aware_objective_frontier_review"] is False
    assert payload["summary"]["support"] == 0


@pytest.mark.parametrize(
    ("support", "verdict", "a32_write"),
    [(1, False, False), (0, True, False), (0, False, True)],
)
def test_risk_aware_post_stop_frontier_records_rejects_bad_source(
    tmp_path: Path,
    support: int,
    verdict: bool,
    a32_write: bool,
) -> None:
    source = _write_json(
        tmp_path / "p3g4.json",
        _p3g4_payload(support=support, verdict=verdict, a32_write=a32_write),
    )

    with pytest.raises(ValueError):
        run_risk_aware_post_stop_frontier_records(
            p3_risk_targeted_validation_path=source,
        )


def test_validate_risk_aware_frontier_rejects_direct_handoff_or_completion() -> None:
    record = build_risk_aware_post_stop_frontier_record(
        _p3g4_payload()["summary"],
        source_path="p3g4.json",
    )
    record["ready_for_m2_or_m3"] = True
    with pytest.raises(ValueError, match="not direct M2/M3 handoffs"):
        validate_risk_aware_post_stop_frontier_record(record)

    record["ready_for_m2_or_m3"] = False
    record["evidence"]["objective_completion_signal"] = True
    with pytest.raises(ValueError, match="requires no objective completion"):
        validate_risk_aware_post_stop_frontier_record(record)
