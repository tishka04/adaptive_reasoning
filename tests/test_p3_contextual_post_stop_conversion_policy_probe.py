import json
from pathlib import Path

import pytest

from theory.p3.contextual_post_stop_conversion_policy_probe import (
    ACTION6_ACTION3,
    ACTION6_ONLY,
    POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY,
    build_contextual_post_stop_conversion_policy_adapter,
    run_contextual_post_stop_conversion_policy_probe,
    validate_m3g4_source,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(
    sequence: list[str],
    *,
    hold: float,
    taps: float,
    terminal: bool = False,
    medium: bool = False,
) -> dict:
    return {
        "condition_id": "candidate_" + "_".join(sequence),
        "action_or_sequence": sequence,
        "post_stop_horizon": len(sequence),
        "candidate_terminal_adjusted_progress_after_stop": taps,
        "delta_terminal_adjusted_progress_vs_hold": taps - hold,
        "hold_or_stop_state_terminal_adjusted_progress_after_stop": hold,
        "candidate_terminal_reentry": terminal,
        "objective_completion_signal": False,
        "candidate_levels_completed_after_rollout": 0,
        "beats_hold_or_stop_state": taps > hold and not terminal,
        "beats_relation_progress_policy": medium,
        "medium_signal": medium,
        "weak_signal": taps > hold and not terminal,
        "signal_strength": "MEDIUM_BEATS_HOLD_AND_RELATION_CANDIDATE_ONLY"
        if medium
        else "WEAK_BEATS_HOLD_CANDIDATE_ONLY",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def _safe_stop(
    safe_stop_id: str,
    *,
    family: str,
    horizon_band: str,
    hold_band: str,
    horizon_remaining: int,
    hold: float,
    extension_terminal: bool = False,
) -> dict:
    extension_taps = 0.0 if extension_terminal else hold + 15.0
    return {
        "safe_stop_id": safe_stop_id,
        "sampling_family": family,
        "terminal_horizon_remaining": horizon_remaining,
        "terminal_horizon_band": horizon_band,
        "hold_baseline_terminal_adjusted_progress": hold,
        "hold_baseline_band": hold_band,
        "relation_progress_control_summary": {
            "best_relation_taps": hold + 20.0,
            "any_relation_control_terminal_reentry": False,
        },
        "candidate_records": [
            _candidate(["ACTION6"], hold=hold, taps=hold + 5.0),
            _candidate(
                ["ACTION6", "ACTION3"],
                hold=hold,
                taps=extension_taps,
                terminal=extension_terminal,
            ),
            _candidate(
                ["ACTION6", "ACTION4"],
                hold=hold,
                taps=extension_taps,
                terminal=extension_terminal,
            ),
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def _m3g4_payload(*, support: int = 0) -> dict:
    records = [
        _safe_stop(
            "safe",
            family="single_action_burst",
            horizon_band="horizon_far_ge_55",
            hold_band="hold_low_lt_50",
            horizon_remaining=60,
            hold=40.0,
        ),
        _safe_stop(
            "risky",
            family="base_prefix_truncation",
            horizon_band="horizon_mid_45_54",
            hold_band="hold_high_ge_120",
            horizon_remaining=50,
            hold=140.0,
            extension_terminal=True,
        ),
    ]
    return {
        "summary": {
            "support": support,
            "validation_outcome_status": "MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY",
        },
        "validation_outcome_status": "MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY",
        "per_safe_stop_validation_records": records,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "validation_outcome_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _p3g0_reference() -> dict:
    return {
        "summary": {
            "condition_aggregates": {
                "abstract_mechanic_policy_from_m3g0": {
                    "terminal_rate": 0.5,
                    "mean_terminal_adjusted_progress": 20.0,
                    "mean_progress_proxy": 120.0,
                }
            },
            "support": 0,
        },
        "support": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _p3g1_reference() -> dict:
    return {
        "summary": {
            "best_objective_aware_condition": "objective_aware_abstract_policy_lambda_0",
            "best_objective_terminal_rate": 0.0,
            "best_objective_terminal_adjusted_progress": 132.5,
            "best_objective_mean_progress_proxy": 132.5,
            "support": 0,
        },
        "support": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def test_adapter_builds_contextual_risk_model_without_confirmation(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "g4.json", _m3g4_payload())

    payload = build_contextual_post_stop_conversion_policy_adapter(
        source_m3g4_path=source
    )

    adapter = payload["contextual_post_stop_policy_adapter"]
    risk_model = adapter["risk_model"]
    assert payload["summary"]["ready_for_contextual_policy_probe"] is True
    assert "base_prefix_truncation|horizon_mid_45_54" in risk_model[
        "blocked_sampling_family_horizon_pairs"
    ]
    assert "hold_high_ge_120" in risk_model["blocked_hold_baseline_bands"]
    assert risk_model["unsafe_extension_options_observed"] == 2
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_scientific_verdict"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False


def test_probe_selects_extension_only_in_safe_scope(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "g4.json", _m3g4_payload())
    adapter = build_contextual_post_stop_conversion_policy_adapter(
        source_m3g4_path=source
    )
    adapter_path = _write_json(tmp_path / "adapter.json", adapter)
    p3g0_path = _write_json(tmp_path / "p3g0.json", _p3g0_reference())
    p3g1_path = _write_json(tmp_path / "p3g1.json", _p3g1_reference())

    payload = run_contextual_post_stop_conversion_policy_probe(
        source_m3g4_path=source,
        adapter_path=adapter_path,
        p3g0_policy_probe_path=p3g0_path,
        p3g1_consolidation_path=p3g1_path,
    )

    decisions = {
        row["safe_stop_id"]: row["selected_option"]
        for row in payload["policy_decision_records"]
    }
    assert decisions["safe"] == ACTION6_ACTION3
    assert decisions["risky"] == ACTION6_ONLY
    assert payload["summary"]["policy_utility_status"] == (
        POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY
    )
    assert payload["summary"]["terminal_rate"] == 0.0
    assert payload["summary"]["unsafe_extension_options_avoided"] == 2
    assert payload["summary"]["improvement_over_action6_only"] is True
    assert payload["summary"]["p3_reference_comparison"][
        "terminal_rate_delta_vs_p3g1"
    ] == 0.0
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_scientific_verdict"] is False


def test_m3g4_source_rejects_support_or_downstream_writes() -> None:
    with pytest.raises(ValueError):
        validate_m3g4_source(_m3g4_payload(support=1))

    payload = _m3g4_payload()
    payload["a33_write_performed"] = True
    with pytest.raises(ValueError):
        validate_m3g4_source(payload)
