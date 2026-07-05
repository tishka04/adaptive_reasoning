import json
from pathlib import Path

import pytest

from theory.m3.objective_conversion_experiment_executor import CapturedSafeStop
from theory.m3.objective_conversion_safe_stop_diversity_sampler import SafeStopCandidatePlan
from theory.p3.contextual_post_stop_conversion_policy_probe import (
    ACTION6_ACTION3,
    ACTION6_ONLY,
)
from theory.p3.out_of_sample_contextual_post_stop_policy_validation import (
    IN_SAMPLE_SAFE_STOP_REJECTED,
    POST_STOP_CONTEXTUAL_POLICY_GENERALIZES_CANDIDATE_ONLY,
    classify_out_of_sample_safe_stops,
    run_out_of_sample_contextual_post_stop_policy_validation,
    validate_p3g2_probe_source,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _adapter_payload() -> dict:
    return {
        "summary": {
            "ready_for_contextual_policy_probe": True,
            "support": 0,
        },
        "contextual_post_stop_policy_adapter": {
            "policy_adapter_id": "p3g2::test::adapter",
            "extension_sequence_preference": [ACTION6_ACTION3, "ACTION6,ACTION4"],
            "min_extension_margin_vs_action6": 5.0,
            "risk_model": {
                "blocked_sampling_family_horizon_pairs": [
                    "blocked_family|horizon_mid_45_54"
                ],
                "blocked_hold_baseline_bands": ["hold_high_ge_120"],
                "support": 0,
            },
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
            "policy_adapter_counted_as_confirmation": False,
            "policy_result_counted_as_scientific_verdict": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        "policy_adapter_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _m3g4_source() -> dict:
    return {
        "summary": {"support": 0},
        "per_safe_stop_validation_records": [
            {
                "safe_stop_id": "seen",
                "safe_stop_state_hash": "state::seen",
                "captured_prefix_hash": "prefix::seen",
                "candidate_records": [],
            }
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "validation_outcome_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _p3g2_source(*, support: int = 0) -> dict:
    return {
        "summary": {
            "policy_utility_status": "POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY",
            "safe_stops_evaluated": 13,
            "mean_terminal_adjusted_progress": 80.0,
            "terminal_rate": 0.0,
            "support": support,
        },
        "policy_decision_records": [{"safe_stop_id": "seen"}],
        "policy_utility_status": "POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY",
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        "policy_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _plan(plan_id: str) -> SafeStopCandidatePlan:
    action = {
        "seen": "ACTION3",
        "allowed": "ACTION4",
        "blocked": "ACTION6",
    }.get(plan_id, "ACTION3")
    return SafeStopCandidatePlan(
        plan_id=plan_id,
        planned_prefix=({"action": action, "action_args": {}},),
        sampling_family="test_family",
        anti_attractor_rationale="test",
        tie_break_seed=0,
    )


def _candidate_result(
    plan: SafeStopCandidatePlan,
    *,
    family: str,
    horizon: int,
    hold: float,
    state_hash: str,
    prefix_hash: str,
) -> dict:
    return {
        **plan.to_dict(),
        "execution_performed": True,
        "captured_prefix": [{"action": "ACTION3", "action_args": {}}],
        "captured_prefix_len": 1,
        "captured_prefix_hash": prefix_hash,
        "safe_stop_state_hash": state_hash,
        "relation_state_signature": f"relation::{state_hash}",
        "terminal_horizon_estimate": {
            "estimated_moves_remaining": horizon,
            "source": "test",
        },
        "hold_baseline_terminal_adjusted_progress": hold,
        "hold_baseline_levels_completed": 0,
        "hold_baseline_terminal": False,
        "replay_exact": True,
        "non_terminal": True,
        "terminal_safe": True,
        "hold_baseline_measurable": True,
        "sampling_family": family,
        "blocked_reason": "",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
    }


def _cell_result(cell, captured: CapturedSafeStop) -> dict:
    safe_stop_id = captured.provenance["safe_stop_id"]
    hold = captured.hold_baseline_terminal_adjusted_progress
    if cell.condition_kind == "hold":
        taps = hold
        terminal = False
    elif cell.action_or_sequence == ("ACTION6",):
        taps = hold + 5.0
        terminal = False
    elif safe_stop_id == "p3_g3::oos_safe_stop::002":
        taps = 0.0
        terminal = True
    else:
        taps = hold + 15.0
        terminal = False
    return {
        **cell.to_dict(),
        "execution_performed": True,
        "safe_stop_replay_exact": True,
        "measurements": {
            "delta_terminal_adjusted_progress_vs_hold": taps - hold,
            "candidate_terminal_adjusted_progress_after_stop": taps,
            "candidate_terminal_reentry": terminal,
            "objective_completion_signal": False,
            "candidate_levels_completed_after_rollout": 0,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
    }


def _captured_builder(record: dict) -> CapturedSafeStop:
    return CapturedSafeStop(
        prefix=tuple(record["captured_prefix"]),
        captured_prefix_hash=record["captured_prefix_hash"],
        safe_stop_state_hash=record["safe_stop_state_hash"],
        hold_baseline_terminal_adjusted_progress=float(
            record["hold_baseline_terminal_adjusted_progress"]
        ),
        hold_baseline_levels_completed=0,
        hold_baseline_terminal=False,
        provenance={"safe_stop_id": record["safe_stop_id"]},
        capture_config={},
        adapter={},
        prefix_step_dicts=(),
    )


def test_oos_validation_freezes_adapter_and_selects_without_oos_leakage(
    tmp_path: Path,
) -> None:
    adapter_path = _write_json(tmp_path / "adapter.json", _adapter_payload())
    m3g4_path = _write_json(tmp_path / "m3g4.json", _m3g4_source())
    p3g2_path = _write_json(tmp_path / "p3g2.json", _p3g2_source())
    plans = [_plan("seen"), _plan("allowed"), _plan("blocked")]

    def fake_candidate_executor(plan: SafeStopCandidatePlan) -> dict:
        if plan.plan_id == "seen":
            return _candidate_result(
                plan,
                family="seen_family",
                horizon=60,
                hold=10.0,
                state_hash="state::seen",
                prefix_hash="prefix::seen",
            )
        if plan.plan_id == "allowed":
            return _candidate_result(
                plan,
                family="new_family",
                horizon=60,
                hold=40.0,
                state_hash="state::allowed",
                prefix_hash="prefix::allowed",
            )
        return _candidate_result(
            plan,
            family="blocked_family",
            horizon=50,
            hold=140.0,
            state_hash="state::blocked",
            prefix_hash="prefix::blocked",
        )

    payload = run_out_of_sample_contextual_post_stop_policy_validation(
        adapter_path=adapter_path,
        source_m3g4_path=m3g4_path,
        source_p3g2_probe_path=p3g2_path,
        min_out_of_sample_safe_stops=2,
        candidate_plans=plans,
        candidate_executor=fake_candidate_executor,
        captured_builder=_captured_builder,
        cell_executor=_cell_result,
    )

    decisions = {
        row["safe_stop_id"]: row["selected_option"]
        for row in payload["policy_decision_records"]
    }
    assert payload["summary"]["accepted_out_of_sample_safe_stops"] == 2
    assert payload["summary"]["in_sample_safe_stops_rejected"] == 1
    assert decisions["p3_g3::oos_safe_stop::001"] == ACTION6_ACTION3
    assert decisions["p3_g3::oos_safe_stop::002"] == ACTION6_ONLY
    assert all(
        row["selection_uses_out_of_sample_candidate_outcomes"] is False
        for row in payload["policy_decision_records"]
    )
    assert payload["summary"]["unsafe_extension_options_avoided"] == 2
    assert payload["summary"]["policy_utility_status"] == (
        POST_STOP_CONTEXTUAL_POLICY_GENERALIZES_CANDIDATE_ONLY
    )
    assert payload["summary"]["mean_delta_vs_action6_only"] > 0
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_scientific_verdict"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False


def test_classification_rejects_in_sample_identity() -> None:
    record = {
        "execution_performed": True,
        "replay_exact": True,
        "non_terminal": True,
        "terminal_safe": True,
        "hold_baseline_measurable": True,
        "safe_stop_state_hash": "state::seen",
        "captured_prefix_hash": "prefix::fresh",
        "relation_state_signature": "relation::fresh",
    }

    classified, accepted = classify_out_of_sample_safe_stops(
        [record],
        in_sample_identity={
            "state_hashes": {"state::seen"},
            "prefix_hashes": {"prefix::seen"},
        },
        min_out_of_sample_safe_stops=1,
    )

    assert accepted == []
    assert classified[0]["out_of_sample_acceptance_status"] == IN_SAMPLE_SAFE_STOP_REJECTED
    assert classified[0]["support"] == 0


def test_p3g2_source_rejects_support() -> None:
    with pytest.raises(ValueError):
        validate_p3g2_probe_source(_p3g2_source(support=1))
