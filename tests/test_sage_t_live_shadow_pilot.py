from __future__ import annotations

import math

from theory.sage_t.live_shadow_pilot import (
    LiveShadowRow,
    _compact_audit_record,
    build_report,
    load_frozen_manifest,
    rows_from_paired_arms,
    summarize_rows,
)


def _row(
    index: int,
    *,
    terminal_prediction: float = 0.1,
    terminal: int = 0,
    goal_prediction: float = 0.2,
    goal: int = 0,
    action_matches: bool = True,
    latency: float = 10.0,
) -> LiveShadowRow:
    return LiveShadowRow(
        game_id="lp85-305b61c3",
        seed=0,
        reset_index=0,
        step=index,
        action_key="ACTION1:{}",
        action_matches_off=action_matches,
        assessment_found=True,
        predicted_terminal=terminal_prediction,
        predicted_goal=goal_prediction,
        predicted_progress=0.0,
        actual_terminal=terminal,
        actual_goal=goal,
        actual_progress=float(goal),
        surprise=1.0 + index,
        entropy_before=0.8,
        entropy_after=0.6,
        entropy_reduction=0.2,
        particles_before=8,
        particles_after=4,
        effective_sample_size_after=3.0,
        repairs_attempted_delta=0,
        repairs_admitted_delta=0,
        top_program_changed=False,
        top_probability_after=0.7,
        decision_latency_ms=latency,
        observation_latency_ms=1.0,
    )


def _manifest(*, minimum_positives: int = 2) -> dict:
    return {
        "manifest_checksum": "manifest",
        "base_t7_1_manifest_checksum": "base",
        "gate": {
            "minimum_actions": 2,
            "minimum_prediction_coverage": 1.0,
            "minimum_finite_surprise_samples": 2,
            "maximum_decision_p95_ms": 100.0,
            "minimum_terminal_positives": minimum_positives,
            "minimum_goal_positives": minimum_positives,
        },
    }


def _condition() -> dict:
    return {
        "same_action_trace": True,
        "same_reset_states": True,
        "controller_errors": 0,
        "illegal_actions": 0,
        "environment_errors": 0,
        "interventions": 0,
        "trace_errors": 0,
    }


def test_frozen_manifest_is_source_train_shadow_only() -> None:
    manifest = load_frozen_manifest()

    assert manifest["authority"]["mode"] == "shadow"
    assert manifest["authority"]["active_gate_passed"] is False
    assert {game.split("-", 1)[0] for game in manifest["source_train_games"]} == {
        "bp35",
        "lp85",
        "su15",
    }
    assert manifest["action_budget_per_reset"] == 25


def test_calibration_and_entropy_metrics_are_computed_from_live_rows() -> None:
    rows = (
        _row(0, terminal_prediction=0.1, terminal=0),
        _row(
            1,
            terminal_prediction=0.9,
            terminal=1,
            goal_prediction=0.8,
            goal=1,
        ),
    )

    metrics = summarize_rows(rows)

    assert math.isclose(metrics["calibration"]["terminal"]["brier"], 0.01)
    assert metrics["calibration"]["terminal"]["positives"] == 1
    assert metrics["entropy"]["positive_reduction_rate"] == 1.0
    assert metrics["teleology"]["levels_completed"] == 1
    assert metrics["stability"]["posterior_collapse_actions"] == 0


def test_report_passes_integration_but_blocks_underidentified_calibration() -> None:
    rows = (_row(0), _row(1, goal=1, terminal=1))

    report = build_report(
        rows,
        manifest=_manifest(minimum_positives=2),
        conditions=(_condition(),),
        runtime={"ready": True},
        wall_clock_seconds=1.0,
    )

    assert report["integration_gate_passed"] is True
    assert report["calibration_identified"] is False
    assert report["diagnosis"] == "live_teleological_underidentification"
    assert report["status"] == "DIAGNOSIS_COMPLETE_FAIL_CLOSED"
    assert report["active_authority_authorized"] is False


def test_action_mismatch_and_latency_fail_the_integration_gate() -> None:
    rows = (_row(0), _row(1, action_matches=False, latency=1000.0))

    report = build_report(
        rows,
        manifest=_manifest(minimum_positives=0),
        conditions=(_condition(),),
        runtime={"ready": True},
        wall_clock_seconds=1.0,
    )

    assert report["checks"]["paired_action_identity"] is False
    assert report["checks"]["decision_latency"] is False
    assert report["integration_gate_passed"] is False
    assert report["bounded_authority_authorized"] is False


def test_compact_audit_keeps_aggregate_prediction_without_particle_matrix() -> None:
    record = {
        "kind": "decision",
        "posterior_before": {"top": [{"program_hash": str(i)} for i in range(12)]},
        "sequences": [
            {
                "sequence": ["ACTION1:{}"],
                "terminal_risk": 0.1,
                "program_predictions": [
                    {
                        "packets": [
                            {"known_channels": ["terminal", "goal"]},
                        ]
                    }
                ],
            }
        ],
    }

    compact = _compact_audit_record(record)

    assert len(compact["posterior_before"]["top"]) == 8
    assert "program_predictions" not in compact["sequences"][0]
    assert compact["sequences"][0]["prediction_particles"] == 1
    assert compact["sequences"][0]["known_channels"] == ["goal", "terminal"]


def test_paired_arm_rows_align_executed_action_prediction_and_observation() -> None:
    step = {
        "step": 0,
        "action": "ACTION1",
        "action_args": {},
        "levels_before": 0,
        "levels_after": 1,
        "game_state_after": "NOT_FINISHED",
    }
    snapshot_before = {
        "particles": 4,
        "normalized_entropy": 0.8,
        "repairs_attempted": 0,
        "repairs_admitted": 0,
        "top": [{"program_hash": "before", "probability": 0.6}],
    }
    snapshot_after = {
        "particles": 3,
        "normalized_entropy": 0.3,
        "effective_sample_size": 2.0,
        "repairs_attempted": 1,
        "repairs_admitted": 1,
        "top": [{"program_hash": "after", "probability": 0.8}],
    }
    off = {"attempts": [{"reset_index": 0, "trace": [step]}]}
    shadow = {
        "attempts": [{"reset_index": 0, "trace": [step]}],
        "controller_summary": {
            "sage_t_joint_program_posterior": {
                "instrumentation": {
                    "decision_latencies_ms": [12.0],
                    "observation_latencies_ms": [2.0],
                    "records": [
                        {
                            "kind": "decision",
                            "sequences": [
                                {
                                    "sequence": ["ACTION1:{}"],
                                    "terminal_risk": 0.05,
                                    "expected_goal": 0.7,
                                    "expected_progress": 0.5,
                                }
                            ],
                        },
                        {
                            "kind": "observation",
                            "surprise": 1.5,
                            "posterior_before": snapshot_before,
                            "posterior_after": snapshot_after,
                        },
                    ],
                }
            }
        },
    }

    row = rows_from_paired_arms(
        game_id="lp85-305b61c3",
        seed=0,
        off=off,
        shadow=shadow,
    )[0]

    assert row.assessment_found is True
    assert row.actual_goal == 1
    assert row.entropy_reduction == 0.5
    assert row.repairs_admitted_delta == 1
    assert row.top_program_changed is True
    assert row.decision_latency_ms == 12.0
