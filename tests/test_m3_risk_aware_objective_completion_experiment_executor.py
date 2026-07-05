"""Tests for M3.G6 risk-aware objective-completion executor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.m3.risk_aware_objective_completion_experiment_executor import (
    BLOCKED_CELL_STATUS,
    DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH,
    PROXY_COMPLETION_DIVERGENCE,
    run_risk_aware_objective_completion_experiment_execution,
)
from theory.m3.risk_aware_objective_completion_experiment_planner import (
    DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH,
)
from theory.p3.risk_targeted_contextual_post_stop_policy_validation import (
    DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH,
)
from theory.m3.objective_conversion_diverse_safe_stop_validation import (
    DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUESTS_PATH = REPO_ROOT / DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH
P3G4_PATH = REPO_ROOT / DEFAULT_RISK_TARGETED_CONTEXTUAL_POST_STOP_POLICY_OUTPUT_PATH
M3G4_PATH = (
    REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH
)


def _load_payload() -> dict:
    return run_risk_aware_objective_completion_experiment_execution(
        requests_path=REQUESTS_PATH,
        source_p3g4_path=P3G4_PATH,
        source_m3g4_path=M3G4_PATH,
    )


def test_executor_consumes_m3g5_requests_and_deduplicates_cells() -> None:
    payload = _load_payload()
    summary = payload["summary"]

    assert summary["risk_aware_objective_experiment_requests_consumed"] == 15
    assert summary["raw_execution_cells_planned"] == 336
    assert summary["deduplicated_execution_cells"] == 72
    assert summary["deduplicated_cells_removed"] == 264
    assert summary["candidate_protocol_cells"] == 48
    assert summary["control_cells"] == 24
    assert payload["objective_completion_experiment_outcome_status"] == (
        PROXY_COMPLETION_DIVERGENCE
    )


def test_executor_preserves_candidate_only_guardrails() -> None:
    payload = _load_payload()
    summary = payload["summary"]

    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["execution_performed"] is True
    assert payload["policy_rollout_performed"] is False
    assert payload["environment_step_performed"] is False
    assert payload["experiment_result_counted_as_scientific_verdict"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert summary["support"] == 0
    assert summary["candidate_signal_counted_as_scientific_verdict"] is False

    for result in payload["execution_results"]:
        assert result["support"] == 0
        assert result["revision_status"] == "CANDIDATE_ONLY"
        assert result["truth_status"] == "NOT_EVALUATED_BY_M3"
        assert result["cell_result_counted_as_scientific_verdict"] is False
        assert result["a32_write_performed"] is False
        assert result["a33_write_performed"] is False


def test_executor_reports_proxy_completion_divergence_without_completion() -> None:
    payload = _load_payload()
    summary = payload["summary"]

    assert summary["objective_completion_signal"] is False
    assert summary["objective_completion_candidate_cells"] == 0
    assert summary["levels_completed_after_rollout_max"] == 0.0
    assert summary["proxy_progress_without_completion_observed"] is True
    assert summary["proxy_completion_divergence_candidate_cells"] == 42
    assert summary["static_extension_terminal_options_from_p3g4"] == 4
    assert summary["unsafe_extension_options_avoided_from_p3g4"] == 4


def test_commit_actions_are_blocked_when_not_measured_in_source_substrate() -> None:
    payload = _load_payload()
    blocked = [
        row
        for row in payload["execution_results"]
        if row["cell_execution_status"] == BLOCKED_CELL_STATUS
    ]

    assert len(blocked) == 6
    assert {row["protocol_family"] for row in blocked} == {
        "post_conversion_commit_action_search"
    }
    assert {
        row["blocked_reason"] for row in blocked
    } == {"commit_action_not_present_in_m3_g6_source_substrate_measurements"}
    assert payload["summary"]["commit_action_cells_blocked"] == 6


def test_action6_led_sequences_remain_controls_not_candidate_targets() -> None:
    payload = _load_payload()
    forbidden = {"ACTION6", "ACTION6,ACTION3", "ACTION6,ACTION4"}
    candidates = [
        row for row in payload["execution_results"] if row["condition_kind"] == "candidate"
    ]
    controls = [
        row for row in payload["execution_results"] if row["condition_kind"] == "control"
    ]

    assert candidates
    assert all(row["target"] not in forbidden for row in candidates)
    assert {
        "static_ACTION6,ACTION3",
        "static_ACTION6,ACTION4",
        "ACTION6",
    } <= {row["condition_id"] for row in controls}


def test_executor_rejects_executed_m3g5_source(tmp_path: Path) -> None:
    payload = json.loads(REQUESTS_PATH.read_text(encoding="utf-8"))
    payload["summary"]["execution_performed"] = True
    path = tmp_path / "executed_m3g5_requests.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="planning-only"):
        run_risk_aware_objective_completion_experiment_execution(
            requests_path=path,
            source_p3g4_path=P3G4_PATH,
            source_m3g4_path=M3G4_PATH,
        )


def test_default_output_path_is_m3_g6_results_artifact() -> None:
    assert DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH.name == (
        "risk_aware_objective_completion_experiment_results.json"
    )
