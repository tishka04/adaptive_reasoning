"""Tests for M3.G1.2 objective-conversion experiment executor (stubbed env)."""

from __future__ import annotations

from pathlib import Path

from theory.m3.objective_conversion_experiment_planner import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
)
from theory.m3.objective_conversion_experiment_executor import (
    CANDIDATE_ACTION_UNAVAILABLE_BLOCKED,
    CapturedSafeStop,
    HOLD_OR_STOP_STATE_OBSERVED,
    LOW_SINGLE_SAFE_STOP,
    OBJECTIVE_CONVERSION_CELL_EXECUTED,
    ObjectiveConversionCell,
    blocked_cell_result,
    measured_cell_result,
    run_objective_conversion_experiment_execution,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUESTS_PATH = REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH


def _captured() -> CapturedSafeStop:
    prefix = (
        {"action": "ACTION6", "action_args": {"x": 18, "y": 0}},
        {"action": "ACTION3", "action_args": {}},
    )
    return CapturedSafeStop(
        prefix=prefix,
        captured_prefix_hash="prefix::stub",
        safe_stop_state_hash="safe_stop::stub",
        hold_baseline_terminal_adjusted_progress=100.0,
        hold_baseline_levels_completed=0,
        hold_baseline_terminal=False,
        provenance={
            "safe_stop_state_source": "P3.G1",
            "safe_stop_policy_condition": "objective_aware_abstract_policy_lambda_0",
            "stop_trigger_reason": "objective_aware_terminal_risk_score_stop",
            "base_state_family": "terminal_safe_stop_or_avoidance_state",
        },
        capture_config={"selection_counted_as_support": False},
        adapter={},
    )


def _stub_cell_executor(cell: ObjectiveConversionCell, captured: CapturedSafeStop):
    """Deterministic measurement per condition kind, exercising all branches."""
    if cell.condition_kind == "hold":
        return measured_cell_result(
            cell,
            captured=captured,
            candidate_terminal_adjusted_progress=100.0,
            candidate_levels_completed=0,
            candidate_terminal_reentry=False,
            objective_completion_signal=False,
            diagnostics={"changed_pixels": 0.0},
            replayed_prefix_hash=captured.captured_prefix_hash,
            safe_stop_state_hash=captured.safe_stop_state_hash,
            post_stop_steps_executed=0,
        )
    actions = cell.action_or_sequence or ()
    if "ACTION6" in actions:
        # Candidate that re-enters terminal (the dangerous reactivation case).
        if cell.condition_kind == "candidate":
            return measured_cell_result(
                cell,
                captured=captured,
                candidate_terminal_adjusted_progress=0.0,
                candidate_levels_completed=0,
                candidate_terminal_reentry=True,
                objective_completion_signal=False,
                diagnostics={"changed_pixels": 47.0},
                replayed_prefix_hash=captured.captured_prefix_hash,
                safe_stop_state_hash=captured.safe_stop_state_hash,
                post_stop_steps_executed=len(actions),
            )
    if cell.condition_kind == "candidate" and actions == ("ACTION9",):
        return blocked_cell_result(
            cell,
            captured=captured,
            status=CANDIDATE_ACTION_UNAVAILABLE_BLOCKED,
            reason="stub_unavailable",
            replayed_prefix_hash=captured.captured_prefix_hash,
            safe_stop_state_hash=captured.safe_stop_state_hash,
            safe_stop_replay_exact=True,
        )
    # Generic candidate / relation control: small positive progress.
    taps = 105.0 if cell.condition_kind == "candidate" else 101.0
    return measured_cell_result(
        cell,
        captured=captured,
        candidate_terminal_adjusted_progress=taps,
        candidate_levels_completed=0,
        candidate_terminal_reentry=False,
        objective_completion_signal=False,
        diagnostics={"changed_pixels": 3.0},
        replayed_prefix_hash=captured.captured_prefix_hash,
        safe_stop_state_hash=captured.safe_stop_state_hash,
        post_stop_steps_executed=max(1, len(actions)),
    )


def _run():
    return run_objective_conversion_experiment_execution(
        requests_path=REQUESTS_PATH,
        safe_stop_capturer=lambda requests: _captured(),
        cell_executor=_stub_cell_executor,
    )


def test_execution_runs_all_unique_cells_and_links_back() -> None:
    payload = _run()
    summary = payload["summary"]
    assert summary["objective_conversion_requests_consumed"] == 12
    assert summary["unique_execution_cells"] >= 1
    assert summary["hypothesis_measurement_links"] == summary["planned_cells"]
    assert summary["cells_executed"] >= 1


def test_safe_stop_capture_block_present_and_low_diversity() -> None:
    payload = _run()
    capture = payload["safe_stop_capture"]
    assert capture["safe_stop_context_diversity"] == LOW_SINGLE_SAFE_STOP
    assert payload["summary"]["safe_stop_context_diversity"] == LOW_SINGLE_SAFE_STOP
    assert capture["provenance"]["safe_stop_policy_condition"] == (
        "objective_aware_abstract_policy_lambda_0"
    )
    assert capture["safe_stop_capture_config"]["selection_counted_as_support"] is False


def test_delta_vs_hold_computed_against_hold_baseline() -> None:
    payload = _run()
    candidate_cells = [
        cell
        for cell in payload["execution_cells"]
        if cell["condition_kind"] == "candidate"
        and cell["status"] == OBJECTIVE_CONVERSION_CELL_EXECUTED
        and not cell["measurements"]["candidate_terminal_reentry"]
    ]
    assert candidate_cells
    for cell in candidate_cells:
        m = cell["measurements"]
        expected = (
            m["candidate_terminal_adjusted_progress_after_stop"]
            - m["hold_or_stop_state_terminal_adjusted_progress_after_stop"]
        )
        assert abs(m["delta_terminal_adjusted_progress_vs_hold"] - expected) < 1e-9


def test_hold_cell_has_zero_delta_and_hold_status() -> None:
    payload = _run()
    hold_cells = [
        cell
        for cell in payload["execution_cells"]
        if cell["condition_kind"] == "hold"
    ]
    assert hold_cells
    for cell in hold_cells:
        assert cell["status"] == HOLD_OR_STOP_STATE_OBSERVED
        assert cell["measurements"]["delta_terminal_adjusted_progress_vs_hold"] == 0.0


def test_replay_exact_flag_true_when_hashes_match() -> None:
    payload = _run()
    for cell in payload["execution_cells"]:
        if cell["execution_performed"]:
            assert cell["safe_stop_replay_exact"] is True
            assert cell["captured_prefix_hash"] == cell["replayed_prefix_hash"]


def test_blocked_cell_routing_and_no_measurement() -> None:
    captured = _captured()
    cell = ObjectiveConversionCell(
        cell_signature="m3_g1::stub",
        game_id="bp35-0a0ad940",
        condition_kind="candidate",
        condition_id="candidate_ACTION9",
        action_or_sequence=("ACTION9",),
        post_stop_horizon=1,
    )
    result = _stub_cell_executor(cell, captured)
    assert result["status"] == CANDIDATE_ACTION_UNAVAILABLE_BLOCKED
    assert result["execution_performed"] is False
    assert result["measurements"] == {}


def test_executor_produces_no_aggregated_outcome_enum() -> None:
    payload = _run()
    assert payload["summary"]["produces_aggregated_outcome_enum"] is False
    text = str(payload)
    for outcome in (
        "CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY",
        "NO_CONVERSION_SIGNAL_CANDIDATE_ONLY",
        "MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY",
    ):
        assert outcome not in text


def test_guardrails_locked() -> None:
    payload = _run()
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["experiment_result_counted_as_scientific_verdict"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    for cell in payload["execution_cells"]:
        assert cell["support"] == 0
        assert cell["truth_status"] == "NOT_EVALUATED_BY_M3"
