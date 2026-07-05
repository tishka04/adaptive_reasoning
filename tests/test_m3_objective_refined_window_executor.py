import json

import pytest

import theory.m3.objective_refined_window_executor as executor


def _threshold_payload(*, support=0, execution=False, verdict=False):
    return {
        "summary": {
            "objective_threshold_consolidations": 1,
            "threshold_type": "PRE_TERMINAL_PREFIX_WINDOW",
            "safe_tested_prefix_max": 48,
            "early_terminal_prefix_min": 64,
            "critical_window": [49, 64],
            "next_experiment_recommendation": "REFINE_PREFIX_WINDOW",
            "recommended_refined_prefixes": [50, 54, 58, 60, 62, 63],
            "recommended_conditions": [
                "continue_action6",
                "stop_policy",
                "switch_ACTION3",
                "switch_ACTION4",
            ],
            "execution_performed": execution,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "threshold_consolidation_counted_as_confirmation": False,
            "stop_switch_effectiveness_counted_as_verdict": verdict,
        },
        "objective_threshold_candidate": {
            "threshold_consolidation_id": "m3_o3::bp35-0a0ad940::objective_threshold",
            "threshold_type": "PRE_TERMINAL_PREFIX_WINDOW",
            "game_id": "bp35-0a0ad940",
            "prefix_action": "ACTION6",
            "prefix_policy": "patch_similarity_soft_stale_action6_prefix",
            "safe_tested_prefix_max": 48,
            "early_terminal_prefix_min": 64,
            "critical_window": [49, 64],
            "next_experiment_recommendation": "REFINE_PREFIX_WINDOW",
            "recommended_refined_prefixes": [50, 54, 58, 60, 62, 63],
            "recommended_conditions": [
                "continue_action6",
                "stop_policy",
                "switch_ACTION3",
                "switch_ACTION4",
            ],
            "excluded_conditions": [
                {"condition_id": "switch_ACTION1"},
                {"condition_id": "switch_ACTION2"},
            ],
            "status": "UNRESOLVED",
            "revision_status": "CANDIDATE_ONLY",
            "support": support,
            "truth_status": "NOT_EVALUATED_BY_M3",
            "execution_performed": False,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": support,
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def _fake_result(cell, *, terminal=False, blocked=False):
    status = (
        "BLOCKED_CONTROL_UNAVAILABLE"
        if blocked
        else "EARLY_TERMINAL_DURING_PREFIX"
        if terminal
        else "STOP_PREFIX_ENDPOINT_OBSERVED"
        if cell.condition_id == "stop_policy"
        else "EXECUTED"
    )
    return {
        **cell.to_dict(),
        "status": status,
        "execution_performed": not (terminal or blocked),
        "early_terminal_during_prefix": terminal,
        "blocked_reason": "fixture_blocked" if blocked else None,
        "objective_metrics": {
            "final_game_state": "GAME_OVER" if terminal else "NOT_FINISHED",
            "terminal_state_after_rollout": terminal,
            "levels_completed_after_rollout": 0,
            "objective_progress_proxy": float(cell.prefix_length),
        },
        "local_diagnostic_metrics": {
            "local_effect_metric": cell.prefix_length,
            "repeated_action6_count": 1,
            "useful_action6_steps": cell.prefix_length,
        },
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 0 if terminal or blocked else 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_refined_window_executor_builds_24_cells_without_blocked_controls(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps(_threshold_payload()), encoding="utf-8")
    calls = []

    def fake_executor(cell):
        calls.append(cell)
        return _fake_result(cell)

    payload = executor.run_objective_refined_window_execution(
        threshold_consolidation_path=path,
        cell_executor=fake_executor,
    )

    summary = payload["summary"]
    assert summary["threshold_consolidations_consumed"] == 1
    assert summary["refined_prefixes"] == [50, 54, 58, 60, 62, 63]
    assert summary["refined_conditions"] == [
        "continue_action6",
        "stop_policy",
        "switch_ACTION3",
        "switch_ACTION4",
    ]
    assert summary["planned_refined_cells"] == 24
    assert summary["unique_execution_cells"] == 24
    assert len(calls) == 24
    assert {cell.condition_id for cell in calls} == {
        "continue_action6",
        "stop_policy",
        "switch_ACTION3",
        "switch_ACTION4",
    }
    assert "switch_ACTION1" not in {cell.condition_id for cell in calls}
    assert "switch_ACTION2" not in {cell.condition_id for cell in calls}
    assert summary["support"] == 0
    assert summary["stop_switch_effectiveness_counted_as_verdict"] is False


def test_refined_window_executor_detects_candidate_terminal_avoidance(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps(_threshold_payload()), encoding="utf-8")

    def fake_executor(cell):
        terminal = cell.prefix_length == 62 and cell.condition_id == "continue_action6"
        return _fake_result(cell, terminal=terminal)

    payload = executor.run_objective_refined_window_execution(
        threshold_consolidation_path=path,
        cell_executor=fake_executor,
    )

    summary = payload["summary"]
    assert summary["stop_switch_effectiveness_status"] == (
        "CANDIDATE_TERMINAL_AVOIDANCE_OBSERVED"
    )
    assert summary["prefixes_with_candidate_terminal_avoidance"] == [62]
    assert summary["support"] == 0
    assert payload["prefix_comparisons"][4]["support"] == 0


def test_refined_window_executor_rejects_source_support_execution_or_verdict(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_threshold_payload(support=1)), encoding="utf-8")
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(
        json.dumps(_threshold_payload(execution=True)),
        encoding="utf-8",
    )
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text(json.dumps(_threshold_payload(verdict=True)), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        executor.run_objective_refined_window_execution(
            threshold_consolidation_path=support_path,
            cell_executor=lambda cell: _fake_result(cell),
        )
    with pytest.raises(ValueError, match="consolidation-only"):
        executor.run_objective_refined_window_execution(
            threshold_consolidation_path=execution_path,
            cell_executor=lambda cell: _fake_result(cell),
        )
    with pytest.raises(ValueError, match="cannot be a verdict"):
        executor.run_objective_refined_window_execution(
            threshold_consolidation_path=verdict_path,
            cell_executor=lambda cell: _fake_result(cell),
        )
