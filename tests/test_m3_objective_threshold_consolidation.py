import json

import pytest

import theory.m3.objective_threshold_consolidation as consolidation


def _condition(status, *, terminal=False, progress=0.0):
    return {
        "status": status,
        "final_game_state": "GAME_OVER" if terminal else "NOT_FINISHED",
        "terminal_state_after_rollout": terminal,
        "levels_completed_after_rollout": 0,
        "objective_progress_proxy": progress,
    }


def _cell(prefix, condition_id, status, *, terminal=False):
    return {
        "game_id": "bp35-0a0ad940",
        "prefix_length": prefix,
        "condition_id": condition_id,
        "status": status,
        "execution_performed": status
        in {"EXECUTED", "STOP_PREFIX_ENDPOINT_OBSERVED"},
        "early_terminal_during_prefix": status == "EARLY_TERMINAL_DURING_PREFIX",
        "objective_metrics": {
            "final_game_state": "GAME_OVER" if terminal else "NOT_FINISHED",
            "terminal_state_after_rollout": terminal,
            "levels_completed_after_rollout": 0,
            "objective_progress_proxy": float(prefix),
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _payload(*, support=0, duplicate_independent=False, verdict=False):
    rows = []
    cells = []
    for prefix in [6, 12, 24, 48]:
        conditions = {
            "continue_action6": _condition("EXECUTED", progress=prefix + 8.0),
            "stop_policy": _condition(
                "STOP_PREFIX_ENDPOINT_OBSERVED",
                progress=prefix,
            ),
            "switch_ACTION3": _condition("EXECUTED", progress=prefix),
            "switch_ACTION4": _condition("EXECUTED", progress=prefix + 7.0),
            "switch_ACTION1": _condition("BLOCKED_CONTROL_UNAVAILABLE", progress=prefix),
            "switch_ACTION2": _condition("BLOCKED_CONTROL_UNAVAILABLE", progress=prefix),
        }
        rows.append(
            {
                "prefix_length": prefix,
                "conditions": conditions,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
        )
        for condition_id, row in conditions.items():
            cells.append(
                _cell(
                    prefix,
                    condition_id,
                    row["status"],
                    terminal=row["terminal_state_after_rollout"],
                )
            )
    terminal_conditions = {
        condition_id: _condition(
            "EARLY_TERMINAL_DURING_PREFIX",
            terminal=True,
            progress=512.0,
        )
        for condition_id in [
            "continue_action6",
            "stop_policy",
            "switch_ACTION3",
            "switch_ACTION4",
            "switch_ACTION1",
            "switch_ACTION2",
        ]
    }
    rows.append(
        {
            "prefix_length": 64,
            "conditions": terminal_conditions,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
        }
    )
    for condition_id, row in terminal_conditions.items():
        cells.append(_cell(64, condition_id, row["status"], terminal=True))
    return {
        "config": {
            "prefix_policy": "patch_similarity_soft_stale_action6_prefix",
            "duplicate_execution_cells_counted_as_independent": False,
        },
        "summary": {
            "objective_requests_consumed": 4,
            "planned_condition_cells": 120,
            "unique_execution_cells": 30,
            "objective_cells_executed": 16,
            "blocked_cells": 14,
            "early_terminal_prefix_cells": 6,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "duplicate_execution_cells_counted_as_independent": duplicate_independent,
            "blocked_cells_counted_as_contradictions": False,
            "policy_result_counted_as_scientific_verdict": verdict,
        },
        "execution_cells": cells,
        "objective_outcome_table": rows,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "objective_result_counted_as_confirmation": False,
    }


def test_objective_threshold_consolidation_extracts_pre_terminal_window(tmp_path):
    path = tmp_path / "objective_results.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = consolidation.run_objective_threshold_consolidation(
        objective_results_path=path,
    )

    summary = payload["summary"]
    assert summary["objective_threshold_consolidations"] == 1
    assert summary["threshold_type"] == "PRE_TERMINAL_PREFIX_WINDOW"
    assert summary["safe_tested_prefix_max"] == 48
    assert summary["early_terminal_prefix_min"] == 64
    assert summary["critical_window"] == [49, 64]
    assert summary["next_experiment_recommendation"] == "REFINE_PREFIX_WINDOW"
    assert summary["recommended_refined_prefixes"] == [50, 54, 58, 60, 62, 63]
    assert summary["recommended_conditions"] == [
        "continue_action6",
        "stop_policy",
        "switch_ACTION3",
        "switch_ACTION4",
    ]
    assert {row["condition_id"] for row in summary["excluded_conditions"]} == {
        "switch_ACTION1",
        "switch_ACTION2",
    }
    assert summary["support"] == 0
    assert summary["truth_status"] == "NOT_EVALUATED_BY_M3"

    candidate = payload["objective_threshold_candidate"]
    assert candidate["stop_switch_effectiveness_status"] == "NOT_ESTABLISHED"
    assert candidate["blocked_controls_not_contradictions"] is True
    assert candidate["early_terminal_prefix_cells_not_stop_switch_tests"] is True
    assert candidate["threshold_consolidation_counted_as_confirmation"] is False


def test_objective_threshold_consolidation_handles_no_early_terminal(tmp_path):
    source = _payload()
    source["objective_outcome_table"] = source["objective_outcome_table"][:-1]
    source["execution_cells"] = [
        row
        for row in source["execution_cells"]
        if int(row.get("prefix_length", 0)) != 64
    ]
    path = tmp_path / "objective_results.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    payload = consolidation.run_objective_threshold_consolidation(
        objective_results_path=path,
    )

    candidate = payload["objective_threshold_candidate"]
    assert candidate["threshold_type"] == "NO_EARLY_TERMINAL_OBSERVED"
    assert candidate["safe_tested_prefix_max"] == 48
    assert candidate["early_terminal_prefix_min"] is None
    assert candidate["critical_window"] == []
    assert candidate["next_experiment_recommendation"] == "EXTEND_PREFIX_SEARCH"
    assert candidate["support"] == 0


def test_objective_threshold_consolidation_rejects_support_or_verdict_flags(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_payload(support=1)), encoding="utf-8")
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps(_payload(duplicate_independent=True)),
        encoding="utf-8",
    )
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text(json.dumps(_payload(verdict=True)), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        consolidation.run_objective_threshold_consolidation(
            objective_results_path=support_path,
        )
    with pytest.raises(ValueError, match="cannot count as independent"):
        consolidation.run_objective_threshold_consolidation(
            objective_results_path=duplicate_path,
        )
    with pytest.raises(ValueError, match="scientific verdict"):
        consolidation.run_objective_threshold_consolidation(
            objective_results_path=verdict_path,
        )
