import json

import pytest

import theory.m3.objective_stop_switch_experiment_executor as executor


def _condition(condition_id, policy, action=None):
    return {
        "condition_id": condition_id,
        "condition_family": (
            "continue_local_affordance"
            if condition_id == "continue_action6"
            else "stop_or_noop"
            if condition_id == "stop_policy"
            else "switch_subgoal"
        ),
        "post_prefix_policy": policy,
        "post_prefix_action": action,
        "role": "test_condition",
    }


def _request(request_id, hypothesis_id, family="stop_switch_criterion"):
    return {
        "request_id": request_id,
        "source_hypothesis_id": hypothesis_id,
        "source_request_id": "p2_o1::bp35-0a0ad940::objective_alignment::001",
        "game_id": "bp35-0a0ad940",
        "hypothesis_family": family,
        "prefix_action": "ACTION6",
        "prefix_lengths": [6, 12],
        "experimental_conditions": [
            _condition("continue_action6", "continue_action", "ACTION6"),
            _condition("stop_policy", "stop_or_hold_if_available", None),
        ],
        "status": "READY_FOR_M3_OBJECTIVE_EXPERIMENT",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "objective_request_counted_as_support": False,
        "policy_result_counted_as_scientific_verdict": False,
    }


def _payload(requests, *, support=0, execution=False, verdict=False):
    return {
        "summary": {
            "objective_experiment_requests_generated": len(requests),
            "planned_condition_cells": len(requests) * 4,
            "execution_performed": execution,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "objective_request_counted_as_support": False,
            "policy_result_counted_as_scientific_verdict": verdict,
        },
        "objective_stop_switch_experiment_requests": list(requests),
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "execution_performed": execution,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _fake_result(cell, *, status=executor.OBJECTIVE_EXECUTED):
    executed = status in {executor.OBJECTIVE_EXECUTED, executor.OBJECTIVE_STOP_OBSERVED}
    return {
        **cell.to_dict(),
        "status": status,
        "execution_performed": executed,
        "prefix_steps_executed": cell.prefix_length if executed else 2,
        "post_prefix_action_executed": bool(
            executed and cell.condition_id != "stop_policy"
        ),
        "early_terminal_during_prefix": status
        == executor.EARLY_TERMINAL_DURING_PREFIX,
        "blocked_reason": None if executed else "fixture_blocked",
        "objective_metrics": {
            "final_game_state": "NOT_FINISHED" if executed else "GAME_OVER",
            "terminal_state_after_rollout": not executed,
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
        "neutral_events": 1 if executed else 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_objective_executor_deduplicates_shared_execution_cells(tmp_path):
    path = tmp_path / "objective_requests.json"
    requests = [
        _request("m3_o1::h001", "m2_o1::h001"),
        _request("m3_o1::h002", "m2_o1::h002", family="terminal_risk_predictor"),
    ]
    path.write_text(json.dumps(_payload(requests)), encoding="utf-8")
    calls = []

    def fake_executor(cell):
        calls.append(cell.cell_signature)
        return _fake_result(cell)

    payload = executor.run_objective_stop_switch_experiment_execution(
        objective_requests_path=path,
        cell_executor=fake_executor,
    )

    summary = payload["summary"]
    assert summary["objective_requests_consumed"] == 2
    assert summary["planned_condition_cells"] == 8
    assert summary["unique_execution_cells"] == 4
    assert summary["duplicate_execution_cells"] == 4
    assert summary["duplicate_execution_cells_counted_as_independent"] is False
    assert len(calls) == 4
    assert summary["hypothesis_observation_links"] == 8
    assert summary["objective_cells_executed"] == 4
    assert summary["support_events"] == 0
    assert summary["contradiction_events"] == 0
    assert summary["neutral_events"] == 4
    assert summary["support"] == 0
    assert payload["hypothesis_observation_links"][0][
        "cell_link_counted_as_independent_execution"
    ] is False


def test_objective_executor_keeps_early_terminal_and_blocked_controls_non_verdict(
    tmp_path,
):
    path = tmp_path / "objective_requests.json"
    path.write_text(json.dumps(_payload([_request("m3_o1::h001", "m2_o1::h001")])), encoding="utf-8")

    def fake_executor(cell):
        if cell.condition_id == "continue_action6":
            return _fake_result(cell, status=executor.EARLY_TERMINAL_DURING_PREFIX)
        return _fake_result(cell, status=executor.BLOCKED_CONTROL_UNAVAILABLE)

    payload = executor.run_objective_stop_switch_experiment_execution(
        objective_requests_path=path,
        cell_executor=fake_executor,
    )

    summary = payload["summary"]
    assert summary["unique_execution_cells"] == 4
    assert summary["objective_cells_executed"] == 0
    assert summary["blocked_cells"] == 4
    assert summary["early_terminal_prefix_cells"] == 2
    assert summary["blocked_cells_counted_as_contradictions"] is False
    assert summary["contradiction_events"] == 0
    assert summary["support"] == 0
    assert all(
        row["support"] == 0 for row in payload["hypothesis_observation_links"]
    )


def test_objective_executor_rejects_source_support_execution_or_verdict(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(_payload([_request("m3_o1::h001", "m2_o1::h001")], support=1)),
        encoding="utf-8",
    )
    executed_path = tmp_path / "executed.json"
    executed_path.write_text(
        json.dumps(_payload([_request("m3_o1::h001", "m2_o1::h001")], execution=True)),
        encoding="utf-8",
    )
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text(
        json.dumps(_payload([_request("m3_o1::h001", "m2_o1::h001")], verdict=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        executor.run_objective_stop_switch_experiment_execution(
            objective_requests_path=support_path,
            cell_executor=lambda cell: _fake_result(cell),
        )
    with pytest.raises(ValueError, match="planning-only"):
        executor.run_objective_stop_switch_experiment_execution(
            objective_requests_path=executed_path,
            cell_executor=lambda cell: _fake_result(cell),
        )
    with pytest.raises(ValueError, match="scientific verdict"):
        executor.run_objective_stop_switch_experiment_execution(
            objective_requests_path=verdict_path,
            cell_executor=lambda cell: _fake_result(cell),
        )


def test_execution_cell_signature_omits_request_identity():
    left = executor.execution_cell_signature(
        game_id="bp35-0a0ad940",
        prefix_policy=executor.PREFIX_POLICY,
        prefix_action="ACTION6",
        prefix_length=6,
        condition_id="continue_action6",
        post_prefix_policy="continue_action",
        post_prefix_action="ACTION6",
        tie_break_seed=0,
    )
    right = executor.execution_cell_signature(
        game_id="bp35-0a0ad940",
        prefix_policy=executor.PREFIX_POLICY,
        prefix_action="ACTION6",
        prefix_length=6,
        condition_id="continue_action6",
        post_prefix_policy="continue_action",
        post_prefix_action="ACTION6",
        tie_break_seed=0,
    )

    assert left == right
    assert "h001" not in left
    assert "h002" not in left
