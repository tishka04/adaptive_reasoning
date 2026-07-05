import json

from theory.m2.schema import FalsificationCriterion, M3CandidateExperimentRequest
from theory.m3.offline_frame_counterfactual_probe import (
    ACTION_UNAVAILABLE_BLOCK,
    ENV_RESTORE_OR_COLLECTION_BLOCK,
    FEASIBLE_FOR_EXECUTOR,
    FRONTIER_TYPE,
    run_offline_frame_counterfactual_probe,
)


def _request(
    *,
    source_transition_id="m2_14d::ar25-e3c63847::ep-a::0001",
    episode_id="ep-a",
    step=1,
    target_action="ACTION3",
):
    return M3CandidateExperimentRequest(
        request_id="m2m3::m2_15_fused::terminal_safe_alternative_action::002",
        source_hypothesis_id="m2_15_fused::terminal_safe_alternative_action::002",
        game_id="ar25-e3c63847",
        context_replay=(),
        context_replay_args=None,
        context_snapshot_hash=source_transition_id,
        target_action=target_action,
        target_action_args=None,
        suggested_control_actions=("ACTION4", "ACTION1"),
        control_policy="m3_dynamic_available_controls",
        metric="terminal_state_after_rollout",
        expected_signal="alternative_action_terminal_rate_below_risk_target",
        falsification_criterion=FalsificationCriterion(
            metric="terminal_state_after_rollout",
            support_condition="alternative_terminal_rate < risk_target_terminal_rate",
            failure_condition="alternative_terminal_rate >= risk_target_terminal_rate",
            minimum_effect_size=1,
        ),
        status="BLOCKED_NOT_TESTABLE",
        source_episode_id=episode_id,
        source_step=step,
        source_transition_id=source_transition_id,
        context_state_origin="human_trace_frame_before",
        replayability="OFFLINE_TRACE_CONTEXT_ONLY",
        blocking_reason="BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME",
    )


def _payload(requests):
    return {
        "config": {"schema_version": "m2.fused_llm_wm_to_m3.v1"},
        "experiment_requests": [request.to_dict() for request in requests],
        "summary": {
            "experiment_requests": len(requests),
            "blocked_not_testable": len(requests),
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
        },
    }


def _transition_row(
    *,
    game_id="ar25-e3c63847",
    episode_id="ep-a",
    step=1,
    action="ACTION1",
    available_actions=("ACTION1", "ACTION3", "ACTION4"),
    extra=None,
):
    row = {
        "game_id": game_id,
        "episode_id": episode_id,
        "step": step,
        "grid_t": [[0, 1], [0, 0]],
        "grid_t1": [[1, 1], [0, 0]],
        "action": action,
        "action_args": {},
        "available_actions_t": list(available_actions),
        "terminal_t1": True,
        "level_delta": 0,
        "source": {
            "source_path": "human_traces/ar25-e3c63847.steps.jsonl",
            "line_number": 10,
        },
    }
    if extra:
        row.update(extra)
    return row


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_counterfactual_probe_blocks_visual_frame_without_env_state(tmp_path):
    requests_path = tmp_path / "requests.json"
    dataset_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "feasibility.json"
    requests_path.write_text(json.dumps(_payload([_request()])), encoding="utf-8")
    _write_jsonl(dataset_path, [_transition_row()])

    payload = run_offline_frame_counterfactual_probe(
        m2_requests_path=requests_path,
        offline_trace_dataset_path=dataset_path,
        output_path=output_path,
    )

    assert output_path.exists()
    summary = payload["summary"]
    assert summary["counterfactual_requests_seen"] == 1
    assert summary["source_transitions_found"] == 1
    assert summary["frame_before_visual_contexts_present"] == 1
    assert summary["full_env_state_payloads_present"] == 0
    assert summary["blocked_counterfactual_requests"] == 1
    assert summary["feasible_counterfactual_requests"] == 0
    assert summary["active_collection_required"] is True
    assert summary["recommended_frontier_type"] == FRONTIER_TYPE
    assert summary["support"] == 0
    assert summary["a32_write_performed"] is False

    result = payload["counterfactual_probe_results"][0]
    assert result["target_action_would_be_counterfactual"] is True
    assert result["target_action_available_in_frame"] is True
    assert result["frame_before_is_visual_observation_only"] is True
    assert result["can_reconstruct_env_state"] is False
    assert result["can_replay_alternative_action"] is False
    assert result["replay_exact_hashable"] is False
    assert result["counterfactual_feasibility"] == ENV_RESTORE_OR_COLLECTION_BLOCK
    assert result["recommended_frontier"]["frontier_type"] == FRONTIER_TYPE
    assert result["recommended_frontier"]["frontier_write_performed"] is False


def test_counterfactual_probe_marks_synthetic_restore_payload_feasible(tmp_path):
    requests_path = tmp_path / "requests.json"
    dataset_path = tmp_path / "transitions.jsonl"
    requests_path.write_text(json.dumps(_payload([_request()])), encoding="utf-8")
    _write_jsonl(
        dataset_path,
        [
            _transition_row(
                extra={
                    "env_state": {
                        "format": "test_full_env_state_v1",
                        "state_hash": "abc123",
                        "payload": {"level": 2, "agent": [1, 1]},
                    },
                    "env_state_restore_contract": {
                        "api": "restore_from_full_env_state",
                        "deterministic": True,
                    },
                }
            )
        ],
    )

    payload = run_offline_frame_counterfactual_probe(
        m2_requests_path=requests_path,
        offline_trace_dataset_path=dataset_path,
    )

    summary = payload["summary"]
    assert summary["full_env_state_payloads_present"] == 1
    assert summary["restore_contracts_detected"] == 1
    assert summary["feasible_counterfactual_requests"] == 1
    assert summary["blocked_counterfactual_requests"] == 0
    assert summary["can_reconstruct_env_state"] is True
    assert summary["can_replay_alternative_action"] is True
    assert summary["replay_exact_hashable"] is True
    assert summary["active_collection_required"] is False

    result = payload["counterfactual_probe_results"][0]
    assert result["full_env_state_payload_key"] == "env_state"
    assert result["env_state_payload_hash"]
    assert result["counterfactual_feasibility"] == FEASIBLE_FOR_EXECUTOR
    assert result["recommended_frontier"] is None


def test_counterfactual_probe_blocks_unavailable_alternative_action(tmp_path):
    requests_path = tmp_path / "requests.json"
    dataset_path = tmp_path / "transitions.jsonl"
    requests_path.write_text(
        json.dumps(_payload([_request(target_action="ACTION7")])),
        encoding="utf-8",
    )
    _write_jsonl(
        dataset_path,
        [
            _transition_row(
                available_actions=("ACTION1", "ACTION3", "ACTION4"),
                extra={
                    "env_state": {"format": "test_full_env_state_v1"},
                    "env_state_restore_contract": {"api": "restore"},
                },
            )
        ],
    )

    payload = run_offline_frame_counterfactual_probe(
        m2_requests_path=requests_path,
        offline_trace_dataset_path=dataset_path,
    )

    result = payload["counterfactual_probe_results"][0]
    assert result["target_action_available_in_frame"] is False
    assert result["can_reconstruct_env_state"] is True
    assert result["can_replay_alternative_action"] is False
    assert result["counterfactual_feasibility"] == ACTION_UNAVAILABLE_BLOCK
    assert payload["summary"]["blocked_by_reason"] == {ACTION_UNAVAILABLE_BLOCK: 1}
