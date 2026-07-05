import json

from theory.m2.schema import FalsificationCriterion, M3CandidateExperimentRequest
from theory.m3.fused_llm_wm_experiment_runner import (
    annotate_fused_experiments,
    run_fused_llm_wm_experiment_runner,
    summarize_fused_experiments,
)


def _request(
    *,
    request_id,
    game_id,
    source_transition_id,
    episode_id,
    step,
    target_action,
    target_action_args=None,
    controls=("ACTION3", "ACTION4"),
    status="READY_FOR_M3",
    blocking_reason=None,
):
    return M3CandidateExperimentRequest(
        request_id=request_id,
        source_hypothesis_id=request_id.replace("m2m3::", "m2_"),
        game_id=game_id,
        context_replay=(),
        context_replay_args=None,
        context_snapshot_hash=source_transition_id,
        target_action=target_action,
        target_action_args=dict(target_action_args or {}) if target_action_args else None,
        suggested_control_actions=tuple(controls),
        control_policy="m3_dynamic_available_controls",
        metric="terminal_state_after_rollout",
        expected_signal="fused_signal_exceeds_matched_controls",
        falsification_criterion=FalsificationCriterion(
            metric="terminal_state_after_rollout",
            support_condition="target_action_terminal_rate > matched_control_terminal_rate",
            failure_condition="target_action_terminal_rate <= matched_control_terminal_rate",
            minimum_effect_size=1,
        ),
        status=status,
        source_episode_id=episode_id,
        source_step=step,
        source_transition_id=source_transition_id,
        context_state_origin="human_trace_frame_before",
        replayability="OFFLINE_TRACE_CONTEXT_ONLY",
        blocking_reason=blocking_reason,
    )


def _payload(requests):
    return {
        "config": {"schema_version": "m2.fused_llm_wm_to_m3.v1"},
        "experiment_requests": [request.to_dict() for request in requests],
        "summary": {
            "experiment_requests": len(requests),
            "ready_for_m3": len([r for r in requests if r.status == "READY_FOR_M3"]),
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
        },
    }


def _transition_row(
    *,
    game_id,
    episode_id,
    step,
    action,
    terminal=False,
    action_args=None,
    available_actions=None,
):
    return {
        "game_id": game_id,
        "episode_id": episode_id,
        "step": step,
        "grid_t": [[0, 1], [0, 0]],
        "grid_t1": [[1, 1], [0, 0]] if terminal else [[0, 1], [1, 0]],
        "action": action,
        "action_args": dict(action_args or {}),
        "available_actions_t": list(
            available_actions or ["ACTION1", "ACTION3", "ACTION4", "ACTION6"]
        ),
        "terminal_t1": terminal,
        "level_delta": 0,
    }


def _m3_7d_results(source_ids):
    return {
        "summary": {
            "support": 0,
            "support_events": len(source_ids),
            "truth_status": "NOT_EVALUATED_BY_M2",
        },
        "controlled_experiments": [
            {
                "request_id": f"m2m3::m2_14_lewm::terminal_risk_replication::{i:03d}",
                "source_hypothesis_id": f"m2_14_lewm::{i:03d}",
                "source_transition_id": source_id,
                "support_events": 1,
                "contradiction_events": 0,
                "matched_control_samples": 2,
                "support": 0,
            }
            for i, source_id in enumerate(source_ids, start=1)
        ],
    }


def test_annotate_fused_experiments_marks_m3_7d_reuse():
    experiments = [
        {
            "request_id": "m2m3::m2_15_fused::terminal_risk_precondition::001",
            "source_transition_id": "source-a",
            "metric": "terminal_state_after_rollout",
            "execution_mode": "offline_trace_context",
            "support_events": 1,
        },
        {
            "request_id": "m2m3::m2_15_fused::wm_llm_disagreement_frontier::003",
            "source_transition_id": "source-new",
            "metric": "terminal_state_after_rollout",
            "execution_mode": "offline_trace_context",
            "support_events": 1,
        },
    ]

    annotated = annotate_fused_experiments(
        experiments,
        reuse_index={"source-a": {"m3_7d_request_id": "m3.7d::001"}},
    )

    assert annotated[0]["source_transition_reused_from_m3_7d"] is True
    assert annotated[0]["new_independent_terminal_risk_evidence"] is False
    assert annotated[0]["m3_7d_reuse_reference"]["m3_7d_request_id"] == "m3.7d::001"
    assert annotated[1]["source_transition_reused_from_m3_7d"] is False
    assert annotated[1]["new_independent_terminal_risk_evidence"] is True


def test_summarize_fused_experiments_separates_reused_from_independent():
    experiments = [
        {
            "source_transition_id": "source-a",
            "support_events": 1,
            "contradiction_events": 0,
            "neutral_events": 0,
            "controlled_experiments_run": 1,
            "matched_control_samples": 10,
            "target_trace_samples": 1,
            "matched_control_policy": "same_game_same_available_actions",
            "source_transition_reused_from_m3_7d": True,
            "new_independent_terminal_risk_evidence": False,
            "fusion_hypothesis_routing_validated": True,
        },
        {
            "source_transition_id": "source-b",
            "support_events": 1,
            "contradiction_events": 0,
            "neutral_events": 0,
            "controlled_experiments_run": 1,
            "matched_control_samples": 11,
            "target_trace_samples": 1,
            "matched_control_policy": "same_game_same_available_actions",
            "source_transition_reused_from_m3_7d": False,
            "new_independent_terminal_risk_evidence": True,
            "fusion_hypothesis_routing_validated": True,
        },
    ]

    summary = summarize_fused_experiments(
        experiments,
        skipped_blocked_requests=[{"request_id": "blocked"}],
    )

    assert summary["support_events"] == 2
    assert summary["independent_source_support_events"] == 1
    assert summary["reused_source_support_events"] == 1
    assert summary["new_independent_terminal_risk_evidence"] is True
    assert summary["source_transition_reused_from_m3_7d"] is False
    assert summary["fused_requests_skipped_blocked"] == 1


def test_fused_runner_executes_ready_requests_and_marks_reused_sources(tmp_path):
    requests_path = tmp_path / "fused_requests.json"
    dataset_path = tmp_path / "transitions.jsonl"
    m3_7d_path = tmp_path / "m3_7d.json"
    output_path = tmp_path / "fused_results.json"
    source_ids = [
        "m2_14d::ar25-e3c63847::ep-a::0001",
        "m2_14d::bp35-0a0ad940::ep-b::0002",
        "m2_14d::bp35-0a0ad940::ep-c::0003",
    ]
    requests = [
        _request(
            request_id="m2m3::m2_15_fused::terminal_risk_precondition::001",
            game_id="ar25-e3c63847",
            source_transition_id=source_ids[0],
            episode_id="ep-a",
            step=1,
            target_action="ACTION1",
            controls=("ACTION3", "ACTION4"),
        ),
        _request(
            request_id="m2m3::m2_15_fused::terminal_safe_alternative_action::002",
            game_id="ar25-e3c63847",
            source_transition_id=source_ids[0],
            episode_id="ep-a",
            step=1,
            target_action="ACTION3",
            status="BLOCKED_NOT_TESTABLE",
            blocking_reason="BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME",
        ),
        _request(
            request_id="m2m3::m2_15_fused::wm_llm_disagreement_frontier::003",
            game_id="bp35-0a0ad940",
            source_transition_id=source_ids[1],
            episode_id="ep-b",
            step=2,
            target_action="ACTION3",
            controls=("ACTION4", "ACTION6"),
        ),
        _request(
            request_id="m2m3::m2_15_fused::objective_completion_vs_terminal_risk_tradeoff::004",
            game_id="bp35-0a0ad940",
            source_transition_id=source_ids[2],
            episode_id="ep-c",
            step=3,
            target_action="ACTION6",
            target_action_args={"x": 22, "y": 33},
            controls=("ACTION3", "ACTION4"),
        ),
    ]
    requests_path.write_text(json.dumps(_payload(requests)), encoding="utf-8")
    rows = [
        _transition_row(
            game_id="ar25-e3c63847",
            episode_id="ep-a",
            step=1,
            action="ACTION1",
            terminal=True,
        ),
        _transition_row(
            game_id="ar25-e3c63847",
            episode_id="ctrl-a",
            step=10,
            action="ACTION3",
            terminal=False,
        ),
        _transition_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-b",
            step=2,
            action="ACTION3",
            terminal=True,
        ),
        _transition_row(
            game_id="bp35-0a0ad940",
            episode_id="ctrl-b",
            step=11,
            action="ACTION4",
            terminal=False,
        ),
        _transition_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-c",
            step=3,
            action="ACTION6",
            action_args={"x": 22, "y": 33},
            terminal=True,
        ),
        _transition_row(
            game_id="bp35-0a0ad940",
            episode_id="ctrl-c",
            step=12,
            action="ACTION3",
            terminal=False,
        ),
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    m3_7d_path.write_text(json.dumps(_m3_7d_results(source_ids)), encoding="utf-8")

    payload = run_fused_llm_wm_experiment_runner(
        m2_requests_path=requests_path,
        m3_7d_results_path=m3_7d_path,
        offline_trace_dataset_path=dataset_path,
        output_path=output_path,
    )

    assert output_path.exists()
    summary = payload["summary"]
    assert summary["fused_requests_executed"] == 3
    assert summary["fused_requests_skipped_blocked"] == 1
    assert summary["support_events"] == 3
    assert summary["contradiction_events"] == 0
    assert summary["source_transition_reused_from_m3_7d"] is True
    assert summary["new_independent_terminal_risk_evidence"] is False
    assert summary["independent_source_support_events"] == 0
    assert summary["reused_source_support_events"] == 3
    assert summary["fusion_hypothesis_routing_validated"] is True
    assert summary["support"] == 0
    assert summary["a32_write_performed"] is False
    assert all(
        row["source_transition_reused_from_m3_7d"]
        for row in payload["controlled_experiments"]
    )
