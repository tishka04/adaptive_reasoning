import json

from theory.m2.validators import validate_m3_request
from theory.m3.arc_lewm_terminal_risk_replication import (
    build_terminal_risk_replication_requests_payload,
    run_arc_lewm_terminal_risk_replication,
    summarize_replication_experiments,
)


def _signal_row(
    *,
    game_id,
    episode_id,
    step,
    action,
    action_args=None,
    available_actions=None,
    terminal=True,
):
    return {
        "signal_id": f"m2_14d::{game_id}::{episode_id}::{step:04d}",
        "game_id": game_id,
        "episode_id": episode_id,
        "step": step,
        "action": action,
        "action_args": dict(action_args or {}),
        "available_actions_t": list(
            available_actions or ["ACTION3", "ACTION4", "ACTION6"]
        ),
        "terminal_t1": terminal,
        "level_delta": 0,
        "latent_surprise_score": 0.9,
        "latent_delta_norm": 1.8,
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M2",
        "world_model_score_counted_as_support": False,
    }


def _signal_report(rows):
    return {
        "schema_version": "m2.arc_lewm_signal_report.v1",
        "signals": {
            "terminal_like_latent_neighborhoods": {
                "terminal_transition_count": len(rows),
                "highest_surprise_terminal_transitions": list(rows),
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M2",
                "world_model_score_counted_as_support": False,
            },
        },
        "summary": {
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
            "world_model_counted_as_evidence": False,
            "world_model_score_counted_as_support": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M2",
        "world_model_counted_as_evidence": False,
        "world_model_score_counted_as_support": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _transition_row(
    *,
    game_id,
    episode_id,
    step,
    action,
    action_args=None,
    terminal=False,
):
    return {
        "game_id": game_id,
        "episode_id": episode_id,
        "step": step,
        "grid_t": [[0, 1], [0, 0]],
        "grid_t1": [[1, 1], [0, 0]] if terminal else [[0, 1], [1, 0]],
        "action": action,
        "action_args": dict(action_args or {}),
        "available_actions_t": ["ACTION3", "ACTION4", "ACTION6"],
        "terminal_t1": terminal,
        "level_delta": 0,
    }


def test_terminal_risk_replication_planner_selects_only_trace_replayable_rows():
    rows = [
        _signal_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-a",
            step=1,
            action="ACTION6",
            action_args={"x": 45, "y": 30},
        ),
        _signal_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-a",
            step=1,
            action="ACTION6",
            action_args={"x": 45, "y": 30},
        ),
        _signal_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-reset",
            step=2,
            action="RESET",
        ),
        _signal_row(
            game_id="unknown_game",
            episode_id="ep-unknown",
            step=3,
            action="ACTION3",
        ),
        _signal_row(
            game_id="ft09-0d8bbf25",
            episode_id="ep-missing-args",
            step=4,
            action="ACTION6",
        ),
        _signal_row(
            game_id="ar25-e3c63847",
            episode_id="ep-not-live",
            step=5,
            action="ACTION5",
            available_actions=["ACTION1", "ACTION2"],
        ),
        _signal_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-b",
            step=6,
            action="ACTION4",
        ),
    ]

    payload = build_terminal_risk_replication_requests_payload(
        _signal_report(rows),
        max_requests=10,
        signal_report_path="diagnostics/m2/arc_lewm_signal_report.json",
    )

    summary = payload["summary"]
    assert summary["source_terminal_rows_selected"] == 2
    assert summary["experiment_requests"] == 2
    assert summary["ready_for_m3"] == 2
    assert summary["distinct_source_transition_ids"] == 2
    assert summary["distinct_source_episodes"] == 2
    assert summary["support"] == 0
    assert summary["world_model_score_counted_as_support"] is False
    assert summary["a32_write_performed"] is False
    requests = payload["experiment_requests"]
    assert requests[0]["target_action"] == "ACTION6"
    assert requests[0]["target_action_args"] == {"x": 45, "y": 30}
    assert requests[0]["replayability"] == "OFFLINE_TRACE_CONTEXT_ONLY"
    assert requests[0]["context_state_origin"] == "human_trace_frame_before"
    assert requests[1]["target_action"] == "ACTION4"
    assert all(validate_m3_request(request).valid for request in requests)


def test_arc_lewm_terminal_risk_replication_runs_offline_trace_batch(tmp_path):
    signal_path = tmp_path / "signal_report.json"
    requests_path = tmp_path / "replication_requests.json"
    output_path = tmp_path / "replication_results.json"
    dataset_path = tmp_path / "transitions.jsonl"
    signal_rows = [
        _signal_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-source-a",
            step=1,
            action="ACTION6",
            action_args={"x": 45, "y": 30},
        ),
        _signal_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-source-b",
            step=2,
            action="ACTION6",
            action_args={"x": 46, "y": 28},
        ),
        _signal_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-source-c",
            step=1,
            action="ACTION4",
        ),
    ]
    signal_path.write_text(json.dumps(_signal_report(signal_rows)), encoding="utf-8")
    dataset_rows = [
        _transition_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-source-a",
            step=1,
            action="ACTION6",
            action_args={"x": 45, "y": 30},
            terminal=True,
        ),
        _transition_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-source-b",
            step=2,
            action="ACTION6",
            action_args={"x": 46, "y": 28},
            terminal=True,
        ),
        _transition_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-control-a",
            step=3,
            action="ACTION3",
            terminal=False,
        ),
        _transition_row(
            game_id="dc22-4c9bff3e",
            episode_id="ep-control-b",
            step=4,
            action="ACTION3",
            terminal=False,
        ),
        _transition_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-source-c",
            step=1,
            action="ACTION4",
            terminal=True,
        ),
        _transition_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-control-c",
            step=2,
            action="ACTION3",
            terminal=True,
        ),
        _transition_row(
            game_id="bp35-0a0ad940",
            episode_id="ep-control-d",
            step=3,
            action="ACTION3",
            terminal=True,
        ),
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in dataset_rows),
        encoding="utf-8",
    )

    payload = run_arc_lewm_terminal_risk_replication(
        signal_report_path=signal_path,
        offline_trace_dataset_path=dataset_path,
        requests_output_path=requests_path,
        output_path=output_path,
        max_requests=3,
    )

    assert requests_path.exists()
    assert output_path.exists()
    summary = payload["summary"]
    assert summary["m2_requests_ready_for_m3"] == 3
    assert summary["m2_requests_executed"] == 3
    assert summary["replication_experiments"] == 3
    assert summary["support_events"] == 2
    assert summary["contradiction_events"] == 0
    assert summary["neutral_events"] == 1
    assert summary["independent_source_support_events"] == 2
    assert summary["duplicate_source_support_events"] == 0
    assert summary["unique_context_support_events"] == 1
    assert summary["context_reused_support_events"] == 1
    assert summary["supporting_source_episodes"] == 2
    assert summary["supporting_games"] == 1
    assert summary["target_trace_samples_total"] == 3
    assert summary["matched_control_samples_total"] == 6
    assert summary["all_controls_same_game_same_available_actions"] is True
    assert summary["replication_breadth_low"] is False
    assert summary["support"] == 0
    assert summary["a32_write_performed"] is False
    assert payload["status"] == "UNRESOLVED"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M2"


def test_replication_summary_separates_duplicate_and_contradiction_events():
    experiments = [
        {
            "support_events": 1,
            "contradiction_events": 0,
            "controlled_experiments_run": 1,
            "source_transition_id": "source-a",
            "source_episode_id": "ep-a",
            "game_id": "game-a",
            "target_action": "ACTION6",
            "control_action": "ACTION3",
            "dynamic_available_actions": ["ACTION3", "ACTION6"],
            "target_trace_samples": 1,
            "matched_control_samples": 10,
            "matched_control_policy": "same_game_same_available_actions",
            "metric_grounding_status": "grounded_metric_extractor",
        },
        {
            "support_events": 1,
            "contradiction_events": 0,
            "controlled_experiments_run": 1,
            "source_transition_id": "source-a",
            "source_episode_id": "ep-a",
            "game_id": "game-a",
            "target_action": "ACTION6",
            "control_action": "ACTION3",
            "dynamic_available_actions": ["ACTION3", "ACTION6"],
            "target_trace_samples": 1,
            "matched_control_samples": 10,
            "matched_control_policy": "same_game_same_available_actions",
            "metric_grounding_status": "grounded_metric_extractor",
        },
        {
            "support_events": 0,
            "contradiction_events": 1,
            "controlled_experiments_run": 1,
            "source_transition_id": "source-b",
            "source_episode_id": "ep-b",
            "game_id": "game-b",
            "target_action": "ACTION4",
            "control_action": "ACTION3",
            "dynamic_available_actions": ["ACTION3", "ACTION4"],
            "target_trace_samples": 1,
            "matched_control_samples": 7,
            "matched_control_policy": "same_game_same_available_actions",
            "metric_grounding_status": "grounded_metric_extractor",
        },
    ]

    summary = summarize_replication_experiments(experiments, blocked=())

    assert summary["support_events"] == 2
    assert summary["contradiction_events"] == 1
    assert summary["supporting_source_transition_ids"] == 1
    assert summary["independent_source_support_events"] == 1
    assert summary["duplicate_source_support_events"] == 1
    assert summary["unique_context_support_events"] == 1
    assert summary["context_reused_support_events"] == 1
    assert summary["contradicting_source_transition_ids"] == 1
    assert summary["target_trace_samples_total"] == 3
    assert summary["matched_control_samples_total"] == 27
