import json

import theory.m3.m2_candidate_experiment_runner as runner
from theory.m2.schema import (
    FalsificationCriterion,
    M3CandidateExperimentRequest,
)


def _request(metric="local_patch_before_after", status="READY_FOR_M3"):
    return M3CandidateExperimentRequest(
        request_id="m2m3::after_ACTION3_live_after_ACTION6::h001",
        source_hypothesis_id="m2::after_ACTION3_live_after_ACTION6::h001",
        game_id="bp35-0a0ad940",
        context_replay=("ACTION6", "ACTION3"),
        context_replay_args=None,
        context_snapshot_hash=None,
        target_action="ACTION4",
        target_action_args=None,
        suggested_control_actions=("ACTION3", "ACTION6"),
        control_policy="m3_dynamic_available_controls",
        metric=metric,
        expected_signal="target_action_changes_local_patch_more_than_control",
        falsification_criterion=FalsificationCriterion(
            metric=metric,
            support_condition="target_action_signal > best_control_signal",
            failure_condition="target_action_signal <= best_control_signal",
            minimum_effect_size=1,
        ),
        status=status,
    )


def _offline_trace_request():
    return M3CandidateExperimentRequest(
        request_id="m2m3::m2_14_lewm::terminal_like_latent_neighborhoods::004",
        source_hypothesis_id="m2_14_lewm::terminal_like_latent_neighborhoods::004",
        game_id="dc22-4c9bff3e",
        context_replay=(),
        context_replay_args=None,
        context_snapshot_hash="m2_14d::dc22-4c9bff3e::ep-source::0002",
        target_action="ACTION6",
        target_action_args={"x": 45, "y": 30},
        suggested_control_actions=("ACTION3", "ACTION4"),
        control_policy="m3_dynamic_available_controls",
        metric="terminal_state_after_rollout",
        expected_signal="target_action_terminal_rate_exceeds_matched_dynamic_controls",
        falsification_criterion=FalsificationCriterion(
            metric="terminal_state_after_rollout",
            support_condition="target_action_terminal_rate > best_control_terminal_rate",
            failure_condition="target_action_terminal_rate <= best_control_terminal_rate",
            minimum_effect_size=1,
        ),
        status="READY_FOR_M3",
        source_episode_id="ep-source",
        source_step=2,
        source_transition_id="m2_14d::dc22-4c9bff3e::ep-source::0002",
        context_state_origin="human_trace_frame_before",
        replayability="OFFLINE_TRACE_CONTEXT_ONLY",
    )


def _payload(requests):
    return {
        "config": {"schema_version": "m2_to_m3.v1"},
        "experiment_requests": [request.to_dict() for request in requests],
    }


def _transition_row(
    *,
    episode_id,
    step,
    action,
    action_args=None,
    terminal=False,
):
    return {
        "game_id": "dc22-4c9bff3e",
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


def test_batch_runner_escalates_metrics_when_primary_local_metric_is_neutral(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "m3_candidate_experiment_requests.json"
    path.write_text(json.dumps(_payload([_request()])), encoding="utf-8")

    signatures_seen = []

    def fake_source_hypothesis_live_state_signature(payload, source_hypothesis_id):
        assert source_hypothesis_id == "m2::after_ACTION3_live_after_ACTION6::h001"
        return "live-signature-after-action6-action3"

    def fake_execute_contextual_m2_request(
        request,
        *,
        environments_dir=None,
        target_context_signature="",
    ):
        signatures_seen.append(target_context_signature)
        if request.metric == "local_patch_before_after":
            support = 0
            target_pixels = 47
            baseline_pixels = 1
            effect_size = 0.0
        elif request.metric == "changed_pixels":
            support = 1
            target_pixels = 47
            baseline_pixels = 1
            effect_size = 46.0
        elif request.metric == "object_positions_before_after":
            support = 1
            target_pixels = 47
            baseline_pixels = 1
            effect_size = 3.0
        elif request.metric == "topology_before_after":
            support = 1
            target_pixels = 47
            baseline_pixels = 1
            effect_size = 46.0
        else:
            support = 0
            target_pixels = 47
            baseline_pixels = 1
            effect_size = 0.0
        if request.metric == "object_positions_before_after":
            observed_baseline = {
                "metric": request.metric,
                "measurable": True,
                "changed_pixels": baseline_pixels,
                "moved_component_count": 2,
            }
            observed_perturbation = {
                "metric": request.metric,
                "measurable": True,
                "changed_pixels": target_pixels,
                "moved_component_count": 5,
            }
        elif request.metric == "contact_graph_before_after":
            observed_baseline = {
                "metric": request.metric,
                "measurable": True,
                "changed_pixels": baseline_pixels,
                "contact_pairs_added": [],
                "contact_pairs_removed": [],
            }
            observed_perturbation = {
                "metric": request.metric,
                "measurable": True,
                "changed_pixels": target_pixels,
                "contact_pairs_added": [],
                "contact_pairs_removed": [],
            }
        elif request.metric == "local_patch_before_after":
            observed_baseline = {
                "metric": request.metric,
                "measurable": True,
                "changed_pixels": baseline_pixels,
                "local_changed_pixels": 0,
            }
            observed_perturbation = {
                "metric": request.metric,
                "measurable": True,
                "changed_pixels": target_pixels,
                "local_changed_pixels": 0,
            }
        else:
            observed_baseline = {
                "metric": "unknown",
                "measurable": False,
                "changed_pixels": baseline_pixels,
            }
            observed_perturbation = {
                "metric": "unknown",
                "measurable": False,
                "changed_pixels": target_pixels,
            }
        return {
            "hypothesis_key": request.source_hypothesis_id,
            "game_id": request.game_id,
            "target_action": request.target_action,
            "control_action": "ACTION3",
            "context_replay": list(request.context_replay),
            "predicted_metric": request.metric,
            "observed_baseline": observed_baseline,
            "observed_perturbation": observed_perturbation,
            "delta": {"effect_size": effect_size, "direction": "support" if support else "neutral"},
            "support_events": support,
            "contradiction_events": 0,
            "controlled_experiments_run": 1,
            "support": 0,
            "revision_performed": False,
            "wrong_confirmations": 0,
        }

    monkeypatch.setattr(
        runner,
        "execute_contextual_m2_request",
        fake_execute_contextual_m2_request,
    )
    monkeypatch.setattr(
        runner,
        "source_hypothesis_live_state_signature",
        fake_source_hypothesis_live_state_signature,
    )

    payload = runner.run_m2_candidate_experiment_queue(
        m2_requests_path=path,
        secondary_metrics=(
            "changed_pixels",
            "object_positions_before_after",
            "contact_graph_before_after",
            "topology_before_after",
        ),
    )

    summary = payload["summary"]
    assert summary["m2_requests_ready_for_m3"] == 1
    assert summary["primary_metric_experiments"] == 1
    assert summary["secondary_metric_experiments"] == 4
    assert summary["metric_escalations"] == 1
    assert summary["support_events"] == 2
    assert summary["neutral_events"] == 3
    assert summary["diagnostic_only_experiments"] == 1
    assert summary["grounding_suppressed_support_events"] == 1
    assert summary["raw_changed_pixels_experiments"] == 1
    assert summary["support"] == 0
    assert summary["a32_remains_only_verdict_location"] is True
    assert signatures_seen == ["live-signature-after-action6-action3"] * 5
    assert payload["updated_candidate_records"][0]["support"] == 0
    assert payload["updated_candidate_records"][0]["support_events"] == 2
    by_metric = {row["metric"]: row for row in payload["controlled_experiments"]}
    assert by_metric["changed_pixels"]["signal_source"] == "raw_changed_pixels"
    assert by_metric["changed_pixels"]["observed_baseline"]["measurable"] is True
    assert by_metric["object_positions_before_after"]["signal_source"] == "metric_extractor"
    assert by_metric["topology_before_after"]["diagnostic_only"] is True
    assert by_metric["topology_before_after"]["support_events"] == 0
    assert by_metric["topology_before_after"]["raw_support_events"] == 1


def test_batch_runner_executes_offline_trace_context_request(tmp_path):
    requests_path = tmp_path / "arc_lewm_m3_candidate_requests_v2.json"
    dataset_path = tmp_path / "m2_arc_lewm_transitions.jsonl"
    requests_path.write_text(
        json.dumps(_payload([_offline_trace_request()])),
        encoding="utf-8",
    )
    rows = [
        _transition_row(
            episode_id="ep-source",
            step=2,
            action="ACTION6",
            action_args={"x": 45, "y": 30},
            terminal=True,
        ),
        _transition_row(
            episode_id="ep-control-a",
            step=5,
            action="ACTION3",
            terminal=False,
        ),
        _transition_row(
            episode_id="ep-control-b",
            step=6,
            action="ACTION3",
            terminal=False,
        ),
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    payload = runner.run_m2_candidate_experiment_queue(
        m2_requests_path=requests_path,
        offline_trace_dataset_path=dataset_path,
        secondary_metrics=(),
    )

    summary = payload["summary"]
    assert summary["m2_requests_ready_for_m3"] == 1
    assert summary["m2_requests_executed"] == 1
    assert summary["controlled_experiments_run"] == 1
    assert summary["support_events"] == 1
    assert summary["support"] == 0
    assert summary["a32_remains_only_verdict_location"] is True
    experiment = payload["controlled_experiments"][0]
    assert experiment["request_id"] == "m2m3::m2_14_lewm::terminal_like_latent_neighborhoods::004"
    assert experiment["execution_mode"] == "offline_trace_context"
    assert experiment["source_transition_id"] == "m2_14d::dc22-4c9bff3e::ep-source::0002"
    assert experiment["target_action_args"] == {"x": 45, "y": 30}
    assert experiment["control_action"] == "ACTION3"
    assert experiment["matched_control_samples"] == 2
    assert experiment["observed_perturbation"]["terminal_rate"] == 1.0
    assert experiment["observed_baseline"]["terminal_rate"] == 0.0
    assert experiment["delta"]["direction"] == "support"
    assert experiment["metric_grounding_status"] == "grounded_metric_extractor"


def test_batch_runner_skips_blocked_m2_requests(monkeypatch, tmp_path):
    path = tmp_path / "m3_candidate_experiment_requests.json"
    path.write_text(json.dumps(_payload([_request(status="BLOCKED_NOT_TESTABLE")])), encoding="utf-8")

    payload = runner.run_m2_candidate_experiment_queue(m2_requests_path=path)

    assert payload["summary"]["m2_requests_ready_for_m3"] == 0
    assert payload["summary"]["controlled_experiments_run"] == 0
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
