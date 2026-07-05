import json

import pytest

from theory.m2.arc_lewm_hypothesis_adapter import run_arc_lewm_hypothesis_adapter
from theory.m2.testability_compiler import requests_from_payload
from theory.m2.validators import validate_m3_request


def _signal_row(game_id, step, action, score, terminal=False):
    return {
        "signal_id": f"m2_14d::{game_id}::{step}",
        "game_id": game_id,
        "episode_id": f"{game_id}::episode",
        "step": step,
        "action": action,
        "action_args": {"x": step + 1, "y": step + 2} if action == "ACTION6" else {},
        "available_actions_t": ["ACTION3", "ACTION4", "ACTION6"],
        "terminal_t1": terminal,
        "level_delta": 0,
        "latent_surprise_score": score,
        "latent_delta_norm": score * 2,
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M2",
        "world_model_score_counted_as_support": False,
    }


def _signal_report(**overrides):
    report = {
        "schema_version": "m2.arc_lewm_signal_report.v1",
        "latent_prediction_quality": {
            "prediction_loss_val": 0.25,
            "collapse_detected": False,
            "latent_variance_above_min": True,
            "nan_or_inf_detected": False,
        },
        "candidate_signal_families": [
            "high_surprise_transitions",
            "low_surprise_stable_transitions",
            "action_conditioned_delta_clusters",
            "terminal_like_latent_neighborhoods",
            "proxy_completion_gap_candidates",
        ],
        "signals": {
            "high_surprise_transitions": [
                _signal_row("bp35-0a0ad940", 0, "RESET", 0.9)
            ],
            "low_surprise_stable_transitions": [
                _signal_row("bp35-0a0ad940", 8, "ACTION3", 0.1)
            ],
            "action_conditioned_delta_clusters": [
                {
                    "cluster_id": "m2_14d::action_delta::RESET",
                    "action": "RESET",
                    "transitions": 4,
                    "latent_surprise_mean": 0.6,
                    "latent_delta_norm_mean": 1.2,
                    "support": 0,
                    "truth_status": "NOT_EVALUATED_BY_M2",
                    "world_model_score_counted_as_support": False,
                }
            ],
            "terminal_like_latent_neighborhoods": {
                "terminal_transition_count": 1,
                "highest_surprise_terminal_transitions": [
                    _signal_row("ft09-0d8bbf25", 11, "ACTION6", 0.7, terminal=True)
                ],
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M2",
                "world_model_score_counted_as_support": False,
            },
            "proxy_completion_gap_candidates": [
                _signal_row("bp35-0a0ad940", 0, "RESET", 0.8)
            ],
        },
        "summary": {
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
            "world_model_counted_as_evidence": False,
            "world_model_score_counted_as_support": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "ready_for_a32": False,
            "ready_for_a33": False,
        },
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M2",
        "world_model_counted_as_evidence": False,
        "world_model_score_counted_as_support": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "ready_for_a32": False,
        "ready_for_a33": False,
    }
    report.update(overrides)
    return report


def test_arc_lewm_hypothesis_adapter_generates_m2_and_m3_candidates(tmp_path):
    signal_path = tmp_path / "signal_report.json"
    hypotheses_path = tmp_path / "arc_lewm_hypotheses.json"
    m3_path = tmp_path / "arc_lewm_m3_candidate_requests.json"
    signal_path.write_text(json.dumps(_signal_report()), encoding="utf-8")

    payloads = run_arc_lewm_hypothesis_adapter(
        signal_report_path=signal_path,
        hypotheses_output_path=hypotheses_path,
        m3_requests_output_path=m3_path,
    )

    assert hypotheses_path.exists()
    assert m3_path.exists()
    hypotheses_payload = payloads["hypotheses_payload"]
    assert hypotheses_payload["summary"]["hypotheses_generated"] == 4
    assert hypotheses_payload["summary"]["testable_hypotheses"] == 1
    assert hypotheses_payload["summary"]["support"] == 0
    assert hypotheses_payload["summary"]["truth_status"] == "NOT_EVALUATED_BY_M2"
    assert hypotheses_payload["summary"]["world_model_counted_as_evidence"] is False
    assert hypotheses_payload["summary"]["world_model_score_counted_as_support"] is False
    assert hypotheses_payload["summary"]["a32_write_performed"] is False
    assert hypotheses_payload["summary"]["a33_write_performed"] is False

    hypotheses = hypotheses_payload["arc_lewm_hypothesis_batches"][0][
        "candidate_hypotheses"
    ]
    families = {row["hypothesis_family"] for row in hypotheses}
    assert families == {
        "latent_surprise_frontier",
        "proxy_completion_gap_candidate",
        "action_conditioned_delta_cluster",
        "terminal_like_latent_neighborhood",
    }
    for row in hypotheses:
        assert row["status"] == "UNRESOLVED"
        assert row["support"] == 0
        assert row["controlled_test_required"] is True
        assert row["truth_status"] == "NOT_EVALUATED_BY_M2"
        assert row["revision_status"] == "CANDIDATE_ONLY"
        assert row["world_model_score_counted_as_support"] is False
        assert row["world_model_counted_as_evidence"] is False
        assert row["source_generation"]["priority_score_counted_as_support"] is False
        assert row["ready_for_a32"] is False
        assert row["ready_for_a33"] is False
    ready_hypotheses = [
        row
        for row in hypotheses
        if row["ready_for_m3_candidate_experiment_request"]
    ]
    assert len(ready_hypotheses) == 1
    assert ready_hypotheses[0]["testability"]["target_action"] == "ACTION6"
    blocking_reasons = {
        row["testability"]["blocking_reason"]
        for row in hypotheses
        if not row["ready_for_m3_candidate_experiment_request"]
    }
    assert blocking_reasons == {
        "BLOCKED_RESET_BOUNDARY",
        "DIAGNOSTIC_ONLY_AGGREGATE",
    }

    requests = requests_from_payload(payloads["m3_payload"])
    assert len(requests) == 4
    ready_requests = [
        request for request in requests if request.status == "READY_FOR_M3"
    ]
    blocked_requests = [
        request for request in requests if request.status == "BLOCKED_NOT_TESTABLE"
    ]
    assert len(ready_requests) == 1
    assert len(blocked_requests) == 3
    assert all(request.target_action != "RESET" for request in ready_requests)
    assert all(request.game_id != "unknown_game" for request in ready_requests)
    ready_request = ready_requests[0]
    assert ready_request.target_action == "ACTION6"
    assert ready_request.target_action_args == {"x": 12, "y": 13}
    assert ready_request.context_replay == ()
    assert ready_request.context_replay_args is None
    assert ready_request.source_episode_id == "ft09-0d8bbf25::episode"
    assert ready_request.source_step == 11
    assert ready_request.source_transition_id is not None
    assert ready_request.context_state_origin == "human_trace_frame_before"
    assert ready_request.replayability == "OFFLINE_TRACE_CONTEXT_ONLY"
    assert {
        request.blocking_reason for request in blocked_requests
    } == {
        "BLOCKED_RESET_BOUNDARY",
        "DIAGNOSTIC_ONLY_AGGREGATE",
    }
    assert all(validate_m3_request(request).valid for request in requests)
    assert payloads["m3_payload"]["summary"]["ready_for_m3"] == 1
    assert payloads["m3_payload"]["summary"]["blocked_not_testable"] == 3
    assert (
        payloads["m3_payload"]["summary"]["reset_actions_marked_ready_for_m3"]
        == 0
    )
    assert (
        payloads["m3_payload"]["summary"]["unknown_game_marked_ready_for_m3"]
        == 0
    )
    assert payloads["m3_payload"]["summary"]["support"] == 0
    assert payloads["m3_payload"]["summary"]["world_model_score_counted_as_support"] is False


def test_arc_lewm_hypothesis_adapter_rejects_bad_signal_report(tmp_path):
    signal_path = tmp_path / "bad_signal_report.json"
    signal_path.write_text(
        json.dumps(_signal_report(support=1)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        run_arc_lewm_hypothesis_adapter(signal_report_path=signal_path)


def test_arc_lewm_hypothesis_adapter_rejects_world_model_evidence_flag(tmp_path):
    signal_path = tmp_path / "bad_signal_report.json"
    signal_path.write_text(
        json.dumps(_signal_report(world_model_counted_as_evidence=True)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="world_model_counted_as_evidence"):
        run_arc_lewm_hypothesis_adapter(signal_report_path=signal_path)
