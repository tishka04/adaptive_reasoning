import json

from theory.m2.fused_hypothesis_generator import (
    FUSED_HYPOTHESIS_FAMILIES,
    build_fused_input_packet,
    build_fused_m3_requests_payload,
    generate_fused_hypotheses,
    run_fused_hypothesis_generation,
)
from theory.m2.validators import validate_hypothesis, validate_m3_request


def _candidate_source(summary=None, **extra):
    payload = {
        "summary": {
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "a32_write_performed": False,
            "a33_write_performed": False,
            "revision_performed": False,
        },
        "support": 0,
    }
    payload["summary"].update(summary or {})
    payload.update(extra)
    return payload


def _llm_hypotheses():
    payload = _candidate_source(
        {
            "hypotheses_generated": 1,
            "ready_for_m3_candidate_experiment_request": 1,
            "llm_output_counted_as_evidence": False,
        }
    )
    payload["candidate_hypotheses"] = [
        {
            "hypothesis_id": "m2::p2g4::bp35::h001",
            "hypothesis_family": "objective_readiness_detection",
            "candidate_action": "ACTION3",
            "predicted_metric": "objective_completion_signal",
            "status": "UNRESOLVED",
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
        }
    ]
    return payload


def _terminal_experiment(
    *,
    request_id,
    game_id,
    target_action,
    source_transition_id,
    source_episode_id,
    source_step,
    target_action_args=None,
    control_action="ACTION3",
):
    return {
        "request_id": request_id,
        "game_id": game_id,
        "target_action": target_action,
        "target_action_args": dict(target_action_args or {}),
        "control_action": control_action,
        "source_transition_id": source_transition_id,
        "source_episode_id": source_episode_id,
        "source_step": source_step,
        "dynamic_available_actions": [
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION6",
        ],
        "matched_control_policy": "same_game_same_available_actions",
        "matched_control_samples": 10,
        "target_trace_samples": 1,
        "observed_perturbation": {
            "terminal_rate": 1.0,
            "terminal_state_after_rollout": True,
        },
        "observed_baseline": {"terminal_rate": 0.0},
        "metric": "terminal_state_after_rollout",
        "execution_mode": "offline_trace_context",
        "support_events": 1,
        "contradiction_events": 0,
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M2",
    }


def _m37d_results():
    rows = [
        _terminal_experiment(
            request_id="m2m3::m2_14_lewm::terminal_risk_replication::001",
            game_id="dc22-4c9bff3e",
            target_action="ACTION6",
            target_action_args={"x": 45, "y": 30},
            source_transition_id="m2_14d::dc22-4c9bff3e::ep-a::0001",
            source_episode_id="ep-a",
            source_step=1,
        ),
        _terminal_experiment(
            request_id="m2m3::m2_14_lewm::terminal_risk_replication::002",
            game_id="bp35-0a0ad940",
            target_action="ACTION3",
            source_transition_id="m2_14d::bp35-0a0ad940::ep-b::0002",
            source_episode_id="ep-b",
            source_step=2,
            control_action="ACTION4",
        ),
        _terminal_experiment(
            request_id="m2m3::m2_14_lewm::terminal_risk_replication::003",
            game_id="bp35-0a0ad940",
            target_action="ACTION6",
            target_action_args={"x": 22, "y": 31},
            source_transition_id="m2_14d::bp35-0a0ad940::ep-c::0003",
            source_episode_id="ep-c",
            source_step=3,
        ),
    ]
    payload = _candidate_source(
        {
            "support_events": 3,
            "contradiction_events": 0,
            "supporting_source_transition_ids": 3,
            "supporting_source_episodes": 3,
            "supporting_games": 2,
            "supporting_target_actions": 2,
            "all_controls_same_game_same_available_actions": True,
            "support": 0,
        }
    )
    payload["controlled_experiments"] = rows
    return payload


def _arc_signal_report():
    payload = _candidate_source(
        {
            "world_model_score_counted_as_support": False,
            "world_model_counted_as_evidence": False,
        }
    )
    payload["signals"] = {
        "terminal_like_latent_neighborhoods": {
            "terminal_transition_count": 3,
        }
    }
    return payload


def _sources():
    return {
        "llm_hypotheses": _llm_hypotheses(),
        "llm_m3_requests": _candidate_source({"ready_for_m3": 1}),
        "wm_invariant_packet": _candidate_source(),
        "arc_lewm_signal_report": _arc_signal_report(),
        "arc_lewm_hypotheses": _candidate_source({"hypotheses_generated": 4}),
        "arc_lewm_m3_requests": _candidate_source({"ready_for_m3": 1}),
        "m3_arc_lewm_single_result": _candidate_source({"support_events": 1}),
        "m3_arc_lewm_replication_results": _m37d_results(),
        "m3g6_results": _candidate_source(
            {
                "objective_completion_signal": False,
                "proxy_progress_without_completion_observed": True,
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
        ),
    }


def test_fused_input_packet_consumes_llm_wm_and_m3_grounding_context():
    packet = build_fused_input_packet(_sources())

    assert packet["allowed_hypothesis_families"] == list(FUSED_HYPOTHESIS_FAMILIES)
    assert packet["summary"]["terminal_risk_support_events_read"] == 3
    assert packet["summary"]["terminal_risk_support_events_counted_as_support"] is False
    assert packet["summary"]["support"] == 0
    assert packet["fusion_roles"]["llm"] == "abductive_hypothesis_generator_not_evidence"
    assert packet["world_model_context"]["terminal_transition_count"] == 3
    assert len(packet["m3_grounding_context"]["terminal_risk_observations"]) == 3


def test_fused_generator_creates_four_candidate_only_hypotheses():
    packet = build_fused_input_packet(_sources())
    hypotheses = generate_fused_hypotheses(packet)

    assert len(hypotheses) == 4
    assert {h.hypothesis_family for h in hypotheses} == set(FUSED_HYPOTHESIS_FAMILIES)
    assert all(validate_hypothesis(h).valid for h in hypotheses)
    for hypothesis in hypotheses:
        assert hypothesis.status == "UNRESOLVED"
        assert hypothesis.support == 0
        assert hypothesis.truth_status == "NOT_EVALUATED_BY_M2"
        assert set(hypothesis.source_generation.sources) == {"llm", "world_model"}
        assert hypothesis.source_generation.priority_score_counted_as_support is False
    blocked = [h for h in hypotheses if not h.testability.testable]
    assert len(blocked) == 1
    assert (
        blocked[0].testability.blocking_reason
        == "BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME"
    )


def test_fused_m3_handoff_preserves_offline_locators_and_blocks_counterfactual():
    packet = build_fused_input_packet(_sources())
    hypotheses = generate_fused_hypotheses(packet)
    payload = build_fused_m3_requests_payload(
        hypotheses,
        source_hypothesis_path="diagnostics/m2/fused_llm_wm_hypotheses.json",
    )

    summary = payload["summary"]
    assert summary["experiment_requests"] == 4
    assert summary["ready_for_m3"] == 3
    assert summary["blocked_not_testable"] == 1
    assert summary["offline_trace_context_requests"] == 3
    assert summary["support"] == 0
    requests = payload["experiment_requests"]
    assert all(validate_m3_request(request).valid for request in requests)
    ready = [row for row in requests if row["status"] == "READY_FOR_M3"]
    assert all(row["replayability"] == "OFFLINE_TRACE_CONTEXT_ONLY" for row in ready)
    assert all(row["source_transition_id"] for row in ready)
    assert all(row["context_replay_args"] is None for row in ready if row["context_replay"] == [])
    action6_ready = [row for row in ready if row["target_action"] == "ACTION6"]
    assert action6_ready
    assert all(row["target_action_args"] for row in action6_ready)
    blocked = [row for row in requests if row["status"] == "BLOCKED_NOT_TESTABLE"]
    assert blocked[0]["blocking_reason"] == (
        "BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME"
    )


def test_run_fused_hypothesis_generation_writes_all_artifacts(tmp_path):
    paths = {}
    for name, payload in _sources().items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    packet_out = tmp_path / "fused_llm_wm_input_packet.json"
    hypotheses_out = tmp_path / "fused_llm_wm_hypotheses.json"
    m3_out = tmp_path / "fused_llm_wm_m3_candidate_requests.json"
    handoff_out = tmp_path / "fused_llm_wm_handoff_validation.json"

    outputs = run_fused_hypothesis_generation(
        llm_hypotheses_path=paths["llm_hypotheses"],
        llm_m3_requests_path=paths["llm_m3_requests"],
        wm_invariant_packet_path=paths["wm_invariant_packet"],
        arc_lewm_signal_report_path=paths["arc_lewm_signal_report"],
        arc_lewm_hypotheses_path=paths["arc_lewm_hypotheses"],
        arc_lewm_m3_requests_path=paths["arc_lewm_m3_requests"],
        m3_arc_lewm_single_result_path=paths["m3_arc_lewm_single_result"],
        m3_arc_lewm_replication_results_path=paths[
            "m3_arc_lewm_replication_results"
        ],
        m3g6_results_path=paths["m3g6_results"],
        input_packet_output_path=packet_out,
        hypotheses_output_path=hypotheses_out,
        m3_requests_output_path=m3_out,
        handoff_validation_output_path=handoff_out,
    )

    assert packet_out.exists()
    assert hypotheses_out.exists()
    assert m3_out.exists()
    assert handoff_out.exists()
    assert outputs["hypotheses_payload"]["summary"]["hypotheses_generated"] == 4
    assert outputs["m3_payload"]["summary"]["ready_for_m3"] == 3
    assert outputs["handoff_validation_payload"]["summary"]["invalid_m3_requests"] == 0
