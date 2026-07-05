import json

from theory.sage.known_game_closed_loop_scaffold import (
    SAGE_TRUTH_STATUS,
    run_sage0_known_game_scaffold,
)


def _m2_fused_requests():
    return {
        "summary": {
            "experiment_requests": 4,
            "ready_for_m3": 3,
            "blocked_not_testable": 1,
            "llm_output_counted_as_evidence": False,
            "world_model_counted_as_evidence": False,
            "support": 0,
        }
    }


def _m3_fused_results():
    return {
        "summary": {
            "fused_requests_executed": 3,
            "fused_requests_skipped_blocked": 1,
            "fusion_hypothesis_routing_validated": True,
            "new_independent_terminal_risk_evidence": False,
            "support": 0,
        }
    }


def _m3_counterfactual_feasibility():
    return {
        "summary": {
            "counterfactual_requests_seen": 1,
            "feasible_counterfactual_requests": 0,
            "frontier_recommendations": 1,
            "recommended_frontier_type": (
                "NEED_ACTIVE_COUNTERFACTUAL_COLLECTION_FROM_TRACE_CONTEXT"
            ),
            "support": 0,
        }
    }


def _p1_policy_probe():
    return {
        "candidate_policy_counted_as_confirmation": False,
        "candidate_policy_memory": {
            "enabled": True,
            "ready_for_agent_policy_probe": True,
            "a33_ready": False,
            "target_action": "ACTION6",
            "repositioning_action": "ACTION4",
            "known_success_args": [
                {"x": 12, "y": 0},
                {"x": 24, "y": 0},
            ],
            "support": 0,
        },
        "summary": {
            "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
            "support": 0,
        },
    }


def _p1_utility_handoff():
    return {
        "summary": {
            "agentic_utility_status": "SUPPORTED_CANDIDATE_ONLY",
            "support": 0,
        },
        "a33_ready": False,
        "policy_result_counted_as_mechanistic_confirmation": False,
    }


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path):
    paths = {
        "m2": tmp_path / "m2_requests.json",
        "m3": tmp_path / "m3_fused.json",
        "m3_7f": tmp_path / "m3_7f.json",
        "p1": tmp_path / "p1_probe.json",
        "p1_handoff": tmp_path / "p1_handoff.json",
    }
    _write(paths["m2"], _m2_fused_requests())
    _write(paths["m3"], _m3_fused_results())
    _write(paths["m3_7f"], _m3_counterfactual_feasibility())
    _write(paths["p1"], _p1_policy_probe())
    _write(paths["p1_handoff"], _p1_utility_handoff())
    return paths


def test_sage0_scaffold_logs_closed_loop_and_counterfactual_boundary(tmp_path):
    paths = _write_inputs(tmp_path)
    out = tmp_path / "sage0.json"

    payload = run_sage0_known_game_scaffold(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        output_path=out,
        budget=2,
    )

    assert out.exists()
    summary = payload["summary"]
    assert summary["closed_loop_scaffold_completed"] is True
    assert summary["benchmark_run"] is False
    assert summary["env_steps_executed"] == 2
    assert summary["offline_counterfactual_allowed"] is False
    assert summary["active_counterfactual_collection_allowed"] is True
    assert summary["env_state_restore_available"] is False
    assert summary["blocked_capability_frontiers_logged"] is True
    assert summary["support"] == 0
    assert summary["truth_status"] == SAGE_TRUTH_STATUS
    assert summary["a32_write_performed"] is False
    assert summary["a33_write_performed"] is False

    logger = payload["logger_contract"]
    assert logger["offline_trace_context_role"] == "observation_diagnostic_grounding_only"
    assert logger["live_env_context_role"] == (
        "only_authorized_context_for_alternative_actions"
    )


def test_sage0_scaffold_selects_actions_from_live_context(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = run_sage0_known_game_scaffold(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        budget=2,
    )

    decisions = payload["action_decisions"]
    assert decisions[0]["action"] == "ACTION4"
    assert decisions[0]["selection_context"] == "live_env_context"
    assert decisions[0]["offline_counterfactual_allowed"] is False
    assert decisions[1]["action"] == "ACTION6"
    assert decisions[1]["action_args"] == {"x": 12, "y": 0}
    assert decisions[1]["active_counterfactual_collection_allowed"] is True

    steps = payload["env_steps"]
    assert all(step["live_env_context"] for step in steps)
    assert all(not step["offline_trace_context"] for step in steps)
    assert all(step["state_changed"] for step in steps)


def test_sage0_scaffold_preserves_candidate_only_input_roles(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = run_sage0_known_game_scaffold(
        m2_fused_requests_path=paths["m2"],
        m3_fused_results_path=paths["m3"],
        m3_counterfactual_feasibility_path=paths["m3_7f"],
        p1_policy_probe_path=paths["p1"],
        p1_utility_handoff_path=paths["p1_handoff"],
        budget=1,
    )

    inputs = payload["input_summaries"]
    assert inputs["hypothesis_context"]["m2_ready_for_m3"] == 3
    assert inputs["hypothesis_context"]["llm_output_counted_as_evidence"] is False
    assert inputs["hypothesis_context"]["world_model_counted_as_evidence"] is False
    assert inputs["m3_tests"]["m3_fused_requests_executed"] == 3
    assert inputs["m3_tests"]["new_independent_terminal_risk_evidence"] is False
    assert inputs["m3_tests"]["offline_counterfactual_feasible"] is False
    assert inputs["policy_context"]["policy_memory_enabled"] is True
    assert inputs["policy_context"]["a33_ready"] is False
    assert inputs["policy_context"]["policy_result_counted_as_confirmation"] is False
    assert payload["policy_result_counted_as_confirmation"] is False
    assert payload["support"] == 0
