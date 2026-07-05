import json

import theory.p3.terminal_horizon_policy_probe as probe


def _refined_payload():
    return {
        "summary": {
            "prefixes_with_candidate_terminal_avoidance": [63],
            "refined_prefixes": [50, 54, 58, 60, 62, 63],
            "refined_conditions": [
                "continue_action6",
                "stop_policy",
                "switch_ACTION3",
                "switch_ACTION4",
            ],
            "stop_switch_effectiveness_status": "CANDIDATE_TERMINAL_AVOIDANCE_OBSERVED",
            "stop_switch_effectiveness_counted_as_verdict": False,
            "refined_window_result_counted_as_confirmation": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _scope_payload():
    return {
        "scope_consolidations": [
            {
                "scope_consolidation_id": "m3_24::bp35::scope",
                "scope_assessment": "SCOPE_EXPANDED_CANDIDATE_ONLY",
                "ready_for_agent_policy_probe": True,
                "a33_ready": False,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
                "game_id": "bp35-0a0ad940",
                "target_action": "ACTION6",
                "candidate_rule_family": "local_patch_transformability",
                "known_success_args": [{"x": 12, "y": 0}],
                "known_failed_args": [{"x": 30, "y": 0}],
                "outside_boundary_args": [],
                "alternate_context_success_args": [{"x": 24, "y": 0}],
                "success_metrics": [
                    "local_patch_before_after",
                    "object_positions_before_after",
                ],
                "diagnostic_metrics": ["changed_pixels", "contact_graph_before_after"],
                "recommended_agent_policy_probe": {
                    "use_repositioning_action": "ACTION4"
                },
            }
        ]
    }


def _summary(condition, *, budget, seed, k_objective, k_stop, game_state, levels=0):
    return {
        "condition": condition,
        "budget": budget,
        "tie_break_seed": seed,
        "terminal_budget_estimate": 64,
        "terminal_horizon_source": "empirical_fallback",
        "estimated_moves_remaining": 1,
        "moves_used": 63,
        "k_objective": k_objective,
        "k_stop": k_stop,
        "terminal_horizon_triggered": condition != probe.BASELINE_POLICY,
        "objective_mode_entered": condition == probe.OBJECTIVE_MODE_POLICY,
        "trigger_reason": "moves_remaining_below_objective_threshold"
        if condition == probe.OBJECTIVE_MODE_POLICY
        else "moves_remaining_below_candidate_stop_threshold",
        "final_game_state": game_state,
        "terminal_state_after_rollout": game_state == "GAME_OVER",
        "final_levels_completed": levels,
        "progress_proxy": 10.0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
    }


def test_terminal_horizon_policy_probe_uses_moves_remaining_k_matrix(tmp_path):
    refined = tmp_path / "refined.json"
    scope = tmp_path / "scope.json"
    refined.write_text(json.dumps(_refined_payload()), encoding="utf-8")
    scope.write_text(json.dumps(_scope_payload()), encoding="utf-8")
    calls = []

    def fake_executor(condition, budget, seed, terminal_budget, k_objective, k_stop, memory, env_dir, game_id):
        calls.append((condition, terminal_budget, k_objective, k_stop))
        if condition == probe.BASELINE_POLICY:
            return _summary(
                condition,
                budget=budget,
                seed=seed,
                k_objective=None,
                k_stop=None,
                game_state="GAME_OVER",
            )
        return _summary(
            condition,
            budget=budget,
            seed=seed,
            k_objective=k_objective,
            k_stop=k_stop,
            game_state="NOT_FINISHED",
        )

    payload = probe.run_terminal_horizon_policy_probe(
        refined_window_results_path=refined,
        scope_consolidation_path=scope,
        budgets=(64,),
        tie_break_seeds=(0,),
        k_values=(1, 4),
        condition_executor=fake_executor,
    )

    assert payload["terminal_horizon_estimator"]["terminal_budget_estimate"] == 64
    assert payload["terminal_horizon_estimator"][
        "action6_prefix_count_used_as_decision_variable"
    ] is False
    assert payload["summary"]["candidate_policy_runs"] == 4
    assert payload["summary"]["terminal_avoidance_signal_runs"] == 4
    assert payload["summary"]["objective_completion_signal_runs"] == 0
    assert payload["summary"]["terminal_avoidance_counted_as_completion"] is False
    assert (probe.STOP_AT_HORIZON_POLICY, 64, 4, 4) in calls
    assert (probe.OBJECTIVE_MODE_POLICY, 64, 4, 1) in calls


def test_terminal_horizon_policy_probe_records_objective_completion_separately(tmp_path):
    refined = tmp_path / "refined.json"
    scope = tmp_path / "scope.json"
    refined.write_text(json.dumps(_refined_payload()), encoding="utf-8")
    scope.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def fake_executor(condition, budget, seed, terminal_budget, k_objective, k_stop, memory, env_dir, game_id):
        if condition == probe.BASELINE_POLICY:
            return _summary(
                condition,
                budget=budget,
                seed=seed,
                k_objective=None,
                k_stop=None,
                game_state="GAME_OVER",
            )
        return _summary(
            condition,
            budget=budget,
            seed=seed,
            k_objective=k_objective,
            k_stop=k_stop,
            game_state="NOT_FINISHED",
            levels=1 if condition == probe.OBJECTIVE_MODE_POLICY else 0,
        )

    payload = probe.run_terminal_horizon_policy_probe(
        refined_window_results_path=refined,
        scope_consolidation_path=scope,
        budgets=(64,),
        tie_break_seeds=(0,),
        k_values=(8,),
        condition_executor=fake_executor,
    )

    assert payload["summary"]["objective_completion_signal_runs"] == 1
    assert payload["summary"]["terminal_avoidance_counted_as_completion"] is False
    objective_rows = [
        row
        for row in payload["condition_k_summaries"]
        if row["condition"] == probe.OBJECTIVE_MODE_POLICY
    ]
    assert objective_rows[0]["objective_completion_signal_runs"] == 1

