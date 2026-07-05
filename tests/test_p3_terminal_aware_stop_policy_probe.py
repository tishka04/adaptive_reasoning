import json

import pytest

import theory.p3.terminal_aware_stop_policy_probe as probe


def _refined_payload(*, support=0, verdict=False, status=None):
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
            "stop_switch_effectiveness_status": status
            or "CANDIDATE_TERMINAL_AVOIDANCE_OBSERVED",
            "stop_switch_effectiveness_counted_as_verdict": verdict,
            "refined_window_result_counted_as_confirmation": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "support": support,
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


def _summary(condition, *, budget, seed, game_state, levels, progress=0.0):
    return {
        "condition": condition,
        "budget": budget,
        "tie_break_seed": seed,
        "final_game_state": game_state,
        "terminal_state_after_rollout": game_state == "GAME_OVER",
        "final_levels_completed": levels,
        "progress_proxy": progress,
        "action6_steps": 63,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
    }


def test_terminal_aware_probe_counts_avoidance_not_completion(tmp_path):
    refined = tmp_path / "refined.json"
    scope = tmp_path / "scope.json"
    refined.write_text(json.dumps(_refined_payload()), encoding="utf-8")
    scope.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def fake_pair_executor(budget, seed, threshold, memory, env_dir, game_id):
        assert threshold == 63
        baseline = _summary(
            "patch_similarity_soft_stale_action6_prefix",
            budget=budget,
            seed=seed,
            game_state="GAME_OVER",
            levels=0,
            progress=100.0,
        )
        candidate = _summary(
            "terminal_aware_stop_at_threshold",
            budget=budget,
            seed=seed,
            game_state="NOT_FINISHED",
            levels=0,
            progress=80.0,
        )
        candidate["terminal_aware_stop_triggered"] = True
        candidate["terminal_avoidance_counted_as_completion"] = False
        return baseline, candidate

    payload = probe.run_terminal_aware_stop_policy_probe(
        refined_window_results_path=refined,
        scope_consolidation_path=scope,
        budgets=(64, 96),
        tie_break_seeds=(0, 1),
        pair_executor=fake_pair_executor,
    )

    summary = payload["summary"]
    assert summary["runs_per_condition"] == 4
    assert summary["baseline_game_over_runs"] == 4
    assert summary["candidate_game_over_runs"] == 0
    assert summary["terminal_avoidance_signal_runs"] == 4
    assert summary["objective_completion_signal_runs"] == 0
    assert summary["terminal_avoidance_only_is_not_objective_completion"] is True
    assert summary["support"] == 0
    assert payload["terminal_avoidance_counted_as_completion"] is False


def test_terminal_aware_probe_records_objective_completion_separately(tmp_path):
    refined = tmp_path / "refined.json"
    scope = tmp_path / "scope.json"
    refined.write_text(json.dumps(_refined_payload()), encoding="utf-8")
    scope.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def fake_pair_executor(budget, seed, threshold, memory, env_dir, game_id):
        return (
            _summary(
                "patch_similarity_soft_stale_action6_prefix",
                budget=budget,
                seed=seed,
                game_state="GAME_OVER",
                levels=0,
            ),
            _summary(
                "terminal_aware_stop_at_threshold",
                budget=budget,
                seed=seed,
                game_state="NOT_FINISHED",
                levels=1,
            ),
        )

    payload = probe.run_terminal_aware_stop_policy_probe(
        refined_window_results_path=refined,
        scope_consolidation_path=scope,
        budgets=(64,),
        tie_break_seeds=(0,),
        pair_executor=fake_pair_executor,
    )

    assert payload["summary"]["objective_completion_signal_runs"] == 1
    assert payload["summary"]["candidate_improves_level_completion_candidate_only"] is True
    assert payload["summary"]["terminal_avoidance_counted_as_completion"] is False


def test_terminal_aware_probe_rejects_source_support_or_verdict(tmp_path):
    support_path = tmp_path / "support.json"
    verdict_path = tmp_path / "verdict.json"
    wrong_status_path = tmp_path / "wrong_status.json"
    support_path.write_text(json.dumps(_refined_payload(support=1)), encoding="utf-8")
    verdict_path.write_text(json.dumps(_refined_payload(verdict=True)), encoding="utf-8")
    wrong_status_path.write_text(
        json.dumps(_refined_payload(status="NO_TERMINAL_SEPARATION_IN_REFINED_WINDOW")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support must remain 0"):
        probe.validate_refined_window_terminal_signal(json.loads(support_path.read_text()))
    with pytest.raises(ValueError, match="cannot be a verdict"):
        probe.validate_refined_window_terminal_signal(json.loads(verdict_path.read_text()))
    with pytest.raises(ValueError, match="must expose candidate terminal avoidance"):
        probe.validate_refined_window_terminal_signal(
            json.loads(wrong_status_path.read_text())
        )

