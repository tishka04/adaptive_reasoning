import json

from theory.p2 import long_budget_saturation_probe as probe


def _scope_payload():
    return {
        "scope_consolidations": [
            {
                "scope_consolidation_id": (
                    "m3_24::bp35-0a0ad940::patch_similarity_scope::ACTION6"
                ),
                "game_id": "bp35-0a0ad940",
                "candidate_rule_family": "local_patch_transformability",
                "target_action": "ACTION6",
                "scope_assessment": "SCOPE_EXPANDED_CANDIDATE_ONLY",
                "known_success_args": [{"x": 12, "y": 0}],
                "known_failed_args": [{"x": 30, "y": 0}],
                "outside_boundary_args": [{"x": 18, "y": 0}],
                "alternate_context_success_args": [{"x": 24, "y": 0}],
                "success_metrics": [
                    "local_patch_before_after",
                    "object_positions_before_after",
                ],
                "diagnostic_metrics": [
                    "changed_pixels",
                    "contact_graph_before_after",
                ],
                "recommended_agent_policy_probe": {
                    "use_repositioning_action": "ACTION4",
                },
                "ready_for_agent_policy_probe": True,
                "a33_ready": False,
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M3",
                "wrong_confirmations": 0,
            }
        ]
    }


def _summary(
    *,
    steps=48,
    triggers=0,
    action6_after=0,
    useful_after=0,
    new_after=0,
):
    return {
        "condition": "conditional_movement_refresh",
        "policy_steps": steps,
        "conditional_movement_refresh_triggers": triggers,
        "action6_after_conditional_movement_refresh_steps": action6_after,
        "useful_action6_after_conditional_movement_refresh_steps": useful_after,
        "new_action6_affordances_after_movement_refresh": new_after,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_budget_exhaustion_without_refresh_is_not_saturation():
    result = probe.classify_saturation_outcome(_summary(steps=256), budget=256)

    assert result["outcome"] == probe.NO_SATURATION
    assert result["budget_exhausted"] is True
    assert result["budget_exhausted_counted_as_saturation"] is False
    assert result["true_frontier_triggered"] is False
    assert result["support"] == 0


def test_refresh_unlock_is_not_frontier():
    result = probe.classify_saturation_outcome(
        _summary(triggers=1, action6_after=1, useful_after=1, new_after=1),
        budget=96,
    )

    assert result["outcome"] == probe.REFRESH_UNLOCKED
    assert result["movement_refresh_unlocked_action6"] is True
    assert result["true_frontier_triggered"] is False


def test_refresh_attempt_without_post_refresh_observation_is_not_frontier():
    result = probe.classify_saturation_outcome(
        _summary(triggers=1, action6_after=0, useful_after=0, new_after=0),
        budget=96,
    )

    assert result["outcome"] == probe.REFRESH_POST_UNOBSERVED
    assert result["post_refresh_action6_observed"] is False
    assert result["true_frontier_triggered"] is False


def test_failed_refresh_with_observed_action6_is_true_frontier_candidate():
    result = probe.classify_saturation_outcome(
        _summary(triggers=1, action6_after=1, useful_after=0, new_after=0),
        budget=96,
    )

    assert result["outcome"] == probe.TRUE_FRONTIER
    assert result["movement_refresh_attempted"] is True
    assert result["post_refresh_action6_observed"] is True
    assert result["true_frontier_triggered"] is True
    assert result["ready_for_p2_4_handoff"] is True
    assert result["support"] == 0


def test_aggregate_keeps_candidate_only_and_write_guards():
    rows = [
        {"budget": 48, "tie_break_seed": 0, "classification": probe.classify_saturation_outcome(_summary(steps=48), budget=48)},
        {"budget": 96, "tie_break_seed": 1, "classification": probe.classify_saturation_outcome(_summary(triggers=1, action6_after=1), budget=96)},
    ]

    aggregate = probe.aggregate_saturation_probe_results(rows)

    assert aggregate["budget_runs"] == 2
    assert aggregate["budget_exhausted_runs"] == 1
    assert aggregate["budget_exhausted_counted_as_saturation"] is False
    assert aggregate["true_frontier_runs"] == 1
    assert aggregate["real_frontier_ready_for_p2_4"] is True
    assert aggregate["a40_write_performed"] is False
    assert aggregate["m2_write_performed"] is False
    assert aggregate["m3_write_performed"] is False
    assert aggregate["support"] == 0


def test_run_long_budget_probe_uses_movement_refresh_without_writes(
    monkeypatch,
    tmp_path,
):
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(_scope_payload()), encoding="utf-8")
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)
    monkeypatch.setattr(probe, "_env_dir", lambda: tmp_path)
    monkeypatch.setattr(probe, "execute_probe_condition", lambda **kwargs: ())
    monkeypatch.setattr(
        probe,
        "summarize_probe_steps",
        lambda condition, steps: _summary(steps=48),
    )

    payload = probe.run_long_budget_saturation_probe(
        scope_consolidation_path=scope_path,
        budgets=(48,),
        tie_break_seeds=(0, 1),
    )

    assert payload["summary"]["budget_runs"] == 2
    assert payload["summary"]["no_saturation_runs"] == 2
    assert payload["summary"]["real_frontier_ready_for_p2_4"] is False
    assert payload["summary"]["a40_write_performed"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["support"] == 0
    assert payload["long_budget_probe_counted_as_confirmation"] is False
