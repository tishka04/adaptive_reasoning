import json

import numpy as np

import theory.p1.bp35_sage_candidate_policy_probe as probe


class _Action:
    def __init__(self, name, action_args=None):
        self.name = name
        self.action_args = dict(action_args or {})


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
                "initial_context_supported": True,
                "alternate_context_supported": True,
                "outside_region_boundary_reinforced": True,
                "known_success_args": [
                    {"x": 12, "y": 0},
                    {"x": 24, "y": 0},
                    {"x": 30, "y": 12},
                ],
                "known_failed_args": [{"x": 30, "y": 0}],
                "outside_boundary_args": [{"x": 18, "y": 0}, {"x": 30, "y": 0}],
                "alternate_context_success_args": [
                    {"x": 12, "y": 0},
                    {"x": 24, "y": 0},
                ],
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
                    "select_target_action": "ACTION6",
                    "prefer_patch_similar_success_like_args": True,
                    "avoid_known_failure_like_args": [{"x": 30, "y": 0}],
                    "do_not_treat_candidate_as_confirmed_skill": True,
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


def test_candidate_policy_memory_loads_without_a33_readiness(tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    memory = probe.candidate_policy_memory_from_scope(payload)

    assert memory.enabled is True
    assert memory.ready_for_agent_policy_probe is True
    assert memory.a33_ready is False
    assert memory.support == 0
    assert memory.target_action == "ACTION6"
    assert memory.repositioning_action == "ACTION4"
    assert {"x": 30, "y": 0} in memory.known_failed_args


def test_score_action6_candidate_prefers_success_and_penalizes_failure():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 40), 5, dtype=np.int32)
    grid[0:2, 11:14] = [[5, 5, 5], [5, 5, 3]]
    grid[0:2, 17:20] = [[5, 5, 5], [3, 5, 10]]
    grid[0:2, 29:32] = [[5, 5, 10], [3, 5, 10]]

    success = probe.score_action6_candidate(
        {"x": 12, "y": 0},
        before_grid=grid,
        memory=memory,
        used_action6_args=(),
    )
    failure = probe.score_action6_candidate(
        {"x": 30, "y": 0},
        before_grid=grid,
        memory=memory,
        used_action6_args=(),
    )
    boundary = probe.score_action6_candidate(
        {"x": 18, "y": 0},
        before_grid=grid,
        memory=memory,
        used_action6_args=(),
    )

    assert success["score"] < boundary["score"]
    assert success["score"] < failure["score"]
    assert success["exact_known_success"] is True
    assert failure["exact_known_failure"] is True
    assert boundary["exact_outside_boundary"] is True


def test_candidate_policy_repositions_then_selects_success_like_action6():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 40), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 30, "y": 0}),
        _Action("ACTION6", {"x": 12, "y": 0}),
    ]

    first = probe.select_probe_decision(
        condition=probe.CANDIDATE_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=(),
        used_action6_args=(),
        action_counts={},
    )
    assert first.action_name == "ACTION4"
    assert first.candidate_policy_used is True

    second = probe.select_probe_decision(
        condition=probe.CANDIDATE_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION4",),
        used_action6_args=(),
        action_counts={"ACTION4": 1},
    )
    assert second.action_name == "ACTION6"
    assert second.action_args == {"x": 12, "y": 0}
    assert second.candidate_policy_used is True
    assert second.decision_reason == "candidate_policy_patch_similarity_retarget"


def test_action4_only_ablation_repositions_without_patch_scoring():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 40), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 30, "y": 0}),
        _Action("ACTION6", {"x": 12, "y": 0}),
    ]

    first = probe.select_probe_decision(
        condition=probe.ACTION4_ONLY_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=(),
        used_action6_args=(),
        action_counts={},
    )
    assert first.action_name == "ACTION4"
    assert first.decision_reason == "action4_only_reposition_without_patch_similarity"

    second = probe.select_probe_decision(
        condition=probe.ACTION4_ONLY_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION4",),
        used_action6_args=(),
        action_counts={"ACTION4": 1},
    )
    assert second.action_name == "ACTION6"
    assert second.candidate_score is None
    assert second.candidate_score_details["ablation"] == "no_patch_similarity_scoring"


def test_patch_similarity_only_ablation_scores_action6_without_repositioning():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 40), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 30, "y": 0}),
        _Action("ACTION6", {"x": 12, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.PATCH_SIMILARITY_ONLY_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=(),
        used_action6_args=(),
        action_counts={},
    )

    assert decision.action_name == "ACTION6"
    assert decision.action_args == {"x": 12, "y": 0}
    assert decision.decision_reason == "patch_similarity_only_action6_without_reposition"
    assert decision.candidate_score_details["repositioning_action_allowed"] is False


def test_patch_similarity_stale_guard_skips_used_action6_args():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 12, "y": 0}),
        _Action("ACTION6", {"x": 24, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.PATCH_SIMILARITY_STALE_GUARD_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION6",),
        used_action6_args=({"x": 12, "y": 0},),
        action_counts={"ACTION6": 1},
    )

    assert decision.action_name == "ACTION6"
    assert decision.action_args == {"x": 24, "y": 0}
    assert decision.decision_reason == "patch_similarity_stale_guard_fresh_action6"
    assert decision.candidate_score_details["repeated_args_filtered"] == 1
    assert decision.candidate_score_details["failure_like_args_filtered"] == 1


def test_patch_similarity_stale_guard_falls_back_when_no_fresh_safe_action6():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION3"),
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 12, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.PATCH_SIMILARITY_STALE_GUARD_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION6",),
        used_action6_args=({"x": 12, "y": 0},),
        action_counts={"ACTION6": 1},
    )

    assert decision.action_name == "ACTION3"
    assert decision.decision_reason == "patch_similarity_stale_guard_no_fresh_action6_fallback"
    assert decision.candidate_score_details["stale_guard_triggered"] is True
    assert decision.candidate_score_details["fresh_safe_action6_available"] is False


def _action6_step(action_args, *, useful=True, repeated=False):
    return probe.ProbeStep(
        step=1 if repeated else 0,
        condition=probe.PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        policy_selected_action="ACTION6",
        action_args=action_args,
        decision_reason="previous",
        candidate_policy_used=True,
        candidate_score=-20.0,
        candidate_score_details={},
        action6_arg_class="known_success",
        failure_like_action6_arg=False,
        success_like_action6_arg=True,
        repositioning_action=False,
        local_patch_signal=2.0 if useful else 0.0,
        object_positions_signal=4.0 if useful else 0.0,
        changed_pixels=8.0 if useful else 0.0,
        contact_graph_signal=0.0,
        useful_action6=useful,
        useful_repositioning=False,
        useful_new_state=useful,
        dead_end_or_cycle=not useful,
        state_signature_before="a",
        state_signature_after="b" if useful else "a",
        levels_before=0,
        levels_after=0,
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        measurements={},
    )


def test_soft_stale_guard_allows_repeated_action6_when_effectful():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION6", {"x": 12, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION6",),
        used_action6_args=({"x": 12, "y": 0},),
        previous_steps=(_action6_step({"x": 12, "y": 0}, useful=True),),
        action_counts={"ACTION6": 1},
    )

    assert decision.action_name == "ACTION6"
    assert decision.action_args == {"x": 12, "y": 0}
    assert decision.decision_reason == "patch_similarity_soft_stale_guard_effective_action6"
    assert decision.candidate_score_details["effect_memory"]["last_effect_useful"] is True
    assert decision.candidate_score_details["soft_stale_blocked"] is False


def test_soft_stale_guard_blocks_repeated_action6_after_no_effect():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION6", {"x": 12, "y": 0}),
        _Action("ACTION6", {"x": 24, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION6",),
        used_action6_args=({"x": 12, "y": 0},),
        previous_steps=(_action6_step({"x": 12, "y": 0}, useful=False),),
        action_counts={"ACTION6": 1},
    )

    assert decision.action_name == "ACTION6"
    assert decision.action_args == {"x": 24, "y": 0}
    assert decision.candidate_score_details["blocked_stale_args"] == 1


def test_conditional_refresh_prefers_effective_action6_before_action4():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 12, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.CONDITIONAL_ACTION4_REFRESH_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=(),
        used_action6_args=(),
        previous_steps=(),
        action_counts={},
    )

    assert decision.action_name == "ACTION6"
    assert decision.action_args == {"x": 12, "y": 0}
    assert decision.decision_reason == "conditional_refresh_effective_action6"
    assert decision.candidate_score_details["refresh_action_selected"] is False
    assert decision.candidate_score_details["support"] == 0


def test_conditional_refresh_uses_action4_when_no_effective_action6():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 18, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.CONDITIONAL_ACTION4_REFRESH_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION6",),
        used_action6_args=({"x": 18, "y": 0},),
        previous_steps=(),
        action_counts={"ACTION6": 1},
    )

    assert decision.action_name == "ACTION4"
    assert decision.decision_reason == "conditional_action4_refresh_after_soft_stale_exhausted"
    assert decision.candidate_score_details["conditional_refresh_triggered"] is True
    assert decision.candidate_score_details["effective_action6_available"] is False
    assert decision.candidate_score_details["support"] == 0


def test_conditional_movement_refresh_prefers_effective_action6_before_movement():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION3"),
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 12, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.CONDITIONAL_MOVEMENT_REFRESH_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=(),
        used_action6_args=(),
        previous_steps=(),
        action_counts={},
    )

    assert decision.action_name == "ACTION6"
    assert decision.action_args == {"x": 12, "y": 0}
    assert decision.decision_reason == "conditional_movement_refresh_effective_action6"
    assert decision.candidate_score_details["refresh_action_selected"] is False
    assert decision.candidate_score_details["support"] == 0


def test_conditional_movement_refresh_can_choose_action3_under_saturation():
    memory = probe.candidate_policy_memory_from_scope(_scope_payload())
    grid = np.full((16, 50), 5, dtype=np.int32)
    valid_actions = [
        _Action("ACTION3"),
        _Action("ACTION4"),
        _Action("ACTION6", {"x": 18, "y": 0}),
        _Action("ACTION6", {"x": 30, "y": 0}),
    ]

    decision = probe.select_probe_decision(
        condition=probe.CONDITIONAL_MOVEMENT_REFRESH_POLICY,
        memory=memory,
        before_grid=grid,
        valid_actions=valid_actions,
        action_history=("ACTION6",),
        used_action6_args=({"x": 18, "y": 0},),
        previous_steps=(),
        action_counts={"ACTION6": 1},
    )

    assert decision.action_name == "ACTION3"
    assert decision.decision_reason == "conditional_movement_refresh_after_soft_stale_exhausted"
    assert decision.candidate_score_details["conditional_movement_refresh_triggered"] is True
    assert decision.candidate_score_details["refresh_candidates_considered"] == [
        "ACTION3",
        "ACTION4",
        "ACTION1",
        "ACTION2",
    ]
    assert decision.candidate_score_details["support"] == 0


def test_run_probe_summarizes_comparison_without_verdict(monkeypatch, tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")
    baseline_steps = (
        probe.ProbeStep(
            step=0,
            condition=probe.BASELINE_POLICY,
            policy_selected_action="ACTION6",
            action_args={"x": 30, "y": 0},
            decision_reason="baseline",
            candidate_policy_used=False,
            candidate_score=None,
            candidate_score_details={},
            action6_arg_class="known_failure",
            failure_like_action6_arg=True,
            success_like_action6_arg=False,
            repositioning_action=False,
            local_patch_signal=0.0,
            object_positions_signal=0.0,
            changed_pixels=1.0,
            contact_graph_signal=0.0,
            useful_action6=False,
            useful_repositioning=False,
            useful_new_state=False,
            dead_end_or_cycle=True,
            state_signature_before="a",
            state_signature_after="a",
            levels_before=0,
            levels_after=0,
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            measurements={},
        ),
    )
    candidate_steps = (
        probe.ProbeStep(
            step=0,
            condition=probe.CANDIDATE_POLICY,
            policy_selected_action="ACTION4",
            action_args={},
            decision_reason="candidate_policy_reposition_before_retarget",
            candidate_policy_used=True,
            candidate_score=None,
            candidate_score_details={},
            action6_arg_class="",
            failure_like_action6_arg=False,
            success_like_action6_arg=False,
            repositioning_action=True,
            local_patch_signal=0.0,
            object_positions_signal=4.0,
            changed_pixels=47.0,
            contact_graph_signal=0.0,
            useful_action6=False,
            useful_repositioning=True,
            useful_new_state=True,
            dead_end_or_cycle=False,
            state_signature_before="a",
            state_signature_after="b",
            levels_before=0,
            levels_after=0,
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            measurements={},
        ),
        probe.ProbeStep(
            step=1,
            condition=probe.CANDIDATE_POLICY,
            policy_selected_action="ACTION6",
            action_args={"x": 12, "y": 0},
            decision_reason="candidate_policy_patch_similarity_retarget",
            candidate_policy_used=True,
            candidate_score=-20.0,
            candidate_score_details={},
            action6_arg_class="alternate_context_success",
            failure_like_action6_arg=False,
            success_like_action6_arg=True,
            repositioning_action=False,
            local_patch_signal=2.0,
            object_positions_signal=49.0,
            changed_pixels=36.0,
            contact_graph_signal=0.0,
            useful_action6=True,
            useful_repositioning=False,
            useful_new_state=True,
            dead_end_or_cycle=False,
            state_signature_before="b",
            state_signature_after="c",
            levels_before=0,
            levels_after=0,
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            measurements={},
        ),
    )

    def fake_execute(*, condition, memory, environments_dir, budget, game_id):
        return baseline_steps if condition == probe.BASELINE_POLICY else candidate_steps

    monkeypatch.setattr(probe, "execute_probe_condition", fake_execute)
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)

    payload = probe.run_bp35_sage_candidate_policy_probe(
        scope_consolidation_path=path,
        environments_dir=tmp_path,
        budget=2,
    )

    assert payload["summary"]["candidate_policy_probe_ready"] is True
    assert payload["summary"]["candidate_policy_status"] == "EXPERIMENTAL_POLICY_CANDIDATE_ONLY"
    assert payload["summary"]["a33_ready"] is False
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["comparison"]["candidate_policy_better_than_baseline_on_any_axis"] is True
    assert "useful_action6_steps" in payload["comparison"]["candidate_policy_improved_axes"]
    assert payload["comparison"]["candidate_policy_counted_as_confirmation"] is False


def test_matrix_probe_includes_action4_ablation_without_verdict(monkeypatch, tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def make_step(condition, progress_kind):
        return probe.ProbeStep(
            step=0,
            condition=condition,
            policy_selected_action="ACTION6",
            action_args={"x": 12, "y": 0} if progress_kind == "candidate" else {"x": 30, "y": 0},
            decision_reason=progress_kind,
            candidate_policy_used=condition != probe.BASELINE_POLICY,
            candidate_score=-20.0 if progress_kind == "candidate" else None,
            candidate_score_details={},
            action6_arg_class="known_success" if progress_kind == "candidate" else "known_failure",
            failure_like_action6_arg=progress_kind != "candidate",
            success_like_action6_arg=progress_kind == "candidate",
            repositioning_action=False,
            local_patch_signal=2.0 if progress_kind == "candidate" else 0.0,
            object_positions_signal=4.0 if progress_kind == "candidate" else 0.0,
            changed_pixels=10.0,
            contact_graph_signal=0.0,
            useful_action6=progress_kind == "candidate",
            useful_repositioning=False,
            useful_new_state=progress_kind == "candidate",
            dead_end_or_cycle=progress_kind != "candidate",
            state_signature_before="a",
            state_signature_after="b" if progress_kind == "candidate" else "a",
            levels_before=0,
            levels_after=0,
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            measurements={},
        )

    def fake_execute(*, condition, memory, environments_dir, budget, game_id, tie_break_seed=0):
        if condition == probe.CANDIDATE_POLICY:
            return (make_step(condition, "candidate"),)
        if condition == probe.ACTION4_ONLY_POLICY:
            return (make_step(condition, "ablation"),)
        return (make_step(condition, "baseline"),)

    monkeypatch.setattr(probe, "execute_probe_condition", fake_execute)
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)

    payload = probe.run_bp35_sage_candidate_policy_probe_matrix(
        scope_consolidation_path=path,
        environments_dir=tmp_path,
        budgets=(4, 8),
        tie_break_seeds=(0, 1),
    )

    assert payload["summary"]["budget_runs"] == 4
    assert payload["summary"]["conditions_per_run"] == 3
    assert payload["summary"]["support"] == 0
    assert payload["aggregate"]["candidate_beats_baseline_runs"] == 4
    assert payload["aggregate"]["candidate_beats_action4_only_runs"] == 4
    assert payload["aggregate"]["patch_similarity_attribution_signal_candidate_only"] is True
    assert payload["aggregate"]["candidate_policy_counted_as_confirmation"] is False
    assert payload["aggregate"]["ablation_counted_as_confirmation"] is False


def test_matrix_probe_can_include_patch_similarity_only_ablation(monkeypatch, tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def make_step(condition, kind):
        return probe.ProbeStep(
            step=0,
            condition=condition,
            policy_selected_action="ACTION6",
            action_args={"x": 12, "y": 0} if kind == "candidate" else {"x": 30, "y": 0},
            decision_reason=kind,
            candidate_policy_used=condition != probe.BASELINE_POLICY,
            candidate_score=-20.0 if condition in {
                probe.CANDIDATE_POLICY,
                probe.PATCH_SIMILARITY_ONLY_POLICY,
            } else None,
            candidate_score_details={},
            action6_arg_class="known_success" if kind == "candidate" else "known_failure",
            failure_like_action6_arg=kind != "candidate",
            success_like_action6_arg=kind == "candidate",
            repositioning_action=False,
            local_patch_signal=2.0 if kind == "candidate" else 0.0,
            object_positions_signal=4.0 if kind == "candidate" else 0.0,
            changed_pixels=10.0,
            contact_graph_signal=0.0,
            useful_action6=kind == "candidate",
            useful_repositioning=False,
            useful_new_state=kind == "candidate",
            dead_end_or_cycle=kind != "candidate",
            state_signature_before="a",
            state_signature_after="b" if kind == "candidate" else "a",
            levels_before=0,
            levels_after=0,
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            measurements={},
        )

    def fake_execute(*, condition, memory, environments_dir, budget, game_id, tie_break_seed=0):
        if condition == probe.CANDIDATE_POLICY:
            return (make_step(condition, "candidate"),)
        return (make_step(condition, "ablation"),)

    monkeypatch.setattr(probe, "execute_probe_condition", fake_execute)
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)

    payload = probe.run_bp35_sage_candidate_policy_probe_matrix(
        scope_consolidation_path=path,
        environments_dir=tmp_path,
        budgets=(4,),
        tie_break_seeds=(0, 1),
        include_patch_similarity_only=True,
    )

    assert payload["summary"]["budget_runs"] == 2
    assert payload["summary"]["conditions_per_run"] == 4
    assert payload["aggregate"]["patch_similarity_only_runs"] == 2
    assert payload["aggregate"]["candidate_beats_patch_similarity_only_runs"] == 2
    assert payload["aggregate"]["repositioning_context_dependency_signal_candidate_only"] is True
    assert payload["aggregate"]["patch_similarity_only_counted_as_confirmation"] is False
    assert payload["aggregate"]["support"] == 0


def test_matrix_probe_can_include_stale_guard_ablation(monkeypatch, tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def make_steps(condition):
        if condition == probe.PATCH_SIMILARITY_ONLY_POLICY:
            args = [{"x": 12, "y": 0}, {"x": 12, "y": 0}]
        elif condition == probe.PATCH_SIMILARITY_STALE_GUARD_POLICY:
            args = [{"x": 12, "y": 0}, {"x": 24, "y": 0}]
        else:
            args = [{"x": 30, "y": 0}, {"x": 30, "y": 0}]
        steps = []
        for index, action_args in enumerate(args):
            useful = condition in {
                probe.PATCH_SIMILARITY_ONLY_POLICY,
                probe.PATCH_SIMILARITY_STALE_GUARD_POLICY,
            }
            steps.append(
                probe.ProbeStep(
                    step=index,
                    condition=condition,
                    policy_selected_action="ACTION6",
                    action_args=action_args,
                    decision_reason=condition,
                    candidate_policy_used=condition != probe.BASELINE_POLICY,
                    candidate_score=-20.0 if useful else None,
                    candidate_score_details={},
                    action6_arg_class="known_success" if useful else "known_failure",
                    failure_like_action6_arg=not useful,
                    success_like_action6_arg=useful,
                    repositioning_action=False,
                    local_patch_signal=2.0 if useful else 0.0,
                    object_positions_signal=4.0 if useful else 0.0,
                    changed_pixels=10.0,
                    contact_graph_signal=0.0,
                    useful_action6=useful,
                    useful_repositioning=False,
                    useful_new_state=useful,
                    dead_end_or_cycle=not useful,
                    state_signature_before=f"s{index}",
                    state_signature_after=f"s{index + 1}" if useful else f"s{index}",
                    levels_before=0,
                    levels_after=0,
                    game_state_before="NOT_FINISHED",
                    game_state_after="NOT_FINISHED",
                    measurements={},
                )
            )
        return tuple(steps)

    def fake_execute(*, condition, memory, environments_dir, budget, game_id, tie_break_seed=0):
        return make_steps(condition)

    monkeypatch.setattr(probe, "execute_probe_condition", fake_execute)
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)

    payload = probe.run_bp35_sage_candidate_policy_probe_matrix(
        scope_consolidation_path=path,
        environments_dir=tmp_path,
        budgets=(4,),
        tie_break_seeds=(0, 1),
        include_stale_guard=True,
    )

    assert payload["summary"]["conditions_per_run"] == 5
    assert payload["aggregate"]["stale_guard_runs"] == 2
    assert payload["aggregate"]["stale_guard_repetition_reduction_signal_candidate_only"] is True
    assert payload["aggregate"]["stale_guard_mean_repeated_action6_delta_vs_patch_similarity_only"] == -1.0
    assert payload["aggregate"]["stale_guard_counted_as_confirmation"] is False
    assert payload["aggregate"]["support"] == 0


def test_matrix_probe_can_include_soft_stale_guard_ablation(monkeypatch, tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def make_steps(condition):
        if condition in {
            probe.PATCH_SIMILARITY_ONLY_POLICY,
            probe.PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        }:
            args = [{"x": 12, "y": 0}, {"x": 12, "y": 0}]
            useful_flags = [True, True]
        elif condition == probe.PATCH_SIMILARITY_STALE_GUARD_POLICY:
            args = [{"x": 12, "y": 0}, {"x": 24, "y": 0}]
            useful_flags = [True, True]
        else:
            args = [{"x": 30, "y": 0}, {"x": 30, "y": 0}]
            useful_flags = [False, False]
        steps = []
        for index, (action_args, useful) in enumerate(zip(args, useful_flags)):
            steps.append(
                probe.ProbeStep(
                    step=index,
                    condition=condition,
                    policy_selected_action="ACTION6",
                    action_args=action_args,
                    decision_reason=condition,
                    candidate_policy_used=condition != probe.BASELINE_POLICY,
                    candidate_score=-20.0 if useful else None,
                    candidate_score_details={},
                    action6_arg_class="known_success" if useful else "known_failure",
                    failure_like_action6_arg=not useful,
                    success_like_action6_arg=useful,
                    repositioning_action=False,
                    local_patch_signal=2.0 if useful else 0.0,
                    object_positions_signal=4.0 if useful else 0.0,
                    changed_pixels=10.0,
                    contact_graph_signal=0.0,
                    useful_action6=useful,
                    useful_repositioning=False,
                    useful_new_state=useful,
                    dead_end_or_cycle=not useful,
                    state_signature_before=f"s{index}",
                    state_signature_after=f"s{index + 1}" if useful else f"s{index}",
                    levels_before=0,
                    levels_after=0,
                    game_state_before="NOT_FINISHED",
                    game_state_after="NOT_FINISHED",
                    measurements={},
                )
            )
        return tuple(steps)

    def fake_execute(*, condition, memory, environments_dir, budget, game_id, tie_break_seed=0):
        return make_steps(condition)

    monkeypatch.setattr(probe, "execute_probe_condition", fake_execute)
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)

    payload = probe.run_bp35_sage_candidate_policy_probe_matrix(
        scope_consolidation_path=path,
        environments_dir=tmp_path,
        budgets=(4,),
        tie_break_seeds=(0, 1),
        include_soft_stale_guard=True,
    )

    assert payload["summary"]["conditions_per_run"] == 6
    assert payload["aggregate"]["soft_stale_guard_runs"] == 2
    assert payload["aggregate"]["soft_stale_guard_preserves_patch_only_progress_signal_candidate_only"] is True
    assert payload["aggregate"]["soft_stale_guard_sterile_repetition_nonincrease_signal_candidate_only"] is True
    assert payload["aggregate"]["soft_stale_guard_counted_as_confirmation"] is False
    assert payload["aggregate"]["support"] == 0


def test_matrix_probe_can_include_conditional_refresh_ablation(monkeypatch, tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def make_step(
        *,
        condition,
        index,
        action,
        action_args=None,
        reason=None,
        useful=True,
    ):
        return probe.ProbeStep(
            step=index,
            condition=condition,
            policy_selected_action=action,
            action_args=dict(action_args or {}),
            decision_reason=reason or condition,
            candidate_policy_used=condition != probe.BASELINE_POLICY,
            candidate_score=-20.0 if useful and action == "ACTION6" else None,
            candidate_score_details={},
            action6_arg_class="known_success" if useful and action == "ACTION6" else "",
            failure_like_action6_arg=False,
            success_like_action6_arg=bool(useful and action == "ACTION6"),
            repositioning_action=action == "ACTION4",
            local_patch_signal=2.0 if useful and action == "ACTION6" else 0.0,
            object_positions_signal=4.0 if useful and action == "ACTION6" else 0.0,
            changed_pixels=10.0 if useful else 0.0,
            contact_graph_signal=0.0,
            useful_action6=bool(useful and action == "ACTION6"),
            useful_repositioning=action == "ACTION4",
            useful_new_state=useful,
            dead_end_or_cycle=not useful,
            state_signature_before=f"s{index}",
            state_signature_after=f"s{index + 1}" if useful else f"s{index}",
            levels_before=0,
            levels_after=0,
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            measurements={},
        )

    def make_steps(condition):
        if condition == probe.CONDITIONAL_ACTION4_REFRESH_POLICY:
            return (
                make_step(
                    condition=condition,
                    index=0,
                    action="ACTION4",
                    reason="conditional_action4_refresh_after_soft_stale_exhausted",
                    useful=True,
                ),
                make_step(
                    condition=condition,
                    index=1,
                    action="ACTION6",
                    action_args={"x": 42, "y": 12},
                    reason="conditional_refresh_effective_action6",
                    useful=True,
                ),
            )
        if condition in {
            probe.PATCH_SIMILARITY_ONLY_POLICY,
            probe.PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        }:
            return (
                make_step(
                    condition=condition,
                    index=0,
                    action="ACTION6",
                    action_args={"x": 12, "y": 0},
                    useful=True,
                ),
                make_step(
                    condition=condition,
                    index=1,
                    action="ACTION6",
                    action_args={"x": 12, "y": 0},
                    useful=True,
                ),
            )
        if condition == probe.PATCH_SIMILARITY_STALE_GUARD_POLICY:
            return (
                make_step(
                    condition=condition,
                    index=0,
                    action="ACTION6",
                    action_args={"x": 12, "y": 0},
                    useful=True,
                ),
            )
        return (
            make_step(
                condition=condition,
                index=0,
                action="ACTION6",
                action_args={"x": 30, "y": 0},
                useful=False,
            ),
        )

    def fake_execute(*, condition, memory, environments_dir, budget, game_id, tie_break_seed=0):
        return make_steps(condition)

    monkeypatch.setattr(probe, "execute_probe_condition", fake_execute)
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)

    payload = probe.run_bp35_sage_candidate_policy_probe_matrix(
        scope_consolidation_path=path,
        environments_dir=tmp_path,
        budgets=(4,),
        tie_break_seeds=(0, 1),
        include_conditional_refresh=True,
    )

    assert payload["summary"]["conditions_per_run"] == 7
    assert payload["summary"]["conditional_refresh_policy_status"] == "EXPERIMENTAL_POLICY_ABLATION_ONLY"
    assert payload["aggregate"]["conditional_refresh_runs"] == 2
    assert payload["aggregate"]["conditional_refresh_total_triggers"] == 2
    assert payload["aggregate"]["conditional_refresh_runs_with_triggers"] == 2
    assert payload["aggregate"]["conditional_refresh_mean_new_action6_affordances_after_refresh"] == 1.0
    assert payload["aggregate"]["conditional_refresh_counted_as_confirmation"] is False
    assert payload["aggregate"]["support"] == 0


def test_matrix_probe_can_include_movement_refresh_ablation(monkeypatch, tmp_path):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(_scope_payload()), encoding="utf-8")

    def make_step(
        *,
        condition,
        index,
        action,
        action_args=None,
        reason=None,
        useful=True,
    ):
        return probe.ProbeStep(
            step=index,
            condition=condition,
            policy_selected_action=action,
            action_args=dict(action_args or {}),
            decision_reason=reason or condition,
            candidate_policy_used=condition != probe.BASELINE_POLICY,
            candidate_score=-20.0 if useful and action == "ACTION6" else None,
            candidate_score_details={},
            action6_arg_class="known_success" if useful and action == "ACTION6" else "",
            failure_like_action6_arg=False,
            success_like_action6_arg=bool(useful and action == "ACTION6"),
            repositioning_action=action in {"ACTION3", "ACTION4"},
            local_patch_signal=2.0 if useful and action == "ACTION6" else 0.0,
            object_positions_signal=4.0 if useful and action == "ACTION6" else 0.0,
            changed_pixels=10.0 if useful else 0.0,
            contact_graph_signal=0.0,
            useful_action6=bool(useful and action == "ACTION6"),
            useful_repositioning=action in {"ACTION3", "ACTION4"},
            useful_new_state=useful,
            dead_end_or_cycle=not useful,
            state_signature_before=f"s{index}",
            state_signature_after=f"s{index + 1}" if useful else f"s{index}",
            levels_before=0,
            levels_after=0,
            game_state_before="NOT_FINISHED",
            game_state_after="NOT_FINISHED",
            measurements={},
        )

    def make_steps(condition):
        if condition == probe.CONDITIONAL_MOVEMENT_REFRESH_POLICY:
            return (
                make_step(
                    condition=condition,
                    index=0,
                    action="ACTION3",
                    reason="conditional_movement_refresh_after_soft_stale_exhausted",
                    useful=True,
                ),
                make_step(
                    condition=condition,
                    index=1,
                    action="ACTION6",
                    action_args={"x": 42, "y": 12},
                    reason="conditional_movement_refresh_effective_action6",
                    useful=True,
                ),
            )
        if condition == probe.CONDITIONAL_ACTION4_REFRESH_POLICY:
            return (
                make_step(
                    condition=condition,
                    index=0,
                    action="ACTION4",
                    reason="conditional_action4_refresh_after_soft_stale_exhausted",
                    useful=True,
                ),
                make_step(
                    condition=condition,
                    index=1,
                    action="ACTION6",
                    action_args={"x": 42, "y": 12},
                    reason="conditional_refresh_effective_action6",
                    useful=True,
                ),
            )
        if condition in {
            probe.PATCH_SIMILARITY_ONLY_POLICY,
            probe.PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        }:
            return (
                make_step(
                    condition=condition,
                    index=0,
                    action="ACTION6",
                    action_args={"x": 12, "y": 0},
                    useful=True,
                ),
                make_step(
                    condition=condition,
                    index=1,
                    action="ACTION6",
                    action_args={"x": 12, "y": 0},
                    useful=True,
                ),
            )
        if condition == probe.PATCH_SIMILARITY_STALE_GUARD_POLICY:
            return (
                make_step(
                    condition=condition,
                    index=0,
                    action="ACTION6",
                    action_args={"x": 12, "y": 0},
                    useful=True,
                ),
            )
        return (
            make_step(
                condition=condition,
                index=0,
                action="ACTION6",
                action_args={"x": 30, "y": 0},
                useful=False,
            ),
        )

    def fake_execute(*, condition, memory, environments_dir, budget, game_id, tie_break_seed=0):
        return make_steps(condition)

    monkeypatch.setattr(probe, "execute_probe_condition", fake_execute)
    monkeypatch.setattr(probe, "_configure_offline_env", lambda env_dir: None)

    payload = probe.run_bp35_sage_candidate_policy_probe_matrix(
        scope_consolidation_path=path,
        environments_dir=tmp_path,
        budgets=(4,),
        tie_break_seeds=(0, 1),
        include_movement_refresh=True,
    )

    assert payload["summary"]["conditions_per_run"] == 8
    assert payload["summary"]["movement_refresh_policy_status"] == "EXPERIMENTAL_POLICY_ABLATION_ONLY"
    assert payload["aggregate"]["movement_refresh_runs"] == 2
    assert payload["aggregate"]["movement_refresh_total_triggers"] == 2
    assert payload["aggregate"]["movement_refresh_runs_with_triggers"] == 2
    assert payload["aggregate"]["movement_refresh_mean_new_action6_affordances_after_refresh"] == 1.0
    assert payload["aggregate"]["movement_refresh_actions_selected_counts"] == {"ACTION3": 2}
    assert payload["aggregate"]["movement_refresh_counted_as_confirmation"] is False
    assert payload["aggregate"]["support"] == 0
