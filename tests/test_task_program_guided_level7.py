from task_program_guided_level7 import (
    GlobalCorrespondenceScore,
    GuidedNode,
    MatchScore,
    RuleModel,
    _auto_levelup_feature_gap_report,
    _biggest_second_breaker_candidate_report,
    _build_auto_levelup_state_classifier_from_references,
    _filter_auto_levelup_references,
    _fragmentation_guidance_report,
    global_correspondence_score,
    infer_pair_colors_from_transitions,
    match_score,
    _load_action_ontology_model,
    _make_guided_child,
    _offending_component_diagnostics,
    _score_action_affordance,
    _score_auto_levelup_state,
    _score_break_context_similarity,
    _rewrite_logs,
    _rewrite_is_saturated,
    _submit_gate_allows,
)


class _RawFrame:
    def __init__(self, grid, *, state="NOT_FINISHED", levels_completed=7):
        self.frame = [grid]
        self.state = state
        self.levels_completed = levels_completed
        self.available_actions = [1, 2, 3, 4, 5, 7]


def test_infer_pair_colors_prefers_active_non_background_colors():
    colors = infer_pair_colors_from_transitions(
        {
            "9->10": 100,
            "10->9": 80,
            "9->11": 90,
            "11->9": 70,
            "4->9": 30,
        }
    )

    assert colors == (10, 11)


def test_match_score_rewards_shape_correspondence_and_reports_unmatched():
    aligned = [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11],
        [0, 10, 10, 0, 11, 11],
        [0, 0, 0, 0, 0, 0],
    ]
    unmatched = [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 0, 0],
        [0, 10, 10, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    good = match_score(aligned, pair_colors=(10, 11))
    bad = match_score(unmatched, pair_colors=(10, 11))

    assert good.matched_pairs == 1
    assert bad.unmatched_first == 1
    assert good.score > bad.score


def test_action2_submit_not_ready_is_penalized():
    parent_grid = [
        [0, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    raw = _RawFrame(parent_grid)
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=5.0,
        ready_scores=[5.0, 6.0],
        ready_level_up_actions={"ACTION2": 2},
        expected_matched_pairs=1,
        expected_unmatched_total=0,
    )
    parent_match = match_score(parent_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="root",
        search_score=0.0,
        match=parent_match,
        depth=0,
        path_hashes=["root"],
    )

    child = _make_guided_child(
        parent,
        raw=raw,
        env=None,
        action="ACTION2",
        action_data=None,
        rule_model=rule,
        root_match=parent_match,
        parent_grid=parent_grid,
        danger_actions=["ACTION2", "ACTION2"],
    )

    assert child.submit_not_ready is True
    assert child.search_score < 0


def test_submit_gate_requires_extra_match_no_dotted_and_unmatched_reduction():
    root = match_score(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        pair_colors=(10, 11),
        pair_threshold=3.0,
    )
    ready = match_score(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 10, 10, 0, 11, 11, 0],
            [0, 10, 10, 0, 11, 11, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        pair_colors=(10, 11),
        pair_threshold=3.0,
    )
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={"ACTION2": 1},
        expected_matched_pairs=0,
        expected_unmatched_total=2,
    )
    node = GuidedNode(
        actions=["ACTION1"],
        action_data=[None],
        state="NOT_FINISHED",
        level=7,
        grid_hash="ready",
        search_score=0.0,
        match=ready,
        depth=1,
    )

    allowed, _info = _submit_gate_allows(node, rule_model=rule, root_match=root)
    assert allowed is True


def test_rewrite_logs_track_productive_action3_and_first_noop():
    root_grid = [
        [0, 10, 0],
        [0, 0, 11],
        [0, 0, 0],
    ]
    changed_grid = [
        [0, 0, 10],
        [0, 0, 11],
        [0, 0, 0],
    ]
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    root_match = match_score(root_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="root",
        search_score=0.0,
        match=root_match,
        depth=0,
        path_hashes=["root"],
    )

    child = _make_guided_child(
        parent,
        raw=_RawFrame(changed_grid),
        env=None,
        action="ACTION3",
        action_data=None,
        rule_model=rule,
        root_match=root_match,
        parent_grid=root_grid,
        danger_actions=[],
        scoring_mode="rewrite_until_saturation",
    )
    noop = _make_guided_child(
        child,
        raw=_RawFrame(changed_grid),
        env=None,
        action="ACTION3",
        action_data=None,
        rule_model=rule,
        root_match=root_match,
        parent_grid=changed_grid,
        danger_actions=[],
        scoring_mode="rewrite_until_saturation",
    )
    logs = _rewrite_logs(noop)

    assert logs["action3_count"] == 2
    assert logs["action3_productive_count"] == 1
    assert logs["changed_cells_sequence"] == [2, 0]
    assert logs["cumulative_changed_cells"] == 2
    assert logs["first_noop_index"] == 2
    assert logs["first_noop_action3_ordinal"] == 2


def test_rewrite_saturation_threshold_marks_post_probe_child():
    grid = [
        [0, 10, 0],
        [0, 0, 11],
        [0, 0, 0],
    ]
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    match = match_score(grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=["ACTION3"] * 48,
        action_data=[None] * 48,
        state="NOT_FINISHED",
        level=7,
        grid_hash="sat",
        search_score=0.0,
        match=match,
        depth=48,
        path_hashes=["sat"],
    )

    assert _rewrite_is_saturated(parent, action3_threshold=48) is True

    child = _make_guided_child(
        parent,
        raw=_RawFrame(grid),
        env=None,
        action="ACTION2",
        action_data=None,
        rule_model=rule,
        root_match=match,
        parent_grid=grid,
        danger_actions=[],
        scoring_mode="rewrite_until_saturation",
        post_saturation_probe=True,
        parent_rewrite_saturated=True,
        rewrite_saturation_action3_threshold=48,
    )

    assert child.post_saturation_probe_phase is True


def test_semantic_scoring_prefers_more_matched_pairs():
    root_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11, 0],
        [0, 10, 10, 0, 11, 11, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    two_pair_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11, 0],
        [0, 10, 10, 0, 11, 11, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 11, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    root_match = match_score(root_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="root",
        search_score=0.0,
        match=root_match,
        depth=0,
        path_hashes=["root"],
    )

    child = _make_guided_child(
        parent,
        raw=_RawFrame(two_pair_grid),
        env=None,
        action="ACTION1",
        action_data=None,
        rule_model=rule,
        root_match=root_match,
        parent_grid=root_grid,
        danger_actions=[],
        scoring_mode="semantic",
    )

    assert child.match.matched_pairs >= parent.match.matched_pairs
    assert child.search_score > 0


def test_global_correspondence_scoring_prefers_family_structure():
    mismatched_grid = [
        [0, 10, 10, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11, 11, 0],
        [0, 0, 0, 0, 11, 11, 11, 0],
        [0, 10, 10, 0, 11, 11, 11, 0],
        [0, 10, 10, 0, 0, 0, 0, 0],
    ]
    coherent_grid = [
        [0, 10, 10, 0, 11, 11, 0, 0],
        [0, 10, 10, 0, 11, 11, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11, 0, 0],
        [0, 10, 10, 0, 11, 11, 0, 0],
    ]
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    root_match = match_score(mismatched_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="root",
        search_score=0.0,
        match=root_match,
        depth=0,
        global_correspondence=global_correspondence_score(
            mismatched_grid,
            pair_colors=rule.pair_colors,
        ),
        path_hashes=["root"],
    )

    child = _make_guided_child(
        parent,
        raw=_RawFrame(coherent_grid),
        env=None,
        action="ACTION1",
        action_data=None,
        rule_model=rule,
        root_match=root_match,
        parent_grid=mismatched_grid,
        danger_actions=[],
        scoring_mode="global_correspondence",
    )

    assert child.global_correspondence is not None
    assert child.global_correspondence.score > parent.global_correspondence.score
    assert child.global_delta_from_parent > 0
    assert child.search_score > 0

    hybrid_child = _make_guided_child(
        parent,
        raw=_RawFrame(coherent_grid),
        env=None,
        action="ACTION1",
        action_data=None,
        rule_model=rule,
        root_match=root_match,
        parent_grid=mismatched_grid,
        danger_actions=[],
        scoring_mode="global_semantic_hybrid",
    )

    assert hybrid_child.global_delta_from_parent > 0
    assert hybrid_child.readiness_gate_passed is True
    assert hybrid_child.search_score > 0


def test_offending_component_diagnostics_reports_concrete_targets():
    grid = [
        [10, 10, 0, 0, 0, 11],
        [10, 10, 0, 0, 0, 11],
        [0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 11, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    report = _offending_component_diagnostics(
        grid,
        pair_colors=(10, 11),
        limit=8,
    )

    assert report["offending_count"] > 0
    assert report["largest_offender"] is not None
    assert report["target_components"]
    assert all("bbox" in item for item in report["target_components"])
    assert any(
        "dotted_boundary_violation" in item["reason"]
        or "low_alignment_pair" in item["reason"]
        for item in report["target_components"]
    )


def test_action_ontology_model_learns_empirical_auto_levelup_prior(tmp_path):
    ontology_path = tmp_path / "ontology.json"
    ontology_path.write_text(
        """
{
  "probe_count": 3,
  "action_ontology": {
    "ACTION1": {
      "count": 2,
      "operator_type_counts": {"auto_levelup_trigger": 1, "transform_like": 1},
      "dominant_operator_type": "auto_levelup_trigger",
      "context_dependent": true,
      "no_op_rate": 0.0,
      "avg_changed_cells": 550.0,
      "direction_candidates": {"up": 1, "none": 1}
    },
    "ACTION6": {
      "count": 1,
      "operator_type_counts": {"no_op": 1},
      "dominant_operator_type": "no_op",
      "context_dependent": false,
      "no_op_rate": 1.0,
      "avg_changed_cells": 0.0,
      "direction_candidates": {"none": 1}
    }
  },
  "probes": [
    {
      "action": "ACTION1",
      "level_delta": 1,
      "changed_cells": 1000,
      "affected_colors": [10, 11],
      "sample": {"source": "human_pre_auto_levelup"}
    },
    {
      "action": "ACTION1",
      "level_delta": 0,
      "changed_cells": 100,
      "affected_colors": [10],
      "sample": {"source": "agent_guided_search"}
    },
    {
      "action": "ACTION6",
      "level_delta": 0,
      "changed_cells": 0,
      "affected_colors": [],
      "sample": {"source": "agent_guided_search"}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    model = _load_action_ontology_model(ontology_path, pair_colors=(10, 11))

    assert model["available"] is True
    assert model["human_auto_actions"] == {"ACTION1": 1}
    assert model["action_priors"]["ACTION1"]["human_auto_levelup_rate"] == 1.0
    assert model["action_priors"]["ACTION6"]["no_op_rate"] == 1.0


def test_action_ontology_guided_scoring_uses_affordance_without_direction_hardcode():
    root_grid = [
        [0, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 11, 0],
        [0, 0, 0, 0],
    ]
    child_grid = [
        [0, 0, 0, 0],
        [0, 10, 11, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    root_match = match_score(root_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="root",
        search_score=0.0,
        match=root_match,
        depth=0,
        global_correspondence=global_correspondence_score(
            root_grid,
            pair_colors=rule.pair_colors,
        ),
        path_hashes=["root"],
    )
    model = {
        "available": True,
        "human_auto_changed_cells_avg": 1000.0,
        "action_priors": {
            "ACTION2": {
                "count": 4,
                "dominant_operator_type": "transform_like",
                "context_dependent": True,
                "operator_type_rates": {"auto_levelup_trigger": 0.5, "transform_like": 0.5},
                "direction_candidates": {"down": 2, "up": 1, "none": 1},
                "no_op_rate": 0.0,
                "auto_levelup_rate": 0.5,
                "human_auto_levelup_rate": 0.5,
                "pair_affect_rate": 1.0,
            }
        },
    }

    child = _make_guided_child(
        parent,
        raw=_RawFrame(child_grid),
        env=None,
        action="ACTION2",
        action_data=None,
        rule_model=rule,
        root_match=root_match,
        parent_grid=root_grid,
        danger_actions=[],
        scoring_mode="action_ontology_guided",
        action_ontology_model=model,
    )

    assert child.submit_not_ready is False
    assert child.action_affordance["available"] is True
    assert child.action_affordance["human_auto_levelup_rate"] == 0.5
    assert child.action_affordance["pair_colors_affected"] is True
    assert child.search_score > 0


def test_auto_levelup_state_classifier_scores_reference_similarity():
    reference_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11],
        [0, 10, 10, 0, 11, 11],
        [0, 0, 0, 0, 0, 0],
    ]
    distant_grid = [
        [10, 10, 10, 10, 0, 0],
        [10, 10, 10, 10, 0, 0],
        [10, 10, 10, 10, 0, 0],
        [10, 10, 10, 10, 0, 0],
    ]
    classifier = _build_auto_levelup_state_classifier_from_references(
        [
            {
                "label": "human_pre_auto_levelup_1",
                "level": 1,
                "action": "ACTION2",
                "grid": reference_grid,
            }
        ],
        pair_colors=(10, 11),
    )

    exact = _score_auto_levelup_state(
        reference_grid,
        pair_colors=(10, 11),
        classifier=classifier,
    )
    distant = _score_auto_levelup_state(
        distant_grid,
        pair_colors=(10, 11),
        classifier=classifier,
    )

    assert classifier["available"] is True
    assert exact["score"] == 100.0
    assert exact["nearest_references"][0]["action"] == "ACTION2"
    assert exact["score"] > distant["score"]


def test_auto_reference_scope_filters_by_target_level():
    references = [
        {"label": "l2", "level": 2, "action": "ACTION2", "grid": [[0]]},
        {"label": "l7", "level": 7, "action": "ACTION3", "grid": [[1]]},
        {"label": "l8", "level": 8, "action": "ACTION1", "grid": [[2]]},
    ]

    same = _filter_auto_levelup_references(
        references,
        target_level=7,
        reference_scope="same-level",
    )
    nearby = _filter_auto_levelup_references(
        references,
        target_level=7,
        reference_scope="nearby-level",
    )
    all_refs = _filter_auto_levelup_references(
        references,
        target_level=7,
        reference_scope="all",
    )

    assert [item["label"] for item in same] == ["l7"]
    assert [item["label"] for item in nearby] == ["l7", "l8"]
    assert [item["label"] for item in all_refs] == ["l2", "l7", "l8"]


def test_auto_levelup_state_classifier_mode_uses_state_similarity_and_keeps_actions_unknown():
    root_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 0, 0],
        [0, 10, 10, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    reference_like_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11],
        [0, 10, 10, 0, 11, 11],
        [0, 0, 0, 0, 0, 0],
    ]
    classifier = _build_auto_levelup_state_classifier_from_references(
        [
            {
                "label": "human_pre_auto_levelup_1",
                "level": 1,
                "action": "ACTION2",
                "grid": reference_like_grid,
            }
        ],
        pair_colors=(10, 11),
    )
    action_model = {
        "available": True,
        "human_auto_changed_cells_avg": 1000.0,
        "action_priors": {
            "ACTION2": {
                "count": 1,
                "dominant_operator_type": "transform_like",
                "context_dependent": True,
                "operator_type_rates": {"auto_levelup_trigger": 1.0},
                "direction_candidates": {"none": 1},
                "no_op_rate": 0.0,
                "auto_levelup_rate": 1.0,
                "human_auto_levelup_rate": 1.0,
                "pair_affect_rate": 1.0,
            }
        },
    }
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    root_match = match_score(root_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="root",
        search_score=0.0,
        match=root_match,
        depth=0,
        global_correspondence=global_correspondence_score(
            root_grid,
            pair_colors=rule.pair_colors,
        ),
        auto_levelup_state=_score_auto_levelup_state(
            root_grid,
            pair_colors=rule.pair_colors,
            classifier=classifier,
        ),
        path_hashes=["root"],
    )

    child = _make_guided_child(
        parent,
        raw=_RawFrame(reference_like_grid),
        env=None,
        action="ACTION2",
        action_data=None,
        rule_model=rule,
        root_match=root_match,
        parent_grid=root_grid,
        danger_actions=[],
        scoring_mode="auto_levelup_state_classifier",
        action_ontology_model=action_model,
        auto_levelup_state_classifier=classifier,
    )

    assert child.submit_not_ready is False
    assert child.auto_levelup_state["available"] is True
    assert child.auto_levelup_state["nearest_references"][0]["label"] == "human_pre_auto_levelup_1"
    assert child.auto_levelup_state["score"] > parent.auto_levelup_state["score"]
    assert child.search_score > 0


def test_auto_levelup_feature_gap_reports_actionable_feature_deltas():
    reference_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 11, 11],
        [0, 10, 10, 0, 11, 11],
        [0, 0, 0, 0, 0, 0],
    ]
    agent_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 0, 0],
        [0, 10, 10, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    classifier = _build_auto_levelup_state_classifier_from_references(
        [
            {
                "label": "human_pre_auto_levelup_1",
                "level": 1,
                "action": "ACTION2",
                "grid": reference_grid,
            }
        ],
        pair_colors=(10, 11),
    )
    match = match_score(agent_grid, pair_colors=(10, 11))
    node = GuidedNode(
        actions=["ACTION5"],
        action_data=[None],
        state="NOT_FINISHED",
        level=7,
        grid_hash="agent",
        search_score=12.0,
        match=match,
        depth=1,
        global_correspondence=global_correspondence_score(
            agent_grid,
            pair_colors=(10, 11),
        ),
        auto_levelup_state=_score_auto_levelup_state(
            agent_grid,
            pair_colors=(10, 11),
            match=match,
            classifier=classifier,
        ),
    )

    gap = _auto_levelup_feature_gap_report(node, classifier=classifier)

    assert gap["available"] is True
    assert gap["nearest_reference"]["label"] == "human_pre_auto_levelup_1"
    assert gap["tracked_features"]["second_count"]["delta_agent_minus_reference"] < 0
    assert "second_size_sum" in gap["largest_gaps"]


def test_level7_fragmentation_guided_rewards_moving_toward_fragmented_reference():
    reference_grid = [
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
    ]
    parent_grid = [
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
    ]
    child_grid = [
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 11, 0, 0, 0, 0],
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
    ]
    classifier = _build_auto_levelup_state_classifier_from_references(
        [
            {
                "label": "human_pre_auto_levelup_7",
                "level": 7,
                "action": "ACTION1",
                "grid": reference_grid,
            }
        ],
        pair_colors=(10, 11),
        reference_scope="same-level",
        target_level=7,
    )
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    parent_match = match_score(parent_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="parent",
        search_score=0.0,
        match=parent_match,
        depth=0,
        global_correspondence=global_correspondence_score(
            parent_grid,
            pair_colors=rule.pair_colors,
        ),
        auto_levelup_state=_score_auto_levelup_state(
            parent_grid,
            pair_colors=rule.pair_colors,
            match=parent_match,
            classifier=classifier,
        ),
        path_hashes=["parent"],
    )

    child = _make_guided_child(
        parent,
        raw=_RawFrame(child_grid),
        env=None,
        action="ACTION3",
        action_data=None,
        rule_model=rule,
        root_match=parent_match,
        parent_grid=parent_grid,
        danger_actions=[],
        scoring_mode="level7_fragmentation_guided",
        auto_levelup_state_classifier=classifier,
    )

    assert child.submit_not_ready is False
    assert child.fragmentation_guidance["available"] is True
    assert child.fragmentation_guidance["improvement_score"] > 0
    assert child.search_score > 0


def test_level7_fragmentation_guarded_reports_structural_guardrails():
    reference_grid = [
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
    ]
    parent_grid = [
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
    ]
    child_grid = [
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 11, 0, 0, 0, 0],
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
    ]
    classifier = _build_auto_levelup_state_classifier_from_references(
        [
            {
                "label": "human_pre_auto_levelup_7",
                "level": 7,
                "action": "ACTION1",
                "grid": reference_grid,
            }
        ],
        pair_colors=(10, 11),
        reference_scope="same-level",
        target_level=7,
    )
    rule = RuleModel(
        pair_colors=(10, 11),
        ready_threshold=0.0,
        ready_scores=[],
        ready_level_up_actions={},
        expected_matched_pairs=0,
        expected_unmatched_total=0,
    )
    parent_match = match_score(parent_grid, pair_colors=rule.pair_colors)
    parent = GuidedNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=7,
        grid_hash="parent",
        search_score=0.0,
        match=parent_match,
        depth=0,
        global_correspondence=global_correspondence_score(
            parent_grid,
            pair_colors=rule.pair_colors,
        ),
        auto_levelup_state=_score_auto_levelup_state(
            parent_grid,
            pair_colors=rule.pair_colors,
            match=parent_match,
            classifier=classifier,
        ),
        path_hashes=["parent"],
    )

    child = _make_guided_child(
        parent,
        raw=_RawFrame(child_grid),
        env=None,
        action="ACTION3",
        action_data=None,
        rule_model=rule,
        root_match=parent_match,
        parent_grid=parent_grid,
        danger_actions=[],
        scoring_mode="level7_fragmentation_guarded",
        auto_levelup_state_classifier=classifier,
    )

    assert child.fragmentation_guardrails["available"] is True
    assert "penalty" in child.fragmentation_guardrails
    assert child.fragmentation_guidance["improvement_score"] > 0


def test_largest_second_pressure_overweights_largest_block_reduction():
    reference_grid = [
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
    ]
    parent_grid = [
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
        [0, 10, 0, 11, 11, 11, 11, 0],
    ]
    child_grid = [
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 11, 0, 0, 0, 0],
        [0, 10, 0, 11, 0, 11, 0, 11],
        [0, 10, 0, 0, 0, 0, 0, 0],
    ]
    classifier = _build_auto_levelup_state_classifier_from_references(
        [
            {
                "label": "human_pre_auto_levelup_7",
                "level": 7,
                "action": "ACTION1",
                "grid": reference_grid,
            }
        ],
        pair_colors=(10, 11),
        reference_scope="same-level",
        target_level=7,
    )
    parent_state = _score_auto_levelup_state(
        parent_grid,
        pair_colors=(10, 11),
        classifier=classifier,
    )
    child_state = _score_auto_levelup_state(
        child_grid,
        pair_colors=(10, 11),
        classifier=classifier,
    )

    normal = _fragmentation_guidance_report(
        parent_state=parent_state,
        child_state=child_state,
        classifier=classifier,
        largest_second_pressure=False,
    )
    pressured = _fragmentation_guidance_report(
        parent_state=parent_state,
        child_state=child_state,
        classifier=classifier,
        largest_second_pressure=True,
    )

    assert pressured["largest_second_pressure"] is True
    assert pressured["score"] > normal["score"]


def _breaker_test_node(
    *,
    actions,
    second_count,
    second_largest_size,
    largest_offender_size,
    global_score=50.0,
    dotted_violations=2,
    grid=None,
):
    return GuidedNode(
        actions=list(actions),
        action_data=[None for _ in actions],
        state="NOT_FINISHED",
        level=7,
        grid_hash="-".join(actions) or "root",
        search_score=0.0,
        match=MatchScore(
            score=0.0,
            pair_colors=(10, 11),
            matched_pairs=0,
            unmatched_first=0,
            unmatched_second=0,
            shape_overlap_or_alignment=0.0,
            cursor_near_target=0,
            dotted_constraint_violations=dotted_violations,
            component_counts={},
        ),
        depth=len(actions),
        global_correspondence=GlobalCorrespondenceScore(
            score=global_score,
            pair_colors=(10, 11),
            structure_similarity=0.0,
            count_similarity=0.0,
            size_distribution_similarity=0.0,
            orientation_similarity=0.0,
            spatial_order_similarity=0.0,
            boundary_relation_similarity=0.0,
            centroid_relation_similarity=0.0,
            component_counts={},
            family_signatures={},
        ),
        auto_levelup_state={
            "available": True,
            "score": 0.0,
            "features": {
                "second_count": second_count,
                "second_largest_size": second_largest_size,
                "largest_offender_size": largest_offender_size,
                "second_size_sum": second_largest_size,
                "unmatched_total": second_count,
            },
        },
        fragmentation_guardrails={"passes": True, "penalty": 0.0},
        grid=grid,
    )


def test_biggest_second_breaker_candidate_report_tracks_target_break():
    seed = _breaker_test_node(
        actions=["ACTION5"],
        second_count=16,
        second_largest_size=224,
        largest_offender_size=224,
    )
    child = _breaker_test_node(
        actions=["ACTION5", "ACTION3", "ACTION4"],
        second_count=24,
        second_largest_size=188,
        largest_offender_size=188,
    )

    report = _biggest_second_breaker_candidate_report(seed, child)

    assert report["probe_actions"] == ["ACTION3", "ACTION4"]
    assert report["delta_second_largest_size"] == 36
    assert report["delta_component_count_11"] == 8
    assert report["passes_constraints"] is True
    assert report["breaks_under_target"] is True


def test_break_context_similarity_prefers_human_like_pre_break_state():
    model = {
        "available": True,
        "references": [
            {
                "label": "human_break_512_ACTION1",
                "trace_index": 512,
                "trace_step": 512,
                "action": "ACTION1",
                "previous_actions": ["ACTION1", "ACTION1", "ACTION1", "ACTION1"],
                "deltas": {"second_largest_drop": 118.0, "second_count_gain": 5.0},
                "numeric": {
                    "second_largest_size": 291.0,
                    "second_count": 7.0,
                    "largest_offender_size": 291.0,
                    "unmatched_total": 24.0,
                    "dotted_violations": 3.0,
                    "nearest_distance_mean": 8.0,
                    "global_score": 26.0,
                },
                "largest_second": {
                    "centroid": {"y": 2.0, "x": 2.0},
                    "bbox": {"min_y": 0, "min_x": 0, "max_y": 3, "max_x": 3},
                },
                "largest_second_boundary_contact": {
                    "top": True,
                    "bottom": False,
                    "left": True,
                    "right": False,
                },
                "cursor_to_largest_distance": 2.0,
            }
        ],
    }
    close_grid = [
        [11, 11, 11, 11, 0, 0],
        [11, 11, 11, 11, 4, 0],
        [11, 11, 11, 11, 4, 0],
        [11, 11, 11, 11, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    far_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 11, 11],
        [0, 0, 0, 0, 11, 11],
        [0, 0, 0, 0, 4, 0],
    ]
    close = _breaker_test_node(
        actions=["ACTION1", "ACTION1", "ACTION1", "ACTION1"],
        second_count=7,
        second_largest_size=291,
        largest_offender_size=291,
        global_score=26,
        dotted_violations=3,
        grid=close_grid,
    )
    far = _breaker_test_node(
        actions=["ACTION5", "ACTION2"],
        second_count=38,
        second_largest_size=45,
        largest_offender_size=45,
        global_score=70,
        dotted_violations=0,
        grid=far_grid,
    )

    close_score = _score_break_context_similarity(
        close,
        model=model,
        pair_colors=(10, 11),
    )
    far_score = _score_break_context_similarity(
        far,
        model=model,
        pair_colors=(10, 11),
    )

    assert close_score["available"] is True
    assert close_score["score"] > far_score["score"]
