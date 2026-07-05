import numpy as np

from human_trace.schema import EpisodeRecord, StepRecord
from level7_frontier_recovery import (
    BranchNode,
    LevelFrontier,
    _bbox_targeted_action6_candidates,
    _change_mechanism_report,
    _component_delta,
    _coordinate_candidates,
    _has_structural_action,
    _histogram_delta,
    _hash_grid,
    _make_child,
    _max_consecutive,
    _mechanism_score,
    find_level_frontier,
)
from trace_replay_verifier import SelectedEpisode


class _RawFrame:
    def __init__(self, grid, *, state="NOT_FINISHED", levels_completed=1):
        self.frame = [np.array(grid, dtype=np.int32)]
        self.state = state
        self.levels_completed = levels_completed
        self.available_actions = [1, 2, 3, 4]


def _step(idx, before, action, after, *, state="NOT_FINISHED", level=0):
    return StepRecord(
        game_id="ar25-e3c63847",
        episode_id="ep",
        step=idx,
        frame_before=before,
        available_actions=[1, 2, 3, 4],
        action=action,
        action_args=None,
        frame_after=after,
        game_state_after=state,
        levels_completed_after=level,
        intent="test_move",
    )


def _selection(steps):
    return SelectedEpisode(
        game_id="ar25-e3c63847",
        episode_id="ep",
        selection_reason="test",
        episode=EpisodeRecord(
            game_id="ar25-e3c63847",
            episode_id="ep",
            started_at="2026-06-08T00:00:00+00:00",
            ended_at="2026-06-08T00:00:01+00:00",
            n_steps=len(steps),
            final_state="GAME_OVER",
            levels_completed=1,
        ),
        steps=steps,
    )


def test_find_level_frontier_uses_last_safe_pre_death_state():
    steps = [
        _step(0, [[1]], "ACTION1", [[2]], level=0),
        _step(1, [[2]], "ACTION2", [[3]], level=1),
        _step(2, [[3]], "ACTION3", [[4]], level=1),
        _step(3, [[4]], "ACTION4", [[9]], state="GAME_OVER", level=1),
    ]

    frontier = find_level_frontier(
        _selection(steps),
        target_level=1,
        danger_window=2,
    )

    assert frontier.level_start_index == 2
    assert frontier.frontier_index == 3
    assert frontier.frontier_frame == [[4]]
    assert frontier.immediate_danger_action == "ACTION4"
    assert frontier.danger_actions == ["ACTION3", "ACTION4"]
    assert _hash_grid([[9]]) in frontier.danger_state_hashes
    assert _hash_grid([[4]]) in frontier.visited_safe_hashes
    assert _hash_grid([[9]]) not in frontier.visited_safe_hashes


def test_branch_scoring_prefers_novel_safe_state_over_known_death_suffix():
    frontier = LevelFrontier(
        target_level=1,
        level_start_index=0,
        frontier_index=1,
        terminal_index=1,
        level_start_frame=[[1]],
        frontier_frame=[[4]],
        terminal_frame=[[9]],
        immediate_danger_action="ACTION4",
        danger_actions=["ACTION4"],
        visited_safe_hashes={_hash_grid([[4]])},
        danger_state_hashes={_hash_grid([[9]])},
    )
    parent = BranchNode(
        actions=[],
        action_data=[],
        state="NOT_FINISHED",
        level=1,
        grid_hash=_hash_grid([[4]]),
        score=0.0,
        depth=0,
        available_actions=["ACTION1", "ACTION4"],
        path_hashes=[_hash_grid([[4]])],
    )

    safe = _make_child(
        parent,
        raw=_RawFrame([[5]], state="NOT_FINISHED", levels_completed=1),
        env=None,
        action="ACTION1",
        action_data=None,
        frontier=frontier,
        frontier_grid=[[4]],
    )
    dead = _make_child(
        parent,
        raw=_RawFrame([[9]], state="GAME_OVER", levels_completed=1),
        env=None,
        action="ACTION4",
        action_data=None,
        frontier=frontier,
        frontier_grid=[[4]],
    )

    assert safe.died is False
    assert safe.novel_safe_states == 1
    assert dead.died is True
    assert dead.danger_prefix_len == 1
    assert safe.score > dead.score


def test_coordinate_candidates_include_object_centers_and_grid_points():
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[2:4, 4:6] = 7

    candidates = _coordinate_candidates(
        grid,
        grid_size=2,
        max_candidates=12,
        seeds=[{"x": 5, "y": 6}],
    )

    assert {"x": 5, "y": 6} in candidates
    assert {"x": 32, "y": 32} in candidates
    assert any(38 <= item["x"] <= 42 and 20 <= item["y"] <= 24 for item in candidates)


def test_bbox_targeted_candidates_include_bbox_anchors_and_mirrors():
    before = np.zeros((8, 8), dtype=np.int32)
    after = before.copy()
    after[1:4, 2:5] = 9

    candidates = _bbox_targeted_action6_candidates(
        before,
        after,
        max_candidates=32,
    )

    assert {"x": 2, "y": 1} in candidates
    assert {"x": 4, "y": 3} in candidates
    assert {"x": 3, "y": 2} in candidates
    assert {"x": 5, "y": 1} in candidates
    assert {"x": 2, "y": 6} in candidates


def test_spatial_deltas_separate_geometry_from_histogram_change():
    before = [[1, 0], [0, 0]]
    moved = [[0, 0], [0, 1]]
    recolored = [[2, 0], [0, 0]]

    assert _histogram_delta(before, moved) == 0
    assert _component_delta(before, moved) > 0
    assert _histogram_delta(before, recolored) > 0


def test_action7_cycle_helpers_detect_structural_actions():
    assert _max_consecutive(["ACTION7", "ACTION7", "ACTION1", "ACTION7"], "ACTION7") == 2
    assert _max_consecutive(["ACTION7", "ACTION7", "ACTION7"], "ACTION7") == 3
    assert _has_structural_action(["ACTION7", "ACTION6"]) is False
    assert _has_structural_action(["ACTION7", "ACTION3"]) is True


def test_change_mechanism_report_exposes_bbox_values_and_symmetry():
    before = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    after = [
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    report = _change_mechanism_report(before, after)

    assert report["changed_values"]["before_values"] == [1]
    assert report["changed_values"]["after_values"] == [2]
    assert report["bounding_box_of_diff"]["changed_cells"] == 2
    assert report["distance_to_center"]["normalized"] >= 0
    assert report["symmetry_score"]["vertical"] == 1.0


def test_mechanism_score_prioritizes_level_up_and_penalizes_death():
    base = [[1, 0], [0, 0]]
    changed = [[0, 0], [0, 1]]

    level_up = _mechanism_score(
        base_grid=base,
        final_grid=changed,
        final_state="NOT_FINISHED",
        final_level=2,
        target_level=1,
        new_safe_state=True,
        no_op_loop=False,
        repeatable_effect=True,
    )
    death = _mechanism_score(
        base_grid=base,
        final_grid=changed,
        final_state="GAME_OVER",
        final_level=1,
        target_level=1,
        new_safe_state=False,
        no_op_loop=False,
        repeatable_effect=True,
    )

    assert level_up["level_up"] is True
    assert death["death"] is True
    assert level_up["score"] > death["score"]
