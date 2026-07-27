from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np

from theory.live_transition_loop import build_observation
from theory.sage11.splits import SOURCE_TRAIN
from theory.sage12.action_target_collection import (
    _replace_with_retry,
    allocate_adaptive_game_quotas,
    select_adaptive_topup_game,
)
from theory.sage12.action_target_data import (
    EFFECT_LABELS,
    ActionTargetTrace,
    build_action_target_trace,
    conservative_match_objects,
    feature_row,
    resolve_action_target,
    validate_model_projection,
)
from v3.schemas import ObjectInfo


def _grid(player=(2, 2), target=None, target_value=2):
    grid = np.zeros((7, 7), dtype=np.int32)
    if player is not None:
        grid[player] = 1
    if target is not None:
        grid[target] = target_value
    return grid


def _trace(
    *,
    game="bp35",
    action="ACTION4",
    action_data=None,
    before=None,
    after=None,
):
    before = _grid() if before is None else before
    after = _grid(player=(2, 3)) if after is None else after
    return build_action_target_trace(
        game_id=game,
        source_split="source_train",
        policy_seed=12,
        reset_index=0,
        step_index=0,
        collection_phase="base",
        available_action_names=("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"),
        selected_action_name=action,
        selected_action_data=action_data or {},
        frame_before=before,
        frame_after=after,
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        levels_completed_before=0,
        levels_completed_after=0,
    )


def _object(object_id, value, cells):
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    return ObjectInfo(
        object_id=object_id,
        value=value,
        cells=list(cells),
        bbox=(min(rows), min(cols), max(rows), max(cols)),
        center=(sum(rows) / len(rows), sum(cols) / len(cols)),
        area=len(cells),
        shape_signature=hash(tuple(sorted(cells))),
    )


def test_movement_anchor_is_relative_and_model_view_has_no_coordinates():
    observation = build_observation(
        _grid(),
        available_actions=("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
    )
    anchor = resolve_action_target(observation, "ACTION4", {})

    assert anchor.kind == "move_destination"
    assert anchor.requested_direction == "right"
    assert (anchor.row, anchor.col) == (2, 3)
    assert anchor.actor_relation == "adjacent"
    assert anchor.path_status == "open"
    assert "row" not in anchor.model_view("full")
    assert "col" not in anchor.model_view("full")


def test_click_anchor_binds_exact_occupied_object():
    observation = build_observation(
        _grid(target=(4, 5)),
        available_actions=("ACTION6",),
    )
    anchor = resolve_action_target(
        observation,
        "ACTION6",
        {"x": 5, "y": 4},
    )

    assert anchor.kind == "clicked_object"
    assert anchor.occupied
    assert anchor.target_object_id is not None
    assert anchor.action_family == "click"


def test_click_empty_anchor_is_distinct():
    observation = build_observation(
        _grid(),
        available_actions=("ACTION6",),
    )
    anchor = resolve_action_target(
        observation,
        "ACTION6",
        {"x": 5, "y": 4},
    )

    assert anchor.kind == "clicked_empty"
    assert not anchor.occupied
    assert anchor.target_object_id is None


def test_conservative_matcher_tracks_translation():
    before = [_object(0, 3, [(2, 2)])]
    after = [_object(0, 3, [(2, 3)])]

    result = conservative_match_objects(before, after)

    assert result.matched == {0: 0}
    assert result.created == ()
    assert result.removed == ()


def test_conservative_matcher_masks_near_tie():
    before = [_object(0, 3, [(2, 2)])]
    after = [
        _object(0, 3, [(2, 1)]),
        _object(1, 3, [(2, 3)]),
    ]

    result = conservative_match_objects(before, after)

    assert result.matched == {}
    assert result.ambiguous_before == (0,)
    assert set(result.ambiguous_after) == set()


def test_observed_click_removal_is_target_grounded():
    before = _grid(target=(4, 5))
    after = _grid()
    trace = _trace(
        action="ACTION6",
        action_data={"x": 5, "y": 4},
        before=before,
        after=after,
    )

    assert trace.effects.applicable["target_removed"]
    assert trace.effects.labels["target_removed"]
    assert not trace.effects.labels["target_moved"]


def test_observed_click_transformation_is_remove_plus_create():
    before = _grid(target=(4, 5), target_value=2)
    after = _grid(target=(4, 5), target_value=3)
    trace = _trace(
        action="ACTION6",
        action_data={"x": 5, "y": 4},
        before=before,
        after=after,
    )

    assert trace.effects.labels["target_removed"]
    assert trace.effects.labels["target_created"]


def test_model_projection_excludes_provenance_and_future_state():
    trace = _trace()

    validate_model_projection(trace, "full")
    rendered = json.dumps(trace.model_features("full"), sort_keys=True)

    assert trace.game_id not in rendered
    assert "frame_before" not in rendered
    assert "frame_after" not in rendered
    assert "policy_seed" not in rendered
    assert "row" not in rendered
    assert "col" not in rendered


def test_trace_round_trip_preserves_digest_and_labels():
    trace = _trace()
    restored = ActionTargetTrace.from_dict(trace.to_dict())

    assert restored.trace_digest == trace.trace_digest
    assert restored.effects.labels == trace.effects.labels
    assert restored.model_features("coarse") == trace.model_features("coarse")


def test_feature_ladder_removes_shape_then_direction():
    trace = _trace()
    full = feature_row(trace, "full")
    no_shape = feature_row(trace, "no_shape")
    coarse = feature_row(trace, "coarse")

    assert any(key.startswith("target_area_bucket") for key in full)
    assert not any(key.startswith("target_area_bucket") for key in no_shape)
    assert not any(key.startswith("actor_relative_direction") for key in coarse)


def test_adaptive_allocation_is_exact_deterministic_and_capped():
    template = _trace()
    records = {
        game: [
            replace(
                template,
                game_id=game,
                trace_digest=f"{index:064x}",
            )
            for index in range(8)
        ]
        for game in SOURCE_TRAIN
    }

    first = allocate_adaptive_game_quotas(
        records,
        extra_total=453,
        maximum_extra_per_game=64,
    )
    second = allocate_adaptive_game_quotas(
        records,
        extra_total=453,
        maximum_extra_per_game=64,
    )

    assert first == second
    assert sum(first.values()) == 453
    assert max(first.values()) <= 64
    assert "lp85" not in first


def test_adaptive_reallocation_choice_is_deterministic():
    template = _trace()
    records = {
        "bp35": [replace(template, game_id="bp35") for _ in range(8)],
        "cd82": [replace(template, game_id="cd82") for _ in range(8)],
    }

    first = select_adaptive_topup_game(records, ("bp35", "cd82"))
    second = select_adaptive_topup_game(records, ("bp35", "cd82"))

    assert first == second
    assert first in records


def test_frozen_label_set_is_component_wise():
    assert EFFECT_LABELS == (
        "actor_displaced",
        "target_created",
        "target_removed",
        "target_moved",
    )


def test_atomic_replace_retries_transient_windows_lock(tmp_path, monkeypatch):
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(left, right):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("transient lock")
        return real_replace(left, right)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr(
        "theory.sage12.action_target_collection.time.sleep",
        lambda _seconds: None,
    )

    _replace_with_retry(source, destination)

    assert calls["count"] == 3
    assert destination.read_text(encoding="utf-8") == "new"
