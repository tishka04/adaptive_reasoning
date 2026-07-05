"""A2 live-transition belief loop tests."""

from __future__ import annotations

import numpy as np

from theory.live_transition_loop import (
    LiveTransitionBeliefLoop,
    build_transition_record,
)


def test_build_transition_record_from_real_grids_and_updates_theory():
    before = np.zeros((5, 5), dtype=np.int32)
    after = np.zeros((5, 5), dtype=np.int32)
    before[2, 2] = 2
    after[2, 3] = 2

    record = build_transition_record(
        action="ACTION1",
        grid_before=before,
        grid_after=after,
        available_actions=["ACTION1"],
        background_value=0,
        infer_players=True,
    )

    assert record.diff.num_changed == 2
    assert record.diff.player_displacement == (0, 1)
    assert len(record.obs_before.objects) == 1
    assert len(record.obs_after.objects) == 1

    loop = LiveTransitionBeliefLoop(
        "synthetic",
        available_actions=["ACTION1"],
        background_value=0,
        infer_players=True,
    )
    for idx in range(8):
        loop.observe_grids(
            action="ACTION1",
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION1"],
            timestamp=idx,
        )

    dominant = loop.theory.dominant("ACTION1")
    assert dominant is not None
    assert dominant.kind == "move"

    click_hyp = [h for h in loop.theory.for_action("ACTION1") if h.kind == "click"][0]
    assert click_hyp.contradictions >= 4


def test_live_loop_verifies_removal_rules_from_transition_records():
    before = np.zeros((5, 5), dtype=np.int32)
    before[1, 1] = 3
    before[3, 3] = 4
    after = before.copy()
    after[1, 1] = 0

    loop = LiveTransitionBeliefLoop(
        "synthetic",
        available_actions=["ACTION6"],
        background_value=0,
        infer_players=False,
    )
    for idx in range(2):
        update = loop.observe_grids(
            action="ACTION6",
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION6"],
            timestamp=idx,
        )

    by_id = {rule.rule_id: rule for rule in update.rules}
    assert "remove_v3_with_ACTION6" in by_id
    assert by_id["remove_v3_with_ACTION6"].confidence >= 0.9


def test_live_loop_verifies_global_transform_rules():
    before = np.zeros((5, 5), dtype=np.int32)
    before[1:4, 1:4] = 2
    after = before.copy()
    after[1:4, 1:4] = 3

    loop = LiveTransitionBeliefLoop(
        "synthetic",
        available_actions=["ACTION2"],
        background_value=0,
        infer_players=False,
    )
    for idx in range(3):
        update = loop.observe_grids(
            action="ACTION2",
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION2"],
            timestamp=idx,
        )

    by_id = {rule.rule_id: rule for rule in update.rules}
    assert "global_transform_with_ACTION2" in by_id
    assert by_id["global_transform_with_ACTION2"].confidence >= 0.9
