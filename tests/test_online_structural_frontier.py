"""SAGE.9i structural-frontier generation tests."""

import numpy as np

from theory.live_transition_loop import build_transition_record
from theory.online_structural_frontier import OnlineStructuralFrontierDetector


def test_structural_detector_labels_entity_and_relation_change():
    before = np.zeros((7, 7), dtype=np.int32)
    before[3, 1] = 2
    before[3, 5] = 3
    after = before.copy()
    after[3, 5] = 0
    after[4, 4] = 3
    record = build_transition_record(
        action="ACTION1",
        grid_before=before,
        grid_after=after,
    )
    detector = OnlineStructuralFrontierDetector()

    signal = detector.observe_transition(
        grid_before=before,
        grid_after=after,
        objects_before=record.obs_before.objects,
        objects_after=record.obs_after.objects,
        diff=record.diff,
    )

    assert signal is not None
    assert "entity_motion" in signal.families
    assert "relation_change" in signal.families
    assert signal.trigger_signature.startswith("structural-trigger::")
    assert detector.summary()["signals_generated"] == 1


def test_structural_detector_suppresses_noop_and_terminal_transition():
    grid = np.zeros((5, 5), dtype=np.int32)
    noop = build_transition_record(
        action="ACTION1",
        grid_before=grid,
        grid_after=grid.copy(),
    )
    changed = grid.copy()
    changed[2, 2] = 4
    terminal = build_transition_record(
        action="ACTION1",
        grid_before=grid,
        grid_after=changed,
        levels_completed_before=0,
        levels_completed_after=1,
    )
    detector = OnlineStructuralFrontierDetector()

    assert detector.observe_transition(
        grid_before=grid,
        grid_after=grid,
        objects_before=noop.obs_before.objects,
        objects_after=noop.obs_after.objects,
        diff=noop.diff,
    ) is None
    assert detector.observe_transition(
        grid_before=grid,
        grid_after=changed,
        objects_before=terminal.obs_before.objects,
        objects_after=terminal.obs_after.objects,
        diff=terminal.diff,
        terminal_success=True,
    ) is None
    summary = detector.summary()
    assert summary["noop_suppressions"] == 1
    assert summary["terminal_suppressions"] == 1
