from __future__ import annotations

import pytest

from theory.sage_t.contracts import ActionCandidate
from theory.sage_t.terminal_calibration_v9 import (
    ObservedSafetyCalibrator,
    TerminalCalibrationPolicy,
)


def test_repeated_safe_evidence_caps_unsupported_high_terminal_risk() -> None:
    policy = TerminalCalibrationPolicy("test", minimum_safe_observations=3)
    calibrator = ObservedSafetyCalibrator(policy)
    action = ActionCandidate("ACTION6", {"x": 10, "y": 12})

    for _ in range(3):
        assert calibrator.calibrate(action, 1.0) == 1.0
        calibrator.observe_outcome(action, False)

    assert calibrator.calibrate(action, 1.0) == pytest.approx(0.2)


def test_observed_danger_is_never_calibrated_down() -> None:
    policy = TerminalCalibrationPolicy("test", minimum_safe_observations=1)
    calibrator = ObservedSafetyCalibrator(policy)
    danger = ActionCandidate("ACTION6", {"x": 48, "y": 15})
    other = ActionCandidate("ACTION6", {"x": 8, "y": 58})

    calibrator.observe_outcome(other, False)
    calibrator.observe_outcome(danger, True)

    assert calibrator.calibrate(danger, 1.0) == 1.0
    assert calibrator.calibrate(other, 1.0) < 0.8
    assert calibrator.snapshot()["danger_actions"] == 1


def test_regimes_are_separate_but_reset_does_not_clear_evidence() -> None:
    policy = TerminalCalibrationPolicy("test", minimum_safe_observations=1)
    calibrator = ObservedSafetyCalibrator(policy)
    action = ActionCandidate("ACTION6", {"x": 1, "y": 2})

    calibrator.observe_outcome(action, False, regime_index=0)

    assert calibrator.calibrate(action, 1.0, regime_index=0) < 0.8
    assert calibrator.calibrate(action, 1.0, regime_index=1) == 1.0
