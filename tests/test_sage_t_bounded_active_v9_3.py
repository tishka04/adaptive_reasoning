from __future__ import annotations

from theory.sage_t import bounded_active_v9_3 as t9_3
from theory.sage_t.contracts import ActionCandidate


def test_bounded_controller_is_fail_closed_and_budgeted() -> None:
    manifest = {
        "controller_caps": {
            "maximum_programs": 32,
            "maximum_sequences": 32,
            "maximum_particles_per_decision": 8,
            "ordinary_horizon": 3,
            "maximum_structural_macros": 8,
        },
        "selected_terminal_policy": "safe_after_3",
        "authority": {
            "maximum_interventions_per_reset": 5,
            "maximum_marginal_terminal_risk": 0.05,
            "strong_surprise_lockout_threshold": 8.0,
        },
    }
    controller = t9_3.build_controller(manifest)

    assert controller.effective_mode.value == "bounded"
    assert controller.config.bounded_maximum_interventions_per_reset == 5
    assert controller.config.bounded_maximum_terminal_risk == 0.05
    assert controller._surprise_lockout is False


def test_observed_danger_latch_survives_branch_reset() -> None:
    manifest = t9_3.load_manifest()
    controller = t9_3.build_controller(manifest)
    action = ActionCandidate("ACTION6", {"x": 48, "y": 15})

    controller.terminal_calibrator.observe_outcome(action, True)
    controller.start_branch()

    assert controller.terminal_calibrator.is_observed_danger(action)
