from __future__ import annotations

from theory.sage_t import paired_active_gate_v9_4 as active


def test_paired_bootstrap_is_deterministic_and_strictly_positive() -> None:
    values = (0.15, 0.15, 0.15, 0.0, 0.0, 0.0)

    first = active._paired_interval(values)
    second = active._paired_interval(values)

    assert first == second
    assert first["n"] == 6
    assert first["lower_95"] > 0.0


def test_active_manifest_is_paired_and_keeps_holdout_closed() -> None:
    manifest = active.load_manifest()

    assert manifest["seeds"] == [0, 1, 2]
    assert manifest["pairing"]["same_game_seed_reset_and_action_budget"] is True
    assert manifest["experimental_authority"]["mode"] == "active"
    assert manifest["firewall"]["production_active_authority"] is False
    assert manifest["firewall"]["holdout_opened"] is False


def test_safe_active_controller_reaches_active_mode() -> None:
    controller = active.build_controller(active.load_manifest())

    assert controller.effective_mode.value == "active"
    assert controller.maximum_marginal_terminal_risk == 0.05
    assert controller.posterior.maximum_repair_contexts == 0
