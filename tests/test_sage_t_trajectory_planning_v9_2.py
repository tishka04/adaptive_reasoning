from __future__ import annotations

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import reachability_audit_v9 as t9_0
from theory.sage_t import trajectory_planning_v9_2 as t9_2
from theory.sage_t.replay_gate import fast_panel_from_binding_pair


def _winning_root():
    pairs = load_pairs(t9_0.DEFAULT_SHARD_DIR, t9_0.SOURCE_GAMES)
    paths = t9_0.winner_paths(pairs)
    root_key = min(paths)
    return next(pair for pair in pairs if pair.root_key == root_key and pair.path == "")


def test_structural_macro_contains_the_coordinate_free_rrr_instantiation() -> None:
    pair = _winning_root()
    panel = fast_panel_from_binding_pair(pair)
    macros = t9_2.structural_macros(panel.state, tuple(arm.action for arm in panel.arms))
    right = panel.arms[1].action

    assert len(macros) <= 8
    assert tuple(right for _ in range(3)) in macros
    assert all(len(macro) <= 8 for macro in macros)


def test_balanced_controller_has_registered_long_horizon_caps() -> None:
    controller = t9_2.build_controller(
        t9_2.CHALLENGERS["balanced_h3"],
        terminal_policy_name="safe_after_3",
    )

    assert controller.config.ordinary_horizon == 3
    assert controller.config.maximum_sequences == 32
    assert controller.config.maximum_particles_per_decision == 8
    assert controller.maximum_structural_macros == 8
