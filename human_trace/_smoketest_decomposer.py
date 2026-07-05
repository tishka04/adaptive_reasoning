"""Smoke test: verify GoalDecomposer honours human priors.

Run with:
    .\\ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m human_trace._smoketest_decomposer
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from v4_1_reasoning_system.arc_agi.game_memory import GameMemory
from v4_1_reasoning_system.arc_agi.goal_decomposer import GoalDecomposer
from v4_1_reasoning_system.arc_agi.state_describer import GameObservation


def _make_fake_obs() -> GameObservation:
    """A plausible navigation scene (player + 3 coloured objects)."""
    grid = np.zeros((16, 16), dtype=np.int32)
    grid[5, 5] = 1   # player-ish
    grid[3, 9] = 4
    grid[10, 2] = 4
    grid[12, 12] = 5
    return GameObservation(
        grid_description="16x16 grid with player + 3 coloured objects",
        objects=[
            {"value": 1, "size": 1, "center_y": 5, "center_x": 5, "is_player": True},
            {"value": 4, "size": 1, "center_y": 3, "center_x": 9, "is_player": False},
            {"value": 4, "size": 1, "center_y": 10, "center_x": 2, "is_player": False},
            {"value": 5, "size": 1, "center_y": 12, "center_x": 12, "is_player": False},
        ],
        player_info={"y": 5, "x": 5, "value": 1},
        action_semantics={},
        memory_summary={},
        raw_grid=grid,
        level=0,
        game_state="NOT_FINISHED",
        action_counter=0,
    )


def main() -> int:
    print("\n--- Test 1: unprimed memory (baseline) ---")
    gm = GameMemory()
    dec = GoalDecomposer(use_llm=False)
    goal = dec.decompose(_make_fake_obs(), gm)
    print(f"  overarching: {goal.overarching_goal}")
    print(f"  hypothesis : {goal.hypothesis}")
    bank = dec.generate_goal_bank(_make_fake_obs(), gm)
    print(f"  goal_bank  : {[(g.id, round(g.confidence, 2)) for g in bank]}")
    assert not any(g.id.startswith("human_prior") for g in bank), \
        "Unprimed bank should not contain human prior objectives"

    print("\n--- Test 2: primed with navigate_puzzle game_type + objective ---")
    gm2 = GameMemory()
    gm2.add_hypothesis("game_type::navigate_puzzle", 0.65)
    gm2.add_hypothesis("objective::correspond_shapes", 0.60)
    gm2.add_hypothesis("mechanic::moves_left_sideline", 0.55)
    gm2.add_hypothesis("mechanic::black_doted_shape_is_player", 0.55)
    gm2.add_hypothesis("human::doted_line_act_as_miror", 0.80)

    goal2 = dec.decompose(_make_fake_obs(), gm2)
    print(f"  overarching: {goal2.overarching_goal}")
    print(f"  hypothesis : {goal2.hypothesis}")

    # Expectations:
    # - overarching replaced with the human objective (game_type 0.65 > 0.6 → override)
    # - hypothesis string mentions mechanics
    assert "correspond_shapes" in goal2.overarching_goal, goal2.overarching_goal
    assert "mechanic" in goal2.hypothesis, goal2.hypothesis

    bank2 = dec.generate_goal_bank(_make_fake_obs(), gm2)
    print(f"  goal_bank  : {[(g.id, round(g.confidence, 2)) for g in bank2]}")
    ids = [g.id for g in bank2]
    assert any(i.startswith("human_prior") for i in ids), ids
    # And it should be at or near the top (confidence-sorted).
    top = bank2[0]
    assert top.id.startswith("human_prior"), f"expected human_prior at top, got {top.id}"
    print(f"  top goal   : {top.id}  (conf={top.confidence:.2f})")
    print(f"  top desc   : {top.description}")

    print("\n--- Test 3: low-confidence prior should NOT override game_type ---")
    gm3 = GameMemory()
    gm3.add_hypothesis("game_type::navigate_puzzle", 0.40)  # below 0.6 threshold
    gm3.add_hypothesis("objective::reach_the_blue_tile", 0.40)

    goal3 = dec.decompose(_make_fake_obs(), gm3)
    print(f"  overarching: {goal3.overarching_goal}")
    print(f"  hypothesis : {goal3.hypothesis}")
    # Low confidence: objective string still injected (no threshold on objective)
    # but game_type heuristic is preserved (no "Human prior classifies" in hyp).
    assert "Human prior" not in (goal3.hypothesis or ""), goal3.hypothesis

    print("\nALL GOOD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
