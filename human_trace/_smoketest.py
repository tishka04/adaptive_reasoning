"""End-to-end smoketest: write synthetic traces, load them, seed a GameMemory.

Run with the project venv, e.g.:
    .\\ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m human_trace._smoketest
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# Ensure repo root on sys.path when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from human_trace import (
    build_prior_pack,
    load_traces,
    seed_cross_game_memory,
    seed_game_memory,
)
from human_trace.schema import EpisodeRecord, IntentTag, StepRecord, TraceWriter


def _make_fake_traces(tmp: Path) -> str:
    game_id = "ar25-e3c63847"
    writer = TraceWriter(tmp, game_id=game_id, stamp="00000000-000000")

    # Build a tiny 4x4 grid that changes after every action.
    base = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    def move(player_pos):
        g = [row[:] for row in base]
        y, x = player_pos
        g[y][x] = 1
        return g

    episode_id = "ep0000000001"
    # Winning trace: player moves right by ACTION4 three times.
    positions = [(1, 1), (1, 2), (1, 3), (1, 3)]
    actions = ["ACTION4", "ACTION4", "ACTION6"]
    action_args = [None, None, {"x": 3, "y": 1}]
    states = ["NOT_FINISHED", "NOT_FINISHED", "WIN"]
    levels = [0, 0, 1]

    for i, (a, args, s, lv) in enumerate(zip(actions, action_args, states, levels)):
        before = move(positions[i])
        after = move(positions[i + 1])
        writer.write_step(StepRecord(
            game_id=game_id,
            episode_id=episode_id,
            step=i,
            frame_before=before,
            available_actions=[1, 2, 3, 4, 5, 6],
            action=a,
            action_args=args,
            frame_after=after,
            game_state_after=s,
            levels_completed_after=lv,
            intent=IntentTag.REACH_TARGET.value if a != "ACTION6" else IntentTag.TEST_CLICK.value,
            hypothesis="blue tile might be goal",
            t_ms=i * 500,
        ))

    writer.write_episode(EpisodeRecord(
        game_id=game_id,
        episode_id=episode_id,
        started_at="2026-04-20T17:00:00+00:00",
        ended_at="2026-04-20T17:00:05+00:00",
        n_steps=len(actions),
        final_state="WIN",
        levels_completed=1,
        game_type_guess="navigate_exit",
        objective_guess="reach the blue tile and click it",
        discovered_mechanics=["ACTION4 moves player right", "ACTION6 interacts with a cell"],
        discovered_mistakes=[],
    ))
    return game_id


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="human_trace_smoke_"))
    try:
        game_id = _make_fake_traces(tmp)
        corpus = load_traces(tmp)
        assert corpus.n_episodes == 1, corpus.n_episodes
        assert corpus.n_steps == 3, corpus.n_steps
        print(f"[ok] corpus: {corpus.n_episodes} ep / {corpus.n_steps} steps")

        pack = build_prior_pack(corpus, game_id)
        assert pack.steps and pack.episodes, pack
        assert "ACTION4" in pack.action_stats, pack.action_stats
        print(f"[ok] pack: {len(pack.steps)} steps, "
              f"{len(pack.goal_hints)} goal hints, "
              f"{len(pack.hypothesis_priors)} hypotheses")

        # Seed GameMemory
        from v4_1_reasoning_system.arc_agi.game_memory import GameMemory
        from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory

        gm = GameMemory()
        stats = seed_game_memory(pack, gm)
        print(f"[ok] seed_game_memory stats: {stats}")
        assert stats["steps_replayed"] == 3, stats
        assert "ACTION4" in gm.action_profiles, list(gm.action_profiles)
        assert gm.action_profiles["ACTION4"].times_tried == 2

        # Seed CrossGameMemory
        cross = CrossGameMemory()
        xstats = seed_cross_game_memory(pack, cross)
        print(f"[ok] seed_cross_game_memory stats: {xstats}")
        assert "navigate_exit" in cross.goal_strategy_hints, cross.goal_strategy_hints
        hint = cross.goal_strategy_hints["navigate_exit"][0]
        assert hint["source"] == "human"
        assert hint["won"] is True

        print("\nALL GOOD.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
