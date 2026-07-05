"""Play each V4 game until WIN (or stagnation/time ceiling), then move on.

For every game we:
  1. Run the V4 agent with a large wall-clock ceiling.
  2. Abort early if no level progress happens for `--stagnation` seconds.
  3. Extract a compact "what was useful / useless" digest via
     `v4.memory.win_digest.build_digest` and merge it into the V4
     cross-game memory keyed by game_id.
  4. Continue to the next game.

The idea: keep cross-game memory compact (~KB per game), learn per-game
priors that can steer future runs, and let the agent actually solve a
game rather than being timed out arbitrarily.

Usage:
    python run_until_win_v4.py [num_games] [time_ceiling] [stagnation]

Examples:
    # 25 games, 15-min ceiling per game, skip if no level progress for 3 min
    python run_until_win_v4.py 25 900 180

    # Just one game (tr87) with a long ceiling
    python run_until_win_v4.py --only tr87 --ceiling 1800 --stagnation 300
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_DIR = PROJECT_ROOT / "environment_files"

os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", "recordings")
os.environ["TQDM_DISABLE"] = "1"

logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi import Arcade, OperationMode
from arcengine import GameState

from v4.memory.cross_game_memory import CrossGameMemoryV4
from v4.memory.win_digest import build_digest, merge_into_cross_game


MEMORY_PATH = PROJECT_ROOT / "cross_game_memory_v4.pkl"
WIDTH = 108


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play each game until WIN, then harvest a compact digest."
    )
    parser.add_argument("num_games", nargs="?", type=int, default=25)
    parser.add_argument("ceiling_pos", nargs="?", type=int, default=None,
                        help="Hard per-game wall-clock ceiling in seconds.")
    parser.add_argument("stagnation_pos", nargs="?", type=int, default=None,
                        help="Abort a game after N seconds without a level transition.")
    parser.add_argument("--ceiling", type=int, default=900,
                        help="Hard per-game wall-clock ceiling in seconds (flag form).")
    parser.add_argument("--stagnation", type=int, default=180,
                        help="Abort a game after N seconds without a level transition (flag form).")
    parser.add_argument("--only", default=None,
                        help="Run only a single game_id (overrides num_games).")
    parser.add_argument("--memory-path", default=str(MEMORY_PATH))
    parser.add_argument("--fresh", action="store_true",
                        help="Start from an empty cross-game memory.")
    parser.add_argument("--freeze-transfer", action="store_true",
                        help="Do not seed from or export to cross-game memory.")
    parser.add_argument("--progress-profile", default="strict_plus",
                        choices=["strict", "strict_plus", "relaxed_sp", "relaxed_tp"])
    parser.add_argument("--retry-on-loss", type=int, default=0,
                        help="If >0, replay a game up to N extra times after a non-WIN stop.")
    args = parser.parse_args()
    # Positional overrides (only if provided).
    if args.ceiling_pos is not None:
        args.ceiling = args.ceiling_pos
    if args.stagnation_pos is not None:
        args.stagnation = args.stagnation_pos
    return args


def run_single_game(
    arc: Arcade,
    game_id: str,
    *,
    ceiling: float,
    stagnation: float,
    cross_game: Optional[CrossGameMemoryV4],
    freeze_transfer: bool,
    progress_profile: str,
) -> dict[str, Any]:
    """Run one game, return a combined result dict including the digest."""
    from agents.templates.adaptive_reasoning_v4_agent import AdaptiveReasoningV4

    original_time = AdaptiveReasoningV4.TIME_BUDGET
    AdaptiveReasoningV4.TIME_BUDGET = float(ceiling)

    env = arc.make(game_id)
    agent = AdaptiveReasoningV4(
        card_id=None,
        game_id=game_id,
        agent_name="adaptivev4",
        ROOT_URL="",
        record=False,
        arc_env=env,
        cross_game=cross_game,
        arcade=arc,
        freeze_transfer=freeze_transfer,
        progress_profile=progress_profile,
        stagnation_seconds=float(stagnation) if stagnation > 0 else None,
    )

    start = time.time()
    error: Optional[str] = None
    try:
        agent.main()
    except Exception as exc:
        import traceback

        error = f"{exc}"
        print(f"\n  ERROR in {game_id}: {exc}")
        traceback.print_exc()

    elapsed = time.time() - start
    AdaptiveReasoningV4.TIME_BUDGET = original_time

    won = bool(agent.frames and agent.frames[-1].state == GameState.WIN)
    levels = int(agent.frames[-1].levels_completed) if agent.frames else 0

    # end_game() runs the standard V4 export_game + returns summary.
    summary = agent.end_game()
    wrapper = summary.get("wrapper", {})
    stop_reason = wrapper.get("stop_reason", "unknown")

    # Build compact digest from inside the agent memory.
    digest = build_digest(
        game_id=game_id,
        memory=agent.v4.memory,
        won=won,
        stop_reason=stop_reason,
        elapsed_seconds=elapsed,
    )

    # Merge digest into cross-game memory (only when we're persisting).
    if cross_game is not None and not freeze_transfer:
        merge_into_cross_game(cross_game, digest)

    return {
        "game_id": game_id,
        "won": won,
        "levels": levels,
        "actions": agent.action_counter,
        "stop_reason": stop_reason,
        "elapsed": round(elapsed, 1),
        "digest": digest.to_dict(),
        "summary": {
            "operators": summary.get("operators", 0),
            "rules": summary.get("rules", 0),
            "teleology": summary.get("teleology", 0),
            "motifs": summary.get("motifs", 0),
            "rituals": summary.get("rituals", 0),
            "knowledge": summary.get("knowledge_level", 0.0),
            "pred_acc": summary.get("pred_accuracy", 0.0),
            "ctrl_suc": summary.get("control_success", 0.0),
        },
        "error": error,
    }


def format_digest_line(digest: dict[str, Any]) -> str:
    useful_prims = ",".join(digest.get("useful_primitives", [])[:6]) or "-"
    useful_kinds = ",".join(digest.get("useful_operator_kinds", [])[:4]) or "-"
    return (
        f"useful_prim=[{useful_prims}] "
        f"op_kinds=[{useful_kinds}] "
        f"solLen={digest.get('solution_length', 0)} "
        f"rituals={len(digest.get('useful_ritual_ids', []))} "
        f"motifs={len(digest.get('useful_motif_ids', []))}"
    )


def main() -> None:
    args = parse_args()

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    all_game_ids = [env.game_id for env in envs]

    if args.only:
        game_ids = [args.only] if args.only in all_game_ids else []
        if not game_ids:
            print(f"Game {args.only!r} not found. Available: {all_game_ids[:10]} ...")
            sys.exit(1)
    else:
        game_ids = all_game_ids[: args.num_games]

    memory_path = Path(args.memory_path)
    cross_game = (
        CrossGameMemoryV4()
        if args.fresh
        else CrossGameMemoryV4.load(str(memory_path))
    )

    print()
    print("=" * WIDTH)
    print("  V4 PLAY-UNTIL-WIN LOOP")
    print("=" * WIDTH)
    print(f"  Games:           {len(game_ids)}")
    print(f"  Ceiling/game:    {args.ceiling}s")
    print(f"  Stagnation:      {args.stagnation}s")
    print(f"  Progress profile: {args.progress_profile}")
    print(f"  Memory path:     {memory_path}")
    if args.fresh:
        print("  Starting FRESH cross-game memory.")
    elif cross_game.games_played > 0:
        print(
            f"  Resuming: {cross_game.games_played} games played, "
            f"{cross_game.games_won} won, trust={cross_game.trust:.2f}, "
            f"digests={len(cross_game.game_digests)}"
        )
    print("=" * WIDTH)

    results: list[dict[str, Any]] = []
    overall_start = time.time()

    for index, game_id in enumerate(game_ids, 1):
        print(f"\n  [{index}/{len(game_ids)}] {game_id} ...", flush=True)

        attempt = 0
        result: dict[str, Any] = {}
        while True:
            attempt += 1
            if attempt > 1:
                print(f"    retry #{attempt - 1} ...", flush=True)
            result = run_single_game(
                arc,
                game_id,
                ceiling=args.ceiling,
                stagnation=args.stagnation,
                cross_game=cross_game,
                freeze_transfer=args.freeze_transfer,
                progress_profile=args.progress_profile,
            )
            if result["won"] or attempt > args.retry_on_loss:
                break

        results.append(result)

        if result["won"]:
            tag = f"WIN  L{result['levels']}"
        else:
            tag = f"{result['stop_reason']:<12} L{result['levels']}"

        print(
            f"    {tag}  acts={result['actions']:>5}  "
            f"time={result['elapsed']:>6.1f}s  "
            f"{format_digest_line(result['digest'])}"
        )

        if cross_game is not None and not args.freeze_transfer:
            cross_game.save(str(memory_path))

    total_elapsed = time.time() - overall_start

    # ---------- SUMMARY ----------
    print()
    print("=" * WIDTH)
    print("  FINAL SUMMARY")
    print("=" * WIDTH)
    won = sum(1 for r in results if r["won"])
    total_levels = sum(r["levels"] for r in results)
    total_actions = sum(r["actions"] for r in results)
    print(f"  Games won:      {won}/{len(results)}")
    print(f"  Total levels:   {total_levels}")
    print(f"  Total actions:  {total_actions:,}")
    print(f"  Wall-clock:     {total_elapsed:.1f}s "
          f"({total_elapsed / max(len(results), 1):.1f}s avg/game)")
    print()
    print("  Per-game outcomes:")
    print(f"  {'Game':<14} {'Result':<14} {'Lv':>3} {'Acts':>6} {'Time':>6}  {'Digest'}")
    print("  " + "-" * (WIDTH - 4))
    for r in results:
        outcome = "WIN" if r["won"] else r["stop_reason"]
        print(
            f"  {r['game_id'][:13]:<13} "
            f"{outcome[:13]:<13} "
            f"{r['levels']:>3} "
            f"{r['actions']:>6} "
            f"{r['elapsed']:>5.0f}s  "
            f"{format_digest_line(r['digest'])}"
        )

    print()
    print("-" * WIDTH)
    print("  CROSS-GAME MEMORY")
    print("-" * WIDTH)
    print(f"  Trust:               {cross_game.trust:.2f}")
    print(f"  Games played (all):  {cross_game.games_played}")
    print(f"  Games won (all):     {cross_game.games_won}")
    print(f"  Game digests stored: {len(cross_game.game_digests)}")
    print(f"  Operator templates:  {sum(len(v) for v in cross_game.operator_templates.values())}")
    print(f"  Law families:        {len(cross_game.law_families)}")
    print(f"  Ritual signatures:   {len(cross_game.ritual_signatures)}")
    print(f"  Terminal motifs:     {len(cross_game.terminal_motifs)}")

    if not args.freeze_transfer:
        cross_game.save(str(memory_path))
        print(f"\n  Cross-game memory saved to {memory_path}")


if __name__ == "__main__":
    main()
