"""CLI entry point for the human recorder.

Usage:
    python -m human_trace.run_recorder --game ar25 [--mode offline] [--out human_traces]

The transport is selected via `--mode`:
  offline  : drive the local simulator in `environment_files/` only (default)
  online   : always hit the ARC-AGI-3 API
  normal   : prefer local; download+fallback to API if missing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `human_trace` resolves when the
# user invokes `python human_trace/run_recorder.py` instead of `-m`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from human_trace.recorder import HumanRecorder
from human_trace.schema import TraceWriter


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="human_trace.run_recorder",
        description="Human-controlled ARC-AGI-3 recorder (pygame UI).",
    )
    parser.add_argument("--game", required=True, help="game_id prefix (e.g. ar25)")
    parser.add_argument(
        "--mode",
        choices=["offline", "online", "normal"],
        default="offline",
        help="Transport mode (default: offline = local simulator).",
    )
    parser.add_argument(
        "--environments-dir",
        default="environment_files",
        help="Directory holding local env metadata.json files.",
    )
    parser.add_argument(
        "--out",
        default="human_traces",
        help="Output directory for JSONL traces.",
    )
    parser.add_argument(
        "--cell-size", type=int, default=10,
        help="Pixels per grid cell (default 10 → 640px side for 64x64).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose >= 2 else (logging.INFO if args.verbose == 1 else logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Lazy import so `--help` works even without the arc_agi SDK installed.
    from arc_agi import Arcade, OperationMode

    mode_map = {
        "offline": OperationMode.OFFLINE,
        "online": OperationMode.ONLINE,
        "normal": OperationMode.NORMAL,
    }

    arcade = Arcade(
        operation_mode=mode_map[args.mode],
        environments_dir=args.environments_dir,
    )
    env = arcade.make(args.game, seed=args.seed)
    if env is None:
        print(f"Failed to make environment for {args.game!r} in mode {args.mode}.", file=sys.stderr)
        return 2

    # Resolve the full game_id (with version suffix) from the wrapper.
    resolved_game_id = env.info.game_id if hasattr(env, "info") else args.game

    writer = TraceWriter(args.out, game_id=resolved_game_id)
    print(f"Recording to {writer.steps_path}")
    print(f"      and    {writer.episodes_path}")

    recorder = HumanRecorder(
        env=env,
        writer=writer,
        game_id=resolved_game_id,
        cell_size=args.cell_size,
    )
    try:
        recorder.run()
    except KeyboardInterrupt:
        pass

    # Close the scorecard if the SDK created one, to keep the local store tidy.
    try:
        arcade.close_scorecard()
    except Exception:
        pass

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
