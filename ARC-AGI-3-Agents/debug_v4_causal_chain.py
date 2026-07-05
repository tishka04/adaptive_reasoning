"""
Run a single V4 game in diagnostic mode with transfer and learning updates frozen.

Usage:
    python debug_v4_causal_chain.py [game_id] [time_budget]
    python debug_v4_causal_chain.py ar25-e3c63847 5 --dump-path diagnostics/v4/ar25.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from test_v4_agent import ENV_DIR, PROJECT_ROOT, run_game

from arc_agi import Arcade, OperationMode

from v4.memory.cross_game_memory import CrossGameMemoryV4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ARC game through V4 with heavy causal-chain diagnostics."
    )
    parser.add_argument("game_id", nargs="?", default=None)
    parser.add_argument("time_budget", nargs="?", type=int, default=5)
    parser.add_argument(
        "--dump-path",
        default=None,
        help="Optional diagnostic JSON path. Defaults to diagnostics/v4/<game_id>_causal_chain.json",
    )
    parser.add_argument(
        "--progress-profile",
        default="strict_plus",
        choices=["strict", "strict_plus", "relaxed_sp", "relaxed_tp"],
        help="Progress profile to use during the diagnostic run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    game_ids = [env.game_id for env in envs]
    game_id = args.game_id or game_ids[0]
    if game_id not in game_ids:
        raise SystemExit(f"Unknown game_id: {game_id}")

    dump_path = Path(args.dump_path) if args.dump_path else (
        PROJECT_ROOT / "diagnostics" / "v4" / f"{game_id}_causal_chain.json"
    )
    cross_game = CrossGameMemoryV4()
    result = run_game(
        arc,
        game_id,
        args.time_budget,
        cross_game,
        agent_options={
            "freeze_transfer": True,
            "freeze_learning_updates": True,
            "progress_profile": args.progress_profile,
            "diagnostics": {
                "enabled": True,
                "dump_path": str(dump_path),
                "assert_causal_chain": True,
            },
        },
    )

    diagnostics = result.get("diagnostics", {})
    print()
    print("=" * 108)
    print(f"  V4 CAUSAL-CHAIN DEBUG  |  Game: {game_id}  |  Budget: {args.time_budget}s")
    print("=" * 108)
    print(f"  Won:               {result['won']}")
    print(f"  Levels completed:  {result['levels']}")
    print(f"  Actions:           {result['actions']}")
    print(
        f"  Progress:          LP={result['lp']:.2f}  SP={result['sp']:.2f}  TP={result['tp']:.2f}"
    )
    progress_detail = result.get("progress_detail", {})
    print(f"  Progress profile:  {progress_detail.get('profile', args.progress_profile)}")
    print(
        f"  Structures:        ops={result['operators']}  rules={result['rules']}  "
        f"tel={result['teleology']}  mot={result['motifs']}  rit={result['rituals']}"
    )
    print(
        f"  Diagnostics:       records={diagnostics.get('records', 0)}  "
        f"assert_failures={len(diagnostics.get('assertion_failures', []))}"
    )
    print(f"  Dump path:         {dump_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
