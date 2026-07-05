"""
Run a small frozen V4 progress ablation across multiple progress profiles.

Usage:
    python run_progress_ablation_v4.py
    python run_progress_ablation_v4.py 2 --num-games 4
    python run_progress_ablation_v4.py 3 --games ar25-e3c63847 bp35-0a0ad940
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from test_v4_agent import ENV_DIR, PROJECT_ROOT, run_game

from arc_agi import Arcade, OperationMode

from v4.memory.cross_game_memory import CrossGameMemoryV4

PROFILES = ["strict", "strict_plus", "relaxed_sp", "relaxed_tp"]
DEFAULT_OUTPUT = PROJECT_ROOT / "progress_ablation_v4.json"
DEFAULT_DIAGNOSTIC_ROOT = PROJECT_ROOT / "diagnostics" / "v4" / "progress_ablation"
WIDTH = 108


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strict and slightly relaxed V4 progress profiles."
    )
    parser.add_argument("time_budget", nargs="?", type=int, default=2)
    parser.add_argument("--num-games", type=int, default=4)
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--diagnostic-root",
        default=str(DEFAULT_DIAGNOSTIC_ROOT),
        help="Directory for per-profile per-game diagnostic dumps.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT),
        help="JSON file for saving the ablation summary.",
    )
    parser.add_argument(
        "--assert-causal-chain",
        action="store_true",
        help="Enable strict causal assertions during the ablation sweep.",
    )
    return parser.parse_args()


def merge_counter(target: Counter[str], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] += int(value)


def summarize_profile(profile: str, results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    count = max(len(results), 1)
    observed = {"lp": Counter(), "sp": Counter(), "tp": Counter()}
    awarded = {"lp": Counter(), "sp": Counter(), "tp": Counter()}
    missed = {"sp": Counter(), "tp": Counter()}

    for result in results:
        progress_detail = result.get("progress_detail", {})
        observed_counts = progress_detail.get("observed_event_counts", {})
        awarded_counts = progress_detail.get("awarded_event_counts", {})
        missed_counts = progress_detail.get("missed_event_counts", {})
        for channel in ("lp", "sp", "tp"):
            merge_counter(observed[channel], observed_counts.get(channel, {}))
            merge_counter(awarded[channel], awarded_counts.get(channel, {}))
        for channel in ("sp", "tp"):
            merge_counter(missed[channel], missed_counts.get(channel, {}))

    return {
        "profile": profile,
        "games": len(results),
        "solved": sum(1 for item in results if item["won"]),
        "total_levels": sum(item["levels"] for item in results),
        "total_actions": sum(item["actions"] for item in results),
        "avg_pred_acc": sum(item["pred_acc"] for item in results) / count,
        "avg_ctrl_suc": sum(item["ctrl_suc"] for item in results) / count,
        "avg_lp": sum(item["lp"] for item in results) / count,
        "avg_sp": sum(item["sp"] for item in results) / count,
        "avg_tp": sum(item["tp"] for item in results) / count,
        "avg_knowledge": sum(item["knowledge"] for item in results) / count,
        "wall_clock": round(elapsed, 1),
        "observed_event_counts": {
            channel: dict(counter)
            for channel, counter in observed.items()
        },
        "awarded_event_counts": {
            channel: dict(counter)
            for channel, counter in awarded.items()
        },
        "missed_event_counts": {
            channel: dict(counter)
            for channel, counter in missed.items()
        },
        "results": results,
    }


def print_profile_summary(summary: dict[str, Any]) -> None:
    print()
    print("-" * WIDTH)
    print(f"  PROFILE {summary['profile']}")
    print("-" * WIDTH)
    print(f"  Solved:            {summary['solved']}/{summary['games']} games")
    print(f"  Total levels:      {summary['total_levels']}")
    print(f"  Total actions:     {summary['total_actions']:,}")
    print(f"  Avg knowledge:     {summary['avg_knowledge']:.2f}")
    print(f"  Avg pred accuracy: {summary['avg_pred_acc']:.1%}")
    print(f"  Avg ctrl success:  {summary['avg_ctrl_suc']:.1%}")
    print(
        f"  Avg progress:      LP={summary['avg_lp']:.2f}  SP={summary['avg_sp']:.2f}  TP={summary['avg_tp']:.2f}"
    )
    print(f"  Wall-clock:        {summary['wall_clock']:.1f}s")

    sp_awarded = summary["awarded_event_counts"].get("sp", {})
    sp_missed = summary["missed_event_counts"].get("sp", {})
    tp_awarded = summary["awarded_event_counts"].get("tp", {})
    tp_missed = summary["missed_event_counts"].get("tp", {})
    print(f"  SP awarded:        {sp_awarded}")
    print(f"  SP missed:         {sp_missed}")
    print(f"  TP awarded:        {tp_awarded}")
    print(f"  TP missed:         {tp_missed}")


def print_comparison_table(summaries: list[dict[str, Any]]) -> None:
    print()
    print("=" * WIDTH)
    print("  PROGRESS ABLATION COMPARISON")
    print("=" * WIDTH)
    print(
        f"  {'Profile':<12} {'Solved':>8} {'Pred':>7} {'Ctrl':>7} {'LP':>5} {'SP':>5} {'TP':>5} {'Acts':>8}"
    )
    print("  " + "-" * (WIDTH - 4))
    for summary in summaries:
        print(
            f"  {summary['profile']:<12} "
            f"{summary['solved']:>2}/{summary['games']:<5} "
            f"{summary['avg_pred_acc']:>6.1%} "
            f"{summary['avg_ctrl_suc']:>6.1%} "
            f"{summary['avg_lp']:>5.2f} "
            f"{summary['avg_sp']:>5.2f} "
            f"{summary['avg_tp']:>5.2f} "
            f"{summary['total_actions']:>8,}"
        )


def save_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    env_game_ids = [env.game_id for env in envs]
    if args.games:
        game_ids = list(args.games)
    else:
        game_ids = env_game_ids[: args.num_games]
    if args.shuffle:
        random.shuffle(game_ids)

    diagnostic_root = Path(args.diagnostic_root)
    output_path = Path(args.output_path)
    payload = {
        "time_budget": args.time_budget,
        "games": game_ids,
        "profiles": {},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print()
    print("=" * WIDTH)
    print(
        f"  V4 PROGRESS ABLATION  |  Games: {len(game_ids)}  |  Budget: {args.time_budget}s  |  Profiles: {', '.join(PROFILES)}"
    )
    print("=" * WIDTH)
    print(f"  Frozen transfer: yes")
    print(f"  Frozen learning: yes")
    print(f"  Games: {', '.join(game_ids)}")

    summaries: list[dict[str, Any]] = []
    for profile in PROFILES:
        print()
        print("=" * WIDTH)
        print(f"  RUNNING PROFILE: {profile}")
        print("=" * WIDTH)
        results: list[dict[str, Any]] = []
        start = time.time()
        for index, game_id in enumerate(game_ids, 1):
            print(f"  [{index}/{len(game_ids)}] {game_id} ...", end="", flush=True)
            diagnostic_path = diagnostic_root / profile / f"{index:02d}_{game_id}.json"
            result = run_game(
                arc,
                game_id,
                args.time_budget,
                CrossGameMemoryV4(),
                agent_options={
                    "freeze_transfer": True,
                    "freeze_learning_updates": True,
                    "progress_profile": profile,
                    "diagnostics": {
                        "enabled": True,
                        "dump_path": str(diagnostic_path),
                        "assert_causal_chain": args.assert_causal_chain,
                    },
                },
            )
            results.append(result)
            print(
                f"  L{result['levels']} k={result['knowledge']:.2f} "
                f"LP={result['lp']:.2f} SP={result['sp']:.2f} TP={result['tp']:.2f} ({result['time']:.0f}s)"
            )

        summary = summarize_profile(profile, results, time.time() - start)
        summaries.append(summary)
        payload["profiles"][profile] = summary
        print_profile_summary(summary)

    save_report(output_path, payload)
    print_comparison_table(summaries)
    print()
    print(f"  Ablation report saved to {output_path}")
    print(f"  Diagnostic dumps saved under {diagnostic_root}")


if __name__ == "__main__":
    main()
