"""
Loop V4 evaluation over multiple full sweeps so cross-game learning compounds.

Usage:
    python run_training_loop_v4.py [iterations] [num_games] [time_budget]

Examples:
    python run_training_loop_v4.py
    python run_training_loop_v4.py 10 25 10
    python run_training_loop_v4.py 5 50 20 --fresh
    python run_training_loop_v4.py 8 25 15 --shuffle
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from test_v4_agent import ENV_DIR, PROJECT_ROOT, run_game

from arc_agi import Arcade, OperationMode

from v4.memory.cross_game_memory import CrossGameMemoryV4

MEMORY_PATH = PROJECT_ROOT / "cross_game_memory_v4.pkl"
FRESH_MEMORY_PATH = PROJECT_ROOT / "cross_game_memory_v4_fresh.pkl"
HISTORY_PATH = PROJECT_ROOT / "training_history_v4.json"
WIDTH = 108


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated V4 training sweeps over ARC-AGI-3 games."
    )
    parser.add_argument("iterations", nargs="?", type=int, default=5)
    parser.add_argument("num_games", nargs="?", type=int, default=25)
    parser.add_argument("time_budget", nargs="?", type=int, default=60)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved cross-game memory and start from a fresh in-memory V4 memory.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the game order independently on each iteration.",
    )
    parser.add_argument(
        "--history-path",
        default=str(HISTORY_PATH),
        help="Optional path for writing JSON training history.",
    )
    parser.add_argument(
        "--memory-path",
        default=None,
        help=(
            "Optional cross-game memory path. By default this uses cross_game_memory_v4.pkl, "
            "or cross_game_memory_v4_fresh.pkl when --fresh is set."
        ),
    )
    parser.add_argument(
        "--freeze-transfer",
        action="store_true",
        help="Do not seed from or export to cross-game memory during the run.",
    )
    parser.add_argument(
        "--freeze-learning",
        action="store_true",
        help="Freeze within-game learning updates while keeping the symbolic V4 stack active.",
    )
    parser.add_argument(
        "--diagnostic-dir",
        default=None,
        help="Optional directory for per-game diagnostic JSON dumps.",
    )
    parser.add_argument(
        "--assert-causal-chain",
        action="store_true",
        help="Enable strict causal-chain assertions during diagnostic runs.",
    )
    parser.add_argument(
        "--progress-profile",
        default="strict_plus",
        choices=["strict", "strict_plus", "relaxed_sp", "relaxed_tp"],
        help="Progress profile to use for all games in this run.",
    )
    return parser.parse_args()


def aggregate_progress_counts(
    results: list[dict[str, Any]],
    bucket_name: str,
    channel: str,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in results:
        progress_detail = item.get("progress_detail", {})
        counts = progress_detail.get(bucket_name, {}).get(channel, {})
        for key, value in counts.items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


def summarize_selected_counts(counts: dict[str, int], keys: list[str]) -> str:
    parts = [f"{key}={counts.get(key, 0)}" for key in keys if counts.get(key, 0) > 0]
    return ", ".join(parts) if parts else "-"


def format_cross_game(cross_game: CrossGameMemoryV4) -> list[str]:
    return [
        f"Games: {cross_game.games_played}  |  Won: {cross_game.games_won}  |  Trust: {cross_game.trust:.2f}",
        f"Ontology priors: {len(cross_game.ontology_priors)}",
        (
            f"Operator templates: {sum(len(items) for items in cross_game.operator_templates.values())} "
            f"across {len(cross_game.operator_templates)} kinds"
        ),
        f"Law families: {len(cross_game.law_families)}",
        f"Terminal motifs: {len(cross_game.terminal_motifs)}",
        f"Ritual signatures: {len(cross_game.ritual_signatures)}",
        f"World frame embeddings: {len(getattr(cross_game, 'learned_world_frame_embeddings', {}))}",
        f"World episode embeddings: {len(getattr(cross_game, 'learned_world_episode_embeddings', {}))}",
    ]


def run_iteration(
    arc: Arcade,
    game_ids: list[str],
    time_budget: int,
    cross_game: CrossGameMemoryV4,
    iteration_index: int,
    iterations: int,
    *,
    freeze_transfer: bool = False,
    freeze_learning: bool = False,
    diagnostic_dir: Path | None = None,
    assert_causal_chain: bool = False,
    progress_profile: str = "strict",
) -> dict[str, Any]:
    print()
    print("=" * WIDTH)
    print(f"  ITERATION {iteration_index}/{iterations}  |  Games: {len(game_ids)}  |  Budget: {time_budget}s/game")
    print("=" * WIDTH)
    print()

    results: list[dict[str, Any]] = []
    start = time.time()
    for index, game_id in enumerate(game_ids, 1):
        print(f"  [{index}/{len(game_ids)}] {game_id} ...", end="", flush=True)
        diagnostic_path = None
        if diagnostic_dir is not None:
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            diagnostic_path = diagnostic_dir / f"{index:02d}_{game_id}.json"
        agent_options = {
            "freeze_transfer": freeze_transfer,
            "freeze_learning_updates": freeze_learning,
            "progress_profile": progress_profile,
            "diagnostics": {
                "enabled": diagnostic_path is not None or assert_causal_chain,
                "dump_path": str(diagnostic_path) if diagnostic_path is not None else None,
                "assert_causal_chain": assert_causal_chain,
            },
        }
        result = run_game(arc, game_id, time_budget, cross_game, agent_options=agent_options)
        results.append(result)
        if result["won"]:
            print(f"  WIN  L{result['levels']} ({result['time']:.0f}s)")
        else:
            print(
                f"  ---  L{result['levels']} ops={result['operators']} rules={result['rules']} tel={result['teleology']} "
                f"mot={result['motifs']} rit={result['rituals']} "
                f"onto={result['ontology']} k={result['knowledge']:.2f} "
                f"LP={result['lp']:.2f} SP={result['sp']:.2f} TP={result['tp']:.2f} "
                f"({result['time']:.0f}s)"
            )

    elapsed = time.time() - start
    summary = summarize_iteration(results, elapsed, cross_game)
    print_iteration_summary(summary)
    return summary


def summarize_iteration(
    results: list[dict[str, Any]],
    elapsed: float,
    cross_game: CrossGameMemoryV4,
) -> dict[str, Any]:
    solved = sum(1 for item in results if item["won"])
    total_levels = sum(item["levels"] for item in results)
    total_actions = sum(item["actions"] for item in results)
    total_time = sum(item["time"] for item in results)
    count = max(len(results), 1)

    learning_world = [
        float(item.get("learning", {}).get("world_reliability", 0.0))
        for item in results
        if isinstance(item.get("learning"), dict)
    ]
    learning_sterility = [
        float(item.get("learning", {}).get("sterility_risk", 0.0))
        for item in results
        if isinstance(item.get("learning"), dict)
    ]

    return {
        "results": results,
        "games": len(results),
        "solved": solved,
        "total_levels": total_levels,
        "total_actions": total_actions,
        "avg_actions": total_actions / count,
        "avg_operators": sum(item["operators"] for item in results) / count,
        "avg_rules": sum(item["rules"] for item in results) / count,
        "avg_teleology": sum(item["teleology"] for item in results) / count,
        "avg_motifs": sum(item["motifs"] for item in results) / count,
        "avg_rituals": sum(item["rituals"] for item in results) / count,
        "avg_knowledge": sum(item["knowledge"] for item in results) / count,
        "avg_pred_acc": sum(item["pred_acc"] for item in results) / count,
        "avg_ctrl_suc": sum(item["ctrl_suc"] for item in results) / count,
        "avg_lp": sum(item["lp"] for item in results) / count,
        "avg_sp": sum(item["sp"] for item in results) / count,
        "avg_tp": sum(item["tp"] for item in results) / count,
        "avg_time": total_time / count,
        "wall_clock": round(elapsed, 1),
        "avg_world_reliability": (sum(learning_world) / len(learning_world)) if learning_world else 0.0,
        "avg_sterility_risk": (sum(learning_sterility) / len(learning_sterility)) if learning_sterility else 0.0,
        "progress_profile": next(
            (item.get("progress_detail", {}).get("profile") for item in results if item.get("progress_detail")),
            "unknown",
        ),
        "sp_awarded": aggregate_progress_counts(results, "awarded_event_counts", "sp"),
        "sp_missed": aggregate_progress_counts(results, "missed_event_counts", "sp"),
        "cross_game": {
            "games_played": cross_game.games_played,
            "games_won": cross_game.games_won,
            "trust": cross_game.trust,
            "ontology_priors": len(cross_game.ontology_priors),
            "operator_templates": sum(len(items) for items in cross_game.operator_templates.values()),
            "operator_kinds": len(cross_game.operator_templates),
            "law_families": len(cross_game.law_families),
            "terminal_motifs": len(cross_game.terminal_motifs),
            "ritual_signatures": len(cross_game.ritual_signatures),
            "world_frame_embeddings": len(getattr(cross_game, "learned_world_frame_embeddings", {})),
            "world_episode_embeddings": len(getattr(cross_game, "learned_world_episode_embeddings", {})),
        },
    }


def print_iteration_summary(summary: dict[str, Any]) -> None:
    print()
    print("-" * WIDTH)
    print("  ITERATION SUMMARY")
    print("-" * WIDTH)
    print(f"  Solved:            {summary['solved']}/{summary['games']} games")
    print(f"  Total levels:      {summary['total_levels']}")
    print(f"  Total actions:     {summary['total_actions']:,}")
    print(f"  Avg operators:     {summary['avg_operators']:.1f}")
    print(f"  Avg rules:         {summary['avg_rules']:.1f}")
    print(f"  Avg teleology:     {summary['avg_teleology']:.1f}")
    print(f"  Avg motifs:        {summary['avg_motifs']:.1f}")
    print(f"  Avg rituals:       {summary['avg_rituals']:.1f}")
    print(f"  Avg knowledge:     {summary['avg_knowledge']:.2f}")
    print(f"  Avg pred accuracy: {summary['avg_pred_acc']:.1%}")
    print(f"  Avg ctrl success:  {summary['avg_ctrl_suc']:.1%}")
    print(f"  Progress profile:  {summary['progress_profile']}")
    print(
        f"  Avg progress:      LP={summary['avg_lp']:.2f}  SP={summary['avg_sp']:.2f}  TP={summary['avg_tp']:.2f}"
    )
    print(
        "  SP awarded:       "
        + summarize_selected_counts(
            summary["sp_awarded"],
            ["object_change", "structural_change", "novel_state", "region_unlock", "new_rules", "class_depletion"],
        )
    )
    print(
        "  SP missed:        "
        + summarize_selected_counts(
            summary["sp_missed"],
            ["object_change", "structural_change", "grid_change_without_sp", "class_depletion"],
        )
    )
    print(
        f"  Learned eval:      world={summary['avg_world_reliability']:.2f}  sterility={summary['avg_sterility_risk']:.2f}"
    )
    print(f"  Time:              {summary['wall_clock']:.1f}s wall-clock ({summary['avg_time']:.1f}s avg/game)")
    print()
    print("  CROSS-GAME MEMORY (V4)")
    for line in format_cross_game_object(summary["cross_game"]):
        print(f"  {line}")


def format_cross_game_object(data: dict[str, Any]) -> list[str]:
    return [
        f"Games: {data['games_played']}  |  Won: {data['games_won']}  |  Trust: {data['trust']:.2f}",
        f"Ontology priors: {data['ontology_priors']}",
        f"Operator templates: {data['operator_templates']} across {data['operator_kinds']} kinds",
        f"Law families: {data['law_families']}",
        f"Terminal motifs: {data['terminal_motifs']}",
        f"Ritual signatures: {data['ritual_signatures']}",
        f"World frame embeddings: {data['world_frame_embeddings']}",
        f"World episode embeddings: {data['world_episode_embeddings']}",
    ]


def print_training_history(history: list[dict[str, Any]]) -> None:
    print()
    print("=" * WIDTH)
    print("  TRAINING HISTORY")
    print("=" * WIDTH)
    print(
        f"  {'Iter':>4} {'Solved':>8} {'Levels':>6} {'Actions':>8} {'Pred':>7} {'Ctrl':>7} "
        f"{'LP':>5} {'SP':>5} {'TP':>5} {'Trust':>6} {'WRel':>6} {'Ster':>6}"
    )
    print("  " + "-" * (WIDTH - 4))
    for index, item in enumerate(history, 1):
        print(
            f"  {index:>4} "
            f"{item['solved']:>2}/{item['games']:<5} "
            f"{item['total_levels']:>6} "
            f"{item['total_actions']:>8,} "
            f"{item['avg_pred_acc']:>6.1%} "
            f"{item['avg_ctrl_suc']:>6.1%} "
            f"{item['avg_lp']:>5.2f} "
            f"{item['avg_sp']:>5.2f} "
            f"{item['avg_tp']:>5.2f} "
            f"{item['cross_game']['trust']:>6.2f} "
            f"{item['avg_world_reliability']:>6.2f} "
            f"{item['avg_sterility_risk']:>6.2f}"
        )


def history_payload(
    args: argparse.Namespace,
    game_ids: list[str],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "iterations": args.iterations,
        "num_games": args.num_games,
        "time_budget": args.time_budget,
        "fresh": args.fresh,
        "shuffle": args.shuffle,
        "freeze_transfer": args.freeze_transfer,
        "freeze_learning": args.freeze_learning,
        "diagnostic_dir": args.diagnostic_dir,
        "assert_causal_chain": args.assert_causal_chain,
        "progress_profile": args.progress_profile,
        "game_ids": game_ids,
        "history": history,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_history(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    memory_path = Path(args.memory_path) if args.memory_path else (FRESH_MEMORY_PATH if args.fresh else MEMORY_PATH)

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    base_game_ids = [env.game_id for env in envs][: args.num_games]

    cross_game = CrossGameMemoryV4() if args.fresh else CrossGameMemoryV4.load(str(memory_path))

    print()
    print("=" * WIDTH)
    print(
        f"  V4 TRAINING LOOP  |  Iterations: {args.iterations}  |  Games/iter: {len(base_game_ids)}  |  Budget: {args.time_budget}s"
    )
    print("=" * WIDTH)
    if args.fresh:
        print(f"  Starting from fresh V4 cross-game memory at {memory_path}")
    elif cross_game.games_played > 0:
        print(
            f"  Resuming: {cross_game.games_played} games, {cross_game.games_won} won, trust={cross_game.trust:.2f}"
        )
    else:
        print(f"  Starting from saved-memory path, but no prior V4 games were found: {memory_path}")
    if args.freeze_transfer:
        print("  Cross-game transfer is frozen for this run")
    if args.freeze_learning:
        print("  Learned updates are frozen for this run")
    if args.diagnostic_dir:
        print(f"  Diagnostic dumps will be written under {args.diagnostic_dir}")
    print(f"  Progress profile: {args.progress_profile}")
    print()

    history: list[dict[str, Any]] = []
    overall_start = time.time()

    for iteration in range(1, args.iterations + 1):
        game_ids = list(base_game_ids)
        if args.shuffle:
            random.shuffle(game_ids)
        iteration_diag_dir = None
        if args.diagnostic_dir:
            iteration_diag_dir = Path(args.diagnostic_dir) / f"iter_{iteration:03d}"
        iteration_summary = run_iteration(
            arc=arc,
            game_ids=game_ids,
            time_budget=args.time_budget,
            cross_game=cross_game,
            iteration_index=iteration,
            iterations=args.iterations,
            freeze_transfer=args.freeze_transfer,
            freeze_learning=args.freeze_learning,
            diagnostic_dir=iteration_diag_dir,
            assert_causal_chain=args.assert_causal_chain,
            progress_profile=args.progress_profile,
        )
        history.append(iteration_summary)
        if not args.freeze_transfer:
            cross_game.save(str(memory_path))
        save_history(Path(args.history_path), history_payload(args, game_ids, history))
        if not args.freeze_transfer:
            print(f"\n  Cross-game memory saved to {memory_path}")
        print(f"  Training history saved to {args.history_path}")

    total_elapsed = time.time() - overall_start
    print_training_history(history)
    print()
    print("=" * WIDTH)
    print(f"  TRAINING COMPLETE  |  Total wall-clock: {total_elapsed:.1f}s")
    print("=" * WIDTH)
    for line in format_cross_game(cross_game):
        print(f"  {line}")
    print()
    if not args.freeze_transfer:
        print(f"  Cross-game memory saved to {memory_path}")
    print(f"  Training history saved to {args.history_path}")


if __name__ == "__main__":
    main()
