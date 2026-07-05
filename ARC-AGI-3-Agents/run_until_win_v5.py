"""Play each V5 game until WIN (or stagnation / time ceiling).

Like `run_until_win_v4.py` but for V5. After each game we harvest a
compact digest into the V5 cross-game memory and report productive-
minority metrics at the end:

  - %_games_with_SP_positive   (sp > 0)
  - %_games_with_TP_positive   (tp > 0)
  - %_games_with_LP_above_0.5
  - %_games_with_partial_credit (any of SP>=0.3 / TP>=0.15 / LP>=0.5)

Usage:
    python run_until_win_v5.py [num_games] [ceiling] [stagnation]

Examples:
    python run_until_win_v5.py --only tr87-cd924810 --ceiling 1800 --stagnation 300
    python run_until_win_v5.py 25 900 180
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
import types
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

from v5.memory import CrossGameMemoryV5


MEMORY_PATH = PROJECT_ROOT / "cross_game_memory_v5.pkl"
WIDTH = 108


def _load_v5_wrapper() -> Any:
    """Load only the V5 bridge, bypassing eager optional agent imports."""
    package_name = "_adaptive_v5_agents"
    agents_dir = Path(__file__).parent / "agents"
    templates_dir = agents_dir / "templates"
    for name, path in (
        (package_name, agents_dir),
        (f"{package_name}.templates", templates_dir),
    ):
        if name in sys.modules:
            continue
        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(path)]
        sys.modules[name] = package
    module = importlib.import_module(
        f"{package_name}.templates.adaptive_reasoning_v5_agent"
    )
    return module.AdaptiveReasoningV5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play each V5 game until WIN, then harvest a compact digest."
    )
    parser.add_argument("num_games", nargs="?", type=int, default=25)
    parser.add_argument("ceiling_pos", nargs="?", type=int, default=None)
    parser.add_argument("stagnation_pos", nargs="?", type=int, default=None)
    parser.add_argument("--ceiling", type=int, default=900)
    parser.add_argument("--stagnation", type=int, default=180)
    parser.add_argument("--only", default=None)
    parser.add_argument("--memory-path", default=str(MEMORY_PATH))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--freeze-transfer", action="store_true")
    parser.add_argument("--retry-on-loss", type=int, default=0)
    # Ablation flags
    parser.add_argument("--no-dissent", action="store_true")
    parser.add_argument("--no-ontology", action="store_true")
    parser.add_argument("--no-rituals", action="store_true")
    parser.add_argument("--no-goal-skeleton", action="store_true")
    parser.add_argument("--no-bissociation", action="store_true")
    parser.add_argument("--learned-priors", action="store_true")
    parser.add_argument("--prior-band", type=float, default=0.10)
    parser.add_argument("--prior-w-break", type=float, default=0.10)
    parser.add_argument("--prior-w-progress", type=float, default=0.0)
    parser.add_argument(
        "--break-classifier",
        default=str(PROJECT_ROOT / "models" / "break_classifier.ar25_human.joblib"),
    )
    parser.add_argument(
        "--value",
        default=str(PROJECT_ROOT / "models" / "value_history.joblib"),
    )
    parser.add_argument(
        "--action-effect",
        default=str(PROJECT_ROOT / "models" / "action_effect.joblib"),
    )
    parser.add_argument(
        "--macro-scores",
        default=str(PROJECT_ROOT / "models" / "macro_scores_history.joblib"),
    )
    args = parser.parse_args()
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
    cross_game: Optional[CrossGameMemoryV5],
    args: argparse.Namespace,
) -> dict[str, Any]:
    AdaptiveReasoningV5 = _load_v5_wrapper()

    learned_priors = None
    if args.learned_priors:
        from v5.control.learned_priors import LearnedPriors

        learned_priors = LearnedPriors.from_paths(
            action_effect=args.action_effect,
            value=args.value,
            break_classifier=args.break_classifier,
            macro_scores=args.macro_scores,
            band=args.prior_band,
            w_break=args.prior_w_break,
            w_progress=args.prior_w_progress,
        )
        if learned_priors is None:
            print("      learned priors unavailable; continuing with baseline V5")

    original_time = AdaptiveReasoningV5.TIME_BUDGET
    AdaptiveReasoningV5.TIME_BUDGET = float(ceiling)

    env = arc.make(game_id)
    agent = AdaptiveReasoningV5(
        card_id=None,
        game_id=game_id,
        agent_name="adaptivev5",
        ROOT_URL="",
        record=False,
        arc_env=env,
        cross_game=cross_game if not args.freeze_transfer else None,
        arcade=arc,
        stagnation_seconds=float(stagnation) if stagnation > 0 else None,
        use_dissent=not args.no_dissent,
        use_ontology=not args.no_ontology,
        use_rituals=not args.no_rituals,
        use_goal_skeleton=not args.no_goal_skeleton,
        use_bissociation=not args.no_bissociation,
        use_learned_priors=learned_priors is not None,
        learned_priors=learned_priors,
    )

    start = time.time()
    error: Optional[str] = None
    try:
        agent.main()
    except Exception as exc:
        import traceback
        error = str(exc)
        print(f"\n  ERROR in {game_id}: {exc}")
        traceback.print_exc()

    elapsed = time.time() - start
    AdaptiveReasoningV5.TIME_BUDGET = original_time

    won = bool(agent.frames and agent.frames[-1].state == GameState.WIN)
    levels = int(agent.frames[-1].levels_completed) if agent.frames else 0
    summary = agent.end_game()
    wrapper = summary.get("wrapper", {})
    stop_reason = wrapper.get("stop_reason", "unknown")
    digest = summary.get("digest", {})
    progress = summary.get("progress", {})

    return {
        "game_id": game_id,
        "won": won,
        "levels": levels,
        "actions": agent.action_counter,
        "stop_reason": stop_reason,
        "elapsed": round(elapsed, 1),
        "lp": float(progress.get("lp", digest.get("lp", 0.0))),
        "sp": float(progress.get("sp", digest.get("sp", 0.0))),
        "tp": float(progress.get("tp", digest.get("tp", 0.0))),
        "pred_acc": float(summary.get("pred_accuracy", 0.0)),
        "ctrl_suc": float(summary.get("control_success", 0.0)),
        "knowledge": float(summary.get("knowledge_level", 0.0)),
        "operators": int(summary.get("operators", 0)),
        "rules": int(summary.get("rules", 0)),
        "rituals": int(summary.get("rituals", 0)),
        "ontology": summary.get("ontology", []),
        "dissent_interrupts": int(summary.get("dissent_interrupts", 0)),
        "bissociation_probes": int(summary.get("bissociation_probes", 0)),
        "ontology_flips_used": int(summary.get("ontology_flips_used", 0)),
        "anti_attractor_escapes": int(summary.get("anti_attractor_escapes", 0)),
        "danger_vetoes": int(summary.get("danger_vetoes", 0)),
        "noop_bans": int(summary.get("noop_bans", 0)),
        "danger_memory_size": int(summary.get("danger_memory_size", 0)),
        "prior_reorders": int(summary.get("prior_reorders", 0)),
        "prior_promotions": int(summary.get("prior_promotions", 0)),
        "digest": digest,
        "error": error,
    }


def format_digest(d: dict[str, Any]) -> str:
    prims = ",".join(d.get("useful_primitives", [])[:5]) or "-"
    kinds = ",".join(d.get("useful_operator_kinds", [])[:3]) or "-"
    return (
        f"prim=[{prims}] kinds=[{kinds}] sol={d.get('solution_length', 0)} "
        f"rit={len(d.get('useful_ritual_ids', []))}"
    )


def main() -> None:
    args = parse_args()

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    all_ids = [env.game_id for env in arc.get_environments()]
    if args.only:
        game_ids = [args.only] if args.only in all_ids else []
        if not game_ids:
            print(f"Game {args.only!r} not found.")
            sys.exit(1)
    else:
        game_ids = all_ids[: args.num_games]

    memory_path = Path(args.memory_path)
    cross_game = (
        CrossGameMemoryV5()
        if args.fresh
        else CrossGameMemoryV5.load(str(memory_path))
    )

    print()
    print("=" * WIDTH)
    print("  V5 PLAY-UNTIL-WIN LOOP")
    print("=" * WIDTH)
    print(f"  Games:           {len(game_ids)}")
    print(f"  Ceiling/game:    {args.ceiling}s")
    print(f"  Stagnation:      {args.stagnation}s")
    print(f"  Memory path:     {memory_path}")
    ablations = []
    if args.no_dissent: ablations.append("no-dissent")
    if args.no_ontology: ablations.append("no-ontology")
    if args.no_rituals: ablations.append("no-rituals")
    if args.no_goal_skeleton: ablations.append("no-goal")
    if args.no_bissociation: ablations.append("no-biss")
    if ablations:
        print(f"  Ablations:       {', '.join(ablations)}")
    if args.learned_priors:
        print(
            "  Learned priors:  "
            f"band={args.prior_band:.3f} break={args.prior_w_break:.3f} "
            f"progress={args.prior_w_progress:.3f}"
        )
    if args.fresh:
        print("  Starting FRESH cross-game memory.")
    elif cross_game.games_played > 0:
        print(
            f"  Resuming: {cross_game.games_played} played, "
            f"{cross_game.games_won} won, trust={cross_game.trust:.2f}, "
            f"digests={len(cross_game.game_digests)}, "
            f"partial={cross_game.partial_credit_games}"
        )
    print("=" * WIDTH)

    results: list[dict[str, Any]] = []
    overall_start = time.time()

    total_attempts_budget = 1 + max(0, args.retry_on_loss)
    worst_case_min = total_attempts_budget * (args.ceiling / 60)
    if total_attempts_budget > 1:
        print(
            f"  Retry policy:   up to {total_attempts_budget} attempts per game "
            f"(worst case ~{worst_case_min:.0f} min/game)"
        )
        print("=" * WIDTH)

    for i, game_id in enumerate(game_ids, 1):
        print(f"\n  [{i}/{len(game_ids)}] {game_id}", flush=True)
        attempt = 0
        result: dict[str, Any] = {}
        while True:
            attempt += 1
            attempt_label = (
                "attempt 1" if attempt == 1 else f"retry #{attempt - 1}"
            )
            print(f"    {attempt_label} ...", flush=True)
            attempt_start = time.time()
            result = run_single_game(
                arc, game_id,
                ceiling=args.ceiling,
                stagnation=args.stagnation,
                cross_game=cross_game,
                args=args,
            )

            # Per-attempt outcome (now printed immediately so you can
            # monitor progress instead of waiting for all retries)
            tag = "WIN" if result["won"] else result["stop_reason"]
            attempt_elapsed = time.time() - attempt_start
            print(
                f"      -> {tag:<12} L{result['levels']} "
                f"acts={result['actions']:>5}  "
                f"time={attempt_elapsed:>5.0f}s  "
                f"LP={result['lp']:.2f} SP={result['sp']:.2f} TP={result['tp']:.2f}  "
                f"diss={result['dissent_interrupts']} "
                f"biss={result['bissociation_probes']} "
                f"flip={result['ontology_flips_used']} "
                f"esc={result['anti_attractor_escapes']} "
                f"veto={result['danger_vetoes']} "
                f"walls={result['danger_memory_size']} "
                f"prior={result['prior_reorders']}/{result['prior_promotions']}",
                flush=True,
            )

            if cross_game is not None and not args.freeze_transfer:
                cross_game.save(str(memory_path))

            if result["won"] or attempt > args.retry_on_loss:
                break

        results.append(result)

        # Per-game summary (digest line)
        tag = "WIN" if result["won"] else result["stop_reason"]
        print(
            f"    final:        {tag:<12} L{result['levels']} "
            f"{format_digest(result['digest'])}",
            flush=True,
        )

    total_elapsed = time.time() - overall_start

    # ---------- FINAL SUMMARY ----------
    print()
    print("=" * WIDTH)
    print("  FINAL SUMMARY")
    print("=" * WIDTH)
    n = max(len(results), 1)
    won_n = sum(1 for r in results if r["won"])
    total_levels = sum(r["levels"] for r in results)
    total_actions = sum(r["actions"] for r in results)

    # Productive-minority metrics
    sp_pos = sum(1 for r in results if r["sp"] > 0.0)
    tp_pos = sum(1 for r in results if r["tp"] > 0.0)
    lp_high = sum(1 for r in results if r["lp"] > 0.5)
    partial = sum(
        1 for r in results
        if (not r["won"])
        and (r["sp"] >= 0.30 or r["tp"] >= 0.15 or r["lp"] >= 0.5)
    )

    print(f"  Games won:       {won_n}/{n}")
    print(f"  Total levels:    {total_levels}")
    print(f"  Total actions:   {total_actions:,}")
    print(f"  Wall-clock:      {total_elapsed:.1f}s "
          f"({total_elapsed / n:.1f}s avg/game)")
    print()
    print("  Productive-minority metrics:")
    print(f"    SP > 0:          {sp_pos}/{n}  ({100*sp_pos/n:.0f}%)")
    print(f"    TP > 0:          {tp_pos}/{n}  ({100*tp_pos/n:.0f}%)")
    print(f"    LP > 0.5:        {lp_high}/{n}  ({100*lp_high/n:.0f}%)")
    print(f"    Partial credit:  {partial}/{n}  ({100*partial/n:.0f}%)")
    print()
    print("  Aggregate totals:")
    total_interrupts = sum(r["dissent_interrupts"] for r in results)
    total_probes = sum(r["bissociation_probes"] for r in results)
    total_flips = sum(r["ontology_flips_used"] for r in results)
    total_prior_reorders = sum(r["prior_reorders"] for r in results)
    total_prior_promotions = sum(r["prior_promotions"] for r in results)
    print(f"    Dissent interrupts:  {total_interrupts}")
    print(f"    Bissociation probes: {total_probes}")
    print(f"    Ontology flips:      {total_flips}")
    print(f"    Prior reorders:      {total_prior_reorders}")
    print(f"    Prior promotions:    {total_prior_promotions}")

    print()
    print("  Per-game outcomes:")
    print(f"  {'Game':<14} {'Result':<12} {'Lv':>3} {'Acts':>6} {'LP':>5} {'SP':>5} {'TP':>5}  Digest")
    print("  " + "-" * (WIDTH - 4))
    for r in results:
        outcome = "WIN" if r["won"] else r["stop_reason"]
        print(
            f"  {r['game_id'][:13]:<13} "
            f"{outcome[:11]:<11} "
            f"{r['levels']:>3} {r['actions']:>6} "
            f"{r['lp']:>5.2f} {r['sp']:>5.2f} {r['tp']:>5.2f}  "
            f"{format_digest(r['digest'])}"
        )

    print()
    print("-" * WIDTH)
    print("  CROSS-GAME MEMORY (V5)")
    print("-" * WIDTH)
    print(f"  Trust:                {cross_game.trust:.2f}")
    print(f"  Games played (all):   {cross_game.games_played}")
    print(f"  Games won (all):      {cross_game.games_won}")
    print(f"  Game digests:         {len(cross_game.game_digests)}")
    print(f"  Partial-credit games: {cross_game.partial_credit_games}")
    print(f"  Ritual signatures:    {len(cross_game.ritual_signatures)}")
    print(f"  Ontology priors:      {dict(cross_game.ontology_priors)}")
    print(f"  Operator-kind priors: {dict(cross_game.operator_kind_priors)}")

    if not args.freeze_transfer:
        cross_game.save(str(memory_path))
        print(f"\n  Cross-game memory saved to {memory_path}")


if __name__ == "__main__":
    main()
