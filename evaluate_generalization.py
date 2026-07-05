"""Phase 5: evaluate generalization of the learned-guided search.

Runs the search on a split (ar25 regression benchmark, public_seen, or the
held-out public_unseen_split) at one or more learned weights and reports, per
weight: level reached, actions-to-level-up, no-op rate, game-over rate, unique
states. It also surfaces offline model quality (Model A held-out MAE, Model C
macro accuracy) when a dataset / metrics are available.

ar25 is reported separately as a regression guard: w=0 is the current handmade
behavior; learned weights must not regress it.

Run with the bundled env interpreter:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe evaluate_generalization.py \\
        --games public_unseen_split --weights 0,0.5,1.0 \\
        --action-effect models\\action_effect.joblib --value models\\value.joblib
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from level7_frontier_recovery import (
    Arcade,
    OperationMode,
    ENV_DIR,
    PROJECT_ROOT,
    _resolve_full_game_id,
)

import game_splits
from learned_guided_search import run_game
from learned_scoring import LearnedScorer

DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "evaluate_generalization"


def _traj_summary(traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact summary of a step trajectory for quick inspection."""
    if not traj:
        return {}
    radar_steps = [s for s in traj if s.get("is_radar")]
    escape_steps = [s for s in traj if s.get("is_escape")]
    return {
        "steps": len(traj),
        "radar_injections": len(radar_steps),
        "escape_steps": len(escape_steps),
        "mean_break_prob": round(float(np.mean([s["break_probability"] for s in traj])), 5),
        "max_break_prob": round(float(np.max([s["break_probability"] for s in traj])), 5),
        "mean_break_prob_radar": (
            round(float(np.mean([s["break_probability"] for s in radar_steps])), 5)
            if radar_steps else None
        ),
        "mean_lc_size": round(float(np.mean([s["largest_component_size"] for s in traj])), 3),
        "mean_component_count": round(float(np.mean([s["component_count"] for s in traj])), 3),
        "mean_changed_cells": round(float(np.mean([s["changed_cells"] for s in traj])), 2),
        "action_counts": {a: sum(1 for s in traj if s["action"] == a) for a in sorted({s["action"] for s in traj})},
    }


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    levels = [r["level_reached"] for r in results]
    no_ops = [r["no_op_rate"] for r in results]
    unique = [r.get("unique_states", 0) for r in results]
    a2l = [r["actions_to_level_up"] for r in results if r["actions_to_level_up"] >= 0]
    return {
        "games": len(results),
        "mean_level_reached": round(float(np.mean(levels)), 4),
        "max_level_reached": int(np.max(levels)),
        "level_up_rate": round(float(np.mean([1.0 if l > 0 else 0.0 for l in levels])), 4),
        "game_over_rate": round(float(np.mean([1.0 if r["game_over"] else 0.0 for r in results])), 4),
        "mean_no_op_rate": round(float(np.mean(no_ops)), 4),
        "mean_actions_to_level_up": round(float(np.mean(a2l)), 4) if a2l else None,
        "win_rate": round(float(np.mean([1.0 if r["won"] else 0.0 for r in results])), 4),
        "mean_unique_states": round(float(np.mean(unique)), 2),
        "total_radar_steps": int(sum(r.get("radar_steps", 0) for r in results)),
        "total_radar_overrides": int(sum(r.get("radar_overrides", 0) for r in results)),
        "total_escape_steps": int(sum(r.get("escape_steps", 0) for r in results)),
    }


def _load_metrics(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def evaluate(
    *,
    games: List[str],
    weights: List[float],
    action_effect: str,
    value: str,
    max_steps: int,
    quiet: bool,
    with_break_expert: bool = False,
    break_classifier: Optional[str] = None,
    macro_scores: Optional[str] = None,
    break_bonus_weight: float = 2.0,
    break_radar_fraction: float = 0.0,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    full_ids = [_resolve_full_game_id(arc, g) for g in games]

    scorer: Optional[LearnedScorer] = None
    if any(w > 1e-9 for w in weights):
        scorer = LearnedScorer(
            action_effect,
            value,
            macro_scores_path=macro_scores if with_break_expert else None,
            break_classifier_path=break_classifier if with_break_expert else None,
            break_bonus_weight=break_bonus_weight,
        )
    radar_fraction = break_radar_fraction if with_break_expert else 0.0

    seed_list: List[Optional[int]] = list(seeds) if seeds else [None]

    # Danger memory is shared per game across seeds and weights: deaths observed
    # in one run teach the next run where the walls are.
    danger_memories: Dict[str, Dict[Any, float]] = {}

    per_weight: Dict[str, Any] = {}
    for weight in weights:
        results = []
        for full_id in full_ids:
            for seed in seed_list:
                rng = random.Random(seed) if seed is not None else None
                res = run_game(
                    arc, full_id,
                    learned_weight=weight, scorer=scorer, max_steps=max_steps,
                    break_radar_fraction=radar_fraction,
                    rng=rng,
                    danger_memory=danger_memories.setdefault(full_id, {}),
                )
                res_dict = res.__dict__.copy()
                res_dict["seed"] = seed
                # Convert StepRecord objects to plain dicts for JSON serialisation.
                res_dict["trajectory"] = [
                    s.__dict__ if hasattr(s, "__dict__") else s
                    for s in res_dict.get("trajectory", [])
                ]
                res_dict["trajectory_summary"] = _traj_summary(res_dict["trajectory"])
                results.append(res_dict)
                if not quiet:
                    ts = res_dict["trajectory_summary"]
                    seed_tag = f" seed={seed}" if seed is not None else ""
                    print(
                        f"[w={weight}]{seed_tag} {full_id} level={res.level_reached} "
                        f"no_op={res.no_op_rate} died={res.died} a2l={res.actions_to_level_up} "
                        f"unique={res.unique_states} radar={res.radar_overrides} "
                        f"escape={res.escape_steps} walls={res.danger_memory_size} "
                        f"max_bp={ts.get('max_break_prob')} mean_lc={ts.get('mean_lc_size')}"
                    )
        per_weight[str(weight)] = {"aggregate": _aggregate(results), "per_game": results}
        if not quiet:
            print(f"  => w={weight} aggregate: {per_weight[str(weight)]['aggregate']}\n")

    summary = {
        "games": full_ids,
        "weights": weights,
        "seeds": [s for s in seed_list if s is not None],
        "max_steps": max_steps,
        "with_break_expert": bool(with_break_expert),
        "break_radar_fraction": float(radar_fraction),
        "per_weight": per_weight,
        "model_a_metrics": _load_metrics(Path(action_effect).with_suffix(".metrics.json")),
        "model_c_metrics": _load_metrics(str(Path(value).parent / "macro.metrics.json")),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generalization across splits and weights.")
    parser.add_argument("--games", default="public_unseen_split")
    parser.add_argument("--weights", default="0,0.5,1.0", help="Comma list of learned weights.")
    parser.add_argument("--action-effect", default="models/action_effect.joblib")
    parser.add_argument("--value", default="models/value.joblib")
    parser.add_argument(
        "--with-break-expert",
        action="store_true",
        help="Load the rare-event break expert (+ macro scores) and enable the break-radar head.",
    )
    parser.add_argument("--break-classifier", default="models/break_classifier.joblib")
    parser.add_argument("--macro-scores", default=None)
    parser.add_argument("--break-bonus-weight", type=float, default=2.0)
    parser.add_argument(
        "--break-radar-fraction",
        type=float,
        default=0.2,
        help="Fraction of steps where the guarded break-radar overrides selection.",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma list of RNG seeds (e.g. 0,1,2,3,4); each game is run once per seed.",
    )
    parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR / "report.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
    games = game_splits.resolve(args.games, full_ids=True)
    summary = evaluate(
        games=games,
        weights=weights,
        action_effect=args.action_effect,
        value=args.value,
        max_steps=args.max_steps,
        quiet=args.quiet,
        with_break_expert=args.with_break_expert,
        break_classifier=args.break_classifier,
        macro_scores=args.macro_scores,
        break_bonus_weight=args.break_bonus_weight,
        break_radar_fraction=args.break_radar_fraction,
        seeds=[int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.quiet:
        print("\n=== Comparison table (mean level / level-up rate / no-op / game-over) ===")
        for weight, payload in summary["per_weight"].items():
            agg = payload["aggregate"]
            print(
                f"  w={weight:>4}: level={agg.get('mean_level_reached')} "
                f"lvlup={agg.get('level_up_rate')} no_op={agg.get('mean_no_op_rate')} "
                f"game_over={agg.get('game_over_rate')} "
                f"unique={agg.get('mean_unique_states')} "
                f"radar_overrides={agg.get('total_radar_overrides')} "
                f"escape_steps={agg.get('total_escape_steps')}"
            )
        print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
