"""Post-evaluation multi-game protocol for natural structural breaks.

SAGE.9r runs the same online learner with and without SAGE.9q-u on fresh,
paired episodes.  It never perturbs a learned relation and never feeds one
game's outcomes into another controller.  Break candidates are classified
only after every episode has finished.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import game_splits

from .unified_cognition_ab_benchmark import (
    DEFAULT_HELD_OUT_GAMES,
    run_unified_cognition_ab_benchmark,
)


SCHEMA_VERSION = "sage.natural_structural_break.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage9r_natural_structural_break_benchmark.json"
)


def run_natural_structural_break_benchmark(
    *,
    game_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (0,),
    action_budget_per_reset: int = 80,
    resets: int = 4,
    environments_dir: str | Path | None = None,
    env_factory: Callable[[str], Any] | None = None,
    include_traces: bool = False,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Run a natural, non-interventional SAGE.9r active/ablation screen."""
    games = tuple(str(game) for game in (game_ids or DEFAULT_HELD_OUT_GAMES))
    common = {
        "game_ids": games,
        "seeds": tuple(int(seed) for seed in seeds),
        "action_budget_per_reset": int(action_budget_per_reset),
        "resets": int(resets),
        "environments_dir": environments_dir,
        "env_factory": env_factory,
        "include_traces": include_traces,
        "permute_terminal_relational_stencil_relation": False,
    }
    active = run_unified_cognition_ab_benchmark(
        **common,
        enable_online_structural_break_detection=True,
        enable_active_structural_hypothesis_arbitration=True,
        enable_structural_regime_abstraction=True,
        enable_hierarchical_structural_theory_composition=True,
    )
    ablated = run_unified_cognition_ab_benchmark(
        **common,
        enable_online_structural_break_detection=False,
        enable_active_structural_hypothesis_arbitration=False,
        enable_structural_regime_abstraction=False,
        enable_hierarchical_structural_theory_composition=False,
    )
    payload = summarize_natural_structural_break_protocol(
        active,
        ablated,
    )
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return payload


def summarize_natural_structural_break_protocol(
    active: Mapping[str, Any],
    ablated: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify natural breaks after paired evaluations have completed."""
    active_games = tuple(str(game) for game in active["held_out_games"])
    ablated_games = tuple(str(game) for game in ablated["held_out_games"])
    active_protocol = dict(active["paired_protocol"])
    ablated_protocol = dict(ablated["paired_protocol"])
    protocol_gate = bool(
        active_games == ablated_games
        and tuple(active["seeds"]) == tuple(ablated["seeds"])
        and active["action_budget_per_reset"]
        == ablated["action_budget_per_reset"]
        and active["resets_per_game_seed_arm"]
        == ablated["resets_per_game_seed_arm"]
        and active_protocol.get("protocol_gate_passed")
        and ablated_protocol.get("protocol_gate_passed")
        and not active_protocol.get(
            "terminal_relational_stencil_relation_permuted_in_unified"
        )
        and not ablated_protocol.get(
            "terminal_relational_stencil_relation_permuted_in_unified"
        )
        and active_protocol.get(
            "online_structural_break_detection_enabled_in_unified"
        )
        and not ablated_protocol.get(
            "online_structural_break_detection_enabled_in_unified"
        )
    )

    game_reports = []
    for game_id in active_games:
        active_metrics = _aggregate_game(active, game_id)
        ablated_metrics = _aggregate_game(ablated, game_id)
        natural_candidate = bool(
            active_metrics["structural_breaks_detected"] > 0
            and active_metrics[
                "structural_revision_hypotheses_generated"
            ]
            > 0
        )
        causal_frontier_advantage = bool(
            active_metrics["max_level_reached"]
            > ablated_metrics["max_level_reached"]
            or active_metrics["levels_completed"]
            > ablated_metrics["levels_completed"]
            or active_metrics["wins"] > ablated_metrics["wins"]
        )
        terminal_revision = bool(
            active_metrics["structural_revision_confirmations"] > 0
        )
        report = {
            "game_id": game_id,
            "active": active_metrics,
            "structural_revision_ablated": ablated_metrics,
            "natural_break_candidate": natural_candidate,
            "terminal_revision_observed": terminal_revision,
            "causal_frontier_advantage": causal_frontier_advantage,
            "natural_revision_gate_passed": bool(
                natural_candidate
                and terminal_revision
                and causal_frontier_advantage
            ),
            "strong_natural_revision_gate_passed": bool(
                natural_candidate
                and terminal_revision
                and causal_frontier_advantage
                and active_metrics["max_level_reached"] >= 5
            ),
        }
        game_reports.append(report)

    candidates = [
        report
        for report in game_reports
        if report["natural_break_candidate"]
    ]
    causal = [
        report
        for report in game_reports
        if report["natural_revision_gate_passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "post_hoc_natural_structural_break_screen",
        "held_out_games": list(active_games),
        "seeds": list(active["seeds"]),
        "action_budget_per_reset": active["action_budget_per_reset"],
        "resets_per_game_seed_arm": (
            active["resets_per_game_seed_arm"]
        ),
        "protocol": {
            "protocol_gate_passed": protocol_gate,
            "relation_permutation_used": False,
            "game_specific_rules_used": False,
            "cross_game_online_memory_used": False,
            "outcomes_reused_for_control": False,
            "classification_is_post_evaluation_only": True,
            "active_and_ablation_have_identical_exam_budget": True,
        },
        "natural_break_candidates": len(candidates),
        "causal_natural_revisions": len(causal),
        "any_natural_break_candidate": bool(candidates),
        "any_natural_revision_gate_passed": bool(causal),
        "games": game_reports,
        "active_benchmark": dict(active),
        "structural_revision_ablation_benchmark": dict(ablated),
    }


def _aggregate_game(
    payload: Mapping[str, Any],
    game_id: str,
) -> Dict[str, int]:
    rows = [
        dict(pair["unified"])
        for pair in payload["pairs"]
        if str(pair["game_id"]) == str(game_id)
    ]
    summed = (
        "levels_completed_delta",
        "wins",
        "controller_errors",
        "structural_breaks_detected",
        "structural_old_theory_suspensions",
        "structural_revision_hypotheses_generated",
        "structural_revision_actions",
        "structural_revision_confirmations",
        "structural_revision_refutations",
        "structural_arbitration_decisions",
        "structural_family_transfers",
        "structural_family_transfer_actions",
        "structural_theory_switches",
        "structural_theory_reactivations",
    )
    totals: Counter[str] = Counter()
    for row in rows:
        for key in summed:
            totals[key] += int(row.get(key, 0) or 0)
    return {
        "game_seed_runs": len(rows),
        "levels_completed": totals["levels_completed_delta"],
        "max_level_reached": max(
            (int(row.get("max_level_reached", 0) or 0) for row in rows),
            default=0,
        ),
        "wins": totals["wins"],
        "controller_errors": totals["controller_errors"],
        "structural_breaks_detected": totals[
            "structural_breaks_detected"
        ],
        "structural_old_theory_suspensions": totals[
            "structural_old_theory_suspensions"
        ],
        "structural_revision_hypotheses_generated": totals[
            "structural_revision_hypotheses_generated"
        ],
        "structural_revision_actions": totals[
            "structural_revision_actions"
        ],
        "structural_revision_confirmations": totals[
            "structural_revision_confirmations"
        ],
        "structural_revision_refutations": totals[
            "structural_revision_refutations"
        ],
        "structural_arbitration_decisions": totals[
            "structural_arbitration_decisions"
        ],
        "structural_family_transfers": totals[
            "structural_family_transfers"
        ],
        "structural_family_transfer_actions": totals[
            "structural_family_transfer_actions"
        ],
        "structural_theory_switches": totals[
            "structural_theory_switches"
        ],
        "structural_theory_reactivations": totals[
            "structural_theory_reactivations"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.9r natural structural-break evaluation.",
    )
    parser.add_argument(
        "--games",
        default=",".join(
            game_splits.resolve("public_unseen_split", full_ids=False)
        ),
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--resets", type=int, default=4)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--include-traces", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    games = [
        game_splits.resolve_full_game_id(item.strip())
        for item in str(args.games).split(",")
        if item.strip()
    ]
    seeds = [
        int(item.strip())
        for item in str(args.seeds).split(",")
        if item.strip()
    ]
    payload = run_natural_structural_break_benchmark(
        game_ids=games,
        seeds=seeds,
        action_budget_per_reset=args.budget,
        resets=args.resets,
        environments_dir=args.environments_dir,
        include_traces=args.include_traces,
        write_path=args.out,
    )
    print(
        json.dumps(
            {
                "natural_break_candidates": (
                    payload["natural_break_candidates"]
                ),
                "causal_natural_revisions": (
                    payload["causal_natural_revisions"]
                ),
                "protocol_gate_passed": payload["protocol"][
                    "protocol_gate_passed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["protocol"]["protocol_gate_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "run_natural_structural_break_benchmark",
    "summarize_natural_structural_break_protocol",
]
