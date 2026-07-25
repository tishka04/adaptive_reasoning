"""Procedural causal validation for SAGE.9s structural arbitration.

Every episode presents the same three live theories and interventions to both
arms.  The hidden theory is never passed to either selector.  The active arm
uses the production disagreement arbiter; the ablation optimizes the first
generated theory only.  One action is allowed, so a changed terminal outcome
is attributable to experiment order rather than extra budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .structural_hypothesis_arbitration import (
    StructuralArbitrationChoice,
    StructuralExperimentOption,
    select_discriminating_structural_experiment,
    sequential_structural_experiment,
    surviving_hypotheses,
)


SCHEMA_VERSION = "sage.structural_arbitration_causal.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage9x_structural_arbitration_benchmark.json"
)


@dataclass(frozen=True)
class ProceduralArbitrationEpisode:
    seed: int
    hypothesis_priority: Tuple[str, ...]
    true_hypothesis_id: str
    terminal_action_key: str
    options: Tuple[StructuralExperimentOption, ...]
    action_budget: int = 1


def generate_procedural_arbitration_episode(
    seed: int,
) -> ProceduralArbitrationEpisode:
    """Create an order-sensitive theory contest without exposing its answer."""
    rng = random.Random(int(seed))
    hypothesis_ids = [
        _opaque_id("hypothesis", seed, index)
        for index in range(3)
    ]
    action_keys = [
        _opaque_id("intervention", seed, index)
        for index in range(4)
    ]
    rng.shuffle(action_keys)
    scale = 1 + (abs(int(seed)) % 3)

    # h0 is the historical first candidate.  The hidden regime alternates
    # between the other two candidates.  Its intervention yields three
    # distinct predictions, whereas the h0 test leaves both alternatives
    # observationally aliased.  Names and action positions are seed-permuted.
    true_index = 1 + (abs(int(seed)) % 2)
    other_index = 1 if true_index == 2 else 2
    historical_row = [-scale, -scale, -scale]
    historical_row[0] = scale
    terminal_row = [-scale, -scale, -scale]
    terminal_row[0] = 0
    terminal_row[true_index] = 2 * scale
    weak_row = [0, 0, 0]
    weak_row[other_index] = scale
    shared_row = [0, 0, 0]
    shared_row[true_index] = scale
    shared_row[other_index] = scale
    rows = (
        tuple(historical_row),
        tuple(terminal_row),
        tuple(weak_row),
        tuple(shared_row),
    )
    options = []
    for index, action_key in enumerate(action_keys):
        options.append(
            StructuralExperimentOption(
                action_key=action_key,
                predicted_reductions=tuple(
                    (hypothesis_ids[hypothesis_index], reduction)
                    for hypothesis_index, reduction in enumerate(
                        rows[index]
                    )
                ),
                hypothesis_support=tuple(
                    (hypothesis_id, 1 + ((seed + index) % 2))
                    for hypothesis_id in hypothesis_ids
                ),
                tie_break=-index,
            )
        )
    return ProceduralArbitrationEpisode(
        seed=int(seed),
        hypothesis_priority=tuple(hypothesis_ids),
        true_hypothesis_id=hypothesis_ids[true_index],
        terminal_action_key=action_keys[1],
        options=tuple(options),
    )


def evaluate_procedural_arbitration_episode(
    episode: ProceduralArbitrationEpisode,
) -> Dict[str, Any]:
    active = select_discriminating_structural_experiment(
        options=episode.options,
        hypothesis_priority=episode.hypothesis_priority,
    )
    sequential = sequential_structural_experiment(
        options=episode.options,
        hypothesis_priority=episode.hypothesis_priority,
    )
    true_first_priority = (
        episode.true_hypothesis_id,
        *tuple(
            hypothesis_id
            for hypothesis_id in episode.hypothesis_priority
            if hypothesis_id != episode.true_hypothesis_id
        ),
    )
    true_first = sequential_structural_experiment(
        options=episode.options,
        hypothesis_priority=true_first_priority,
    )
    reversed_active = select_discriminating_structural_experiment(
        options=episode.options,
        hypothesis_priority=tuple(
            reversed(episode.hypothesis_priority)
        ),
    )
    plausible = sum(
        any(
            dict(option.predicted_reductions).get(
                hypothesis_id,
                0,
            )
            > 0
            for option in episode.options
        )
        for hypothesis_id in episode.hypothesis_priority
    )
    active_result = _choice_outcome(episode, active)
    sequential_result = _choice_outcome(episode, sequential)
    true_first_result = _choice_outcome(episode, true_first)
    return {
        "seed": episode.seed,
        "action_budget": episode.action_budget,
        "hypotheses": len(episode.hypothesis_priority),
        "plausible_hypotheses": plausible,
        "true_hypothesis_priority_index": (
            episode.hypothesis_priority.index(
                episode.true_hypothesis_id
            )
        ),
        "active": active_result,
        "sequential_ablation": sequential_result,
        "true_first_order_control": true_first_result,
        "active_priority_permutation_invariant": bool(
            active is not None
            and reversed_active is not None
            and active.action_key == reversed_active.action_key
        ),
        "active_and_sequential_actions_differ": bool(
            active is not None
            and sequential is not None
            and active.action_key != sequential.action_key
        ),
        "sequential_order_changes_terminal_outcome": bool(
            sequential_result["terminal_success"]
            != true_first_result["terminal_success"]
        ),
    }


def run_structural_arbitration_benchmark(
    *,
    seeds: Sequence[int] = tuple(range(64)),
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    runs = [
        evaluate_procedural_arbitration_episode(
            generate_procedural_arbitration_episode(int(seed))
        )
        for seed in seeds
    ]
    payload = summarize_structural_arbitration_runs(runs)
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def summarize_structural_arbitration_runs(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    reports = [dict(run) for run in runs]
    episodes = len(reports)
    active_terminal = sum(
        bool(dict(report["active"]).get("terminal_success"))
        for report in reports
    )
    sequential_terminal = sum(
        bool(
            dict(report["sequential_ablation"]).get(
                "terminal_success"
            )
        )
        for report in reports
    )
    active_identifications = sum(
        int(dict(report["active"]).get("surviving_hypotheses", 0))
        == 1
        for report in reports
    )
    sequential_identifications = sum(
        int(
            dict(report["sequential_ablation"]).get(
                "surviving_hypotheses",
                0,
            )
        )
        == 1
        for report in reports
    )
    protocol_gate = bool(
        episodes > 0
        and all(int(report.get("action_budget", 0)) == 1 for report in reports)
        and all(
            dict(report.get("active", {})).get("action_selected")
            and dict(report.get("sequential_ablation", {})).get(
                "action_selected"
            )
            for report in reports
        )
    )
    ambiguity_gate = bool(
        reports
        and all(
            int(report.get("plausible_hypotheses", 0)) >= 3
            for report in reports
        )
    )
    order_sensitivity_gate = bool(
        reports
        and all(
            report.get("active_and_sequential_actions_differ")
            and report.get("sequential_order_changes_terminal_outcome")
            for report in reports
        )
    )
    priority_permutation_gate = bool(
        reports
        and all(
            report.get("active_priority_permutation_invariant")
            for report in reports
        )
    )
    causal_gate = bool(
        protocol_gate
        and ambiguity_gate
        and order_sensitivity_gate
        and priority_permutation_gate
        and active_terminal > sequential_terminal
        and active_identifications > sequential_identifications
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": (
            "paired_procedural_causal_structural_hypothesis_arbitration"
        ),
        "episodes": episodes,
        "action_budget_per_arm": 1,
        "protocol": {
            "protocol_gate_passed": protocol_gate,
            "same_episode_and_prediction_matrix_per_arm": True,
            "same_action_budget_per_arm": True,
            "hidden_true_hypothesis_not_exposed_to_selectors": True,
            "production_sage9s_arbiter_used": True,
            "sequential_ablation_uses_first_generated_hypothesis": True,
            "classification_is_post_evaluation_only": True,
            "arc_level_or_win_claimed": False,
        },
        "ambiguous_episode_gate_passed": ambiguity_gate,
        "order_sensitivity_gate_passed": order_sensitivity_gate,
        "priority_permutation_gate_passed": priority_permutation_gate,
        "active_terminal_successes": active_terminal,
        "sequential_terminal_successes": sequential_terminal,
        "terminal_success_advantage": (
            active_terminal - sequential_terminal
        ),
        "active_single_theory_identifications": active_identifications,
        "sequential_single_theory_identifications": (
            sequential_identifications
        ),
        "causal_arbitration_gate_passed": causal_gate,
        "runs": reports,
    }


def _choice_outcome(
    episode: ProceduralArbitrationEpisode,
    choice: StructuralArbitrationChoice | None,
) -> Dict[str, Any]:
    if choice is None:
        return {
            "action_selected": False,
            "action_key": "",
            "sponsor_hypothesis_id": "",
            "observed_reduction": 0,
            "surviving_hypotheses": 0,
            "terminal_success": False,
            "disagreement_score": 0.0,
        }
    observed = int(
        dict(choice.predicted_reductions)[
            episode.true_hypothesis_id
        ]
    )
    survivors = surviving_hypotheses(
        choice=choice,
        observed_reduction=observed,
    )
    return {
        "action_selected": True,
        "action_key": choice.action_key,
        "sponsor_hypothesis_id": choice.hypothesis_id,
        "observed_reduction": observed,
        "surviving_hypotheses": len(survivors),
        "survivor_ids": list(survivors),
        "terminal_success": bool(
            choice.action_key == episode.terminal_action_key
        ),
        "disagreement_score": choice.disagreement_score,
        "distinct_predictions": choice.distinct_predictions,
        "polarity_pairs": choice.polarity_pairs,
    }


def _opaque_id(kind: str, seed: int, index: int) -> str:
    return (
        f"{kind}-"
        + hashlib.sha1(
            f"{kind}:{int(seed)}:{int(index)}".encode("utf-8")
        ).hexdigest()[:10]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.9x procedural arbitration benchmark.",
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(list(argv) if argv is not None else None)
    count = max(1, int(args.episodes))
    offset = int(args.seed_offset)
    payload = run_structural_arbitration_benchmark(
        seeds=tuple(range(offset, offset + count)),
        write_path=args.out,
    )
    print(json.dumps({
        "episodes": payload["episodes"],
        "active_terminal_successes": (
            payload["active_terminal_successes"]
        ),
        "sequential_terminal_successes": (
            payload["sequential_terminal_successes"]
        ),
        "causal_arbitration_gate_passed": (
            payload["causal_arbitration_gate_passed"]
        ),
    }, indent=2, sort_keys=True))
    return 0 if payload["causal_arbitration_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProceduralArbitrationEpisode",
    "evaluate_procedural_arbitration_episode",
    "generate_procedural_arbitration_episode",
    "run_structural_arbitration_benchmark",
    "summarize_structural_arbitration_runs",
]
