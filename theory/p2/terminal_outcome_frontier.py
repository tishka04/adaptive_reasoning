"""P2 terminal-objective frontier classifier from long-budget policy runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.long_budget_saturation_probe import (
    DEFAULT_P2_LONG_BUDGET_SATURATION_PROBE_OUTPUT_PATH,
)
from theory.p2.policy_frontier_records import TRUTH_STATUS


DEFAULT_P2_TERMINAL_OUTCOME_FRONTIER_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "bp35_terminal_outcome_frontier.json"
)

OBJECTIVE_ALIGNMENT_FRONTIER = "OBJECTIVE_ALIGNMENT_FRONTIER"
LOCAL_PRODUCTIVE_TERMINAL_FAILED = (
    "LOCAL_AFFORDANCE_PRODUCTIVE_BUT_TERMINAL_OBJECTIVE_FAILED"
)
NO_TERMINAL_OBJECTIVE_FRONTIER = "NO_TERMINAL_OBJECTIVE_FRONTIER"


def run_terminal_outcome_frontier_classifier(
    *,
    long_budget_saturation_probe_path: str | Path = (
        DEFAULT_P2_LONG_BUDGET_SATURATION_PROBE_OUTPUT_PATH
    ),
    min_useful_action6_steps: int = 10,
) -> Dict[str, Any]:
    payload = _load_json(long_budget_saturation_probe_path)
    _validate_source_payload(payload)
    run_results = [
        dict(row)
        for row in payload.get("run_results", []) or []
        if isinstance(row, Mapping)
    ]
    evaluations = [
        classify_run_for_terminal_frontier(
            row,
            min_useful_action6_steps=min_useful_action6_steps,
        )
        for row in run_results
    ]
    candidates = [
        row for row in evaluations if row["terminal_objective_frontier_triggered"]
    ]
    frontier = build_terminal_objective_frontier(
        candidates,
        source_payload=payload,
        source_path=long_budget_saturation_probe_path,
        min_useful_action6_steps=min_useful_action6_steps,
    )
    summary = summarize_terminal_frontier_evaluations(evaluations, frontier)
    return {
        "config": {
            "schema_version": "p2.terminal_outcome_frontier.v1",
            "long_budget_saturation_probe_path": str(long_budget_saturation_probe_path),
            "min_useful_action6_steps": int(min_useful_action6_steps),
            "inputs_read": ["P2.3"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "saturation_handoff_not_performed": True,
        },
        "run_evaluations": evaluations,
        "terminal_outcome_frontier": frontier,
        "summary": summary,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "terminal_outcome_frontier_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
    }


def classify_run_for_terminal_frontier(
    run_result: Mapping[str, Any],
    *,
    min_useful_action6_steps: int,
) -> Dict[str, Any]:
    summary = dict(run_result.get("summary", {}) or {})
    classification = dict(run_result.get("classification", {}) or {})
    final_game_state = str(summary.get("final_game_state", ""))
    final_levels = int(summary.get("final_levels_completed", 0) or 0)
    useful_action6 = int(summary.get("useful_action6_steps", 0) or 0)
    movement_refresh_triggers = int(
        summary.get("conditional_movement_refresh_triggers", 0) or 0
    )
    true_saturation = bool(classification.get("true_frontier_triggered", False))
    productive_local_affordance = useful_action6 >= int(min_useful_action6_steps)
    terminal_objective_failure = final_game_state == "GAME_OVER" and final_levels == 0
    terminal_frontier = bool(
        terminal_objective_failure
        and productive_local_affordance
        and movement_refresh_triggers == 0
        and not true_saturation
    )
    return {
        "budget": int(run_result.get("budget", 0) or 0),
        "tie_break_seed": int(run_result.get("tie_break_seed", 0) or 0),
        "final_game_state": final_game_state,
        "final_levels_completed": final_levels,
        "useful_action6_steps": useful_action6,
        "action6_steps": int(summary.get("action6_steps", 0) or 0),
        "unique_action6_args_selected": int(
            summary.get("unique_action6_args_selected", 0) or 0
        ),
        "repeated_action6_args_selected": int(
            summary.get("repeated_action6_args_selected", 0) or 0
        ),
        "progress_proxy": float(summary.get("progress_proxy", 0.0) or 0.0),
        "movement_refresh_triggers": movement_refresh_triggers,
        "true_saturation_frontier": true_saturation,
        "productive_local_affordance": productive_local_affordance,
        "terminal_objective_failure": terminal_objective_failure,
        "terminal_objective_frontier_triggered": terminal_frontier,
        "frontier_type": (
            OBJECTIVE_ALIGNMENT_FRONTIER if terminal_frontier else ""
        ),
        "frontier_reason": (
            LOCAL_PRODUCTIVE_TERMINAL_FAILED
            if terminal_frontier
            else NO_TERMINAL_OBJECTIVE_FRONTIER
        ),
        "budget_exhausted_counted_as_saturation": False,
        "saturation_handoff_ready": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_terminal_objective_frontier(
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_payload: Mapping[str, Any],
    source_path: str | Path,
    min_useful_action6_steps: int,
) -> Dict[str, Any] | None:
    if not candidates:
        return None
    config = dict(source_payload.get("config", {}) or {})
    game_id = str(config.get("game_id", ""))
    budgets = sorted({int(row.get("budget", 0) or 0) for row in candidates})
    seeds = sorted({int(row.get("tie_break_seed", 0) or 0) for row in candidates})
    useful_counts = [int(row.get("useful_action6_steps", 0) or 0) for row in candidates]
    progress_values = [float(row.get("progress_proxy", 0.0) or 0.0) for row in candidates]
    return {
        "frontier_id": (
            "p2_terminal::bp35::conditional_movement_refresh::"
            "local_affordance_productive_but_terminal"
        ),
        "frontier_type": OBJECTIVE_ALIGNMENT_FRONTIER,
        "frontier_reason": LOCAL_PRODUCTIVE_TERMINAL_FAILED,
        "game_id": game_id,
        "source": "P2.3",
        "source_long_budget_saturation_probe_path": str(source_path),
        "source_saturation_handoff_ready": bool(
            source_payload.get("summary", {}).get("real_frontier_ready_for_p2_4", False)
        ),
        "ready_for_p2_4_saturation_handoff": False,
        "ready_for_objective_frontier_review": True,
        "ready_for_m2_or_m3": False,
        "terminal_runs": len(candidates),
        "terminal_budgets": budgets,
        "terminal_tie_break_seeds": seeds,
        "min_useful_action6_steps_threshold": int(min_useful_action6_steps),
        "max_useful_action6_steps": max(useful_counts or [0]),
        "min_useful_action6_steps": min(useful_counts or [0]),
        "max_progress_proxy": max(progress_values or [0.0]),
        "movement_refresh_triggers_total": sum(
            int(row.get("movement_refresh_triggers", 0) or 0) for row in candidates
        ),
        "true_saturation_frontier_runs": len(
            [row for row in candidates if bool(row.get("true_saturation_frontier"))]
        ),
        "interpretation": (
            "ACTION6 remains locally productive, but the rollout reaches GAME_OVER "
            "with no level completion and no saturation-refresh trigger."
        ),
        "proposed_question": (
            "When do locally useful ACTION6 effects stop serving the global objective?"
        ),
        "policy_result_counted_as_scientific_verdict": False,
        "objective_frontier_counted_as_confirmation": False,
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def summarize_terminal_frontier_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
    frontier: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    final_state_counts = Counter(
        str(row.get("final_game_state", "")) for row in evaluations
    )
    terminal_runs = [
        row for row in evaluations if bool(row.get("terminal_objective_failure"))
    ]
    objective_frontier_runs = [
        row
        for row in evaluations
        if bool(row.get("terminal_objective_frontier_triggered"))
    ]
    return {
        "runs_seen": len(evaluations),
        "terminal_objective_failure_runs": len(terminal_runs),
        "objective_alignment_frontier_runs": len(objective_frontier_runs),
        "final_game_state_counts": dict(sorted(final_state_counts.items())),
        "productive_local_affordance_terminal_runs": len(
            [
                row
                for row in terminal_runs
                if bool(row.get("productive_local_affordance"))
            ]
        ),
        "true_saturation_frontier_runs": len(
            [row for row in evaluations if bool(row.get("true_saturation_frontier"))]
        ),
        "movement_refresh_triggers_total": sum(
            int(row.get("movement_refresh_triggers", 0) or 0) for row in evaluations
        ),
        "ready_for_p2_4_saturation_handoff": False,
        "ready_for_objective_frontier_review": frontier is not None,
        "frontier_type": (
            str(frontier.get("frontier_type", "")) if frontier is not None else ""
        ),
        "frontier_reason": (
            str(frontier.get("frontier_reason", "")) if frontier is not None else ""
        ),
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(summary.get("revision_performed", False)):
        raise ValueError("source summary revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source summary wrong_confirmations must remain 0")
    if bool(payload.get("long_budget_probe_counted_as_confirmation", False)):
        raise ValueError("long-budget probe cannot be counted as confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be counted as scientific verdict")


def write_terminal_outcome_frontier(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P2_TERMINAL_OUTCOME_FRONTIER_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P2 terminal-objective frontier classifier.",
    )
    parser.add_argument(
        "--long-budget-saturation-probe",
        type=Path,
        default=DEFAULT_P2_LONG_BUDGET_SATURATION_PROBE_OUTPUT_PATH,
    )
    parser.add_argument("--min-useful-action6-steps", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_TERMINAL_OUTCOME_FRONTIER_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_terminal_outcome_frontier_classifier(
        long_budget_saturation_probe_path=args.long_budget_saturation_probe,
        min_useful_action6_steps=args.min_useful_action6_steps,
    )
    write_terminal_outcome_frontier(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
