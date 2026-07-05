"""M3.O3 candidate-only consolidation of objective threshold observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_stop_switch_experiment_executor import (
    BLOCKED_CONTROL_UNAVAILABLE,
    DEFAULT_OBJECTIVE_STOP_SWITCH_RESULTS_OUTPUT_PATH,
    EARLY_TERMINAL_DURING_PREFIX,
    OBJECTIVE_EXECUTED,
    OBJECTIVE_STOP_OBSERVED,
)


DEFAULT_OBJECTIVE_THRESHOLD_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "objective_threshold_consolidation.json"
)
PRE_TERMINAL_PREFIX_WINDOW = "PRE_TERMINAL_PREFIX_WINDOW"
NO_EARLY_TERMINAL_OBSERVED = "NO_EARLY_TERMINAL_OBSERVED"
NO_SAFE_PREFIX_OBSERVED = "NO_SAFE_PREFIX_OBSERVED"
STOP_SWITCH_NOT_ESTABLISHED = "NOT_ESTABLISHED"
REFINE_PREFIX_WINDOW = "REFINE_PREFIX_WINDOW"
EXTEND_PREFIX_SEARCH = "EXTEND_PREFIX_SEARCH"
REPLAN_FROM_SHORTER_PREFIX = "REPLAN_FROM_SHORTER_PREFIX"
DEFAULT_REFINED_CONDITIONS = (
    "continue_action6",
    "stop_policy",
    "switch_ACTION3",
    "switch_ACTION4",
)


def run_objective_threshold_consolidation(
    *,
    objective_results_path: str | Path = DEFAULT_OBJECTIVE_STOP_SWITCH_RESULTS_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(objective_results_path)
    validate_objective_results_source(payload)
    candidate = consolidate_objective_threshold(payload)
    return {
        "config": {
            "schema_version": "m3.objective_threshold_consolidation.v1",
            "objective_results_path": str(objective_results_path),
            "inputs_read": ["M3.O2"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "execution_performed": False,
            "consolidation_policy": "threshold_window_candidate_only",
        },
        "summary": summarize_threshold_candidate(payload, candidate),
        "objective_threshold_candidate": candidate,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "threshold_consolidation_counted_as_confirmation": False,
        "stop_switch_effectiveness_counted_as_verdict": False,
        "a32_remains_only_verdict_location": True,
    }


def consolidate_objective_threshold(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = outcome_rows(payload)
    safe_prefixes = safe_tested_prefixes(rows)
    early_prefixes = early_terminal_prefixes(rows)
    safe_max = max(safe_prefixes) if safe_prefixes else None
    early_min = min(early_prefixes) if early_prefixes else None
    threshold_type = threshold_type_for(safe_max=safe_max, early_min=early_min)
    recommended_prefixes = refined_prefixes_between(safe_max, early_min)
    condition_summary = condition_availability_summary(payload)
    recommended_conditions, excluded_conditions = recommended_conditions_for_refinement(
        condition_summary
    )
    critical_window = (
        [int(safe_max) + 1, int(early_min)]
        if safe_max is not None and early_min is not None and safe_max < early_min
        else []
    )
    return {
        "threshold_consolidation_id": threshold_consolidation_id(payload),
        "threshold_type": threshold_type,
        "game_id": first_game_id(payload),
        "prefix_policy": str((payload.get("config", {}) or {}).get("prefix_policy", "")),
        "prefix_action": "ACTION6",
        "safe_tested_prefixes": safe_prefixes,
        "safe_tested_prefix_max": safe_max,
        "early_terminal_prefixes": early_prefixes,
        "early_terminal_prefix_min": early_min,
        "critical_window": critical_window,
        "interpretation": interpretation_for_threshold(
            threshold_type=threshold_type,
            safe_max=safe_max,
            early_min=early_min,
        ),
        "stop_switch_effectiveness_status": STOP_SWITCH_NOT_ESTABLISHED,
        "stop_switch_effectiveness_rationale": (
            "No post-prefix stop/switch condition established level completion "
            "or terminal avoidance as a verdict; prefix 64 is early-terminal, "
            "so post-prefix intervention is too late there."
        ),
        "next_experiment_recommendation": recommendation_for_threshold(
            threshold_type
        ),
        "recommended_refined_prefixes": recommended_prefixes,
        "recommended_conditions": recommended_conditions,
        "excluded_conditions": excluded_conditions,
        "condition_availability_summary": condition_summary,
        "blocked_controls_not_contradictions": True,
        "early_terminal_prefix_cells_not_stop_switch_tests": True,
        "support_events_counted_as_proof": False,
        "contradiction_events_counted_as_refutation": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "threshold_consolidation_counted_as_confirmation": False,
    }


def outcome_rows(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for row in payload.get("objective_outcome_table", []) or []:
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return tuple(rows)


def safe_tested_prefixes(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    safe = []
    for row in rows:
        prefix = int(row.get("prefix_length", 0) or 0)
        conditions = dict(row.get("conditions", {}) or {})
        if not conditions:
            continue
        if any(
            bool((condition or {}).get("terminal_state_after_rollout", False))
            for condition in conditions.values()
            if isinstance(condition, Mapping)
        ):
            continue
        if any(
            str((condition or {}).get("status", ""))
            in {OBJECTIVE_EXECUTED, OBJECTIVE_STOP_OBSERVED}
            for condition in conditions.values()
            if isinstance(condition, Mapping)
        ):
            safe.append(prefix)
    return sorted(set(safe))


def early_terminal_prefixes(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    prefixes = []
    for row in rows:
        prefix = int(row.get("prefix_length", 0) or 0)
        conditions = dict(row.get("conditions", {}) or {})
        if any(
            str((condition or {}).get("status", "")) == EARLY_TERMINAL_DURING_PREFIX
            for condition in conditions.values()
            if isinstance(condition, Mapping)
        ):
            prefixes.append(prefix)
    return sorted(set(prefixes))


def threshold_type_for(*, safe_max: int | None, early_min: int | None) -> str:
    if safe_max is None:
        return NO_SAFE_PREFIX_OBSERVED
    if early_min is None:
        return NO_EARLY_TERMINAL_OBSERVED
    if safe_max < early_min:
        return PRE_TERMINAL_PREFIX_WINDOW
    return NO_SAFE_PREFIX_OBSERVED


def refined_prefixes_between(safe_max: int | None, early_min: int | None) -> list[int]:
    if safe_max is None or early_min is None or safe_max >= early_min:
        return []
    interior = list(range(int(safe_max) + 1, int(early_min)))
    if len(interior) <= 6:
        return interior
    candidates = [
        int(safe_max) + 2,
        int(safe_max) + 6,
        int(safe_max) + 10,
        int(early_min) - 4,
        int(early_min) - 2,
        int(early_min) - 1,
    ]
    return [value for value in ordered_unique_ints(candidates) if value in interior]


def condition_availability_summary(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for cell in payload.get("execution_cells", []) or []:
        if not isinstance(cell, Mapping):
            continue
        condition_id = str(cell.get("condition_id", ""))
        row = summary.setdefault(
            condition_id,
            {
                "condition_id": condition_id,
                "executed_or_observed_cells": 0,
                "blocked_control_unavailable_cells": 0,
                "early_terminal_cells": 0,
                "safe_nonterminal_cells": 0,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            },
        )
        status = str(cell.get("status", ""))
        if status in {OBJECTIVE_EXECUTED, OBJECTIVE_STOP_OBSERVED}:
            row["executed_or_observed_cells"] += 1
        if status == BLOCKED_CONTROL_UNAVAILABLE:
            row["blocked_control_unavailable_cells"] += 1
        if status == EARLY_TERMINAL_DURING_PREFIX:
            row["early_terminal_cells"] += 1
        objective = dict(cell.get("objective_metrics", {}) or {})
        if not bool(objective.get("terminal_state_after_rollout", False)):
            row["safe_nonterminal_cells"] += 1
    return {key: summary[key] for key in sorted(summary)}


def recommended_conditions_for_refinement(
    condition_summary: Mapping[str, Mapping[str, Any]],
) -> Tuple[list[str], list[Dict[str, Any]]]:
    recommended = []
    excluded = []
    for condition_id in DEFAULT_REFINED_CONDITIONS:
        row = dict(condition_summary.get(condition_id, {}) or {})
        if int(row.get("executed_or_observed_cells", 0) or 0) > 0:
            recommended.append(condition_id)
    for condition_id, row_raw in condition_summary.items():
        row = dict(row_raw)
        if condition_id in recommended:
            continue
        if int(row.get("blocked_control_unavailable_cells", 0) or 0) > 0:
            excluded.append(
                {
                    "condition_id": condition_id,
                    "reason": "blocked_control_unavailable_in_safe_prefixes",
                    "blocked_control_unavailable_cells": int(
                        row.get("blocked_control_unavailable_cells", 0) or 0
                    ),
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                }
            )
    return recommended, excluded


def interpretation_for_threshold(
    *,
    threshold_type: str,
    safe_max: int | None,
    early_min: int | None,
) -> str:
    if threshold_type == PRE_TERMINAL_PREFIX_WINDOW:
        return (
            f"ACTION6 remains post-prefix testable through prefix {safe_max}, "
            f"but prefix {early_min} reaches terminality before stop/switch can "
            "be evaluated."
        )
    if threshold_type == NO_EARLY_TERMINAL_OBSERVED:
        return (
            "No early-terminal prefix was observed in M3.O2; extend the prefix "
            "search before inducing a threshold."
        )
    return (
        "No safe tested prefix was observed before terminality; replan from "
        "shorter prefixes before testing stop/switch."
    )


def recommendation_for_threshold(threshold_type: str) -> str:
    if threshold_type == PRE_TERMINAL_PREFIX_WINDOW:
        return REFINE_PREFIX_WINDOW
    if threshold_type == NO_EARLY_TERMINAL_OBSERVED:
        return EXTEND_PREFIX_SEARCH
    return REPLAN_FROM_SHORTER_PREFIX


def summarize_threshold_candidate(
    payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    source = dict(payload.get("summary", {}) or {})
    return {
        "objective_threshold_consolidations": 1,
        "source_objective_requests_consumed": int(
            source.get("objective_requests_consumed", 0) or 0
        ),
        "source_planned_condition_cells": int(
            source.get("planned_condition_cells", 0) or 0
        ),
        "source_unique_execution_cells": int(
            source.get("unique_execution_cells", 0) or 0
        ),
        "source_objective_cells_executed": int(
            source.get("objective_cells_executed", 0) or 0
        ),
        "source_blocked_cells": int(source.get("blocked_cells", 0) or 0),
        "source_early_terminal_prefix_cells": int(
            source.get("early_terminal_prefix_cells", 0) or 0
        ),
        "threshold_type": str(candidate.get("threshold_type", "")),
        "safe_tested_prefix_max": candidate.get("safe_tested_prefix_max"),
        "early_terminal_prefix_min": candidate.get("early_terminal_prefix_min"),
        "critical_window": list(candidate.get("critical_window", []) or []),
        "stop_switch_effectiveness_status": str(
            candidate.get("stop_switch_effectiveness_status", "")
        ),
        "next_experiment_recommendation": str(
            candidate.get("next_experiment_recommendation", "")
        ),
        "recommended_refined_prefixes": list(
            candidate.get("recommended_refined_prefixes", []) or []
        ),
        "recommended_conditions": list(candidate.get("recommended_conditions", []) or []),
        "excluded_conditions": [
            dict(row) for row in candidate.get("excluded_conditions", []) or []
        ],
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "threshold_consolidation_counted_as_confirmation": False,
        "stop_switch_effectiveness_counted_as_verdict": False,
        "a32_remains_only_verdict_location": True,
    }


def validate_objective_results_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source objective results support must remain 0")
    if bool(summary.get("revision_performed", False)):
        raise ValueError("source objective results revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source objective results wrong_confirmations must remain 0")
    if bool(summary.get("duplicate_execution_cells_counted_as_independent", False)):
        raise ValueError("duplicate execution cells cannot count as independent")
    if bool(summary.get("blocked_cells_counted_as_contradictions", False)):
        raise ValueError("blocked cells cannot count as contradictions")
    if bool(summary.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")
    if int(payload.get("support", 0) or 0) != 0:
        raise ValueError("source objective results top-level support must remain 0")
    if bool(payload.get("objective_result_counted_as_confirmation", False)):
        raise ValueError("objective result cannot count as confirmation")


def first_game_id(payload: Mapping[str, Any]) -> str:
    for cell in payload.get("execution_cells", []) or []:
        if isinstance(cell, Mapping) and cell.get("game_id"):
            return str(cell.get("game_id", ""))
    return ""


def threshold_consolidation_id(payload: Mapping[str, Any]) -> str:
    game_id = first_game_id(payload) or "unknown_game"
    return f"m3_o3::{game_id}::objective_threshold"


def ordered_unique_ints(values: Sequence[int]) -> list[int]:
    result = []
    seen = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_objective_threshold_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OBJECTIVE_THRESHOLD_CONSOLIDATION_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate M3.O2 objective threshold observations.",
    )
    parser.add_argument(
        "--objective-results",
        type=Path,
        default=DEFAULT_OBJECTIVE_STOP_SWITCH_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_THRESHOLD_CONSOLIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_threshold_consolidation(
        objective_results_path=args.objective_results,
    )
    write_objective_threshold_consolidation(payload, args.out)
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
