"""M3.O4 executor for the refined objective pre-terminal prefix window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir
from theory.p1.bp35_sage_candidate_policy_probe import candidate_policy_memory_from_scope

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_stop_switch_experiment_executor import (
    BLOCKED_CONTROL_UNAVAILABLE,
    DEFAULT_OBJECTIVE_STOP_SWITCH_RESULTS_OUTPUT_PATH,
    EARLY_TERMINAL_DURING_PREFIX,
    ObjectiveExecutionCell,
    execute_objective_cell,
    execution_cell_signature,
    objective_outcome_table,
)
from .objective_threshold_consolidation import (
    DEFAULT_OBJECTIVE_THRESHOLD_CONSOLIDATION_OUTPUT_PATH,
    PRE_TERMINAL_PREFIX_WINDOW,
    REFINE_PREFIX_WINDOW,
)


DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "objective_refined_window_results.json"
)
REFINED_WINDOW_SCHEMA_VERSION = "m3.objective_refined_window_results.v1"
PREFIX_POLICY = "patch_similarity_soft_stale_action6_prefix"
TERMINAL_AVOIDANCE_CANDIDATE = "CANDIDATE_TERMINAL_AVOIDANCE_OBSERVED"
NO_TERMINAL_SEPARATION = "NO_TERMINAL_SEPARATION_IN_REFINED_WINDOW"
ALL_CONDITIONS_TERMINAL = "ALL_CONDITIONS_TERMINAL_OR_TOO_LATE"
MIXED_OBJECTIVE_OUTCOME = "MIXED_OBJECTIVE_OUTCOME_CANDIDATE_ONLY"


def run_objective_refined_window_execution(
    *,
    threshold_consolidation_path: str | Path = (
        DEFAULT_OBJECTIVE_THRESHOLD_CONSOLIDATION_OUTPUT_PATH
    ),
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    tie_break_seed: int = 0,
    max_cells: int | None = None,
    cell_executor: Callable[[ObjectiveExecutionCell], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Execute the M3.O3 recommended refined prefix window candidate-only."""
    threshold_payload = _load_json(threshold_consolidation_path)
    candidate = validate_threshold_source(threshold_payload)
    cells = build_refined_window_cells(
        candidate,
        tie_break_seed=tie_break_seed,
    )
    if max_cells is not None:
        cells = cells[: max(0, int(max_cells))]

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    if cell_executor is None:
        _configure_offline_env(env_dir)
        memory = candidate_policy_memory_from_scope(_load_json(scope_consolidation_path))

        def default_executor(cell: ObjectiveExecutionCell) -> Mapping[str, Any]:
            return execute_objective_cell(
                cell,
                memory=memory,
                environments_dir=env_dir,
            )

        cell_executor = default_executor

    cell_results = [dict(cell_executor(cell)) for cell in cells]
    table = objective_outcome_table(cell_results)
    comparisons = compare_refined_prefix_outcomes(table)
    summary = summarize_refined_window(
        candidate=candidate,
        cells=cells,
        cell_results=cell_results,
        prefix_comparisons=comparisons,
    )
    return {
        "config": {
            "schema_version": REFINED_WINDOW_SCHEMA_VERSION,
            "threshold_consolidation_path": str(threshold_consolidation_path),
            "scope_consolidation_path": str(scope_consolidation_path),
            "environments_dir": str(env_dir),
            "inputs_read": ["M3.O3", "M3.24"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "execution_performed": True,
            "prefix_policy": PREFIX_POLICY,
            "conditions_source": "M3.O3.recommended_conditions",
            "prefixes_source": "M3.O3.recommended_refined_prefixes",
        },
        "summary": summary,
        "source_threshold_candidate": {
            "threshold_consolidation_id": str(
                candidate.get("threshold_consolidation_id", "")
            ),
            "threshold_type": str(candidate.get("threshold_type", "")),
            "critical_window": list(candidate.get("critical_window", []) or []),
            "safe_tested_prefix_max": candidate.get("safe_tested_prefix_max"),
            "early_terminal_prefix_min": candidate.get("early_terminal_prefix_min"),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        },
        "execution_cells": cell_results,
        "objective_outcome_table": table,
        "prefix_comparisons": comparisons,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "refined_window_result_counted_as_confirmation": False,
        "stop_switch_effectiveness_counted_as_verdict": False,
        "a32_remains_only_verdict_location": True,
    }


def build_refined_window_cells(
    candidate: Mapping[str, Any],
    *,
    tie_break_seed: int = 0,
) -> Tuple[ObjectiveExecutionCell, ...]:
    game_id = str(candidate.get("game_id", ""))
    prefix_action = str(candidate.get("prefix_action", "ACTION6") or "ACTION6")
    prefixes = [int(value) for value in candidate.get("recommended_refined_prefixes", [])]
    conditions = [
        condition_for_id(str(condition_id))
        for condition_id in candidate.get("recommended_conditions", []) or []
    ]
    cells = [
        make_refined_cell(
            game_id=game_id,
            prefix_action=prefix_action,
            prefix_length=prefix,
            condition=condition,
            tie_break_seed=tie_break_seed,
        )
        for prefix in prefixes
        for condition in conditions
    ]
    return tuple(cells)


def condition_for_id(condition_id: str) -> Dict[str, Any]:
    if condition_id == "continue_action6":
        return {
            "condition_id": "continue_action6",
            "condition_family": "continue_local_affordance",
            "post_prefix_policy": "continue_action",
            "post_prefix_action": "ACTION6",
        }
    if condition_id == "stop_policy":
        return {
            "condition_id": "stop_policy",
            "condition_family": "stop_or_noop",
            "post_prefix_policy": "stop_or_hold_if_available",
            "post_prefix_action": None,
        }
    if condition_id.startswith("switch_"):
        action = condition_id.replace("switch_", "", 1)
        return {
            "condition_id": condition_id,
            "condition_family": "switch_subgoal",
            "post_prefix_policy": "switch_to_action",
            "post_prefix_action": action,
        }
    raise ValueError(f"unsupported refined condition:{condition_id}")


def make_refined_cell(
    *,
    game_id: str,
    prefix_action: str,
    prefix_length: int,
    condition: Mapping[str, Any],
    tie_break_seed: int,
) -> ObjectiveExecutionCell:
    condition_id = str(condition.get("condition_id", ""))
    post_prefix_policy = str(condition.get("post_prefix_policy", ""))
    post_prefix_action_raw = condition.get("post_prefix_action")
    post_prefix_action = (
        None if post_prefix_action_raw is None else str(post_prefix_action_raw)
    )
    return ObjectiveExecutionCell(
        cell_signature=execution_cell_signature(
            game_id=game_id,
            prefix_policy=PREFIX_POLICY,
            prefix_action=prefix_action,
            prefix_length=int(prefix_length),
            condition_id=condition_id,
            post_prefix_policy=post_prefix_policy,
            post_prefix_action=post_prefix_action,
            tie_break_seed=tie_break_seed,
        ),
        game_id=game_id,
        prefix_policy=PREFIX_POLICY,
        prefix_action=prefix_action,
        prefix_length=int(prefix_length),
        condition_id=condition_id,
        condition_family=str(condition.get("condition_family", "")),
        post_prefix_policy=post_prefix_policy,
        post_prefix_action=post_prefix_action,
        tie_break_seed=int(tie_break_seed),
    )


def compare_refined_prefix_outcomes(
    outcome_table: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows = []
    for prefix_row in outcome_table:
        prefix = int(prefix_row.get("prefix_length", 0) or 0)
        conditions = dict(prefix_row.get("conditions", {}) or {})
        continue_row = dict(conditions.get("continue_action6", {}) or {})
        alternatives = {
            key: dict(value)
            for key, value in conditions.items()
            if key != "continue_action6" and isinstance(value, Mapping)
        }
        continue_terminal = bool(
            continue_row.get("terminal_state_after_rollout", False)
        )
        nonterminal_alternatives = [
            key
            for key, value in alternatives.items()
            if not bool(value.get("terminal_state_after_rollout", False))
            and str(value.get("status", "")) != BLOCKED_CONTROL_UNAVAILABLE
        ]
        terminal_alternatives = [
            key
            for key, value in alternatives.items()
            if bool(value.get("terminal_state_after_rollout", False))
        ]
        if continue_terminal and nonterminal_alternatives:
            interpretation = "stop_or_switch_avoids_terminal_candidate_only"
        elif continue_terminal and not nonterminal_alternatives:
            interpretation = "all_available_conditions_terminal_or_too_late"
        elif not continue_terminal and not terminal_alternatives:
            interpretation = "pre_terminal_no_terminal_separation"
        else:
            interpretation = "mixed_objective_outcome_candidate_only"
        rows.append(
            {
                "prefix_length": prefix,
                "continue_terminal": continue_terminal,
                "nonterminal_alternative_conditions": nonterminal_alternatives,
                "terminal_alternative_conditions": terminal_alternatives,
                "interpretation": interpretation,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            }
        )
    return rows


def summarize_refined_window(
    *,
    candidate: Mapping[str, Any],
    cells: Sequence[ObjectiveExecutionCell],
    cell_results: Sequence[Mapping[str, Any]],
    prefix_comparisons: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    executed = [row for row in cell_results if bool(row.get("execution_performed"))]
    blocked = [row for row in cell_results if not bool(row.get("execution_performed"))]
    early = [
        row for row in cell_results if bool(row.get("early_terminal_during_prefix"))
    ]
    avoidance_prefixes = [
        int(row.get("prefix_length", 0) or 0)
        for row in prefix_comparisons
        if str(row.get("interpretation", ""))
        == "stop_or_switch_avoids_terminal_candidate_only"
    ]
    all_terminal_prefixes = [
        int(row.get("prefix_length", 0) or 0)
        for row in prefix_comparisons
        if str(row.get("interpretation", ""))
        == "all_available_conditions_terminal_or_too_late"
    ]
    no_separation_prefixes = [
        int(row.get("prefix_length", 0) or 0)
        for row in prefix_comparisons
        if str(row.get("interpretation", "")) == "pre_terminal_no_terminal_separation"
    ]
    status = refined_stop_switch_status(
        avoidance_prefixes=avoidance_prefixes,
        all_terminal_prefixes=all_terminal_prefixes,
    )
    return {
        "threshold_consolidations_consumed": 1,
        "source_threshold_type": str(candidate.get("threshold_type", "")),
        "source_critical_window": list(candidate.get("critical_window", []) or []),
        "refined_prefixes": [int(value) for value in candidate.get("recommended_refined_prefixes", []) or []],
        "refined_conditions": list(candidate.get("recommended_conditions", []) or []),
        "planned_refined_cells": len(cells),
        "unique_execution_cells": len({cell.cell_signature for cell in cells}),
        "execution_performed": True,
        "objective_cells_executed": len(executed),
        "blocked_cells": len(blocked),
        "blocked_control_unavailable_cells": len(
            [row for row in blocked if row.get("status") == BLOCKED_CONTROL_UNAVAILABLE]
        ),
        "early_terminal_prefix_cells": len(early),
        "neutral_events": sum(int(row.get("neutral_events", 0) or 0) for row in cell_results),
        "support_events": 0,
        "contradiction_events": 0,
        "stop_switch_effectiveness_status": status,
        "prefixes_with_candidate_terminal_avoidance": avoidance_prefixes,
        "prefixes_all_conditions_terminal_or_too_late": all_terminal_prefixes,
        "prefixes_without_terminal_separation": no_separation_prefixes,
        "result_interpretation": interpretation_for_refined_status(status),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "refined_window_result_counted_as_confirmation": False,
        "stop_switch_effectiveness_counted_as_verdict": False,
        "blocked_cells_counted_as_contradictions": False,
        "a32_remains_only_verdict_location": True,
    }


def refined_stop_switch_status(
    *,
    avoidance_prefixes: Sequence[int],
    all_terminal_prefixes: Sequence[int],
) -> str:
    if avoidance_prefixes:
        return TERMINAL_AVOIDANCE_CANDIDATE
    if all_terminal_prefixes:
        return ALL_CONDITIONS_TERMINAL
    return NO_TERMINAL_SEPARATION


def interpretation_for_refined_status(status: str) -> str:
    if status == TERMINAL_AVOIDANCE_CANDIDATE:
        return (
            "At least one refined prefix has terminal continue-ACTION6 while a "
            "stop/switch alternative remains non-terminal; candidate-only."
        )
    if status == ALL_CONDITIONS_TERMINAL:
        return (
            "Refined prefixes reached terminality before useful stop/switch "
            "separation; switch timing or subgoal remains unresolved."
        )
    return (
        "Refined prefixes did not separate terminal outcomes; either the window "
        "is still too early or the objective signal changes abruptly near 64."
    )


def validate_threshold_source(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    candidate = dict(payload.get("objective_threshold_candidate", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("threshold source support must remain 0")
    if bool(summary.get("execution_performed", False)):
        raise ValueError("threshold source must be consolidation-only")
    if bool(summary.get("threshold_consolidation_counted_as_confirmation", False)):
        raise ValueError("threshold consolidation cannot count as confirmation")
    if bool(summary.get("stop_switch_effectiveness_counted_as_verdict", False)):
        raise ValueError("stop/switch effectiveness cannot be a verdict")
    if str(candidate.get("threshold_type", "")) != PRE_TERMINAL_PREFIX_WINDOW:
        raise ValueError("M3.O4 requires a pre-terminal prefix window")
    if str(candidate.get("next_experiment_recommendation", "")) != REFINE_PREFIX_WINDOW:
        raise ValueError("M3.O4 requires REFINE_PREFIX_WINDOW recommendation")
    if not candidate.get("recommended_refined_prefixes"):
        raise ValueError("recommended_refined_prefixes are required")
    if not candidate.get("recommended_conditions"):
        raise ValueError("recommended_conditions are required")
    if int(candidate.get("support", 0) or 0) != 0:
        raise ValueError("threshold candidate support must remain 0")
    return candidate


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_objective_refined_window_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3.O4 refined objective prefix window.",
    )
    parser.add_argument(
        "--threshold-consolidation",
        type=Path,
        default=DEFAULT_OBJECTIVE_THRESHOLD_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument(
        "--scope-consolidation",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--tie-break-seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_refined_window_execution(
        threshold_consolidation_path=args.threshold_consolidation,
        scope_consolidation_path=args.scope_consolidation,
        environments_dir=args.environments_dir,
        tie_break_seed=args.tie_break_seed,
        max_cells=args.max_cells,
    )
    write_objective_refined_window_results(payload, args.out)
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
