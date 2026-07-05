"""P3.1 terminal-aware stop/hold policy probe for bp35."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
)
from theory.m3.objective_refined_window_executor import (
    DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
    TERMINAL_AVOIDANCE_CANDIDATE,
)
from theory.m3.objective_stop_switch_experiment_executor import (
    execute_decision_step,
    is_terminal_game_state,
    update_rollout_memory,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.p1.bp35_sage_candidate_policy_probe import (
    DEFAULT_GAME_ID,
    DEFAULT_TIE_BREAK_SEEDS,
    PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
    CandidatePolicyMemory,
    ProbeStep,
    candidate_policy_memory_from_scope,
    execute_probe_condition,
    select_probe_decision,
    state_signature,
    summarize_probe_steps,
)
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_P3_TERMINAL_AWARE_STOP_POLICY_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "bp35_terminal_aware_stop_policy_probe.json"
)
DEFAULT_P3_BUDGETS = (64, 96, 128)
P3_TERMINAL_AWARE_SCHEMA_VERSION = "p3.terminal_aware_stop_policy_probe.v1"
TRUTH_STATUS = "NOT_EVALUATED_BY_P3_AGENT_PROBE"
BASELINE_POLICY = "patch_similarity_soft_stale_action6_prefix"
TERMINAL_AWARE_STOP_POLICY = "terminal_aware_stop_at_threshold"
STOP_OR_HOLD_SEMANTICS = "stop_rollout_without_extra_env_action_at_threshold"


def validate_refined_window_terminal_signal(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the candidate-only terminal avoidance threshold from M3.O4."""
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.O4 support must remain 0 for P3.1")
    if str(payload.get("revision_status", summary.get("revision_status", ""))) != "CANDIDATE_ONLY":
        raise ValueError("M3.O4 source must remain candidate-only")
    if bool(payload.get("revision_performed", summary.get("revision_performed", False))):
        raise ValueError("M3.O4 source cannot have revision_performed=true")
    if int(payload.get("wrong_confirmations", summary.get("wrong_confirmations", 0)) or 0) != 0:
        raise ValueError("M3.O4 source must have wrong_confirmations=0")
    if bool(summary.get("stop_switch_effectiveness_counted_as_verdict")):
        raise ValueError("M3.O4 terminal avoidance cannot be a verdict")
    if bool(summary.get("refined_window_result_counted_as_confirmation")):
        raise ValueError("M3.O4 refined result cannot be confirmation")
    if str(summary.get("stop_switch_effectiveness_status", "")) != TERMINAL_AVOIDANCE_CANDIDATE:
        raise ValueError("M3.O4 must expose candidate terminal avoidance")

    prefixes = [
        int(value)
        for value in summary.get("prefixes_with_candidate_terminal_avoidance", []) or []
    ]
    if not prefixes:
        raise ValueError("M3.O4 has no candidate terminal avoidance prefix")
    threshold = min(prefixes)
    return {
        "source_module": "M3.O4",
        "source_status": str(summary.get("stop_switch_effectiveness_status", "")),
        "stop_threshold_action6_prefix_count": threshold,
        "prefixes_with_candidate_terminal_avoidance": prefixes,
        "source_refined_prefixes": [
            int(value) for value in summary.get("refined_prefixes", []) or []
        ],
        "source_refined_conditions": [
            str(value) for value in summary.get("refined_conditions", []) or []
        ],
        "source_result_interpretation": str(summary.get("result_interpretation", "")),
        "source_counted_as_verdict": False,
        "source_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def action6_count(steps: Sequence[ProbeStep]) -> int:
    return len([step for step in steps if step.policy_selected_action == "ACTION6"])


def execute_terminal_aware_stop_condition(
    *,
    memory: CandidatePolicyMemory,
    environments_dir: str | Path,
    budget: int,
    game_id: str,
    stop_threshold_action6_prefix_count: int,
    tie_break_seed: int = 0,
) -> Tuple[Tuple[ProbeStep, ...], Dict[str, Any]]:
    """Run soft-stale ACTION6 until the candidate terminal threshold, then stop."""
    env = _make_env(game_id, environments_dir)
    current_frame = _reset_env(env)
    action_history: list[str] = []
    used_action6_args: list[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    seen_states: set[str] = set()
    steps: list[ProbeStep] = []
    initial = snapshot_frame(current_frame)
    seen_states.add(state_signature(initial.grid, initial.levels_completed, initial.game_state))

    stop_event: Dict[str, Any] = {
        "terminal_aware_stop_triggered": False,
        "stop_trigger_step": None,
        "stop_threshold_action6_prefix_count": int(stop_threshold_action6_prefix_count),
        "action6_prefix_count_at_stop": 0,
        "stop_reason": None,
        "stop_or_hold_semantics": STOP_OR_HOLD_SEMANTICS,
    }

    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(current_frame)
        if is_terminal_game_state(before.game_state):
            stop_event.update(
                {
                    "action6_prefix_count_at_stop": action6_count(steps),
                    "stop_reason": "terminal_state_reached_before_stop_policy",
                }
            )
            break
        if action6_count(steps) >= int(stop_threshold_action6_prefix_count):
            stop_event.update(
                {
                    "terminal_aware_stop_triggered": True,
                    "stop_trigger_step": step_index,
                    "action6_prefix_count_at_stop": action6_count(steps),
                    "stop_reason": "terminal_threshold_reached_before_next_action",
                }
            )
            break

        decision = select_probe_decision(
            condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            memory=memory,
            before_grid=before.grid,
            valid_actions=list(_valid_actions(env)),
            action_history=tuple(action_history),
            used_action6_args=tuple(used_action6_args),
            action_counts=action_counts,
            previous_steps=tuple(steps),
            tie_break_seed=tie_break_seed,
        )
        if (
            decision.action_name == "ACTION6"
            and action6_count(steps) >= int(stop_threshold_action6_prefix_count)
        ):
            stop_event.update(
                {
                    "terminal_aware_stop_triggered": True,
                    "stop_trigger_step": step_index,
                    "action6_prefix_count_at_stop": action6_count(steps),
                    "stop_reason": "terminal_threshold_reached_before_next_action6",
                }
            )
            break

        current_frame, step = execute_decision_step(
            env,
            current_frame,
            decision=decision,
            memory=memory,
            condition_label=TERMINAL_AWARE_STOP_POLICY,
            step_index=step_index,
            seen_states=seen_states,
        )
        steps.append(step)
        seen_states.add(step.state_signature_after)
        update_rollout_memory(
            step,
            action_history=action_history,
            used_action6_args=used_action6_args,
            action_counts=action_counts,
        )

    stop_event["action6_prefix_count_at_stop"] = action6_count(steps)
    return tuple(steps), stop_event


def terminal_aware_summary(
    *,
    condition: str,
    steps: Sequence[ProbeStep],
    budget: int,
    tie_break_seed: int,
    stop_threshold_action6_prefix_count: int,
    stop_event: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    summary = summarize_probe_steps(condition, steps)
    final_game_state = str(summary.get("final_game_state", ""))
    terminal = is_terminal_game_state(final_game_state)
    return {
        **summary,
        "budget": int(budget),
        "tie_break_seed": int(tie_break_seed),
        "stop_threshold_action6_prefix_count": int(stop_threshold_action6_prefix_count),
        "terminal_state_after_rollout": terminal,
        "objective_completion_signal": int(summary.get("final_levels_completed", 0) or 0) > 0,
        "terminal_aware_stop_triggered": bool(
            (stop_event or {}).get("terminal_aware_stop_triggered", False)
        ),
        "stop_trigger_step": (stop_event or {}).get("stop_trigger_step"),
        "action6_prefix_count_at_stop": int(
            (stop_event or {}).get(
                "action6_prefix_count_at_stop",
                summary.get("action6_steps", 0),
            )
            or 0
        ),
        "stop_reason": (stop_event or {}).get("stop_reason"),
        "stop_or_hold_semantics": (stop_event or {}).get("stop_or_hold_semantics"),
        "terminal_avoidance_counted_as_completion": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def compare_terminal_policy_pair(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_terminal = bool(baseline.get("terminal_state_after_rollout"))
    candidate_terminal = bool(candidate.get("terminal_state_after_rollout"))
    baseline_levels = int(baseline.get("final_levels_completed", 0) or 0)
    candidate_levels = int(candidate.get("final_levels_completed", 0) or 0)
    game_over_avoided = (
        str(baseline.get("final_game_state", "")) == "GAME_OVER"
        and str(candidate.get("final_game_state", "")) != "GAME_OVER"
    )
    levels_delta = candidate_levels - baseline_levels
    progress_delta = round(
        float(candidate.get("progress_proxy", 0.0) or 0.0)
        - float(baseline.get("progress_proxy", 0.0) or 0.0),
        4,
    )
    return {
        "budget": int(candidate.get("budget", baseline.get("budget", 0)) or 0),
        "tie_break_seed": int(
            candidate.get("tie_break_seed", baseline.get("tie_break_seed", 0)) or 0
        ),
        "baseline_final_game_state": str(baseline.get("final_game_state", "")),
        "candidate_final_game_state": str(candidate.get("final_game_state", "")),
        "baseline_terminal": baseline_terminal,
        "candidate_terminal": candidate_terminal,
        "game_over_avoided": game_over_avoided,
        "terminal_avoidance_signal": bool(baseline_terminal and not candidate_terminal),
        "baseline_levels_completed": baseline_levels,
        "candidate_levels_completed": candidate_levels,
        "levels_completed_delta": levels_delta,
        "objective_completion_signal": bool(candidate_levels > baseline_levels),
        "objective_progress_proxy_delta": progress_delta,
        "terminal_avoidance_only": bool(game_over_avoided and levels_delta <= 0),
        "terminal_avoidance_counted_as_completion": False,
        "policy_result_counted_as_scientific_verdict": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def summarize_terminal_probe(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    tie_break_seeds: Sequence[int],
    threshold: int,
) -> Dict[str, Any]:
    baseline_game_over = len(
        [row for row in comparisons if row.get("baseline_final_game_state") == "GAME_OVER"]
    )
    candidate_game_over = len(
        [row for row in comparisons if row.get("candidate_final_game_state") == "GAME_OVER"]
    )
    terminal_avoidance = len(
        [row for row in comparisons if bool(row.get("terminal_avoidance_signal"))]
    )
    objective_completion = len(
        [row for row in comparisons if bool(row.get("objective_completion_signal"))]
    )
    return {
        "conditions_run": 2,
        "runs_per_condition": len(comparisons),
        "budgets_tested": [int(value) for value in budgets],
        "tie_break_seeds_tested": [int(value) for value in tie_break_seeds],
        "stop_threshold_action6_prefix_count": int(threshold),
        "baseline_game_over_runs": baseline_game_over,
        "candidate_game_over_runs": candidate_game_over,
        "game_over_run_delta_candidate_minus_baseline": (
            candidate_game_over - baseline_game_over
        ),
        "terminal_avoidance_signal_runs": terminal_avoidance,
        "objective_completion_signal_runs": objective_completion,
        "terminal_avoidance_only_runs": len(
            [row for row in comparisons if bool(row.get("terminal_avoidance_only"))]
        ),
        "candidate_reduces_game_over_rate_candidate_only": candidate_game_over
        < baseline_game_over,
        "candidate_improves_level_completion_candidate_only": objective_completion > 0,
        "terminal_avoidance_only_is_not_objective_completion": objective_completion == 0,
        "execution_performed": True,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "terminal_avoidance_counted_as_completion": False,
        "candidate_policy_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a33_ready": False,
    }


PairExecutor = Callable[
    [int, int, int, CandidatePolicyMemory, Path, str],
    Tuple[Mapping[str, Any], Mapping[str, Any]],
]


def default_pair_executor(
    budget: int,
    tie_break_seed: int,
    threshold: int,
    memory: CandidatePolicyMemory,
    env_dir: Path,
    game_id: str,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    baseline_steps = execute_probe_condition(
        condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        memory=memory,
        environments_dir=env_dir,
        budget=budget,
        game_id=game_id,
        tie_break_seed=tie_break_seed,
    )
    candidate_steps, stop_event = execute_terminal_aware_stop_condition(
        memory=memory,
        environments_dir=env_dir,
        budget=budget,
        game_id=game_id,
        stop_threshold_action6_prefix_count=threshold,
        tie_break_seed=tie_break_seed,
    )
    baseline = terminal_aware_summary(
        condition=BASELINE_POLICY,
        steps=baseline_steps,
        budget=budget,
        tie_break_seed=tie_break_seed,
        stop_threshold_action6_prefix_count=threshold,
    )
    candidate = terminal_aware_summary(
        condition=TERMINAL_AWARE_STOP_POLICY,
        steps=candidate_steps,
        budget=budget,
        tie_break_seed=tie_break_seed,
        stop_threshold_action6_prefix_count=threshold,
        stop_event=stop_event,
    )
    return baseline, candidate


def run_terminal_aware_stop_policy_probe(
    *,
    refined_window_results_path: str | Path = (
        DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH
    ),
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    budgets: Sequence[int] = DEFAULT_P3_BUDGETS,
    tie_break_seeds: Sequence[int] = DEFAULT_TIE_BREAK_SEEDS,
    game_id: str = DEFAULT_GAME_ID,
    pair_executor: PairExecutor | None = None,
) -> Dict[str, Any]:
    refined_payload = _load_json(refined_window_results_path)
    terminal_signal = validate_refined_window_terminal_signal(refined_payload)
    threshold = int(terminal_signal["stop_threshold_action6_prefix_count"])
    scope_payload = _load_json(scope_consolidation_path)
    memory = candidate_policy_memory_from_scope(scope_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    if pair_executor is None:
        _configure_offline_env(env_dir)
        pair_executor = default_pair_executor

    run_results: list[Dict[str, Any]] = []
    comparisons: list[Dict[str, Any]] = []
    for budget in budgets:
        for seed in tie_break_seeds:
            baseline, candidate = pair_executor(
                int(budget),
                int(seed),
                threshold,
                memory,
                env_dir,
                game_id,
            )
            comparison = compare_terminal_policy_pair(
                baseline=baseline,
                candidate=candidate,
            )
            run_results.append(
                {
                    "budget": int(budget),
                    "tie_break_seed": int(seed),
                    "baseline": dict(baseline),
                    "candidate": dict(candidate),
                    "comparison": comparison,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": TRUTH_STATUS,
                }
            )
            comparisons.append(comparison)

    summary = summarize_terminal_probe(
        comparisons,
        budgets=budgets,
        tie_break_seeds=tie_break_seeds,
        threshold=threshold,
    )
    return {
        "config": {
            "schema_version": P3_TERMINAL_AWARE_SCHEMA_VERSION,
            "refined_window_results_path": str(refined_window_results_path),
            "scope_consolidation_path": str(scope_consolidation_path),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "baseline_policy": BASELINE_POLICY,
            "candidate_policy": TERMINAL_AWARE_STOP_POLICY,
            "inputs_read": ["M3.O4", "M3.24"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33"],
            "stop_or_hold_semantics": STOP_OR_HOLD_SEMANTICS,
        },
        "source_terminal_signal": terminal_signal,
        "candidate_policy_memory": memory.to_dict(),
        "run_results": run_results,
        "comparisons": comparisons,
        "summary": summary,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": TRUTH_STATUS,
        "execution_performed": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "terminal_avoidance_counted_as_completion": False,
        "candidate_policy_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a33_ready": False,
    }


def write_terminal_aware_stop_policy_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P3_TERMINAL_AWARE_STOP_POLICY_PROBE_OUTPUT_PATH,
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
        description="Run P3.1 terminal-aware stop policy probe.",
    )
    parser.add_argument(
        "--refined-window-results",
        type=Path,
        default=DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--scope-consolidation",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budgets", type=int, nargs="*", default=list(DEFAULT_P3_BUDGETS))
    parser.add_argument(
        "--tie-break-seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_TIE_BREAK_SEEDS),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P3_TERMINAL_AWARE_STOP_POLICY_PROBE_OUTPUT_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_terminal_aware_stop_policy_probe(
        refined_window_results_path=args.refined_window_results,
        scope_consolidation_path=args.scope_consolidation,
        environments_dir=args.environments_dir,
        budgets=tuple(args.budgets or DEFAULT_P3_BUDGETS),
        tie_break_seeds=tuple(args.tie_break_seeds or DEFAULT_TIE_BREAK_SEEDS),
        game_id=args.game_id,
    )
    write_terminal_aware_stop_policy_probe(payload, args.out)


if __name__ == "__main__":
    main()

