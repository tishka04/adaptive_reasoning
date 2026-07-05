"""M3.G7 - State-conditioned LLM hypothesis execution.

Consumes the M2.13a compiled M3 candidate requests
(`diagnostics/m2/state_conditioned_llm_m3_candidate_requests.json`), executes
only the `READY_FOR_M3` requests against the already-measured M3.G6 post-stop
substrate, ignores `BLOCKED_NOT_TESTABLE` requests (the unlock/affordance
metric), and compares the result to the M3.G6 outcome.

This executor does not step the environment, does not optimize a P3 policy, and
never turns a candidate signal into a scientific verdict. It answers one
question: do the LLM-sourced requests add a *new* testable completion/safety
signal beyond M3.G6, or do they reduce to the existing G6 proxy/completion
divergence failure?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.m2.schema import (
    M2_BLOCKED_FOR_M3_STATUS,
    M2_READY_FOR_M3_STATUS,
    M2_TRUTH_STATUS,
)

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .risk_aware_objective_completion_experiment_executor import (
    DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH,
)


DEFAULT_STATE_CONDITIONED_LLM_M3_REQUESTS_PATH = (
    Path("diagnostics") / "m2" / "state_conditioned_llm_m3_candidate_requests.json"
)
DEFAULT_STATE_CONDITIONED_LLM_EXECUTION_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "state_conditioned_llm_hypothesis_execution_results.json"
)
STATE_CONDITIONED_LLM_EXECUTION_SCHEMA_VERSION = (
    "m3.state_conditioned_llm_hypothesis_execution_results.v1"
)

# Outcome statuses (candidate-only).
LLM_ADDS_NEW_TESTABLE_SIGNAL = "LLM_ADDS_NEW_TESTABLE_SIGNAL_CANDIDATE_ONLY"
LLM_REDUCES_TO_EXISTING_G6_FAILURE = "LLM_REDUCES_TO_EXISTING_G6_FAILURE_CANDIDATE_ONLY"
LLM_REQUESTS_TOO_WEAK_NEED_M2_14 = "LLM_REQUESTS_TOO_WEAK_NEED_M2_14_CANDIDATE_ONLY"
SEMANTIC_COMPILATION_GAP = "SEMANTIC_COMPILATION_GAP_CANDIDATE_ONLY"
NO_READY_REQUESTS = "NO_READY_REQUESTS_CANDIDATE_ONLY"

EXECUTED_CELL_STATUS = "M3_G7_CELL_EXECUTED_CANDIDATE_ONLY"
UNBOUND_CELL_STATUS = "M3_G7_CELL_UNBOUND_CANDIDATE_ONLY"
IGNORED_BLOCKED_STATUS = "M3_G7_REQUEST_IGNORED_BLOCKED_NOT_TESTABLE"

# Control conditions as keyed in the M3.G6 source_measurement_context.
HOLD_CONDITION = "hold_or_stop_state"
ACTION6_ONLY_CONDITION = "ACTION6"
FROZEN_SELECTOR_CONDITION = "contextual_post_stop_conversion_policy"
RELATION_PROGRESS_CONDITION = "relation_progress_policy"

# Dynamic controls a candidate target sequence is compared against.
DYNAMIC_CONTROL_CONDITIONS: Tuple[str, ...] = (
    HOLD_CONDITION,
    ACTION6_ONLY_CONDITION,
    FROZEN_SELECTOR_CONDITION,
    RELATION_PROGRESS_CONDITION,
)

COMPLETION_METRIC = "objective_completion_signal"
TERMINAL_METRIC = "terminal_reentry_rate"

FORBIDDEN_INTERPRETATIONS: Tuple[str, ...] = (
    "Candidate cell signals are never scientific support or verdicts.",
    "Reproducing the M3.G6 substrate measurement is not new evidence.",
    "A negative completion result does not refute the LLM; it is candidate-only.",
    "No environment step, no policy rollout, no A32/A33 write occurs here.",
)


def run_state_conditioned_llm_hypothesis_execution(
    *,
    requests_path: str | Path = DEFAULT_STATE_CONDITIONED_LLM_M3_REQUESTS_PATH,
    m3g6_results_path: str
    | Path = DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH,
) -> Dict[str, Any]:
    request_payload = _load_json(requests_path)
    _validate_request_payload(request_payload)
    g6_payload = _load_json(m3g6_results_path)
    _validate_g6_payload(g6_payload)

    ready_requests = ready_llm_requests(request_payload)
    ignored_requests = ignored_blocked_requests(request_payload)
    context = source_measurement_context_from_g6(g6_payload)
    g6_comparison = g6_comparison_reference(g6_payload)

    execution_results = [
        execute_llm_request_cell(request, context=context, g6_comparison=g6_comparison)
        for request in ready_requests
    ]
    ignored_records = [
        ignored_request_record(request) for request in ignored_requests
    ]
    outcome_status = state_conditioned_llm_execution_outcome_status(
        execution_results=execution_results,
    )
    summary = summarize_state_conditioned_llm_execution(
        ready_requests=ready_requests,
        ignored_requests=ignored_requests,
        execution_results=execution_results,
        g6_comparison=g6_comparison,
        outcome_status=outcome_status,
    )

    return {
        "config": {
            "schema_version": STATE_CONDITIONED_LLM_EXECUTION_SCHEMA_VERSION,
            "stage": "M3.G7",
            "requests_path": str(requests_path),
            "m3g6_results_path": str(m3g6_results_path),
            "inputs_read": ["M2.13a", "M3.G6"],
            "primary_input": str(requests_path),
            "artifacts_not_modified": ["M2", "P2", "P3", "M3.G6", "A32", "A33"],
            "execution_mode": "g6_source_substrate_reuse_no_environment_step",
            "consume_only_status": M2_READY_FOR_M3_STATUS,
            "ignore_status": M2_BLOCKED_FOR_M3_STATUS,
            "context_replay_required": ["ACTION6"],
            "policy_optimization_performed": False,
            "policy_relearning_performed": False,
            "environment_step_performed": False,
            "source_cells_rerun": False,
        },
        "source_measurement_context": context,
        "g6_comparison_reference": g6_comparison,
        "execution_results": execution_results,
        "ignored_requests": ignored_records,
        "summary": summary,
        "state_conditioned_llm_execution_outcome_status": outcome_status,
        "forbidden_interpretations": list(FORBIDDEN_INTERPRETATIONS),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": bool(execution_results),
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "candidate_signal_counted_as_scientific_verdict": False,
        "llm_request_counted_as_evidence": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


# --------------------------------------------------------------------------- #
# Request selection
# --------------------------------------------------------------------------- #


def ready_llm_requests(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(row)
        for row in payload.get("experiment_requests", []) or []
        if isinstance(row, Mapping)
        and str(row.get("status", "")) == M2_READY_FOR_M3_STATUS
    ]


def ignored_blocked_requests(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(row)
        for row in payload.get("experiment_requests", []) or []
        if isinstance(row, Mapping)
        and str(row.get("status", "")) == M2_BLOCKED_FOR_M3_STATUS
    ]


def ignored_request_record(request: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "metric": str(request.get("metric", "")),
        "target_action": str(request.get("target_action", "")),
        "status": IGNORED_BLOCKED_STATUS,
        "reason": "blocked_not_testable_in_current_executor",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


# --------------------------------------------------------------------------- #
# Measurement substrate (reused from M3.G6)
# --------------------------------------------------------------------------- #


def source_measurement_context_from_g6(g6_payload: Mapping[str, Any]) -> Dict[str, Any]:
    context = dict(g6_payload.get("source_measurement_context", {}) or {})
    control_aggregates = {
        str(key): dict(value)
        for key, value in (context.get("control_aggregates", {}) or {}).items()
        if isinstance(value, Mapping)
    }
    return {
        "control_aggregates": control_aggregates,
        "measured_conditions": sorted(control_aggregates),
        "best_control_objective_completion_runs": int(
            context.get("best_control_objective_completion_runs", 0) or 0
        ),
        "best_control_mean_levels_completed": float(
            context.get("best_control_mean_levels_completed", 0.0) or 0.0
        ),
        "frozen_selector_terminal_reentry_rate": float(
            context.get("frozen_selector_terminal_reentry_rate", 0.0) or 0.0
        ),
        "proxy_progress_without_completion_observed": bool(
            context.get("proxy_progress_without_completion_observed", False)
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def g6_comparison_reference(g6_payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(g6_payload.get("summary", {}) or {})
    return {
        "g6_outcome_status": str(
            g6_payload.get("objective_completion_experiment_outcome_status", "")
            or summary.get("objective_completion_experiment_outcome_status", "")
        ),
        "g6_objective_completion_signal": bool(
            summary.get("objective_completion_signal", False)
        ),
        "g6_objective_completion_candidate_cells": int(
            summary.get("objective_completion_candidate_cells", 0) or 0
        ),
        "g6_levels_completed_after_rollout_max": float(
            summary.get("levels_completed_after_rollout_max", 0.0) or 0.0
        ),
        "g6_commit_action_cells_blocked": int(
            summary.get("commit_action_cells_blocked", 0) or 0
        ),
        "g6_proxy_progress_without_completion_observed": bool(
            summary.get("proxy_progress_without_completion_observed", False)
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


# --------------------------------------------------------------------------- #
# Per-request execution
# --------------------------------------------------------------------------- #


def candidate_target_sequence(request: Mapping[str, Any]) -> Tuple[str, ...]:
    replay = [str(a) for a in request.get("context_replay", []) or []]
    target = str(request.get("target_action", ""))
    return tuple(replay + ([target] if target else []))


def dynamic_control_aggregates(
    request: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    candidate_key: str,
) -> Dict[str, Dict[str, Any]]:
    control_aggregates = context.get("control_aggregates", {}) or {}
    controls: Dict[str, Dict[str, Any]] = {}
    for condition in DYNAMIC_CONTROL_CONDITIONS:
        if condition == candidate_key:
            continue
        aggregate = control_aggregates.get(condition)
        if isinstance(aggregate, Mapping):
            controls[condition] = dict(aggregate)
    return controls


def execute_llm_request_cell(
    request: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    g6_comparison: Mapping[str, Any],
) -> Dict[str, Any]:
    metric = str(request.get("metric", ""))
    target_sequence = candidate_target_sequence(request)
    candidate_key = ",".join(target_sequence)
    control_aggregates = context.get("control_aggregates", {}) or {}
    candidate_aggregate = control_aggregates.get(candidate_key)

    base = {
        "cell_id": "m3_g7::" + _stable_fragment(str(request.get("request_id", ""))),
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "game_id": str(request.get("game_id", "")),
        "metric": metric,
        "expected_signal": str(request.get("expected_signal", "")),
        "context_replay": [str(a) for a in request.get("context_replay", []) or []],
        "target_action": str(request.get("target_action", "")),
        "candidate_target_sequence": list(target_sequence),
        "candidate_condition_key": candidate_key,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "cell_result_counted_as_confirmation": False,
        "cell_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }

    if not isinstance(candidate_aggregate, Mapping):
        return {
            **base,
            "cell_execution_status": UNBOUND_CELL_STATUS,
            "bind_status": "UNBOUND",
            "unbound_reason": "target_sequence_not_present_in_g6_source_substrate",
            "execution_performed": False,
            "objective_completion_signal": False,
            "terminal_reentry_rate": None,
            "candidate_supported": False,
            "candidate_falsified": False,
            "already_measured_in_g6": False,
            "reduces_to_g6_failure": False,
        }

    candidate_aggregate = dict(candidate_aggregate)
    controls = dynamic_control_aggregates(
        request, context=context, candidate_key=candidate_key
    )
    completion_runs = int(candidate_aggregate.get("objective_completion_runs", 0) or 0)
    levels = float(candidate_aggregate.get("mean_levels_completed", 0.0) or 0.0)
    terminal_rate = float(candidate_aggregate.get("terminal_rate", 0.0) or 0.0)
    objective_completion_signal = completion_runs > 0

    best_control_completion = max(
        (int(agg.get("objective_completion_runs", 0) or 0) for agg in controls.values()),
        default=0,
    )
    best_control_terminal = min(
        (float(agg.get("terminal_rate", 0.0) or 0.0) for agg in controls.values()),
        default=0.0,
    )

    if metric == COMPLETION_METRIC:
        candidate_supported = (
            objective_completion_signal and completion_runs > best_control_completion
        )
        candidate_falsified = completion_runs <= best_control_completion and levels <= 0.0
    elif metric == TERMINAL_METRIC:
        candidate_supported = terminal_rate < best_control_terminal
        candidate_falsified = terminal_rate > best_control_terminal
    else:
        candidate_supported = False
        candidate_falsified = False

    reduces_to_g6_failure = bool(
        not candidate_supported
        and not objective_completion_signal
    )

    return {
        **base,
        "cell_execution_status": EXECUTED_CELL_STATUS,
        "bind_status": "BOUND",
        "bound_condition": candidate_key,
        "source_runs": int(candidate_aggregate.get("runs", 0) or 0),
        "execution_performed": True,
        "source_substrate_measurement_performed": True,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "objective_completion_signal": objective_completion_signal,
        "objective_completion_runs": completion_runs,
        "levels_completed_after_rollout": levels,
        "terminal_reentry_rate": round(terminal_rate, 6),
        "terminal_state_after_rollout": terminal_rate > 0.0,
        "mean_terminal_adjusted_progress": round(
            float(candidate_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0),
            6,
        ),
        "best_control_objective_completion_runs": best_control_completion,
        "best_control_terminal_reentry_rate": round(best_control_terminal, 6),
        "dynamic_controls_compared": sorted(controls),
        "candidate_supported": candidate_supported,
        "candidate_falsified": candidate_falsified,
        "already_measured_in_g6": True,
        "reduces_to_g6_failure": reduces_to_g6_failure,
        "candidate_signal_status": (
            "CANDIDATE_SIGNAL_PRESENT_CANDIDATE_ONLY"
            if candidate_supported
            else "NO_CANDIDATE_SIGNAL_REDUCES_TO_G6_CANDIDATE_ONLY"
        ),
    }


# --------------------------------------------------------------------------- #
# Outcome and summary
# --------------------------------------------------------------------------- #


def state_conditioned_llm_execution_outcome_status(
    *,
    execution_results: Sequence[Mapping[str, Any]],
) -> str:
    if not execution_results:
        return NO_READY_REQUESTS
    bound = [row for row in execution_results if str(row.get("bind_status", "")) == "BOUND"]
    unbound = [
        row for row in execution_results if str(row.get("bind_status", "")) == "UNBOUND"
    ]
    if not bound and unbound:
        return SEMANTIC_COMPILATION_GAP
    new_signal = [row for row in bound if bool(row.get("candidate_supported", False))]
    if new_signal:
        return LLM_ADDS_NEW_TESTABLE_SIGNAL
    all_reduce = bound and all(
        bool(row.get("reduces_to_g6_failure", False))
        and bool(row.get("already_measured_in_g6", False))
        for row in bound
    )
    if all_reduce:
        return LLM_REDUCES_TO_EXISTING_G6_FAILURE
    return LLM_REQUESTS_TOO_WEAK_NEED_M2_14


def summarize_state_conditioned_llm_execution(
    *,
    ready_requests: Sequence[Mapping[str, Any]],
    ignored_requests: Sequence[Mapping[str, Any]],
    execution_results: Sequence[Mapping[str, Any]],
    g6_comparison: Mapping[str, Any],
    outcome_status: str,
) -> Dict[str, Any]:
    bound = [row for row in execution_results if str(row.get("bind_status", "")) == "BOUND"]
    unbound = [
        row for row in execution_results if str(row.get("bind_status", "")) == "UNBOUND"
    ]
    completion_cells = [
        row for row in bound if bool(row.get("objective_completion_signal", False))
    ]
    supported_cells = [row for row in bound if bool(row.get("candidate_supported", False))]
    reduces_cells = [row for row in bound if bool(row.get("reduces_to_g6_failure", False))]
    metrics_executed = sorted({str(row.get("metric", "")) for row in bound})
    g7_completion = bool(completion_cells)
    reproduces_g6 = bool(
        not g7_completion
        and bool(g6_comparison.get("g6_proxy_progress_without_completion_observed", False))
        and len(reduces_cells) == len(bound)
        and bound
    )
    recommends_m2_14 = outcome_status in {
        LLM_REDUCES_TO_EXISTING_G6_FAILURE,
        LLM_REQUESTS_TOO_WEAK_NEED_M2_14,
    }
    return {
        "ready_requests_consumed": len(ready_requests),
        "blocked_requests_ignored": len(ignored_requests),
        "cells_executed": len(bound),
        "cells_unbound": len(unbound),
        "metrics_executed": metrics_executed,
        "objective_completion_signal": g7_completion,
        "objective_completion_candidate_cells": len(completion_cells),
        "candidate_supported_cells": len(supported_cells),
        "cells_reduce_to_g6_failure": len(reduces_cells),
        "levels_completed_after_rollout_max": max(
            (float(row.get("levels_completed_after_rollout", 0.0) or 0.0) for row in bound),
            default=0.0,
        ),
        "terminal_reentry_rate_max_executed_candidate": max(
            (float(row.get("terminal_reentry_rate", 0.0) or 0.0) for row in bound),
            default=0.0,
        ),
        "g6_outcome_status": str(g6_comparison.get("g6_outcome_status", "")),
        "g6_objective_completion_signal": bool(
            g6_comparison.get("g6_objective_completion_signal", False)
        ),
        "reproduces_g6_proxy_completion_divergence": reproduces_g6,
        "adds_signal_beyond_g6": bool(supported_cells),
        "recommends_m2_14_world_model": recommends_m2_14,
        "state_conditioned_llm_execution_outcome_status": outcome_status,
        "execution_performed": bool(bound),
        "source_substrate_measurement_performed": bool(bound),
        "policy_optimization_performed": False,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "candidate_signal_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_request_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("state-conditioned LLM M3 requests payload is empty")
    requests = payload.get("experiment_requests", [])
    if not isinstance(requests, list) or not requests:
        raise ValueError("experiment_requests are required")
    for row in requests:
        if not isinstance(row, Mapping):
            raise ValueError("each experiment request must be a mapping")
        if int(row.get("support", 0) or 0) != 0:
            raise ValueError("input requests must remain support=0")
        if str(row.get("truth_status", "")) != M2_TRUTH_STATUS:
            raise ValueError("input requests must remain M2-local truth_status")
        if bool(row.get("revision_performed", False)):
            raise ValueError("input requests must not be revised")
    summary = payload.get("summary", {}) or {}
    if int(summary.get("ready_for_m3", 0) or 0) < 0:
        raise ValueError("ready_for_m3 count is invalid")


def _validate_g6_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("M3.G6 results payload is empty")
    if not payload.get("source_measurement_context"):
        raise ValueError("M3.G6 results must carry source_measurement_context")
    if not bool(payload.get("execution_performed", False)):
        raise ValueError("M3.G6 results must have execution_performed=True")
    if int(payload.get("support", 0) or 0) != 0:
        raise ValueError("M3.G6 results must remain support=0")


# --------------------------------------------------------------------------- #
# IO and CLI
# --------------------------------------------------------------------------- #


def _stable_fragment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    return cleaned.strip("_") or "cell"


def _load_json(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_payload(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_STATE_CONDITIONED_LLM_EXECUTION_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute M2.13a state-conditioned LLM hypotheses (M3.G7).",
    )
    parser.add_argument(
        "--requests", type=Path, default=DEFAULT_STATE_CONDITIONED_LLM_M3_REQUESTS_PATH
    )
    parser.add_argument(
        "--m3g6-results",
        type=Path,
        default=DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_STATE_CONDITIONED_LLM_EXECUTION_OUTPUT_PATH
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_state_conditioned_llm_hypothesis_execution(
        requests_path=args.requests,
        m3g6_results_path=args.m3g6_results,
    )
    write_payload(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "outcome_status": payload[
                    "state_conditioned_llm_execution_outcome_status"
                ],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
