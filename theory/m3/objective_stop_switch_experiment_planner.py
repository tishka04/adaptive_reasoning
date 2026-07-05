"""M3.O1 planner for objective stop/switch experiments from M2.O1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.m2.objective_conditioned_hypotheses import (
    DEFAULT_M2_OBJECTIVE_CONDITIONED_HYPOTHESES_OUTPUT_PATH,
)

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_OBJECTIVE_STOP_SWITCH_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "objective_stop_switch_experiment_requests.json"
)
READY_FOR_M3_OBJECTIVE_EXPERIMENT = "READY_FOR_M3_OBJECTIVE_EXPERIMENT"
BLOCKED_OBJECTIVE_EXPERIMENT = "BLOCKED_OBJECTIVE_EXPERIMENT"
DEFAULT_PREFIX_LENGTHS = (6, 12, 24, 48, 64)
DEFAULT_OBJECTIVE_METRICS = (
    "final_game_state",
    "terminal_state_after_rollout",
    "levels_completed_after_rollout",
    "objective_progress_proxy",
)
DEFAULT_LOCAL_DIAGNOSTIC_METRICS = (
    "local_effect_metric",
    "repeated_action6_count",
    "useful_action6_steps",
)
DEFAULT_SWITCH_ACTIONS = ("ACTION3", "ACTION4", "ACTION1", "ACTION2")


@dataclass(frozen=True)
class ObjectiveStopSwitchExperimentRequest:
    """Candidate-only protocol request for M3.O2 execution."""

    request_id: str
    source_hypothesis_id: str
    source_request_id: str
    game_id: str
    hypothesis_family: str
    hypothesis_tested: str
    prefix_action: str
    prefix_lengths: Tuple[int, ...]
    experimental_conditions: Tuple[Dict[str, Any], ...]
    primary_objective_metrics: Tuple[str, ...]
    local_diagnostic_metrics: Tuple[str, ...]
    metrics: Tuple[str, ...]
    expected_signal: str
    falsification_criteria: Tuple[Dict[str, Any], ...]
    controlled_comparison: str
    planning_rationale: str
    status: str = READY_FOR_M3_OBJECTIVE_EXPERIMENT
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    execution_performed: bool = False
    revision_performed: bool = False
    wrong_confirmations: int = 0
    hypothesis_counted_as_confirmation: bool = False
    objective_request_counted_as_support: bool = False
    policy_result_counted_as_scientific_verdict: bool = False
    a32_remains_only_verdict_location: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_hypothesis_id": self.source_hypothesis_id,
            "source_request_id": self.source_request_id,
            "game_id": self.game_id,
            "hypothesis_family": self.hypothesis_family,
            "hypothesis_tested": self.hypothesis_tested,
            "prefix_action": self.prefix_action,
            "prefix_lengths": [int(value) for value in self.prefix_lengths],
            "experimental_conditions": [
                dict(condition) for condition in self.experimental_conditions
            ],
            "primary_objective_metrics": list(self.primary_objective_metrics),
            "local_diagnostic_metrics": list(self.local_diagnostic_metrics),
            "metrics": list(self.metrics),
            "expected_signal": self.expected_signal,
            "falsification_criteria": [
                dict(item) for item in self.falsification_criteria
            ],
            "controlled_comparison": self.controlled_comparison,
            "planning_rationale": self.planning_rationale,
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "execution_performed": self.execution_performed,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "hypothesis_counted_as_confirmation": (
                self.hypothesis_counted_as_confirmation
            ),
            "objective_request_counted_as_support": (
                self.objective_request_counted_as_support
            ),
            "policy_result_counted_as_scientific_verdict": (
                self.policy_result_counted_as_scientific_verdict
            ),
            "a32_remains_only_verdict_location": (
                self.a32_remains_only_verdict_location
            ),
        }


def run_objective_stop_switch_experiment_planning(
    *,
    objective_hypotheses_path: str | Path = (
        DEFAULT_M2_OBJECTIVE_CONDITIONED_HYPOTHESES_OUTPUT_PATH
    ),
    prefix_lengths: Sequence[int] = DEFAULT_PREFIX_LENGTHS,
) -> Dict[str, Any]:
    payload = _load_json(objective_hypotheses_path)
    _validate_source_payload(payload)
    hypotheses = objective_hypotheses_from_payload(payload)
    requests = [
        build_objective_stop_switch_request(
            hypothesis,
            prefix_lengths=prefix_lengths,
        )
        for hypothesis in hypotheses
        if is_ready_objective_hypothesis(hypothesis)
    ]
    skipped = [
        skipped_objective_hypothesis(hypothesis)
        for hypothesis in hypotheses
        if not is_ready_objective_hypothesis(hypothesis)
    ]
    for request in requests:
        validate_objective_stop_switch_request(request)

    return {
        "config": {
            "objective_hypotheses_path": str(objective_hypotheses_path),
            "schema_version": "m3.objective_stop_switch_requests.v1",
            "inputs_read": ["M2.O1"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "execution_performed": False,
            "prefix_lengths": [int(value) for value in prefix_lengths],
            "condition_policy": "continue_vs_stop_vs_switch",
        },
        "summary": summarize_objective_stop_switch_requests(
            hypotheses=hypotheses,
            requests=requests,
            skipped=skipped,
            prefix_lengths=prefix_lengths,
        ),
        "objective_stop_switch_experiment_requests": [
            request.to_dict() for request in requests
        ],
        "skipped_objective_hypotheses": skipped,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "hypothesis_counted_as_confirmation": False,
        "objective_request_counted_as_support": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a32_remains_only_verdict_location": True,
    }


def build_objective_stop_switch_request(
    hypothesis: Mapping[str, Any],
    *,
    prefix_lengths: Sequence[int],
) -> ObjectiveStopSwitchExperimentRequest:
    family = str(hypothesis.get("hypothesis_family", ""))
    prefix_action = str(hypothesis.get("candidate_action", "ACTION6") or "ACTION6")
    if family == "subgoal_switch_after_local_affordance":
        prefix_action = "ACTION6"
    metrics = ordered_unique(
        (*DEFAULT_OBJECTIVE_METRICS, *DEFAULT_LOCAL_DIAGNOSTIC_METRICS)
    )
    return ObjectiveStopSwitchExperimentRequest(
        request_id=objective_request_id(hypothesis),
        source_hypothesis_id=str(hypothesis.get("hypothesis_id", "")),
        source_request_id=str(hypothesis.get("source_request_id", "")),
        game_id=str(hypothesis.get("game_id", "")),
        hypothesis_family=family,
        hypothesis_tested=str(hypothesis.get("predicted_effect", "")),
        prefix_action=prefix_action,
        prefix_lengths=tuple(int(value) for value in prefix_lengths),
        experimental_conditions=objective_conditions(
            prefix_action=prefix_action,
            switch_actions=DEFAULT_SWITCH_ACTIONS,
        ),
        primary_objective_metrics=DEFAULT_OBJECTIVE_METRICS,
        local_diagnostic_metrics=DEFAULT_LOCAL_DIAGNOSTIC_METRICS,
        metrics=metrics,
        expected_signal=str(
            (hypothesis.get("testability", {}) or {}).get(
                "expected_signal_type",
                "objective_condition_separates_terminal_outcomes",
            )
        ),
        falsification_criteria=falsification_criteria_for_hypothesis(
            hypothesis,
            metrics=metrics,
        ),
        controlled_comparison="same_prefix_length_compare_continue_stop_switch",
        planning_rationale=planning_rationale_for_family(family),
    )


def objective_conditions(
    *,
    prefix_action: str,
    switch_actions: Sequence[str],
) -> Tuple[Dict[str, Any], ...]:
    conditions = [
        {
            "condition_id": "continue_action6",
            "condition_family": "continue_local_affordance",
            "post_prefix_policy": "continue_action",
            "post_prefix_action": prefix_action,
            "role": "target_or_risk_condition",
        },
        {
            "condition_id": "stop_policy",
            "condition_family": "stop_or_noop",
            "post_prefix_policy": "stop_or_hold_if_available",
            "post_prefix_action": None,
            "role": "stop_control",
        },
    ]
    for action in switch_actions:
        conditions.append(
            {
                "condition_id": f"switch_{action}",
                "condition_family": "switch_subgoal",
                "post_prefix_policy": "switch_to_action",
                "post_prefix_action": action,
                "role": "switch_control",
            }
        )
    return tuple(conditions)


def falsification_criteria_for_hypothesis(
    hypothesis: Mapping[str, Any],
    *,
    metrics: Sequence[str],
) -> Tuple[Dict[str, Any], ...]:
    source = dict(hypothesis.get("falsification", {}) or {})
    criteria = []
    source_metric = str(source.get("metric", ""))
    if source_metric:
        criteria.append(
            {
                "metric": source_metric,
                "support_condition": str(source.get("support_condition", "")),
                "failure_condition": str(source.get("failure_condition", "")),
                "minimum_effect_size": source.get("minimum_effect_size", 1),
                "source": "m2_o1_hypothesis",
            }
        )
    for metric in metrics:
        if metric == source_metric:
            continue
        criteria.append(
            {
                "metric": metric,
                "support_condition": (
                    "candidate_condition_signal > best_control_condition_signal"
                ),
                "failure_condition": (
                    "candidate_condition_signal <= best_control_condition_signal"
                ),
                "minimum_effect_size": 1,
                "source": "m3_o1_default_metric_guard",
            }
        )
    return tuple(criteria)


def planning_rationale_for_family(family: str) -> str:
    if family == "stop_switch_criterion":
        return (
            "Compare continued ACTION6 exploitation against stop and switch controls "
            "at matched prefix lengths to test terminal risk from continuing."
        )
    if family == "terminal_risk_predictor":
        return (
            "Vary ACTION6 prefix length to test whether terminal outcome risk grows "
            "while local effects remain productive."
        )
    if family == "subgoal_switch_after_local_affordance":
        return (
            "Test whether switching to ACTION3/ACTION4/ACTION1/ACTION2 after local "
            "ACTION6 exploitation improves objective metrics."
        )
    if family == "global_objective_alignment_metric":
        return (
            "Compare local-effect diagnostics to objective metrics so M3 can test "
            "whether local productivity predicts or hides terminal failure."
        )
    return "Objective stop/switch follow-up for M2.O1 hypothesis."


def summarize_objective_stop_switch_requests(
    *,
    hypotheses: Sequence[Mapping[str, Any]],
    requests: Sequence[ObjectiveStopSwitchExperimentRequest],
    skipped: Sequence[Mapping[str, Any]],
    prefix_lengths: Sequence[int],
) -> Dict[str, Any]:
    condition_count = len(objective_conditions(prefix_action="ACTION6", switch_actions=DEFAULT_SWITCH_ACTIONS))
    return {
        "objective_hypotheses_consumed": len(hypotheses),
        "objective_experiment_requests_generated": len(requests),
        "skipped_objective_hypotheses": len(skipped),
        "prefix_lengths": [int(value) for value in prefix_lengths],
        "conditions_per_request": condition_count,
        "planned_condition_cells": len(requests) * len(prefix_lengths) * condition_count,
        "continue_conditions": len(requests),
        "stop_conditions": len(requests),
        "switch_conditions": len(requests) * len(DEFAULT_SWITCH_ACTIONS),
        "primary_objective_metrics": list(DEFAULT_OBJECTIVE_METRICS),
        "local_diagnostic_metrics": list(DEFAULT_LOCAL_DIAGNOSTIC_METRICS),
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "hypothesis_counted_as_confirmation": False,
        "objective_request_counted_as_support": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a32_remains_only_verdict_location": True,
    }


def objective_hypotheses_from_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for batch in payload.get("objective_hypothesis_batches", []) or []:
        if not isinstance(batch, Mapping):
            continue
        for hypothesis in batch.get("candidate_hypotheses", []) or []:
            if isinstance(hypothesis, Mapping):
                rows.append(dict(hypothesis))
    return tuple(rows)


def is_ready_objective_hypothesis(hypothesis: Mapping[str, Any]) -> bool:
    return bool(
        str(hypothesis.get("status", "")) == "UNRESOLVED"
        and int(hypothesis.get("support", 0) or 0) == 0
        and str(hypothesis.get("revision_status", "")) == "CANDIDATE_ONLY"
        and str(hypothesis.get("truth_status", "")) == "NOT_EVALUATED_BY_M2"
        and not bool(hypothesis.get("revision_performed", False))
        and int(hypothesis.get("wrong_confirmations", 0) or 0) == 0
        and bool(hypothesis.get("ready_for_m3_candidate_experiment_request", False))
        and bool((hypothesis.get("testability", {}) or {}).get("testable", False))
    )


def skipped_objective_hypothesis(hypothesis: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_hypothesis_id": str(hypothesis.get("hypothesis_id", "")),
        "hypothesis_family": str(hypothesis.get("hypothesis_family", "")),
        "reason": "not_ready_for_m3_objective_experiment",
        "status": BLOCKED_OBJECTIVE_EXPERIMENT,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def validate_objective_stop_switch_request(
    request: ObjectiveStopSwitchExperimentRequest | Mapping[str, Any],
) -> None:
    data = request.to_dict() if isinstance(request, ObjectiveStopSwitchExperimentRequest) else dict(request)
    if not str(data.get("request_id", "")):
        raise ValueError("request_id is required")
    if not str(data.get("source_hypothesis_id", "")):
        raise ValueError("source_hypothesis_id is required")
    if not data.get("prefix_lengths"):
        raise ValueError("prefix_lengths are required")
    if not data.get("experimental_conditions"):
        raise ValueError("experimental_conditions are required")
    if str(data.get("status", "")) != READY_FOR_M3_OBJECTIVE_EXPERIMENT:
        raise ValueError("objective request must be ready for M3 objective experiment")
    if str(data.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("objective request must remain candidate-only")
    if int(data.get("support", 0) or 0) != 0:
        raise ValueError("objective request support must remain 0")
    if str(data.get("truth_status", "")) != M3_REFINEMENT_TRUTH_STATUS:
        raise ValueError("objective request truth_status must remain M3-local")
    if bool(data.get("execution_performed", False)):
        raise ValueError("objective planner cannot execute requests")
    if bool(data.get("revision_performed", False)):
        raise ValueError("objective request revision_performed must be false")
    if int(data.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("objective request wrong_confirmations must remain 0")
    if bool(data.get("hypothesis_counted_as_confirmation", False)):
        raise ValueError("objective request cannot count hypothesis as confirmation")
    if bool(data.get("objective_request_counted_as_support", False)):
        raise ValueError("objective request cannot count as support")
    if bool(data.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot count as scientific verdict")


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(summary.get("revision_performed", False)):
        raise ValueError("source summary revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source summary wrong_confirmations must remain 0")
    if bool(payload.get("objective_hypotheses_counted_as_confirmation", False)):
        raise ValueError("objective hypotheses cannot count as confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")


def objective_request_id(hypothesis: Mapping[str, Any]) -> str:
    source = str(hypothesis.get("hypothesis_id", "")).replace("::", "_")
    family = str(hypothesis.get("hypothesis_family", "objective"))
    return f"m3_o1::{source}::{family}"


def ordered_unique(values: Sequence[str]) -> Tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def write_objective_stop_switch_experiment_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OBJECTIVE_STOP_SWITCH_REQUESTS_OUTPUT_PATH,
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
        description="Build M3.O1 objective stop/switch experiment requests.",
    )
    parser.add_argument(
        "--objective-hypotheses",
        type=Path,
        default=DEFAULT_M2_OBJECTIVE_CONDITIONED_HYPOTHESES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_STOP_SWITCH_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_stop_switch_experiment_planning(
        objective_hypotheses_path=args.objective_hypotheses,
        prefix_lengths=tuple(args.prefix_lengths or DEFAULT_PREFIX_LENGTHS),
    )
    write_objective_stop_switch_experiment_requests(payload, args.out)
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
