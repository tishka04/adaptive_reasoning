"""M3.G1.1 planner for stop-state objective-conversion experiments from M2.G1.

This module compiles the 12 M2.G1 objective-conversion hypotheses
(``diagnostics/m2/objective_conversion_hypotheses.json``) into candidate-only
controlled experiment requests for the M3.G1.2 executor.

Discipline (never violated here):
- The planner does not execute anything.
- The planner does not confirm, refute, or revise any hypothesis.
- Every artifact keeps ``support=0``, ``revision_status=CANDIDATE_ONLY``,
  ``truth_status=NOT_EVALUATED_BY_M3``, and never writes A32/A33.

Central experimental unit: ``safe_stop_state -> candidate_action_or_sequence``
compared against two controls:
- ``hold_or_stop_state``      (zero-action endpoint observation)
- ``relation_progress_policy`` (P3.G0-style relation-progress continuation,
  horizon-matched to ``candidate_len``)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.m2.objective_conversion_hypothesis_generator import (
    DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH,
)

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "objective_conversion_experiment_requests.json"
)
OBJECTIVE_CONVERSION_REQUESTS_SCHEMA_VERSION = (
    "m3.objective_conversion_experiment_requests.v1"
)
READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT = (
    "READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT"
)
BLOCKED_OBJECTIVE_CONVERSION_EXPERIMENT = "BLOCKED_OBJECTIVE_CONVERSION_EXPERIMENT"

SAFE_STOP_POLICY_CONDITION = "objective_aware_abstract_policy_lambda_0"
SAFE_STOP_STATE_SOURCE = "P3.G1"
SAFE_STOP_STOP_TRIGGER_REASON = "objective_aware_terminal_risk_score_stop"
SAFE_STOP_BASE_STATE_FAMILY = "terminal_safe_stop_or_avoidance_state"

HOLD_OR_STOP_STATE_CONTROL = "hold_or_stop_state"
RELATION_PROGRESS_POLICY_CONTROL = "relation_progress_policy"

# Decisive metric is a *delta* against the zero-action hold control, never the
# raw safe-stop progress (which may already be good on its own).
DELTA_VS_HOLD_METRIC = "delta_terminal_adjusted_progress_vs_hold"
PRIMARY_METRICS = (
    DELTA_VS_HOLD_METRIC,
    "terminal_adjusted_progress_after_stop",
    "terminal_reentry_rate",
    "objective_completion_signal",
    "levels_completed_after_rollout",
)
DIAGNOSTIC_METRICS = (
    "relation_delta_after_stop",
    "changed_pixels",
    "new_relation_states",
    "distance_decreases_count",
    "objective_readiness_signature_delta",
)
CENTRAL_DECISION_RULE = (
    "candidate beats hold_or_stop_state on "
    "delta_terminal_adjusted_progress_vs_hold > 0 AND terminal_reentry == false"
)
SECONDARY_DECISION_RULE = "candidate beats relation_progress_policy"


@dataclass(frozen=True)
class ObjectiveConversionExperimentRequest:
    """Candidate-only objective-conversion protocol request for M3.G1.2."""

    request_id: str
    source_hypothesis_id: str
    source_request_id: str
    game_id: str
    hypothesis_family: str
    hypothesis_tested: str
    requested_experiment_style: str
    falsification_signal: str
    safe_stop_spec: Dict[str, Any]
    candidate_conditions: Tuple[Dict[str, Any], ...]
    control_conditions: Tuple[Dict[str, Any], ...]
    primary_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    decision_rule: Dict[str, str]
    falsification_criteria: Tuple[Dict[str, Any], ...]
    planning_rationale: str
    status: str = READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    execution_performed: bool = False
    revision_performed: bool = False
    wrong_confirmations: int = 0
    m2_hypothesis_counted_as_confirmation: bool = False
    objective_conversion_request_counted_as_support: bool = False
    experiment_result_counted_as_scientific_verdict: bool = False
    a32_write_performed: bool = False
    a33_write_performed: bool = False
    a32_remains_only_verdict_location: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_hypothesis_id": self.source_hypothesis_id,
            "source_request_id": self.source_request_id,
            "game_id": self.game_id,
            "hypothesis_family": self.hypothesis_family,
            "hypothesis_tested": self.hypothesis_tested,
            "requested_experiment_style": self.requested_experiment_style,
            "falsification_signal": self.falsification_signal,
            "safe_stop_spec": dict(self.safe_stop_spec),
            "candidate_conditions": [
                dict(condition) for condition in self.candidate_conditions
            ],
            "control_conditions": [
                dict(condition) for condition in self.control_conditions
            ],
            "primary_metrics": list(self.primary_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "decision_rule": dict(self.decision_rule),
            "falsification_criteria": [
                dict(item) for item in self.falsification_criteria
            ],
            "planning_rationale": self.planning_rationale,
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "execution_performed": self.execution_performed,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "m2_hypothesis_counted_as_confirmation": (
                self.m2_hypothesis_counted_as_confirmation
            ),
            "objective_conversion_request_counted_as_support": (
                self.objective_conversion_request_counted_as_support
            ),
            "experiment_result_counted_as_scientific_verdict": (
                self.experiment_result_counted_as_scientific_verdict
            ),
            "a32_write_performed": self.a32_write_performed,
            "a33_write_performed": self.a33_write_performed,
            "a32_remains_only_verdict_location": (
                self.a32_remains_only_verdict_location
            ),
        }


def run_objective_conversion_experiment_planning(
    *,
    objective_conversion_hypotheses_path: str | Path = (
        DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(objective_conversion_hypotheses_path)
    _validate_source_payload(payload)
    hypotheses = objective_conversion_hypotheses_from_payload(payload)
    requests = [
        build_objective_conversion_experiment_request(hypothesis)
        for hypothesis in hypotheses
        if is_ready_objective_conversion_hypothesis(hypothesis)
    ]
    skipped = [
        skipped_objective_conversion_hypothesis(hypothesis)
        for hypothesis in hypotheses
        if not is_ready_objective_conversion_hypothesis(hypothesis)
    ]
    for request in requests:
        validate_objective_conversion_experiment_request(request)

    return {
        "config": {
            "objective_conversion_hypotheses_path": str(
                objective_conversion_hypotheses_path
            ),
            "schema_version": OBJECTIVE_CONVERSION_REQUESTS_SCHEMA_VERSION,
            "inputs_read": ["M2.G1"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "execution_performed": False,
            "central_experimental_unit": (
                "safe_stop_state -> candidate_action_or_sequence"
            ),
            "controls": [
                HOLD_OR_STOP_STATE_CONTROL,
                RELATION_PROGRESS_POLICY_CONTROL,
            ],
            "relation_progress_policy_horizon_rule": (
                "post_stop_horizon = candidate_len"
            ),
        },
        "summary": summarize_objective_conversion_requests(
            hypotheses=hypotheses,
            requests=requests,
            skipped=skipped,
        ),
        "objective_conversion_experiment_requests": [
            request.to_dict() for request in requests
        ],
        "skipped_objective_conversion_hypotheses": skipped,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "objective_conversion_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def build_objective_conversion_experiment_request(
    hypothesis: Mapping[str, Any],
) -> ObjectiveConversionExperimentRequest:
    family = str(hypothesis.get("hypothesis_family", ""))
    candidate_conditions = candidate_conditions_for_hypothesis(hypothesis)
    return ObjectiveConversionExperimentRequest(
        request_id=objective_conversion_request_id(hypothesis),
        source_hypothesis_id=str(hypothesis.get("hypothesis_id", "")),
        source_request_id=str(hypothesis.get("source_request_id", "")),
        game_id=str(hypothesis.get("game_id", "")),
        hypothesis_family=family,
        hypothesis_tested=str(hypothesis.get("claim", "") or hypothesis.get("predicted_effect", "")),
        requested_experiment_style=str(
            hypothesis.get("requested_experiment_style", "")
        ),
        falsification_signal=str(hypothesis.get("falsification_signal", "")),
        safe_stop_spec=default_safe_stop_spec(),
        candidate_conditions=candidate_conditions,
        control_conditions=control_conditions_for_candidates(candidate_conditions),
        primary_metrics=PRIMARY_METRICS,
        diagnostic_metrics=DIAGNOSTIC_METRICS,
        decision_rule={
            "central": CENTRAL_DECISION_RULE,
            "secondary": SECONDARY_DECISION_RULE,
        },
        falsification_criteria=falsification_criteria_for_hypothesis(hypothesis),
        planning_rationale=planning_rationale_for_family(family),
    )


def candidate_conditions_for_hypothesis(
    hypothesis: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Build candidate single-step / short-sequence conditions for a hypothesis.

    The candidate matrix is taken directly from the M2.G1 hypothesis
    ``candidate_sequences`` (which already encode single steps as length-1
    sequences and short sequences as length-2 sequences).
    """
    sequences = _normalized_candidate_sequences(hypothesis)
    conditions: list[Dict[str, Any]] = []
    seen: set[Tuple[str, ...]] = set()
    for sequence in sequences:
        key = tuple(sequence)
        if not key or key in seen:
            continue
        seen.add(key)
        candidate_len = len(key)
        conditions.append(
            {
                "condition_id": "candidate_" + "_".join(key),
                "condition_kind": "candidate",
                "condition_family": (
                    "single_step" if candidate_len == 1 else "short_sequence"
                ),
                "action_or_sequence": list(key),
                "candidate_len": candidate_len,
                "post_stop_horizon": candidate_len,
                "role": "candidate_objective_conversion",
            }
        )
    return tuple(conditions)


def control_conditions_for_candidates(
    candidate_conditions: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Build the two controls, with relation_progress_policy horizon-matched.

    ``relation_progress_policy`` is replicated once per distinct candidate
    horizon so that each candidate is compared against a relation-progress
    continuation of the *same* length.
    """
    horizons = sorted(
        {
            int(condition.get("post_stop_horizon", 1) or 1)
            for condition in candidate_conditions
        }
    ) or [1]
    controls: list[Dict[str, Any]] = [
        {
            "condition_id": HOLD_OR_STOP_STATE_CONTROL,
            "condition_kind": "control",
            "condition_family": "hold_or_stop_state",
            "action_or_sequence": [],
            "post_stop_horizon": 0,
            "role": "zero_action_control",
            "semantics": "observe_safe_stop_endpoint_without_extra_action",
        }
    ]
    for horizon in horizons:
        controls.append(
            {
                "condition_id": f"{RELATION_PROGRESS_POLICY_CONTROL}_h{horizon}",
                "condition_kind": "control",
                "condition_family": "relation_progress_policy",
                "action_or_sequence": None,
                "post_stop_horizon": int(horizon),
                "role": "relation_progress_continuation_control",
                "policy": "abstract_mechanic_policy_p3g0_style",
                "horizon_matched_to_candidate_len": int(horizon),
            }
        )
    return tuple(controls)


def falsification_criteria_for_hypothesis(
    hypothesis: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    source = dict(hypothesis.get("falsification", {}) or {})
    criteria: list[Dict[str, Any]] = []
    source_metric = str(source.get("metric", ""))
    if source_metric:
        criteria.append(
            {
                "metric": source_metric,
                "support_condition": str(source.get("support_condition", "")),
                "failure_condition": str(source.get("failure_condition", "")),
                "minimum_effect_size": source.get("minimum_effect_size", 1),
                "source": "m2_g1_hypothesis",
            }
        )
    criteria.append(
        {
            "metric": DELTA_VS_HOLD_METRIC,
            "support_condition": (
                "candidate_delta_terminal_adjusted_progress_vs_hold > 0 "
                "and candidate_terminal_reentry == false"
            ),
            "failure_condition": (
                "candidate_delta_terminal_adjusted_progress_vs_hold <= 0 "
                "or candidate_terminal_reentry == true"
            ),
            "minimum_effect_size": 1,
            "source": "m3_g1_central_decision_guard",
        }
    )
    return tuple(criteria)


def planning_rationale_for_family(family: str) -> str:
    if family == "post_safe_stop_objective_conversion":
        return (
            "From a replayed P3.G1 safe-stop state, test whether a single post-stop "
            "action raises terminal-adjusted progress over hold_or_stop_state without "
            "terminal re-entry."
        )
    if family == "subgoal_target_reselection":
        return (
            "From the safe-stop state, test whether candidate actions targeting a "
            "non-closest relation beat generic relation-progress continuation."
        )
    if family == "objective_readiness_condition":
        return (
            "From the safe-stop state, test whether conversion actions only help "
            "when an objective-readiness signature is present, vs hold and relation "
            "progress controls."
        )
    if family == "terminal_safe_sequence_search":
        return (
            "From the safe-stop state, test whether a short post-stop sequence "
            "converts better than single steps and horizon-matched relation progress "
            "without terminal re-entry."
        )
    return "Objective-conversion controlled experiment for M2.G1 hypothesis."


def default_safe_stop_spec() -> Dict[str, Any]:
    return {
        "source": SAFE_STOP_STATE_SOURCE,
        "policy_condition": SAFE_STOP_POLICY_CONDITION,
        "stop_trigger_reason": SAFE_STOP_STOP_TRIGGER_REASON,
        "base_state_family": SAFE_STOP_BASE_STATE_FAMILY,
    }


def summarize_objective_conversion_requests(
    *,
    hypotheses: Sequence[Mapping[str, Any]],
    requests: Sequence[ObjectiveConversionExperimentRequest],
    skipped: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    requested_families = sorted(
        {request.hypothesis_family for request in requests}
    )
    candidate_cells = sum(len(request.candidate_conditions) for request in requests)
    return {
        "objective_conversion_hypotheses_consumed": len(hypotheses),
        "objective_conversion_experiment_requests_generated": len(requests),
        "skipped_objective_conversion_hypotheses": len(skipped),
        "covered_hypothesis_families": requested_families,
        "all_four_families_covered": set(requested_families)
        >= {
            "post_safe_stop_objective_conversion",
            "subgoal_target_reselection",
            "objective_readiness_condition",
            "terminal_safe_sequence_search",
        },
        "candidate_conditions_total": candidate_cells,
        "controls_per_request": [
            HOLD_OR_STOP_STATE_CONTROL,
            RELATION_PROGRESS_POLICY_CONTROL,
        ],
        "relation_progress_policy_horizon_rule": "post_stop_horizon = candidate_len",
        "primary_metrics": list(PRIMARY_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "central_decision_rule": CENTRAL_DECISION_RULE,
        "secondary_decision_rule": SECONDARY_DECISION_RULE,
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "objective_conversion_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def objective_conversion_hypotheses_from_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for batch in payload.get("objective_conversion_hypothesis_batches", []) or []:
        if not isinstance(batch, Mapping):
            continue
        for hypothesis in batch.get("candidate_hypotheses", []) or []:
            if isinstance(hypothesis, Mapping):
                rows.append(dict(hypothesis))
    return tuple(rows)


def is_ready_objective_conversion_hypothesis(hypothesis: Mapping[str, Any]) -> bool:
    return bool(
        str(hypothesis.get("status", "")) == "UNRESOLVED"
        and int(hypothesis.get("support", 0) or 0) == 0
        and str(hypothesis.get("revision_status", "")) == "CANDIDATE_ONLY"
        and str(hypothesis.get("truth_status", "")) == "NOT_EVALUATED_BY_M2"
        and not bool(hypothesis.get("revision_performed", False))
        and int(hypothesis.get("wrong_confirmations", 0) or 0) == 0
        and bool(hypothesis.get("ready_for_m3_candidate_experiment_request", False))
        and bool((hypothesis.get("testability", {}) or {}).get("testable", False))
        and bool(_normalized_candidate_sequences(hypothesis))
    )


def skipped_objective_conversion_hypothesis(
    hypothesis: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "source_hypothesis_id": str(hypothesis.get("hypothesis_id", "")),
        "hypothesis_family": str(hypothesis.get("hypothesis_family", "")),
        "reason": "not_ready_for_m3_objective_conversion_experiment",
        "status": BLOCKED_OBJECTIVE_CONVERSION_EXPERIMENT,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def validate_objective_conversion_experiment_request(
    request: ObjectiveConversionExperimentRequest | Mapping[str, Any],
) -> None:
    data = (
        request.to_dict()
        if isinstance(request, ObjectiveConversionExperimentRequest)
        else dict(request)
    )
    if not str(data.get("request_id", "")):
        raise ValueError("request_id is required")
    if not str(data.get("source_hypothesis_id", "")):
        raise ValueError("source_hypothesis_id is required")
    if not data.get("candidate_conditions"):
        raise ValueError("candidate_conditions are required")
    controls = {
        str(control.get("condition_family", ""))
        for control in data.get("control_conditions", []) or []
    }
    if HOLD_OR_STOP_STATE_CONTROL not in controls:
        raise ValueError("hold_or_stop_state control is required")
    if RELATION_PROGRESS_POLICY_CONTROL not in controls:
        raise ValueError("relation_progress_policy control is required")
    if str(data.get("status", "")) != READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT:
        raise ValueError("request must be ready for M3 objective-conversion experiment")
    if str(data.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("request must remain candidate-only")
    if int(data.get("support", 0) or 0) != 0:
        raise ValueError("request support must remain 0")
    if str(data.get("truth_status", "")) != M3_REFINEMENT_TRUTH_STATUS:
        raise ValueError("request truth_status must remain M3-local")
    if bool(data.get("execution_performed", False)):
        raise ValueError("planner cannot execute requests")
    if bool(data.get("revision_performed", False)):
        raise ValueError("request revision_performed must be false")
    if int(data.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("request wrong_confirmations must remain 0")
    if bool(data.get("m2_hypothesis_counted_as_confirmation", False)):
        raise ValueError("request cannot count M2 hypothesis as confirmation")
    if bool(data.get("objective_conversion_request_counted_as_support", False)):
        raise ValueError("request cannot count as support")
    if bool(data.get("experiment_result_counted_as_scientific_verdict", False)):
        raise ValueError("experiment result cannot count as scientific verdict")
    if bool(data.get("a32_write_performed", False)) or bool(
        data.get("a33_write_performed", False)
    ):
        raise ValueError("planner must not write A32/A33")


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(summary.get("revision_performed", False)) or bool(
        payload.get("revision_performed", False)
    ):
        raise ValueError("source revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source summary wrong_confirmations must remain 0")
    if bool(summary.get("execution_performed", False)) or bool(
        payload.get("execution_performed", False)
    ):
        raise ValueError("M2.G1 source must be planning-only")
    if bool(
        payload.get("objective_conversion_hypotheses_counted_as_confirmation", False)
    ):
        raise ValueError("objective-conversion hypotheses cannot count as confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")
    for key in ("a32_write_performed", "a33_write_performed", "m3_write_performed"):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"source must not have {key}")


def _normalized_candidate_sequences(
    hypothesis: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], ...]:
    sequences = hypothesis.get("candidate_sequences")
    result: list[Tuple[str, ...]] = []
    if isinstance(sequences, Sequence) and not isinstance(sequences, (str, bytes)):
        for sequence in sequences:
            if isinstance(sequence, Sequence) and not isinstance(
                sequence, (str, bytes)
            ):
                actions = tuple(str(action) for action in sequence if action)
                if actions:
                    result.append(actions)
    if result:
        return tuple(result)
    candidate_action = str(hypothesis.get("candidate_action", "") or "")
    if candidate_action:
        return ((candidate_action,),)
    return tuple()


def objective_conversion_request_id(hypothesis: Mapping[str, Any]) -> str:
    source = str(hypothesis.get("hypothesis_id", "")).replace("::", "_")
    family = str(hypothesis.get("hypothesis_family", "objective_conversion"))
    return f"m3_g1::{source}::{family}"


def write_objective_conversion_experiment_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH
    ),
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
        description="Build M3.G1.1 objective-conversion experiment requests.",
    )
    parser.add_argument(
        "--hypotheses",
        type=Path,
        default=DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_experiment_planning(
        objective_conversion_hypotheses_path=args.hypotheses,
    )
    write_objective_conversion_experiment_requests(payload, args.out)
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
