"""SAGE.5j executor for the A32.4 pre-registered control protocol."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from theory.a32.unknown_game_control_protocol_decisions import (
    A32_4_PROTOCOL_AUTHORIZED,
    A32_4_SCHEMA_VERSION,
    AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL,
    DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH,
)
from theory.m1.controlled_followup_experiment import controlled_delta

from .distributed_live_mini_frontier_generation import (
    DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH,
)
from .live_mini_frontier_m3_executor import _execute_request_arm


DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH = (
    Path("diagnostics") / "sage" / "sage5j_parameterized_control_acquisition.json"
)
SAGE5J_SCHEMA_VERSION = "sage.parameterized_control_acquisition.v1"
SAGE5J_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5J"

SAGE5J_ALL_DISCRIMINATING = (
    "SAGE_PARAMETERIZED_CONTROL_ACQUISITION_ALL_DISCRIMINATING_CANDIDATE_ONLY"
)
SAGE5J_MIXED_DISCRIMINATION = (
    "SAGE_PARAMETERIZED_CONTROL_ACQUISITION_MIXED_CANDIDATE_ONLY"
)
SAGE5J_NO_DISCRIMINATION = (
    "SAGE_PARAMETERIZED_CONTROL_ACQUISITION_NO_DISCRIMINATION_CANDIDATE_ONLY"
)
SAGE5J_EXECUTION_INCOMPLETE = (
    "SAGE_PARAMETERIZED_CONTROL_ACQUISITION_INCOMPLETE_CANDIDATE_ONLY"
)

DISCRIMINATING_TARGET_EFFECT = "DISCRIMINATING_TARGET_EFFECT_CANDIDATE_ONLY"
NON_DISCRIMINATING_EQUAL_EFFECT = "NON_DISCRIMINATING_EQUAL_EFFECT_CANDIDATE_ONLY"
CONTROL_EXCEEDS_TARGET_EFFECT = "CONTROL_EXCEEDS_TARGET_EFFECT_CANDIDATE_ONLY"

CANDIDATE_DISCRIMINATING = "PARAMETERIZED_CONTROLS_DISCRIMINATING_CANDIDATE_ONLY"
CANDIDATE_NON_DISCRIMINATING = (
    "PARAMETERIZED_CONTROLS_NON_DISCRIMINATING_CANDIDATE_ONLY"
)
CANDIDATE_CONTROL_EXCEEDS = "PARAMETERIZED_CONTROL_EXCEEDS_TARGET_CANDIDATE_ONLY"
CANDIDATE_MIXED = "PARAMETERIZED_CONTROL_RESPONSE_MIXED_CANDIDATE_ONLY"
CANDIDATE_INCOMPLETE = "PARAMETERIZED_CONTROL_EXECUTION_INCOMPLETE_CANDIDATE_ONLY"

EnvFactory = Callable[[str], Any]


def run_sage5j_parameterized_control_acquisition(
    *,
    source_a32_4_path: str | Path = (
        DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH
    ),
    source_sage5e_path: str | Path = (
        DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH
    ),
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute exactly the eight paired experiments pre-registered by A32.4."""
    source_a32 = _load_json(source_a32_4_path)
    source_sage5e = _load_json(source_sage5e_path)
    validate_sage5j_sources(source_a32, source_sage5e)

    source_requests = {
        str(row.get("request_id", "")): dict(row)
        for row in source_sage5e.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping) and str(row.get("request_id", ""))
    }
    protocol_experiments = [
        dict(row)
        for row in source_a32.get("requested_followup_experiments", []) or []
        if isinstance(row, Mapping)
    ]
    executed: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for protocol in protocol_experiments:
        result = execute_preregistered_parameterized_control_experiment(
            protocol=protocol,
            source_request=source_requests[str(protocol.get("source_request_id", ""))],
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        if str(result.get("execution_status", "")) == "EXECUTED":
            executed.append(result)
        else:
            blocked.append(result)

    candidate_assessments = build_candidate_protocol_assessments(
        source_a32=source_a32,
        executed=executed,
        blocked=blocked,
    )
    summary = summarize_sage5j(
        source_a32=source_a32,
        requested=protocol_experiments,
        executed=executed,
        blocked=blocked,
        candidate_assessments=candidate_assessments,
    )
    payload = {
        "config": {
            "schema_version": SAGE5J_SCHEMA_VERSION,
            "source_a32_4_path": str(source_a32_4_path),
            "source_sage5e_path": str(source_sage5e_path),
            "environments_dir": (
                str(environments_dir) if environments_dir is not None else None
            ),
            "inputs_read": ["A32.4", "SAGE.5e"],
            "execution_mode": "a32_pre_registered_live_prefix_paired_control",
            "execution_policy": {
                "execute_only_pre_registered_experiments": True,
                "exact_context_replay_required": True,
                "exact_target_action_args_required": True,
                "exact_control_action_args_required": True,
                "same_measurement_for_both_arms_required": True,
                "variant_substitution_allowed": False,
                "context_substitution_allowed": False,
                "parameterized_controls_are_not_action_distinct_controls": True,
                "results_require_later_a32_review": True,
            },
            "artifacts_not_modified": ["A32.4", "M2", "M3", "A33", "A40", "P2"],
        },
        "source_a32_4_summary": dict(source_a32.get("summary", {}) or {}),
        "pre_registered_experiments": protocol_experiments,
        "executed_parameterized_control_experiments": executed,
        "blocked_parameterized_control_experiments": blocked,
        "candidate_protocol_assessments": candidate_assessments,
        "summary": summary,
        "comparison": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5J_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": True,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "parameterized_control_events_counted_as_scientific_support": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "neutral_events_counted_as_refutation": False,
        "protocol_result_counted_as_a32_decision": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5j_parameterized_control_acquisition(payload, output_path)
    return payload


def execute_preregistered_parameterized_control_experiment(
    *,
    protocol: Mapping[str, Any],
    source_request: Mapping[str, Any],
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    request = copy.deepcopy(dict(source_request))
    request["metric"] = str(protocol.get("measurement", ""))
    request["target_action"] = str(protocol.get("target_action", ""))
    request["target_action_args"] = copy.deepcopy(protocol.get("target_action_args"))

    target = _execute_request_arm(
        request,
        action_name=str(protocol.get("target_action", "")),
        action_args=dict(protocol.get("target_action_args", {}) or {}),
        arm="pre_registered_target",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if str(target.get("status", "")) != "EXECUTED":
        return _blocked_protocol_result(
            protocol,
            reason=str(target.get("reason", "target_execution_blocked")),
            target_arm=target,
        )

    control = _execute_request_arm(
        request,
        action_name=str(protocol.get("control_action", "")),
        action_args=dict(protocol.get("control_action_args", {}) or {}),
        arm="pre_registered_parameterized_control",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if str(control.get("status", "")) != "EXECUTED":
        return _blocked_protocol_result(
            protocol,
            reason=str(control.get("reason", "control_execution_blocked")),
            target_arm=target,
            control_arm=control,
        )

    substitutions = protocol_substitution_reasons(
        protocol=protocol,
        request=request,
        target_arm=target,
        control_arm=control,
    )
    if substitutions:
        return _blocked_protocol_result(
            protocol,
            reason="protocol_substitution_detected:" + ",".join(substitutions),
            target_arm=target,
            control_arm=control,
        )

    delta = controlled_delta(
        control["measurement_for_delta"],
        target["measurement_for_delta"],
        predicted_metric=str(protocol.get("measurement", "")),
    )
    raw_support, raw_contradiction = _event_counts_from_delta(delta)
    raw_neutral = 1 if raw_support == 0 and raw_contradiction == 0 else 0
    discrimination_status = discrimination_status_from_delta(delta)
    target_signature = measurement_effect_signature(target["measurement"])
    control_signature = measurement_effect_signature(control["measurement"])
    return {
        "execution_status": "EXECUTED",
        "protocol_experiment_id": str(protocol.get("experiment_id", "")),
        "candidate_id": str(protocol.get("candidate_id", "")),
        "candidate_key": str(protocol.get("candidate_key", "")),
        "source_request_id": str(protocol.get("source_request_id", "")),
        "source_audit_id": str(protocol.get("source_audit_id", "")),
        "source_transition_id": str(request.get("source_transition_id", "")),
        "game_id": str(request.get("game_id", "")),
        "budget": int(protocol.get("budget", 0) or 0),
        "variant_index": int(protocol.get("variant_index", 0) or 0),
        "measurement": str(protocol.get("measurement", "")),
        "target_action": str(protocol.get("target_action", "")),
        "target_action_args": protocol.get("target_action_args"),
        "control_action": str(protocol.get("control_action", "")),
        "control_action_args": dict(protocol.get("control_action_args", {}) or {}),
        "context_snapshot_hash": str(protocol.get("context_snapshot_hash", "")),
        "target_context_signature_verified": bool(
            target.get("context_snapshot_hash_verified", False)
        ),
        "control_context_signature_verified": bool(
            control.get("context_snapshot_hash_verified", False)
        ),
        "live_prefix_replay_exact": bool(
            target.get("context_snapshot_hash_verified", False)
            and control.get("context_snapshot_hash_verified", False)
        ),
        "protocol_exact_match": True,
        "protocol_substitution_detected": False,
        "target_measurement": dict(target["measurement"]),
        "control_measurement": dict(control["measurement"]),
        "target_effect_signature": target_signature,
        "control_effect_signature": control_signature,
        "paired_effect_signatures_equal": target_signature == control_signature,
        "target_signal": float(target["measurement_for_delta"]["local_changed_pixels"]),
        "control_signal": float(
            control["measurement_for_delta"]["local_changed_pixels"]
        ),
        "controlled_delta": dict(delta),
        "discrimination_status": discrimination_status,
        "support_events": raw_support,
        "contradiction_events": raw_contradiction,
        "neutral_events": raw_neutral,
        "support": 0,
        "truth_status": SAGE5J_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "parameterized_control_event_counted_as_scientific_support": False,
        "parameterized_control_counted_as_distinct_action": False,
        "neutral_event_counted_as_refutation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def protocol_substitution_reasons(
    *,
    protocol: Mapping[str, Any],
    request: Mapping[str, Any],
    target_arm: Mapping[str, Any],
    control_arm: Mapping[str, Any],
) -> Tuple[str, ...]:
    reasons: List[str] = []
    if str(request.get("context_snapshot_hash", "")) != str(
        protocol.get("context_snapshot_hash", "")
    ):
        reasons.append("context_snapshot_hash")
    if str(request.get("metric", "")) != str(protocol.get("measurement", "")):
        reasons.append("measurement")
    if str(target_arm.get("action", "")) != str(protocol.get("target_action", "")):
        reasons.append("target_action")
    if _canonical_json(target_arm.get("action_args", {}) or {}) != _canonical_json(
        protocol.get("target_action_args", {}) or {}
    ):
        reasons.append("target_action_args")
    if str(control_arm.get("action", "")) != str(protocol.get("control_action", "")):
        reasons.append("control_action")
    if _canonical_json(control_arm.get("action_args", {}) or {}) != _canonical_json(
        protocol.get("control_action_args", {}) or {}
    ):
        reasons.append("control_action_args")
    return tuple(reasons)


def measurement_effect_signature(measurement: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "metric": str(measurement.get("metric", "")),
        "changed": bool(measurement.get("changed", False)),
        "changed_pixels": int(measurement.get("changed_pixels", 0) or 0),
        "local_patch_available": bool(measurement.get("local_patch_available", False)),
        "local_changed_pixels": int(measurement.get("local_changed_pixels", 0) or 0),
        "observed_signal": float(measurement.get("observed_signal", 0) or 0),
        "observed_signal_source": str(measurement.get("observed_signal_source", "")),
    }


def discrimination_status_from_delta(delta: Mapping[str, Any]) -> str:
    effect = float(delta.get("effect_size", 0.0) or 0.0)
    if effect > 0:
        return DISCRIMINATING_TARGET_EFFECT
    if effect < 0:
        return CONTROL_EXCEEDS_TARGET_EFFECT
    return NON_DISCRIMINATING_EQUAL_EFFECT


def _event_counts_from_delta(delta: Mapping[str, Any]) -> Tuple[int, int]:
    effect = float(delta.get("effect_size", 0.0) or 0.0)
    if effect > 0:
        return 1, 0
    if effect < 0:
        return 0, 1
    return 0, 0


def build_candidate_protocol_assessments(
    *,
    source_a32: Mapping[str, Any],
    executed: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    assessments: List[Dict[str, Any]] = []
    for decision in source_a32.get("protocol_decisions", []) or []:
        if not isinstance(decision, Mapping):
            continue
        candidate_id = str(decision.get("candidate_id", ""))
        candidate_executed = [
            row for row in executed if str(row.get("candidate_id", "")) == candidate_id
        ]
        candidate_blocked = [
            row for row in blocked if str(row.get("candidate_id", "")) == candidate_id
        ]
        requested = list(decision.get("preregistered_experiments", []) or [])
        exact = [
            row
            for row in candidate_executed
            if bool(row.get("live_prefix_replay_exact", False))
            and bool(row.get("protocol_exact_match", False))
        ]
        variant_counts = Counter(
            _variant_key(row) for row in exact if str(row.get("control_action", ""))
        )
        expected_variant_keys = {
            _variant_key(row)
            for row in decision.get("authorized_control_variants", []) or []
            if isinstance(row, Mapping)
        }
        variant_replication_complete = bool(expected_variant_keys) and all(
            variant_counts.get(key, 0) >= 2 for key in expected_variant_keys
        )
        complete = (
            bool(requested) and len(exact) == len(requested) and not candidate_blocked
        )
        support_events = sum(int(row.get("support_events", 0) or 0) for row in exact)
        contradiction_events = sum(
            int(row.get("contradiction_events", 0) or 0) for row in exact
        )
        neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in exact)
        result = candidate_protocol_result(
            complete=complete and variant_replication_complete,
            experiment_count=len(exact),
            support_events=support_events,
            contradiction_events=contradiction_events,
            neutral_events=neutral_events,
        )
        assessments.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": str(decision.get("candidate_key", "")),
                "game_id": str(decision.get("game_id", "")),
                "action": str(decision.get("action", "")),
                "action_args": decision.get("action_args"),
                "protocol_decision": str(decision.get("decision", "")),
                "requested_experiments": len(requested),
                "executed_experiments": len(candidate_executed),
                "blocked_experiments": len(candidate_blocked),
                "replay_exact_protocol_matches": len(exact),
                "distinct_parameterized_interventions_executed": len(variant_counts),
                "distinct_control_action_names_executed": len(
                    {
                        str(row.get("control_action", ""))
                        for row in exact
                        if str(row.get("control_action", ""))
                    }
                ),
                "variant_replication_counts": dict(sorted(variant_counts.items())),
                "variant_replication_complete": variant_replication_complete,
                "protocol_execution_complete": complete,
                "raw_support_events": support_events,
                "raw_contradiction_events": contradiction_events,
                "raw_neutral_events": neutral_events,
                "parameterized_protocol_result": result,
                "ready_for_A32_protocol_result_review": complete
                and variant_replication_complete,
                "scientific_status": "UNRESOLVED",
                "support": 0,
                "truth_status": SAGE5J_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "parameterized_interventions_counted_as_distinct_actions": False,
                "protocol_events_counted_as_scientific_support": False,
                "neutral_events_counted_as_refutation": False,
                "candidate_assessment_counted_as_a32_decision": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return assessments


def candidate_protocol_result(
    *,
    complete: bool,
    experiment_count: int,
    support_events: int,
    contradiction_events: int,
    neutral_events: int,
) -> str:
    if not complete:
        return CANDIDATE_INCOMPLETE
    if contradiction_events > 0:
        return CANDIDATE_CONTROL_EXCEEDS
    if experiment_count > 0 and support_events == experiment_count:
        return CANDIDATE_DISCRIMINATING
    if experiment_count > 0 and neutral_events == experiment_count:
        return CANDIDATE_NON_DISCRIMINATING
    return CANDIDATE_MIXED


def summarize_sage5j(
    *,
    source_a32: Mapping[str, Any],
    requested: Sequence[Mapping[str, Any]],
    executed: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    candidate_assessments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    complete = [
        row
        for row in candidate_assessments
        if bool(row.get("protocol_execution_complete", False))
        and bool(row.get("variant_replication_complete", False))
    ]
    discriminating = [
        row
        for row in complete
        if str(row.get("parameterized_protocol_result", "")) == CANDIDATE_DISCRIMINATING
    ]
    non_discriminating = [
        row
        for row in complete
        if str(row.get("parameterized_protocol_result", ""))
        == CANDIDATE_NON_DISCRIMINATING
    ]
    control_exceeds = [
        row
        for row in complete
        if str(row.get("parameterized_protocol_result", ""))
        == CANDIDATE_CONTROL_EXCEEDS
    ]
    if len(complete) != len(candidate_assessments) or blocked:
        outcome_status = SAGE5J_EXECUTION_INCOMPLETE
    elif len(discriminating) == len(complete) and complete:
        outcome_status = SAGE5J_ALL_DISCRIMINATING
    elif len(non_discriminating) == len(complete) and complete:
        outcome_status = SAGE5J_NO_DISCRIMINATION
    else:
        outcome_status = SAGE5J_MIXED_DISCRIMINATION
    exact = [
        row
        for row in executed
        if bool(row.get("live_prefix_replay_exact", False))
        and bool(row.get("protocol_exact_match", False))
        and not bool(row.get("protocol_substitution_detected", False))
    ]
    gate_passed = bool(requested) and len(exact) == len(requested) and not blocked
    return {
        "source_a32_4_outcome_status": str(source_a32.get("outcome_status", "")),
        "source_protocol_decisions": len(
            source_a32.get("protocol_decisions", []) or []
        ),
        "pre_registered_experiments_consumed": len(requested),
        "experiments_executed": len(executed),
        "experiments_blocked": len(blocked),
        "live_prefix_replay_exact_experiments": sum(
            bool(row.get("live_prefix_replay_exact", False)) for row in executed
        ),
        "protocol_exact_match_experiments": len(exact),
        "protocol_substitutions_detected": sum(
            bool(row.get("protocol_substitution_detected", False))
            for row in [*executed, *blocked]
        ),
        "target_control_pairs_executed": len(executed),
        "parameterized_variants_executed": len({_variant_key(row) for row in executed}),
        "variant_replications_completed": sum(
            len(row.get("variant_replication_counts", {}) or {})
            for row in candidate_assessments
            if bool(row.get("variant_replication_complete", False))
        ),
        "raw_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in executed
        ),
        "raw_contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in executed
        ),
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in executed
        ),
        "candidates_evaluated": len(candidate_assessments),
        "candidate_protocols_complete": len(complete),
        "candidates_with_discriminating_parameterized_controls": len(discriminating),
        "candidates_with_non_discriminating_parameterized_controls": len(
            non_discriminating
        ),
        "candidates_with_control_exceeding_target": len(control_exceeds),
        "candidates_ready_for_A32_protocol_result_review": sum(
            bool(row.get("ready_for_A32_protocol_result_review", False))
            for row in candidate_assessments
        ),
        "all_pre_registered_experiments_resolved": (
            len(executed) + len(blocked) == len(requested)
        ),
        "all_pre_registered_experiments_executed_exactly": gate_passed,
        "gate_passed": gate_passed,
        "outcome_status": outcome_status,
        "support": 0,
        "truth_status": SAGE5J_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "parameterized_control_events_counted_as_scientific_support": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "neutral_events_counted_as_refutation": False,
        "protocol_result_counted_as_a32_decision": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_sage5j_sources(
    source_a32: Mapping[str, Any],
    source_sage5e: Mapping[str, Any],
) -> None:
    validate_a32_4_protocol_source(source_a32)
    validate_sage5e_execution_source(source_sage5e)

    requests = {
        str(row.get("request_id", "")): row
        for row in source_sage5e.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping) and str(row.get("request_id", ""))
    }
    decisions = {
        str(row.get("candidate_id", "")): row
        for row in source_a32.get("protocol_decisions", []) or []
        if isinstance(row, Mapping) and str(row.get("candidate_id", ""))
    }
    experiments = [
        row
        for row in source_a32.get("requested_followup_experiments", []) or []
        if isinstance(row, Mapping)
    ]
    for row in experiments:
        request_id = str(row.get("source_request_id", ""))
        if request_id not in requests:
            raise ValueError(f"A32.4 source request is unavailable: {request_id}")
        request = requests[request_id]
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id not in decisions:
            raise ValueError("A32.4 experiment candidate is not authorized")
        decision = decisions[candidate_id]
        if str(request.get("game_id", "")) != str(decision.get("game_id", "")):
            raise ValueError("A32.4 experiment game does not match SAGE.5e")
        if str(request.get("target_action", "")) != str(row.get("target_action", "")):
            raise ValueError("A32.4 target action does not match SAGE.5e")
        if _canonical_json(
            request.get("target_action_args", {}) or {}
        ) != _canonical_json(row.get("target_action_args", {}) or {}):
            raise ValueError("A32.4 target args do not match SAGE.5e")
        if str(request.get("context_snapshot_hash", "")) != str(
            row.get("context_snapshot_hash", "")
        ):
            raise ValueError("A32.4 context hash does not match SAGE.5e")
        if str(request.get("metric", "")) != str(row.get("measurement", "")):
            raise ValueError("A32.4 measurement does not match SAGE.5e")
        if _budget_from_request(request) != int(row.get("budget", 0) or 0):
            raise ValueError("A32.4 budget does not match SAGE.5e")


def validate_a32_4_protocol_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != A32_4_SCHEMA_VERSION:
        raise ValueError("A32.4 schema version is not supported by SAGE.5j")
    if str(source.get("outcome_status", "")) != A32_4_PROTOCOL_AUTHORIZED:
        raise ValueError("A32.4 must authorize the parameterized protocol")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("A32.4 support must remain 0 before execution")
    if not bool(source.get("scientific_review_performed", False)):
        raise ValueError("A32.4 scientific review must be complete")
    if not bool(source.get("protocol_decision_performed", False)):
        raise ValueError("A32.4 protocol decision must be complete")
    if not bool(source.get("experimental_protocol_authorized", False)):
        raise ValueError("A32.4 experimental protocol must be authorized")
    if bool(source.get("execution_performed", False)):
        raise ValueError("A32.4 cannot pre-execute the SAGE.5j protocol")
    if bool(source.get("revision_performed", False)) or bool(
        source.get("confirmation_performed", False)
    ):
        raise ValueError("A32.4 cannot revise or confirm before SAGE.5j")
    if bool(source.get("refutation_performed", False)):
        raise ValueError("A32.4 cannot refute before SAGE.5j")
    if bool(source.get("a33_ready", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("A32.4 cannot make the candidates A33-ready")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("A32.4 wrong_confirmations must remain 0")
    if bool(source.get("parameterized_controls_counted_as_distinct_actions", False)):
        raise ValueError("A32.4 cannot relabel parameterized controls")
    if bool(
        source.get("parameterized_controls_counted_as_evidence_before_execution", False)
    ):
        raise ValueError("A32.4 cannot count unexecuted controls as evidence")
    if bool(source.get("bounded_exhaustion_counted_as_refutation", False)):
        raise ValueError("A32.4 bounded exhaustion cannot count as refutation")

    decisions = [
        row
        for row in source.get("protocol_decisions", []) or []
        if isinstance(row, Mapping)
    ]
    experiments = [
        row
        for row in source.get("requested_followup_experiments", []) or []
        if isinstance(row, Mapping)
    ]
    if len(decisions) != int(summary.get("protocol_decisions", 0) or 0):
        raise ValueError("A32.4 protocol decision count must be exact")
    if len(experiments) != int(
        summary.get("preregistered_followup_experiments", 0) or 0
    ):
        raise ValueError("A32.4 pre-registered experiment count must be exact")
    if not decisions or not experiments:
        raise ValueError("A32.4 must provide decisions and experiments")
    experiment_ids = [str(row.get("experiment_id", "")) for row in experiments]
    request_ids = [str(row.get("source_request_id", "")) for row in experiments]
    if "" in experiment_ids or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("A32.4 experiment ids must be unique")
    if "" in request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError("A32.4 source request ids must be unique")

    flattened = [
        dict(experiment)
        for decision in decisions
        for experiment in decision.get("preregistered_experiments", []) or []
        if isinstance(experiment, Mapping)
    ]
    if _canonical_json(flattened) != _canonical_json(experiments):
        raise ValueError("A32.4 flattened experiment plan must be immutable")
    for decision in decisions:
        if (
            str(decision.get("decision", ""))
            != AUTHORIZE_PARAMETERIZED_CONTROL_PROTOCOL
        ):
            raise ValueError("SAGE.5j executes only authorized A32.4 decisions")
        if bool(decision.get("revision_performed", False)) or bool(
            decision.get("confirmation_performed", False)
        ):
            raise ValueError("A32.4 candidate cannot be revised or confirmed")
        if bool(decision.get("refutation_performed", False)) or bool(
            decision.get("a33_ready", False)
        ):
            raise ValueError("A32.4 candidate must remain unresolved")
        if str((decision.get("decision_record", {}) or {}).get("status", "")) != (
            "unresolved"
        ):
            raise ValueError("A32.4 decision record must remain unresolved")
        candidate_experiments = list(
            decision.get("preregistered_experiments", []) or []
        )
        if len(candidate_experiments) != 4:
            raise ValueError("A32.4 must pre-register four experiments per candidate")
        variants = Counter(_variant_key(row) for row in candidate_experiments)
        if len(variants) != 2 or any(count != 2 for count in variants.values()):
            raise ValueError("A32.4 must replicate two variants twice per candidate")
        authorized_variants = {
            _variant_key(row)
            for row in decision.get("authorized_control_variants", []) or []
            if isinstance(row, Mapping)
        }
        if set(variants) != authorized_variants:
            raise ValueError("A32.4 experiments must use only authorized variants")
        if {int(row.get("budget", 0) or 0) for row in candidate_experiments} != {
            50,
            300,
        }:
            raise ValueError("A32.4 variants must span budgets 50 and 300")
        for variant in variants:
            variant_budgets = {
                int(row.get("budget", 0) or 0)
                for row in candidate_experiments
                if _variant_key(row) == variant
            }
            if variant_budgets != {50, 300}:
                raise ValueError("each A32.4 variant must span budgets 50 and 300")
    for row in experiments:
        if str(row.get("status", "")) != "PRE_REGISTERED_NOT_EXECUTED":
            raise ValueError("A32.4 experiment must remain unexecuted")
        if int(row.get("support", 0) or 0) != 0:
            raise ValueError("A32.4 experiment support must remain 0")
        if not bool(row.get("paired_target_control_required", False)):
            raise ValueError("A32.4 experiment must require paired arms")
        if not bool(row.get("exact_context_replay_required", False)):
            raise ValueError("A32.4 experiment must require exact replay")
        if not bool(row.get("same_measurement_for_target_and_control_required", False)):
            raise ValueError("A32.4 experiment must lock the measurement")
        if str(row.get("measurement", "")) != "local_patch_before_after":
            raise ValueError("A32.4 experiment measurement must remain pre-registered")
        if str(row.get("evaluation_rule", "")) != (
            "compare_paired_target_and_control_effect_signatures_"
            "without_post_hoc_metric_change"
        ):
            raise ValueError("A32.4 evaluation rule must remain pre-registered")
        if bool(row.get("parameterized_control_counted_as_distinct_action", False)):
            raise ValueError("A32.4 experiment cannot relabel its control")


def validate_sage5e_execution_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    if str(config.get("schema_version", "")) != (
        "sage.distributed_live_mini_frontier_generation.v1"
    ):
        raise ValueError("SAGE.5e schema version is not supported by SAGE.5j")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.5e must remain unresolved")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.5e must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.5e support must remain 0")
    if bool(source.get("revision_performed", False)):
        raise ValueError("SAGE.5e must not perform revision")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.5e wrong_confirmations must remain 0")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.5e cannot write A32/A33")


def _blocked_protocol_result(
    protocol: Mapping[str, Any],
    *,
    reason: str,
    target_arm: Mapping[str, Any] | None = None,
    control_arm: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "execution_status": "BLOCKED",
        "protocol_experiment_id": str(protocol.get("experiment_id", "")),
        "candidate_id": str(protocol.get("candidate_id", "")),
        "candidate_key": str(protocol.get("candidate_key", "")),
        "source_request_id": str(protocol.get("source_request_id", "")),
        "blocked_reason": reason,
        "target_arm": dict(target_arm or {}),
        "control_arm": dict(control_arm or {}),
        "protocol_exact_match": False,
        "protocol_substitution_detected": "substitution" in reason,
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 0,
        "support": 0,
        "truth_status": SAGE5J_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "parameterized_control_event_counted_as_scientific_support": False,
        "parameterized_control_counted_as_distinct_action": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _variant_key(row: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "action": str(row.get("control_action", row.get("action", ""))),
            "action_args": dict(
                row.get("control_action_args", row.get("action_args", {})) or {}
            ),
        }
    )


def _budget_from_request(request: Mapping[str, Any]) -> int:
    transition = str(request.get("source_transition_id", ""))
    marker = "::budget_"
    if marker not in transition:
        return 0
    return int(transition.split(marker, 1)[1].split("::", 1)[0].split("_", 1)[0])


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_sage5j_parameterized_control_acquisition(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH,
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
        description="Execute the A32.4 pre-registered parameterized controls.",
    )
    parser.add_argument(
        "--source-a32-4",
        default=str(DEFAULT_A32_UNKNOWN_GAME_CONTROL_PROTOCOL_DECISIONS_OUTPUT_PATH),
    )
    parser.add_argument(
        "--source-sage5e",
        default=str(DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH),
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_SAGE5J_PARAMETERIZED_CONTROL_ACQUISITION_PATH),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_sage5j_parameterized_control_acquisition(
        source_a32_4_path=args.source_a32_4,
        source_sage5e_path=args.source_sage5e,
        environments_dir=args.environments_dir,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
