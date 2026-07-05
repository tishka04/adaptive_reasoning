"""Strict validators for M2 artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from .schema import (
    M2_BLOCKED_FOR_M3_STATUS,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_HYPOTHESIS_STATUS,
    M2_READY_FOR_M3_STATUS,
    M2_TRUTH_STATUS,
    ValidationResult,
)


def validate_testability(value: Any) -> ValidationResult:
    data = _data(value)
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ValidationResult(False, ("not_mapping",))

    testable = bool(data.get("testable", False))
    if testable and not str(data.get("metric", "")):
        errors.append("testable_missing_metric")
    if testable and not str(data.get("target_action", "")):
        errors.append("testable_missing_target_action")
    if testable and not str(data.get("expected_signal_type", "")):
        errors.append("testable_missing_expected_signal_type")
    if str(data.get("control_policy", "")) not in {
        M2_DYNAMIC_CONTROL_POLICY,
        "fixed_suggested_controls",
    }:
        errors.append("invalid_control_policy")
    return ValidationResult(not errors, tuple(errors))


def validate_hypothesis(value: Any) -> ValidationResult:
    data = _data(value)
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ValidationResult(False, ("not_mapping",))

    for field in (
        "hypothesis_id",
        "source_request_id",
        "game_id",
        "frontier_context_id",
        "frontier_reason",
        "hypothesis_family",
        "candidate_action",
        "predicted_metric",
        "predicted_effect",
        "rationale",
    ):
        if not str(data.get(field, "")):
            errors.append(f"missing_{field}")

    if str(data.get("status", "")) != M2_HYPOTHESIS_STATUS:
        errors.append("status_must_be_unresolved")
    if int(data.get("support", -1) or 0) != 0:
        errors.append("support_must_be_zero")
    if bool(data.get("controlled_test_required")) is not True:
        errors.append("controlled_test_required_must_be_true")
    if str(data.get("truth_status", "")) != M2_TRUTH_STATUS:
        errors.append("truth_status_must_be_not_evaluated_by_m2")
    if bool(data.get("revision_performed")):
        errors.append("revision_performed_must_be_false")
    if int(data.get("wrong_confirmations", 0) or 0) != 0:
        errors.append("wrong_confirmations_must_be_zero")
    if bool(data.get("trace_support_counted_as_proof")):
        errors.append("trace_support_counted_as_proof_must_be_false")
    if bool(data.get("prior_counted_as_proof")):
        errors.append("prior_counted_as_proof_must_be_false")

    falsification = data.get("falsification") or {}
    if not isinstance(falsification, Mapping):
        errors.append("missing_falsification_criterion")
    else:
        if not str(falsification.get("metric", "")):
            errors.append("falsification_missing_metric")
        if not str(falsification.get("support_condition", "")):
            errors.append("falsification_missing_support_condition")
        if not str(falsification.get("failure_condition", "")):
            errors.append("falsification_missing_failure_condition")

    testability = data.get("testability") or {}
    testability_result = validate_testability(testability)
    errors.extend(testability_result.errors)

    audit = data.get("source_generation") or {}
    if not isinstance(audit, Mapping):
        errors.append("missing_source_generation")
    else:
        if bool(audit.get("priority_score_counted_as_support", True)):
            errors.append("priority_score_counted_as_support_must_be_false")

    return ValidationResult(not errors, tuple(errors))


def validate_m3_request(value: Any) -> ValidationResult:
    data = _data(value)
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ValidationResult(False, ("not_mapping",))

    for field in ("request_id", "source_hypothesis_id", "game_id", "target_action"):
        if not str(data.get(field, "")):
            errors.append(f"missing_{field}")
    if str(data.get("metric", "")) == "":
        errors.append("missing_metric")
    if str(data.get("expected_signal", "")) == "":
        errors.append("missing_expected_signal")
    if str(data.get("control_policy", "")) != M2_DYNAMIC_CONTROL_POLICY:
        errors.append("control_policy_must_be_m3_dynamic")
    if str(data.get("status", "")) not in {
        M2_READY_FOR_M3_STATUS,
        M2_BLOCKED_FOR_M3_STATUS,
    }:
        errors.append("invalid_status")
    if str(data.get("truth_status", "")) != M2_TRUTH_STATUS:
        errors.append("truth_status_must_be_not_evaluated_by_m2")
    if int(data.get("support", -1) or 0) != 0:
        errors.append("support_must_be_zero")
    if bool(data.get("controlled_test_required")) is not True:
        errors.append("controlled_test_required_must_be_true")
    if bool(data.get("revision_performed")):
        errors.append("revision_performed_must_be_false")
    if int(data.get("wrong_confirmations", 0) or 0) != 0:
        errors.append("wrong_confirmations_must_be_zero")

    falsification = data.get("falsification_criterion") or {}
    if not isinstance(falsification, Mapping):
        errors.append("missing_falsification_criterion")
    elif str(falsification.get("metric", "")) == "":
        errors.append("falsification_missing_metric")

    return ValidationResult(not errors, tuple(errors))


def _data(value: Any) -> Any:
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return asdict(value)
    return value
