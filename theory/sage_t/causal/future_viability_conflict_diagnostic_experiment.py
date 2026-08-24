"""Offline-only execution of the SAGE.T12.6.1a conflict diagnostic."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .future_viability_conflict_diagnostic import extract_conflict_sensitivities
from .future_viability_conflict_diagnostic_protocol import (
    FutureViabilityConflictDiagnosticProtocol,
    conflict_diagnostic_receipt,
    load_conflict_diagnostic_receipt,
    load_future_viability_conflict_diagnostic_manifest,
)
from .future_viability_hierarchy import evaluate_hierarchical_viability_ranking
from .future_viability_hierarchy_experiment import (
    _evaluation_checks,
    _load_model_bundle,
    _stratified,
)
from .future_viability_hierarchy_protocol import (
    FutureViabilityHierarchyProtocol,
    load_future_viability_hierarchy_manifest,
    load_hierarchy_receipt,
)
from .future_viability_protocol import _resolve_bound

_CORE_METRICS = (
    "binding_swap_hits",
    "binding_swap_top1_accuracy",
    "eligible_groups",
    "future_binding_hits",
    "future_binding_top1_accuracy",
    "future_gain_over_binding_swap",
    "future_gain_over_immediate",
    "future_gain_over_incumbent",
    "hierarchy_coverage",
    "immediate_binding_hits",
    "immediate_binding_top1_accuracy",
    "incumbent_binding_hits",
    "incumbent_binding_top1_accuracy",
    "incumbent_signature_coverage",
    "recommendation_accuracy",
    "recommendation_coverage",
    "recommendation_hits",
    "recommendations",
    "unique_top_rate",
)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _same_number(left: Any, right: Any, *, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= float(tolerance)
    return left == right


def _metrics_reproduced(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    tolerance: float,
) -> bool:
    return all(
        key in observed
        and key in expected
        and _same_number(observed[key], expected[key], tolerance=tolerance)
        for key in _CORE_METRICS
    )


def _strata_reproduced(
    observed: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float,
) -> bool:
    if set(observed) != set(expected):
        return False
    keys = (
        "eligible_groups",
        "future_binding_hits",
        "immediate_binding_hits",
        "incumbent_binding_hits",
        "binding_swap_hits",
    )
    return all(
        _same_number(
            observed[stratum][key],
            expected[stratum][key],
            tolerance=tolerance,
        )
        for stratum in observed
        for key in keys
    )


def _metric_ranges(
    policies: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    names = (
        "eligible_groups",
        "future_binding_top1_accuracy",
        "future_gain_over_immediate",
        "future_gain_over_binding_swap",
        "future_gain_over_incumbent",
    )
    return {
        name: {
            "maximum": max(float(value["metrics"][name]) for value in policies.values()),
            "minimum": min(float(value["metrics"][name]) for value in policies.values()),
        }
        for name in names
    }


def run_future_viability_conflict_diagnostic(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_conflict_diagnostic_manifest(
        manifest_path, root=repo_root
    )
    if not manifest["firewall"].get("diagnostic_authorized", False):
        raise ValueError("T12.6.1a manifest does not authorize diagnosis")
    protocol = FutureViabilityConflictDiagnosticProtocol(**dict(manifest["protocol"]))
    parent_manifest_path = _resolve_bound(
        str(manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    parent_manifest = load_future_viability_hierarchy_manifest(
        parent_manifest_path, root=repo_root, open_evaluation=True
    )
    parent_protocol = FutureViabilityHierarchyProtocol(
        **dict(parent_manifest["protocol"])
    )
    parent_receipt_path = _resolve_bound(
        str(manifest["parent"]["evaluation_receipt"]["path"]), root=repo_root
    )
    parent_receipt = load_hierarchy_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        expected_phase="evaluation",
    )
    model_path = _resolve_bound(
        str(manifest["parent"]["evaluation_artifacts"]["models"]["path"]),
        root=repo_root,
    )
    future_model, immediate_model, incumbent_model, bundle = _load_model_bundle(
        model_path, manifest=parent_manifest
    )

    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1a diagnostic: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    extraction = extract_conflict_sensitivities(
        archive_metas=manifest["inputs"]["evaluation_archives"],
        root=repo_root,
        corpus="evaluation",
        expected_search_seeds=parent_protocol.evaluation_search_seeds,
        expected_lineages=parent_protocol.source_lineages,
        expected_arms=parent_protocol.evaluation_arms,
        future_horizon=parent_protocol.future_horizon,
        local_radius=parent_protocol.local_radius,
        policies=protocol.consolidation_policies,
    )

    policy_results: dict[str, dict[str, Any]] = {}
    for policy in protocol.consolidation_policies:
        ranking = evaluate_hierarchical_viability_ranking(
            extraction.observations_by_policy[policy],
            future_model=future_model,
            immediate_model=immediate_model,
            incumbent_model=incumbent_model,
            binding_shift=parent_protocol.binding_shift,
        )
        stratified = _stratified(ranking["cells"])
        policy_extraction_metrics = {
            "all_archive_conditions_present": extraction.metrics[
                "all_archive_conditions_present"
            ],
            "duplicate_action_conflicts": 0,
            "search_seeds": extraction.metrics["search_seeds"],
            "source_lineages": extraction.metrics["source_lineages"],
        }
        _, scientific, derived = _evaluation_checks(
            protocol=parent_protocol,
            extraction_metrics=policy_extraction_metrics,
            stratified=stratified,
            elapsed_seconds=0.0,
        )
        policy_results[policy] = {
            "all_parent_scientific_gates_passed": all(scientific.values()),
            "derived": derived,
            "metrics": stratified["metrics"],
            "per_lineage": stratified["per_lineage"],
            "per_search_seed": stratified["per_search_seed"],
            "scientific_checks": scientific,
        }

    parent_metrics = dict(parent_receipt["metrics"])
    parent_order = policy_results["parent_order"]
    expected_patterns = dict(protocol.expected_difference_pattern_counts)
    elapsed = max(0.0, time.monotonic() - started)
    integrity_checks = {
        "all_archive_conditions_present": bool(
            extraction.metrics["all_archive_conditions_present"]
        ),
        "all_evaluation_search_seeds_present": set(
            extraction.metrics["search_seeds"]
        )
        == set(parent_protocol.evaluation_search_seeds),
        "all_source_lineages_present": set(extraction.metrics["source_lineages"])
        == set(parent_protocol.source_lineages),
        "conflict_count_reproduced": int(
            extraction.metrics["parent_duplicate_action_conflicts"]
        )
        == protocol.expected_parent_conflicts,
        "conflict_difference_patterns_reproduced": dict(
            extraction.metrics["conflict_difference_pattern_counts"]
        )
        == expected_patterns,
        "conflicted_archive_conditions_reproduced": int(
            extraction.metrics["conflicted_archive_condition_count"]
        )
        == protocol.expected_conflicted_archive_conditions,
        "unique_conflicted_archive_payloads_reproduced": int(
            extraction.metrics["unique_conflicted_archive_payloads"]
        )
        == protocol.expected_unique_conflicted_archive_payloads,
        "future_label_conflicts_reproduced": int(
            extraction.metrics["future_label_conflicts"]
        )
        == protocol.expected_future_label_conflicts,
        "immediate_label_conflicts_reproduced": int(
            extraction.metrics["immediate_label_conflicts"]
        )
        == protocol.expected_immediate_label_conflicts,
        "parent_overall_ranking_reproduced": _metrics_reproduced(
            parent_order["metrics"],
            parent_metrics,
            tolerance=protocol.parent_reproduction_tolerance,
        ),
        "parent_seed_strata_reproduced": _strata_reproduced(
            parent_order["per_search_seed"],
            parent_metrics["per_search_seed"],
            tolerance=protocol.parent_reproduction_tolerance,
        ),
        "parent_lineage_strata_reproduced": _strata_reproduced(
            parent_order["per_lineage"],
            parent_metrics["per_lineage"],
            tolerance=protocol.parent_reproduction_tolerance,
        ),
        "policy_set_matches_freeze": tuple(policy_results)
        == protocol.consolidation_policies,
        "sdk_budget_respected": protocol.maximum_sdk_calls == 0,
        "wall_time_respected": elapsed <= protocol.maximum_wall_seconds,
    }
    passed = all(integrity_checks.values())
    passing_policies = [
        policy
        for policy, result in policy_results.items()
        if result["all_parent_scientific_gates_passed"]
    ]
    robust_misses = sorted(
        name
        for name in next(iter(policy_results.values()))["scientific_checks"]
        if all(
            not result["scientific_checks"][name]
            for result in policy_results.values()
        )
    )
    classification = (
        "DIAGNOSTIC_INTEGRITY_FAILURE"
        if not passed
        else (
            "POSTHOC_TRANSFER_VERDICT_POLICY_SENSITIVE"
            if passing_policies
            else "POSTHOC_TRANSFER_GATE_MISS_ACROSS_REGISTERED_CONSOLIDATIONS"
        )
    )
    status = (
        "PASS_T12_6_1A_DIAGNOSTIC_COMPLETE"
        if passed
        else "FAIL_T12_6_1A_DIAGNOSTIC_INTEGRITY_GATE"
    )

    conflicts_path = destination / "conflict_rows.json"
    sensitivity_path = destination / "policy_sensitivities.json"
    report_path = destination / "diagnostic_report.json"
    _write_json_once(
        conflicts_path,
        {
            "format_version": "sage-t12.6.1a-conflict-rows-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "rows": list(extraction.conflict_rows),
        },
        storage_budget=storage,
    )
    _write_json_once(
        sensitivity_path,
        {
            "format_version": "sage-t12.6.1a-policy-sensitivities-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "model_bundle_checksum": bundle["bundle_checksum"],
            "policies": policy_results,
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    metrics = {
        **dict(extraction.metrics),
        "checks": integrity_checks,
        "classification": classification,
        "confirmatory_claim_authorized": False,
        "elapsed_seconds": elapsed,
        "metric_ranges": _metric_ranges(policy_results),
        "new_archive_confirmation_required": True,
        "passing_consolidation_policies": passing_policies,
        "policy_results": policy_results,
        "robust_scientific_gate_misses": robust_misses,
        "same_archive_reconfirmation_authorized": False,
        "sdk_calls_used": 0,
        "t12_6_2_freeze_authorized": False,
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6.1a-conflict-diagnostic-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = conflict_diagnostic_receipt(
        manifest=manifest,
        phase="diagnostic",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "conflict_rows": _artifact(conflicts_path),
            "policy_sensitivities": _artifact(sensitivity_path),
            "report": _artifact(report_path),
        },
    )
    _write_json_once(
        destination / "diagnostic_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def future_viability_conflict_diagnostic_status(
    *,
    manifest_path: str | Path,
    diagnostic_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_conflict_diagnostic_manifest(
        manifest_path, root=repo_root
    )
    receipt = (
        None
        if diagnostic_receipt_path is None
        or not Path(diagnostic_receipt_path).is_file()
        else load_conflict_diagnostic_receipt(
            diagnostic_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="diagnostic",
        )
    )
    ready = bool(
        receipt is None and manifest["firewall"].get("diagnostic_authorized", False)
    )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "diagnostic_receipt": (
            None
            if receipt is None
            else {
                "classification": receipt["metrics"]["classification"],
                "passed": receipt["passed"],
                "receipt_checksum": receipt["receipt_checksum"],
                "status": receipt["status"],
            }
        ),
        "firewall": {
            "diagnostic_authorized": ready,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "new_holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "t12_6_2_freeze_authorized": False,
            "new_archive_protocol_freeze_authorized": False,
        },
        "format_version": "sage-t12.6.1a-conflict-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": False,
        "protocol_checksum": manifest["protocol_checksum"],
    }


__all__ = [
    "future_viability_conflict_diagnostic_status",
    "run_future_viability_conflict_diagnostic",
]
