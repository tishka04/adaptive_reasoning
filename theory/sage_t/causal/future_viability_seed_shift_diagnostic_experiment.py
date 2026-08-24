"""Offline-only execution of the SAGE.T12.6.1b seed-shift diagnostic."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .experiment import (
    RunStorageBudget,
    _file_sha256,
    _read_json,
    _write_json_once,
)
from .future_viability_conflict_diagnostic import extract_conflict_sensitivities
from .future_viability_conflict_diagnostic_protocol import (
    load_conflict_diagnostic_receipt,
)
from .future_viability_hierarchy import (
    extract_hierarchical_viability_observations,
)
from .future_viability_hierarchy_experiment import _load_model_bundle
from .future_viability_hierarchy_protocol import (
    FutureViabilityHierarchyProtocol,
    load_future_viability_hierarchy_manifest,
)
from .future_viability_protocol import _resolve_bound
from .future_viability_seed_shift_diagnostic import (
    diagnose_future_viability_seed_shift,
)
from .future_viability_seed_shift_diagnostic_protocol import (
    FutureViabilitySeedShiftDiagnosticProtocol,
    load_future_viability_seed_shift_diagnostic_manifest,
    load_seed_shift_diagnostic_receipt,
    seed_shift_diagnostic_receipt,
)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def run_future_viability_seed_shift_diagnostic(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_seed_shift_diagnostic_manifest(
        manifest_path, root=repo_root
    )
    if not manifest["firewall"].get("diagnostic_authorized", False):
        raise ValueError("T12.6.1b manifest does not authorize diagnosis")
    protocol = FutureViabilitySeedShiftDiagnosticProtocol(**dict(manifest["protocol"]))
    hierarchy_manifest_path = _resolve_bound(
        str(manifest["parents"]["hierarchy_manifest"]["path"]), root=repo_root
    )
    hierarchy = load_future_viability_hierarchy_manifest(
        hierarchy_manifest_path, root=repo_root, open_evaluation=True
    )
    hierarchy_protocol = FutureViabilityHierarchyProtocol(
        **dict(hierarchy["protocol"])
    )
    model_path = _resolve_bound(
        str(
            manifest["parents"]["hierarchy_evaluation_artifacts"]["models"][
                "path"
            ]
        ),
        root=repo_root,
    )
    future_model, _, _, bundle = _load_model_bundle(
        model_path, manifest=hierarchy
    )
    conflict_receipt_path = _resolve_bound(
        str(manifest["parents"]["conflict_diagnostic_receipt"]["path"]),
        root=repo_root,
    )
    conflict_receipt = load_conflict_diagnostic_receipt(conflict_receipt_path)
    sensitivity_path = _resolve_bound(
        str(conflict_receipt["artifacts"]["policy_sensitivities"]["path"]),
        root=repo_root,
    )
    sensitivity_payload = _read_json(sensitivity_path)

    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1b diagnostic: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    training = extract_hierarchical_viability_observations(
        archive_metas=manifest["inputs"]["training_archives"],
        root=repo_root,
        corpus="training",
        expected_search_seeds=hierarchy_protocol.training_search_seeds,
        expected_lineages=hierarchy_protocol.source_lineages,
        expected_arms=hierarchy_protocol.training_arms,
        future_horizon=hierarchy_protocol.future_horizon,
        local_radius=hierarchy_protocol.local_radius,
    )
    evaluation = extract_conflict_sensitivities(
        archive_metas=manifest["inputs"]["evaluation_archives"],
        root=repo_root,
        corpus="evaluation",
        expected_search_seeds=hierarchy_protocol.evaluation_search_seeds,
        expected_lineages=hierarchy_protocol.source_lineages,
        expected_arms=hierarchy_protocol.evaluation_arms,
        future_horizon=hierarchy_protocol.future_horizon,
        local_radius=hierarchy_protocol.local_radius,
    )
    observations = evaluation.observations_by_policy[
        protocol.evaluation_consolidation_policy
    ]
    diagnostic = diagnose_future_viability_seed_shift(
        training.observations,
        observations,
        future_model=future_model,
        focal_search_seed=protocol.focal_search_seed,
        reference_search_seeds=protocol.reference_search_seeds,
        training_search_seeds=protocol.training_search_seeds,
        radius=hierarchy_protocol.local_radius,
        minimum_signature_support=hierarchy_protocol.minimum_signature_support,
    )
    elapsed = max(0.0, time.monotonic() - started)
    focal = diagnostic["focal_summary"]
    expected_parent = conflict_receipt["metrics"]["policy_results"][
        protocol.evaluation_consolidation_policy
    ]["per_search_seed"]
    per_seed_reproduced = all(
        int(diagnostic["per_search_seed"][seed]["eligible_groups"])
        == int(expected_parent[seed]["eligible_groups"])
        and int(diagnostic["per_search_seed"][seed]["hits"])
        == int(expected_parent[seed]["future_binding_hits"])
        for seed in expected_parent
    )
    checks = {
        "all_training_archive_conditions_present": bool(
            training.metrics["all_archive_conditions_present"]
        ),
        "all_evaluation_archive_conditions_present": bool(
            evaluation.metrics["all_archive_conditions_present"]
        ),
        "all_training_search_seeds_present": set(training.metrics["search_seeds"])
        == set(hierarchy_protocol.training_search_seeds),
        "all_evaluation_search_seeds_present": set(
            evaluation.metrics["search_seeds"]
        )
        == set(hierarchy_protocol.evaluation_search_seeds),
        "all_source_lineages_present": set(training.metrics["source_lineages"])
        == set(hierarchy_protocol.source_lineages)
        and set(evaluation.metrics["source_lineages"])
        == set(hierarchy_protocol.source_lineages),
        "training_duplicate_action_conflicts_absent": int(
            training.metrics["duplicate_action_conflicts"]
        )
        == 0,
        "parent_evaluation_conflicts_reproduced": int(
            evaluation.metrics["parent_duplicate_action_conflicts"]
        )
        == 37,
        "diagnostic_axes_match_freeze": tuple(diagnostic["diagnostic_axes"])
        == protocol.diagnostic_axes,
        "focal_group_count_reproduced": int(focal["eligible_groups"])
        == protocol.expected_focal_eligible_groups,
        "focal_hit_count_reproduced": int(focal["hits"])
        == protocol.expected_focal_hits,
        "focal_accuracy_reproduced": abs(
            float(focal["accuracy"]) - protocol.expected_focal_accuracy
        )
        <= 1e-12,
        "all_parent_seed_rankings_reproduced": per_seed_reproduced,
        "fixed_model_bundle_reused": bundle["bundle_checksum"]
        == sensitivity_payload.get("model_bundle_checksum"),
        "sdk_budget_respected": protocol.maximum_sdk_calls == 0,
        "wall_time_respected": elapsed <= protocol.maximum_wall_seconds,
    }
    passed = all(checks.values())
    status = (
        "PASS_T12_6_1B_DIAGNOSTIC_COMPLETE"
        if passed
        else "FAIL_T12_6_1B_DIAGNOSTIC_INTEGRITY_GATE"
    )
    classification = (
        diagnostic["classification"]
        if passed
        else "DIAGNOSTIC_INTEGRITY_FAILURE"
    )

    rows_path = destination / "seed_shift_rows.json"
    report_path = destination / "diagnostic_report.json"
    _write_json_once(
        rows_path,
        {
            "format_version": "sage-t12.6.1b-seed-shift-rows-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "rows": diagnostic["rows"],
        },
        storage_budget=storage,
    )
    metrics = {
        "checks": checks,
        "classification": classification,
        "confirmatory_claim_authorized": False,
        "diagnostic_axes": diagnostic["diagnostic_axes"],
        "elapsed_seconds": elapsed,
        "evaluation_observations": len(observations),
        "focal_search_seed": protocol.focal_search_seed,
        "focal_summary": focal,
        "future_protocol_freeze_authorized": False,
        "leave_one_training_seed_out": diagnostic[
            "leave_one_training_seed_out"
        ],
        "model_or_descriptor_change_authorized": False,
        "per_arm": diagnostic["per_arm"],
        "per_lineage": diagnostic["per_lineage"],
        "per_search_seed": diagnostic["per_search_seed"],
        "reference_contrast": diagnostic["reference_contrast"],
        "sdk_calls_used": 0,
        "training_observations": len(training.observations),
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6.1b-seed-shift-diagnostic-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = seed_shift_diagnostic_receipt(
        manifest=manifest,
        phase="diagnostic",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "report": _artifact(report_path),
            "rows": _artifact(rows_path),
        },
    )
    _write_json_once(
        destination / "diagnostic_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def future_viability_seed_shift_diagnostic_status(
    *,
    manifest_path: str | Path,
    diagnostic_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_seed_shift_diagnostic_manifest(
        manifest_path, root=repo_root
    )
    receipt = (
        None
        if diagnostic_receipt_path is None
        or not Path(diagnostic_receipt_path).is_file()
        else load_seed_shift_diagnostic_receipt(
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
            "future_protocol_freeze_authorized": False,
        },
        "format_version": "sage-t12.6.1b-seed-shift-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": False,
        "protocol_checksum": manifest["protocol_checksum"],
    }


__all__ = [
    "future_viability_seed_shift_diagnostic_status",
    "run_future_viability_seed_shift_diagnostic",
]
