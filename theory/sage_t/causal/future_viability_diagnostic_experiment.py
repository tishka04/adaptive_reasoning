"""Offline-only execution of the frozen SAGE.T12.6a post-hoc diagnostic."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .experiment import RunStorageBudget, _file_sha256, _read_json, _write_json_once
from .future_viability import extract_future_viability_observations
from .future_viability_diagnostic import diagnose_future_viability_fold
from .future_viability_diagnostic_protocol import (
    FutureViabilityDiagnosticProtocol,
    future_viability_diagnostic_receipt,
    load_future_viability_diagnostic_manifest,
    load_future_viability_diagnostic_receipt,
)
from .future_viability_protocol import FutureViabilityProtocol, _resolve_bound


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _extract_training(
    manifest: Mapping[str, Any], *, root: Path
):
    parent_path = _resolve_bound(
        str(manifest["parent"]["manifest"]["path"]), root=root
    )
    parent = _read_json(parent_path)
    parent_protocol = FutureViabilityProtocol(**dict(parent["protocol"]))
    return extract_future_viability_observations(
        archive_metas=manifest["inputs"]["training_archives"],
        root=root,
        corpus="training",
        expected_search_seeds=parent_protocol.training_search_seeds,
        expected_lineages=parent_protocol.source_lineages,
        expected_arms=parent_protocol.training_arms,
        future_horizon=parent_protocol.future_horizon,
        local_radius=parent_protocol.local_radius,
    ), parent_protocol


def run_future_viability_diagnostic(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_diagnostic_manifest(
        manifest_path, root=repo_root
    )
    if not manifest["firewall"].get("diagnostic_authorized", False):
        raise ValueError("T12.6a manifest does not authorize diagnosis")
    protocol = FutureViabilityDiagnosticProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6a diagnostic: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    extracted, parent_protocol = _extract_training(manifest, root=repo_root)
    diagnostic = diagnose_future_viability_fold(
        extracted.observations,
        holdout_search_seed=protocol.focal_search_seed,
        focal_lineage_seed=protocol.focal_lineage_seed,
        reference_lineage_seed=protocol.reference_lineage_seed,
        radius=parent_protocol.local_radius,
        minimum_signature_support=parent_protocol.minimum_signature_support,
        binding_shift=parent_protocol.binding_shift,
    )
    elapsed = max(0.0, time.monotonic() - started)
    focal = diagnostic["focal_metrics"]
    checks = {
        "all_training_archive_conditions_present": bool(
            extracted.metrics["all_archive_conditions_present"]
        ),
        "all_training_search_seeds_present": set(extracted.metrics["search_seeds"])
        == set(parent_protocol.training_search_seeds),
        "all_source_lineages_present": set(extracted.metrics["source_lineages"])
        == set(parent_protocol.source_lineages),
        "diagnostic_axes_match_freeze": tuple(diagnostic["diagnostic_axes"])
        == protocol.diagnostic_axes,
        "duplicate_action_conflicts_absent": int(
            extracted.metrics["duplicate_action_conflicts"]
        )
        == 0,
        "evaluation_archive_payloads_excluded": True,
        "focal_group_count_reproduced": int(focal["eligible_groups"])
        == protocol.expected_focal_eligible_groups,
        "focal_hit_count_reproduced": int(focal["hits"])
        == protocol.expected_focal_hits,
        "sdk_budget_respected": protocol.maximum_sdk_calls == 0,
        "wall_time_respected": elapsed <= protocol.maximum_wall_seconds,
    }
    passed = all(checks.values())
    status = (
        "PASS_T12_6A_DIAGNOSTIC_COMPLETE"
        if passed
        else "FAIL_T12_6A_DIAGNOSTIC_INTEGRITY_GATE"
    )
    rows_path = destination / "diagnostic_rows.json"
    report_path = destination / "diagnostic_report.json"
    _write_json_once(
        rows_path,
        {
            "format_version": "sage-t12.6a-future-viability-diagnostic-rows-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "rows": diagnostic["rows"],
        },
        storage_budget=storage,
    )
    metrics = {
        **dict(extracted.metrics),
        "checks": checks,
        "classification": diagnostic["classification"],
        "confirmatory_claim_authorized": False,
        "counterfactual_sensitivities": diagnostic[
            "counterfactual_sensitivities"
        ],
        "diagnostic_axes": diagnostic["diagnostic_axes"],
        "elapsed_seconds": elapsed,
        "error_summary": diagnostic["error_summary"],
        "evaluation_archive_payloads_loaded": 0,
        "focal_metrics": focal,
        "future_protocol_freeze_authorized": False,
        "sdk_calls_used": 0,
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6a-future-viability-diagnostic-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = future_viability_diagnostic_receipt(
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


def future_viability_diagnostic_status(
    *,
    manifest_path: str | Path,
    diagnostic_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_diagnostic_manifest(
        manifest_path, root=repo_root
    )
    receipt = (
        None
        if diagnostic_receipt_path is None
        or not Path(diagnostic_receipt_path).is_file()
        else load_future_viability_diagnostic_receipt(
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
                "classification": receipt.get("metrics", {}).get("classification"),
                "passed": receipt["passed"],
                "receipt_checksum": receipt["receipt_checksum"],
                "status": receipt["status"],
            }
        ),
        "firewall": {
            "diagnostic_authorized": ready,
            "evaluation_authorized": False,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "future_protocol_freeze_authorized": False,
        },
        "format_version": "sage-t12.6a-future-viability-diagnostic-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": False,
        "protocol_checksum": manifest["protocol_checksum"],
    }


__all__ = [
    "future_viability_diagnostic_status",
    "run_future_viability_diagnostic",
]
