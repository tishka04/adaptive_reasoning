"""Offline compile and sealed temporal evaluation for SAGE.T12.6."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .experiment import (
    RunStorageBudget,
    _file_sha256,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .future_viability import (
    FutureViabilityModel,
    crossfit_future_viability,
    evaluate_future_viability_ranking,
    extract_future_viability_observations,
)
from .future_viability_protocol import (
    FutureViabilityProtocol,
    _resolve_bound,
    future_viability_receipt,
    load_future_viability_manifest,
    load_future_viability_receipt,
)

MODEL_BUNDLE_FORMAT = "sage-t12.6-future-viability-model-bundle-v1"


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _extract(
    manifest: Mapping[str, Any],
    *,
    protocol: FutureViabilityProtocol,
    corpus: str,
    root: Path,
):
    if corpus == "training":
        metas = manifest["inputs"]["training_archives"]
        seeds = protocol.training_search_seeds
        arms = protocol.training_arms
    elif corpus == "evaluation":
        metas = manifest["inputs"]["evaluation_archives"]
        seeds = protocol.evaluation_search_seeds
        arms = protocol.evaluation_arms
    else:
        raise ValueError("unsupported T12.6 corpus")
    return extract_future_viability_observations(
        archive_metas=metas,
        root=root,
        corpus=corpus,
        expected_search_seeds=seeds,
        expected_lineages=protocol.source_lineages,
        expected_arms=arms,
        future_horizon=protocol.future_horizon,
        local_radius=protocol.local_radius,
    )


def _compile_checks(
    *,
    protocol: FutureViabilityProtocol,
    extraction_metrics: Mapping[str, Any],
    crossfit: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[dict[str, bool], dict[str, bool]]:
    micro = crossfit["micro_metrics"]
    integrity = {
        "all_archive_conditions_present": bool(
            extraction_metrics["all_archive_conditions_present"]
        ),
        "all_compile_search_seeds_present": set(
            extraction_metrics["search_seeds"]
        )
        == set(protocol.training_search_seeds),
        "all_source_lineages_present": set(extraction_metrics["source_lineages"])
        == set(protocol.source_lineages),
        "duplicate_action_conflicts_absent": int(
            extraction_metrics["duplicate_action_conflicts"]
        )
        == 0,
        "sdk_budget_respected": protocol.maximum_sdk_calls == 0,
        "wall_time_respected": elapsed_seconds <= protocol.maximum_wall_seconds_per_phase,
    }
    every_fold_specific = all(
        float(fold["metrics"]["future_binding_top1_accuracy"])
        > float(fold["metrics"]["immediate_binding_top1_accuracy"])
        and float(fold["metrics"]["future_binding_top1_accuracy"])
        > float(fold["metrics"]["binding_swap_top1_accuracy"])
        for fold in crossfit["folds"]
    )
    lineage_accuracy = all(
        float(lineage["future_binding_top1_accuracy"])
        >= protocol.minimum_compile_lineage_accuracy
        for fold in crossfit["folds"]
        for lineage in fold["metrics"]["per_lineage"].values()
    )
    scientific = {
        "compile_eligible_support_sufficient": int(micro["eligible_groups"])
        >= protocol.minimum_compile_eligible_groups,
        "compile_future_top1_sufficient": float(
            micro["future_binding_top1_accuracy"]
        )
        >= protocol.minimum_compile_top1_accuracy,
        "compile_gain_over_immediate_sufficient": float(
            micro["future_gain_over_immediate"]
        )
        >= protocol.minimum_compile_gain_over_immediate,
        "compile_gain_over_binding_swap_sufficient": float(
            micro["future_gain_over_binding_swap"]
        )
        >= protocol.minimum_compile_gain_over_binding_swap,
        "compile_signature_coverage_sufficient": float(
            micro["target_local_signature_coverage"]
        )
        >= protocol.minimum_compile_signature_coverage,
        "every_compile_fold_is_binding_specific": every_fold_specific,
        "every_compile_lineage_accuracy_sufficient": lineage_accuracy,
    }
    return integrity, scientific


def compile_future_viability(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_manifest(manifest_path, root=repo_root)
    if not manifest["firewall"].get("compile_authorized", False):
        raise ValueError("T12.6 manifest does not authorize compile")
    protocol = FutureViabilityProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable compile: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_phase)
    extracted = _extract(
        manifest,
        protocol=protocol,
        corpus="training",
        root=repo_root,
    )
    crossfit = crossfit_future_viability(
        extracted.observations,
        search_seeds=protocol.training_search_seeds,
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
        binding_shift=protocol.binding_shift,
    )
    future_model = FutureViabilityModel.fit(
        extracted.observations,
        target_field="productive_reach",
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
    )
    immediate_model = FutureViabilityModel.fit(
        extracted.observations,
        target_field="immediate_score",
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
    )
    elapsed = max(0.0, time.monotonic() - started)
    integrity, scientific = _compile_checks(
        protocol=protocol,
        extraction_metrics=extracted.metrics,
        crossfit=crossfit,
        elapsed_seconds=elapsed,
    )
    integrity_passed = all(integrity.values())
    support_passed = bool(
        scientific["compile_eligible_support_sufficient"]
        and scientific["compile_signature_coverage_sufficient"]
    )
    identification_passed = all(scientific.values())
    passed = bool(integrity_passed and identification_passed)
    if not integrity_passed:
        classification = "COMPILE_INTEGRITY_FAILURE"
        status = "FAIL_T12_6_COMPILE_INTEGRITY_GATE"
    elif not support_passed:
        classification = "INSUFFICIENT_FUTURE_VIABILITY_SUPPORT"
        status = "FAIL_T12_6_INSUFFICIENT_FUTURE_VIABILITY_SUPPORT"
    elif not identification_passed:
        classification = "FUTURE_VIABILITY_NOT_IDENTIFIED"
        status = "FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE"
    else:
        classification = "FUTURE_VIABILITY_CROSSFIT_IDENTIFIED"
        status = "PASS_T12_6_COMPILE_GATE"

    model_path = destination / "future_viability_models.sealed.json"
    crossfit_path = destination / "future_viability_crossfit.json"
    report_path = destination / "compile_report.json"
    model_bundle = _signed(
        {
            "format_version": MODEL_BUNDLE_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "future_model": future_model.to_dict(),
            "immediate_model": immediate_model.to_dict(),
            "training_archive_count": len(manifest["inputs"]["training_archives"]),
            "training_observation_count": len(extracted.observations),
        },
        "bundle_checksum",
    )
    _write_json_once(model_path, model_bundle, storage_budget=storage)
    _write_json_once(
        crossfit_path,
        {
            "format_version": "sage-t12.6-future-viability-crossfit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "crossfit": crossfit,
            "extraction_metrics": dict(extracted.metrics),
        },
        storage_budget=storage,
    )
    metrics = {
        **dict(extracted.metrics),
        **dict(crossfit["micro_metrics"]),
        "checks": {**integrity, **scientific},
        "classification": classification,
        "elapsed_seconds": elapsed,
        "evaluation_authorized": passed,
        "sdk_calls_used": 0,
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6-future-viability-compile-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = future_viability_receipt(
        manifest=manifest,
        phase="compile",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "crossfit": _artifact(crossfit_path),
            "models": _artifact(model_path),
            "report": _artifact(report_path),
        },
    )
    _write_json_once(
        destination / "compile_receipt.json", receipt, storage_budget=storage
    )
    return receipt


def _load_model_bundle(
    path: Path,
    *,
    manifest: Mapping[str, Any],
) -> tuple[FutureViabilityModel, FutureViabilityModel, Mapping[str, Any]]:
    payload = _read_json(path)
    _verify_signed(payload, "bundle_checksum")
    if payload.get("format_version") != MODEL_BUNDLE_FORMAT:
        raise ValueError("unsupported T12.6 model bundle")
    if payload.get("manifest_checksum") != manifest.get("manifest_checksum"):
        raise ValueError("T12.6 model bundle belongs to another manifest")
    if payload.get("protocol_checksum") != manifest.get("protocol_checksum"):
        raise ValueError("T12.6 model bundle protocol changed")
    return (
        FutureViabilityModel.from_dict(payload["future_model"]),
        FutureViabilityModel.from_dict(payload["immediate_model"]),
        payload,
    )


def _evaluation_checks(
    *,
    protocol: FutureViabilityProtocol,
    extraction_metrics: Mapping[str, Any],
    ranking_metrics: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[dict[str, bool], dict[str, bool]]:
    integrity = {
        "all_archive_conditions_present": bool(
            extraction_metrics["all_archive_conditions_present"]
        ),
        "all_evaluation_search_seeds_present": set(
            extraction_metrics["search_seeds"]
        )
        == set(protocol.evaluation_search_seeds),
        "all_source_lineages_present": set(extraction_metrics["source_lineages"])
        == set(protocol.source_lineages),
        "duplicate_action_conflicts_absent": int(
            extraction_metrics["duplicate_action_conflicts"]
        )
        == 0,
        "sdk_budget_respected": protocol.maximum_sdk_calls == 0,
        "wall_time_respected": elapsed_seconds <= protocol.maximum_wall_seconds_per_phase,
    }
    every_seed_specific = all(
        float(seed["future_binding_top1_accuracy"])
        > float(seed["immediate_binding_top1_accuracy"])
        and float(seed["future_binding_top1_accuracy"])
        > float(seed["binding_swap_top1_accuracy"])
        for seed in ranking_metrics["per_search_seed"].values()
    )
    lineage_accuracy = all(
        float(lineage["future_binding_top1_accuracy"])
        >= protocol.minimum_evaluation_lineage_accuracy
        for lineage in ranking_metrics["per_lineage"].values()
    )
    scientific = {
        "evaluation_eligible_support_sufficient": int(
            ranking_metrics["eligible_groups"]
        )
        >= protocol.minimum_evaluation_eligible_groups,
        "evaluation_future_top1_sufficient": float(
            ranking_metrics["future_binding_top1_accuracy"]
        )
        >= protocol.minimum_evaluation_top1_accuracy,
        "evaluation_gain_over_immediate_sufficient": float(
            ranking_metrics["future_gain_over_immediate"]
        )
        >= protocol.minimum_evaluation_gain_over_immediate,
        "evaluation_gain_over_binding_swap_sufficient": float(
            ranking_metrics["future_gain_over_binding_swap"]
        )
        >= protocol.minimum_evaluation_gain_over_binding_swap,
        "evaluation_signature_coverage_sufficient": float(
            ranking_metrics["target_local_signature_coverage"]
        )
        >= protocol.minimum_evaluation_signature_coverage,
        "every_evaluation_seed_is_binding_specific": every_seed_specific,
        "every_evaluation_lineage_accuracy_sufficient": lineage_accuracy,
    }
    return integrity, scientific


def evaluate_future_viability(
    *,
    manifest_path: str | Path,
    compile_receipt_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_manifest(manifest_path, root=repo_root)
    compile_receipt = load_future_viability_receipt(
        compile_receipt_path,
        manifest=manifest,
        root=repo_root,
        require_passed=True,
        expected_phase="compile",
    )
    if compile_receipt.get("status") != "PASS_T12_6_COMPILE_GATE":
        raise ValueError("T12.6 evaluation requires the passed compile gate")
    protocol = FutureViabilityProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable evaluation: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_phase)
    model_path = _resolve_bound(
        str(compile_receipt["artifacts"]["models"]["path"]), root=repo_root
    )
    future_model, immediate_model, bundle = _load_model_bundle(
        model_path, manifest=manifest
    )
    extracted = _extract(
        manifest,
        protocol=protocol,
        corpus="evaluation",
        root=repo_root,
    )
    ranking = evaluate_future_viability_ranking(
        extracted.observations,
        future_model=future_model,
        immediate_model=immediate_model,
        binding_shift=protocol.binding_shift,
    )
    elapsed = max(0.0, time.monotonic() - started)
    integrity, scientific = _evaluation_checks(
        protocol=protocol,
        extraction_metrics=extracted.metrics,
        ranking_metrics=ranking["metrics"],
        elapsed_seconds=elapsed,
    )
    integrity_passed = all(integrity.values())
    support_passed = bool(
        scientific["evaluation_eligible_support_sufficient"]
        and scientific["evaluation_signature_coverage_sufficient"]
    )
    identification_passed = all(scientific.values())
    passed = bool(integrity_passed and identification_passed)
    if not integrity_passed:
        classification = "EVALUATION_INTEGRITY_FAILURE"
        status = "FAIL_T12_6_EVALUATION_INTEGRITY_GATE"
    elif not support_passed:
        classification = "FUTURE_VIABILITY_SUPPORT_DID_NOT_TRANSFER"
        status = "FAIL_T12_6_FUTURE_VIABILITY_SUPPORT_TRANSFER_GATE"
    elif not identification_passed:
        classification = "FUTURE_VIABILITY_BINDING_DID_NOT_TRANSFER"
        status = "FAIL_T12_6_FUTURE_VIABILITY_TRANSFER_GATE"
    else:
        classification = "TARGET_LOCAL_FUTURE_VIABILITY_TRANSFERS"
        status = "PASS_T12_6_FUTURE_VIABILITY_GATE"

    ranking_path = destination / "evaluation_rankings.json"
    report_path = destination / "evaluation_report.json"
    _write_json_once(
        ranking_path,
        {
            "format_version": "sage-t12.6-future-viability-rankings-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "model_bundle_checksum": bundle["bundle_checksum"],
            "cells": ranking["cells"],
        },
        storage_budget=storage,
    )
    parent_receipt_path = _resolve_bound(
        str(manifest["evaluation_parent"]["receipt"]["path"]), root=repo_root
    )
    parent_receipt = _read_json(parent_receipt_path)
    _verify_signed(parent_receipt, "receipt_checksum")
    if parent_receipt.get("receipt_checksum") != manifest["evaluation_parent"][
        "receipt"
    ]["receipt_checksum"]:
        raise ValueError("T12.6 evaluation parent receipt changed")
    metrics = {
        **dict(extracted.metrics),
        **dict(ranking["metrics"]),
        "checks": {**integrity, **scientific},
        "classification": classification,
        "elapsed_seconds": elapsed,
        "evaluation_parent_progress_edges": dict(
            parent_receipt.get("metrics", {}).get("progress_edges", {})
        ),
        "sdk_calls_used": 0,
        "t12_6b_physical_freeze_authorized": passed,
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6-future-viability-evaluation-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = future_viability_receipt(
        manifest=manifest,
        phase="evaluation",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "compile_receipt": _artifact(Path(compile_receipt_path)),
            "models": _artifact(model_path),
            "rankings": _artifact(ranking_path),
            "report": _artifact(report_path),
        },
    )
    _write_json_once(
        destination / "evaluation_receipt.json", receipt, storage_budget=storage
    )
    return receipt


def future_viability_status(
    *,
    manifest_path: str | Path,
    compile_receipt_path: str | Path | None = None,
    evaluation_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_manifest(manifest_path, root=repo_root)
    compile_receipt = (
        None
        if compile_receipt_path is None or not Path(compile_receipt_path).is_file()
        else load_future_viability_receipt(
            compile_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="compile",
        )
    )
    evaluation_receipt = (
        None
        if evaluation_receipt_path is None
        or not Path(evaluation_receipt_path).is_file()
        else load_future_viability_receipt(
            evaluation_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="evaluation",
        )
    )
    compile_passed = bool(
        compile_receipt
        and compile_receipt.get("passed") is True
        and compile_receipt.get("status") == "PASS_T12_6_COMPILE_GATE"
    )
    evaluation_passed = bool(
        evaluation_receipt
        and evaluation_receipt.get("passed") is True
        and evaluation_receipt.get("status") == "PASS_T12_6_FUTURE_VIABILITY_GATE"
    )
    compile_ready = bool(
        compile_receipt is None and manifest["firewall"].get("compile_authorized")
    )
    evaluation_ready = bool(compile_passed and evaluation_receipt is None)
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "compile_authorized": compile_ready,
            "evaluation_authorized": evaluation_ready,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "t12_6b_physical_freeze_authorized": evaluation_passed,
        },
        "format_version": "sage-t12.6-future-viability-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": evaluation_passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "compile_receipt": (
            None
            if compile_receipt is None
            else {
                "classification": compile_receipt.get("metrics", {}).get(
                    "classification"
                ),
                "passed": compile_receipt["passed"],
                "receipt_checksum": compile_receipt["receipt_checksum"],
                "status": compile_receipt["status"],
            }
        ),
        "evaluation_receipt": (
            None
            if evaluation_receipt is None
            else {
                "classification": evaluation_receipt.get("metrics", {}).get(
                    "classification"
                ),
                "passed": evaluation_receipt["passed"],
                "receipt_checksum": evaluation_receipt["receipt_checksum"],
                "status": evaluation_receipt["status"],
            }
        ),
    }


__all__ = [
    "compile_future_viability",
    "evaluate_future_viability",
    "future_viability_status",
]
