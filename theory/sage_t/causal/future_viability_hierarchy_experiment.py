"""Offline compile and sealed evaluation for SAGE.T12.6.1."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
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
from .future_viability import FutureViabilityModel
from .future_viability_hierarchy import (
    HierarchicalFutureViabilityModel,
    crossfit_hierarchical_viability,
    evaluate_hierarchical_viability_ranking,
    extract_hierarchical_viability_observations,
    summarize_hierarchical_cells,
)
from .future_viability_hierarchy_protocol import (
    FutureViabilityHierarchyProtocol,
    _resolve_bound,
    hierarchy_receipt,
    load_future_viability_hierarchy_manifest,
    load_hierarchy_receipt,
)

HIERARCHY_MODEL_BUNDLE_FORMAT = "sage-t12.6.1-hierarchy-model-bundle-v1"


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _extract(
    manifest: Mapping[str, Any],
    *,
    protocol: FutureViabilityHierarchyProtocol,
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
        raise ValueError("unsupported T12.6.1 corpus")
    return extract_hierarchical_viability_observations(
        archive_metas=metas,
        root=root,
        corpus=corpus,
        expected_search_seeds=seeds,
        expected_lineages=protocol.source_lineages,
        expected_arms=arms,
        future_horizon=protocol.future_horizon,
        local_radius=protocol.local_radius,
    )


def _stratified(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "metrics": summarize_hierarchical_cells(cells),
        "per_lineage": {
            str(lineage): summarize_hierarchical_cells(
                [cell for cell in cells if int(cell["lineage_seed"]) == lineage]
            )
            for lineage in sorted({int(cell["lineage_seed"]) for cell in cells})
        },
        "per_search_seed": {
            str(seed): summarize_hierarchical_cells(
                [cell for cell in cells if int(cell["search_seed"]) == seed]
            )
            for seed in sorted({int(cell["search_seed"]) for cell in cells})
        },
    }


def _compile_checks(
    *,
    protocol: FutureViabilityHierarchyProtocol,
    extraction_metrics: Mapping[str, Any],
    crossfit: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, float]]:
    micro = crossfit["micro_metrics"]
    folds = list(crossfit["folds"])
    worst_future = min(
        float(fold["metrics"]["future_binding_top1_accuracy"]) for fold in folds
    )
    worst_incumbent = min(
        float(fold["metrics"]["incumbent_binding_top1_accuracy"])
        for fold in folds
    )
    derived = {
        "compile_coverage_gain_over_incumbent": float(
            micro["hierarchy_coverage"]
        )
        - float(micro["incumbent_signature_coverage"]),
        "compile_worst_fold_gain_over_incumbent": worst_future - worst_incumbent,
    }
    integrity = {
        "all_archive_conditions_present": bool(
            extraction_metrics["all_archive_conditions_present"]
        ),
        "all_compile_search_seeds_present": set(extraction_metrics["search_seeds"])
        == set(protocol.training_search_seeds),
        "all_source_lineages_present": set(extraction_metrics["source_lineages"])
        == set(protocol.source_lineages),
        "duplicate_action_conflicts_absent": int(
            extraction_metrics["duplicate_action_conflicts"]
        )
        == 0,
        "evaluation_archive_payloads_excluded": True,
        "sdk_budget_respected": protocol.maximum_sdk_calls == 0,
        "wall_time_respected": elapsed_seconds
        <= protocol.maximum_wall_seconds_per_phase,
    }
    every_fold_specific = all(
        float(fold["metrics"]["future_binding_top1_accuracy"])
        > float(fold["metrics"]["immediate_binding_top1_accuracy"])
        and float(fold["metrics"]["future_binding_top1_accuracy"])
        > float(fold["metrics"]["binding_swap_top1_accuracy"])
        for fold in folds
    )
    every_lineage = all(
        float(lineage["future_binding_top1_accuracy"])
        >= protocol.minimum_compile_lineage_fold_accuracy
        for fold in folds
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
        "compile_hierarchy_coverage_sufficient": float(
            micro["hierarchy_coverage"]
        )
        >= protocol.minimum_compile_hierarchy_coverage,
        "compile_coverage_gain_over_incumbent_sufficient": derived[
            "compile_coverage_gain_over_incumbent"
        ]
        >= protocol.minimum_compile_coverage_gain_over_incumbent,
        "compile_unique_top_rate_sufficient": float(micro["unique_top_rate"])
        >= protocol.minimum_compile_unique_top_rate,
        "compile_recommendation_coverage_sufficient": float(
            micro["recommendation_coverage"]
        )
        >= protocol.minimum_compile_recommendation_coverage,
        "compile_incumbent_noninferiority": float(
            micro["future_gain_over_incumbent"]
        )
        >= protocol.minimum_compile_gain_over_incumbent,
        "compile_worst_fold_gain_over_incumbent_sufficient": derived[
            "compile_worst_fold_gain_over_incumbent"
        ]
        >= protocol.minimum_compile_worst_fold_gain_over_incumbent,
        "every_compile_fold_is_binding_specific": every_fold_specific,
        "every_compile_lineage_fold_accuracy_sufficient": every_lineage,
    }
    return integrity, scientific, derived


def compile_future_viability_hierarchy(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_hierarchy_manifest(
        manifest_path, root=repo_root, open_evaluation=False
    )
    if not manifest["firewall"].get("compile_authorized", False):
        raise ValueError("T12.6.1 manifest does not authorize compile")
    protocol = FutureViabilityHierarchyProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1 compile: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_phase)
    extracted = _extract(
        manifest, protocol=protocol, corpus="training", root=repo_root
    )
    crossfit = crossfit_hierarchical_viability(
        extracted.observations,
        search_seeds=protocol.training_search_seeds,
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
        binding_shift=protocol.binding_shift,
    )
    future_model = HierarchicalFutureViabilityModel.fit(
        extracted.observations,
        target_field="productive_reach",
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
    )
    immediate_model = HierarchicalFutureViabilityModel.fit(
        extracted.observations,
        target_field="immediate_score",
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
    )
    incumbent_model = FutureViabilityModel.fit(
        tuple(item.base for item in extracted.observations),
        target_field="productive_reach",
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
    )
    elapsed = max(0.0, time.monotonic() - started)
    integrity, scientific, derived = _compile_checks(
        protocol=protocol,
        extraction_metrics=extracted.metrics,
        crossfit=crossfit,
        elapsed_seconds=elapsed,
    )
    integrity_passed = all(integrity.values())
    support_passed = all(
        scientific[name]
        for name in (
            "compile_eligible_support_sufficient",
            "compile_hierarchy_coverage_sufficient",
            "compile_coverage_gain_over_incumbent_sufficient",
            "compile_unique_top_rate_sufficient",
            "compile_recommendation_coverage_sufficient",
        )
    )
    passed = bool(integrity_passed and all(scientific.values()))
    if not integrity_passed:
        classification = "COMPILE_INTEGRITY_FAILURE"
        status = "FAIL_T12_6_1_COMPILE_INTEGRITY_GATE"
    elif not support_passed:
        classification = "INSUFFICIENT_HIERARCHICAL_SUPPORT"
        status = "FAIL_T12_6_1_HIERARCHICAL_SUPPORT_GATE"
    elif not passed:
        classification = "HIERARCHICAL_VIABILITY_NOT_IDENTIFIED"
        status = "FAIL_T12_6_1_HIERARCHICAL_IDENTIFICATION_GATE"
    else:
        classification = "HIERARCHICAL_VIABILITY_CROSSFIT_IDENTIFIED"
        status = "PASS_T12_6_1_COMPILE_GATE"

    model_path = destination / "hierarchical_viability_models.sealed.json"
    crossfit_path = destination / "hierarchical_viability_crossfit.json"
    report_path = destination / "compile_report.json"
    model_bundle = _signed(
        {
            "format_version": HIERARCHY_MODEL_BUNDLE_FORMAT,
            "future_model": future_model.to_dict(),
            "immediate_model": immediate_model.to_dict(),
            "incumbent_model": incumbent_model.to_dict(),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "training_archive_count": len(manifest["inputs"]["training_archives"]),
            "training_observation_count": len(extracted.observations),
        },
        "bundle_checksum",
    )
    _write_json_once(model_path, model_bundle, storage_budget=storage)
    _write_json_once(
        crossfit_path,
        {
            "crossfit": crossfit,
            "extraction_metrics": dict(extracted.metrics),
            "format_version": "sage-t12.6.1-hierarchy-crossfit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    metrics = {
        **dict(extracted.metrics),
        **dict(crossfit["micro_metrics"]),
        **derived,
        "checks": {**integrity, **scientific},
        "classification": classification,
        "elapsed_seconds": elapsed,
        "evaluation_archive_payloads_loaded": 0,
        "evaluation_authorized": passed,
        "folds": crossfit["folds"],
        "sdk_calls_used": 0,
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6.1-hierarchy-compile-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = hierarchy_receipt(
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
) -> tuple[
    HierarchicalFutureViabilityModel,
    HierarchicalFutureViabilityModel,
    FutureViabilityModel,
    Mapping[str, Any],
]:
    payload = _read_json(path)
    _verify_signed(payload, "bundle_checksum")
    if payload.get("format_version") != HIERARCHY_MODEL_BUNDLE_FORMAT:
        raise ValueError("unsupported T12.6.1 hierarchy model bundle")
    if (
        payload.get("manifest_checksum") != manifest.get("manifest_checksum")
        or payload.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6.1 hierarchy model bundle changed")
    return (
        HierarchicalFutureViabilityModel.from_dict(payload["future_model"]),
        HierarchicalFutureViabilityModel.from_dict(payload["immediate_model"]),
        FutureViabilityModel.from_dict(payload["incumbent_model"]),
        payload,
    )


def _evaluation_checks(
    *,
    protocol: FutureViabilityHierarchyProtocol,
    extraction_metrics: Mapping[str, Any],
    stratified: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, Any]]:
    metrics = stratified["metrics"]
    seeds = stratified["per_search_seed"]
    per_seed_gains = {
        seed: float(value["future_gain_over_incumbent"])
        for seed, value in seeds.items()
    }
    derived = {
        "evaluation_coverage_gain_over_incumbent": float(
            metrics["hierarchy_coverage"]
        )
        - float(metrics["incumbent_signature_coverage"]),
        "evaluation_seed_wins_over_incumbent": sum(
            value > 0.0 for value in per_seed_gains.values()
        ),
        "evaluation_worst_seed_gain_over_incumbent": min(
            per_seed_gains.values(), default=-1.0
        ),
        "per_seed_gain_over_incumbent": per_seed_gains,
    }
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
        "wall_time_respected": elapsed_seconds
        <= protocol.maximum_wall_seconds_per_phase,
    }
    every_seed_specific = all(
        float(value["future_binding_top1_accuracy"])
        > float(value["immediate_binding_top1_accuracy"])
        and float(value["future_binding_top1_accuracy"])
        > float(value["binding_swap_top1_accuracy"])
        for value in seeds.values()
    )
    every_lineage = all(
        float(value["future_binding_top1_accuracy"])
        >= protocol.minimum_evaluation_lineage_accuracy
        for value in stratified["per_lineage"].values()
    )
    scientific = {
        "evaluation_eligible_support_sufficient": int(metrics["eligible_groups"])
        >= protocol.minimum_evaluation_eligible_groups,
        "evaluation_future_top1_sufficient": float(
            metrics["future_binding_top1_accuracy"]
        )
        >= protocol.minimum_evaluation_top1_accuracy,
        "evaluation_gain_over_immediate_sufficient": float(
            metrics["future_gain_over_immediate"]
        )
        >= protocol.minimum_evaluation_gain_over_immediate,
        "evaluation_gain_over_binding_swap_sufficient": float(
            metrics["future_gain_over_binding_swap"]
        )
        >= protocol.minimum_evaluation_gain_over_binding_swap,
        "evaluation_gain_over_incumbent_sufficient": float(
            metrics["future_gain_over_incumbent"]
        )
        >= protocol.minimum_evaluation_gain_over_incumbent,
        "evaluation_hierarchy_coverage_sufficient": float(
            metrics["hierarchy_coverage"]
        )
        >= protocol.minimum_evaluation_hierarchy_coverage,
        "evaluation_coverage_gain_over_incumbent_sufficient": derived[
            "evaluation_coverage_gain_over_incumbent"
        ]
        >= protocol.minimum_evaluation_coverage_gain_over_incumbent,
        "evaluation_unique_top_rate_sufficient": float(metrics["unique_top_rate"])
        >= protocol.minimum_evaluation_unique_top_rate,
        "evaluation_recommendation_coverage_sufficient": float(
            metrics["recommendation_coverage"]
        )
        >= protocol.minimum_evaluation_recommendation_coverage,
        "evaluation_worst_seed_incumbent_noninferiority": derived[
            "evaluation_worst_seed_gain_over_incumbent"
        ]
        >= protocol.minimum_evaluation_worst_seed_gain_over_incumbent,
        "evaluation_seed_wins_over_incumbent_sufficient": int(
            derived["evaluation_seed_wins_over_incumbent"]
        )
        >= protocol.minimum_evaluation_seed_wins_over_incumbent,
        "every_evaluation_seed_is_binding_specific": every_seed_specific,
        "every_evaluation_lineage_accuracy_sufficient": every_lineage,
    }
    return integrity, scientific, derived


def evaluate_future_viability_hierarchy(
    *,
    manifest_path: str | Path,
    compile_receipt_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_hierarchy_manifest(
        manifest_path, root=repo_root, open_evaluation=False
    )
    compile_receipt = load_hierarchy_receipt(
        compile_receipt_path,
        manifest=manifest,
        root=repo_root,
        expected_phase="compile",
        require_passed=True,
    )
    if compile_receipt.get("status") != "PASS_T12_6_1_COMPILE_GATE":
        raise ValueError("T12.6.1 evaluation requires the passed compile gate")
    manifest = load_future_viability_hierarchy_manifest(
        manifest_path, root=repo_root, open_evaluation=True
    )
    protocol = FutureViabilityHierarchyProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1 evaluation: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_phase)
    model_path = _resolve_bound(
        str(compile_receipt["artifacts"]["models"]["path"]), root=repo_root
    )
    future, immediate, incumbent, bundle = _load_model_bundle(
        model_path, manifest=manifest
    )
    extracted = _extract(
        manifest, protocol=protocol, corpus="evaluation", root=repo_root
    )
    ranking = evaluate_hierarchical_viability_ranking(
        extracted.observations,
        future_model=future,
        immediate_model=immediate,
        incumbent_model=incumbent,
        binding_shift=protocol.binding_shift,
    )
    stratified = _stratified(ranking["cells"])
    elapsed = max(0.0, time.monotonic() - started)
    integrity, scientific, derived = _evaluation_checks(
        protocol=protocol,
        extraction_metrics=extracted.metrics,
        stratified=stratified,
        elapsed_seconds=elapsed,
    )
    integrity_passed = all(integrity.values())
    support_passed = all(
        scientific[name]
        for name in (
            "evaluation_eligible_support_sufficient",
            "evaluation_hierarchy_coverage_sufficient",
            "evaluation_coverage_gain_over_incumbent_sufficient",
            "evaluation_unique_top_rate_sufficient",
            "evaluation_recommendation_coverage_sufficient",
        )
    )
    passed = bool(integrity_passed and all(scientific.values()))
    if not integrity_passed:
        classification = "EVALUATION_INTEGRITY_FAILURE"
        status = "FAIL_T12_6_1_EVALUATION_INTEGRITY_GATE"
    elif not support_passed:
        classification = "HIERARCHICAL_SUPPORT_DID_NOT_TRANSFER"
        status = "FAIL_T12_6_1_HIERARCHICAL_SUPPORT_TRANSFER_GATE"
    elif not passed:
        classification = "HIERARCHICAL_BINDING_DID_NOT_TRANSFER"
        status = "FAIL_T12_6_1_HIERARCHICAL_TRANSFER_GATE"
    else:
        classification = "HIERARCHICAL_FUTURE_VIABILITY_TRANSFERS"
        status = "PASS_T12_6_1_HIERARCHICAL_VIABILITY_GATE"

    ranking_path = destination / "evaluation_rankings.json"
    report_path = destination / "evaluation_report.json"
    _write_json_once(
        ranking_path,
        {
            "cells": ranking["cells"],
            "format_version": "sage-t12.6.1-hierarchy-evaluation-rankings-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "model_bundle_checksum": bundle["bundle_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    metrics = {
        **dict(extracted.metrics),
        **dict(stratified["metrics"]),
        **derived,
        "checks": {**integrity, **scientific},
        "classification": classification,
        "elapsed_seconds": elapsed,
        "per_lineage": stratified["per_lineage"],
        "per_search_seed": stratified["per_search_seed"],
        "sdk_calls_used": 0,
        "t12_6_2_freeze_authorized": passed,
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6.1-hierarchy-evaluation-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = hierarchy_receipt(
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


def future_viability_hierarchy_status(
    *,
    manifest_path: str | Path,
    compile_receipt_path: str | Path | None = None,
    evaluation_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_hierarchy_manifest(
        manifest_path, root=repo_root, open_evaluation=False
    )
    compile_receipt = (
        None
        if compile_receipt_path is None or not Path(compile_receipt_path).is_file()
        else load_hierarchy_receipt(
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
        else load_hierarchy_receipt(
            evaluation_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="evaluation",
        )
    )
    compile_passed = bool(
        compile_receipt
        and compile_receipt.get("passed") is True
        and compile_receipt.get("status") == "PASS_T12_6_1_COMPILE_GATE"
    )
    evaluation_passed = bool(
        evaluation_receipt
        and evaluation_receipt.get("passed") is True
        and evaluation_receipt.get("status")
        == "PASS_T12_6_1_HIERARCHICAL_VIABILITY_GATE"
    )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "compile_receipt": (
            None
            if compile_receipt is None
            else {
                "classification": compile_receipt["metrics"]["classification"],
                "passed": compile_receipt["passed"],
                "receipt_checksum": compile_receipt["receipt_checksum"],
                "status": compile_receipt["status"],
            }
        ),
        "evaluation_receipt": (
            None
            if evaluation_receipt is None
            else {
                "classification": evaluation_receipt["metrics"]["classification"],
                "passed": evaluation_receipt["passed"],
                "receipt_checksum": evaluation_receipt["receipt_checksum"],
                "status": evaluation_receipt["status"],
            }
        ),
        "firewall": {
            "compile_authorized": bool(
                compile_receipt is None
                and manifest["firewall"].get("compile_authorized", False)
            ),
            "evaluation_authorized": bool(
                compile_passed and evaluation_receipt is None
            ),
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "t12_6_2_freeze_authorized": evaluation_passed,
        },
        "format_version": "sage-t12.6.1-hierarchy-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": evaluation_passed,
        "protocol_checksum": manifest["protocol_checksum"],
    }


__all__ = [
    "compile_future_viability_hierarchy",
    "evaluate_future_viability_hierarchy",
    "future_viability_hierarchy_status",
]
