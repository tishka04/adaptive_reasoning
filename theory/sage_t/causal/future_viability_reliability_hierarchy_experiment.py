"""Source-train-only compile for SAGE.T12.6.1c."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .experiment import RunStorageBudget, _file_sha256, _signed, _write_json_once
from .future_viability_hierarchy import (
    HierarchicalFutureViabilityModel,
    extract_hierarchical_viability_observations,
)
from .future_viability_reliability_hierarchy import (
    RELIABILITY_CANDIDATES,
    ReliabilityGatedFutureViabilityModel,
    evaluate_reliability_candidates,
)
from .future_viability_reliability_hierarchy_protocol import (
    FutureViabilityReliabilityProtocol,
    load_future_viability_reliability_manifest,
    load_reliability_hierarchy_receipt,
    reliability_hierarchy_receipt,
)

RELIABILITY_MODEL_BUNDLE_FORMAT = (
    "sage-t12.6.1c-reliability-gated-model-bundle-v1"
)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _extract(
    manifest: Mapping[str, Any],
    *,
    protocol: FutureViabilityReliabilityProtocol,
    root: Path,
):
    if set(manifest.get("inputs", {})) != {"training_archives"}:
        raise ValueError("T12.6.1c compile refuses non-training inputs")
    return extract_hierarchical_viability_observations(
        archive_metas=manifest["inputs"]["training_archives"],
        root=root,
        corpus="training",
        expected_search_seeds=protocol.training_search_seeds,
        expected_lineages=protocol.source_lineages,
        expected_arms=protocol.training_arms,
        future_horizon=protocol.future_horizon,
        local_radius=protocol.local_radius,
    )


def _reliable_exact_support_count(
    model: ReliabilityGatedFutureViabilityModel,
) -> int:
    return sum(
        len(support.search_seeds) >= model.minimum_exact_seed_span
        and support.value_range <= model.maximum_exact_label_range
        for support in model.exact_support.values()
    )


def _compile_checks(
    *,
    protocol: FutureViabilityReliabilityProtocol,
    extraction_metrics: Mapping[str, Any],
    selection: Mapping[str, Any],
    reliable_exact_supports: int,
    elapsed_seconds: float,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, Any]]:
    selected_name = str(selection["selected_candidate"])
    selected = selection["candidate_results"][selected_name]
    micro = selected["micro_metrics"]
    folds = selected["folds"]
    worst_fold = min(
        float(fold["metrics"]["future_binding_top1_accuracy"])
        for fold in folds
    )
    derived = {
        "compile_worst_fold_top1_accuracy": worst_fold,
        "full_fit_reliable_exact_supports": reliable_exact_supports,
        "selected_candidate": selected_name,
        "selected_maximum_exact_label_range": RELIABILITY_CANDIDATES[
            selected_name
        ][1],
        "selected_minimum_exact_seed_span": RELIABILITY_CANDIDATES[selected_name][
            0
        ],
    }
    integrity = {
        "all_archive_conditions_present": bool(
            extraction_metrics["all_archive_conditions_present"]
        ),
        "all_training_search_seeds_present": set(extraction_metrics["search_seeds"])
        == set(protocol.training_search_seeds),
        "all_source_lineages_present": set(extraction_metrics["source_lineages"])
        == set(protocol.source_lineages),
        "candidate_registry_complete": set(selection["candidate_results"])
        == set(protocol.reliability_candidates),
        "candidate_selection_criterion_match": selection["selection_criterion"]
        == protocol.candidate_selection_criterion,
        "duplicate_action_conflicts_absent": int(
            extraction_metrics["duplicate_action_conflicts"]
        )
        == 0,
        "evaluation_archive_payloads_excluded": True,
        "evaluation_archive_registry_excluded": True,
        "sdk_budget_respected": protocol.maximum_sdk_calls == 0,
        "wall_time_respected": elapsed_seconds <= protocol.maximum_wall_seconds,
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
        "compile_worst_fold_top1_sufficient": worst_fold
        >= protocol.minimum_compile_worst_fold_accuracy,
        "compile_gain_over_immediate_sufficient": float(
            micro["future_gain_over_immediate"]
        )
        >= protocol.minimum_compile_gain_over_immediate,
        "compile_gain_over_binding_swap_sufficient": float(
            micro["future_gain_over_binding_swap"]
        )
        >= protocol.minimum_compile_gain_over_binding_swap,
        "compile_exact_first_noninferiority": float(
            micro["future_gain_over_incumbent"]
        )
        >= protocol.minimum_compile_gain_over_exact_first,
        "compile_hierarchy_coverage_sufficient": float(
            micro["hierarchy_coverage"]
        )
        >= protocol.minimum_compile_hierarchy_coverage,
        "compile_unique_top_rate_sufficient": float(micro["unique_top_rate"])
        >= protocol.minimum_compile_unique_top_rate,
        "compile_recommendation_coverage_sufficient": float(
            micro["recommendation_coverage"]
        )
        >= protocol.minimum_compile_recommendation_coverage,
        "compile_exact_rejection_exercised": float(
            micro["exact_rejection_exercised_rate"]
        )
        >= protocol.minimum_exact_rejection_exercised_rate,
        "compile_reliable_exact_support_retained": reliable_exact_supports
        >= protocol.minimum_full_fit_reliable_exact_supports,
        "every_compile_fold_is_binding_specific": every_fold_specific,
        "every_compile_lineage_fold_accuracy_sufficient": every_lineage,
    }
    return integrity, scientific, derived


def compile_future_viability_reliability_hierarchy(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_reliability_manifest(
        manifest_path, root=repo_root
    )
    if not manifest["firewall"].get("compile_authorized", False):
        raise ValueError("T12.6.1c manifest does not authorize compile")
    protocol = FutureViabilityReliabilityProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1c compile: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    extracted = _extract(manifest, protocol=protocol, root=repo_root)
    selection = evaluate_reliability_candidates(
        extracted.observations,
        search_seeds=protocol.training_search_seeds,
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
        candidate_names=protocol.reliability_candidates,
        binding_shift=protocol.binding_shift,
    )
    selected_name = str(selection["selected_candidate"])
    seed_span, label_range = RELIABILITY_CANDIDATES[selected_name]
    common = {
        "radius": protocol.local_radius,
        "minimum_signature_support": protocol.minimum_signature_support,
        "minimum_exact_seed_span": seed_span,
        "maximum_exact_label_range": label_range,
    }
    future_model = ReliabilityGatedFutureViabilityModel.fit(
        extracted.observations,
        target_field="productive_reach",
        **common,
    )
    immediate_model = ReliabilityGatedFutureViabilityModel.fit(
        extracted.observations,
        target_field="immediate_score",
        **common,
    )
    incumbent_model = HierarchicalFutureViabilityModel.fit(
        extracted.observations,
        target_field="productive_reach",
        radius=protocol.local_radius,
        minimum_signature_support=protocol.minimum_signature_support,
    )
    reliable_exact_supports = _reliable_exact_support_count(future_model)
    elapsed = max(0.0, time.monotonic() - started)
    integrity, scientific, derived = _compile_checks(
        protocol=protocol,
        extraction_metrics=extracted.metrics,
        selection=selection,
        reliable_exact_supports=reliable_exact_supports,
        elapsed_seconds=elapsed,
    )
    integrity_passed = all(integrity.values())
    passed = bool(integrity_passed and all(scientific.values()))
    if not integrity_passed:
        classification = "SOURCE_TRAIN_COMPILE_INTEGRITY_FAILURE"
        status = "FAIL_T12_6_1C_SOURCE_TRAIN_INTEGRITY_GATE"
    elif not passed:
        classification = "RELIABILITY_GATED_HIERARCHY_NOT_QUALIFIED"
        status = "FAIL_T12_6_1C_SOURCE_TRAIN_QUALIFICATION_GATE"
    else:
        classification = "RELIABILITY_GATED_HIERARCHY_SOURCE_TRAIN_QUALIFIED"
        status = "PASS_T12_6_1C_SOURCE_TRAIN_COMPILE_GATE"

    model_path = destination / "reliability_gated_models.sealed.json"
    crossfit_path = destination / "source_train_candidate_crossfit.json"
    report_path = destination / "compile_report.json"
    model_bundle = _signed(
        {
            "format_version": RELIABILITY_MODEL_BUNDLE_FORMAT,
            "future_model": future_model.to_dict(),
            "immediate_model": immediate_model.to_dict(),
            "incumbent_exact_first_model": incumbent_model.to_dict(),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "selected_candidate": selected_name,
            "training_archive_count": len(manifest["inputs"]["training_archives"]),
            "training_observation_count": len(extracted.observations),
        },
        "bundle_checksum",
    )
    _write_json_once(model_path, model_bundle, storage_budget=storage)
    _write_json_once(
        crossfit_path,
        {
            "candidate_results": selection["candidate_results"],
            "candidate_summary": selection["candidate_summary"],
            "format_version": "sage-t12.6.1c-source-train-candidate-crossfit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "selected_candidate": selected_name,
            "selection_criterion": selection["selection_criterion"],
        },
        storage_budget=storage,
    )
    selected_metrics = selection["candidate_results"][selected_name][
        "micro_metrics"
    ]
    metrics = {
        **dict(extracted.metrics),
        **dict(selected_metrics),
        **derived,
        "candidate_summary": selection["candidate_summary"],
        "checks": {**integrity, **scientific},
        "classification": classification,
        "confirmatory_claim_authorized": False,
        "elapsed_seconds": elapsed,
        "evaluation_archive_count": 0,
        "evaluation_archive_payloads_loaded": 0,
        "folds": selection["candidate_results"][selected_name]["folds"],
        "new_archive_protocol_freeze_authorized": passed,
        "physical_collection_authorized": False,
        "sdk_calls_used": 0,
    }
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.6.1c-source-train-compile-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = reliability_hierarchy_receipt(
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


def future_viability_reliability_status(
    *,
    manifest_path: str | Path,
    compile_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_reliability_manifest(
        manifest_path, root=repo_root
    )
    receipt = (
        None
        if compile_receipt_path is None or not Path(compile_receipt_path).is_file()
        else load_reliability_hierarchy_receipt(
            compile_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="compile",
        )
    )
    passed = bool(
        receipt
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_6_1C_SOURCE_TRAIN_COMPILE_GATE"
    )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "compile_receipt": (
            None
            if receipt is None
            else {
                "classification": receipt["metrics"]["classification"],
                "passed": receipt["passed"],
                "receipt_checksum": receipt["receipt_checksum"],
                "selected_candidate": receipt["metrics"]["selected_candidate"],
                "status": receipt["status"],
            }
        ),
        "firewall": {
            "compile_authorized": bool(
                receipt is None and manifest["firewall"].get("compile_authorized")
            ),
            "evaluation_authorized": False,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "new_holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "new_archive_protocol_freeze_authorized": passed,
        },
        "format_version": "sage-t12.6.1c-reliability-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": passed,
        "protocol_checksum": manifest["protocol_checksum"],
    }


__all__ = [
    "RELIABILITY_MODEL_BUNDLE_FORMAT",
    "compile_future_viability_reliability_hierarchy",
    "future_viability_reliability_status",
]
