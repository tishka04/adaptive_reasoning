"""Staged collection and prospective adjudication for SAGE.T12.6.1d."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
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
from .future_viability_hierarchy import HierarchicalFutureViabilityModel
from .future_viability_prospective_confirmation import (
    adjudicate_prediction_commitment,
    commit_label_blind_predictions,
    extract_exact_state_candidates,
    verify_prediction_commitment,
)
from .future_viability_prospective_protocol import (
    FutureViabilityProspectiveProtocol,
    load_future_viability_prospective_manifest,
    load_prospective_receipt,
    prospective_receipt,
)
from .future_viability_reliability_hierarchy import (
    ReliabilityGatedFutureViabilityModel,
)
from .future_viability_reliability_hierarchy_experiment import (
    RELIABILITY_MODEL_BUNDLE_FORMAT,
)
from .graph_experiment import _write_archive
from .hazard_diversity_experiment import (
    _load_frozen_inputs,
    _parent_artifacts,
    run_hazard_diversity_arm,
)
from .hazard_diversity_model import AbstractHazardModel
from .hazard_diversity_protocol import (
    load_hazard_diversity_manifest,
    load_hazard_diversity_receipt,
)

PREDICTION_ARTIFACT_FORMAT = "sage-t12.6.1d-bound-label-blind-prediction-artifact-v2"

EnvFactory = Callable[[str], Any]


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _artifact(path: Path, **extra: Any) -> dict[str, Any]:
    return {**extra, "path": str(path.resolve()), "sha256": _file_sha256(path)}


def _clone_shield(shield: Any) -> Any:
    from .shield_model import ProgressProtectedTerminalShield

    return ProgressProtectedTerminalShield.from_dict(shield.to_dict())


def _load_model_bundle(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[
    ReliabilityGatedFutureViabilityModel,
    ReliabilityGatedFutureViabilityModel,
    HierarchicalFutureViabilityModel,
    Mapping[str, Any],
]:
    path = _resolve(
        str(manifest["parents"]["reliability_model_bundle"]["path"]), root=root
    )
    payload = _read_json(path)
    _verify_signed(payload, "bundle_checksum")
    if payload.get("format_version") != RELIABILITY_MODEL_BUNDLE_FORMAT:
        raise ValueError("unsupported T12.6.1d parent model bundle")
    if payload.get("selected_candidate") != "exact_span2_range0":
        raise ValueError("T12.6.1d parent selected model changed")
    future = ReliabilityGatedFutureViabilityModel.from_dict(payload["future_model"])
    immediate = ReliabilityGatedFutureViabilityModel.from_dict(
        payload["immediate_model"]
    )
    incumbent = HierarchicalFutureViabilityModel.from_dict(
        payload["incumbent_exact_first_model"]
    )
    if (
        future.minimum_exact_seed_span != 2
        or future.maximum_exact_label_range != 0.0
        or future.target_field != "productive_reach"
        or immediate.target_field != "immediate_score"
        or incumbent.target_field != "productive_reach"
    ):
        raise ValueError("T12.6.1d frozen model contract changed")
    return future, immediate, incumbent, payload


def _load_collector(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[Sequence[Any], Any, Any, Any, AbstractHazardModel]:
    hazard_manifest_path = _resolve(
        str(manifest["parents"]["hazard_manifest"]["path"]), root=root
    )
    hazard_receipt_path = _resolve(
        str(manifest["parents"]["hazard_compile_receipt"]["path"]), root=root
    )
    hazard_manifest = load_hazard_diversity_manifest(
        hazard_manifest_path, root=root, verify_code=False
    )
    hazard_receipt = load_hazard_diversity_receipt(
        hazard_receipt_path,
        manifest=hazard_manifest,
        root=root,
    )
    parent_manifest, _ = _parent_artifacts(hazard_manifest, root=root)
    witnesses, registry, posterior, shield = _load_frozen_inputs(
        parent_manifest, root=root
    )
    model_path = _resolve(
        str(hazard_receipt["artifacts"]["hazard_model"]["path"]), root=root
    )
    model = AbstractHazardModel.from_dict(_read_json(model_path))
    return witnesses, registry, posterior, shield, model


def preflight_future_viability_confirmation(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_prospective_manifest(manifest_path, root=repo_root)
    if not manifest["firewall"].get("preflight_authorized", False):
        raise ValueError("T12.6.1d manifest does not authorize preflight")
    protocol = FutureViabilityProspectiveProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1d preflight: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    future, immediate, incumbent, bundle = _load_model_bundle(manifest, root=repo_root)
    witnesses, _, _, _, hazard_model = _load_collector(manifest, root=repo_root)
    environment_path = _resolve(environments_dir, root=repo_root) / str(
        manifest["game_id"]
    )
    elapsed = max(0.0, time.monotonic() - started)
    checks = {
        "collector_environment_present": environment_path.is_dir(),
        "collector_has_two_lineage_witnesses": len(witnesses) == 2
        and {int(witness.source_seed) for witness in witnesses}
        == set(protocol.source_lineages),
        "hazard_model_loaded": bool(hazard_model.support),
        "model_bundle_checksum_match": bundle.get("bundle_checksum")
        == manifest["parents"]["reliability_model_bundle"]["bundle_checksum"],
        "model_refit_disabled": protocol.model_refit_authorized is False,
        "old_evaluation_archives_excluded": (
            protocol.old_evaluation_archives_authorized is False
            and "inputs" not in manifest
            and manifest["design"].get(
                "aborted_parent_archive_excluded_from_scoring", False
            )
        ),
        "prospective_seed_matrix_fresh": set(protocol.prospective_search_seeds)
        == {9_401, 9_402, 9_403}
        and not set(protocol.prospective_search_seeds)
        & set(protocol.retired_search_seeds),
        "reliability_model_is_strict": (
            future.minimum_exact_seed_span == 2
            and future.maximum_exact_label_range == 0.0
        ),
        "control_models_loaded": (
            immediate.target_field == "immediate_score"
            and incumbent.target_field == "productive_reach"
        ),
        "collection_not_executed": True,
        "sdk_budget_frozen": protocol.maximum_total_sdk_calls == 38_000,
        "cumulative_sdk_ledger_frozen": (
            protocol.parent_aborted_sdk_calls == 2_048
            and protocol.maximum_cumulative_sdk_calls == 40_048
        ),
        "cumulative_artifact_ledger_frozen": (
            protocol.parent_aborted_artifact_bytes == 20_911_530
            and protocol.maximum_cumulative_artifact_bytes == 1_094_653_354
        ),
        "wall_time_respected": elapsed <= protocol.maximum_offline_wall_seconds,
    }
    passed = all(checks.values())
    status = (
        "PASS_T12_6_1D_PREFLIGHT"
        if passed
        else "FAIL_T12_6_1D_PREFLIGHT_INTEGRITY_GATE"
    )
    report_path = destination / "preflight_report.json"
    metrics = {
        "checks": checks,
        "elapsed_seconds": elapsed,
        "environment_collection_executed": False,
        "model_bundle_checksum": bundle["bundle_checksum"],
        "maximum_cumulative_artifact_bytes": (
            protocol.maximum_cumulative_artifact_bytes
        ),
        "maximum_cumulative_sdk_calls": protocol.maximum_cumulative_sdk_calls,
        "parent_aborted_sdk_calls": protocol.parent_aborted_sdk_calls,
        "parent_aborted_artifact_bytes": protocol.parent_aborted_artifact_bytes,
        "pilot_collection_authorized": passed,
        "sdk_calls_used": 0,
        "witness_count": len(witnesses),
    }
    _write_json_once(
        report_path,
        {
            "format_version": "sage-t12.6.1d-preflight-report-v2",
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": metrics,
            "passed": passed,
            "protocol_checksum": manifest["protocol_checksum"],
            "status": status,
        },
        storage_budget=storage,
    )
    metrics["storage"] = storage.snapshot()
    receipt = prospective_receipt(
        manifest=manifest,
        phase="preflight",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={"report": _artifact(report_path)},
    )
    _write_json_once(
        destination / "preflight_receipt.json", receipt, storage_budget=storage
    )
    return receipt


def _batch_seeds(
    protocol: FutureViabilityProspectiveProtocol, batch: str
) -> tuple[int, ...]:
    if batch == "pilot":
        return protocol.pilot_search_seeds
    if batch == "completion":
        return protocol.completion_search_seeds
    raise ValueError("T12.6.1d batch must be pilot or completion")


def _collection_integrity_metrics(run: Any, *, arm: str) -> dict[str, Any]:
    """Project the canonical archive metrics without opening outcomes."""

    raw_metrics = run.metrics()
    return {
        "arm": str(arm),
        "candidate_catalog_checksum": str(run.candidate_catalog_checksum),
        "entry_exact": bool(run.entry_exact),
        "replay_exact_rate": float(raw_metrics["replay_exact_rate"]),
        "sdk_calls": int(raw_metrics["sdk_calls"]),
        "symbolic_cells": int(raw_metrics["symbolic_cells"]),
    }


def collect_future_viability_batch(
    *,
    manifest_path: str | Path,
    preflight_receipt_path: str | Path,
    output_dir: str | Path,
    batch: str,
    pilot_receipt_path: str | Path | None = None,
    environments_dir: str | Path = "environment_files",
    root: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_prospective_manifest(
        manifest_path, root=repo_root, verify_code=env_factory is None
    )
    protocol = FutureViabilityProspectiveProtocol(**dict(manifest["protocol"]))
    preflight = load_prospective_receipt(
        preflight_receipt_path,
        manifest=manifest,
        root=repo_root,
        expected_phase="preflight",
        require_passed=True,
    )
    if preflight.get("status") != "PASS_T12_6_1D_PREFLIGHT":
        raise ValueError("T12.6.1d collection requires the passed preflight")
    seeds = _batch_seeds(protocol, batch)
    pilot = None
    if batch == "completion":
        if pilot_receipt_path is None:
            raise ValueError("T12.6.1d completion requires the pilot receipt")
        pilot = load_prospective_receipt(
            pilot_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="collection_pilot",
            require_passed=True,
        )
        if pilot.get("status") != "PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY":
            raise ValueError("T12.6.1d pilot did not authorize completion")
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1d {batch} collection: "
            f"{destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    witnesses, registry, posterior, frozen_shield, hazard_model = _load_collector(
        manifest, root=repo_root
    )
    archives = []
    conditions = []
    total_sdk_calls = 0
    for search_seed in seeds:
        for witness in witnesses:
            arms = {}
            for arm in protocol.search_arms:
                shield = _clone_shield(frozen_shield)
                run = run_hazard_diversity_arm(
                    game_id=str(manifest["game_id"]),
                    witness=witness,
                    registry=registry,
                    posterior=posterior,
                    shield=shield,
                    hazard_model=hazard_model,
                    arm=arm,
                    search_seed=search_seed,
                    sdk_call_budget=protocol.sdk_calls_per_archive,
                    maximum_excursions=protocol.maximum_excursions_per_archive,
                    maximum_cells=protocol.maximum_cells_per_archive,
                    burst_schedule=protocol.burst_schedule,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
                path = (
                    destination
                    / str(manifest["game_id"])
                    / str(search_seed)
                    / str(witness.source_seed)
                    / f"{arm}.json"
                )
                artifact = _write_archive(path, run.archive, storage_budget=storage)
                artifact.update(
                    {
                        "arm": arm,
                        "lineage_seed": int(witness.source_seed),
                        "search_seed": int(search_seed),
                    }
                )
                archives.append(artifact)
                allowed_metrics = _collection_integrity_metrics(run, arm=arm)
                total_sdk_calls += allowed_metrics["sdk_calls"]
                arms[arm] = allowed_metrics
            conditions.append(
                {
                    "arms": arms,
                    "lineage_seed": int(witness.source_seed),
                    "search_seed": int(search_seed),
                }
            )
    elapsed = max(0.0, time.monotonic() - started)
    expected_count = (
        len(seeds) * len(protocol.source_lineages) * len(protocol.search_arms)
    )
    all_arms = [arm for condition in conditions for arm in condition["arms"].values()]
    prior_r1_sdk_calls = 0 if pilot is None else int(pilot["metrics"]["sdk_calls_used"])
    r1_sdk_calls_to_date = prior_r1_sdk_calls + total_sdk_calls
    cumulative_sdk_calls = protocol.parent_aborted_sdk_calls + r1_sdk_calls_to_date
    batch_archive_bytes = sum(
        _resolve(str(row["path"]), root=repo_root).stat().st_size for row in archives
    )
    prior_r1_archive_bytes = (
        0 if pilot is None else int(pilot["metrics"]["r1_archive_bytes_used"])
    )
    r1_archive_bytes = prior_r1_archive_bytes + batch_archive_bytes
    cumulative_archive_bytes = protocol.parent_aborted_artifact_bytes + r1_archive_bytes
    checks = {
        "all_anchor_replays_exact": all(bool(arm["entry_exact"]) for arm in all_arms),
        "all_archive_replays_exact": all(
            float(arm["replay_exact_rate"]) == 1.0 for arm in all_arms
        ),
        "archive_count_complete": len(archives) == expected_count,
        "batch_seed_registry_exact": {int(row["search_seed"]) for row in conditions}
        == set(seeds),
        "cell_budget_respected": all(
            int(arm["symbolic_cells"]) <= protocol.maximum_cells_per_archive
            for arm in all_arms
        ),
        "no_scores_or_labels_computed": True,
        "paired_candidate_catalogs_identical": all(
            len(
                {
                    condition["arms"][arm]["candidate_catalog_checksum"]
                    for arm in protocol.search_arms
                }
            )
            == 1
            for condition in conditions
        ),
        "per_archive_sdk_budget_respected": all(
            int(arm["sdk_calls"]) <= protocol.sdk_calls_per_archive for arm in all_arms
        ),
        "r1_total_sdk_budget_respected": r1_sdk_calls_to_date
        <= protocol.maximum_total_sdk_calls,
        "cumulative_sdk_budget_respected": cumulative_sdk_calls
        <= protocol.maximum_cumulative_sdk_calls,
        "r1_artifact_budget_respected": r1_archive_bytes
        <= protocol.maximum_artifact_bytes,
        "cumulative_artifact_budget_respected": cumulative_archive_bytes
        <= protocol.maximum_cumulative_artifact_bytes,
        "storage_budget_respected": bool(storage.snapshot()["within_budget"]),
        "wall_time_respected": elapsed <= protocol.maximum_wall_seconds_per_batch,
    }
    passed = all(checks.values())
    phase = f"collection_{batch}"
    status = (
        "PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY"
        if passed and batch == "pilot"
        else (
            "PASS_T12_6_1D_COMPLETION_COLLECTION_INTEGRITY"
            if passed
            else "FAIL_T12_6_1D_COLLECTION_INTEGRITY_GATE"
        )
    )
    report_path = destination / "collection_report.json"
    metrics = {
        "archive_count": len(archives),
        "batch": batch,
        "checks": checks,
        "completion_collection_authorized": bool(passed and batch == "pilot"),
        "batch_archive_bytes_used": batch_archive_bytes,
        "cumulative_artifact_bytes_used": cumulative_archive_bytes,
        "cumulative_sdk_calls_used": cumulative_sdk_calls,
        "elapsed_seconds": elapsed,
        "model_scores_computed": False,
        "parent_aborted_sdk_calls": protocol.parent_aborted_sdk_calls,
        "parent_aborted_artifact_bytes": protocol.parent_aborted_artifact_bytes,
        "productive_reach_labels_computed": False,
        "r1_archive_bytes_used": r1_archive_bytes,
        "sdk_calls_used": total_sdk_calls,
        "search_seeds": list(seeds),
        "storage": storage.snapshot(),
    }
    _write_json_once(
        report_path,
        {
            "conditions": conditions,
            "format_version": f"sage-t12.6.1d-{batch}-collection-report-v2",
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": metrics,
            "passed": passed,
            "protocol_checksum": manifest["protocol_checksum"],
            "status": status,
        },
        storage_budget=storage,
    )
    artifacts = {
        "archives": archives,
        "preflight_receipt": _artifact(Path(preflight_receipt_path)),
        "report": _artifact(report_path),
    }
    if pilot_receipt_path is not None:
        artifacts["pilot_receipt"] = _artifact(Path(pilot_receipt_path))
    receipt = prospective_receipt(
        manifest=manifest,
        phase=phase,
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts=artifacts,
    )
    _write_json_once(
        destination / "collection_receipt.json", receipt, storage_budget=storage
    )
    return receipt


def seal_future_viability_collection(
    *,
    manifest_path: str | Path,
    pilot_receipt_path: str | Path,
    completion_receipt_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_prospective_manifest(manifest_path, root=repo_root)
    protocol = FutureViabilityProspectiveProtocol(**dict(manifest["protocol"]))
    pilot = load_prospective_receipt(
        pilot_receipt_path,
        manifest=manifest,
        root=repo_root,
        expected_phase="collection_pilot",
        require_passed=True,
    )
    completion = load_prospective_receipt(
        completion_receipt_path,
        manifest=manifest,
        root=repo_root,
        expected_phase="collection_completion",
        require_passed=True,
    )
    if pilot.get("status") != "PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY":
        raise ValueError("T12.6.1d collection seal requires the passed pilot")
    if completion.get("status") != "PASS_T12_6_1D_COMPLETION_COLLECTION_INTEGRITY":
        raise ValueError("T12.6.1d collection seal requires the passed completion")
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1d collection seal: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    archives = [
        *pilot["artifacts"]["archives"],
        *completion["artifacts"]["archives"],
    ]
    conditions = {
        (int(row["search_seed"]), int(row["lineage_seed"]), str(row["arm"]))
        for row in archives
    }
    expected = {
        (seed, lineage, arm)
        for seed in protocol.prospective_search_seeds
        for lineage in protocol.source_lineages
        for arm in protocol.search_arms
    }
    unique_hashes = {str(row["sha256"]) for row in archives}
    total_sdk_calls = int(pilot["metrics"]["sdk_calls_used"]) + int(
        completion["metrics"]["sdk_calls_used"]
    )
    total_archive_bytes = sum(
        _resolve(str(row["path"]), root=repo_root).stat().st_size for row in archives
    )
    integrity = {
        "archive_condition_matrix_exact": conditions == expected,
        "archive_count_exact": len(archives) == protocol.expected_archive_count,
        "batch_seed_sets_disjoint": not set(pilot["metrics"]["search_seeds"])
        & set(completion["metrics"]["search_seeds"]),
        "batch_seed_sets_complete": set(pilot["metrics"]["search_seeds"])
        | set(completion["metrics"]["search_seeds"])
        == set(protocol.prospective_search_seeds),
        "model_scores_absent": (
            pilot["metrics"].get("model_scores_computed") is False
            and completion["metrics"].get("model_scores_computed") is False
        ),
        "productive_reach_labels_absent": (
            pilot["metrics"].get("productive_reach_labels_computed") is False
            and completion["metrics"].get("productive_reach_labels_computed") is False
        ),
        "sdk_budget_respected": total_sdk_calls <= protocol.maximum_total_sdk_calls,
        "cumulative_sdk_budget_respected": (
            protocol.parent_aborted_sdk_calls + total_sdk_calls
            <= protocol.maximum_cumulative_sdk_calls
        ),
        "storage_budget_respected": total_archive_bytes
        <= protocol.maximum_artifact_bytes,
        "cumulative_artifact_budget_respected": (
            protocol.parent_aborted_artifact_bytes + total_archive_bytes
            <= protocol.maximum_cumulative_artifact_bytes
        ),
    }
    support = {
        "unique_archive_support_sufficient": len(unique_hashes)
        >= protocol.minimum_unique_archive_count
    }
    integrity_passed = all(integrity.values())
    passed = bool(integrity_passed and all(support.values()))
    if not integrity_passed:
        status = "FAIL_T12_6_1D_COLLECTION_SEAL_INTEGRITY_GATE"
        classification = "PROSPECTIVE_COLLECTION_INTEGRITY_FAILURE"
    elif not passed:
        status = "FAIL_T12_6_1D_COLLECTION_DIVERSITY_GATE"
        classification = "INSUFFICIENT_UNIQUE_PROSPECTIVE_ARCHIVES"
    else:
        status = "PASS_T12_6_1D_COLLECTION_SEAL"
        classification = "PROSPECTIVE_COLLECTION_SEALED_LABEL_BLIND"
    report_path = destination / "collection_seal_report.json"
    metrics = {
        "archive_count": len(archives),
        "checks": {**integrity, **support},
        "classification": classification,
        "cumulative_sdk_calls_used": protocol.parent_aborted_sdk_calls
        + total_sdk_calls,
        "cumulative_artifact_bytes_used": protocol.parent_aborted_artifact_bytes
        + total_archive_bytes,
        "prediction_authorized": passed,
        "productive_reach_labels_computed": False,
        "sdk_calls_used": total_sdk_calls,
        "total_archive_bytes": total_archive_bytes,
        "unique_archive_count": len(unique_hashes),
    }
    _write_json_once(
        report_path,
        {
            "format_version": "sage-t12.6.1d-collection-seal-report-v2",
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": metrics,
            "passed": passed,
            "protocol_checksum": manifest["protocol_checksum"],
            "status": status,
        },
        storage_budget=storage,
    )
    receipt = prospective_receipt(
        manifest=manifest,
        phase="collection_seal",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "archives": archives,
            "completion_receipt": _artifact(Path(completion_receipt_path)),
            "pilot_receipt": _artifact(Path(pilot_receipt_path)),
            "report": _artifact(report_path),
        },
    )
    _write_json_once(
        destination / "collection_seal_receipt.json", receipt, storage_budget=storage
    )
    return receipt


def commit_future_viability_predictions(
    *,
    manifest_path: str | Path,
    collection_seal_receipt_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_prospective_manifest(manifest_path, root=repo_root)
    protocol = FutureViabilityProspectiveProtocol(**dict(manifest["protocol"]))
    collection = load_prospective_receipt(
        collection_seal_receipt_path,
        manifest=manifest,
        root=repo_root,
        expected_phase="collection_seal",
        require_passed=True,
    )
    if collection.get("status") != "PASS_T12_6_1D_COLLECTION_SEAL":
        raise ValueError("T12.6.1d prediction requires the sealed collection")
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1d predictions: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    extraction = extract_exact_state_candidates(
        archive_metas=collection["artifacts"]["archives"],
        root=repo_root,
        expected_search_seeds=protocol.prospective_search_seeds,
        expected_lineages=protocol.source_lineages,
        expected_arms=protocol.search_arms,
        future_horizon=protocol.future_horizon,
        local_radius=protocol.local_radius,
        include_labels=False,
    )
    future, immediate, incumbent, bundle = _load_model_bundle(manifest, root=repo_root)
    elapsed = max(0.0, time.monotonic() - started)
    checks = {
        "all_archive_conditions_present": bool(
            extraction.metrics["all_archive_conditions_present"]
        ),
        "candidate_registry_nonempty": bool(extraction.candidates),
        "exact_state_abstraction_conflicts_absent": int(
            extraction.metrics["exact_state_abstraction_conflicts"]
        )
        == 0,
        "exact_transition_conflicts_absent": int(
            extraction.metrics["exact_transition_conflicts"]
        )
        == 0,
        "labels_remain_closed": not extraction.labels,
        "raw_archive_count_exact": int(extraction.metrics["raw_archive_count"])
        == protocol.expected_archive_count,
        "unique_archive_support_sufficient": int(
            extraction.metrics["unique_archive_count"]
        )
        >= protocol.minimum_unique_archive_count,
        "wall_time_respected": elapsed <= protocol.maximum_offline_wall_seconds,
    }
    passed = all(checks.values())
    commitment_path = destination / "label_blind_predictions.sealed.json"
    report_path = destination / "prediction_report.json"
    artifacts: dict[str, Any] = {
        "collection_seal_receipt": _artifact(Path(collection_seal_receipt_path))
    }
    commitment_checksum = None
    if passed:
        core = commit_label_blind_predictions(
            extraction,
            future_model=future,
            immediate_model=immediate,
            incumbent_model=incumbent,
            binding_shift=protocol.binding_shift,
        )
        bound = _signed(
            {
                "collection_seal_receipt_checksum": collection["receipt_checksum"],
                "commitment": core,
                "format_version": PREDICTION_ARTIFACT_FORMAT,
                "manifest_checksum": manifest["manifest_checksum"],
                "model_bundle_checksum": bundle["bundle_checksum"],
                "protocol_checksum": manifest["protocol_checksum"],
            },
            "artifact_checksum",
        )
        _write_json_once(commitment_path, bound, storage_budget=storage)
        artifacts["predictions"] = _artifact(commitment_path)
        commitment_checksum = core["prediction_checksum"]
    status = (
        "PASS_T12_6_1D_LABEL_BLIND_PREDICTION_COMMITMENT"
        if passed
        else "FAIL_T12_6_1D_PREDICTION_INTEGRITY_GATE"
    )
    metrics = {
        **dict(extraction.metrics),
        "adjudication_authorized": passed,
        "checks": checks,
        "elapsed_seconds": elapsed,
        "labels_opened": False,
        "prediction_checksum": commitment_checksum,
        "sdk_calls_used": 0,
    }
    _write_json_once(
        report_path,
        {
            "format_version": "sage-t12.6.1d-prediction-report-v2",
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": metrics,
            "passed": passed,
            "protocol_checksum": manifest["protocol_checksum"],
            "status": status,
        },
        storage_budget=storage,
    )
    artifacts["report"] = _artifact(report_path)
    receipt = prospective_receipt(
        manifest=manifest,
        phase="prediction",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts=artifacts,
    )
    _write_json_once(
        destination / "prediction_receipt.json", receipt, storage_budget=storage
    )
    return receipt


def _load_bound_prediction(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    collection: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "artifact_checksum")
    if payload.get("format_version") != PREDICTION_ARTIFACT_FORMAT:
        raise ValueError("unsupported T12.6.1d bound prediction artifact")
    if (
        payload.get("manifest_checksum") != manifest.get("manifest_checksum")
        or payload.get("protocol_checksum") != manifest.get("protocol_checksum")
        or payload.get("collection_seal_receipt_checksum")
        != collection.get("receipt_checksum")
    ):
        raise ValueError("T12.6.1d prediction binding changed")
    verify_prediction_commitment(payload["commitment"])
    return payload


def classify_prospective_adjudication(
    *,
    protocol: FutureViabilityProspectiveProtocol,
    extraction_metrics: Mapping[str, Any],
    ranked: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Apply the frozen integrity, support, then superiority gates."""

    seed_metrics = dict(ranked["per_search_seed"])
    seed_gains = {
        seed: float(metrics["future_gain_over_incumbent"])
        for seed, metrics in seed_metrics.items()
    }
    integrity = {
        "all_archive_conditions_present": bool(
            extraction_metrics["all_archive_conditions_present"]
        ),
        "exact_state_abstraction_conflicts_absent": int(
            extraction_metrics.get("exact_state_abstraction_conflicts", 0)
        )
        == 0,
        "exact_transition_conflicts_absent": int(
            extraction_metrics["exact_transition_conflicts"]
        )
        == 0,
        "prediction_commitment_verified": True,
        "prospective_search_seeds_complete": set(seed_metrics)
        == {str(seed) for seed in protocol.prospective_search_seeds},
        "raw_archive_count_exact": int(extraction_metrics["raw_archive_count"])
        == protocol.expected_archive_count,
        "source_lineages_complete": set(ranked["per_lineage"])
        == {str(lineage) for lineage in protocol.source_lineages},
        "sdk_calls_absent": True,
        "wall_time_respected": float(elapsed_seconds)
        <= protocol.maximum_offline_wall_seconds,
    }
    support = {
        "eligible_groups_sufficient": int(ranked["eligible_groups"])
        >= protocol.minimum_eligible_groups,
        "exact_rejection_exercised": float(ranked["exact_rejection_exercised_rate"])
        >= protocol.minimum_exact_rejection_exercised_rate,
        "hierarchy_coverage_sufficient": float(ranked["hierarchy_coverage"])
        >= protocol.minimum_hierarchy_coverage,
        "recommendation_coverage_sufficient": float(ranked["recommendation_coverage"])
        >= protocol.minimum_recommendation_coverage,
        "unique_archive_support_sufficient": int(
            extraction_metrics["unique_archive_count"]
        )
        >= protocol.minimum_unique_archive_count,
        "unique_top_rate_sufficient": float(ranked["unique_top_rate"])
        >= protocol.minimum_unique_top_rate,
    }
    scientific = {
        "bootstrap_gain_lower_bound_nonnegative": float(
            ranked["bootstrap_gain_lower_bound_90"]
        )
        >= protocol.minimum_bootstrap_gain_lower_bound,
        "future_top1_sufficient": float(ranked["future_binding_top1_accuracy"])
        >= protocol.minimum_top1_accuracy,
        "gain_over_binding_swap_sufficient": float(
            ranked["future_gain_over_binding_swap"]
        )
        >= protocol.minimum_gain_over_binding_swap,
        "gain_over_exact_first_sufficient": float(ranked["future_gain_over_incumbent"])
        >= protocol.minimum_gain_over_exact_first,
        "gain_over_immediate_sufficient": float(ranked["future_gain_over_immediate"])
        >= protocol.minimum_gain_over_immediate,
        "seed_wins_over_exact_first_sufficient": sum(
            gain > 0.0 for gain in seed_gains.values()
        )
        >= protocol.minimum_seed_wins_over_exact_first,
        "worst_seed_exact_first_noninferiority": min(seed_gains.values(), default=-1.0)
        >= protocol.minimum_worst_seed_gain_over_exact_first,
        "every_lineage_accuracy_sufficient": all(
            float(metrics["future_binding_top1_accuracy"])
            >= protocol.minimum_lineage_accuracy
            for metrics in ranked["per_lineage"].values()
        ),
    }
    integrity_passed = all(integrity.values())
    support_passed = all(support.values())
    scientific_passed = all(scientific.values())
    passed = bool(integrity_passed and support_passed and scientific_passed)
    if not integrity_passed:
        classification = "PROSPECTIVE_ADJUDICATION_INTEGRITY_FAILURE"
        status = "FAIL_T12_6_1D_ADJUDICATION_INTEGRITY_GATE"
    elif not support_passed:
        classification = "INSUFFICIENT_PROSPECTIVE_SUPPORT"
        status = "FAIL_T12_6_1D_PROSPECTIVE_SUPPORT_GATE"
    elif not scientific_passed:
        classification = "NO_PROSPECTIVE_RELIABILITY_SUPERIORITY"
        status = "FAIL_T12_6_1D_PROSPECTIVE_SUPERIORITY_GATE"
    else:
        classification = "PROSPECTIVE_RELIABILITY_SUPERIORITY_CONFIRMED"
        status = "PASS_T12_6_1D_PROSPECTIVE_RELIABILITY_GATE"
    return {
        "checks": {**integrity, **support, **scientific},
        "classification": classification,
        "integrity_passed": integrity_passed,
        "passed": passed,
        "scientific_passed": scientific_passed,
        "status": status,
        "support_passed": support_passed,
    }


def adjudicate_future_viability_confirmation(
    *,
    manifest_path: str | Path,
    collection_seal_receipt_path: str | Path,
    prediction_receipt_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_prospective_manifest(manifest_path, root=repo_root)
    protocol = FutureViabilityProspectiveProtocol(**dict(manifest["protocol"]))
    collection = load_prospective_receipt(
        collection_seal_receipt_path,
        manifest=manifest,
        root=repo_root,
        expected_phase="collection_seal",
        require_passed=True,
    )
    prediction = load_prospective_receipt(
        prediction_receipt_path,
        manifest=manifest,
        root=repo_root,
        expected_phase="prediction",
        require_passed=True,
    )
    if collection.get("status") != "PASS_T12_6_1D_COLLECTION_SEAL":
        raise ValueError("T12.6.1d adjudication requires the sealed collection")
    if prediction.get("status") != "PASS_T12_6_1D_LABEL_BLIND_PREDICTION_COMMITMENT":
        raise ValueError("T12.6.1d adjudication requires committed predictions")
    prediction_path = _resolve(
        str(prediction["artifacts"]["predictions"]["path"]), root=repo_root
    )
    bound = _load_bound_prediction(
        prediction_path, manifest=manifest, collection=collection
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(
            f"refusing to append to immutable T12.6.1d adjudication: {destination}"
        )
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    extraction = extract_exact_state_candidates(
        archive_metas=collection["artifacts"]["archives"],
        root=repo_root,
        expected_search_seeds=protocol.prospective_search_seeds,
        expected_lineages=protocol.source_lineages,
        expected_arms=protocol.search_arms,
        future_horizon=protocol.future_horizon,
        local_radius=protocol.local_radius,
        include_labels=True,
    )
    result = adjudicate_prediction_commitment(
        bound["commitment"],
        extraction,
        bootstrap_repetitions=protocol.bootstrap_repetitions,
        bootstrap_seed=protocol.bootstrap_seed,
        bootstrap_lower_quantile=protocol.bootstrap_lower_quantile,
    )
    ranked = result["metrics"]
    elapsed = max(0.0, time.monotonic() - started)
    verdict = classify_prospective_adjudication(
        protocol=protocol,
        extraction_metrics=extraction.metrics,
        ranked=ranked,
        elapsed_seconds=elapsed,
    )
    passed = bool(verdict["passed"])
    classification = str(verdict["classification"])
    status = str(verdict["status"])
    rankings_path = destination / "adjudicated_rankings.json"
    report_path = destination / "adjudication_report.json"
    _write_json_once(
        rankings_path,
        {
            "cells": result["cells"],
            "format_version": "sage-t12.6.1d-adjudicated-rankings-v2",
            "manifest_checksum": manifest["manifest_checksum"],
            "prediction_checksum": bound["commitment"]["prediction_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    metrics = {
        **dict(extraction.metrics),
        **dict(ranked),
        "checks": dict(verdict["checks"]),
        "classification": classification,
        "controller_authority": False,
        "elapsed_seconds": elapsed,
        "neural_training_authorized": False,
        "production_authority": False,
        "sdk_calls_used": 0,
        "t12_6_2_freeze_authorized": passed,
    }
    metrics["storage"] = storage.snapshot()
    _write_json_once(
        report_path,
        {
            "claim_boundary": manifest["claim_boundary"],
            "format_version": "sage-t12.6.1d-adjudication-report-v2",
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": metrics,
            "passed": passed,
            "protocol_checksum": manifest["protocol_checksum"],
            "status": status,
        },
        storage_budget=storage,
    )
    receipt = prospective_receipt(
        manifest=manifest,
        phase="adjudication",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "collection_seal_receipt": _artifact(Path(collection_seal_receipt_path)),
            "prediction_receipt": _artifact(Path(prediction_receipt_path)),
            "rankings": _artifact(rankings_path),
            "report": _artifact(report_path),
        },
    )
    _write_json_once(
        destination / "adjudication_receipt.json", receipt, storage_budget=storage
    )
    return receipt


def future_viability_prospective_status(
    *,
    manifest_path: str | Path,
    preflight_receipt_path: str | Path | None = None,
    pilot_receipt_path: str | Path | None = None,
    completion_receipt_path: str | Path | None = None,
    collection_seal_receipt_path: str | Path | None = None,
    prediction_receipt_path: str | Path | None = None,
    adjudication_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_future_viability_prospective_manifest(manifest_path, root=repo_root)

    def optional(path: str | Path | None, phase: str) -> Mapping[str, Any] | None:
        if path is None or not Path(path).is_file():
            return None
        return load_prospective_receipt(
            path, manifest=manifest, root=repo_root, expected_phase=phase
        )

    preflight = optional(preflight_receipt_path, "preflight")
    pilot = optional(pilot_receipt_path, "collection_pilot")
    completion = optional(completion_receipt_path, "collection_completion")
    seal = optional(collection_seal_receipt_path, "collection_seal")
    prediction = optional(prediction_receipt_path, "prediction")
    adjudication = optional(adjudication_receipt_path, "adjudication")
    preflight_pass = bool(
        preflight
        and preflight.get("passed") is True
        and preflight.get("status") == "PASS_T12_6_1D_PREFLIGHT"
    )
    pilot_pass = bool(
        pilot
        and pilot.get("passed") is True
        and pilot.get("status") == "PASS_T12_6_1D_PILOT_COLLECTION_INTEGRITY"
    )
    completion_pass = bool(
        completion
        and completion.get("passed") is True
        and completion.get("status") == "PASS_T12_6_1D_COMPLETION_COLLECTION_INTEGRITY"
    )
    seal_pass = bool(
        seal
        and seal.get("passed") is True
        and seal.get("status") == "PASS_T12_6_1D_COLLECTION_SEAL"
    )
    prediction_pass = bool(
        prediction
        and prediction.get("passed") is True
        and prediction.get("status")
        == "PASS_T12_6_1D_LABEL_BLIND_PREDICTION_COMMITMENT"
    )
    adjudication_pass = bool(
        adjudication
        and adjudication.get("passed") is True
        and adjudication.get("status") == "PASS_T12_6_1D_PROSPECTIVE_RELIABILITY_GATE"
    )
    receipts = {}
    for name, receipt in (
        ("preflight", preflight),
        ("pilot", pilot),
        ("completion", completion),
        ("collection_seal", seal),
        ("prediction", prediction),
        ("adjudication", adjudication),
    ):
        receipts[name] = (
            None
            if receipt is None
            else {
                "passed": receipt["passed"],
                "receipt_checksum": receipt["receipt_checksum"],
                "status": receipt["status"],
            }
        )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "preflight_authorized": bool(
                manifest["firewall"].get("preflight_authorized", False)
                and preflight is None
            ),
            "pilot_collection_authorized": preflight_pass and pilot is None,
            "completion_collection_authorized": pilot_pass and completion is None,
            "collection_seal_authorized": pilot_pass
            and completion_pass
            and seal is None,
            "prediction_authorized": seal_pass and prediction is None,
            "adjudication_authorized": prediction_pass and adjudication is None,
            "t12_6_2_freeze_authorized": adjudication_pass,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
        },
        "format_version": "sage-t12.6.1d-prospective-status-v2",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": any(
            (
                bool(
                    manifest["firewall"].get("preflight_authorized", False)
                    and preflight is None
                ),
                preflight_pass and pilot is None,
                pilot_pass and completion is None,
                pilot_pass and completion_pass and seal is None,
                seal_pass and prediction is None,
                prediction_pass and adjudication is None,
                adjudication_pass,
            )
        ),
        "protocol_checksum": manifest["protocol_checksum"],
        "receipts": receipts,
    }


__all__ = [
    "PREDICTION_ARTIFACT_FORMAT",
    "_collection_integrity_metrics",
    "adjudicate_future_viability_confirmation",
    "classify_prospective_adjudication",
    "collect_future_viability_batch",
    "commit_future_viability_predictions",
    "future_viability_prospective_status",
    "preflight_future_viability_confirmation",
    "seal_future_viability_collection",
]
