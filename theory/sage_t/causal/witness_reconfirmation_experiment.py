"""Exact paired replay for the two SAGE.T12.4a.2 progress witnesses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .witness_experiment import (
    ReplayTrial,
    SdkCallBudget,
    _intervention_bundles,
    _metrics,
    replay_trial,
)
from .witness_reconfirmation_protocol import (
    WitnessReconfirmationProtocol,
    load_reconfirmation_manifest,
    load_reconfirmation_receipt,
    load_reconfirmation_registry,
    reconfirmation_phase_receipt,
)

EnvFactory = Callable[[str], Any]


def _resolve_registry_path(manifest: Mapping[str, Any]) -> Path:
    path = Path(str(manifest["witness_registry"]["path"]))
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def run_witness_reconfirmation(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_reconfirmation_manifest(
        manifest_path,
        verify_code=env_factory is None,
    )
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.2 run requires a clean scientific freeze")
    if not manifest.get("firewall", {}).get(
        "witness_reconfirmation_authorized",
        False,
    ):
        raise ValueError("T12.4a.2 witness replay is not authorized")
    protocol = WitnessReconfirmationProtocol(**dict(manifest["protocol"]))
    _, witnesses = load_reconfirmation_registry(
        _resolve_registry_path(manifest),
        protocol=protocol,
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    budget = SdkCallBudget(protocol.maximum_sdk_calls)
    trials: list[ReplayTrial] = []
    suffix_length = len(protocol.expected_common_suffix)
    for witness in witnesses:
        for repetition in range(protocol.repetitions_per_route):
            trials.append(
                replay_trial(
                    witness=witness,
                    trial_type="full_route",
                    repetition=repetition,
                    budget=budget,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                    suffix_length=suffix_length,
                )
            )
        for trial_type in ("common_suffix", "delete_last_suffix_action"):
            for repetition in range(protocol.repetitions_per_suffix_branch):
                trials.append(
                    replay_trial(
                        witness=witness,
                        trial_type=trial_type,
                        repetition=repetition,
                        budget=budget,
                        environments_dir=environments_dir,
                        env_factory=env_factory,
                        suffix_length=suffix_length,
                    )
                )
    intervention_bundles = _intervention_bundles(
        witnesses=witnesses,
        trials=trials,
        suffix_length=suffix_length,
    )
    passed, metrics = _metrics(
        witnesses=witnesses,
        trials=trials,
        protocol=protocol,
        budget=budget,
        intervention_bundles=intervention_bundles,
    )
    metrics = {
        **metrics,
        "source_seeds": list(protocol.source_seeds),
        "route_lengths": list(protocol.expected_route_lengths),
        "common_suffix": list(protocol.expected_common_suffix),
        "common_suffix_length": suffix_length,
    }
    trial_path = destination / "replay_trials.json"
    _write_json_once(
        trial_path,
        {
            "format_version": "sage-t12.4a.2-replay-trials-v1",
            "trials": [trial.to_dict() for trial in trials],
        },
        storage_budget=storage,
    )
    bundle_path = destination / "intervention_bundles.json"
    _write_json_once(
        bundle_path,
        {
            "format_version": "sage-t12.4a.2-intervention-bundles-v1",
            "bundles": intervention_bundles,
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_4A_2_WITNESS_GATE"
        if passed
        else "FAIL_T12_4A_2_WITNESS_GATE"
    )
    report = {
        "format_version": "sage-t12.4a.2-witness-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_4a_1_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "collection_receipt_checksum": manifest["parent"]["collection_receipt"][
            "receipt_checksum"
        ],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "witness_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = reconfirmation_phase_receipt(
        manifest=manifest,
        phase="witness_reconfirmation",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "trials": {
                "path": str(trial_path.resolve()),
                "sha256": _file_sha256(trial_path),
            },
            "intervention_bundles": {
                "path": str(bundle_path.resolve()),
                "sha256": _file_sha256(bundle_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "witness_registry": dict(manifest["witness_registry"]),
        },
    )
    _write_json_once(
        destination / "witness_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def witness_reconfirmation_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_reconfirmation_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_reconfirmation_receipt(receipt_path, manifest=manifest)
    )
    freeze_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_2_FREEZE"
    )
    witness_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_2_WITNESS_GATE"
    )
    return {
        "format_version": "sage-t12.4a.2-witness-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_4a_1_status": manifest["parent"]["receipt"]["status"],
        "parent_failure_class": manifest["parent"]["receipt"]["failure_class"],
        "receipt": (
            None
            if receipt is None
            else {
                "phase": receipt["phase"],
                "passed": receipt["passed"],
                "status": receipt["status"],
                "receipt_checksum": receipt["receipt_checksum"],
            }
        ),
        "next_phase_authorized": bool(freeze_passed or witness_passed),
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "neural_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "calibration_training_authorized": False,
            "witness_reconfirmation_authorized": bool(
                manifest.get("scientific_claims_authorized", False)
            ),
            "option_extraction_authorized": False,
            "t12_4a_3_option_freeze_authorized": witness_passed,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
        },
    }


__all__ = ["run_witness_reconfirmation", "witness_reconfirmation_status"]
