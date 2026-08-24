"""Two-phase physical experiment for SAGE.T12.5b.5 goal viability."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .goal_viability import (
    ViabilityBranch,
    audit_calibration_trials,
    audit_evaluation_trials,
    evaluation_registry_payload,
)
from .goal_viability_protocol import (
    GoalViabilityProtocol,
    _checksum,
    _resolve_bound,
    goal_viability_receipt,
    load_goal_viability_manifest,
    load_goal_viability_receipt,
    load_signed_evaluation_registry,
)
from .local_program_utility_experiment import (
    LocalProgramTrial,
    _annotate_context_exactness,
    _execute_program_trial,
    _runtime_inputs,
)
from .local_program_utility_protocol import (
    LocalProgramUtilityProtocol,
    load_local_program_utility_manifest,
)

EnvFactory = Callable[[str], Any]
CALIBRATION_TRIALS_FORMAT = "sage-t12.5b.5-calibration-trials-v1"
EVALUATION_TRIALS_FORMAT = "sage-t12.5b.5-evaluation-trials-v1"


@dataclass
class GoalViabilityBudget:
    maximum_sdk_calls: int
    maximum_wall_seconds: int
    used_sdk_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def consume(self, count: int = 1, *, reason: str = "replay") -> None:
        additional = max(0, int(count))
        if self.used_sdk_calls + additional > self.maximum_sdk_calls:
            raise RuntimeError(
                "T12.5b.5 SDK call budget exceeded: "
                f"used={self.used_sdk_calls} additional={additional} "
                f"maximum={self.maximum_sdk_calls} reason={reason}"
            )
        if self.elapsed_seconds > self.maximum_wall_seconds:
            raise RuntimeError(
                "T12.5b.5 wall-time budget exceeded before environment action"
            )
        self.used_sdk_calls += additional

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def snapshot(self) -> dict[str, Any]:
        elapsed = self.elapsed_seconds
        return {
            "elapsed_seconds": elapsed,
            "maximum_sdk_calls": self.maximum_sdk_calls,
            "maximum_wall_seconds": self.maximum_wall_seconds,
            "remaining_sdk_calls": self.maximum_sdk_calls - self.used_sdk_calls,
            "used_sdk_calls": self.used_sdk_calls,
            "within_sdk_budget": self.used_sdk_calls <= self.maximum_sdk_calls,
            "within_wall_time": elapsed <= self.maximum_wall_seconds,
        }


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _parent_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    verify_code: bool,
) -> dict[str, Any]:
    path = _resolve_bound(str(manifest["parent"]["manifest"]["path"]), root=root)
    return load_local_program_utility_manifest(
        path,
        root=root,
        verify_code=verify_code,
    )


def _collect_branches(
    *,
    phase: str,
    manifest: Mapping[str, Any],
    protocol: GoalViabilityProtocol,
    lineage_seed: int,
    branches: Sequence[ViabilityBranch],
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    budget: GoalViabilityBudget,
    root: Path,
) -> tuple[LocalProgramTrial, ...]:
    parent = _parent_manifest(
        manifest,
        root=root,
        verify_code=env_factory is None,
    )
    parent_protocol = LocalProgramUtilityProtocol(**dict(parent["protocol"]))
    witnesses, option_actions, _posterior, milestones, expected_hashes = (
        _runtime_inputs(parent, protocol=parent_protocol, root=root)
    )
    witness = witnesses[int(lineage_seed)]
    prefix_length = int(manifest["inputs"]["successful_prefix_lengths"][str(lineage_seed)])
    trials = tuple(
        replace(
            _execute_program_trial(
                phase=phase,
                game_id=str(manifest["game_id"]),
                environments_dir=environments_dir,
                env_factory=env_factory,
                witness=witness,
                witness_prefix_length=prefix_length,
                option_actions=option_actions,
                target_stage=protocol.target_stage,
                detour_action="ACTION4",
                detour_depth=0,
                actions=branch.program_actions,
                repetition=repetition,
                expected_stage_hash=expected_hashes[(lineage_seed, protocol.target_stage)],
                features=protocol.allowed_effect_features,
                milestones=milestones,
                budget=budget,
            ),
            context_id=protocol.context_id,
        )
        for branch in branches
        for repetition in range(protocol.repetitions_per_branch)
    )
    return _annotate_context_exactness(trials)


def run_goal_viability_calibration(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_goal_viability_manifest(
        manifest_path,
        root=repo_root,
        verify_code=env_factory is None,
    )
    if not manifest["firewall"].get("calibration_collection_authorized", False):
        raise ValueError("T12.5b.5 manifest does not authorize calibration")
    protocol = GoalViabilityProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_phase)
    budget = GoalViabilityBudget(
        maximum_sdk_calls=protocol.maximum_calibration_sdk_calls,
        maximum_wall_seconds=protocol.maximum_wall_seconds_per_phase,
    )
    trials = _collect_branches(
        phase="calibration",
        manifest=manifest,
        protocol=protocol,
        lineage_seed=protocol.calibration_lineage_seed,
        branches=protocol.calibration_branches,
        environments_dir=environments_dir,
        env_factory=env_factory,
        budget=budget,
        root=repo_root,
    )
    audit = audit_calibration_trials(
        trials=[item.to_dict() for item in trials],
        expected_branches=protocol.calibration_branches,
        repetitions_per_branch=protocol.repetitions_per_branch,
    )
    metrics = dict(audit["metrics"])
    sdk = budget.snapshot()
    integrity_checks = {
        "availability_is_deterministic": metrics["availability_is_deterministic"],
        "context_replay_is_exact": metrics["context_replay_is_exact"],
        "effects_are_deterministic": metrics["effects_are_deterministic"],
        "fixed_branch_schedule_completed": metrics["fixed_branch_schedule_completed"],
        "outcomes_are_deterministic": metrics["outcomes_are_deterministic"],
        "repetition_count_is_exact": metrics["repetition_count_is_exact"],
        "sdk_budget_respected": sdk["within_sdk_budget"],
        "transport_first_actions_available": metrics[
            "transport_first_actions_available"
        ],
        "wall_time_respected": sdk["within_wall_time"],
    }
    scientific_checks = {
        "cursor_advance_safe_progress_exists": metrics[
            "cursor_advance_safe_progress_count"
        ]
        >= 1,
        "goal_viability_contrast_exists": metrics["viability_contrast_count"] == 1,
    }
    integrity_passed = all(integrity_checks.values())
    progress_passed = scientific_checks["cursor_advance_safe_progress_exists"]
    contrast_passed = scientific_checks["goal_viability_contrast_exists"]
    passed = bool(integrity_passed and progress_passed and contrast_passed)
    if not integrity_passed:
        classification = "CALIBRATION_INTEGRITY_FAILURE"
        status = "FAIL_T12_5B_5_CALIBRATION_INTEGRITY_GATE"
    elif not progress_passed:
        classification = "GOAL_CONTINUATION_NO_SAFE_PROGRESS"
        status = "FAIL_T12_5B_5_GOAL_CONTINUATION_GATE"
    elif not contrast_passed:
        classification = "NO_GOAL_VIABILITY_CONTRAST"
        status = "FAIL_T12_5B_5_NO_VIABILITY_CONTRAST"
    else:
        classification = "GOAL_VIABILITY_CALIBRATED"
        status = "PASS_T12_5B_5_CALIBRATION_GATE"
    metrics.update(
        {
            "checks": {**integrity_checks, **scientific_checks},
            "classification": classification,
            "evaluation_collection_authorized": passed,
            "sdk_calls": sdk,
        }
    )

    trials_path = destination / "calibration_trials.json"
    registry_path = destination / "viability_branch_registry.json"
    report_path = destination / "calibration_report.json"
    evaluation_path = destination / "evaluation_registry.json"
    _write_json_once(
        trials_path,
        {
            "format_version": CALIBRATION_TRIALS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [item.to_dict() for item in trials],
        },
        storage_budget=storage,
    )
    branch_registry = {
        **audit["branch_registry"],
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
    }
    _write_json_once(registry_path, branch_registry, storage_budget=storage)
    artifacts = {
        "branches": _artifact(registry_path),
        "trials": _artifact(trials_path),
    }
    if passed:
        evidence_checksum = _checksum(
            {
                "branches_sha256": artifacts["branches"]["sha256"],
                "selection": audit["selection"],
                "trials_sha256": artifacts["trials"]["sha256"],
            }
        )
        evaluation_registry = evaluation_registry_payload(
            manifest_checksum=manifest["manifest_checksum"],
            protocol_checksum=manifest["protocol_checksum"],
            calibration_evidence_checksum=evidence_checksum,
            selection=audit["selection"],
        )
        _write_json_once(
            evaluation_path,
            evaluation_registry,
            storage_budget=storage,
        )
        artifacts["evaluation_registry"] = _artifact(evaluation_path)
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.5b.5-calibration-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "selection": audit["selection"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    artifacts["report"] = _artifact(report_path)
    receipt = goal_viability_receipt(
        manifest=manifest,
        phase="calibration",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts=artifacts,
    )
    _write_json_once(
        destination / "calibration_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def run_goal_viability_evaluation(
    *,
    manifest_path: str | Path,
    calibration_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_goal_viability_manifest(
        manifest_path,
        root=repo_root,
        verify_code=env_factory is None,
    )
    calibration = load_goal_viability_receipt(
        calibration_receipt_path,
        manifest=manifest,
        root=repo_root,
        require_passed=True,
        expected_phase="calibration",
    )
    if calibration.get("status") != "PASS_T12_5B_5_CALIBRATION_GATE":
        raise ValueError("T12.5b.5 evaluation requires the calibration pass")
    registry_meta = calibration["artifacts"].get("evaluation_registry")
    if not isinstance(registry_meta, Mapping):
        raise ValueError("T12.5b.5 calibration did not seal an evaluation registry")
    registry_path = Path(str(registry_meta["path"]))
    evaluation_registry = load_signed_evaluation_registry(
        registry_path,
        manifest=manifest,
    )
    protocol = GoalViabilityProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_phase)
    budget = GoalViabilityBudget(
        maximum_sdk_calls=protocol.maximum_evaluation_sdk_calls,
        maximum_wall_seconds=protocol.maximum_wall_seconds_per_phase,
    )
    registered = dict(evaluation_registry["branches"])
    branches = tuple(
        ViabilityBranch(
            branch_id=str(registered[name]["branch_id"]),
            first_action=str(registered[name]["first_action"]),
            program_actions=tuple(registered[name]["program_actions"]),
            goal_cursor_advance=bool(registered[name]["goal_cursor_advance"]),
            transport_eligible=bool(registered[name]["transport_eligible"]),
        )
        for name in ("progress", "control")
    )
    trials = _collect_branches(
        phase="evaluation",
        manifest=manifest,
        protocol=protocol,
        lineage_seed=protocol.evaluation_lineage_seed,
        branches=branches,
        environments_dir=environments_dir,
        env_factory=env_factory,
        budget=budget,
        root=repo_root,
    )
    audit = audit_evaluation_trials(
        trials=[item.to_dict() for item in trials],
        evaluation_registry=evaluation_registry,
        repetitions_per_branch=protocol.repetitions_per_branch,
    )
    metrics = dict(audit["metrics"])
    sdk = budget.snapshot()
    calibration_sdk_calls = int(
        calibration.get("metrics", {}).get("sdk_calls", {}).get("used_sdk_calls", 0)
    )
    total_sdk_calls = calibration_sdk_calls + int(sdk["used_sdk_calls"])
    integrity_checks = {
        "availability_is_deterministic": metrics["availability_is_deterministic"],
        "context_replay_is_exact": metrics["context_replay_is_exact"],
        "effects_are_deterministic": metrics["effects_are_deterministic"],
        "fixed_branch_schedule_completed": metrics["fixed_branch_schedule_completed"],
        "outcomes_are_deterministic": metrics["outcomes_are_deterministic"],
        "repetition_count_is_exact": metrics["repetition_count_is_exact"],
        "sdk_budget_respected": sdk["within_sdk_budget"],
        "total_sdk_budget_respected": total_sdk_calls
        <= protocol.maximum_total_sdk_calls,
        "transport_first_actions_available": metrics[
            "transport_first_actions_available"
        ],
        "wall_time_respected": sdk["within_wall_time"],
    }
    scientific_checks = {
        "control_branch_rejected": metrics["control_branch_rejected"],
        "goal_viability_contrast_transferred": metrics[
            "goal_viability_contrast_transferred"
        ],
        "progress_branch_transferred": metrics["progress_branch_transferred"],
    }
    integrity_passed = all(integrity_checks.values())
    progress_passed = scientific_checks["progress_branch_transferred"]
    control_passed = scientific_checks["control_branch_rejected"]
    contrast_passed = scientific_checks["goal_viability_contrast_transferred"]
    passed = bool(
        integrity_passed and progress_passed and control_passed and contrast_passed
    )
    if not integrity_passed:
        classification = "EVALUATION_INTEGRITY_FAILURE"
        status = "FAIL_T12_5B_5_EVALUATION_INTEGRITY_GATE"
    elif not progress_passed:
        classification = "GOAL_CONTINUATION_DID_NOT_TRANSFER"
        status = "FAIL_T12_5B_5_GOAL_CONTINUATION_TRANSFER_GATE"
    elif not control_passed:
        classification = "DETOUR_VIABILITY_CONTROL_UNSTABLE"
        status = "FAIL_T12_5B_5_DETOUR_CONTROL_GATE"
    elif not contrast_passed:
        classification = "GOAL_VIABILITY_CONTRAST_DID_NOT_TRANSFER"
        status = "FAIL_T12_5B_5_VIABILITY_TRANSFER_GATE"
    else:
        classification = "GOAL_VIABILITY_CONTRAST_TRANSFERS"
        status = "PASS_T12_5B_5_GOAL_VIABILITY_GATE"
    metrics.update(
        {
            "checks": {**integrity_checks, **scientific_checks},
            "classification": classification,
            "sdk_calls": sdk,
            "t12_5c_control_freeze_authorized": passed,
            "total_sdk_calls": total_sdk_calls,
        }
    )

    trials_path = destination / "evaluation_trials.json"
    registry_out_path = destination / "evaluation_branch_registry.json"
    report_path = destination / "evaluation_report.json"
    _write_json_once(
        trials_path,
        {
            "format_version": EVALUATION_TRIALS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [item.to_dict() for item in trials],
        },
        storage_budget=storage,
    )
    registry_out = {
        **audit["branch_registry"],
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "source_evaluation_registry_checksum": evaluation_registry[
            "registry_checksum"
        ],
    }
    _write_json_once(registry_out_path, registry_out, storage_budget=storage)
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.5b.5-evaluation-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "registered_control": audit["registered_control"],
        "registered_progress": audit["registered_progress"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    artifacts = {
        "calibration_receipt": _artifact(Path(calibration_receipt_path)),
        "evaluation_registry": _artifact(registry_path),
        "branches": _artifact(registry_out_path),
        "report": _artifact(report_path),
        "trials": _artifact(trials_path),
    }
    receipt = goal_viability_receipt(
        manifest=manifest,
        phase="evaluation",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts=artifacts,
    )
    _write_json_once(
        destination / "evaluation_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def goal_viability_status(
    *,
    manifest_path: str | Path,
    calibration_receipt_path: str | Path | None = None,
    evaluation_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_goal_viability_manifest(manifest_path, root=repo_root)
    calibration = (
        None
        if calibration_receipt_path is None
        or not Path(calibration_receipt_path).is_file()
        else load_goal_viability_receipt(
            calibration_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="calibration",
        )
    )
    evaluation = (
        None
        if evaluation_receipt_path is None
        or not Path(evaluation_receipt_path).is_file()
        else load_goal_viability_receipt(
            evaluation_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="evaluation",
        )
    )
    calibration_passed = bool(
        calibration
        and calibration.get("passed") is True
        and calibration.get("status") == "PASS_T12_5B_5_CALIBRATION_GATE"
    )
    evaluation_passed = bool(
        evaluation
        and evaluation.get("passed") is True
        and evaluation.get("status") == "PASS_T12_5B_5_GOAL_VIABILITY_GATE"
    )
    calibration_ready = bool(
        calibration is None
        and manifest["firewall"].get("calibration_collection_authorized", False)
    )
    evaluation_ready = bool(calibration_passed and evaluation is None)
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "calibration_collection_authorized": calibration_ready,
            "evaluation_collection_authorized": evaluation_ready,
            "environment_collection_authorized": calibration_ready or evaluation_ready,
            "causal_progress_control_authorized": False,
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_5c_control_freeze_authorized": evaluation_passed,
            "t12_6_freeze_authorized": False,
        },
        "format_version": "sage-t12.5b.5-goal-viability-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": evaluation_passed,
        "parent_t12_5b_4_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "calibration_receipt": (
            None
            if calibration is None
            else {
                "classification": calibration.get("metrics", {}).get("classification"),
                "passed": calibration["passed"],
                "receipt_checksum": calibration["receipt_checksum"],
                "status": calibration["status"],
            }
        ),
        "evaluation_receipt": (
            None
            if evaluation is None
            else {
                "classification": evaluation.get("metrics", {}).get("classification"),
                "passed": evaluation["passed"],
                "receipt_checksum": evaluation["receipt_checksum"],
                "status": evaluation["status"],
            }
        ),
    }


__all__ = [
    "GoalViabilityBudget",
    "goal_viability_status",
    "run_goal_viability_calibration",
    "run_goal_viability_evaluation",
]
