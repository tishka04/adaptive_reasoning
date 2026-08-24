"""Physical paired-control runner for SAGE.T12.5c."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .goal_cursor_control import audit_goal_cursor_control
from .goal_cursor_control_protocol import (
    GoalCursorControlProtocol,
    _resolve_bound,
    goal_cursor_control_receipt,
    load_goal_cursor_control_manifest,
    load_goal_cursor_control_receipt,
)
from .local_program_utility_experiment import (
    LocalProgramTrial,
    _execute_program_trial,
    _runtime_inputs,
)
from .local_program_utility_protocol import (
    LocalProgramUtilityProtocol,
    load_local_program_utility_manifest,
)

EnvFactory = Callable[[str], Any]
CONTROL_TRIALS_FORMAT = "sage-t12.5c-goal-cursor-control-trials-v1"


@dataclass
class GoalCursorControlBudget:
    maximum_sdk_calls: int
    maximum_wall_seconds: int
    used_sdk_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def consume(self, count: int = 1, *, reason: str = "replay") -> None:
        additional = max(0, int(count))
        if self.used_sdk_calls + additional > self.maximum_sdk_calls:
            raise RuntimeError(
                "T12.5c SDK call budget exceeded: "
                f"used={self.used_sdk_calls} additional={additional} "
                f"maximum={self.maximum_sdk_calls} reason={reason}"
            )
        if self.elapsed_seconds > self.maximum_wall_seconds:
            raise RuntimeError(
                "T12.5c wall-time budget exceeded before environment action"
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


def _collect_control_trials(
    *,
    manifest: Mapping[str, Any],
    protocol: GoalCursorControlProtocol,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    budget: GoalCursorControlBudget,
    root: Path,
) -> tuple[LocalProgramTrial, ...]:
    parent_path = _resolve_bound(
        str(manifest["inputs"]["runtime_parent_manifest"]["path"]), root=root
    )
    parent = load_local_program_utility_manifest(
        parent_path,
        root=root,
        verify_code=env_factory is None,
    )
    parent_protocol = LocalProgramUtilityProtocol(**dict(parent["protocol"]))
    witnesses, option_actions, _posterior, milestones, expected_hashes = (
        _runtime_inputs(parent, protocol=parent_protocol, root=root)
    )
    arm_by_name = {arm.name: arm for arm in protocol.arms}
    trials: list[LocalProgramTrial] = []
    for entry in protocol.schedule:
        lineage = int(entry.lineage_seed)
        arm = arm_by_name[entry.arm_name]
        trial = _execute_program_trial(
            phase="paired_control",
            game_id=str(manifest["game_id"]),
            environments_dir=environments_dir,
            env_factory=env_factory,
            witness=witnesses[lineage],
            witness_prefix_length=int(
                manifest["inputs"]["successful_prefix_lengths"][str(lineage)]
            ),
            option_actions=option_actions,
            target_stage=protocol.target_stage,
            detour_action="ACTION4",
            detour_depth=0,
            actions=arm.program_actions,
            repetition=entry.repetition,
            expected_stage_hash=expected_hashes[(lineage, protocol.target_stage)],
            features=protocol.allowed_effect_features,
            milestones=milestones,
            budget=budget,
        )
        trials.append(
            replace(
                trial,
                context_id=protocol.context_id,
                trial_id=(
                    f"paired_{entry.order_index:02d}_lineage_{lineage}_"
                    f"{arm.name}_rep_{entry.repetition}"
                ),
            )
        )
    return tuple(trials)


def run_goal_cursor_control(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_goal_cursor_control_manifest(
        manifest_path,
        root=repo_root,
        verify_code=env_factory is None,
    )
    if not manifest["firewall"].get("paired_control_collection_authorized", False):
        raise ValueError("T12.5c manifest does not authorize paired collection")
    protocol = GoalCursorControlProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes)
    budget = GoalCursorControlBudget(
        maximum_sdk_calls=protocol.maximum_sdk_calls,
        maximum_wall_seconds=protocol.maximum_wall_seconds,
    )
    trials = _collect_control_trials(
        manifest=manifest,
        protocol=protocol,
        environments_dir=environments_dir,
        env_factory=env_factory,
        budget=budget,
        root=repo_root,
    )
    audit = audit_goal_cursor_control(
        trials=[item.to_dict() for item in trials],
        arms=protocol.arms,
        schedule=protocol.schedule,
    )
    metrics = dict(audit["metrics"])
    sdk = budget.snapshot()
    integrity_checks = {
        "actions_available_or_terminal": metrics["actions_available_or_terminal"],
        "availability_is_deterministic": metrics["availability_is_deterministic"],
        "context_replay_is_exact": metrics["context_replay_is_exact"],
        "effects_are_deterministic": metrics["effects_are_deterministic"],
        "equal_capacity_horizon": metrics["equal_capacity_horizon"],
        "fixed_counterbalanced_schedule_completed": metrics[
            "fixed_counterbalanced_schedule_completed"
        ],
        "outcomes_are_deterministic": metrics["outcomes_are_deterministic"],
        "repetition_count_is_exact": metrics["repetition_count_is_exact"],
        "sdk_budget_respected": sdk["within_sdk_budget"],
        "wall_time_respected": sdk["within_wall_time"],
    }
    scientific_checks = {
        "binding_swap_control_rejected": metrics[
            "binding_swap_control_rejected"
        ],
        "goal_cursor_safe_progress": metrics["goal_cursor_safe_progress"],
        "paired_advantage_all_lineages": metrics[
            "paired_advantage_all_lineages"
        ],
    }
    integrity_passed = all(integrity_checks.values())
    treatment_passed = scientific_checks["goal_cursor_safe_progress"]
    control_passed = scientific_checks["binding_swap_control_rejected"]
    paired_passed = scientific_checks["paired_advantage_all_lineages"]
    passed = bool(
        integrity_passed and treatment_passed and control_passed and paired_passed
    )
    if not integrity_passed:
        classification = "PAIRED_CONTROL_INTEGRITY_FAILURE"
        status = "FAIL_T12_5C_COLLECTION_INTEGRITY_GATE"
    elif not treatment_passed:
        classification = "GOAL_CURSOR_DID_NOT_PRESERVE_PROGRESS"
        status = "FAIL_T12_5C_GOAL_CURSOR_PROGRESS_GATE"
    elif not control_passed:
        classification = "BINDING_SWAP_DID_NOT_REMOVE_PROGRESS"
        status = "FAIL_T12_5C_BINDING_SWAP_CONTROL_GATE"
    elif not paired_passed:
        classification = "NO_PAIRED_GOAL_CURSOR_ADVANTAGE"
        status = "FAIL_T12_5C_PAIRED_ADVANTAGE_GATE"
    else:
        classification = "GOAL_CURSOR_BINDING_CAUSALLY_SUPPORTED"
        status = "PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE"
    metrics.update(
        {
            "checks": {**integrity_checks, **scientific_checks},
            "classification": classification,
            "sdk_calls": sdk,
            "t12_6_freeze_authorized": passed,
        }
    )

    trials_path = destination / "control_trials.json"
    registry_path = destination / "control_arm_registry.json"
    report_path = destination / "control_report.json"
    _write_json_once(
        trials_path,
        {
            "format_version": CONTROL_TRIALS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "schedule": [entry.to_dict() for entry in protocol.schedule],
            "trials": [item.to_dict() for item in trials],
        },
        storage_budget=storage,
    )
    registry = {
        **audit["arm_registry"],
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
    }
    _write_json_once(registry_path, registry, storage_budget=storage)
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.5c-goal-cursor-control-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "paired_lineages": audit["arm_registry"]["paired_lineages"],
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    artifacts = {
        "arms": _artifact(registry_path),
        "parent_evaluation_receipt": _artifact(
            _resolve_bound(str(manifest["parent"]["receipt"]["path"]), root=repo_root)
        ),
        "report": _artifact(report_path),
        "trials": _artifact(trials_path),
    }
    receipt = goal_cursor_control_receipt(
        manifest=manifest,
        phase="paired_control",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts=artifacts,
    )
    _write_json_once(
        destination / "control_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def goal_cursor_control_status(
    *,
    manifest_path: str | Path,
    control_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_goal_cursor_control_manifest(manifest_path, root=repo_root)
    receipt = (
        None
        if control_receipt_path is None or not Path(control_receipt_path).is_file()
        else load_goal_cursor_control_receipt(
            control_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="paired_control",
        )
    )
    passed = bool(
        receipt
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE"
    )
    collection_ready = bool(
        receipt is None
        and manifest["firewall"].get("paired_control_collection_authorized", False)
    )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "paired_control_collection_authorized": collection_ready,
            "environment_collection_authorized": collection_ready,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "t12_6_freeze_authorized": passed,
        },
        "format_version": "sage-t12.5c-goal-cursor-control-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": passed,
        "parent_t12_5b_5_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "control_receipt": (
            None
            if receipt is None
            else {
                "classification": receipt.get("metrics", {}).get("classification"),
                "passed": receipt["passed"],
                "receipt_checksum": receipt["receipt_checksum"],
                "status": receipt["status"],
            }
        ),
    }


__all__ = [
    "GoalCursorControlBudget",
    "goal_cursor_control_status",
    "run_goal_cursor_control",
]
