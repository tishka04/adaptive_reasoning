"""Frozen T12.5c protocol for the paired goal-cursor binding control."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .experiment import (
    _bound_path,
    _file_sha256,
    _git_state,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .goal_cursor_control import ControlArm, ControlScheduleEntry
from .goal_viability_protocol import (
    GOAL_VIABILITY_CODE_PATHS,
    load_goal_viability_manifest,
    load_goal_viability_receipt,
)

GOAL_CURSOR_CONTROL_PROTOCOL_FORMAT = "sage-t12.5c-goal-cursor-control-protocol-v1"
GOAL_CURSOR_CONTROL_MANIFEST_FORMAT = "sage-t12.5c-goal-cursor-control-manifest-v1"
GOAL_CURSOR_CONTROL_RECEIPT_FORMAT = "sage-t12.5c-goal-cursor-control-receipt-v1"

GOAL_CURSOR_CONTROL_CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *GOAL_VIABILITY_CODE_PATHS,
            "theory/sage_t/causal/goal_cursor_control.py",
            "theory/sage_t/causal/goal_cursor_control_protocol.py",
            "theory/sage_t/causal/goal_cursor_control_experiment.py",
            "theory/sage_t/causal/goal_cursor_control_cli.py",
        )
    )
)


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _verified_artifact(meta: Mapping[str, Any], *, root: Path) -> dict[str, str]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.5c input artifact mismatch: {path}")
    return {"path": _bound_path(path, root=root), "sha256": expected}


@dataclass(frozen=True)
class GoalCursorControlProtocol:
    """Immutable equal-capacity control matrix and advancement firewall."""

    format_version: str = GOAL_CURSOR_CONTROL_PROTOCOL_FORMAT
    parent_status: str = "PASS_T12_5B_5_GOAL_VIABILITY_GATE"
    parent_classification: str = "GOAL_VIABILITY_CONTRAST_TRANSFERS"
    lineage_seeds: tuple[int, ...] = (8_701, 8_705)
    target_stage: int = 3
    goal_cursor_program: tuple[str, ...] = ("ACTION3", "ACTION3")
    binding_swap_program: tuple[str, ...] = ("ACTION4", "ACTION3")
    repetitions_per_arm_per_lineage: int = 2
    maximum_sdk_calls: int = 1_000
    maximum_wall_seconds: int = 7_200
    maximum_artifact_bytes: int = 3 * 1024 * 1024 * 1024
    allowed_effect_features: tuple[str, ...] = (
        "predicate_counts.adjacent",
        "predicate_counts.aligned",
        "predicate_counts.contact",
        "predicate_counts.near",
        "role_counts.clickable",
        "role_counts.movable",
    )
    persist_raw_frames: bool = False

    def __post_init__(self) -> None:
        for name in (
            "lineage_seeds",
            "goal_cursor_program",
            "binding_swap_program",
            "allowed_effect_features",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        expected = {
            "format_version": GOAL_CURSOR_CONTROL_PROTOCOL_FORMAT,
            "parent_status": "PASS_T12_5B_5_GOAL_VIABILITY_GATE",
            "parent_classification": "GOAL_VIABILITY_CONTRAST_TRANSFERS",
            "lineage_seeds": (8_701, 8_705),
            "target_stage": 3,
            "goal_cursor_program": ("ACTION3", "ACTION3"),
            "binding_swap_program": ("ACTION4", "ACTION3"),
            "repetitions_per_arm_per_lineage": 2,
            "maximum_sdk_calls": 1_000,
            "maximum_wall_seconds": 7_200,
            "maximum_artifact_bytes": 3 * 1024 * 1024 * 1024,
            "allowed_effect_features": (
                "predicate_counts.adjacent",
                "predicate_counts.aligned",
                "predicate_counts.contact",
                "predicate_counts.near",
                "role_counts.clickable",
                "role_counts.movable",
            ),
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.5c preregistered value changed: {name}")

    @property
    def arms(self) -> tuple[ControlArm, ...]:
        return (
            ControlArm("goal_cursor", self.goal_cursor_program, True),
            ControlArm("binding_swap", self.binding_swap_program, False),
        )

    @property
    def schedule(self) -> tuple[ControlScheduleEntry, ...]:
        order = (
            (8_701, "goal_cursor", 0),
            (8_701, "binding_swap", 0),
            (8_701, "binding_swap", 1),
            (8_701, "goal_cursor", 1),
            (8_705, "binding_swap", 0),
            (8_705, "goal_cursor", 0),
            (8_705, "goal_cursor", 1),
            (8_705, "binding_swap", 1),
        )
        return tuple(
            ControlScheduleEntry(index, lineage, arm, repetition)
            for index, (lineage, arm, repetition) in enumerate(order)
        )

    @property
    def expected_trials(self) -> int:
        return len(self.schedule)

    @property
    def context_id(self) -> str:
        return "stage_3_equal_capacity_goal_cursor_control"

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _sealed_parent_diagnostic(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    protocol: GoalCursorControlProtocol,
    root: Path,
) -> dict[str, Any]:
    registry = _read_json(_resolve_bound(artifacts["branches"]["path"], root=root))
    by_id = {str(item["branch_id"]): item for item in registry.get("branches", ())}
    progress = by_id.get("ACTION3>ACTION3")
    parent_control = by_id.get("ACTION4>ACTION3>ACTION3")
    if not isinstance(progress, Mapping) or not isinstance(parent_control, Mapping):
        raise ValueError("T12.5c parent viability pair changed")
    if progress.get("safe_progress") is not True or int(progress.get("level_delta", 0)) < 1:
        raise ValueError("T12.5c parent no longer proves goal-cursor progress")
    if parent_control.get("rejected") is not True or int(parent_control.get("level_delta", 0)) > 0:
        raise ValueError("T12.5c parent no longer rejects the cursor mismatch")
    if tuple(progress.get("program_actions", ())) != protocol.goal_cursor_program:
        raise ValueError("T12.5c treatment program changed")
    if tuple(parent_control.get("program_actions", ())) != (
        "ACTION4",
        "ACTION3",
        "ACTION3",
    ):
        raise ValueError("T12.5c parent control program changed")
    prefix_lengths = manifest["inputs"]["successful_prefix_lengths"]
    if {int(seed) for seed in prefix_lengths} != set(protocol.lineage_seeds):
        raise ValueError("T12.5c route lineages changed")
    metrics = dict(receipt.get("metrics", {}))
    checks = dict(metrics.get("checks", {}))
    if not checks or not all(checks.values()):
        raise ValueError("T12.5c requires an integrity-clean scientific parent")
    return {
        "arms": [arm.to_dict() for arm in protocol.arms],
        "binding_swap_definition": {
            "changed_slot": 0,
            "control_action": "ACTION4",
            "cursor_action": "ACTION3",
            "forced_cursor_consumption": True,
            "unchanged_slot": 1,
        },
        "capacity_match": {
            "maximum_action_slots_per_arm": 2,
            "same_anchor": True,
            "same_live_legal_inventory": True,
            "same_repetitions": True,
        },
        "parent_control_evidence_ids": list(parent_control.get("evidence_ids", ())),
        "parent_evaluation_receipt_checksum": receipt["receipt_checksum"],
        "parent_progress_evidence_ids": list(progress.get("evidence_ids", ())),
        "schedule": [entry.to_dict() for entry in protocol.schedule],
        "score_used_for_arm_assignment_or_labels": False,
    }


def freeze_goal_cursor_control(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: GoalCursorControlProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or GoalCursorControlProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    parent_manifest = load_goal_viability_manifest(
        parent_manifest_path,
        root=repo_root,
        verify_code=True,
    )
    parent_receipt = load_goal_viability_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
        expected_phase="evaluation",
    )
    metrics = dict(parent_receipt.get("metrics", {}))
    if parent_receipt.get("status") != selected.parent_status:
        raise ValueError("T12.5c parent status changed")
    if metrics.get("classification") != selected.parent_classification:
        raise ValueError("T12.5c parent classification changed")
    if metrics.get("t12_5c_control_freeze_authorized") is not True:
        raise ValueError("T12.5c parent does not authorize this freeze")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.5c is restricted to source_train")

    parent_artifacts = {
        name: _verified_artifact(meta, root=repo_root)
        for name, meta in parent_receipt.get("artifacts", {}).items()
    }
    if set(parent_artifacts) != {
        "branches",
        "calibration_receipt",
        "evaluation_registry",
        "report",
        "trials",
    }:
        raise ValueError("T12.5c parent artifact inventory changed")
    selection = _sealed_parent_diagnostic(
        manifest=parent_manifest,
        receipt=parent_receipt,
        artifacts=parent_artifacts,
        protocol=selected,
        root=repo_root,
    )
    runtime_parent = _verified_artifact(
        parent_manifest["parent"]["manifest"], root=repo_root
    )
    minimal_option = _verified_artifact(
        parent_manifest["inputs"]["minimal_option"], root=repo_root
    )
    missing = [
        path for path in GOAL_CURSOR_CONTROL_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.5c code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": GOAL_CURSOR_CONTROL_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_5C_PAIRED_CONTROL",
        "stage": "source_train",
        "game_id": parent_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "parent": {
            "manifest": {
                "path": _bound_path(parent_manifest_path, root=repo_root),
                "sha256": _file_sha256(parent_manifest_path),
                "manifest_checksum": parent_manifest["manifest_checksum"],
            },
            "receipt": {
                "path": _bound_path(parent_receipt_path, root=repo_root),
                "sha256": _file_sha256(parent_receipt_path),
                "receipt_checksum": parent_receipt["receipt_checksum"],
                "status": parent_receipt["status"],
                "classification": metrics["classification"],
                "t12_5c_control_freeze_authorized": True,
            },
            "artifacts": parent_artifacts,
        },
        "inputs": {
            "minimal_option": minimal_option,
            "runtime_parent_manifest": runtime_parent,
            "successful_prefix_lengths": {
                str(seed): int(
                    parent_manifest["inputs"]["successful_prefix_lengths"][str(seed)]
                )
                for seed in selected.lineage_seeds
            },
        },
        "selection": selection,
        "design": {
            "arm_labels_are_score_independent": True,
            "candidate_terminal_is_scientific_risk_not_integrity_failure": True,
            "fixed_counterbalanced_order": True,
            "goal_steps_are_reacquired_from_live_legal_actions": True,
            "maximum_capacity_is_equal": True,
            "parent_goal_viability_result_preserved": True,
            "single_binding_swap_only": True,
            "unavailable_action_is_missing_not_zero_effect": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in GOAL_CURSOR_CONTROL_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "paired_control_collection_authorized": authorized,
            "environment_collection_authorized": authorized,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "production_authority": False,
            "t12_6_freeze_authorized": False,
        },
        "claim_boundary": {
            "authorized": (
                "source-train paired causal control of the stage-3 goal-cursor "
                "binding on two confirmed bp35 route lineages"
            ),
            "not_authorized": [
                "generic ARC-AGI improvement",
                "autonomous environment control",
                "target-game generalization",
                "source validation",
                "holdout performance",
                "neural training",
                "production authority",
            ],
        },
        "storage": {
            "maximum_artifact_bytes": selected.maximum_artifact_bytes,
            "maximum_sdk_calls": selected.maximum_sdk_calls,
            "maximum_wall_seconds": selected.maximum_wall_seconds,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = goal_cursor_control_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_5C_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "arm_count": len(selected.arms),
            "expected_trials": selected.expected_trials,
            "maximum_action_slots_per_arm": 2,
            "parent_goal_viability_pass_preserved": True,
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_goal_cursor_control_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != GOAL_CURSOR_CONTROL_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.5c goal-cursor-control manifest")
    protocol = GoalCursorControlProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.5c protocol checksum mismatch")
    metas: list[tuple[str, Mapping[str, Any]]] = []
    for name, meta in manifest.get("parent", {}).items():
        if name == "artifacts" and isinstance(meta, Mapping):
            metas.extend((f"parent.{key}", value) for key, value in meta.items())
        elif isinstance(meta, Mapping) and "path" in meta:
            metas.append((f"parent.{name}", meta))
    metas.extend(
        (f"inputs.{name}", meta)
        for name, meta in manifest.get("inputs", {}).items()
        if isinstance(meta, Mapping) and "path" in meta
    )
    for name, meta in metas:
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5c bound artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.5c code checksum mismatch: {relative}")
    return manifest


def goal_cursor_control_receipt(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    passed: bool,
    status: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _signed(
        {
            "format_version": GOAL_CURSOR_CONTROL_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_5b_5_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_goal_cursor_control_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != GOAL_CURSOR_CONTROL_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.5c goal-cursor-control receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.5c receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.5c receipt phase mismatch")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.5c gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5c receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "GOAL_CURSOR_CONTROL_MANIFEST_FORMAT",
    "GOAL_CURSOR_CONTROL_PROTOCOL_FORMAT",
    "GOAL_CURSOR_CONTROL_RECEIPT_FORMAT",
    "GoalCursorControlProtocol",
    "freeze_goal_cursor_control",
    "goal_cursor_control_receipt",
    "load_goal_cursor_control_manifest",
    "load_goal_cursor_control_receipt",
]
