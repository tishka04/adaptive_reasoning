"""Frozen T12.3d protocol for confirmed replay-control provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .contracts import GroundedAction
from .experiment import (
    _bound_path,
    _file_sha256,
    _git_state,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .lineage_protocol import (
    ReplayLineageProtocol,
    load_lineage_manifest,
    load_lineage_receipt,
)
from .shield_protocol import load_shield_manifest
from .witness_protocol import (
    ProgressWitness,
    load_witness_manifest,
    load_witness_receipt,
    load_witness_registry,
)

PROVENANCE_PROTOCOL_FORMAT = "sage-t12.3d-confirmed-control-protocol-v1"
PROVENANCE_REGISTRY_FORMAT = "sage-t12.3d-confirmed-control-registry-v1"
PROVENANCE_MANIFEST_FORMAT = "sage-t12.3d-confirmed-control-manifest-v1"
PROVENANCE_RECEIPT_FORMAT = "sage-t12.3d-confirmed-control-receipt-v1"

PROVENANCE_CODE_PATHS = (
    "theory/sage_t/causal/provenance_protocol.py",
    "theory/sage_t/causal/provenance_experiment.py",
    "theory/sage_t/causal/provenance_experiment_cli.py",
    "theory/sage_t/causal/lineage_archive.py",
    "theory/sage_t/causal/lineage_experiment.py",
    "theory/sage_t/causal/lineage_protocol.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/burst_experiment.py",
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/shield_protocol.py",
    "theory/sage_t/causal/witness_protocol.py",
    "theory/sage/live_prefix_counterfactual_collector.py",
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


@dataclass(frozen=True)
class ConfirmedReplayControl:
    control_id: str
    witness_id: str
    game_id: str
    route_checksum: str
    source_seed: int
    source_arm: str
    prior_route_confirmations: int
    actions: tuple[GroundedAction, ...]
    expected_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(
            self, "expected_hashes", tuple(str(value) for value in self.expected_hashes)
        )
        if not self.control_id or not self.witness_id or not self.route_checksum:
            raise ValueError("confirmed replay control identity is incomplete")
        if not self.actions or len(self.expected_hashes) != len(self.actions) + 1:
            raise ValueError("confirmed replay control hash path is incomplete")
        if self.prior_route_confirmations < 3:
            raise ValueError("confirmed replay control lacks three prior confirmations")

    @property
    def depth(self) -> int:
        return len(self.actions)

    @property
    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "witness_id": self.witness_id,
            "game_id": self.game_id,
            "route_checksum": self.route_checksum,
            "source_seed": self.source_seed,
            "source_arm": self.source_arm,
            "prior_route_confirmations": self.prior_route_confirmations,
            "actions": [
                {
                    "action_name": action.action_name,
                    "action_data": dict(action.action_data),
                }
                for action in self.actions
            ],
            "expected_hashes": list(self.expected_hashes),
        }

    @property
    def control_checksum(self) -> str:
        return _checksum(self.unsigned_payload)

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_payload, "control_checksum": self.control_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConfirmedReplayControl:
        control = cls(
            control_id=str(payload["control_id"]),
            witness_id=str(payload["witness_id"]),
            game_id=str(payload["game_id"]),
            route_checksum=str(payload["route_checksum"]),
            source_seed=int(payload["source_seed"]),
            source_arm=str(payload["source_arm"]),
            prior_route_confirmations=int(payload["prior_route_confirmations"]),
            actions=tuple(
                GroundedAction(
                    str(item["action_name"]),
                    dict(item.get("action_data", {}) or {}),
                )
                for item in payload["actions"]
            ),
            expected_hashes=tuple(str(value) for value in payload["expected_hashes"]),
        )
        if control.control_checksum != str(payload.get("control_checksum", "")):
            raise ValueError("confirmed replay control checksum mismatch")
        return control


@dataclass(frozen=True)
class ConfirmedControlProtocol:
    format_version: str = PROVENANCE_PROTOCOL_FORMAT
    expected_unique_controls: int = 2
    control_repetitions: int = 3
    minimum_prior_route_confirmations: int = 3
    evaluation_seeds: tuple[int, ...] = (7401, 7402, 7403)
    evaluation_arms: tuple[str, ...] = (
        "shortest_prefix_control",
        "lineage_preserving",
    )
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    sdk_calls_per_evaluation_arm: int = 3_500
    maximum_total_sdk_calls: int = 30_000
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_cells: int = 50_000
    minimum_confirmed_control_exact_rate: float = 0.95
    minimum_treatment_replay_exact_rate: float = 0.95
    maximum_per_seed_replay_regression: float = 0.0
    minimum_coverage_ratio: float = 0.80
    maximum_progress_regression_seeds: int = 0
    minimum_rebases_avoided: int = 1
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in ("evaluation_seeds", "burst_schedule"):
            object.__setattr__(
                self, name, tuple(int(value) for value in getattr(self, name))
            )
        object.__setattr__(
            self,
            "evaluation_arms",
            tuple(str(value) for value in self.evaluation_arms),
        )
        if self.format_version != PROVENANCE_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.3d confirmed-control protocol")
        fixed = {
            "expected_unique_controls": (self.expected_unique_controls, 2),
            "control_repetitions": (self.control_repetitions, 3),
            "minimum_prior_route_confirmations": (
                self.minimum_prior_route_confirmations,
                3,
            ),
            "sdk_calls_per_evaluation_arm": (
                self.sdk_calls_per_evaluation_arm,
                3_500,
            ),
            "maximum_total_sdk_calls": (self.maximum_total_sdk_calls, 30_000),
            "maximum_artifact_bytes_per_run": (
                self.maximum_artifact_bytes_per_run,
                3 * 1024 * 1024 * 1024,
            ),
            "maximum_cells": (self.maximum_cells, 50_000),
            "minimum_confirmed_control_exact_rate": (
                self.minimum_confirmed_control_exact_rate,
                0.95,
            ),
            "minimum_treatment_replay_exact_rate": (
                self.minimum_treatment_replay_exact_rate,
                0.95,
            ),
            "maximum_per_seed_replay_regression": (
                self.maximum_per_seed_replay_regression,
                0.0,
            ),
            "minimum_coverage_ratio": (self.minimum_coverage_ratio, 0.80),
            "maximum_progress_regression_seeds": (
                self.maximum_progress_regression_seeds,
                0,
            ),
            "minimum_rebases_avoided": (self.minimum_rebases_avoided, 1),
        }
        for name, (observed, expected) in fixed.items():
            if observed != expected:
                raise ValueError(f"T12.3d preregistered value changed: {name}")
        if self.evaluation_seeds != (7401, 7402, 7403):
            raise ValueError("T12.3d prospective seeds are frozen")
        if self.evaluation_arms != (
            "shortest_prefix_control",
            "lineage_preserving",
        ):
            raise ValueError("T12.3d evaluation arms are frozen")
        if self.burst_schedule != (4, 8, 16):
            raise ValueError("T12.3d burst schedule is frozen")
        control_upper_bound = (
            self.expected_unique_controls
            * self.control_repetitions
            * (1 + 128)
        )
        evaluation_upper_bound = (
            len(self.evaluation_seeds)
            * len(self.evaluation_arms)
            * self.sdk_calls_per_evaluation_arm
        )
        if control_upper_bound + evaluation_upper_bound > self.maximum_total_sdk_calls:
            raise ValueError("T12.3d planned calls exceed the global SDK budget")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _parent_failure_is_control_provenance_only(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    if receipt.get("passed") is not False:
        return False
    if receipt.get("status") != "FAIL_T12_3C_REPLAY_LINEAGE_GATE":
        return False
    protocol = ReplayLineageProtocol(**dict(manifest["protocol"]))
    metrics = dict(receipt.get("metrics", {}))
    matched_rate = float(metrics.get("matched_control_exact_rate", 1.0))
    checks = (
        int(metrics.get("reproduced_parent_failures", 0))
        >= protocol.minimum_reproduced_parent_failures,
        float(metrics.get("minimum_treatment_replay_exact_rate", 0.0))
        >= protocol.minimum_treatment_replay_exact_rate,
        float(metrics.get("calibration_seed_replay_gain", float("-inf")))
        >= protocol.minimum_calibration_seed_replay_gain,
        float(metrics.get("minimum_per_seed_coverage_ratio", 0.0))
        >= protocol.minimum_coverage_ratio,
        int(metrics.get("progress_regression_seeds", 99))
        <= protocol.maximum_progress_regression_seeds,
        int(metrics.get("lineage_attached_transitions", 0)) > 0,
        int(metrics.get("shortest_prefix_rebases_avoided", 0)) > 0,
        int(metrics.get("lineage_rebased_transitions", 1)) == 0,
        int(metrics.get("sdk_calls", protocol.maximum_total_sdk_calls + 1))
        <= protocol.maximum_total_sdk_calls,
    )
    return bool(
        all(checks)
        and matched_rate < protocol.minimum_treatment_replay_exact_rate
    )


def _confirmed_controls(
    witnesses: tuple[ProgressWitness, ...],
    witness_receipt: Mapping[str, Any],
    *,
    protocol: ConfirmedControlProtocol,
) -> tuple[ConfirmedReplayControl, ...]:
    metrics = dict(witness_receipt.get("metrics", {}))
    if float(metrics.get("step_exact_rate", 0.0)) != 1.0:
        raise ValueError("T12.3d source witness step exact rate is not one")
    per_witness = {
        str(item["witness_id"]): dict(item)
        for item in metrics.get("per_witness", ())
    }
    by_route: dict[str, ConfirmedReplayControl] = {}
    for witness in witnesses:
        prior = per_witness.get(witness.witness_id)
        if prior is None:
            raise ValueError("T12.3d witness lacks confirmation provenance")
        confirmations = int(prior.get("route_confirmations", 0))
        if confirmations < protocol.minimum_prior_route_confirmations:
            raise ValueError("T12.3d witness lacks three exact route confirmations")
        expected = (witness.initial_exact_hash,) + tuple(
            step.expected_target_hash for step in witness.steps
        )
        control = ConfirmedReplayControl(
            control_id=f"control_{witness.route_checksum[:20]}",
            witness_id=witness.witness_id,
            game_id=witness.game_id,
            route_checksum=witness.route_checksum,
            source_seed=witness.source_seed,
            source_arm=witness.source_arm,
            prior_route_confirmations=confirmations,
            actions=tuple(step.action for step in witness.steps),
            expected_hashes=expected,
        )
        by_route.setdefault(witness.route_checksum, control)
    controls = tuple(by_route[key] for key in sorted(by_route))
    if len(controls) != protocol.expected_unique_controls:
        raise ValueError("T12.3d requires exactly two unique confirmed controls")
    if len({item.control_checksum for item in controls}) != len(controls):
        raise ValueError("T12.3d confirmed controls are not checksum-unique")
    return controls


def freeze_provenance_experiment(
    *,
    output_path: str | Path,
    control_registry_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: ConfirmedControlProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or ConfirmedControlProtocol()
    parent_manifest = load_lineage_manifest(
        parent_manifest_path, root=repo_root, verify_code=False
    )
    parent_receipt = load_lineage_receipt(
        parent_receipt_path, manifest=parent_manifest, root=repo_root
    )
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.3d is restricted to source_train")
    if not _parent_failure_is_control_provenance_only(
        parent_manifest, parent_receipt
    ):
        raise ValueError("T12.3d requires the control-provenance-only T12.3c failure")

    shield_manifest_path = _resolve_bound(
        str(parent_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    shield_manifest = load_shield_manifest(
        shield_manifest_path, root=repo_root, verify_code=False
    )
    witness_manifest_path = _resolve_bound(
        str(shield_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    witness_receipt_path = _resolve_bound(
        str(shield_manifest["parent"]["receipt"]["path"]), root=repo_root
    )
    witness_manifest = load_witness_manifest(witness_manifest_path, root=repo_root)
    witness_receipt = load_witness_receipt(
        witness_receipt_path, manifest=witness_manifest, root=repo_root
    )
    if witness_receipt.get("passed") is not True or witness_receipt.get(
        "status"
    ) != "PASS_T12_3A_WITNESS_GATE":
        raise ValueError("T12.3d confirmed-control source did not pass T12.3a")
    witness_registry_path = _resolve_bound(
        str(witness_manifest["witness_registry"]["path"]), root=repo_root
    )
    _, witnesses = load_witness_registry(witness_registry_path)
    controls = _confirmed_controls(witnesses, witness_receipt, protocol=selected)

    missing = [path for path in PROVENANCE_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.3d code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(
        not git["dirty"]
        and parent_manifest.get("scientific_claims_authorized", False)
        and witness_receipt.get("passed") is True
    )
    registry = _signed(
        {
            "format_version": PROVENANCE_REGISTRY_FORMAT,
            "protocol_checksum": selected.checksum,
            "parent_t12_3c_receipt_checksum": parent_receipt["receipt_checksum"],
            "source_t12_3a_receipt_checksum": witness_receipt["receipt_checksum"],
            "controls": [item.to_dict() for item in controls],
        },
        "registry_checksum",
    )
    _write_json_once(control_registry_path, registry)
    payload = {
        "format_version": PROVENANCE_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_3D_CONFIRMED_CONTROL",
        "stage": "source_train",
        "game_id": parent_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "control_registry": {
            "path": _bound_path(control_registry_path, root=repo_root),
            "sha256": _file_sha256(control_registry_path),
            "registry_checksum": registry["registry_checksum"],
        },
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
                "passed": False,
                "status": "FAIL_T12_3C_REPLAY_LINEAGE_GATE",
                "failure_class": "CONFIRMED_CONTROL_PROVENANCE_ONLY",
            },
        },
        "control_source_t12_3a": {
            "manifest": {
                "path": _bound_path(witness_manifest_path, root=repo_root),
                "sha256": _file_sha256(witness_manifest_path),
                "manifest_checksum": witness_manifest["manifest_checksum"],
            },
            "receipt": {
                "path": _bound_path(witness_receipt_path, root=repo_root),
                "sha256": _file_sha256(witness_receipt_path),
                "receipt_checksum": witness_receipt["receipt_checksum"],
                "passed": True,
                "status": "PASS_T12_3A_WITNESS_GATE",
            },
            "registry": {
                "path": _bound_path(witness_registry_path, root=repo_root),
                "sha256": _file_sha256(witness_registry_path),
            },
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in PROVENANCE_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "confirmed_control_experiment_authorized": authorized,
            "t12_3b_child_rerun_authorized": False,
            "terminal_shield_production_authority": False,
            "neural_training_authorized": False,
            "option_extraction_authorized": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    freeze_receipt = provenance_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_3D_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "unique_confirmed_controls": len(controls),
            "prior_confirmations": sum(
                item.prior_route_confirmations for item in controls
            ),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), freeze_receipt)
    return manifest


def load_provenance_registry(
    path: str | Path, *, protocol: ConfirmedControlProtocol | None = None
) -> tuple[dict[str, Any], tuple[ConfirmedReplayControl, ...]]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != PROVENANCE_REGISTRY_FORMAT:
        raise ValueError("unsupported T12.3d confirmed-control registry")
    selected = protocol or ConfirmedControlProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("T12.3d registry protocol mismatch")
    controls = tuple(
        ConfirmedReplayControl.from_dict(dict(item))
        for item in payload.get("controls", ())
    )
    if len(controls) != selected.expected_unique_controls:
        raise ValueError("T12.3d registry control count mismatch")
    if len({item.route_checksum for item in controls}) != len(controls):
        raise ValueError("T12.3d registry contains duplicate routes")
    if len({item.control_checksum for item in controls}) != len(controls):
        raise ValueError("T12.3d registry contains duplicate control checksums")
    return payload, controls


def load_provenance_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != PROVENANCE_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.3d manifest")
    protocol = ConfirmedControlProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.3d protocol checksum mismatch")
    registry_meta = dict(manifest["control_registry"])
    registry_path = _resolve_bound(str(registry_meta["path"]), root=repo_root)
    if not registry_path.is_file() or _file_sha256(registry_path) != registry_meta["sha256"]:
        raise ValueError("T12.3d control registry checksum mismatch")
    registry, _ = load_provenance_registry(registry_path, protocol=protocol)
    if registry["registry_checksum"] != registry_meta["registry_checksum"]:
        raise ValueError("T12.3d control registry signature mismatch")
    for parent_key in ("parent", "control_source_t12_3a"):
        for artifact_key in ("manifest", "receipt"):
            meta = dict(manifest[parent_key][artifact_key])
            candidate = _resolve_bound(str(meta["path"]), root=repo_root)
            if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
                raise ValueError(
                    f"T12.3d {parent_key} {artifact_key} checksum mismatch"
                )
    source_registry = dict(manifest["control_source_t12_3a"]["registry"])
    source_registry_path = _resolve_bound(
        str(source_registry["path"]), root=repo_root
    )
    if not source_registry_path.is_file() or _file_sha256(
        source_registry_path
    ) != source_registry["sha256"]:
        raise ValueError("T12.3d source witness registry checksum mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.3d code checksum mismatch: {relative}")
    return manifest


def provenance_phase_receipt(
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
            "format_version": PROVENANCE_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_3c_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "source_t12_3a_receipt_checksum": manifest["control_source_t12_3a"][
                "receipt"
            ]["receipt_checksum"],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_provenance_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != PROVENANCE_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.3d receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.3d receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.3d receipt belongs to another protocol")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        artifact = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not artifact.is_file() or _file_sha256(artifact) != meta.get("sha256"):
            raise ValueError(f"T12.3d receipt artifact checksum mismatch: {name}")
        if name == "paired_evaluation":
            evaluation = _read_json(artifact)
            for condition in evaluation.get("conditions", ()):
                for arm_name, arm in dict(condition.get("arms", {})).items():
                    for artifact_name in ("archive", "excursions"):
                        nested = dict(arm[artifact_name])
                        nested_path = _resolve_bound(str(nested["path"]), root=repo_root)
                        if not nested_path.is_file() or _file_sha256(nested_path) != nested["sha256"]:
                            raise ValueError(
                                "T12.3d paired artifact checksum mismatch: "
                                f"{arm_name}:{artifact_name}"
                            )
    return receipt


__all__ = [
    "ConfirmedControlProtocol",
    "ConfirmedReplayControl",
    "freeze_provenance_experiment",
    "load_provenance_manifest",
    "load_provenance_receipt",
    "load_provenance_registry",
    "provenance_phase_receipt",
]
