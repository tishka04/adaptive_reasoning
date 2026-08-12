"""Frozen T12.3e protocol for the lineage-corrected terminal-shield retest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

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
    _parent_failure_is_replay_only,
    load_lineage_manifest,
)
from .provenance_protocol import (
    load_provenance_manifest,
    load_provenance_receipt,
)
from .shield_model import ProgressProtectedTerminalShield
from .shield_protocol import (
    load_shield_manifest,
    load_shield_receipt,
    load_shield_registry,
)
from .witness_protocol import (
    load_witness_manifest,
    load_witness_receipt,
    load_witness_registry,
)

LINEAGE_SHIELD_PROTOCOL_FORMAT = "sage-t12.3e-lineage-shield-protocol-v1"
LINEAGE_SHIELD_REGISTRY_FORMAT = "sage-t12.3e-lineage-shield-inputs-v1"
LINEAGE_SHIELD_MANIFEST_FORMAT = "sage-t12.3e-lineage-shield-manifest-v1"
LINEAGE_SHIELD_RECEIPT_FORMAT = "sage-t12.3e-lineage-shield-receipt-v1"

LINEAGE_SHIELD_CODE_PATHS = (
    "theory/sage_t/causal/lineage_shield_protocol.py",
    "theory/sage_t/causal/lineage_shield_experiment.py",
    "theory/sage_t/causal/lineage_shield_experiment_cli.py",
    "theory/sage_t/causal/lineage_archive.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/burst_experiment.py",
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/shield_experiment.py",
    "theory/sage_t/causal/shield_model.py",
    "theory/sage_t/causal/shield_protocol.py",
    "theory/sage_t/causal/terminal_shield.py",
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
class LineageShieldProtocol:
    format_version: str = LINEAGE_SHIELD_PROTOCOL_FORMAT
    expected_terminal_candidates: int = 12
    expected_protected_action_pairs: int = 99
    expected_witnesses: int = 2
    witness_repetitions: int = 3
    evaluation_seeds: tuple[int, ...] = (7701, 7702, 7703)
    evaluation_arms: tuple[str, ...] = (
        "lineage_control",
        "lineage_terminal_shield",
    )
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    sdk_calls_per_evaluation_arm: int = 4_096
    maximum_total_sdk_calls: int = 30_000
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_cells: int = 50_000
    minimum_evaluation_replay_exact_rate: float = 0.95
    maximum_terminal_rate_ratio: float = 0.90
    minimum_coverage_ratio: float = 0.80
    maximum_terminal_regression_seeds: int = 1
    maximum_progress_regression_seeds: int = 0
    minimum_vetoes: int = 1
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
        if self.format_version != LINEAGE_SHIELD_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.3e lineage-shield protocol")
        fixed = {
            "expected_terminal_candidates": (
                self.expected_terminal_candidates,
                12,
            ),
            "expected_protected_action_pairs": (
                self.expected_protected_action_pairs,
                99,
            ),
            "expected_witnesses": (self.expected_witnesses, 2),
            "witness_repetitions": (self.witness_repetitions, 3),
            "sdk_calls_per_evaluation_arm": (
                self.sdk_calls_per_evaluation_arm,
                4_096,
            ),
            "maximum_total_sdk_calls": (self.maximum_total_sdk_calls, 30_000),
            "maximum_artifact_bytes_per_run": (
                self.maximum_artifact_bytes_per_run,
                3 * 1024 * 1024 * 1024,
            ),
            "maximum_cells": (self.maximum_cells, 50_000),
            "minimum_evaluation_replay_exact_rate": (
                self.minimum_evaluation_replay_exact_rate,
                0.95,
            ),
            "maximum_terminal_rate_ratio": (
                self.maximum_terminal_rate_ratio,
                0.90,
            ),
            "minimum_coverage_ratio": (self.minimum_coverage_ratio, 0.80),
            "maximum_terminal_regression_seeds": (
                self.maximum_terminal_regression_seeds,
                1,
            ),
            "maximum_progress_regression_seeds": (
                self.maximum_progress_regression_seeds,
                0,
            ),
            "minimum_vetoes": (self.minimum_vetoes, 1),
            "minimum_rebases_avoided": (self.minimum_rebases_avoided, 1),
        }
        for name, (observed, expected) in fixed.items():
            if observed != expected:
                raise ValueError(f"T12.3e preregistered value changed: {name}")
        if self.evaluation_seeds != (7701, 7702, 7703):
            raise ValueError("T12.3e evaluation seeds are frozen")
        if self.evaluation_arms != (
            "lineage_control",
            "lineage_terminal_shield",
        ):
            raise ValueError("T12.3e evaluation arms are frozen")
        if self.burst_schedule != (4, 8, 16):
            raise ValueError("T12.3e burst schedule is frozen")
        witness_upper_bound = (
            self.expected_witnesses
            * self.witness_repetitions
            * (1 + 128)
        )
        evaluation_upper_bound = (
            len(self.evaluation_seeds)
            * len(self.evaluation_arms)
            * self.sdk_calls_per_evaluation_arm
        )
        if witness_upper_bound + evaluation_upper_bound > self.maximum_total_sdk_calls:
            raise ValueError("T12.3e planned calls exceed the global SDK budget")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _source_shield_evidence(
    *,
    shield_manifest: Mapping[str, Any],
    shield_receipt: Mapping[str, Any],
    root: Path,
    protocol: LineageShieldProtocol,
) -> dict[str, Any]:
    if not _parent_failure_is_replay_only(shield_manifest, shield_receipt):
        raise ValueError("T12.3e source T12.3b did not fail only on replay")
    registry_path = _resolve_bound(
        str(shield_manifest["terminal_registry"]["path"]), root=root
    )
    _, candidates, protected = load_shield_registry(registry_path)
    if len(candidates) != protocol.expected_terminal_candidates:
        raise ValueError("T12.3e source terminal candidate count mismatch")
    if len(protected) != protocol.expected_protected_action_pairs:
        raise ValueError("T12.3e source protected action count mismatch")

    confirmation_meta = dict(
        shield_receipt["artifacts"]["terminal_confirmations"]
    )
    confirmation_path = _resolve_bound(str(confirmation_meta["path"]), root=root)
    confirmation_payload = _read_json(confirmation_path)
    confirmed_ids = {
        str(value) for value in confirmation_payload.get("confirmed_candidate_ids", ())
    }
    candidate_ids = {item.candidate_id for item in candidates}
    if confirmed_ids != candidate_ids:
        raise ValueError("T12.3e source terminal confirmations are incomplete")

    shield_meta = dict(shield_receipt["artifacts"]["terminal_shield"])
    shield_path = _resolve_bound(str(shield_meta["path"]), root=root)
    shield_payload = _read_json(shield_path)
    shield = ProgressProtectedTerminalShield.from_dict(shield_payload)
    shield_metrics = shield.metrics()
    if int(shield_metrics["confirmed_terminal_traces"]) != len(candidates):
        raise ValueError("T12.3e source shield trace count mismatch")
    if not bool(shield_metrics["multi_step_hazard_observed"]):
        raise ValueError("T12.3e source shield lacks multi-step terminal evidence")
    if int(shield_metrics["protected_action_pairs"]) != len(protected):
        raise ValueError("T12.3e source shield protected pair count mismatch")
    expected_pairs = {(item.cell_id, item.action_key) for item in protected}
    if set(shield.protected_pairs) != expected_pairs:
        raise ValueError("T12.3e source shield/protected registry mismatch")
    if len(shield.witness_ids) != protocol.expected_witnesses:
        raise ValueError("T12.3e source shield witness count mismatch")
    return {
        "terminal_candidate_ids": sorted(candidate_ids),
        "protected_pair_checksum": _checksum(sorted(expected_pairs)),
        "protected_action_pairs": len(expected_pairs),
        "witness_ids": list(shield.witness_ids),
        "shield_metrics": shield_metrics,
        "terminal_registry": dict(shield_manifest["terminal_registry"]),
        "terminal_confirmations": confirmation_meta,
        "terminal_shield": shield_meta,
    }


def freeze_lineage_shield_experiment(
    *,
    output_path: str | Path,
    source_registry_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: LineageShieldProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or LineageShieldProtocol()
    parent_manifest = load_provenance_manifest(
        parent_manifest_path, root=repo_root, verify_code=False
    )
    parent_receipt = load_provenance_receipt(
        parent_receipt_path, manifest=parent_manifest, root=repo_root
    )
    if parent_receipt.get("passed") is not True or parent_receipt.get(
        "status"
    ) != "PASS_T12_3D_CONFIRMED_CONTROL_GATE":
        raise ValueError("T12.3e requires a passed T12.3d parent")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.3e is restricted to source_train")

    lineage_manifest_path = _resolve_bound(
        str(parent_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    lineage_manifest = load_lineage_manifest(
        lineage_manifest_path, root=repo_root, verify_code=False
    )
    shield_manifest_path = _resolve_bound(
        str(lineage_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    shield_receipt_path = _resolve_bound(
        str(lineage_manifest["parent"]["receipt"]["path"]), root=repo_root
    )
    shield_manifest = load_shield_manifest(
        shield_manifest_path, root=repo_root, verify_code=False
    )
    shield_receipt = load_shield_receipt(
        shield_receipt_path, manifest=shield_manifest, root=repo_root
    )
    source = _source_shield_evidence(
        shield_manifest=shield_manifest,
        shield_receipt=shield_receipt,
        root=repo_root,
        protocol=selected,
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
    if witness_receipt.get("passed") is not True:
        raise ValueError("T12.3e witness source is not passed")
    witness_registry_path = _resolve_bound(
        str(witness_manifest["witness_registry"]["path"]), root=repo_root
    )
    _, witnesses = load_witness_registry(witness_registry_path)
    if len(witnesses) != selected.expected_witnesses:
        raise ValueError("T12.3e witness registry count mismatch")
    if {item.witness_id for item in witnesses} != set(source["witness_ids"]):
        raise ValueError("T12.3e shield/witness registry mismatch")

    missing = [
        path for path in LINEAGE_SHIELD_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.3e code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(
        not git["dirty"]
        and parent_manifest.get("scientific_claims_authorized", False)
        and parent_receipt.get("passed") is True
    )
    registry = _signed(
        {
            "format_version": LINEAGE_SHIELD_REGISTRY_FORMAT,
            "protocol_checksum": selected.checksum,
            "parent_t12_3d_receipt_checksum": parent_receipt["receipt_checksum"],
            "source_t12_3b_receipt_checksum": shield_receipt["receipt_checksum"],
            **source,
        },
        "registry_checksum",
    )
    _write_json_once(source_registry_path, registry)
    payload = {
        "format_version": LINEAGE_SHIELD_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_3E_LINEAGE_SHIELD",
        "stage": "source_train",
        "game_id": parent_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "source_registry": {
            "path": _bound_path(source_registry_path, root=repo_root),
            "sha256": _file_sha256(source_registry_path),
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
                "passed": True,
                "status": "PASS_T12_3D_CONFIRMED_CONTROL_GATE",
            },
        },
        "source_t12_3b": {
            "manifest": {
                "path": _bound_path(shield_manifest_path, root=repo_root),
                "sha256": _file_sha256(shield_manifest_path),
                "manifest_checksum": shield_manifest["manifest_checksum"],
            },
            "receipt": {
                "path": _bound_path(shield_receipt_path, root=repo_root),
                "sha256": _file_sha256(shield_receipt_path),
                "receipt_checksum": shield_receipt["receipt_checksum"],
                "passed": False,
                "status": "FAIL_T12_3B_TERMINAL_SHIELD_GATE",
                "failure_class": "REPLAY_EXACT_ONLY",
            },
        },
        "witness_source_t12_3a": {
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
            path: _file_sha256(repo_root / path) for path in LINEAGE_SHIELD_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "lineage_shield_experiment_authorized": authorized,
            "terminal_shield_production_authority": False,
            "t12_4_freeze_authorized": False,
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
    freeze_receipt = lineage_shield_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_3E_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "terminal_candidates": len(source["terminal_candidate_ids"]),
            "protected_action_pairs": source["protected_action_pairs"],
            "witnesses": len(source["witness_ids"]),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), freeze_receipt)
    return manifest


def load_lineage_shield_registry(
    path: str | Path, *, protocol: LineageShieldProtocol | None = None
) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != LINEAGE_SHIELD_REGISTRY_FORMAT:
        raise ValueError("unsupported T12.3e source registry")
    selected = protocol or LineageShieldProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("T12.3e source registry protocol mismatch")
    if len(payload.get("terminal_candidate_ids", ())) != selected.expected_terminal_candidates:
        raise ValueError("T12.3e source registry candidate count mismatch")
    if int(payload.get("protected_action_pairs", 0)) != selected.expected_protected_action_pairs:
        raise ValueError("T12.3e source registry protected pair count mismatch")
    if len(payload.get("witness_ids", ())) != selected.expected_witnesses:
        raise ValueError("T12.3e source registry witness count mismatch")
    return payload


def load_lineage_shield_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != LINEAGE_SHIELD_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.3e manifest")
    protocol = LineageShieldProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.3e protocol checksum mismatch")
    registry_meta = dict(manifest["source_registry"])
    registry_path = _resolve_bound(str(registry_meta["path"]), root=repo_root)
    if not registry_path.is_file() or _file_sha256(registry_path) != registry_meta["sha256"]:
        raise ValueError("T12.3e source registry checksum mismatch")
    registry = load_lineage_shield_registry(registry_path, protocol=protocol)
    if registry["registry_checksum"] != registry_meta["registry_checksum"]:
        raise ValueError("T12.3e source registry signature mismatch")
    for parent_key in ("parent", "source_t12_3b", "witness_source_t12_3a"):
        for artifact_key in ("manifest", "receipt"):
            meta = dict(manifest[parent_key][artifact_key])
            candidate = _resolve_bound(str(meta["path"]), root=repo_root)
            if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
                raise ValueError(
                    f"T12.3e {parent_key} {artifact_key} checksum mismatch"
                )
    witness_registry = dict(manifest["witness_source_t12_3a"]["registry"])
    witness_registry_path = _resolve_bound(
        str(witness_registry["path"]), root=repo_root
    )
    if not witness_registry_path.is_file() or _file_sha256(
        witness_registry_path
    ) != witness_registry["sha256"]:
        raise ValueError("T12.3e witness registry checksum mismatch")
    for name in ("terminal_registry", "terminal_confirmations", "terminal_shield"):
        meta = dict(registry[name])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.3e source artifact checksum mismatch: {name}")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.3e code checksum mismatch: {relative}")
    return manifest


def lineage_shield_phase_receipt(
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
            "format_version": LINEAGE_SHIELD_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_3d_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "source_t12_3b_receipt_checksum": manifest["source_t12_3b"][
                "receipt"
            ]["receipt_checksum"],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_lineage_shield_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != LINEAGE_SHIELD_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.3e receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.3e receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.3e receipt belongs to another protocol")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        artifact = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not artifact.is_file() or _file_sha256(artifact) != meta.get("sha256"):
            raise ValueError(f"T12.3e receipt artifact checksum mismatch: {name}")
        if name == "paired_evaluation":
            evaluation = _read_json(artifact)
            for condition in evaluation.get("conditions", ()):
                for arm_name, arm in dict(condition.get("arms", {})).items():
                    for artifact_name in ("archive", "excursions"):
                        nested = dict(arm[artifact_name])
                        nested_path = _resolve_bound(str(nested["path"]), root=repo_root)
                        if not nested_path.is_file() or _file_sha256(nested_path) != nested["sha256"]:
                            raise ValueError(
                                "T12.3e paired artifact checksum mismatch: "
                                f"{arm_name}:{artifact_name}"
                            )
    return receipt


__all__ = [
    "LineageShieldProtocol",
    "freeze_lineage_shield_experiment",
    "lineage_shield_phase_receipt",
    "load_lineage_shield_manifest",
    "load_lineage_shield_receipt",
    "load_lineage_shield_registry",
]
