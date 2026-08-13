"""Frozen T12.4a.4d protocol for target-local option re-grounding search."""

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
from .lineage_shield_protocol import (
    load_lineage_shield_manifest,
    load_lineage_shield_receipt,
    load_lineage_shield_registry,
)
from .option_contract_protocol import (
    load_option_contract_manifest,
    load_option_contract_receipt,
)
from .shield_model import ProgressProtectedTerminalShield
from .witness_reconfirmation_protocol import load_reconfirmation_registry

TARGET_REGROUNDING_PROTOCOL_FORMAT = "sage-t12.4a.4d-target-regrounding-protocol-v1"
TARGET_REGROUNDING_MANIFEST_FORMAT = "sage-t12.4a.4d-target-regrounding-manifest-v1"
TARGET_REGROUNDING_RECEIPT_FORMAT = "sage-t12.4a.4d-target-regrounding-receipt-v1"

TARGET_REGROUNDING_CODE_PATHS = (
    "theory/sage_t/causal/target_regrounding_protocol.py",
    "theory/sage_t/causal/target_regrounding_experiment.py",
    "theory/sage_t/causal/target_regrounding_cli.py",
    "theory/sage_t/causal/option_contracts.py",
    "theory/sage_t/causal/option_applicability_experiment.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/lineage_archive.py",
    "theory/sage_t/causal/lineage_shield_protocol.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/contracts.py",
    "theory/sage_t/causal/shield_model.py",
    "theory/sage_t/causal/witness_protocol.py",
    "theory/sage_t/causal/witness_reconfirmation_protocol.py",
    "theory/sage_t/contracts.py",
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
class TargetRegroundingProtocol:
    format_version: str = TARGET_REGROUNDING_PROTOCOL_FORMAT
    source_lineages: tuple[int, ...] = (8_701, 8_705)
    search_seeds: tuple[int, ...] = (9_101, 9_102, 9_103)
    search_arms: tuple[str, ...] = (
        "local_archive_control",
        "contract_regrounded",
    )
    blocked_option_shadow_control: str = "contracted_option_blocked"
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    sdk_calls_per_search_arm: int = 2_048
    maximum_total_sdk_calls: int = 26_000
    maximum_excursions_per_arm: int = 64
    maximum_cells: int = 10_000
    confirmation_repetitions_per_lineage: int = 2
    minimum_confirmation_exact_rate: float = 1.0
    maximum_witness_suffix_actions: int = 64
    maximum_terminal_failure_rate: float = 0.10
    minimum_contract_block_margin: float = 0.80
    minimum_progress_edges: int = 1
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    persist_raw_frames: bool = False
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in (
            "source_lineages",
            "search_seeds",
            "burst_schedule",
        ):
            object.__setattr__(
                self,
                name,
                tuple(int(value) for value in getattr(self, name)),
            )
        object.__setattr__(
            self,
            "search_arms",
            tuple(str(value) for value in self.search_arms),
        )
        expected = {
            "format_version": TARGET_REGROUNDING_PROTOCOL_FORMAT,
            "source_lineages": (8_701, 8_705),
            "search_seeds": (9_101, 9_102, 9_103),
            "search_arms": ("local_archive_control", "contract_regrounded"),
            "blocked_option_shadow_control": "contracted_option_blocked",
            "burst_schedule": (4, 8, 16),
            "sdk_calls_per_search_arm": 2_048,
            "maximum_total_sdk_calls": 26_000,
            "maximum_excursions_per_arm": 64,
            "maximum_cells": 10_000,
            "confirmation_repetitions_per_lineage": 2,
            "minimum_confirmation_exact_rate": 1.0,
            "maximum_witness_suffix_actions": 64,
            "maximum_terminal_failure_rate": 0.10,
            "minimum_contract_block_margin": 0.80,
            "minimum_progress_edges": 1,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.4a.4d preregistered value changed: {name}")
        if len(set(self.search_seeds)) != len(self.search_seeds):
            raise ValueError("T12.4a.4d search seeds must be distinct")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _verified_artifact(
    meta: Mapping[str, Any],
    *,
    root: Path,
    label: str,
) -> tuple[Path, dict[str, str]]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.4a.4d {label} checksum mismatch")
    return path, {"path": _bound_path(path, root=root), "sha256": expected}


def freeze_target_regrounding(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    witness_registry_path: str | Path,
    shield_manifest_path: str | Path,
    shield_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: TargetRegroundingProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or TargetRegroundingProtocol()
    parent_manifest = load_option_contract_manifest(
        parent_manifest_path,
        root=repo_root,
    )
    parent_receipt = load_option_contract_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
    )
    if not (
        parent_receipt.get("phase") == "option_contract"
        and parent_receipt.get("status")
        == "PASS_T12_4A_4C_OPTION_CONTRACT_GATE"
        and parent_manifest.get("stage") == "source_train"
    ):
        raise ValueError("T12.4a.4d requires the passed T12.4a.4c source-train gate")
    if not parent_manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.4d parent did not authorize scientific execution")
    if SAGE11_SPLITS.split_for(str(parent_manifest["game_id"])) != "source_train":
        raise ValueError("T12.4a.4d is restricted to source_train")

    witness_path = _resolve_bound(str(witness_registry_path), root=repo_root)
    _, witnesses = load_reconfirmation_registry(witness_path)
    if tuple(sorted(item.source_seed for item in witnesses)) != selected.source_lineages:
        raise ValueError("T12.4a.4d witness lineages differ from the protocol")
    if len({item.target_exact_hash for item in witnesses}) != 1:
        raise ValueError("T12.4a.4d requires one common exact level-1 anchor")

    shield_manifest = load_lineage_shield_manifest(
        shield_manifest_path,
        root=repo_root,
    )
    shield_receipt = load_lineage_shield_receipt(
        shield_receipt_path,
        manifest=shield_manifest,
        root=repo_root,
    )
    if not (
        shield_receipt.get("passed") is True
        and shield_receipt.get("phase") == "lineage_shield"
        and shield_receipt.get("status") == "PASS_T12_3E_LINEAGE_SHIELD_GATE"
    ):
        raise ValueError("T12.4a.4d requires the passed T12.3e shield gate")
    shield_registry_path = _resolve_bound(
        str(shield_manifest["source_registry"]["path"]), root=repo_root
    )
    shield_registry = load_lineage_shield_registry(shield_registry_path)
    shield_meta = dict(shield_registry["terminal_shield"])
    shield_path = _resolve_bound(str(shield_meta["path"]), root=repo_root)
    if not shield_path.is_file() or _file_sha256(shield_path) != shield_meta["sha256"]:
        raise ValueError("T12.4a.4d validated terminal-shield checksum mismatch")
    shield_payload = _read_json(shield_path)
    ProgressProtectedTerminalShield.from_dict(shield_payload)

    registry_path, registry_meta = _verified_artifact(
        parent_receipt["artifacts"]["contracted_option_registry"],
        root=repo_root,
        label="contracted option registry",
    )
    programs_path, programs_meta = _verified_artifact(
        parent_receipt["artifacts"]["contracted_option_programs"],
        root=repo_root,
        label="contracted programs",
    )
    posterior_path, posterior_meta = _verified_artifact(
        parent_receipt["artifacts"]["contracted_posterior"],
        root=repo_root,
        label="contracted posterior",
    )
    del registry_path, programs_path, posterior_path

    missing = [
        path for path in TARGET_REGROUNDING_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.4a.4d code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": TARGET_REGROUNDING_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_4D_TARGET_REGROUNDING",
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
            },
        },
        "safety_parent": {
            "manifest": {
                "path": _bound_path(shield_manifest_path, root=repo_root),
                "sha256": _file_sha256(shield_manifest_path),
                "manifest_checksum": shield_manifest["manifest_checksum"],
            },
            "receipt": {
                "path": _bound_path(shield_receipt_path, root=repo_root),
                "sha256": _file_sha256(shield_receipt_path),
                "receipt_checksum": shield_receipt["receipt_checksum"],
                "status": shield_receipt["status"],
            },
        },
        "inputs": {
            "contracted_option_registry": registry_meta,
            "contracted_option_programs": programs_meta,
            "contracted_posterior": posterior_meta,
            "terminal_shield": {
                "path": _bound_path(shield_path, root=repo_root),
                "sha256": _file_sha256(shield_path),
            },
            "terminal_shield_registry": {
                "path": _bound_path(shield_registry_path, root=repo_root),
                "sha256": _file_sha256(shield_registry_path),
            },
            "witness_registry": {
                "path": _bound_path(witness_path, root=repo_root),
                "sha256": _file_sha256(witness_path),
            },
            "entry_exact_hash": witnesses[0].target_exact_hash,
            "entry_level": witnesses[0].target_level,
            "route_lengths": [len(item.steps) for item in witnesses],
        },
        "paired_design": {
            "same_action_catalog": True,
            "same_burst_schedule": True,
            "same_lineage_schedule": True,
            "same_sdk_budget": True,
            "same_terminal_shield": True,
            "treatment_difference": (
                "contract guard mismatch plus target-local role grounding reranking"
            ),
            "primary_claim": "confirmed new level-1-to-level-2 progress witness",
            "secondary_claim": "contract reranking improves discovery efficiency",
            "witness_selection_rule": (
                "minimum progress sdk calls, then shortest suffix, then "
                "search seed, lineage seed and arm name"
            ),
            "negative_result_policy": (
                "a miss or failed integrity gate stops option extraction; "
                "no threshold retuning or holdout opening"
            ),
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in TARGET_REGROUNDING_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "neural_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "option_control_authorized": False,
            "target_regrounding_experiment_authorized": authorized,
            "t12_4a_4e_option_extraction_freeze_authorized": False,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "maximum_total_sdk_calls": selected.maximum_total_sdk_calls,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = target_regrounding_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status=(
            "PASS_T12_4A_4D_FREEZE" if authorized else "DIRTY_SMOKE_ONLY"
        ),
        metrics={
            "maximum_total_sdk_calls": selected.maximum_total_sdk_calls,
            "search_arms": list(selected.search_arms),
            "search_seeds": list(selected.search_seeds),
            "source_lineages": list(selected.source_lineages),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_target_regrounding_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != TARGET_REGROUNDING_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a.4d manifest")
    protocol = TargetRegroundingProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a.4d protocol checksum mismatch")
    for meta in manifest["parent"].values():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.4a.4d parent checksum mismatch")
    for meta in manifest["safety_parent"].values():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.4a.4d safety-parent checksum mismatch")
    for name in (
        "contracted_option_registry",
        "contracted_option_programs",
        "contracted_posterior",
        "terminal_shield",
        "terminal_shield_registry",
        "witness_registry",
    ):
        meta = dict(manifest["inputs"][name])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.4d input checksum mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a.4d code checksum mismatch: {relative}")
    return manifest


def target_regrounding_receipt(
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
            "format_version": TARGET_REGROUNDING_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_4c_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_target_regrounding_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != TARGET_REGROUNDING_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a.4d receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.4a.4d receipt belongs to another manifest")
    for name, meta in receipt.get("artifacts", {}).items():
        if isinstance(meta, list):
            values = meta
        elif isinstance(meta, Mapping) and "path" in meta:
            values = [meta]
        else:
            continue
        for item in values:
            candidate = _resolve_bound(str(item["path"]), root=repo_root)
            if not candidate.is_file() or _file_sha256(candidate) != item["sha256"]:
                raise ValueError(f"T12.4a.4d receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "TargetRegroundingProtocol",
    "freeze_target_regrounding",
    "load_target_regrounding_manifest",
    "load_target_regrounding_receipt",
    "target_regrounding_receipt",
]
