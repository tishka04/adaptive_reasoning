"""Frozen T12.4a.4c protocol for guarded causal-option compilation."""

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
from .option_applicability_protocol import (
    load_option_applicability_manifest,
    load_option_applicability_receipt,
)
from .option_transfer_protocol import load_option_transfer_manifest

OPTION_CONTRACT_PROTOCOL_FORMAT = "sage-t12.4a.4c-option-contract-protocol-v1"
OPTION_CONTRACT_MANIFEST_FORMAT = "sage-t12.4a.4c-option-contract-manifest-v1"
OPTION_CONTRACT_RECEIPT_FORMAT = "sage-t12.4a.4c-option-contract-receipt-v1"

OPTION_CONTRACT_CODE_PATHS = (
    "theory/sage_t/causal/option_contracts.py",
    "theory/sage_t/causal/option_contract_protocol.py",
    "theory/sage_t/causal/option_contract_experiment.py",
    "theory/sage_t/causal/option_contract_cli.py",
    "theory/sage_t/causal/option_applicability_protocol.py",
    "theory/sage_t/causal/options.py",
    "theory/sage_t/causal/posterior.py",
    "theory/sage_t/causal/executor.py",
    "theory/sage_t/causal/contracts.py",
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
class OptionContractProtocol:
    format_version: str = OPTION_CONTRACT_PROTOCOL_FORMAT
    source_seeds: tuple[int, ...] = (8_701, 8_705)
    success_context: str = "successful_level0"
    failure_context: str = "failed_level1"
    expected_classification: str = "INITIATION_AND_DYNAMICS_SHIFT"
    expected_option_actions: tuple[str, ...] = (
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    )
    allowed_initiation_features: tuple[str, ...] = (
        "predicate_counts.adjacent",
        "predicate_counts.aligned",
        "predicate_counts.contact",
        "predicate_counts.near",
        "role_counts.clickable",
        "role_counts.movable",
    )
    allowed_effect_features: tuple[str, ...] = (
        "predicate_counts.adjacent",
        "predicate_counts.aligned",
        "predicate_counts.contact",
        "predicate_counts.near",
        "role_counts.clickable",
        "role_counts.movable",
    )
    minimum_initiation_particles: int = 4
    maximum_initiation_particles: int = 6
    atoms_per_initiation_particle: int = 1
    minimum_effect_atoms_per_step: int = 1
    maximum_effect_atoms_per_step: int = 3
    maximum_parent_particles: int = 4
    maximum_child_particles: int = 24
    minimum_applicable_posterior_mass: float = 0.80
    maximum_sdk_calls: int = 0
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    persist_raw_frames: bool = False
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name, caster in (
            ("source_seeds", int),
            ("expected_option_actions", str),
            ("allowed_initiation_features", str),
            ("allowed_effect_features", str),
        ):
            object.__setattr__(
                self,
                name,
                tuple(caster(value) for value in getattr(self, name)),
            )
        expected = {
            "format_version": OPTION_CONTRACT_PROTOCOL_FORMAT,
            "source_seeds": (8_701, 8_705),
            "success_context": "successful_level0",
            "failure_context": "failed_level1",
            "expected_classification": "INITIATION_AND_DYNAMICS_SHIFT",
            "expected_option_actions": (
                "ACTION4",
                "ACTION4",
                "ACTION4",
                "ACTION3",
                "ACTION3",
            ),
            "allowed_initiation_features": (
                "predicate_counts.adjacent",
                "predicate_counts.aligned",
                "predicate_counts.contact",
                "predicate_counts.near",
                "role_counts.clickable",
                "role_counts.movable",
            ),
            "allowed_effect_features": (
                "predicate_counts.adjacent",
                "predicate_counts.aligned",
                "predicate_counts.contact",
                "predicate_counts.near",
                "role_counts.clickable",
                "role_counts.movable",
            ),
            "minimum_initiation_particles": 4,
            "maximum_initiation_particles": 6,
            "atoms_per_initiation_particle": 1,
            "minimum_effect_atoms_per_step": 1,
            "maximum_effect_atoms_per_step": 3,
            "maximum_parent_particles": 4,
            "maximum_child_particles": 24,
            "minimum_applicable_posterior_mass": 0.80,
            "maximum_sdk_calls": 0,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.4a.4c preregistered value changed: {name}")
        if self.maximum_parent_particles * self.maximum_initiation_particles != (
            self.maximum_child_particles
        ):
            raise ValueError("T12.4a.4c child-particle bound is inconsistent")

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
) -> tuple[Path, dict[str, str]]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.4a.4c artifact mismatch: {path.name}")
    return path, {"path": _bound_path(path, root=root), "sha256": expected}


def freeze_option_contract(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: OptionContractProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or OptionContractProtocol()
    parent_manifest = load_option_applicability_manifest(
        parent_manifest_path,
        root=repo_root,
    )
    parent_receipt = load_option_applicability_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
    )
    if not (
        parent_receipt.get("phase") == "option_applicability_audit"
        and parent_receipt.get("status")
        == "PASS_T12_4A_4B_APPLICABILITY_AUDIT_GATE"
        and parent_receipt.get("metrics", {}).get("classification")
        == selected.expected_classification
    ):
        raise ValueError("T12.4a.4c requires the passed initiation+dynamics audit")
    checks = dict(parent_receipt.get("metrics", {}).get("checks", {}))
    if not checks or not all(checks.values()):
        raise ValueError("T12.4a.4c parent audit has an incomplete integrity gate")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.4a.4c is restricted to source_train")
    if not parent_manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.4c parent did not authorize scientific execution")

    trials_path, trials_meta = _verified_artifact(
        parent_receipt["artifacts"]["trials"], root=repo_root
    )
    diagnosis_path, diagnosis_meta = _verified_artifact(
        parent_receipt["artifacts"]["diagnosis"], root=repo_root
    )
    diagnosis = _read_json(diagnosis_path)
    if diagnosis.get("classification") != selected.expected_classification:
        raise ValueError("T12.4a.4c diagnosis artifact disagrees with receipt")
    trials = _read_json(trials_path)
    if len(trials.get("trials", ())) != 16:
        raise ValueError("T12.4a.4c requires all 16 sealed applicability trials")

    option_path, option_meta = _verified_artifact(
        parent_manifest["inputs"]["minimal_option"], root=repo_root
    )
    transfer_manifest_path = _resolve_bound(
        str(parent_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    transfer_manifest = load_option_transfer_manifest(
        transfer_manifest_path,
        root=repo_root,
    )
    compiled_path, compiled_meta = _verified_artifact(
        transfer_manifest["inputs"]["compiled_option_registry"], root=repo_root
    )
    programs_path, programs_meta = _verified_artifact(
        transfer_manifest["inputs"]["option_programs"], root=repo_root
    )
    posterior_path, posterior_meta = _verified_artifact(
        transfer_manifest["inputs"]["posterior_snapshot"], root=repo_root
    )
    del option_path, compiled_path, programs_path, posterior_path

    missing = [
        path for path in OPTION_CONTRACT_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.4a.4c code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": OPTION_CONTRACT_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_4C_OPTION_CONTRACT",
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
        "inputs": {
            "applicability_trials": trials_meta,
            "applicability_diagnosis": diagnosis_meta,
            "minimal_option": option_meta,
            "compiled_option_registry": compiled_meta,
            "option_programs": programs_meta,
            "posterior_snapshot": posterior_meta,
            "option_checksum": parent_manifest["inputs"]["option_checksum"],
        },
        "induction_rule": {
            "guards_are_rival_particles": True,
            "guards_require_cross_lineage_exact_stability": True,
            "guards_must_reject_every_failed_anchor": True,
            "effects_require_cross_lineage_exact_stability": True,
            "effect_vocabulary_is_action_relevant_and_typed": True,
            "forbidden_inputs": [
                "absolute_coordinates",
                "entity_ids",
                "exact_hashes",
                "game_id",
                "level_index",
                "pixels",
                "raw_fact_tokens",
            ],
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in OPTION_CONTRACT_CODE_PATHS
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
            "option_contract_compile_authorized": authorized,
            "t12_4a_4d_target_regrounding_freeze_authorized": False,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "maximum_sdk_calls": 0,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    freeze_receipt = option_contract_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_4A_4C_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "maximum_child_particles": selected.maximum_child_particles,
            "maximum_sdk_calls": 0,
            "source_lineages": list(selected.source_seeds),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), freeze_receipt)
    return manifest


def load_option_contract_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != OPTION_CONTRACT_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a.4c manifest")
    protocol = OptionContractProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a.4c protocol checksum mismatch")
    for meta in manifest["parent"].values():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.4a.4c parent artifact mismatch")
    for name in (
        "applicability_trials",
        "applicability_diagnosis",
        "minimal_option",
        "compiled_option_registry",
        "option_programs",
        "posterior_snapshot",
    ):
        meta = dict(manifest["inputs"][name])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.4c input artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a.4c code checksum mismatch: {relative}")
    return manifest


def option_contract_receipt(
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
            "format_version": OPTION_CONTRACT_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_4b_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_option_contract_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != OPTION_CONTRACT_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a.4c receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.4a.4c receipt belongs to another manifest")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.4a.4c gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.4c receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "OptionContractProtocol",
    "freeze_option_contract",
    "load_option_contract_manifest",
    "load_option_contract_receipt",
    "option_contract_receipt",
]
