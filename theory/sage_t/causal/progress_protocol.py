"""Frozen offline protocol for SAGE.T12.5 causal goal-progress induction."""

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
from .option_contract_protocol import (
    load_option_contract_manifest,
    load_option_contract_receipt,
)
from .option_minimization_protocol import load_option_minimization_receipt

PROGRESS_PROTOCOL_FORMAT = "sage-t12.5-causal-progress-protocol-v1"
PROGRESS_MANIFEST_FORMAT = "sage-t12.5-causal-progress-manifest-v1"
PROGRESS_RECEIPT_FORMAT = "sage-t12.5-causal-progress-receipt-v1"

PROGRESS_CODE_PATHS = (
    "theory/sage_t/causal/__init__.py",
    "theory/sage_t/causal/progress.py",
    "theory/sage_t/causal/progress_protocol.py",
    "theory/sage_t/causal/progress_experiment.py",
    "theory/sage_t/causal/progress_experiment_cli.py",
    "theory/sage_t/causal/option_contracts.py",
    "theory/sage_t/causal/contracts.py",
    "theory/sage_t/causal/posterior.py",
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
class CausalProgressProtocol:
    format_version: str = PROGRESS_PROTOCOL_FORMAT
    induction_lineage_seed: int = 8_701
    replication_lineage_seed: int = 8_705
    rival_progress_kinds: tuple[str, ...] = (
        "terminal_only",
        "change_count",
        "unordered_effects",
        "ordered_effects",
    )
    expected_milestone_count: int = 5
    maximum_owner_programs: int = 24
    maximum_joint_particles: int = 96
    mdl_beta: float = 0.08
    likelihood_match_probability: float = 0.95
    minimum_ordered_posterior_mass: float = 0.95
    minimum_replication_accuracy: float = 1.0
    maximum_parent_mass_error: float = 1e-12
    maximum_sdk_calls: int = 0
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    persist_raw_frames: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rival_progress_kinds", tuple(map(str, self.rival_progress_kinds))
        )
        expected = {
            "format_version": PROGRESS_PROTOCOL_FORMAT,
            "induction_lineage_seed": 8_701,
            "replication_lineage_seed": 8_705,
            "rival_progress_kinds": (
                "terminal_only",
                "change_count",
                "unordered_effects",
                "ordered_effects",
            ),
            "expected_milestone_count": 5,
            "maximum_owner_programs": 24,
            "maximum_joint_particles": 96,
            "mdl_beta": 0.08,
            "likelihood_match_probability": 0.95,
            "minimum_ordered_posterior_mass": 0.95,
            "minimum_replication_accuracy": 1.0,
            "maximum_parent_mass_error": 1e-12,
            "maximum_sdk_calls": 0,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.5 preregistered value changed: {name}")
        if self.maximum_owner_programs * len(self.rival_progress_kinds) != (
            self.maximum_joint_particles
        ):
            raise ValueError("T12.5 joint-particle bound is inconsistent")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _verified_artifact(
    meta: Mapping[str, Any], *, root: Path
) -> tuple[Path, dict[str, str]]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.5 input artifact mismatch: {path}")
    return path, {"path": _bound_path(path, root=root), "sha256": expected}


def freeze_causal_progress(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    ablation_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: CausalProgressProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or CausalProgressProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    ablation_receipt_path = Path(ablation_receipt_path).resolve()

    parent_manifest = load_option_contract_manifest(
        parent_manifest_path, root=repo_root, verify_code=True
    )
    parent_receipt = load_option_contract_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
    )
    if parent_receipt.get("status") != "PASS_T12_4A_4C_OPTION_CONTRACT_GATE":
        raise ValueError("T12.5 requires the passed T12.4a.4c contract gate")
    ablation_receipt = load_option_minimization_receipt(
        ablation_receipt_path, root=repo_root, require_passed=True
    )
    if ablation_receipt.get("status") != "PASS_T12_4A_3_OPTION_ABLATION_GATE":
        raise ValueError("T12.5 requires the passed T12.4a.3 ablation gate")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.5 offline induction is restricted to source_train")

    parent_artifacts = {}
    for name in (
        "contracted_option_programs",
        "contracted_option_registry",
        "contracted_posterior",
        "report",
    ):
        _, parent_artifacts[name] = _verified_artifact(
            parent_receipt["artifacts"][name], root=repo_root
        )
    _, applicability_trials = _verified_artifact(
        parent_manifest["inputs"]["applicability_trials"], root=repo_root
    )
    _, ablation_trials = _verified_artifact(
        ablation_receipt["artifacts"]["ablation_trials"], root=repo_root
    )
    _, minimal_option = _verified_artifact(
        ablation_receipt["artifacts"]["minimal_option"], root=repo_root
    )

    missing = [
        path for path in PROGRESS_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.5 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": PROGRESS_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_5_CAUSAL_PROGRESS",
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
            "ablation_receipt": {
                "path": _bound_path(ablation_receipt_path, root=repo_root),
                "sha256": _file_sha256(ablation_receipt_path),
                "receipt_checksum": ablation_receipt["receipt_checksum"],
                "status": ablation_receipt["status"],
            },
        },
        "inputs": {
            **parent_artifacts,
            "applicability_trials": applicability_trials,
            "ablation_trials": ablation_trials,
            "minimal_option": minimal_option,
        },
        "claim_boundary": {
            "authorized": "source reconstruction and cross-lineage replication",
            "not_authorized": [
                "target-game generalization",
                "production control",
                "holdout performance",
                "neural training",
            ],
            "ablation_order_evidence_has_no_observed_typed_deltas": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in PROGRESS_CODE_PATHS
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
            "causal_progress_compile_authorized": authorized,
            "causal_progress_shadow_experiment_authorized": False,
            "t12_6_freeze_authorized": False,
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
    receipt = causal_progress_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_5_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "maximum_joint_particles": selected.maximum_joint_particles,
            "maximum_sdk_calls": 0,
            "progress_hypotheses": len(selected.rival_progress_kinds),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_causal_progress_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != PROGRESS_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.5 causal-progress manifest")
    protocol = CausalProgressProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.5 protocol checksum mismatch")
    for meta in manifest["parent"].values():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.5 parent artifact mismatch")
    for name, meta in manifest["inputs"].items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5 input artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.5 code checksum mismatch: {relative}")
    return manifest


def causal_progress_receipt(
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
            "format_version": PROGRESS_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_4c_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "parent_t12_4a_3_receipt_checksum": manifest["parent"][
                "ablation_receipt"
            ]["receipt_checksum"],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_causal_progress_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != PROGRESS_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.5 causal-progress receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.5 receipt belongs to another manifest")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.5 gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "CausalProgressProtocol",
    "causal_progress_receipt",
    "freeze_causal_progress",
    "load_causal_progress_manifest",
    "load_causal_progress_receipt",
]
