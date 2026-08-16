"""Frozen offline protocol for SAGE.T12.5b.2 progress discrimination."""

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
from .progress_shadow_protocol import (
    load_progress_shadow_manifest,
    load_progress_shadow_receipt,
)

DISCRIMINATION_PROTOCOL_FORMAT = "sage-t12.5b.2-discrimination-protocol-v1"
DISCRIMINATION_MANIFEST_FORMAT = "sage-t12.5b.2-discrimination-manifest-v1"
DISCRIMINATION_RECEIPT_FORMAT = "sage-t12.5b.2-discrimination-receipt-v1"

DISCRIMINATION_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/progress.py",
    "theory/sage_t/causal/progress_shadow.py",
    "theory/sage_t/causal/progress_discrimination.py",
    "theory/sage_t/causal/progress_discrimination_protocol.py",
    "theory/sage_t/causal/progress_discrimination_experiment.py",
    "theory/sage_t/causal/progress_discrimination_cli.py",
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
class ProgressDiscriminationProtocol:
    format_version: str = DISCRIMINATION_PROTOCOL_FORMAT
    parent_status: str = "FAIL_T12_5B_PROGRESS_SHADOW_GATE"
    authorized_parent_failed_checks: tuple[str, ...] = (
        "all_candidate_actions_available",
        "causal_ranking_beats_non_goal_baselines",
    )
    lineage_seeds: tuple[int, ...] = (8_701, 8_705)
    induction_lineage_seed: int = 8_701
    confirmation_lineage_seed: int = 8_705
    stages: tuple[int, ...] = (0, 1, 2, 3, 4)
    expected_actions: tuple[str, ...] = (
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    )
    repetitions_per_branch: int = 2
    minimum_executable_actions_per_context: int = 2
    minimum_affordance_binding_coverage: float = 1.0
    minimum_distractor_magnitude_gap: float = 1.0
    minimum_hard_contrasts_per_lineage: int = 1
    minimum_hard_contrast_lineages: int = 2
    minimum_causal_hard_contrast_accuracy: float = 1.0
    minimum_hard_contrast_accuracy_gain: float = 0.5
    maximum_sdk_calls: int = 0
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    persist_raw_trials: bool = False

    def __post_init__(self) -> None:
        for name, caster in (
            ("authorized_parent_failed_checks", str),
            ("lineage_seeds", int),
            ("stages", int),
            ("expected_actions", str),
        ):
            object.__setattr__(
                self, name, tuple(caster(item) for item in getattr(self, name))
            )
        expected = {
            "format_version": DISCRIMINATION_PROTOCOL_FORMAT,
            "parent_status": "FAIL_T12_5B_PROGRESS_SHADOW_GATE",
            "authorized_parent_failed_checks": (
                "all_candidate_actions_available",
                "causal_ranking_beats_non_goal_baselines",
            ),
            "lineage_seeds": (8_701, 8_705),
            "induction_lineage_seed": 8_701,
            "confirmation_lineage_seed": 8_705,
            "stages": (0, 1, 2, 3, 4),
            "expected_actions": (
                "ACTION4",
                "ACTION4",
                "ACTION4",
                "ACTION3",
                "ACTION3",
            ),
            "repetitions_per_branch": 2,
            "minimum_executable_actions_per_context": 2,
            "minimum_affordance_binding_coverage": 1.0,
            "minimum_distractor_magnitude_gap": 1.0,
            "minimum_hard_contrasts_per_lineage": 1,
            "minimum_hard_contrast_lineages": 2,
            "minimum_causal_hard_contrast_accuracy": 1.0,
            "minimum_hard_contrast_accuracy_gain": 0.5,
            "maximum_sdk_calls": 0,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "persist_raw_trials": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.5b.2 preregistered value changed: {name}")
        if set(self.lineage_seeds) != {
            self.induction_lineage_seed,
            self.confirmation_lineage_seed,
        }:
            raise ValueError("T12.5b.2 lineage split is inconsistent")
        if len(self.stages) != len(self.expected_actions):
            raise ValueError("T12.5b.2 stage/action schedule is inconsistent")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _verified_artifact(
    meta: Mapping[str, Any], *, root: Path
) -> dict[str, str]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.5b.2 input artifact mismatch: {path}")
    return {"path": _bound_path(path, root=root), "sha256": expected}


def freeze_progress_discrimination(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: ProgressDiscriminationProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or ProgressDiscriminationProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    parent_manifest = load_progress_shadow_manifest(
        parent_manifest_path, root=repo_root, verify_code=True
    )
    parent_receipt = load_progress_shadow_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=False,
    )
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.5b.2 requires the sealed negative T12.5b-r1 result")
    if parent_receipt.get("status") != selected.parent_status:
        raise ValueError("T12.5b.2 parent failure status changed")
    checks = dict(parent_receipt.get("metrics", {}).get("checks", {}))
    failed_checks = tuple(sorted(name for name, passed in checks.items() if not passed))
    if failed_checks != tuple(sorted(selected.authorized_parent_failed_checks)):
        raise ValueError("T12.5b.2 parent failure class changed")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.5b.2 is restricted to source_train")

    artifacts = {
        name: _verified_artifact(meta, root=repo_root)
        for name, meta in parent_receipt["artifacts"].items()
        if name in {"effect_model", "rankings", "report", "trials"}
    }
    if set(artifacts) != {"effect_model", "rankings", "report", "trials"}:
        raise ValueError("T12.5b.2 parent artifact inventory is incomplete")
    parent_inputs = {
        name: _verified_artifact(parent_manifest["inputs"][name], root=repo_root)
        for name in ("posterior", "program_registry")
    }
    missing = [
        path for path in DISCRIMINATION_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.5b.2 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": DISCRIMINATION_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_5B_2_OFFLINE_AUDIT",
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
            "failed_checks": list(failed_checks),
        },
        "inputs": {**artifacts, **parent_inputs},
        "design": {
            "existing_observations_only": True,
            "unavailable_action_is_not_a_zero_effect": True,
            "candidate_sets_are_local_to_each_exact_context": True,
            "affordance_binding_fields": ["stage", "milestone_signature"],
            "action_name_is_not_an_affordance_binding_field": True,
            "hard_contrast_requires_larger_nonprogress_magnitude": True,
            "parent_negative_result_is_preserved": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in DISCRIMINATION_CODE_PATHS
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
            "causal_progress_control_authorized": False,
            "affordance_discrimination_audit_authorized": authorized,
            "environment_collection_authorized": False,
            "t12_5b_3_collection_freeze_authorized": False,
            "t12_5c_control_freeze_authorized": False,
            "t12_6_freeze_authorized": False,
        },
        "claim_boundary": {
            "authorized": "source-train offline affordance and contrast audit",
            "not_authorized": [
                "new environment evidence",
                "policy improvement",
                "environment control",
                "target-game generalization",
                "source validation",
                "holdout performance",
                "neural training",
            ],
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "maximum_sdk_calls": 0,
            "persist_raw_trials": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = progress_discrimination_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_5B_2_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "environment_calls_authorized": 0,
            "parent_failed_checks": list(failed_checks),
            "parent_negative_result_preserved": True,
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_progress_discrimination_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != DISCRIMINATION_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.5b.2 discrimination manifest")
    protocol = ProgressDiscriminationProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.5b.2 protocol checksum mismatch")
    for meta in manifest["parent"].values():
        if not isinstance(meta, Mapping) or "path" not in meta:
            continue
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.5b.2 parent artifact mismatch")
    for name, meta in manifest["inputs"].items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.2 input artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.5b.2 code checksum mismatch: {relative}")
    return manifest


def progress_discrimination_receipt(
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
            "format_version": DISCRIMINATION_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_5b_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_progress_discrimination_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != DISCRIMINATION_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.5b.2 discrimination receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.5b.2 receipt belongs to another manifest")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.5b.2 gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.2 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "ProgressDiscriminationProtocol",
    "freeze_progress_discrimination",
    "load_progress_discrimination_manifest",
    "load_progress_discrimination_receipt",
    "progress_discrimination_receipt",
]
