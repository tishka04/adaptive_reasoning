"""Frozen protocol for T12.5b observed-effect shadow ranking."""

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
from .progress_protocol import (
    load_causal_progress_manifest,
    load_causal_progress_receipt,
)

SHADOW_PROTOCOL_FORMAT = "sage-t12.5b-progress-shadow-protocol-v2"
SHADOW_MANIFEST_FORMAT = "sage-t12.5b-progress-shadow-manifest-v2"
SHADOW_RECEIPT_FORMAT = "sage-t12.5b-progress-shadow-receipt-v2"

SHADOW_CODE_PATHS = (
    "theory/sage_t/causal/__init__.py",
    "theory/sage_t/causal/progress.py",
    "theory/sage_t/causal/progress_shadow.py",
    "theory/sage_t/causal/progress_shadow_protocol.py",
    "theory/sage_t/causal/progress_shadow_experiment.py",
    "theory/sage_t/causal/progress_shadow_cli.py",
    "theory/sage_t/causal/option_applicability_experiment.py",
    "theory/sage_t/causal/witness_experiment.py",
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
class ProgressShadowProtocol:
    format_version: str = SHADOW_PROTOCOL_FORMAT
    lineage_seeds: tuple[int, ...] = (8_701, 8_705)
    induction_lineage_seed: int = 8_701
    confirmation_lineage_seed: int = 8_705
    stages: tuple[int, ...] = (0, 1, 2, 3, 4)
    candidate_actions: tuple[str, ...] = (
        "ACTION3",
        "ACTION4",
        "ACTION6",
    )
    excluded_non_executable_actions: tuple[str, ...] = ("ACTION7",)
    expected_actions: tuple[str, ...] = (
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    )
    allowed_effect_features: tuple[str, ...] = (
        "predicate_counts.adjacent",
        "predicate_counts.aligned",
        "predicate_counts.contact",
        "predicate_counts.near",
        "role_counts.clickable",
        "role_counts.movable",
    )
    repetitions_per_branch: int = 2
    maximum_sdk_calls: int = 5_000
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    minimum_exact_prefix_rate: float = 1.0
    minimum_branch_availability_rate: float = 1.0
    minimum_effect_determinism_rate: float = 1.0
    minimum_causal_top1_accuracy: float = 1.0
    minimum_causal_mean_reciprocal_rank: float = 1.0
    minimum_positive_margin: float = 1e-6
    minimum_baseline_mrr_gain: float = 0.05
    maximum_terminal_failures: int = 0
    persist_raw_frames: bool = False

    def __post_init__(self) -> None:
        for name, caster in (
            ("lineage_seeds", int),
            ("stages", int),
            ("candidate_actions", str),
            ("excluded_non_executable_actions", str),
            ("expected_actions", str),
            ("allowed_effect_features", str),
        ):
            object.__setattr__(
                self, name, tuple(caster(item) for item in getattr(self, name))
            )
        expected = {
            "format_version": SHADOW_PROTOCOL_FORMAT,
            "lineage_seeds": (8_701, 8_705),
            "induction_lineage_seed": 8_701,
            "confirmation_lineage_seed": 8_705,
            "stages": (0, 1, 2, 3, 4),
            "candidate_actions": ("ACTION3", "ACTION4", "ACTION6"),
            "excluded_non_executable_actions": ("ACTION7",),
            "expected_actions": (
                "ACTION4",
                "ACTION4",
                "ACTION4",
                "ACTION3",
                "ACTION3",
            ),
            "allowed_effect_features": (
                "predicate_counts.adjacent",
                "predicate_counts.aligned",
                "predicate_counts.contact",
                "predicate_counts.near",
                "role_counts.clickable",
                "role_counts.movable",
            ),
            "repetitions_per_branch": 2,
            "maximum_sdk_calls": 5_000,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "minimum_exact_prefix_rate": 1.0,
            "minimum_branch_availability_rate": 1.0,
            "minimum_effect_determinism_rate": 1.0,
            "minimum_causal_top1_accuracy": 1.0,
            "minimum_causal_mean_reciprocal_rank": 1.0,
            "minimum_positive_margin": 1e-6,
            "minimum_baseline_mrr_gain": 0.05,
            "maximum_terminal_failures": 0,
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.5b preregistered value changed: {name}")
        if self.induction_lineage_seed == self.confirmation_lineage_seed:
            raise ValueError("T12.5b induction and confirmation must be disjoint")
        if set(self.lineage_seeds) != {
            self.induction_lineage_seed,
            self.confirmation_lineage_seed,
        }:
            raise ValueError("T12.5b lineage split is inconsistent")
        if len(self.stages) != len(self.expected_actions):
            raise ValueError("T12.5b stage/action schedule is inconsistent")

    @property
    def expected_trial_count(self) -> int:
        return (
            len(self.lineage_seeds)
            * len(self.stages)
            * len(self.candidate_actions)
            * self.repetitions_per_branch
        )

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
        raise ValueError(f"T12.5b input artifact mismatch: {path}")
    return path, {"path": _bound_path(path, root=root), "sha256": expected}


def freeze_progress_shadow(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: ProgressShadowProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or ProgressShadowProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    parent_manifest = load_causal_progress_manifest(
        parent_manifest_path, root=repo_root, verify_code=True
    )
    parent_receipt = load_causal_progress_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
    )
    if parent_receipt.get("status") != "PASS_T12_5_CAUSAL_PROGRESS_GATE":
        raise ValueError("T12.5b requires the passed T12.5 progress gate")
    checks = dict(parent_receipt.get("metrics", {}).get("checks", {}))
    if not checks or not all(checks.values()):
        raise ValueError("T12.5b parent has an incomplete integrity gate")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.5b is restricted to source_train")

    parent_artifacts = {}
    for name in ("posterior", "program_registry", "report"):
        _, parent_artifacts[name] = _verified_artifact(
            parent_receipt["artifacts"][name], root=repo_root
        )
    applicability_manifest_path = _resolve_bound(
        str(parent_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    contract_manifest = _read_json(applicability_manifest_path)
    option_applicability_manifest_path = _resolve_bound(
        str(contract_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    option_applicability_manifest = _read_json(option_applicability_manifest_path)
    witness_registry_path, witness_registry = _verified_artifact(
        option_applicability_manifest["inputs"]["witness_registry"], root=repo_root
    )
    minimal_option_path, minimal_option = _verified_artifact(
        parent_manifest["inputs"]["minimal_option"], root=repo_root
    )
    applicability_trials_path, applicability_trials = _verified_artifact(
        parent_manifest["inputs"]["applicability_trials"], root=repo_root
    )
    del witness_registry_path, minimal_option_path, applicability_trials_path

    missing = [path for path in SHADOW_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.5b code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": SHADOW_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_5B_PROGRESS_SHADOW",
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
            **parent_artifacts,
            "applicability_trials": applicability_trials,
            "minimal_option": minimal_option,
            "witness_registry": witness_registry,
            "successful_prefix_lengths": {
                str(seed): int(
                    option_applicability_manifest["inputs"]["successful_prefix_lengths"][
                        str(seed)
                    ]
                )
                for seed in selected.lineage_seeds
            },
            "successful_anchor_hashes": {
                str(seed): str(
                    option_applicability_manifest["inputs"]["successful_anchor_hashes"][
                        str(seed)
                    ]
                )
                for seed in selected.lineage_seeds
            },
        },
        "design": {
            "candidate_actions_are_fixed_before_collection": True,
            "candidate_actions_are_sdk_executable_at_every_stage": True,
            "excluded_non_executable_actions": list(
                selected.excluded_non_executable_actions
            ),
            "exclusion_reason": (
                "ACTION7 is advertised in the frame signature but is absent from "
                "the SDK executable action set at all five sealed stage anchors"
            ),
            "every_candidate_executes_from_the_same_exact_stage_prefix": True,
            "effect_model_fit_lineage": selected.induction_lineage_seed,
            "effect_model_confirmation_lineage": selected.confirmation_lineage_seed,
            "rankings_are_computed_after_collection_only": True,
            "rankings_never_select_executed_actions": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in SHADOW_CODE_PATHS
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
            "causal_progress_shadow_collection_authorized": authorized,
            "t12_5c_control_freeze_authorized": False,
            "t12_6_freeze_authorized": False,
        },
        "claim_boundary": {
            "authorized": "source-train observed-effect shadow ranking",
            "not_authorized": [
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
            "maximum_sdk_calls": selected.maximum_sdk_calls,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = progress_shadow_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_5B_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "expected_trials": selected.expected_trial_count,
            "maximum_sdk_calls": selected.maximum_sdk_calls,
            "ranking_authority": False,
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_progress_shadow_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != SHADOW_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.5b progress-shadow manifest")
    protocol = ProgressShadowProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.5b protocol checksum mismatch")
    for meta in manifest["parent"].values():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.5b parent artifact mismatch")
    for name in (
        "posterior",
        "program_registry",
        "report",
        "minimal_option",
        "witness_registry",
        "applicability_trials",
    ):
        meta = manifest["inputs"][name]
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b input artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.5b code checksum mismatch: {relative}")
    return manifest


def progress_shadow_receipt(
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
            "format_version": SHADOW_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_5_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_progress_shadow_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != SHADOW_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.5b progress-shadow receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.5b receipt belongs to another manifest")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.5b gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "ProgressShadowProtocol",
    "freeze_progress_shadow",
    "load_progress_shadow_manifest",
    "load_progress_shadow_receipt",
    "progress_shadow_receipt",
]
