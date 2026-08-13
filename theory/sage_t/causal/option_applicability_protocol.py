"""Frozen T12.4a.4b protocol for causal-option applicability diagnosis."""

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
from .option_minimization_experiment import _load_contextual_option
from .option_transfer_protocol import (
    load_option_transfer_manifest,
    load_option_transfer_receipt,
)
from .options import MinimalCausalOption
from .witness_reconfirmation_protocol import load_reconfirmation_registry

OPTION_APPLICABILITY_PROTOCOL_FORMAT = (
    "sage-t12.4a.4b-option-applicability-protocol-v1"
)
OPTION_APPLICABILITY_MANIFEST_FORMAT = (
    "sage-t12.4a.4b-option-applicability-manifest-v1"
)
OPTION_APPLICABILITY_RECEIPT_FORMAT = (
    "sage-t12.4a.4b-option-applicability-receipt-v1"
)

OPTION_APPLICABILITY_CODE_PATHS = (
    "theory/sage_t/causal/option_applicability_protocol.py",
    "theory/sage_t/causal/option_applicability_experiment.py",
    "theory/sage_t/causal/option_applicability_cli.py",
    "theory/sage_t/causal/option_transfer_protocol.py",
    "theory/sage_t/causal/option_transfer_experiment.py",
    "theory/sage_t/causal/option_minimization_experiment.py",
    "theory/sage_t/causal/options.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/contracts.py",
    "theory/sage/live_prefix_counterfactual_collector.py",
)

AUTHORIZED_DIAGNOSES = frozenset(
    {
        "INITIATION_AND_DYNAMICS_SHIFT",
        "DYNAMICS_CONTEXT_SHIFT",
        "INITIATION_GOAL_CONTEXT_SHIFT",
        "TERMINATION_PREDICATE_CONTEXT_SHIFT",
        "REPRESENTATION_INSUFFICIENT",
    }
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
class OptionApplicabilityProtocol:
    """Immutable diagnostic design; it grants no policy authority."""

    format_version: str = OPTION_APPLICABILITY_PROTOCOL_FORMAT
    source_seeds: tuple[int, ...] = (8_701, 8_705)
    context_names: tuple[str, ...] = (
        "successful_level0",
        "failed_level1",
    )
    branch_names: tuple[str, ...] = ("option_full", "null")
    expected_option_actions: tuple[str, ...] = (
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    )
    omitted_witness_suffix_actions: int = 6
    repetitions_per_context_lineage_branch: int = 2
    maximum_sdk_calls: int = 1_200
    maximum_terminal_failures: int = 0
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    require_exact_prefix_rate: float = 1.0
    require_expected_contrast: bool = True
    persist_raw_frames: bool = False
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name, caster in (
            ("source_seeds", int),
            ("context_names", str),
            ("branch_names", str),
            ("expected_option_actions", str),
        ):
            object.__setattr__(
                self,
                name,
                tuple(caster(value) for value in getattr(self, name)),
            )
        expected = {
            "format_version": OPTION_APPLICABILITY_PROTOCOL_FORMAT,
            "source_seeds": (8_701, 8_705),
            "context_names": ("successful_level0", "failed_level1"),
            "branch_names": ("option_full", "null"),
            "expected_option_actions": (
                "ACTION4",
                "ACTION4",
                "ACTION4",
                "ACTION3",
                "ACTION3",
            ),
            "omitted_witness_suffix_actions": 6,
            "repetitions_per_context_lineage_branch": 2,
            "maximum_sdk_calls": 1_200,
            "maximum_terminal_failures": 0,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "require_exact_prefix_rate": 1.0,
            "require_expected_contrast": True,
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.4a.4b preregistered value changed: {name}")

    @property
    def expected_trial_count(self) -> int:
        return (
            len(self.context_names)
            * len(self.branch_names)
            * len(self.source_seeds)
            * self.repetitions_per_context_lineage_branch
        )

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
        raise ValueError(f"T12.4a.4b artifact mismatch: {path.name}")
    return path, {"path": _bound_path(path, root=root), "sha256": expected}


def _validate_negative_parent(receipt: Mapping[str, Any]) -> None:
    if receipt.get("phase") != "option_transfer":
        raise ValueError("T12.4a.4b requires the option-transfer receipt")
    if receipt.get("status") != "FAIL_T12_4A_4_OPTION_TRANSFER_GATE":
        raise ValueError("T12.4a.4b requires the sealed T12.4a.4 negative result")
    if receipt.get("passed") is not False:
        raise ValueError("T12.4a.4b parent must be a failed scientific gate")
    metrics = dict(receipt.get("metrics", {}))
    checks = dict(metrics.get("checks", {}))
    stages = list(metrics.get("stage_results", ()))
    if not (
        metrics.get("attempted_transfer_levels") == 1
        and metrics.get("confirmed_transfer_levels") == 0
        and metrics.get("prefix_exact_trials") == metrics.get("trial_count") == 20
        and checks.get("all_executed_prefixes_exact") is True
        and checks.get("no_terminal_failures") is True
        and checks.get("strict_lineage_pairing") is True
        and len(stages) == 1
    ):
        raise ValueError("T12.4a.4 failure is not the preregistered clean transfer miss")
    branch_metrics = dict(stages[0].get("branch_metrics", {}))
    full = dict(branch_metrics.get("option_full", {}))
    if not (
        full.get("trials") == 4
        and full.get("prefix_exact_trials") == 4
        and full.get("available_trials") == 4
        and full.get("progressions") == 0
        and full.get("terminal_failures") == 0
        and full.get("deterministic") is True
    ):
        raise ValueError("T12.4a.4 option miss lacks clean deterministic evidence")


def freeze_option_applicability(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: OptionApplicabilityProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or OptionApplicabilityProtocol()
    parent_manifest = load_option_transfer_manifest(
        parent_manifest_path,
        root=repo_root,
    )
    parent_receipt = load_option_transfer_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=False,
    )
    _validate_negative_parent(parent_receipt)
    if not parent_manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.4 parent did not authorize scientific execution")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.4a.4b is restricted to source_train")

    witness_path, witness_meta = _verified_artifact(
        parent_manifest["inputs"]["witness_registry"], root=repo_root
    )
    option_path, option_meta = _verified_artifact(
        parent_manifest["inputs"]["minimal_option"], root=repo_root
    )
    _, witnesses = load_reconfirmation_registry(witness_path)
    if tuple(item.source_seed for item in witnesses) != selected.source_seeds:
        raise ValueError("T12.4a.4b witness lineages differ from preregistration")
    contextual = _load_contextual_option(option_path)
    option = MinimalCausalOption.from_dict(contextual["option"])
    if option.checksum != parent_manifest["inputs"]["option_checksum"]:
        raise ValueError("T12.4a.4b option checksum mismatch")
    if tuple(step.action_name for step in option.steps) != selected.expected_option_actions:
        raise ValueError("T12.4a.4b option actions differ from preregistration")
    bindings = {int(item["seed"]): dict(item) for item in contextual["context_bindings"]}
    for witness in witnesses:
        binding = bindings.get(witness.source_seed)
        prefix_length = len(witness.steps) - selected.omitted_witness_suffix_actions
        if binding is None or int(binding["prefix_length"]) != prefix_length:
            raise ValueError("T12.4a.4b successful-context prefix is not sealed")
        if prefix_length <= 0:
            raise ValueError("T12.4a.4b successful-context prefix is empty")

    parent_trials_path, parent_trials_meta = _verified_artifact(
        parent_receipt["artifacts"]["trials"], root=repo_root
    )
    parent_report_path, parent_report_meta = _verified_artifact(
        parent_receipt["artifacts"]["report"], root=repo_root
    )
    del parent_trials_path, parent_report_path
    missing = [
        path for path in OPTION_APPLICABILITY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.4a.4b code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": OPTION_APPLICABILITY_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_4B_APPLICABILITY_AUDIT",
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
            "negative_receipt": {
                "path": _bound_path(parent_receipt_path, root=repo_root),
                "sha256": _file_sha256(parent_receipt_path),
                "receipt_checksum": parent_receipt["receipt_checksum"],
                "status": parent_receipt["status"],
            },
        },
        "inputs": {
            "witness_registry": witness_meta,
            "minimal_option": option_meta,
            "parent_transfer_trials": parent_trials_meta,
            "parent_transfer_report": parent_report_meta,
            "option_checksum": option.checksum,
            "successful_prefix_lengths": {
                str(seed): int(bindings[seed]["prefix_length"])
                for seed in selected.source_seeds
            },
            "successful_anchor_hashes": {
                str(seed): str(bindings[seed]["initiation_exact_hash"])
                for seed in selected.source_seeds
            },
            "failed_anchor_hash": str(parent_manifest["inputs"]["entry_exact_hash"]),
            "failed_anchor_level": int(parent_manifest["inputs"]["entry_level"]),
        },
        "decision_rule": {
            "authorized_diagnoses": sorted(AUTHORIZED_DIAGNOSES),
            "integrity_precedes_classification": True,
            "classification_is_mutually_exclusive": True,
            "no_threshold_retuning_after_collection": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in OPTION_APPLICABILITY_CODE_PATHS
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
            "option_applicability_audit_authorized": authorized,
            "t12_4a_4c_option_contract_freeze_authorized": False,
            "t12_4a_4c_representation_extension_freeze_authorized": False,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = option_applicability_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_4A_4B_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "expected_trial_count": selected.expected_trial_count,
            "maximum_sdk_calls": selected.maximum_sdk_calls,
            "source_lineages": list(selected.source_seeds),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_option_applicability_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != OPTION_APPLICABILITY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a.4b manifest")
    protocol = OptionApplicabilityProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a.4b protocol checksum mismatch")
    for meta in manifest["parent"].values():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.4a.4b parent artifact mismatch")
    for name in (
        "witness_registry",
        "minimal_option",
        "parent_transfer_trials",
        "parent_transfer_report",
    ):
        meta = dict(manifest["inputs"][name])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.4b input artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a.4b code checksum mismatch: {relative}")
    return manifest


def option_applicability_receipt(
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
            "format_version": OPTION_APPLICABILITY_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_4_receipt_checksum": manifest["parent"][
                "negative_receipt"
            ]["receipt_checksum"],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_option_applicability_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != OPTION_APPLICABILITY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a.4b receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.4a.4b receipt belongs to another manifest")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.4a.4b gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.4b receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "AUTHORIZED_DIAGNOSES",
    "OptionApplicabilityProtocol",
    "freeze_option_applicability",
    "load_option_applicability_manifest",
    "load_option_applicability_receipt",
    "option_applicability_receipt",
]
