"""Frozen T12.4a.4 protocol for exact-prefix multi-level option transfer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .contracts import causal_program_from_dict
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
from .option_minimization_protocol import (
    load_option_minimization_manifest,
    load_option_minimization_receipt,
)
from .options import CompiledCausalOption, MinimalCausalOption
from .witness_reconfirmation_protocol import load_reconfirmation_registry

OPTION_TRANSFER_PROTOCOL_FORMAT = "sage-t12.4a.4-option-transfer-protocol-v1"
OPTION_TRANSFER_MANIFEST_FORMAT = "sage-t12.4a.4-option-transfer-manifest-v1"
OPTION_TRANSFER_RECEIPT_FORMAT = "sage-t12.4a.4-option-transfer-receipt-v1"

OPTION_TRANSFER_CODE_PATHS = (
    "theory/sage_t/causal/option_transfer_protocol.py",
    "theory/sage_t/causal/option_transfer_experiment.py",
    "theory/sage_t/causal/option_transfer_cli.py",
    "theory/sage_t/causal/option_minimization_protocol.py",
    "theory/sage_t/causal/option_minimization_experiment.py",
    "theory/sage_t/causal/options.py",
    "theory/sage_t/causal/posterior.py",
    "theory/sage_t/causal/executor.py",
    "theory/sage_t/causal/contracts.py",
    "theory/sage_t/causal/witness_reconfirmation_protocol.py",
    "theory/sage_t/causal/witness_experiment.py",
    "theory/sage_t/causal/graph_experiment.py",
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
class OptionTransferProtocol:
    format_version: str = OPTION_TRANSFER_PROTOCOL_FORMAT
    source_seeds: tuple[int, ...] = (8_701, 8_705)
    expected_option_actions: tuple[str, ...] = (
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    )
    branch_names: tuple[str, ...] = (
        "option_full",
        "delete_action4",
        "delete_action3",
        "reverse",
        "null",
    )
    repetitions_per_branch: int = 4
    lineage_schedule: tuple[int, ...] = (8_701, 8_705, 8_701, 8_705)
    minimum_transferred_levels: int = 2
    maximum_transfer_levels: int = 3
    require_unit_level_delta: bool = True
    require_progress_on_final_option_action: bool = True
    maximum_terminal_failures: int = 0
    maximum_sdk_calls: int = 4_500
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    minimum_posterior_owner_mass: float = 0.80
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name, caster in (
            ("source_seeds", int),
            ("expected_option_actions", str),
            ("branch_names", str),
            ("lineage_schedule", int),
        ):
            object.__setattr__(
                self,
                name,
                tuple(caster(value) for value in getattr(self, name)),
            )
        expected = {
            "format_version": OPTION_TRANSFER_PROTOCOL_FORMAT,
            "source_seeds": (8_701, 8_705),
            "expected_option_actions": (
                "ACTION4",
                "ACTION4",
                "ACTION4",
                "ACTION3",
                "ACTION3",
            ),
            "branch_names": (
                "option_full",
                "delete_action4",
                "delete_action3",
                "reverse",
                "null",
            ),
            "repetitions_per_branch": 4,
            "lineage_schedule": (8_701, 8_705, 8_701, 8_705),
            "minimum_transferred_levels": 2,
            "maximum_transfer_levels": 3,
            "require_unit_level_delta": True,
            "require_progress_on_final_option_action": True,
            "maximum_terminal_failures": 0,
            "maximum_sdk_calls": 4_500,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "minimum_posterior_owner_mass": 0.80,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.4a.4 preregistered value changed: {name}")
        if len(self.lineage_schedule) != self.repetitions_per_branch:
            raise ValueError("T12.4a.4 needs one lineage per repetition")
        if set(self.lineage_schedule) != set(self.source_seeds):
            raise ValueError("T12.4a.4 lineage schedule is incomplete")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _artifact_meta(
    raw_meta: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    path = _resolve_bound(str(raw_meta["path"]), root=root)
    if not path.is_file() or _file_sha256(path) != str(raw_meta["sha256"]):
        raise ValueError(f"T12.4a.4 parent artifact mismatch: {path.name}")
    return {
        "path": _bound_path(path, root=root),
        "sha256": str(raw_meta["sha256"]),
    }


def _validate_compiled_parent(
    *,
    contextual_payload: Mapping[str, Any],
    compiled_payload: Mapping[str, Any],
    programs_payload: Mapping[str, Any],
    posterior_payload: Mapping[str, Any],
    minimum_owner_mass: float,
) -> tuple[MinimalCausalOption, CompiledCausalOption, float]:
    option = MinimalCausalOption.from_dict(dict(contextual_payload["option"]))
    compiled = CompiledCausalOption.from_dict(compiled_payload)
    if option.checksum != contextual_payload["option_checksum"]:
        raise ValueError("T12.4a.4 contextual option checksum mismatch")
    if compiled.option.checksum != option.checksum:
        raise ValueError("T12.4a.4 compiled option differs from minimal option")
    programs = tuple(
        causal_program_from_dict(dict(item))
        for item in programs_payload.get("programs", ())
    )
    if not programs or {item.canonical_hash for item in programs} != set(
        compiled.owner_program_hashes
    ):
        raise ValueError("T12.4a.4 compiled owner programs mismatch")
    probabilities = {
        str(item["program_hash"]): float(item["probability"])
        for item in posterior_payload.get("particles", ())
    }
    owner_mass = sum(
        probabilities.get(owner_hash, 0.0)
        for owner_hash in compiled.owner_program_hashes
    )
    if owner_mass + 1e-12 < minimum_owner_mass:
        raise ValueError("T12.4a.4 parent posterior owner mass is too low")
    return option, compiled, owner_mass


def freeze_option_transfer(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    ablation_receipt_path: str | Path,
    compile_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: OptionTransferProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or OptionTransferProtocol()
    parent_manifest = load_option_minimization_manifest(
        parent_manifest_path,
        root=repo_root,
    )
    ablation = load_option_minimization_receipt(
        ablation_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
    )
    compilation = load_option_minimization_receipt(
        compile_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
    )
    if not (
        ablation.get("phase") == "option_ablation"
        and ablation.get("status") == "PASS_T12_4A_3_OPTION_ABLATION_GATE"
    ):
        raise ValueError("T12.4a.4 requires the passed option-ablation gate")
    if not (
        compilation.get("phase") == "shadow_compile"
        and compilation.get("status") == "PASS_T12_4A_3_SHADOW_COMPILE_GATE"
        and compilation.get("parent_receipt_checksum")
        == ablation.get("receipt_checksum")
    ):
        raise ValueError("T12.4a.4 requires the chained shadow-compile gate")
    if not parent_manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.4 parent did not authorize scientific claims")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.4a.4 is restricted to source_train")

    option_meta = _artifact_meta(ablation["artifacts"]["minimal_option"], root=repo_root)
    compiled_meta = _artifact_meta(
        compilation["artifacts"]["compiled_option_registry"],
        root=repo_root,
    )
    programs_meta = _artifact_meta(
        compilation["artifacts"]["option_programs"],
        root=repo_root,
    )
    posterior_meta = _artifact_meta(
        compilation["artifacts"]["posterior_snapshot"],
        root=repo_root,
    )
    contextual = _load_contextual_option(
        _resolve_bound(option_meta["path"], root=repo_root)
    )
    compiled_payload = _read_json(_resolve_bound(compiled_meta["path"], root=repo_root))
    programs_payload = _read_json(_resolve_bound(programs_meta["path"], root=repo_root))
    posterior_payload = _read_json(
        _resolve_bound(posterior_meta["path"], root=repo_root)
    )
    option, compiled, owner_mass = _validate_compiled_parent(
        contextual_payload=contextual,
        compiled_payload=compiled_payload,
        programs_payload=programs_payload,
        posterior_payload=posterior_payload,
        minimum_owner_mass=selected.minimum_posterior_owner_mass,
    )
    action_names = tuple(step.action_name for step in option.steps)
    if action_names != selected.expected_option_actions:
        raise ValueError("T12.4a.4 option differs from preregistration")

    witness_meta = dict(parent_manifest["parent"]["witness_registry"])
    witness_path = _resolve_bound(str(witness_meta["path"]), root=repo_root)
    _, witnesses = load_reconfirmation_registry(witness_path)
    if tuple(item.source_seed for item in witnesses) != selected.source_seeds:
        raise ValueError("T12.4a.4 witness lineages differ from preregistration")
    if len({item.target_exact_hash for item in witnesses}) != 1:
        raise ValueError("T12.4a.4 witness routes do not share an exact entry")

    missing = [path for path in OPTION_TRANSFER_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.4a.4 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": OPTION_TRANSFER_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_4_OPTION_TRANSFER",
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
            "ablation_receipt": {
                "path": _bound_path(ablation_receipt_path, root=repo_root),
                "sha256": _file_sha256(ablation_receipt_path),
                "receipt_checksum": ablation["receipt_checksum"],
                "status": ablation["status"],
            },
            "compile_receipt": {
                "path": _bound_path(compile_receipt_path, root=repo_root),
                "sha256": _file_sha256(compile_receipt_path),
                "receipt_checksum": compilation["receipt_checksum"],
                "status": compilation["status"],
            },
        },
        "inputs": {
            "witness_registry": {
                "path": _bound_path(witness_path, root=repo_root),
                "sha256": _file_sha256(witness_path),
                "registry_checksum": witness_meta["registry_checksum"],
            },
            "minimal_option": option_meta,
            "compiled_option_registry": compiled_meta,
            "option_programs": programs_meta,
            "posterior_snapshot": posterior_meta,
            "option_checksum": option.checksum,
            "compiled_registry_checksum": compiled_payload["registry_checksum"],
            "posterior_owner_mass": owner_mass,
            "entry_exact_hash": witnesses[0].target_exact_hash,
            "entry_level": witnesses[0].target_level,
            "route_lengths": [len(item.steps) for item in witnesses],
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in OPTION_TRANSFER_CODE_PATHS
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
            "option_transfer_experiment_authorized": authorized,
            "option_control_authorized": False,
            "t12_4a_5_option_control_freeze_authorized": False,
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
    receipt = option_transfer_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_4A_4_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "branch_names": list(selected.branch_names),
            "maximum_branches": (
                selected.maximum_transfer_levels
                * len(selected.branch_names)
                * selected.repetitions_per_branch
            ),
            "maximum_sdk_calls": selected.maximum_sdk_calls,
            "source_lineages": list(selected.source_seeds),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_option_transfer_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != OPTION_TRANSFER_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a.4 manifest")
    protocol = OptionTransferProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a.4 protocol checksum mismatch")
    for meta in manifest["parent"].values():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.4a.4 parent artifact mismatch")
    for name in (
        "witness_registry",
        "minimal_option",
        "compiled_option_registry",
        "option_programs",
        "posterior_snapshot",
    ):
        meta = dict(manifest["inputs"][name])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.4 input artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a.4 code checksum mismatch: {relative}")
    return manifest


def option_transfer_receipt(
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
            "format_version": OPTION_TRANSFER_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_3_ablation_receipt_checksum": manifest["parent"][
                "ablation_receipt"
            ]["receipt_checksum"],
            "parent_t12_4a_3_compile_receipt_checksum": manifest["parent"][
                "compile_receipt"
            ]["receipt_checksum"],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_option_transfer_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != OPTION_TRANSFER_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a.4 receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.4a.4 receipt belongs to another manifest")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.4a.4 gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.4 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "OptionTransferProtocol",
    "freeze_option_transfer",
    "load_option_transfer_manifest",
    "load_option_transfer_receipt",
    "option_transfer_receipt",
]
