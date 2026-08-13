"""Frozen T12.4a.3 protocol for exhaustive option minimization and shadow compile."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .calibration_protocol import (
    load_calibration_manifest,
    load_calibration_receipt,
)
from .experiment import (
    _bound_path,
    _file_sha256,
    _git_state,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .witness_reconfirmation_protocol import (
    load_reconfirmation_manifest,
    load_reconfirmation_receipt,
    load_reconfirmation_registry,
)

OPTION_MIN_PROTOCOL_FORMAT = "sage-t12.4a.3-option-minimization-protocol-v1"
OPTION_MIN_MANIFEST_FORMAT = "sage-t12.4a.3-option-minimization-manifest-v1"
OPTION_MIN_RECEIPT_FORMAT = "sage-t12.4a.3-option-minimization-receipt-v1"
CONTEXTUAL_OPTION_FORMAT = "sage-t12.4a.3-contextual-minimal-option-v1"

OPTION_MIN_CODE_PATHS = (
    "theory/sage_t/causal/option_minimization_protocol.py",
    "theory/sage_t/causal/option_minimization_experiment.py",
    "theory/sage_t/causal/option_minimization_cli.py",
    "theory/sage_t/causal/options.py",
    "theory/sage_t/causal/posterior.py",
    "theory/sage_t/causal/executor.py",
    "theory/sage_t/causal/compiler.py",
    "theory/sage_t/causal/contracts.py",
    "theory/sage_t/causal/mechanisms.py",
    "theory/sage_t/causal/witness_reconfirmation_protocol.py",
    "theory/sage_t/causal/witness_reconfirmation_experiment.py",
    "theory/sage_t/causal/witness_protocol.py",
    "theory/sage_t/causal/witness_experiment.py",
    "theory/sage_t/causal/archive.py",
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
class OptionMinimizationProtocol:
    format_version: str = OPTION_MIN_PROTOCOL_FORMAT
    source_seeds: tuple[int, ...] = (8_701, 8_705)
    candidate_action_count: int = 6
    expected_common_suffix: tuple[str, ...] = (
        "ACTION3",
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    )
    exhaustive_subsequence_count: int = 64
    repetitions_per_candidate_context: int = 3
    minimum_target_confirmations: int = 3
    maximum_off_target_progressions: int = 0
    require_reversed_no_progress: bool = True
    maximum_sdk_calls: int = 24_000
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_parent_particles: int = 8
    minimum_posterior_owner_mass: float = 0.80
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in ("source_seeds", "expected_common_suffix"):
            caster = str if name == "expected_common_suffix" else int
            object.__setattr__(
                self,
                name,
                tuple(caster(value) for value in getattr(self, name)),
            )
        if self.format_version != OPTION_MIN_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.4a.3 option protocol")
        expected = {
            "source_seeds": (8_701, 8_705),
            "candidate_action_count": 6,
            "expected_common_suffix": (
                "ACTION3",
                "ACTION4",
                "ACTION4",
                "ACTION4",
                "ACTION3",
                "ACTION3",
            ),
            "exhaustive_subsequence_count": 64,
            "repetitions_per_candidate_context": 3,
            "minimum_target_confirmations": 3,
            "maximum_off_target_progressions": 0,
            "require_reversed_no_progress": True,
            "maximum_sdk_calls": 24_000,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "maximum_parent_particles": 8,
            "minimum_posterior_owner_mass": 0.80,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.4a.3 preregistered value changed: {name}")
        if self.exhaustive_subsequence_count != 2**self.candidate_action_count:
            raise ValueError("T12.4a.3 must enumerate every subsequence")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_program_registry(path: str | Path, *, game_id: str) -> dict[str, Any]:
    registry = _read_json(path)
    _verify_signed(registry, "registry_checksum")
    if registry.get("format_version") != "sage-t11-causal-program-registry-v1":
        raise ValueError("unsupported causal program registry")
    game = dict(registry.get("games", {})).get(str(game_id))
    if not isinstance(game, Mapping) or not game.get("programs"):
        raise ValueError(f"causal program registry lacks {game_id}")
    return registry


def _source_archives(
    *,
    parent_manifest: Mapping[str, Any],
    witnesses: tuple[Any, ...],
    repo_root: Path,
) -> tuple[dict[str, Any], ...]:
    calibration_manifest_path = _resolve_bound(
        str(parent_manifest["parent"]["manifest"]["path"]),
        root=repo_root,
    )
    calibration_manifest = load_calibration_manifest(
        calibration_manifest_path,
        root=repo_root,
    )
    collection_receipt_path = _resolve_bound(
        str(parent_manifest["parent"]["collection_receipt"]["path"]),
        root=repo_root,
    )
    collection_receipt = load_calibration_receipt(
        collection_receipt_path,
        manifest=calibration_manifest,
        root=repo_root,
        require_passed=True,
    )
    collection_meta = dict(collection_receipt["artifacts"]["collection"])
    collection_path = _resolve_bound(str(collection_meta["path"]), root=repo_root)
    if not collection_path.is_file() or _file_sha256(collection_path) != str(
        collection_meta["sha256"]
    ):
        raise ValueError("T12.4a.3 collection artifact mismatch")
    collection = _read_json(collection_path)
    by_seed = {int(item.source_seed): item for item in witnesses}
    artifacts = []
    for condition in collection.get("conditions", ()):
        seed = int(condition["seed"])
        witness = by_seed.get(seed)
        if witness is None:
            continue
        archive = dict(condition["archive"])
        archive_path = _resolve_bound(str(archive["path"]), root=repo_root)
        if not archive_path.is_file() or _file_sha256(archive_path) != archive["sha256"]:
            raise ValueError("T12.4a.3 source archive mismatch")
        if archive["sha256"] != witness.source_archive_sha256:
            raise ValueError("T12.4a.3 witness/archive checksum mismatch")
        artifacts.append(
            {
                "seed": seed,
                "path": _bound_path(archive_path, root=repo_root),
                "sha256": archive["sha256"],
                "witness_id": witness.witness_id,
            }
        )
    if {item["seed"] for item in artifacts} != set(by_seed):
        raise ValueError("T12.4a.3 source archives are incomplete")
    return tuple(sorted(artifacts, key=lambda item: item["seed"]))


def freeze_option_minimization(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    program_registry_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: OptionMinimizationProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or OptionMinimizationProtocol()
    parent_manifest = load_reconfirmation_manifest(parent_manifest_path, root=repo_root)
    parent_receipt = load_reconfirmation_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
    )
    if not (
        parent_receipt.get("passed") is True
        and parent_receipt.get("phase") == "witness_reconfirmation"
        and parent_receipt.get("status") == "PASS_T12_4A_2_WITNESS_GATE"
    ):
        raise ValueError("T12.4a.3 requires the passed T12.4a.2 witness gate")
    if not parent_manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.3 parent did not authorize scientific claims")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.4a.3 is restricted to source_train")
    registry_meta = dict(parent_manifest["witness_registry"])
    witness_registry_path = _resolve_bound(str(registry_meta["path"]), root=repo_root)
    _, witnesses = load_reconfirmation_registry(
        witness_registry_path,
        protocol=None,
    )
    if tuple(item.source_seed for item in witnesses) != selected.source_seeds:
        raise ValueError("T12.4a.3 source witnesses differ from preregistration")
    suffixes = {
        tuple(step.action.action_name for step in item.steps[-selected.candidate_action_count :])
        for item in witnesses
    }
    if suffixes != {selected.expected_common_suffix}:
        raise ValueError("T12.4a.3 source suffix differs from preregistration")
    source_archives = _source_archives(
        parent_manifest=parent_manifest,
        witnesses=witnesses,
        repo_root=repo_root,
    )
    program_path = Path(program_registry_path).resolve()
    if not program_path.is_file():
        raise FileNotFoundError(program_path)
    program_registry = _load_program_registry(
        program_path,
        game_id=str(parent_manifest["game_id"]),
    )
    missing = [path for path in OPTION_MIN_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.4a.3 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": OPTION_MIN_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_3_OPTION_MINIMIZATION",
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
                "passed": True,
                "status": "PASS_T12_4A_2_WITNESS_GATE",
            },
            "witness_registry": {
                "path": _bound_path(witness_registry_path, root=repo_root),
                "sha256": _file_sha256(witness_registry_path),
                "registry_checksum": registry_meta["registry_checksum"],
            },
        },
        "source_archives": list(source_archives),
        "program_registry": {
            "path": _bound_path(program_path, root=repo_root),
            "sha256": _file_sha256(program_path),
            "registry_checksum": program_registry["registry_checksum"],
            "source_protocol_checksum": program_registry.get("protocol_checksum"),
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in OPTION_MIN_CODE_PATHS
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
            "option_ablation_authorized": authorized,
            "option_compilation_authorized": False,
            "option_active_authority": False,
            "t12_4a_4_transfer_freeze_authorized": False,
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
    receipt = option_minimization_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_4A_3_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "witnesses": len(witnesses),
            "candidate_subsequences": selected.exhaustive_subsequence_count,
            "planned_trials": (
                (selected.exhaustive_subsequence_count + 1)
                * len(witnesses)
                * selected.repetitions_per_candidate_context
            ),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_option_minimization_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != OPTION_MIN_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a.3 manifest")
    protocol = OptionMinimizationProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a.3 protocol checksum mismatch")
    for key in ("manifest", "receipt", "witness_registry"):
        meta = dict(manifest["parent"][key])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.3 parent artifact mismatch: {key}")
    for meta in manifest.get("source_archives", ()):
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.4a.3 source archive mismatch")
    program_meta = dict(manifest["program_registry"])
    program_path = _resolve_bound(str(program_meta["path"]), root=repo_root)
    if not program_path.is_file() or _file_sha256(program_path) != program_meta["sha256"]:
        raise ValueError("T12.4a.3 program registry mismatch")
    program_registry = _load_program_registry(program_path, game_id=manifest["game_id"])
    if program_registry["registry_checksum"] != program_meta["registry_checksum"]:
        raise ValueError("T12.4a.3 program registry checksum binding mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a.3 code checksum mismatch: {relative}")
    return manifest


def option_minimization_receipt(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    passed: bool,
    status: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    parent_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _signed(
        {
            "format_version": OPTION_MIN_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_2_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "parent_receipt_checksum": (
                None if parent_receipt is None else parent_receipt["receipt_checksum"]
            ),
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_option_minimization_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != OPTION_MIN_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a.3 receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.4a.3 receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.4a.3 receipt belongs to another protocol")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.4a.3 upstream gate failed: {receipt.get('status')}")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        candidate = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta.get("sha256"):
            raise ValueError(f"T12.4a.3 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "CONTEXTUAL_OPTION_FORMAT",
    "OptionMinimizationProtocol",
    "freeze_option_minimization",
    "load_option_minimization_manifest",
    "load_option_minimization_receipt",
    "option_minimization_receipt",
]
