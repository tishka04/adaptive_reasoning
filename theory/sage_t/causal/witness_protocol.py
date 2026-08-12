"""Frozen T12.3a protocol for exact confirmation of progress witnesses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .archive import GoExploreArchive
from .burst_protocol import load_burst_manifest, load_burst_receipt
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

WITNESS_PROTOCOL_FORMAT = "sage-t12.3a-progress-witness-protocol-v1"
WITNESS_REGISTRY_FORMAT = "sage-t12.3a-progress-witness-registry-v1"
WITNESS_MANIFEST_FORMAT = "sage-t12.3a-progress-witness-manifest-v1"
WITNESS_RECEIPT_FORMAT = "sage-t12.3a-progress-witness-receipt-v1"

WITNESS_CODE_PATHS = (
    "theory/sage_t/causal/witness_protocol.py",
    "theory/sage_t/causal/witness_experiment.py",
    "theory/sage_t/causal/witness_experiment_cli.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/burst_protocol.py",
    "theory/sage_t/causal/burst_experiment.py",
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
class WitnessStep:
    expected_source_hash: str
    action: GroundedAction
    expected_target_hash: str
    level_delta: int = 0
    terminal: bool = False
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_source_hash": self.expected_source_hash,
            "action": {
                "action_name": self.action.action_name,
                "action_data": dict(self.action.action_data),
            },
            "expected_target_hash": self.expected_target_hash,
            "level_delta": self.level_delta,
            "terminal": self.terminal,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WitnessStep:
        action = dict(payload["action"])
        return cls(
            expected_source_hash=str(payload["expected_source_hash"]),
            action=GroundedAction(
                str(action["action_name"]),
                dict(action.get("action_data", {}) or {}),
            ),
            expected_target_hash=str(payload["expected_target_hash"]),
            level_delta=int(payload.get("level_delta", 0)),
            terminal=bool(payload.get("terminal", False)),
            success=bool(payload.get("success", False)),
        )


@dataclass(frozen=True)
class ProgressWitness:
    witness_id: str
    game_id: str
    source_seed: int
    source_arm: str
    source_archive_sha256: str
    source_progress_edge_id: str
    initial_exact_hash: str
    initial_level: int
    target_exact_hash: str
    target_level: int
    steps: tuple[WitnessStep, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if not 1 <= len(self.steps) <= 128:
            raise ValueError("a progress witness needs one to 128 steps")
        if self.steps[0].expected_source_hash != self.initial_exact_hash:
            raise ValueError("witness initial hash does not match its first transition")
        if self.steps[-1].expected_target_hash != self.target_exact_hash:
            raise ValueError("witness target hash does not match its final transition")
        for previous, following in zip(self.steps, self.steps[1:]):
            if previous.expected_target_hash != following.expected_source_hash:
                raise ValueError("witness transition hashes are not contiguous")
        if self.target_level <= self.initial_level:
            raise ValueError("witness must increase the completed-level counter")

    @property
    def route_checksum(self) -> str:
        return _checksum(self.unsigned_payload)

    @property
    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "game_id": self.game_id,
            "source_seed": self.source_seed,
            "source_arm": self.source_arm,
            "source_archive_sha256": self.source_archive_sha256,
            "source_progress_edge_id": self.source_progress_edge_id,
            "initial_exact_hash": self.initial_exact_hash,
            "initial_level": self.initial_level,
            "target_exact_hash": self.target_exact_hash,
            "target_level": self.target_level,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_payload, "route_checksum": self.route_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProgressWitness:
        unsigned = dict(payload)
        expected = str(unsigned.pop("route_checksum", ""))
        witness = cls(
            witness_id=str(payload["witness_id"]),
            game_id=str(payload["game_id"]),
            source_seed=int(payload["source_seed"]),
            source_arm=str(payload["source_arm"]),
            source_archive_sha256=str(payload["source_archive_sha256"]),
            source_progress_edge_id=str(payload["source_progress_edge_id"]),
            initial_exact_hash=str(payload["initial_exact_hash"]),
            initial_level=int(payload["initial_level"]),
            target_exact_hash=str(payload["target_exact_hash"]),
            target_level=int(payload["target_level"]),
            steps=tuple(
                WitnessStep.from_dict(dict(item)) for item in payload.get("steps", ())
            ),
        )
        if not expected or witness.route_checksum != expected:
            raise ValueError("progress witness route checksum mismatch")
        return witness


@dataclass(frozen=True)
class WitnessConfirmProtocol:
    format_version: str = WITNESS_PROTOCOL_FORMAT
    repetitions_per_route: int = 3
    repetitions_per_suffix_branch: int = 3
    minimum_successful_route_replays: int = 2
    minimum_successful_suffix_replays: int = 2
    minimum_paired_contrasts: int = 2
    maximum_sdk_calls: int = 2_048
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_witness_steps: int = 128
    expected_common_suffix: tuple[str, ...] = ("ACTION3", "ACTION3", "ACTION3")
    minimum_step_exact_rate: float = 0.99
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_common_suffix",
            tuple(str(value) for value in self.expected_common_suffix),
        )
        if self.format_version != WITNESS_PROTOCOL_FORMAT:
            raise ValueError("unsupported progress-witness protocol")
        if self.repetitions_per_route != 3 or self.repetitions_per_suffix_branch != 3:
            raise ValueError("T12.3a preregisters three route and branch repetitions")
        if self.minimum_successful_route_replays != 2:
            raise ValueError("T12.3a route confirmation threshold is two of three")
        if self.minimum_successful_suffix_replays != 2:
            raise ValueError("T12.3a suffix confirmation threshold is two of three")
        if self.minimum_paired_contrasts != 2:
            raise ValueError("T12.3a paired contrast threshold is two of three")
        if self.maximum_sdk_calls != 2_048:
            raise ValueError("T12.3a preregisters exactly 2,048 SDK calls")
        if self.maximum_witness_steps != 128:
            raise ValueError("T12.3a preregisters a 128-step witness ceiling")
        if self.minimum_step_exact_rate != 0.99:
            raise ValueError("T12.3a preregisters a 0.99 exact-step threshold")
        if self.expected_common_suffix != ("ACTION3", "ACTION3", "ACTION3"):
            raise ValueError("T12.3a seals the observed ACTION3 x3 common suffix")
        if self.maximum_artifact_bytes_per_run != 3 * 1024 * 1024 * 1024:
            raise ValueError("T12.3a preregisters an exact 3 GiB artifact cap")
        if self.maximum_sdk_calls <= 0:
            raise ValueError("T12.3a needs a positive SDK call budget")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _load_archive(meta: Mapping[str, Any]) -> GoExploreArchive:
    path = Path(str(meta["path"]))
    if not path.is_file() or _file_sha256(path) != str(meta["sha256"]):
        raise ValueError(f"T12.2 archive checksum mismatch: {path}")
    return GoExploreArchive.from_dict(_read_json(path))


def extract_progress_witnesses(
    archive_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[ProgressWitness, ...]:
    witnesses = []
    seen_routes = set()
    for raw_meta in archive_artifacts:
        meta = dict(raw_meta)
        arm = str(meta.get("arm", ""))
        if arm not in {"one_step_archive", "burst_archive"}:
            continue
        archive = _load_archive(meta)
        for edge in sorted(archive.edges.values(), key=lambda item: item.ordinal):
            if not (edge.level_delta > 0 or edge.success):
                continue
            target_cell = archive.cells[edge.target_cell_id]
            target_variant = target_cell.variants[edge.target_exact_hash]
            path = archive.path_edges(target_variant)
            if not path:
                continue
            route_key = tuple(item.action.key for item in path)
            if route_key in seen_routes:
                continue
            seen_routes.add(route_key)
            token = _checksum(
                {
                    "seed": meta["seed"],
                    "arm": arm,
                    "edge": edge.edge_id,
                    "route": route_key,
                }
            )[:20]
            witnesses.append(
                ProgressWitness(
                    witness_id=f"witness_{token}",
                    game_id=str(meta["game_id"]),
                    source_seed=int(meta["seed"]),
                    source_arm=arm,
                    source_archive_sha256=str(meta["sha256"]),
                    source_progress_edge_id=edge.edge_id,
                    initial_exact_hash=path[0].source_exact_hash,
                    initial_level=archive.cells[path[0].source_cell_id].level,
                    target_exact_hash=path[-1].target_exact_hash,
                    target_level=target_cell.level,
                    steps=tuple(
                        WitnessStep(
                            expected_source_hash=item.source_exact_hash,
                            action=item.action,
                            expected_target_hash=item.target_exact_hash,
                            level_delta=item.level_delta,
                            terminal=item.terminal,
                            success=item.success,
                        )
                        for item in path
                    ),
                )
            )
    return tuple(
        sorted(
            witnesses,
            key=lambda item: (item.source_seed, item.source_arm, len(item.steps)),
        )
    )


def common_action_suffix(
    witnesses: Sequence[ProgressWitness],
) -> tuple[GroundedAction, ...]:
    if not witnesses:
        return ()
    reversed_actions = [
        tuple(reversed(tuple(step.action for step in witness.steps)))
        for witness in witnesses
    ]
    length = 0
    for candidates in zip(*reversed_actions):
        if len({item.key for item in candidates}) != 1:
            break
        length += 1
    return tuple(step.action for step in witnesses[0].steps[-length:]) if length else ()


def _registry_payload(
    *,
    witnesses: Sequence[ProgressWitness],
    protocol: WitnessConfirmProtocol,
    parent_receipt_checksum: str,
) -> dict[str, Any]:
    common_suffix = common_action_suffix(witnesses)
    return {
        "format_version": WITNESS_REGISTRY_FORMAT,
        "protocol_checksum": protocol.checksum,
        "parent_t12_2_receipt_checksum": parent_receipt_checksum,
        "witnesses": [item.to_dict() for item in witnesses],
        "common_initial_exact_hash": witnesses[0].initial_exact_hash,
        "common_target_exact_hash": witnesses[0].target_exact_hash,
        "common_target_level": witnesses[0].target_level,
        "common_suffix": [
            {
                "action_name": action.action_name,
                "action_data": dict(action.action_data),
            }
            for action in common_suffix
        ],
    }


def freeze_witness_experiment(
    *,
    output_path: str | Path,
    witness_registry_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: WitnessConfirmProtocol | None = None,
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    selected = protocol or WitnessConfirmProtocol()
    parent_manifest = load_burst_manifest(parent_manifest_path, root=repo_root)
    parent_receipt = load_burst_receipt(parent_receipt_path, manifest=parent_manifest)
    if parent_receipt.get("phase") != "burst_archive":
        raise ValueError("T12.3a parent must be the T12.2 burst receipt")
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.3a is a correction to a failed T12.2 gate")
    if parent_receipt.get("status") != "FAIL_T12_2_BURST_GATE":
        raise ValueError("unexpected T12.2 parent status")
    if int(parent_receipt.get("metrics", {}).get("burst_progress_edges", 0)) < 1:
        raise ValueError("T12.3a needs an observed T12.2 progress witness")
    witnesses = extract_progress_witnesses(
        tuple(parent_receipt.get("artifacts", {}).get("archives", ()))
    )
    if len(witnesses) != 2:
        raise ValueError("T12.3a requires exactly two distinct progress witnesses")
    if {item.source_arm for item in witnesses} != {
        "one_step_archive",
        "burst_archive",
    }:
        raise ValueError("T12.3a requires one one-step and one burst witness")
    if len({item.game_id for item in witnesses}) != 1:
        raise ValueError("T12.3a witnesses must belong to one game")
    if len({item.initial_exact_hash for item in witnesses}) != 1:
        raise ValueError("T12.3a witnesses need the same exact initial state")
    if len({item.target_exact_hash for item in witnesses}) != 1:
        raise ValueError("T12.3a witnesses need the same exact target state")
    if any(len(item.steps) > selected.maximum_witness_steps for item in witnesses):
        raise ValueError("T12.3a witness exceeds the sealed maximum route length")
    suffix = common_action_suffix(witnesses)
    if (
        tuple(action.action_name for action in suffix)
        != selected.expected_common_suffix
    ):
        raise ValueError("observed witness suffix differs from ACTION3 x3")
    missing = [path for path in WITNESS_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.3a code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    scientific_claims_authorized = bool(
        not git["dirty"] and parent_manifest.get("scientific_claims_authorized", False)
    )
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    registry = _signed(
        _registry_payload(
            witnesses=witnesses,
            protocol=selected,
            parent_receipt_checksum=parent_receipt["receipt_checksum"],
        ),
        "registry_checksum",
    )
    _write_json_once(witness_registry_path, registry)
    payload = {
        "format_version": WITNESS_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_3A_WITNESS_CONFIRMATION",
        "stage": "source_train",
        "game_id": witnesses[0].game_id,
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "witness_registry": {
            "path": _bound_path(witness_registry_path, root=repo_root),
            "sha256": _file_sha256(witness_registry_path),
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
                "status": "FAIL_T12_2_BURST_GATE",
            },
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in WITNESS_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": scientific_claims_authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_authorized": False,
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
    receipt = witness_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=scientific_claims_authorized,
        status=(
            "PASS_T12_3A_FREEZE" if scientific_claims_authorized else "DIRTY_SMOKE_ONLY"
        ),
        metrics={"witnesses": len(witnesses)},
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_witness_registry(
    path: str | Path,
    *,
    protocol: WitnessConfirmProtocol | None = None,
) -> tuple[dict[str, Any], tuple[ProgressWitness, ...]]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != WITNESS_REGISTRY_FORMAT:
        raise ValueError("unsupported progress-witness registry")
    selected = protocol or WitnessConfirmProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("progress-witness registry protocol mismatch")
    witnesses = tuple(
        ProgressWitness.from_dict(dict(item)) for item in payload.get("witnesses", ())
    )
    if len(witnesses) != 2:
        raise ValueError("progress-witness registry is incomplete")
    if len({item.initial_exact_hash for item in witnesses}) != 1:
        raise ValueError("progress-witness registry initial hashes differ")
    if len({item.target_exact_hash for item in witnesses}) != 1:
        raise ValueError("progress-witness registry target hashes differ")
    if payload.get("common_initial_exact_hash") != witnesses[0].initial_exact_hash:
        raise ValueError("progress-witness registry initial hash mismatch")
    if payload.get("common_target_exact_hash") != witnesses[0].target_exact_hash:
        raise ValueError("progress-witness registry target hash mismatch")
    suffix_payload = tuple(
        GroundedAction(
            str(dict(item)["action_name"]),
            dict(dict(item).get("action_data", {}) or {}),
        )
        for item in payload.get("common_suffix", ())
    )
    if tuple(action.key for action in suffix_payload) != tuple(
        action.key for action in common_action_suffix(witnesses)
    ):
        raise ValueError("progress-witness registry common suffix mismatch")
    return payload, witnesses


def load_witness_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != WITNESS_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.3a manifest")
    protocol = WitnessConfirmProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.3a protocol checksum mismatch")
    registry_meta = dict(manifest["witness_registry"])
    registry_path = Path(registry_meta["path"])
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    if (
        not registry_path.is_file()
        or _file_sha256(registry_path) != registry_meta["sha256"]
    ):
        raise ValueError("T12.3a witness registry checksum mismatch")
    registry, _ = load_witness_registry(registry_path, protocol=protocol)
    if registry["registry_checksum"] != registry_meta["registry_checksum"]:
        raise ValueError("T12.3a witness registry signature mismatch")
    for key in ("manifest", "receipt"):
        meta = dict(manifest["parent"][key])
        parent_path = Path(meta["path"])
        if not parent_path.is_absolute():
            parent_path = repo_root / parent_path
        if not parent_path.is_file() or _file_sha256(parent_path) != meta["sha256"]:
            raise ValueError(f"T12.2 parent {key} checksum mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.3a code checksum mismatch: {relative}")
    return manifest


def witness_phase_receipt(
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
            "format_version": WITNESS_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_2_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_witness_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != WITNESS_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.3a receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.3a receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.3a receipt belongs to another protocol")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        artifact_path = Path(str(meta.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = repo_root / artifact_path
        if not artifact_path.is_file() or _file_sha256(artifact_path) != meta.get(
            "sha256"
        ):
            raise ValueError(f"T12.3a receipt artifact checksum mismatch: {name}")
    return receipt


__all__ = [
    "ProgressWitness",
    "WitnessConfirmProtocol",
    "WitnessStep",
    "common_action_suffix",
    "extract_progress_witnesses",
    "freeze_witness_experiment",
    "load_witness_manifest",
    "load_witness_receipt",
    "load_witness_registry",
    "witness_phase_receipt",
]
