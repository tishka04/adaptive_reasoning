"""Frozen SAGE.T12.3b protocol for a multi-step terminal shield."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .archive import ArchiveEdge, GoExploreArchive
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
from .witness_protocol import (
    ProgressWitness,
    load_witness_manifest,
    load_witness_receipt,
    load_witness_registry,
)

SHIELD_PROTOCOL_FORMAT = "sage-t12.3b-terminal-shield-protocol-v1"
SHIELD_REGISTRY_FORMAT = "sage-t12.3b-terminal-trace-registry-v1"
SHIELD_MANIFEST_FORMAT = "sage-t12.3b-terminal-shield-manifest-v1"
SHIELD_RECEIPT_FORMAT = "sage-t12.3b-terminal-shield-receipt-v1"

SHIELD_CODE_PATHS = (
    "theory/sage_t/causal/shield_protocol.py",
    "theory/sage_t/causal/shield_model.py",
    "theory/sage_t/causal/shield_experiment.py",
    "theory/sage_t/causal/shield_experiment_cli.py",
    "theory/sage_t/causal/terminal_shield.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/burst_experiment.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/witness_protocol.py",
    "theory/sage_t/causal/witness_experiment.py",
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


def _edge_from_dict(payload: Mapping[str, Any]) -> ArchiveEdge:
    action = dict(payload["action"])
    return ArchiveEdge(
        edge_id=str(payload["edge_id"]),
        ordinal=int(payload["ordinal"]),
        source_cell_id=str(payload["source_cell_id"]),
        source_exact_hash=str(payload["source_exact_hash"]),
        action=GroundedAction(
            str(action["action_name"]),
            dict(action.get("action_data", {}) or {}),
        ),
        target_cell_id=str(payload["target_cell_id"]),
        target_exact_hash=str(payload["target_exact_hash"]),
        level_delta=int(payload.get("level_delta", 0)),
        terminal=bool(payload.get("terminal", False)),
        success=bool(payload.get("success", False)),
        changed=bool(payload.get("changed", False)),
        novel=bool(payload.get("novel", False)),
        prefix_id=str(payload["prefix_id"]),
    )


@dataclass(frozen=True)
class TerminalTraceCandidate:
    candidate_id: str
    game_id: str
    source_seed: int
    source_arm: str
    source_archive_sha256: str
    terminal_edge_id: str
    initial_exact_hash: str
    initial_level: int
    terminal_source_level: int
    edges: tuple[ArchiveEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))
        if not 1 <= len(self.edges) <= 64:
            raise ValueError("a T12.3b terminal trace needs one to 64 edges")
        if self.edges[0].source_exact_hash != self.initial_exact_hash:
            raise ValueError("terminal trace initial hash mismatch")
        for previous, following in zip(self.edges, self.edges[1:]):
            if previous.target_exact_hash != following.source_exact_hash:
                raise ValueError("terminal trace exact hashes are not contiguous")
            if previous.target_cell_id != following.source_cell_id:
                raise ValueError("terminal trace symbolic cells are not contiguous")
        if not self.edges[-1].terminal or self.edges[-1].success:
            raise ValueError("terminal trace must end in a failing terminal edge")
        if self.edges[-1].edge_id != self.terminal_edge_id:
            raise ValueError("terminal trace edge identifier mismatch")

    @property
    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "source_seed": self.source_seed,
            "source_arm": self.source_arm,
            "source_archive_sha256": self.source_archive_sha256,
            "terminal_edge_id": self.terminal_edge_id,
            "initial_exact_hash": self.initial_exact_hash,
            "initial_level": self.initial_level,
            "terminal_source_level": self.terminal_source_level,
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @property
    def trace_checksum(self) -> str:
        return _checksum(self.unsigned_payload)

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_payload, "trace_checksum": self.trace_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TerminalTraceCandidate:
        expected = str(payload.get("trace_checksum", ""))
        candidate = cls(
            candidate_id=str(payload["candidate_id"]),
            game_id=str(payload["game_id"]),
            source_seed=int(payload["source_seed"]),
            source_arm=str(payload["source_arm"]),
            source_archive_sha256=str(payload["source_archive_sha256"]),
            terminal_edge_id=str(payload["terminal_edge_id"]),
            initial_exact_hash=str(payload["initial_exact_hash"]),
            initial_level=int(payload["initial_level"]),
            terminal_source_level=int(payload["terminal_source_level"]),
            edges=tuple(_edge_from_dict(dict(item)) for item in payload["edges"]),
        )
        if not expected or candidate.trace_checksum != expected:
            raise ValueError("terminal trace checksum mismatch")
        return candidate


@dataclass(frozen=True)
class ProtectedActionSpec:
    cell_id: str
    action_key: str
    witness_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "witness_ids",
            tuple(sorted({str(value) for value in self.witness_ids})),
        )
        if not self.cell_id or not self.action_key or not self.witness_ids:
            raise ValueError("protected action specification is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProtectedActionSpec:
        return cls(
            cell_id=str(payload["cell_id"]),
            action_key=str(payload["action_key"]),
            witness_ids=tuple(str(value) for value in payload["witness_ids"]),
        )


@dataclass(frozen=True)
class TerminalShieldProtocol:
    format_version: str = SHIELD_PROTOCOL_FORMAT
    terminal_horizon: int = 64
    terminal_minimum_support: int = 2
    candidates_per_source_archive: int = 2
    terminal_confirmation_repetitions: int = 1
    minimum_confirmed_terminal_traces: int = 8
    minimum_terminal_confirmation_rate: float = 2.0 / 3.0
    witness_repetitions: int = 3
    evaluation_seeds: tuple[int, ...] = (6801, 6802, 6803)
    evaluation_arms: tuple[str, ...] = (
        "burst_control",
        "burst_terminal_shield",
    )
    source_seeds: tuple[int, ...] = (6501, 6502, 6503)
    source_arms: tuple[str, ...] = (
        "one_step_archive",
        "burst_archive",
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
    minimum_vetoes: int = 1
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in ("evaluation_seeds", "source_seeds", "burst_schedule"):
            object.__setattr__(
                self,
                name,
                tuple(int(value) for value in getattr(self, name)),
            )
        for name in ("evaluation_arms", "source_arms"):
            object.__setattr__(
                self,
                name,
                tuple(str(value) for value in getattr(self, name)),
            )
        if self.format_version != SHIELD_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.3b terminal-shield protocol")
        fixed = {
            "terminal_horizon": (self.terminal_horizon, 64),
            "terminal_minimum_support": (self.terminal_minimum_support, 2),
            "candidates_per_source_archive": (
                self.candidates_per_source_archive,
                2,
            ),
            "terminal_confirmation_repetitions": (
                self.terminal_confirmation_repetitions,
                1,
            ),
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
            "minimum_confirmed_terminal_traces": (
                self.minimum_confirmed_terminal_traces,
                8,
            ),
            "minimum_terminal_confirmation_rate": (
                self.minimum_terminal_confirmation_rate,
                2.0 / 3.0,
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
            "minimum_vetoes": (self.minimum_vetoes, 1),
        }
        for name, (observed, expected) in fixed.items():
            if observed != expected:
                raise ValueError(f"T12.3b preregistered value changed: {name}")
        if self.source_seeds != (6501, 6502, 6503):
            raise ValueError("T12.3b source seeds are frozen")
        if self.evaluation_seeds != (6801, 6802, 6803):
            raise ValueError("T12.3b evaluation seeds are frozen")
        if self.source_arms != ("one_step_archive", "burst_archive"):
            raise ValueError("T12.3b source arms are frozen")
        if self.evaluation_arms != (
            "burst_control",
            "burst_terminal_shield",
        ):
            raise ValueError("T12.3b evaluation arms are frozen")
        if self.burst_schedule != (4, 8, 16):
            raise ValueError("T12.3b burst schedule is frozen")
        maximum_candidates = (
            len(self.source_seeds)
            * len(self.source_arms)
            * self.candidates_per_source_archive
        )
        planned_upper_bound = (
            maximum_candidates
            * self.terminal_confirmation_repetitions
            * (1 + self.terminal_horizon)
            + 2 * self.witness_repetitions * (1 + 128)
            + len(self.evaluation_seeds)
            * len(self.evaluation_arms)
            * self.sdk_calls_per_evaluation_arm
        )
        if planned_upper_bound > self.maximum_total_sdk_calls:
            raise ValueError("T12.3b planned calls exceed the global SDK budget")

    @property
    def expected_terminal_candidates(self) -> int:
        return (
            len(self.source_seeds)
            * len(self.source_arms)
            * self.candidates_per_source_archive
        )

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _load_archive(meta: Mapping[str, Any]) -> GoExploreArchive:
    path = Path(str(meta["path"]))
    if not path.is_file() or _file_sha256(path) != str(meta["sha256"]):
        raise ValueError(f"T12.2 archive checksum mismatch: {path}")
    return GoExploreArchive.from_dict(_read_json(path))


def extract_terminal_candidates(
    archive_artifacts: Sequence[Mapping[str, Any]],
    *,
    protocol: TerminalShieldProtocol,
) -> tuple[TerminalTraceCandidate, ...]:
    output = []
    for raw_meta in archive_artifacts:
        meta = dict(raw_meta)
        seed = int(meta.get("seed", -1))
        arm = str(meta.get("arm", ""))
        if seed not in protocol.source_seeds or arm not in protocol.source_arms:
            continue
        archive = _load_archive(meta)
        eligible = []
        for edge in archive.edges.values():
            if not edge.terminal or edge.success:
                continue
            variant = archive.cells[edge.target_cell_id].variants[
                edge.target_exact_hash
            ]
            path = archive.path_edges(variant)
            if not path or len(path) > protocol.terminal_horizon:
                continue
            eligible.append((len(path), edge.ordinal, edge.edge_id, path))
        selected = []
        seen_terminal_pairs = set()
        for _, _, _, path in sorted(eligible):
            terminal = path[-1]
            pair = (terminal.source_cell_id, terminal.action.key)
            if pair in seen_terminal_pairs:
                continue
            seen_terminal_pairs.add(pair)
            selected.append(path)
            if len(selected) >= protocol.candidates_per_source_archive:
                break
        for path in selected:
            terminal = path[-1]
            token = _checksum(
                {
                    "seed": seed,
                    "arm": arm,
                    "archive": meta["sha256"],
                    "terminal_edge": terminal.edge_id,
                    "route": [edge.edge_id for edge in path],
                }
            )[:20]
            output.append(
                TerminalTraceCandidate(
                    candidate_id=f"terminal_{token}",
                    game_id=str(meta["game_id"]),
                    source_seed=seed,
                    source_arm=arm,
                    source_archive_sha256=str(meta["sha256"]),
                    terminal_edge_id=terminal.edge_id,
                    initial_exact_hash=path[0].source_exact_hash,
                    initial_level=archive.cells[path[0].source_cell_id].level,
                    terminal_source_level=archive.cells[terminal.source_cell_id].level,
                    edges=tuple(path),
                )
            )
    return tuple(
        sorted(
            output,
            key=lambda item: (item.source_seed, item.source_arm, item.candidate_id),
        )
    )


def extract_protected_actions(
    witnesses: Sequence[ProgressWitness],
    archive_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[ProtectedActionSpec, ...]:
    by_sha = {
        str(meta["sha256"]): dict(meta)
        for meta in archive_artifacts
        if str(meta.get("arm", "")) in {"one_step_archive", "burst_archive"}
    }
    protected: dict[tuple[str, str], set[str]] = defaultdict(set)
    for witness in witnesses:
        meta = by_sha.get(witness.source_archive_sha256)
        if meta is None:
            raise ValueError(
                f"missing source archive for witness: {witness.witness_id}"
            )
        archive = _load_archive(meta)
        edge = archive.edges.get(witness.source_progress_edge_id)
        if edge is None:
            raise ValueError(f"missing witness progress edge: {witness.witness_id}")
        variant = archive.cells[edge.target_cell_id].variants[edge.target_exact_hash]
        path = archive.path_edges(variant)
        if len(path) != len(witness.steps):
            raise ValueError("witness/archive route length mismatch")
        for archived, sealed in zip(path, witness.steps):
            if (
                archived.source_exact_hash != sealed.expected_source_hash
                or archived.target_exact_hash != sealed.expected_target_hash
                or archived.action.key != sealed.action.key
            ):
                raise ValueError("witness/archive transition mismatch")
            protected[(archived.source_cell_id, archived.action.key)].add(
                witness.witness_id
            )
    return tuple(
        ProtectedActionSpec(cell_id, action_key, tuple(witness_ids))
        for (cell_id, action_key), witness_ids in sorted(protected.items())
    )


def _registry_payload(
    *,
    candidates: Sequence[TerminalTraceCandidate],
    protected_actions: Sequence[ProtectedActionSpec],
    protocol: TerminalShieldProtocol,
    parent_receipt_checksum: str,
    source_receipt_checksum: str,
) -> dict[str, Any]:
    return {
        "format_version": SHIELD_REGISTRY_FORMAT,
        "protocol_checksum": protocol.checksum,
        "parent_t12_3a_receipt_checksum": parent_receipt_checksum,
        "source_t12_2_receipt_checksum": source_receipt_checksum,
        "terminal_candidates": [item.to_dict() for item in candidates],
        "protected_actions": [item.to_dict() for item in protected_actions],
    }


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def freeze_shield_experiment(
    *,
    output_path: str | Path,
    terminal_registry_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: TerminalShieldProtocol | None = None,
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    selected = protocol or TerminalShieldProtocol()
    parent_manifest = load_witness_manifest(parent_manifest_path, root=repo_root)
    parent_receipt = load_witness_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
    )
    if parent_receipt.get("phase") != "witness_confirmation":
        raise ValueError("T12.3b parent must be the T12.3a confirmation receipt")
    if parent_receipt.get("passed") is not True:
        raise ValueError("T12.3b requires a passed T12.3a witness gate")
    if parent_receipt.get("status") != "PASS_T12_3A_WITNESS_GATE":
        raise ValueError("unexpected T12.3a parent status")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.3b is restricted to source_train")

    witness_registry_path = _resolve_bound(
        str(parent_manifest["witness_registry"]["path"]), root=repo_root
    )
    _, witnesses = load_witness_registry(witness_registry_path)
    t12_2_manifest_path = _resolve_bound(
        str(parent_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    t12_2_receipt_path = _resolve_bound(
        str(parent_manifest["parent"]["receipt"]["path"]), root=repo_root
    )
    t12_2_manifest = load_burst_manifest(t12_2_manifest_path, root=repo_root)
    t12_2_receipt = load_burst_receipt(t12_2_receipt_path, manifest=t12_2_manifest)
    archive_artifacts = tuple(t12_2_receipt.get("artifacts", {}).get("archives", ()))
    candidates = extract_terminal_candidates(
        archive_artifacts,
        protocol=selected,
    )
    if len(candidates) != selected.expected_terminal_candidates:
        raise ValueError("T12.3b terminal registry lacks its balanced 12 candidates")
    group_counts: dict[tuple[int, str], int] = defaultdict(int)
    for candidate in candidates:
        group_counts[(candidate.source_seed, candidate.source_arm)] += 1
    expected_groups = {
        (seed, arm): selected.candidates_per_source_archive
        for seed in selected.source_seeds
        for arm in selected.source_arms
    }
    if dict(group_counts) != expected_groups:
        raise ValueError("T12.3b terminal candidates are not source-balanced")
    protected_actions = extract_protected_actions(witnesses, archive_artifacts)
    if not protected_actions:
        raise ValueError("T12.3b requires protected progress actions")
    protected_witnesses = {
        witness_id for item in protected_actions for witness_id in item.witness_ids
    }
    if protected_witnesses != {item.witness_id for item in witnesses}:
        raise ValueError("T12.3b did not protect both progress witnesses")

    missing = [path for path in SHIELD_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.3b code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    scientific_claims_authorized = bool(
        not git["dirty"]
        and parent_manifest.get("scientific_claims_authorized", False)
        and parent_receipt.get("passed") is True
    )
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")

    registry = _signed(
        _registry_payload(
            candidates=candidates,
            protected_actions=protected_actions,
            protocol=selected,
            parent_receipt_checksum=parent_receipt["receipt_checksum"],
            source_receipt_checksum=t12_2_receipt["receipt_checksum"],
        ),
        "registry_checksum",
    )
    _write_json_once(terminal_registry_path, registry)
    payload = {
        "format_version": SHIELD_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_3B_TERMINAL_SHIELD",
        "stage": "source_train",
        "game_id": parent_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "terminal_registry": {
            "path": _bound_path(terminal_registry_path, root=repo_root),
            "sha256": _file_sha256(terminal_registry_path),
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
                "status": "PASS_T12_3A_WITNESS_GATE",
            },
        },
        "source_t12_2": {
            "manifest": {
                "path": _bound_path(t12_2_manifest_path, root=repo_root),
                "sha256": _file_sha256(t12_2_manifest_path),
                "manifest_checksum": t12_2_manifest["manifest_checksum"],
            },
            "receipt": {
                "path": _bound_path(t12_2_receipt_path, root=repo_root),
                "sha256": _file_sha256(t12_2_receipt_path),
                "receipt_checksum": t12_2_receipt["receipt_checksum"],
            },
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in SHIELD_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": scientific_claims_authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_experiment_authorized": scientific_claims_authorized,
            "terminal_shield_production_authority": False,
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
    freeze_receipt = shield_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=scientific_claims_authorized,
        status=(
            "PASS_T12_3B_FREEZE" if scientific_claims_authorized else "DIRTY_SMOKE_ONLY"
        ),
        metrics={
            "terminal_candidates": len(candidates),
            "protected_actions": len(protected_actions),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), freeze_receipt)
    return manifest


def load_shield_registry(
    path: str | Path,
    *,
    protocol: TerminalShieldProtocol | None = None,
) -> tuple[
    dict[str, Any],
    tuple[TerminalTraceCandidate, ...],
    tuple[ProtectedActionSpec, ...],
]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != SHIELD_REGISTRY_FORMAT:
        raise ValueError("unsupported T12.3b terminal registry")
    selected = protocol or TerminalShieldProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("T12.3b terminal registry protocol mismatch")
    candidates = tuple(
        TerminalTraceCandidate.from_dict(dict(item))
        for item in payload.get("terminal_candidates", ())
    )
    protected = tuple(
        ProtectedActionSpec.from_dict(dict(item))
        for item in payload.get("protected_actions", ())
    )
    if len(candidates) != selected.expected_terminal_candidates or not protected:
        raise ValueError("T12.3b terminal registry is incomplete")
    return payload, candidates, protected


def load_shield_manifest(
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
    if manifest.get("format_version") != SHIELD_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.3b terminal-shield manifest")
    protocol = TerminalShieldProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.3b protocol checksum mismatch")
    registry_meta = dict(manifest["terminal_registry"])
    registry_path = _resolve_bound(str(registry_meta["path"]), root=repo_root)
    if (
        not registry_path.is_file()
        or _file_sha256(registry_path) != registry_meta["sha256"]
    ):
        raise ValueError("T12.3b terminal registry checksum mismatch")
    registry, _, _ = load_shield_registry(registry_path, protocol=protocol)
    if registry["registry_checksum"] != registry_meta["registry_checksum"]:
        raise ValueError("T12.3b terminal registry signature mismatch")
    for parent_key in ("parent", "source_t12_2"):
        for artifact_key in ("manifest", "receipt"):
            meta = dict(manifest[parent_key][artifact_key])
            candidate = _resolve_bound(str(meta["path"]), root=repo_root)
            if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
                raise ValueError(
                    f"T12.3b {parent_key} {artifact_key} checksum mismatch"
                )
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.3b code checksum mismatch: {relative}")
    return manifest


def shield_phase_receipt(
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
            "format_version": SHIELD_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_3a_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_shield_receipt(
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
    if receipt.get("format_version") != SHIELD_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.3b receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.3b receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.3b receipt belongs to another protocol")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        artifact_path = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not artifact_path.is_file() or _file_sha256(artifact_path) != meta.get(
            "sha256"
        ):
            raise ValueError(f"T12.3b receipt artifact checksum mismatch: {name}")
        if name == "paired_evaluation":
            evaluation = _read_json(artifact_path)
            for condition in evaluation.get("conditions", ()):
                for arm_name, arm in dict(condition.get("arms", {})).items():
                    for artifact_name in ("archive", "excursions"):
                        nested = dict(arm[artifact_name])
                        nested_path = _resolve_bound(
                            str(nested["path"]), root=repo_root
                        )
                        if (
                            not nested_path.is_file()
                            or _file_sha256(nested_path) != nested["sha256"]
                        ):
                            raise ValueError(
                                "T12.3b paired artifact checksum mismatch: "
                                f"{arm_name}:{artifact_name}"
                            )
    return receipt


__all__ = [
    "ProtectedActionSpec",
    "TerminalShieldProtocol",
    "TerminalTraceCandidate",
    "extract_protected_actions",
    "extract_terminal_candidates",
    "freeze_shield_experiment",
    "load_shield_manifest",
    "load_shield_receipt",
    "load_shield_registry",
    "shield_phase_receipt",
]
