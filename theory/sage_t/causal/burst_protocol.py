"""Frozen SAGE.T12.2 protocol for burst-style symbolic Go-Explore."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS, short_game_id

from .experiment import (
    _bound_path,
    _file_sha256,
    _git_state,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .graph_protocol import load_graph_manifest, load_graph_receipt

BURST_PROTOCOL_FORMAT = "sage-t12.2-burst-go-explore-protocol-v1"
BURST_MANIFEST_FORMAT = "sage-t12.2-burst-go-explore-manifest-v1"
BURST_RECEIPT_FORMAT = "sage-t12.2-burst-go-explore-receipt-v1"

BURST_CODE_PATHS = (
    "theory/sage_t/causal/burst_protocol.py",
    "theory/sage_t/causal/burst_experiment.py",
    "theory/sage_t/causal/burst_experiment_cli.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/graph_protocol.py",
    "theory/sage/live_prefix_counterfactual_collector.py",
)


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BurstExploreProtocol:
    format_version: str = BURST_PROTOCOL_FORMAT
    sdk_call_budget_per_seed_arm: int = 8_192
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_cells: int = 50_000
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    seeds: tuple[int, ...] = (6501, 6502, 6503)
    arms: tuple[str, ...] = ("one_step_archive", "burst_archive")
    minimum_exploration_action_ratio: float = 2.0
    minimum_relative_coverage_gain: float = 0.25
    maximum_terminal_failure_rate: float = 0.10
    minimum_progress_edges: int = 1
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "burst_schedule", tuple(int(value) for value in self.burst_schedule)
        )
        object.__setattr__(self, "seeds", tuple(int(value) for value in self.seeds))
        object.__setattr__(self, "arms", tuple(str(value) for value in self.arms))
        if self.format_version != BURST_PROTOCOL_FORMAT:
            raise ValueError("unsupported burst Go-Explore protocol")
        if int(self.sdk_call_budget_per_seed_arm) <= 0:
            raise ValueError("SDK call budget must be positive")
        if int(self.maximum_artifact_bytes_per_run) != 3 * 1024 * 1024 * 1024:
            raise ValueError("T12.2 preregisters an exact 3 GiB artifact cap")
        if self.burst_schedule != (4, 8, 16):
            raise ValueError("T12.2 preregisters the 4/8/16 burst schedule")
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("T12.2 needs three distinct paired seeds")
        if self.arms != ("one_step_archive", "burst_archive"):
            raise ValueError("T12.2 arms are frozen")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def freeze_burst_experiment(
    *,
    output_path: str | Path,
    game_id: str,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: BurstExploreProtocol | None = None,
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    selected = protocol or BurstExploreProtocol()
    game = short_game_id(game_id)
    if SAGE11_SPLITS.split_for(game) != "source_train":
        raise ValueError("T12.2 fitting is restricted to source_train")
    parent_manifest = load_graph_manifest(parent_manifest_path, root=repo_root)
    parent_receipt = load_graph_receipt(
        parent_receipt_path, manifest=parent_manifest, require_passed=False
    )
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.2 parent must be a source_train experiment")
    if tuple(parent_manifest.get("games", ())) != (game,):
        raise ValueError("T12.2 must use the same single game as its T12.1 parent")
    if parent_receipt.get("phase") != "archive":
        raise ValueError("T12.2 parent must be the T12.1 archive receipt")
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.2 is authorized only as a correction to a failed archive gate")
    if parent_receipt.get("status") != "FAIL_ARCHIVE_GATE":
        raise ValueError("unexpected T12.1 parent failure status")
    missing = [path for path in BURST_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.2 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    scientific_claims_authorized = bool(
        not git["dirty"] and parent_manifest.get("scientific_claims_authorized", False)
    )
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    payload = {
        "format_version": BURST_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_2_BURST_EXPLORATION",
        "stage": "source_train",
        "game_id": game,
        "game_split": SAGE11_SPLITS.split_for(game),
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
                "passed": False,
                "status": "FAIL_ARCHIVE_GATE",
            },
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in BURST_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": scientific_claims_authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
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
    freeze_receipt = burst_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=scientific_claims_authorized,
        status=(
            "PASS_T12_2_FREEZE"
            if scientific_claims_authorized
            else "DIRTY_SMOKE_ONLY"
        ),
        metrics={},
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), freeze_receipt)
    return manifest


def load_burst_manifest(
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
    if manifest.get("format_version") != BURST_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.2 manifest")
    protocol = BurstExploreProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.2 protocol checksum mismatch")
    for key in ("manifest", "receipt"):
        meta = dict(manifest["parent"][key])
        parent_path = Path(meta["path"])
        if not parent_path.is_absolute():
            parent_path = repo_root / parent_path
        if not parent_path.is_file() or _file_sha256(parent_path) != meta["sha256"]:
            raise ValueError(f"T12.1 parent {key} checksum mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.2 code checksum mismatch: {relative}")
    return manifest


def burst_phase_receipt(
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
            "format_version": BURST_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_1_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_burst_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != BURST_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.2 receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.2 receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.2 receipt belongs to another protocol")
    return receipt


__all__ = [
    "BURST_MANIFEST_FORMAT",
    "BURST_PROTOCOL_FORMAT",
    "BURST_RECEIPT_FORMAT",
    "BurstExploreProtocol",
    "burst_phase_receipt",
    "freeze_burst_experiment",
    "load_burst_manifest",
    "load_burst_receipt",
]
