"""Frozen SAGE.T12.1 graph-exploration protocol and receipt chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
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

GRAPH_PROTOCOL_FORMAT = "sage-t12.1-graph-explore-protocol-v1"
GRAPH_MANIFEST_FORMAT = "sage-t12.1-graph-explore-manifest-v1"
GRAPH_RECEIPT_FORMAT = "sage-t12.1-graph-explore-receipt-v1"

GRAPH_CODE_PATHS = (
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/terminal_shield.py",
    "theory/sage_t/causal/novelty.py",
    "theory/sage_t/causal/options.py",
    "theory/sage_t/causal/graph_protocol.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/graph_experiment_cli.py",
    "theory/sage_t/causal/contracts.py",
    "theory/sage_t/causal/compiler.py",
    "theory/sage_t/causal/executor.py",
    "theory/sage_t/causal/posterior.py",
    "theory/sage_t/causal/mechanisms.py",
    "theory/sage/live_prefix_counterfactual_collector.py",
)


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GraphExploreProtocol:
    format_version: str = GRAPH_PROTOCOL_FORMAT
    sdk_call_budget_per_seed_arm: int = 8_192
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_cells: int = 50_000
    terminal_horizon: int = 64
    terminal_minimum_support: int = 2
    option_maximum_horizon: int = 32
    archive_minimum_relative_coverage_gain: float = 0.25
    neural_minimum_relative_coverage_gain: float = 0.10
    neural_minimum_examples: int = 512
    neural_minimum_brier_gain: float = 0.01
    neural_minimum_state_shuffle_degradation: float = 0.01
    neural_maximum_ece: float = 0.10
    option_minimum_posterior_mass: float = 0.80
    transfer_levels: int = 3
    archive_seeds: tuple[int, ...] = (6101, 6102, 6103)
    shield_seeds: tuple[int, ...] = (6201, 6202, 6203)
    neural_seeds: tuple[int, ...] = (6301, 6302, 6303)
    transfer_seeds: tuple[int, ...] = (6401, 6402, 6403)
    archive_arms: tuple[str, ...] = ("baseline", "symbolic_archive")
    shield_arms: tuple[str, ...] = ("symbolic_archive", "archive_shield")
    neural_arms: tuple[str, ...] = ("archive_shield", "archive_shield_neural")
    transfer_arms: tuple[str, ...] = (
        "archive_no_option",
        "raw_option",
        "causal_option_full",
        "causal_option_no_posterior_update",
    )
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        if self.format_version != GRAPH_PROTOCOL_FORMAT:
            raise ValueError("unsupported graph-exploration protocol")
        if int(self.sdk_call_budget_per_seed_arm) <= 0:
            raise ValueError("SDK call budget must be positive")
        if int(self.maximum_artifact_bytes_per_run) <= 0:
            raise ValueError("artifact budget must be positive")
        if int(self.terminal_horizon) != 64:
            raise ValueError("SAGE.T12.1 preregisters a 64-step terminal horizon")
        if int(self.transfer_levels) != 3:
            raise ValueError("SAGE.T12.1 preregisters three transfer levels")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def freeze_graph_experiment(
    *,
    output_path: str | Path,
    stage: str,
    game_ids: Sequence[str],
    program_registry_path: str | Path | None = None,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: GraphExploreProtocol | None = None,
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    protocol = protocol or GraphExploreProtocol()
    normalized_stage = str(stage)
    allowed_splits = {
        "source_train": {"source_train"},
        "regression": {"historical_benchmark", "ar25_regression_only"},
        "historical": {"historical_benchmark"},
    }
    if normalized_stage not in allowed_splits:
        raise ValueError("graph exploration stage must be source_train/regression/historical")
    games = tuple(short_game_id(value) for value in game_ids)
    if not games:
        raise ValueError("graph-exploration manifest needs at least one game")
    if len(games) != 1:
        raise ValueError(
            "use one manifest per game so learned archive, shield, model and memory "
            "are reset between validation games"
        )
    split_map = {game: SAGE11_SPLITS.split_for(game) for game in games}
    violations = {
        game: split_name
        for game, split_name in split_map.items()
        if split_name not in allowed_splits[normalized_stage]
    }
    if violations:
        raise ValueError(f"graph-exploration split firewall rejected games: {violations}")
    missing = [path for path in GRAPH_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"graph-exploration code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    scientific_claims_authorized = not git["dirty"]
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    if normalized_stage == "source_train" and program_registry_path is None:
        raise ValueError("source_train graph exploration requires a causal program registry")
    registry = None
    if program_registry_path is not None:
        registry_path = Path(program_registry_path).resolve()
        if not registry_path.is_file():
            raise FileNotFoundError(registry_path)
        registry_payload = _read_json(registry_path)
        _verify_signed(registry_payload, "registry_checksum")
        if registry_payload.get("format_version") != "sage-t11-causal-program-registry-v1":
            raise ValueError("unsupported causal program registry")
        missing_games = sorted(set(games) - set(registry_payload.get("games", {})))
        if missing_games:
            raise ValueError(f"program registry lacks manifest games: {missing_games}")
        registry = {
            "path": _bound_path(registry_path, root=repo_root),
            "sha256": _file_sha256(registry_path),
            "registry_checksum": registry_payload["registry_checksum"],
            "source_protocol_checksum": registry_payload.get("protocol_checksum"),
        }
    payload = {
        "format_version": GRAPH_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_GRAPH_EXPLORATION",
        "stage": normalized_stage,
        "games": list(games),
        "game_splits": split_map,
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in GRAPH_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": scientific_claims_authorized,
        "program_registry": registry,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "historical_used_for_tuning": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": (
                protocol.maximum_artifact_bytes_per_run
            ),
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = _signed(
        {
            "format_version": GRAPH_RECEIPT_FORMAT,
            "phase": "freeze",
            "passed": scientific_claims_authorized,
            "status": (
                "PASS_GRAPH_FREEZE" if scientific_claims_authorized else "DIRTY_SMOKE_ONLY"
            ),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": protocol.checksum,
            "metrics": {},
        },
        "receipt_checksum",
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_graph_manifest(
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
    if manifest.get("format_version") != GRAPH_MANIFEST_FORMAT:
        raise ValueError("unsupported graph-exploration manifest")
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("graph-exploration protocol checksum mismatch")
    if verify_code:
        for relative, expected in dict(manifest.get("code_sha256", {})).items():
            path_value = repo_root / relative
            if not path_value.is_file() or _file_sha256(path_value) != str(expected):
                raise ValueError(f"graph-exploration code checksum mismatch: {relative}")
    return manifest


def phase_receipt(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    passed: bool,
    status: str,
    metrics: Mapping[str, Any],
    parent_receipt: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _signed(
        {
            "format_version": GRAPH_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": str(manifest["manifest_checksum"]),
            "protocol_checksum": str(manifest["protocol_checksum"]),
            "parent_receipt_checksum": (
                None
                if parent_receipt is None
                else str(parent_receipt["receipt_checksum"])
            ),
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_graph_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != GRAPH_RECEIPT_FORMAT:
        raise ValueError("unsupported graph-exploration receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("receipt belongs to another graph-exploration manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("receipt belongs to another protocol")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"upstream graph-exploration gate failed: {receipt.get('status')}")
    return receipt


__all__ = [
    "GRAPH_MANIFEST_FORMAT",
    "GRAPH_PROTOCOL_FORMAT",
    "GRAPH_RECEIPT_FORMAT",
    "GraphExploreProtocol",
    "freeze_graph_experiment",
    "load_graph_manifest",
    "load_graph_receipt",
    "phase_receipt",
]
