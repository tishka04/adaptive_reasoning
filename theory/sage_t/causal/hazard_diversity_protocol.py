"""Frozen T12.4a.4d.1 protocol for hazard abstraction and diverse search."""

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
from .target_regrounding_protocol import (
    load_target_regrounding_manifest,
    load_target_regrounding_receipt,
)

HAZARD_DIVERSITY_PROTOCOL_FORMAT = "sage-t12.4a.4d.1-hazard-diversity-protocol-v1"
HAZARD_DIVERSITY_MANIFEST_FORMAT = "sage-t12.4a.4d.1-hazard-diversity-manifest-v1"
HAZARD_DIVERSITY_RECEIPT_FORMAT = "sage-t12.4a.4d.1-hazard-diversity-receipt-v1"

HAZARD_DIVERSITY_CODE_PATHS = (
    "theory/sage_t/causal/__init__.py",
    "theory/sage_t/causal/hazard_diversity_model.py",
    "theory/sage_t/causal/hazard_diversity_protocol.py",
    "theory/sage_t/causal/hazard_diversity_experiment.py",
    "theory/sage_t/causal/hazard_diversity_cli.py",
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
class HazardDiversityProtocol:
    format_version: str = HAZARD_DIVERSITY_PROTOCOL_FORMAT
    source_lineages: tuple[int, ...] = (8_701, 8_705)
    compile_search_seeds: tuple[int, ...] = (9_101, 9_102, 9_103)
    active_search_seeds: tuple[int, ...] = (9_201, 9_202, 9_203)
    search_arms: tuple[str, ...] = (
        "local_archive_control",
        "diversity_control",
        "abstract_hazard_diversity",
    )
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    local_hazard_radius: int = 7
    minimum_hazard_support: int = 2
    unsafe_rate_threshold: float = 0.75
    minimum_crossfit_recall: float = 0.50
    minimum_crossfit_precision: float = 0.90
    maximum_crossfit_false_positive_rate: float = 0.02
    minimum_recall_passing_folds: int = 2
    sdk_calls_per_search_arm: int = 2_048
    maximum_total_sdk_calls: int = 38_000
    maximum_excursions_per_arm: int = 64
    maximum_cells: int = 10_000
    maximum_action_family_share: float = 0.70
    confirmation_repetitions_per_lineage: int = 2
    minimum_confirmation_exact_rate: float = 1.0
    maximum_witness_suffix_actions: int = 64
    maximum_treatment_terminal_failure_rate: float = 0.10
    minimum_contract_block_margin: float = 0.80
    minimum_progress_edges: int = 1
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    persist_raw_frames: bool = False
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in (
            "source_lineages",
            "compile_search_seeds",
            "active_search_seeds",
            "burst_schedule",
        ):
            object.__setattr__(
                self,
                name,
                tuple(int(value) for value in getattr(self, name)),
            )
        object.__setattr__(
            self,
            "search_arms",
            tuple(str(value) for value in self.search_arms),
        )
        expected = {
            "format_version": HAZARD_DIVERSITY_PROTOCOL_FORMAT,
            "source_lineages": (8_701, 8_705),
            "compile_search_seeds": (9_101, 9_102, 9_103),
            "active_search_seeds": (9_201, 9_202, 9_203),
            "search_arms": (
                "local_archive_control",
                "diversity_control",
                "abstract_hazard_diversity",
            ),
            "burst_schedule": (4, 8, 16),
            "local_hazard_radius": 7,
            "minimum_hazard_support": 2,
            "unsafe_rate_threshold": 0.75,
            "minimum_crossfit_recall": 0.50,
            "minimum_crossfit_precision": 0.90,
            "maximum_crossfit_false_positive_rate": 0.02,
            "minimum_recall_passing_folds": 2,
            "sdk_calls_per_search_arm": 2_048,
            "maximum_total_sdk_calls": 38_000,
            "maximum_excursions_per_arm": 64,
            "maximum_cells": 10_000,
            "maximum_action_family_share": 0.70,
            "confirmation_repetitions_per_lineage": 2,
            "minimum_confirmation_exact_rate": 1.0,
            "maximum_witness_suffix_actions": 64,
            "maximum_treatment_terminal_failure_rate": 0.10,
            "minimum_contract_block_margin": 0.80,
            "minimum_progress_edges": 1,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.4a.4d.1 preregistered value changed: {name}")
        if set(self.compile_search_seeds) & set(self.active_search_seeds):
            raise ValueError("compile and prospective active seeds must be disjoint")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def freeze_hazard_diversity(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: HazardDiversityProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or HazardDiversityProtocol()
    parent_manifest = load_target_regrounding_manifest(
        parent_manifest_path,
        root=repo_root,
    )
    parent_receipt = load_target_regrounding_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
    )
    if not (
        parent_receipt.get("passed") is False
        and parent_receipt.get("phase") == "target_regrounding"
        and parent_receipt.get("status") == "FAIL_T12_4A_4D_TARGET_WITNESS_GATE"
        and parent_manifest.get("stage") == "source_train"
    ):
        raise ValueError("T12.4a.4d.1 requires the failed, sealed T12.4a.4d run")
    checks = dict(parent_receipt.get("metrics", {}).get("checks", {}))
    required_integrity = (
        "all_anchor_replays_exact",
        "all_archive_replays_exact",
        "contracted_option_blocked_at_every_anchor",
        "paired_candidate_catalogs_identical",
        "sdk_budget_respected",
    )
    if not all(bool(checks.get(name)) for name in required_integrity):
        raise ValueError("T12.4a.4d.1 parent integrity gate did not pass")
    if tuple(parent_manifest["protocol"]["search_seeds"]) != (
        selected.compile_search_seeds
    ):
        raise ValueError("T12.4a.4d.1 compile folds differ from parent seeds")
    if SAGE11_SPLITS.split_for(str(parent_manifest["game_id"])) != "source_train":
        raise ValueError("T12.4a.4d.1 is restricted to source_train")
    missing = [
        path for path in HAZARD_DIVERSITY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.4a.4d.1 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": HAZARD_DIVERSITY_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_4D_1_HAZARD_DIVERSITY",
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
            "failure_class": "ACTION_COLLAPSE_AND_EXACT_SHIELD_SUPPORT_MISS",
        },
        "source_artifacts": {
            name: value for name, value in parent_receipt["artifacts"].items()
        },
        "paired_design": {
            "same_action_catalog": True,
            "same_burst_schedule": True,
            "same_lineage_schedule": True,
            "same_sdk_budget": True,
            "three_arms": list(selected.search_arms),
            "contrast_diversity": (
                "local_archive_control versus diversity_control"
            ),
            "contrast_hazard": (
                "diversity_control versus abstract_hazard_diversity"
            ),
            "cross_fit_unit": "held-out T12.4a.4d search seed",
            "prospective_unit": "fresh active search seed",
            "negative_result_policy": (
                "compile miss stops physical execution; active miss stops option "
                "extraction; no threshold retuning or holdout opening"
            ),
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in HAZARD_DIVERSITY_CODE_PATHS
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
            "hazard_compile_authorized": authorized,
            "hazard_diversity_active_run_authorized": False,
            "t12_4a_4e_option_extraction_freeze_authorized": False,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "maximum_total_sdk_calls": selected.maximum_total_sdk_calls,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    _write_json_once(
        Path(output_path).with_name("freeze_receipt.json"),
        hazard_diversity_receipt(
            manifest=manifest,
            phase="freeze",
            passed=authorized,
            status=(
                "PASS_T12_4A_4D_1_FREEZE" if authorized else "DIRTY_SMOKE_ONLY"
            ),
            metrics={
                "active_search_seeds": list(selected.active_search_seeds),
                "compile_search_seeds": list(selected.compile_search_seeds),
                "search_arms": list(selected.search_arms),
            },
        ),
    )
    return manifest


def load_hazard_diversity_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != HAZARD_DIVERSITY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a.4d.1 manifest")
    protocol = HazardDiversityProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a.4d.1 protocol checksum mismatch")
    for meta in manifest["parent"].values():
        if not isinstance(meta, Mapping) or "path" not in meta:
            continue
        candidate = _resolve(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError("T12.4a.4d.1 parent checksum mismatch")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a.4d.1 code checksum mismatch: {relative}")
    return manifest


def hazard_diversity_receipt(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    passed: bool,
    status: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    parent_receipt_checksum: str | None = None,
) -> dict[str, Any]:
    return _signed(
        {
            "format_version": HAZARD_DIVERSITY_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_4d_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "parent_receipt_checksum": parent_receipt_checksum,
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_hazard_diversity_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != HAZARD_DIVERSITY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a.4d.1 receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.4a.4d.1 receipt belongs to another manifest")
    for name, meta in receipt.get("artifacts", {}).items():
        values = meta if isinstance(meta, list) else [meta]
        for item in values:
            if not isinstance(item, Mapping) or "path" not in item:
                continue
            candidate = _resolve(str(item["path"]), root=repo_root)
            if not candidate.is_file() or _file_sha256(candidate) != item["sha256"]:
                raise ValueError(f"T12.4a.4d.1 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "HazardDiversityProtocol",
    "freeze_hazard_diversity",
    "hazard_diversity_receipt",
    "load_hazard_diversity_manifest",
    "load_hazard_diversity_receipt",
]
