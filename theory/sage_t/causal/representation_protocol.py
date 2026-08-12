"""Frozen T12.4a protocol for relational novelty representation repair."""

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
from .neural_novelty_protocol import (
    load_neural_novelty_manifest,
    load_neural_novelty_receipt,
)

REPRESENTATION_PROTOCOL_FORMAT = "sage-t12.4a-representation-protocol-v1"
REPRESENTATION_DATASET_FORMAT = "sage-t12.4a-representation-dataset-v1"
REPRESENTATION_MANIFEST_FORMAT = "sage-t12.4a-representation-manifest-v1"
REPRESENTATION_RECEIPT_FORMAT = "sage-t12.4a-representation-receipt-v1"

REPRESENTATION_CODE_PATHS = (
    "theory/sage_t/causal/relational_novelty.py",
    "theory/sage_t/causal/representation_protocol.py",
    "theory/sage_t/causal/representation_experiment.py",
    "theory/sage_t/causal/representation_experiment_cli.py",
    "theory/sage_t/causal/neural_novelty_protocol.py",
    "theory/sage_t/causal/neural_novelty_experiment.py",
    "theory/sage_t/causal/novelty.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/lineage_archive.py",
    "theory/sage_t/causal/shield_model.py",
    "theory/sage_t/causal/terminal_shield.py",
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
class RepresentationProtocol:
    format_version: str = REPRESENTATION_PROTOCOL_FORMAT
    diagnostic_seed_excluded: int = 7_703
    collection_seeds: tuple[int, ...] = (8_401, 8_402, 8_403)
    training_seeds: tuple[int, ...] = (8_401, 8_402)
    validation_seed: int = 8_403
    collection_arm: str = "lineage_shield_control"
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    sdk_calls_per_seed: int = 4_096
    maximum_total_sdk_calls: int = 15_000
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_cells: int = 50_000
    minimum_collection_replay_exact_rate: float = 0.95
    minimum_shield_vetoes: int = 1
    minimum_training_examples: int = 512
    minimum_validation_examples: int = 200
    minimum_unique_actions: int = 8
    minimum_relational_feature_coverage: float = 0.20
    minimum_unique_archive_contexts: int = 32
    minimum_label_prevalence: float = 0.05
    maximum_label_prevalence: float = 0.95
    hidden_dim: int = 32
    batch_size: int = 32
    training_epochs: int = 8
    learning_rate: float = 1e-3
    torch_seed: int = 8_424
    maximum_parameters: int = 15_000
    minimum_change_brier_gain: float = 0.01
    minimum_novelty_brier_gain: float = 0.01
    minimum_legacy_mean_brier_improvement: float = 0.01
    minimum_state_shuffle_change_degradation: float = 0.01
    minimum_context_shuffle_novelty_degradation: float = 0.01
    minimum_relation_ablation_degradation: float = 0.005
    maximum_ece: float = 0.10
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in ("collection_seeds", "training_seeds", "burst_schedule"):
            object.__setattr__(
                self,
                name,
                tuple(int(value) for value in getattr(self, name)),
            )
        if self.format_version != REPRESENTATION_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.4a representation protocol")
        fixed = {
            "diagnostic_seed_excluded": (self.diagnostic_seed_excluded, 7_703),
            "collection_seeds": (self.collection_seeds, (8_401, 8_402, 8_403)),
            "training_seeds": (self.training_seeds, (8_401, 8_402)),
            "validation_seed": (self.validation_seed, 8_403),
            "collection_arm": (self.collection_arm, "lineage_shield_control"),
            "burst_schedule": (self.burst_schedule, (4, 8, 16)),
            "sdk_calls_per_seed": (self.sdk_calls_per_seed, 4_096),
            "maximum_total_sdk_calls": (self.maximum_total_sdk_calls, 15_000),
            "maximum_artifact_bytes_per_run": (
                self.maximum_artifact_bytes_per_run,
                3 * 1024 * 1024 * 1024,
            ),
            "maximum_cells": (self.maximum_cells, 50_000),
            "minimum_collection_replay_exact_rate": (
                self.minimum_collection_replay_exact_rate,
                0.95,
            ),
            "minimum_shield_vetoes": (self.minimum_shield_vetoes, 1),
            "minimum_training_examples": (self.minimum_training_examples, 512),
            "minimum_validation_examples": (
                self.minimum_validation_examples,
                200,
            ),
            "minimum_unique_actions": (self.minimum_unique_actions, 8),
            "minimum_relational_feature_coverage": (
                self.minimum_relational_feature_coverage,
                0.20,
            ),
            "minimum_unique_archive_contexts": (
                self.minimum_unique_archive_contexts,
                32,
            ),
            "minimum_label_prevalence": (
                self.minimum_label_prevalence,
                0.05,
            ),
            "maximum_label_prevalence": (
                self.maximum_label_prevalence,
                0.95,
            ),
            "hidden_dim": (self.hidden_dim, 32),
            "batch_size": (self.batch_size, 32),
            "training_epochs": (self.training_epochs, 8),
            "learning_rate": (self.learning_rate, 1e-3),
            "torch_seed": (self.torch_seed, 8_424),
            "maximum_parameters": (self.maximum_parameters, 15_000),
            "minimum_change_brier_gain": (
                self.minimum_change_brier_gain,
                0.01,
            ),
            "minimum_novelty_brier_gain": (
                self.minimum_novelty_brier_gain,
                0.01,
            ),
            "minimum_legacy_mean_brier_improvement": (
                self.minimum_legacy_mean_brier_improvement,
                0.01,
            ),
            "minimum_state_shuffle_change_degradation": (
                self.minimum_state_shuffle_change_degradation,
                0.01,
            ),
            "minimum_context_shuffle_novelty_degradation": (
                self.minimum_context_shuffle_novelty_degradation,
                0.01,
            ),
            "minimum_relation_ablation_degradation": (
                self.minimum_relation_ablation_degradation,
                0.005,
            ),
            "maximum_ece": (self.maximum_ece, 0.10),
        }
        for name, (observed, expected) in fixed.items():
            if observed != expected:
                raise ValueError(f"T12.4a preregistered value changed: {name}")
        if set(self.training_seeds) | {self.validation_seed} != set(
            self.collection_seeds
        ):
            raise ValueError("T12.4a split must partition the collection seeds")
        if self.diagnostic_seed_excluded in self.collection_seeds:
            raise ValueError("T12.4a cannot reuse the opened diagnostic seed")
        if len(self.collection_seeds) * self.sdk_calls_per_seed > self.maximum_total_sdk_calls:
            raise ValueError("T12.4a collection exceeds the SDK budget")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _parent_failure_is_representation_only(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> bool:
    protocol = dict(manifest["protocol"])
    metrics = dict(receipt.get("metrics", {}))
    return bool(
        receipt.get("passed") is False
        and receipt.get("phase") == "train"
        and receipt.get("status") == "FAIL_T12_4_NEURAL_FIT_GATE"
        and float(metrics.get("brier_gain", -1.0))
        >= float(protocol["minimum_brier_gain"])
        and float(metrics.get("state_shuffle_degradation", 1.0))
        < float(protocol["minimum_state_shuffle_degradation"])
        and float(metrics.get("maximum_ece", 0.0))
        > float(protocol["maximum_ece"])
        and int(metrics.get("parameter_count", 1_000_000))
        <= int(protocol["maximum_parameters"])
    )


def freeze_representation_experiment(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: RepresentationProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or RepresentationProtocol()
    parent_manifest = load_neural_novelty_manifest(
        parent_manifest_path,
        root=repo_root,
    )
    parent_receipt = load_neural_novelty_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
    )
    if not _parent_failure_is_representation_only(parent_manifest, parent_receipt):
        raise ValueError("T12.4a requires the representation-only T12.4 failure")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.4a is restricted to source_train")
    shield_meta = dict(parent_manifest["shield"])
    shield_path = _resolve_bound(str(shield_meta["path"]), root=repo_root)
    if not shield_path.is_file() or _file_sha256(shield_path) != shield_meta["sha256"]:
        raise ValueError("T12.4a shield checksum mismatch")
    missing = [
        path for path in REPRESENTATION_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.4a code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(
        not git["dirty"] and parent_manifest.get("scientific_claims_authorized", False)
    )
    payload = {
        "format_version": REPRESENTATION_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_REPRESENTATION_REPAIR",
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
                "passed": False,
                "status": "FAIL_T12_4_NEURAL_FIT_GATE",
                "failure_class": "STATE_AND_ARCHIVE_CONTEXT_REPRESENTATION",
            },
        },
        "shield": {
            "path": _bound_path(shield_path, root=repo_root),
            "sha256": shield_meta["sha256"],
        },
        "diagnostic_exclusions": {
            "opened_seed": selected.diagnostic_seed_excluded,
            "reason": "observed_during_t12_4_model_diagnosis",
            "allowed_for_confirmatory_fit_or_validation": False,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in REPRESENTATION_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "representation_collection_authorized": authorized,
            "representation_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "option_extraction_authorized": False,
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
    receipt = representation_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_4A_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={"failure_class": manifest["parent"]["receipt"]["failure_class"]},
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def seal_representation_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["format_version"] = REPRESENTATION_DATASET_FORMAT
    return _signed(body, "dataset_checksum")


def load_representation_dataset(
    path: str | Path,
    *,
    protocol: RepresentationProtocol | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "dataset_checksum")
    if payload.get("format_version") != REPRESENTATION_DATASET_FORMAT:
        raise ValueError("unsupported T12.4a representation dataset")
    selected = protocol or RepresentationProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("T12.4a dataset protocol mismatch")
    train = [item for item in payload.get("examples", ()) if item["split"] == "train"]
    validation = [
        item for item in payload.get("examples", ()) if item["split"] == "validation"
    ]
    if len(train) < selected.minimum_training_examples:
        raise ValueError("T12.4a dataset lost training examples")
    if len(validation) < selected.minimum_validation_examples:
        raise ValueError("T12.4a dataset lost validation examples")
    return payload


def load_representation_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != REPRESENTATION_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a manifest")
    protocol = RepresentationProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a protocol checksum mismatch")
    for section, key in (("parent", "manifest"), ("parent", "receipt")):
        meta = dict(manifest[section][key])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a parent artifact mismatch: {key}")
    shield_meta = dict(manifest["shield"])
    shield_path = _resolve_bound(str(shield_meta["path"]), root=repo_root)
    if not shield_path.is_file() or _file_sha256(shield_path) != shield_meta["sha256"]:
        raise ValueError("T12.4a shield artifact mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a code checksum mismatch: {relative}")
    return manifest


def representation_phase_receipt(
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
            "format_version": REPRESENTATION_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4_receipt_checksum": manifest["parent"]["receipt"][
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


def load_representation_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != REPRESENTATION_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.4a receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.4a receipt belongs to another protocol")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.4a upstream gate failed: {receipt.get('status')}")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        candidate = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta.get("sha256"):
            raise ValueError(f"T12.4a receipt artifact mismatch: {name}")
        if name == "collection":
            collection = _read_json(candidate)
            for condition in collection.get("conditions", ()):
                for artifact_name in ("archive", "excursions"):
                    nested = dict(condition[artifact_name])
                    nested_path = _resolve_bound(str(nested["path"]), root=repo_root)
                    if (
                        not nested_path.is_file()
                        or _file_sha256(nested_path) != nested["sha256"]
                    ):
                        raise ValueError(
                            f"T12.4a collection artifact mismatch: {artifact_name}"
                        )
    return receipt


__all__ = [
    "RepresentationProtocol",
    "freeze_representation_experiment",
    "load_representation_dataset",
    "load_representation_manifest",
    "load_representation_receipt",
    "representation_phase_receipt",
    "seal_representation_dataset",
]
