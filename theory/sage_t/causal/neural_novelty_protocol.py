"""Frozen T12.4 protocol for the action-conditioned novelty predictor."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .archive import (
    GoExploreArchive,
    abstract_state_to_payload,
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
from .lineage_shield_protocol import (
    load_lineage_shield_manifest,
    load_lineage_shield_receipt,
    load_lineage_shield_registry,
)

NEURAL_NOVELTY_PROTOCOL_FORMAT = "sage-t12.4-neural-novelty-protocol-v1"
NEURAL_NOVELTY_DATASET_FORMAT = "sage-t12.4-neural-novelty-dataset-v1"
NEURAL_NOVELTY_MANIFEST_FORMAT = "sage-t12.4-neural-novelty-manifest-v1"
NEURAL_NOVELTY_RECEIPT_FORMAT = "sage-t12.4-neural-novelty-receipt-v1"

NEURAL_NOVELTY_CODE_PATHS = (
    "theory/sage_t/causal/neural_novelty_protocol.py",
    "theory/sage_t/causal/neural_novelty_experiment.py",
    "theory/sage_t/causal/neural_novelty_experiment_cli.py",
    "theory/sage_t/causal/novelty.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/lineage_archive.py",
    "theory/sage_t/causal/lineage_shield_experiment.py",
    "theory/sage_t/causal/lineage_shield_protocol.py",
    "theory/sage_t/causal/shield_model.py",
    "theory/sage_t/causal/terminal_shield.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/compiler.py",
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
class NeuralNoveltyProtocol:
    format_version: str = NEURAL_NOVELTY_PROTOCOL_FORMAT
    source_arm: str = "lineage_control"
    training_seeds: tuple[int, ...] = (7701, 7702)
    validation_seed: int = 7703
    evaluation_seeds: tuple[int, ...] = (8101, 8102, 8103)
    evaluation_arms: tuple[str, ...] = (
        "lineage_shield_control",
        "lineage_shield_neural",
    )
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    sdk_calls_per_evaluation_arm: int = 4_096
    maximum_total_sdk_calls: int = 30_000
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_cells: int = 50_000
    minimum_training_examples: int = 512
    minimum_validation_examples: int = 200
    minimum_unique_actions: int = 8
    minimum_label_prevalence: float = 0.05
    maximum_label_prevalence: float = 0.95
    raw_change_universal_threshold: float = 0.99
    hidden_dim: int = 32
    batch_size: int = 32
    training_epochs: int = 8
    learning_rate: float = 1e-3
    torch_seed: int = 8_124
    maximum_parameters: int = 15_000
    minimum_brier_gain: float = 0.01
    minimum_state_shuffle_degradation: float = 0.01
    maximum_ece: float = 0.10
    minimum_relative_coverage_gain: float = 0.10
    minimum_per_seed_coverage_ratio: float = 0.80
    maximum_terminal_rate_ratio: float = 1.0
    maximum_terminal_regression_seeds: int = 0
    maximum_progress_regression_seeds: int = 0
    minimum_replay_exact_rate: float = 0.95
    minimum_neural_action_changes: int = 1
    maximum_p95_decision_latency_ms: float = 50.0
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in ("training_seeds", "evaluation_seeds", "burst_schedule"):
            object.__setattr__(
                self,
                name,
                tuple(int(value) for value in getattr(self, name)),
            )
        object.__setattr__(
            self,
            "evaluation_arms",
            tuple(str(value) for value in self.evaluation_arms),
        )
        if self.format_version != NEURAL_NOVELTY_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.4 neural-novelty protocol")
        fixed = {
            "source_arm": (self.source_arm, "lineage_control"),
            "training_seeds": (self.training_seeds, (7701, 7702)),
            "validation_seed": (self.validation_seed, 7703),
            "evaluation_seeds": (self.evaluation_seeds, (8101, 8102, 8103)),
            "evaluation_arms": (
                self.evaluation_arms,
                ("lineage_shield_control", "lineage_shield_neural"),
            ),
            "burst_schedule": (self.burst_schedule, (4, 8, 16)),
            "sdk_calls_per_evaluation_arm": (
                self.sdk_calls_per_evaluation_arm,
                4_096,
            ),
            "maximum_total_sdk_calls": (self.maximum_total_sdk_calls, 30_000),
            "maximum_artifact_bytes_per_run": (
                self.maximum_artifact_bytes_per_run,
                3 * 1024 * 1024 * 1024,
            ),
            "maximum_cells": (self.maximum_cells, 50_000),
            "minimum_training_examples": (self.minimum_training_examples, 512),
            "minimum_validation_examples": (
                self.minimum_validation_examples,
                200,
            ),
            "minimum_unique_actions": (self.minimum_unique_actions, 8),
            "minimum_label_prevalence": (
                self.minimum_label_prevalence,
                0.05,
            ),
            "maximum_label_prevalence": (
                self.maximum_label_prevalence,
                0.95,
            ),
            "raw_change_universal_threshold": (
                self.raw_change_universal_threshold,
                0.99,
            ),
            "hidden_dim": (self.hidden_dim, 32),
            "batch_size": (self.batch_size, 32),
            "training_epochs": (self.training_epochs, 8),
            "learning_rate": (self.learning_rate, 1e-3),
            "torch_seed": (self.torch_seed, 8_124),
            "maximum_parameters": (self.maximum_parameters, 15_000),
            "minimum_brier_gain": (self.minimum_brier_gain, 0.01),
            "minimum_state_shuffle_degradation": (
                self.minimum_state_shuffle_degradation,
                0.01,
            ),
            "maximum_ece": (self.maximum_ece, 0.10),
            "minimum_relative_coverage_gain": (
                self.minimum_relative_coverage_gain,
                0.10,
            ),
            "minimum_per_seed_coverage_ratio": (
                self.minimum_per_seed_coverage_ratio,
                0.80,
            ),
            "maximum_terminal_rate_ratio": (
                self.maximum_terminal_rate_ratio,
                1.0,
            ),
            "maximum_terminal_regression_seeds": (
                self.maximum_terminal_regression_seeds,
                0,
            ),
            "maximum_progress_regression_seeds": (
                self.maximum_progress_regression_seeds,
                0,
            ),
            "minimum_replay_exact_rate": (
                self.minimum_replay_exact_rate,
                0.95,
            ),
            "minimum_neural_action_changes": (
                self.minimum_neural_action_changes,
                1,
            ),
            "maximum_p95_decision_latency_ms": (
                self.maximum_p95_decision_latency_ms,
                50.0,
            ),
        }
        for name, (observed, expected) in fixed.items():
            if observed != expected:
                raise ValueError(f"T12.4 preregistered value changed: {name}")
        evaluation_upper_bound = (
            len(self.evaluation_seeds)
            * len(self.evaluation_arms)
            * self.sdk_calls_per_evaluation_arm
        )
        if evaluation_upper_bound > self.maximum_total_sdk_calls:
            raise ValueError("T12.4 active evaluation exceeds the SDK budget")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _label_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(examples)
    if count == 0:
        return {
            "examples": 0,
            "raw_changed_prevalence": 0.0,
            "semantic_changed_prevalence": 0.0,
            "novelty_prevalence": 0.0,
            "unique_actions": 0,
            "unique_states": 0,
        }
    return {
        "examples": count,
        "raw_changed_prevalence": sum(
            bool(item["raw_changed"]) for item in examples
        )
        / count,
        "semantic_changed_prevalence": sum(
            bool(item["semantic_changed"]) for item in examples
        )
        / count,
        "novelty_prevalence": sum(bool(item["novel"]) for item in examples)
        / count,
        "unique_actions": len({str(item["action_key"]) for item in examples}),
        "unique_states": len({str(item["source_state_id"]) for item in examples}),
    }


def compile_neural_novelty_dataset(
    *,
    paired_evaluation: Mapping[str, Any],
    root: Path,
    protocol: NeuralNoveltyProtocol,
) -> dict[str, Any]:
    selected_seeds = {*protocol.training_seeds, protocol.validation_seed}
    conditions = {
        int(item["seed"]): dict(item)
        for item in paired_evaluation.get("conditions", ())
        if int(item["seed"]) in selected_seeds
    }
    if set(conditions) != selected_seeds:
        raise ValueError("T12.4 source evaluation lacks a frozen seed")
    states: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    sources = []
    for seed in (*protocol.training_seeds, protocol.validation_seed):
        condition = conditions[seed]
        arm = dict(condition["arms"])[protocol.source_arm]
        archive_meta = dict(arm["archive"])
        archive_path = _resolve_bound(str(archive_meta["path"]), root=root)
        if not archive_path.is_file() or _file_sha256(archive_path) != archive_meta[
            "sha256"
        ]:
            raise ValueError(f"T12.4 source archive checksum mismatch: {seed}")
        archive = GoExploreArchive.from_dict(_read_json(archive_path))
        split = "train" if seed in protocol.training_seeds else "validation"
        for edge in sorted(archive.edges.values(), key=lambda item: item.ordinal):
            source_state = archive.cells[edge.source_cell_id].state
            target_state = archive.cells[edge.target_cell_id].state
            state_id = source_state.execution_signature
            state_payload = abstract_state_to_payload(source_state)
            previous = states.setdefault(state_id, state_payload)
            if previous != state_payload:
                raise ValueError("T12.4 state signature collision")
            semantic_changed = source_state.signature != target_state.signature
            action_payload = {
                "action_name": edge.action.action_name,
                "action_data": dict(edge.action.action_data),
            }
            example_payload = {
                "seed": seed,
                "split": split,
                "ordinal": edge.ordinal,
                "edge_id": edge.edge_id,
                "source_state_id": state_id,
                "action": action_payload,
                "action_key": edge.action.key,
                "raw_changed": edge.changed,
                "semantic_changed": semantic_changed,
                "novel": edge.novel,
            }
            example_payload["example_id"] = _checksum(example_payload)
            examples.append(example_payload)
        sources.append(
            {
                "seed": seed,
                "arm": protocol.source_arm,
                "path": _bound_path(archive_path, root=root),
                "sha256": archive_meta["sha256"],
                "edges": len(archive.edges),
            }
        )
    if len({item["example_id"] for item in examples}) != len(examples):
        raise ValueError("T12.4 dataset contains duplicate example identities")
    train = [item for item in examples if item["split"] == "train"]
    validation = [item for item in examples if item["split"] == "validation"]
    train_states = {str(item["source_state_id"]) for item in train}
    validation_states = {str(item["source_state_id"]) for item in validation}
    train_state_actions = {
        (str(item["source_state_id"]), str(item["action_key"])) for item in train
    }
    validation_state_actions = {
        (str(item["source_state_id"]), str(item["action_key"]))
        for item in validation
    }
    qa = {
        "train": _label_metrics(train),
        "validation": _label_metrics(validation),
        "all": _label_metrics(examples),
        "raw_label_is_universal": bool(
            _label_metrics(examples)["raw_changed_prevalence"]
            >= protocol.raw_change_universal_threshold
        ),
        "amended_label": "abstract_state.signature_before!=signature_after",
        "train_validation_state_overlap": len(train_states & validation_states),
        "validation_state_overlap_fraction": (
            len(train_states & validation_states) / max(1, len(validation_states))
        ),
        "train_validation_state_action_overlap": len(
            train_state_actions & validation_state_actions
        ),
        "validation_state_action_overlap_fraction": (
            len(train_state_actions & validation_state_actions)
            / max(1, len(validation_state_actions))
        ),
    }
    for split_name, minimum_examples in (
        ("train", protocol.minimum_training_examples),
        ("validation", protocol.minimum_validation_examples),
    ):
        metrics = qa[split_name]
        if int(metrics["examples"]) < minimum_examples:
            raise ValueError(f"T12.4 {split_name} split is too small")
        if int(metrics["unique_actions"]) < protocol.minimum_unique_actions:
            raise ValueError(f"T12.4 {split_name} action support is too small")
        for label in ("semantic_changed_prevalence", "novelty_prevalence"):
            prevalence = float(metrics[label])
            if not (
                protocol.minimum_label_prevalence
                <= prevalence
                <= protocol.maximum_label_prevalence
            ):
                raise ValueError(f"T12.4 implausible {split_name} label: {label}")
    if not qa["raw_label_is_universal"]:
        raise ValueError("T12.4 expected the preregistered raw-label defect")
    return _signed(
        {
            "format_version": NEURAL_NOVELTY_DATASET_FORMAT,
            "protocol_checksum": protocol.checksum,
            "label_amendment": {
                "rejected_label": "edge.changed/raw_pixel_hash_delta",
                "rejection_reason": "universal_derived_label",
                "replacement_label": qa["amended_label"],
                "novelty_label": "first_observation_of_target_symbolic_cell",
            },
            "sources": sources,
            "states": states,
            "examples": examples,
            "qa": qa,
        },
        "dataset_checksum",
    )


def freeze_neural_novelty_experiment(
    *,
    output_path: str | Path,
    dataset_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: NeuralNoveltyProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or NeuralNoveltyProtocol()
    parent_manifest = load_lineage_shield_manifest(
        parent_manifest_path, root=repo_root
    )
    parent_receipt = load_lineage_shield_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
    )
    if parent_receipt.get("passed") is not True or parent_receipt.get(
        "status"
    ) != "PASS_T12_3E_LINEAGE_SHIELD_GATE":
        raise ValueError("T12.4 requires a passed T12.3e parent")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.4 is restricted to source_train")
    paired_meta = dict(parent_receipt["artifacts"]["paired_evaluation"])
    paired_path = _resolve_bound(str(paired_meta["path"]), root=repo_root)
    paired = _read_json(paired_path)
    dataset = compile_neural_novelty_dataset(
        paired_evaluation=paired,
        root=repo_root,
        protocol=selected,
    )
    _write_json_once(dataset_path, dataset)

    source_registry_path = _resolve_bound(
        str(parent_manifest["source_registry"]["path"]), root=repo_root
    )
    source_registry = load_lineage_shield_registry(source_registry_path)
    shield_meta = dict(source_registry["terminal_shield"])
    shield_path = _resolve_bound(str(shield_meta["path"]), root=repo_root)
    if not shield_path.is_file() or _file_sha256(shield_path) != shield_meta["sha256"]:
        raise ValueError("T12.4 terminal shield checksum mismatch")

    missing = [
        path for path in NEURAL_NOVELTY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.4 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(
        not git["dirty"]
        and parent_manifest.get("scientific_claims_authorized", False)
        and parent_receipt.get("passed") is True
    )
    payload = {
        "format_version": NEURAL_NOVELTY_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4_NEURAL_NOVELTY",
        "stage": "source_train",
        "game_id": parent_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "dataset": {
            "path": _bound_path(dataset_path, root=repo_root),
            "sha256": _file_sha256(dataset_path),
            "dataset_checksum": dataset["dataset_checksum"],
            "qa": dataset["qa"],
        },
        "shield": {
            "path": _bound_path(shield_path, root=repo_root),
            "sha256": shield_meta["sha256"],
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
                "status": "PASS_T12_3E_LINEAGE_SHIELD_GATE",
            },
            "paired_evaluation": {
                "path": _bound_path(paired_path, root=repo_root),
                "sha256": paired_meta["sha256"],
            },
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in NEURAL_NOVELTY_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "neural_novelty_training_authorized": authorized,
            "neural_active_evaluation_authorized": False,
            "option_extraction_authorized": False,
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
    receipt = neural_novelty_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_4_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics=dict(dataset["qa"]),
        artifacts={"dataset": dict(manifest["dataset"])},
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_neural_novelty_dataset(
    path: str | Path,
    *,
    protocol: NeuralNoveltyProtocol | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "dataset_checksum")
    if payload.get("format_version") != NEURAL_NOVELTY_DATASET_FORMAT:
        raise ValueError("unsupported T12.4 neural-novelty dataset")
    selected = protocol or NeuralNoveltyProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("T12.4 dataset protocol mismatch")
    train = [item for item in payload.get("examples", ()) if item["split"] == "train"]
    validation = [
        item for item in payload.get("examples", ()) if item["split"] == "validation"
    ]
    if len(train) < selected.minimum_training_examples:
        raise ValueError("T12.4 dataset lost training examples")
    if len(validation) < selected.minimum_validation_examples:
        raise ValueError("T12.4 dataset lost validation examples")
    return payload


def load_neural_novelty_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != NEURAL_NOVELTY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4 manifest")
    protocol = NeuralNoveltyProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4 protocol checksum mismatch")
    for section, key in (
        ("dataset", "dataset"),
        ("shield", "shield"),
        ("parent", "manifest"),
        ("parent", "receipt"),
        ("parent", "paired_evaluation"),
    ):
        meta = (
            dict(manifest[section])
            if section in {"dataset", "shield"}
            else dict(manifest[section][key])
        )
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4 bound artifact mismatch: {section}:{key}")
    dataset_path = _resolve_bound(str(manifest["dataset"]["path"]), root=repo_root)
    dataset = load_neural_novelty_dataset(dataset_path, protocol=protocol)
    if dataset["dataset_checksum"] != manifest["dataset"]["dataset_checksum"]:
        raise ValueError("T12.4 dataset signature mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4 code checksum mismatch: {relative}")
    return manifest


def neural_novelty_phase_receipt(
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
            "format_version": NEURAL_NOVELTY_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_3e_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "parent_receipt_checksum": (
                None
                if parent_receipt is None
                else parent_receipt["receipt_checksum"]
            ),
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_neural_novelty_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != NEURAL_NOVELTY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4 receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.4 receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.4 receipt belongs to another protocol")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.4 upstream gate failed: {receipt.get('status')}")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        candidate = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta.get("sha256"):
            raise ValueError(f"T12.4 receipt artifact mismatch: {name}")
        if name == "paired_evaluation":
            evaluation = _read_json(candidate)
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
                                "T12.4 paired artifact mismatch: "
                                f"{arm_name}:{artifact_name}"
                            )
    return receipt


__all__ = [
    "NeuralNoveltyProtocol",
    "compile_neural_novelty_dataset",
    "freeze_neural_novelty_experiment",
    "load_neural_novelty_dataset",
    "load_neural_novelty_manifest",
    "load_neural_novelty_receipt",
    "neural_novelty_phase_receipt",
]
