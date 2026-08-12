"""Prospective collection and calibration-transport test for SAGE.T12.4a.1."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .archive import abstract_state_from_payload
from .calibration_protocol import (
    CalibrationProtocol,
    calibration_phase_receipt,
    load_calibration_dataset,
    load_calibration_manifest,
    load_calibration_receipt,
    seal_calibration_dataset,
)
from .contracts import GroundedAction
from .experiment import RunStorageBudget, _file_sha256, _read_json, _write_json_once
from .graph_experiment import _write_archive
from .neural_novelty_experiment import run_neural_novelty_arm
from .novelty import ChangeNoveltyMLP, encode_state_action
from .novelty import expected_calibration_error as ece
from .relational_novelty import (
    ArchiveContext,
    RelationalNoveltyPredictor,
    encode_action_entity_relations,
    encode_relational_state_action,
)
from .representation_experiment import (
    _action_priors,
    _fit,
    _head_brier,
    compile_archive_examples,
)


def _resolve_manifest_path(path: str) -> Path:
    candidate = Path(path)
    return (
        candidate
        if candidate.is_absolute()
        else Path(__file__).resolve().parents[3] / candidate
    )


def _split_for_seed(seed: int, protocol: CalibrationProtocol) -> str:
    if seed in protocol.training_seeds:
        return "train"
    if seed == protocol.calibration_seed:
        return "calibration"
    if seed in protocol.validation_seeds:
        return "validation"
    raise ValueError(f"T12.4a.1 seed is outside the frozen split: {seed}")


def _dataset_qa(
    states: Mapping[str, Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    split_states: dict[str, set[str]] = {}
    for split in ("train", "calibration", "validation", "all"):
        rows = [
            item for item in examples if split == "all" or item["split"] == split
        ]
        count = len(rows)
        relational = 0
        contexts = set()
        seeds = set()
        states_seen = set()
        for item in rows:
            state_id = str(item["source_state_id"])
            state = abstract_state_from_payload(states[state_id])
            action = GroundedAction(
                str(item["action"]["action_name"]),
                dict(item["action"].get("action_data", {}) or {}),
            )
            relational += int(any(encode_action_entity_relations(state, action)))
            contexts.add(tuple(sorted(dict(item["archive_context"]).items())))
            seeds.add(int(item["seed"]))
            states_seen.add(state_id)
        if split != "all":
            split_states[split] = states_seen
        output[split] = {
            "examples": count,
            "seeds": sorted(seeds),
            "semantic_changed_prevalence": (
                0.0
                if count == 0
                else sum(bool(item["semantic_changed"]) for item in rows) / count
            ),
            "novelty_prevalence": (
                0.0 if count == 0 else sum(bool(item["novel"]) for item in rows) / count
            ),
            "unique_actions": len({str(item["action_key"]) for item in rows}),
            "unique_states": len(states_seen),
            "unique_archive_contexts": len(contexts),
            "relational_feature_coverage": (
                0.0 if count == 0 else relational / count
            ),
        }
    for left, right in (
        ("train", "calibration"),
        ("train", "validation"),
        ("calibration", "validation"),
    ):
        overlap = split_states[left] & split_states[right]
        output[f"{left}_{right}_state_overlap"] = len(overlap)
        output[f"{left}_{right}_state_overlap_fraction"] = len(overlap) / max(
            1,
            len(split_states[right]),
        )
    return output


def _dataset_gate(qa: Mapping[str, Any], protocol: CalibrationProtocol) -> bool:
    minimums = {
        "train": protocol.minimum_training_examples,
        "calibration": protocol.minimum_calibration_examples,
        "validation": protocol.minimum_validation_examples,
    }
    expected_seeds = {
        "train": set(protocol.training_seeds),
        "calibration": {protocol.calibration_seed},
        "validation": set(protocol.validation_seeds),
    }
    for split, minimum_examples in minimums.items():
        metrics = dict(qa[split])
        if int(metrics["examples"]) < minimum_examples:
            return False
        if set(int(value) for value in metrics["seeds"]) != expected_seeds[split]:
            return False
        if int(metrics["unique_actions"]) < protocol.minimum_unique_actions:
            return False
        if int(metrics["unique_archive_contexts"]) < (
            protocol.minimum_unique_archive_contexts
        ):
            return False
        if float(metrics["relational_feature_coverage"]) < (
            protocol.minimum_relational_feature_coverage
        ):
            return False
        for label in ("semantic_changed_prevalence", "novelty_prevalence"):
            if not (
                protocol.minimum_label_prevalence
                <= float(metrics[label])
                <= protocol.maximum_label_prevalence
            ):
                return False
    return True


def collect_calibration_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: Any | None = None,
) -> dict[str, Any]:
    manifest = load_calibration_manifest(
        manifest_path,
        verify_code=env_factory is None,
    )
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.1 collection requires a clean scientific freeze")
    if not manifest.get("firewall", {}).get(
        "calibration_collection_authorized",
        False,
    ):
        raise ValueError("T12.4a.1 prospective collection is not authorized")
    protocol = CalibrationProtocol(**dict(manifest["protocol"]))
    shield_payload = _read_json(
        _resolve_manifest_path(str(manifest["shield"]["path"]))
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    states: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    game_id = str(manifest["game_id"])
    for seed in protocol.collection_seeds:
        run, shield, _ = run_neural_novelty_arm(
            game_id=game_id,
            seed=seed,
            sdk_call_budget=protocol.sdk_calls_per_seed,
            burst_schedule=protocol.burst_schedule,
            environments_dir=environments_dir,
            shield_payload=shield_payload,
            env_factory=env_factory,
            novelty_scorer=None,
            maximum_cells=protocol.maximum_cells,
        )
        archive_path = destination / game_id / str(seed) / "archive.json"
        archive_meta = _write_archive(
            archive_path,
            run.archive,
            storage_budget=storage,
        )
        excursions_path = destination / game_id / str(seed) / "excursions.json"
        _write_json_once(
            excursions_path,
            {
                "format_version": "sage-t12.4a.1-calibration-excursions-v1",
                "game_id": game_id,
                "seed": seed,
                "excursions": [item.to_dict() for item in run.excursions],
            },
            storage_budget=storage,
        )
        split = _split_for_seed(seed, protocol)
        seed_states, seed_examples = compile_archive_examples(
            run.archive,
            seed=seed,
            split=split,
        )
        for state_id, state_payload in seed_states.items():
            previous = states.setdefault(state_id, state_payload)
            if previous != state_payload:
                raise ValueError("T12.4a.1 cross-seed state signature collision")
        examples.extend(seed_examples)
        conditions.append(
            {
                "game_id": game_id,
                "seed": seed,
                "split": split,
                "metrics": run.metrics(),
                "shield_metrics": shield.metrics(),
                "archive": {
                    **archive_meta,
                    "seed": seed,
                    "arm": protocol.collection_arm,
                },
                "excursions": {
                    "path": str(excursions_path.resolve()),
                    "sha256": _file_sha256(excursions_path),
                },
            }
        )
    if len({item["example_id"] for item in examples}) != len(examples):
        raise ValueError("T12.4a.1 dataset contains duplicate example identities")
    qa = _dataset_qa(states, examples)
    dataset = seal_calibration_dataset(
        {
            "protocol_checksum": protocol.checksum,
            "manifest_checksum": manifest["manifest_checksum"],
            "label_contract": {
                "semantic_changed": "abstract_state_signature_delta",
                "novel": "target_cell_unseen_before_action",
                "context_timing": "strictly_pre_action",
                "calibration_fit": "calibration_seed_only",
                "confirmation": "validation_seeds_never_fit",
            },
            "source_seeds": list(protocol.collection_seeds),
            "states": states,
            "examples": examples,
            "qa": qa,
        }
    )
    dataset_path = destination / "calibration_dataset.sealed.json"
    _write_json_once(dataset_path, dataset, storage_budget=storage)
    collection_path = destination / "collection.json"
    _write_json_once(
        collection_path,
        {
            "format_version": "sage-t12.4a.1-calibration-collection-v1",
            "conditions": conditions,
        },
        storage_budget=storage,
    )
    sdk_calls = sum(int(item["metrics"]["sdk_calls"]) for item in conditions)
    minimum_replay = min(
        float(item["metrics"]["replay_exact_rate"]) for item in conditions
    )
    shield_vetoes = sum(
        int(item["shield_metrics"].get("vetoes", 0)) for item in conditions
    )
    metrics = {
        "sdk_calls": sdk_calls,
        "maximum_total_sdk_calls": protocol.maximum_total_sdk_calls,
        "minimum_replay_exact_rate": minimum_replay,
        "shield_vetoes": shield_vetoes,
        "dataset_qa": qa,
    }
    passed = bool(
        sdk_calls <= protocol.maximum_total_sdk_calls
        and minimum_replay >= protocol.minimum_collection_replay_exact_rate
        and shield_vetoes >= protocol.minimum_shield_vetoes
        and _dataset_gate(qa, protocol)
    )
    status = (
        "PASS_T12_4A_1_COLLECTION_GATE"
        if passed
        else "FAIL_T12_4A_1_COLLECTION_GATE"
    )
    report = {
        "format_version": "sage-t12.4a.1-calibration-collection-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "metrics": metrics,
        "conditions": conditions,
        "storage": storage.snapshot(),
    }
    report_path = destination / "collection_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = calibration_phase_receipt(
        manifest=manifest,
        phase="collect",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "collection": {
                "path": str(collection_path.resolve()),
                "sha256": _file_sha256(collection_path),
            },
            "dataset": {
                "path": str(dataset_path.resolve()),
                "sha256": _file_sha256(dataset_path),
                "dataset_checksum": dataset["dataset_checksum"],
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(
        destination / "collection_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def _examples(
    dataset: Mapping[str, Any],
    *,
    split: str,
) -> list[tuple[Any, GroundedAction, ArchiveContext, bool, bool, str, int]]:
    states = dict(dataset["states"])
    output = []
    for row in dataset["examples"]:
        if row["split"] != split:
            continue
        output.append(
            (
                abstract_state_from_payload(states[str(row["source_state_id"])]),
                GroundedAction(
                    str(row["action"]["action_name"]),
                    dict(row["action"].get("action_data", {}) or {}),
                ),
                ArchiveContext.from_dict(dict(row["archive_context"])),
                bool(row["semantic_changed"]),
                bool(row["novel"]),
                str(row["example_id"]),
                int(row["seed"]),
            )
        )
    return output


class PlattCalibrator(nn.Module):
    """Four-parameter monotone calibration for the two prediction heads."""

    def __init__(self) -> None:
        super().__init__()
        initial = math.log(math.expm1(1.0))
        self.raw_scale = nn.Parameter(torch.full((2,), initial))
        self.bias = nn.Parameter(torch.zeros(2))

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + 1e-4

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.scale + self.bias

    def parameters_payload(self) -> dict[str, list[float]]:
        return {
            "scale": [float(value) for value in self.scale.detach().tolist()],
            "bias": [float(value) for value in self.bias.detach().tolist()],
        }


def _logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(features).detach()


def _fit_calibrator(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    protocol: CalibrationProtocol,
    seed_offset: int,
) -> PlattCalibrator:
    torch.manual_seed(protocol.torch_seed + seed_offset)
    calibrator = PlattCalibrator().cpu()
    optimizer = torch.optim.Adam(
        calibrator.parameters(),
        lr=protocol.calibration_learning_rate,
    )
    for _ in range(protocol.calibration_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(calibrator(logits), targets)
        loss.backward()
        optimizer.step()
    return calibrator


def _probabilities(
    logits: torch.Tensor,
    calibrator: PlattCalibrator | None = None,
) -> list[tuple[float, float]]:
    with torch.no_grad():
        selected = logits if calibrator is None else calibrator(logits)
        values = torch.sigmoid(selected).tolist()
    return [(float(item[0]), float(item[1])) for item in values]


def _maximum_ece(
    predictions: Sequence[tuple[float, float]],
    targets: Sequence[tuple[float, float]],
) -> tuple[float, float, float]:
    change = ece([item[0] for item in predictions], [item[0] for item in targets])
    novelty = ece([item[1] for item in predictions], [item[1] for item in targets])
    return change, novelty, max(change, novelty)


def train_calibration_experiment(
    *,
    manifest_path: str | Path,
    collection_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = load_calibration_manifest(manifest_path)
    collection_receipt = load_calibration_receipt(
        collection_receipt_path,
        manifest=manifest,
        require_passed=True,
    )
    if collection_receipt.get("status") != "PASS_T12_4A_1_COLLECTION_GATE":
        raise ValueError("T12.4a.1 training requires a passed prospective collection")
    protocol = CalibrationProtocol(**dict(manifest["protocol"]))
    dataset_path = _resolve_manifest_path(
        str(collection_receipt["artifacts"]["dataset"]["path"])
    )
    dataset = load_calibration_dataset(dataset_path, protocol=protocol)
    if dataset["dataset_checksum"] != collection_receipt["artifacts"]["dataset"].get(
        "dataset_checksum"
    ):
        raise ValueError("T12.4a.1 dataset signature does not match receipt")
    training = _examples(dataset, split="train")
    calibration = _examples(dataset, split="calibration")
    validation = _examples(dataset, split="validation")
    if not training or not calibration or not validation:
        raise ValueError("T12.4a.1 train/calibration/validation split is empty")

    relational = RelationalNoveltyPredictor(
        seed=protocol.torch_seed,
        hidden_dim=protocol.hidden_dim,
    )
    torch.manual_seed(protocol.torch_seed + 1)
    legacy = ChangeNoveltyMLP(hidden_dim=protocol.hidden_dim).cpu()
    train_targets = torch.tensor(
        [[float(item[3]), float(item[4])] for item in training],
        dtype=torch.float32,
    )
    relational_train = torch.tensor(
        [encode_relational_state_action(item[0], item[1], item[2]) for item in training],
        dtype=torch.float32,
    )
    legacy_train = torch.tensor(
        [encode_state_action(item[0], item[1]) for item in training],
        dtype=torch.float32,
    )
    relational_updates = _fit(
        relational.model,
        relational_train,
        train_targets,
        protocol=protocol,
        seed_offset=0,
    )
    legacy_updates = _fit(
        legacy,
        legacy_train,
        train_targets,
        protocol=protocol,
        seed_offset=1,
    )

    calibration_targets = torch.tensor(
        [[float(item[3]), float(item[4])] for item in calibration],
        dtype=torch.float32,
    )
    relational_calibration_features = torch.tensor(
        [
            encode_relational_state_action(item[0], item[1], item[2])
            for item in calibration
        ],
        dtype=torch.float32,
    )
    legacy_calibration_features = torch.tensor(
        [encode_state_action(item[0], item[1]) for item in calibration],
        dtype=torch.float32,
    )
    relational_calibrator = _fit_calibrator(
        _logits(relational.model, relational_calibration_features),
        calibration_targets,
        protocol=protocol,
        seed_offset=10,
    )
    legacy_calibrator = _fit_calibrator(
        _logits(legacy, legacy_calibration_features),
        calibration_targets,
        protocol=protocol,
        seed_offset=11,
    )

    targets = [(float(item[3]), float(item[4])) for item in validation]
    relational_validation = torch.tensor(
        [
            encode_relational_state_action(item[0], item[1], item[2])
            for item in validation
        ],
        dtype=torch.float32,
    )
    legacy_validation = torch.tensor(
        [encode_state_action(item[0], item[1]) for item in validation],
        dtype=torch.float32,
    )
    state_shuffle = torch.tensor(
        [
            encode_relational_state_action(
                validation[(index + 1) % len(validation)][0],
                item[1],
                item[2],
            )
            for index, item in enumerate(validation)
        ],
        dtype=torch.float32,
    )
    context_shuffle = torch.tensor(
        [
            encode_relational_state_action(
                item[0],
                item[1],
                validation[(index + 1) % len(validation)][2],
            )
            for index, item in enumerate(validation)
        ],
        dtype=torch.float32,
    )
    relation_ablation = torch.tensor(
        [
            encode_relational_state_action(
                item[0],
                item[1],
                item[2],
                include_relations=False,
            )
            for item in validation
        ],
        dtype=torch.float32,
    )
    validation_logits = _logits(relational.model, relational_validation)
    predictions_uncalibrated = _probabilities(validation_logits)
    predictions = _probabilities(validation_logits, relational_calibrator)
    legacy_predictions = _probabilities(
        _logits(legacy, legacy_validation),
        legacy_calibrator,
    )
    state_shuffle_predictions = _probabilities(
        _logits(relational.model, state_shuffle),
        relational_calibrator,
    )
    context_shuffle_predictions = _probabilities(
        _logits(relational.model, context_shuffle),
        relational_calibrator,
    )
    relation_ablation_predictions = _probabilities(
        _logits(relational.model, relation_ablation),
        relational_calibrator,
    )
    priors, global_prior = _action_priors([item[:6] for item in training])
    baseline_predictions = [priors.get(item[1].key, global_prior) for item in validation]

    action_change_brier = _head_brier(baseline_predictions, targets, 0)
    action_novelty_brier = _head_brier(baseline_predictions, targets, 1)
    change_brier = _head_brier(predictions, targets, 0)
    novelty_brier = _head_brier(predictions, targets, 1)
    relational_mean = 0.5 * (change_brier + novelty_brier)
    uncalibrated_mean = 0.5 * (
        _head_brier(predictions_uncalibrated, targets, 0)
        + _head_brier(predictions_uncalibrated, targets, 1)
    )
    legacy_mean = 0.5 * (
        _head_brier(legacy_predictions, targets, 0)
        + _head_brier(legacy_predictions, targets, 1)
    )
    relation_ablation_mean = 0.5 * (
        _head_brier(relation_ablation_predictions, targets, 0)
        + _head_brier(relation_ablation_predictions, targets, 1)
    )
    change_ece, novelty_ece, maximum_ece = _maximum_ece(predictions, targets)
    _, _, uncalibrated_maximum_ece = _maximum_ece(
        predictions_uncalibrated,
        targets,
    )
    per_seed_ece = {}
    for seed in protocol.validation_seeds:
        indices = [index for index, item in enumerate(validation) if item[6] == seed]
        seed_predictions = [predictions[index] for index in indices]
        seed_targets = [targets[index] for index in indices]
        seed_change, seed_novelty, seed_maximum = _maximum_ece(
            seed_predictions,
            seed_targets,
        )
        per_seed_ece[str(seed)] = {
            "examples": len(indices),
            "change_ece": seed_change,
            "novelty_ece": seed_novelty,
            "maximum_ece": seed_maximum,
        }
    worst_seed_ece = max(item["maximum_ece"] for item in per_seed_ece.values())
    calibration_parameter_count = sum(
        parameter.numel() for parameter in relational_calibrator.parameters()
    )
    metrics = {
        "training_examples": len(training),
        "training_seed_count": len({item[6] for item in training}),
        "calibration_examples": len(calibration),
        "calibration_seed_count": len({item[6] for item in calibration}),
        "validation_examples": len(validation),
        "validation_seed_count": len({item[6] for item in validation}),
        "relational_parameter_count": relational.parameter_count,
        "calibration_parameter_count": calibration_parameter_count,
        "total_parameter_count": relational.parameter_count + calibration_parameter_count,
        "relational_optimizer_updates": relational_updates,
        "legacy_optimizer_updates": legacy_updates,
        "calibrator": relational_calibrator.parameters_payload(),
        "action_only_change_brier": action_change_brier,
        "action_only_novelty_brier": action_novelty_brier,
        "calibrated_change_brier": change_brier,
        "calibrated_novelty_brier": novelty_brier,
        "change_brier_gain": action_change_brier - change_brier,
        "novelty_brier_gain": action_novelty_brier - novelty_brier,
        "calibrated_mean_brier": relational_mean,
        "uncalibrated_mean_brier": uncalibrated_mean,
        "calibrated_brier_regression": relational_mean - uncalibrated_mean,
        "legacy_calibrated_mean_brier": legacy_mean,
        "legacy_mean_brier_improvement": legacy_mean - relational_mean,
        "state_shuffle_change_degradation": (
            _head_brier(state_shuffle_predictions, targets, 0) - change_brier
        ),
        "context_shuffle_novelty_degradation": (
            _head_brier(context_shuffle_predictions, targets, 1) - novelty_brier
        ),
        "relation_ablation_mean_brier": relation_ablation_mean,
        "relation_ablation_degradation": relation_ablation_mean - relational_mean,
        "uncalibrated_maximum_ece": uncalibrated_maximum_ece,
        "change_ece": change_ece,
        "novelty_ece": novelty_ece,
        "maximum_ece": maximum_ece,
        "calibration_ece_improvement": uncalibrated_maximum_ece - maximum_ece,
        "per_seed_ece": per_seed_ece,
        "worst_seed_maximum_ece": worst_seed_ece,
    }
    passed = bool(
        metrics["training_examples"] >= protocol.minimum_training_examples
        and metrics["training_seed_count"] == len(protocol.training_seeds)
        and metrics["calibration_examples"] >= protocol.minimum_calibration_examples
        and metrics["calibration_seed_count"] == 1
        and metrics["validation_examples"] >= protocol.minimum_validation_examples
        and metrics["validation_seed_count"] == len(protocol.validation_seeds)
        and metrics["total_parameter_count"] <= protocol.maximum_parameters
        and metrics["calibration_parameter_count"]
        <= protocol.maximum_calibration_parameters
        and metrics["change_brier_gain"] >= protocol.minimum_change_brier_gain
        and metrics["novelty_brier_gain"] >= protocol.minimum_novelty_brier_gain
        and metrics["legacy_mean_brier_improvement"]
        >= protocol.minimum_legacy_mean_brier_improvement
        and metrics["state_shuffle_change_degradation"]
        >= protocol.minimum_state_shuffle_change_degradation
        and metrics["context_shuffle_novelty_degradation"]
        >= protocol.minimum_context_shuffle_novelty_degradation
        and metrics["relation_ablation_degradation"]
        >= protocol.minimum_relation_ablation_degradation
        and metrics["calibration_ece_improvement"]
        >= protocol.minimum_calibration_ece_improvement
        and metrics["calibrated_brier_regression"]
        <= protocol.maximum_calibrated_brier_regression
        and metrics["maximum_ece"] <= protocol.maximum_pooled_ece
        and metrics["worst_seed_maximum_ece"] <= protocol.maximum_per_seed_ece
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    checkpoint_path = destination / "calibrated_relational_novelty_predictor.pt"
    buffer = io.BytesIO()
    torch.save(
        {
            "format_version": "sage-t12.4a.1-calibrated-predictor-v1",
            "base_model_state_dict": relational.model.state_dict(),
            "calibrator_state_dict": relational_calibrator.state_dict(),
            "hidden_dim": protocol.hidden_dim,
            "protocol_checksum": manifest["protocol_checksum"],
            "dataset_checksum": dataset["dataset_checksum"],
            "training_seeds": list(protocol.training_seeds),
            "calibration_seed": protocol.calibration_seed,
            "validation_seeds": list(protocol.validation_seeds),
        },
        buffer,
    )
    encoded = buffer.getvalue()
    storage.reserve(len(encoded))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as handle:
        handle.write(encoded)
    predictions_path = destination / "confirmation_predictions.json"
    _write_json_once(
        predictions_path,
        {
            "format_version": "sage-t12.4a.1-confirmation-predictions-v1",
            "rows": [
                {
                    "example_id": item[5],
                    "seed": item[6],
                    "target": list(targets[index]),
                    "uncalibrated_prediction": list(predictions_uncalibrated[index]),
                    "calibrated_prediction": list(predictions[index]),
                    "legacy_calibrated_prediction": list(legacy_predictions[index]),
                    "action_only_prediction": list(baseline_predictions[index]),
                }
                for index, item in enumerate(validation)
            ],
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_4A_1_CALIBRATION_GATE"
        if passed
        else "FAIL_T12_4A_1_CALIBRATION_GATE"
    )
    report = {
        "format_version": "sage-t12.4a.1-calibration-training-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "collection_receipt_checksum": collection_receipt["receipt_checksum"],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "calibration_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = calibration_phase_receipt(
        manifest=manifest,
        phase="train",
        passed=passed,
        status=status,
        metrics=metrics,
        parent_receipt=collection_receipt,
        artifacts={
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _file_sha256(checkpoint_path),
            },
            "dataset": dict(collection_receipt["artifacts"]["dataset"]),
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": _file_sha256(predictions_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(
        destination / "calibration_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def calibration_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_calibration_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_calibration_receipt(receipt_path, manifest=manifest)
    )
    freeze_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_1_FREEZE"
    )
    collection_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_1_COLLECTION_GATE"
    )
    calibration_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_1_CALIBRATION_GATE"
    )
    return {
        "format_version": "sage-t12.4a.1-calibration-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_4a_status": manifest["parent"]["receipt"]["status"],
        "parent_failure_class": manifest["parent"]["receipt"]["failure_class"],
        "receipt": (
            None
            if receipt is None
            else {
                "phase": receipt["phase"],
                "passed": receipt["passed"],
                "status": receipt["status"],
                "receipt_checksum": receipt["receipt_checksum"],
            }
        ),
        "next_phase_authorized": bool(
            freeze_passed or collection_passed or calibration_passed
        ),
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "calibration_collection_authorized": bool(
                manifest.get("scientific_claims_authorized", False)
            ),
            "calibration_training_authorized": collection_passed,
            "neural_active_evaluation_authorized": False,
            "option_extraction_authorized": False,
            "t12_4b_freeze_authorized": calibration_passed,
            "t12_5_freeze_authorized": False,
        },
    }


__all__ = [
    "PlattCalibrator",
    "calibration_experiment_status",
    "collect_calibration_experiment",
    "train_calibration_experiment",
]
