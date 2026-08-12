"""Prospective collection and offline representation test for SAGE.T12.4a."""

from __future__ import annotations

import io
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .archive import (
    GoExploreArchive,
    abstract_state_from_payload,
    abstract_state_to_payload,
)
from .contracts import GroundedAction
from .experiment import (
    RunStorageBudget,
    _file_sha256,
    _read_json,
    _write_json_once,
)
from .graph_experiment import _write_archive
from .neural_novelty_experiment import run_neural_novelty_arm
from .novelty import (
    ChangeNoveltyMLP,
    brier_score,
    encode_state_action,
    expected_calibration_error,
)
from .relational_novelty import (
    ArchiveContext,
    RelationalNoveltyPredictor,
    encode_action_entity_relations,
    encode_relational_state_action,
)
from .representation_protocol import (
    RepresentationProtocol,
    load_representation_dataset,
    load_representation_manifest,
    load_representation_receipt,
    representation_phase_receipt,
    seal_representation_dataset,
)


def _resolve_manifest_path(path: str) -> Path:
    candidate = Path(path)
    return (
        candidate
        if candidate.is_absolute()
        else Path(__file__).resolve().parents[3] / candidate
    )


def _split_for_seed(seed: int, protocol: RepresentationProtocol) -> str:
    if seed in protocol.training_seeds:
        return "train"
    if seed == protocol.validation_seed:
        return "validation"
    raise ValueError(f"T12.4a seed is outside the frozen split: {seed}")


def _context_from_counts(
    *,
    cell_id: str,
    action_key: str,
    legal_actions: int,
    seen_cells: set[str],
    cell_visits: Counter[str],
    cell_expansions: Counter[str],
    cell_actions: Mapping[tuple[str, str], list[int]],
    global_actions: Mapping[str, list[int]],
    global_edges: int,
) -> ArchiveContext:
    cell_action = cell_actions.get((cell_id, action_key), [0, 0, 0])
    global_action = global_actions.get(action_key, [0, 0, 0])
    tried_actions = sum(
        counts[0] > 0
        for (candidate_cell, _), counts in cell_actions.items()
        if candidate_cell == cell_id
    )
    return ArchiveContext(
        cell_visits=cell_visits[cell_id],
        action_attempts=cell_action[0],
        cell_expansions=cell_expansions[cell_id],
        unique_tried_actions=tried_actions,
        legal_actions=legal_actions,
        archive_cells=len(seen_cells),
        global_edges=global_edges,
        global_action_trials=global_action[0],
        global_action_changed=global_action[1],
        global_action_novel=global_action[2],
        cell_action_trials=cell_action[0],
        cell_action_changed=cell_action[1],
        cell_action_novel=cell_action[2],
    )


def compile_archive_examples(
    archive: GoExploreArchive,
    *,
    seed: int,
    split: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    edges = sorted(archive.edges.values(), key=lambda item: item.ordinal)
    if not edges:
        return {}, []
    seen_cells = {edges[0].source_cell_id}
    cell_visits: Counter[str] = Counter({edges[0].source_cell_id: 1})
    cell_expansions: Counter[str] = Counter()
    cell_actions: dict[tuple[str, str], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    global_actions: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    states: dict[str, dict[str, Any]] = {}
    examples = []
    for global_index, edge in enumerate(edges):
        source = archive.cells[edge.source_cell_id]
        target = archive.cells[edge.target_cell_id]
        if edge.source_cell_id not in seen_cells:
            raise ValueError("T12.4a edge source was not observed before execution")
        expected_novel = edge.target_cell_id not in seen_cells
        if expected_novel != bool(edge.novel):
            raise ValueError("T12.4a cannot reconstruct the pre-action novelty label")
        state_id = source.state.execution_signature
        payload = abstract_state_to_payload(source.state)
        previous = states.setdefault(state_id, payload)
        if previous != payload:
            raise ValueError("T12.4a state signature collision")
        context = _context_from_counts(
            cell_id=edge.source_cell_id,
            action_key=edge.action.key,
            legal_actions=len(source.legal_action_keys),
            seen_cells=seen_cells,
            cell_visits=cell_visits,
            cell_expansions=cell_expansions,
            cell_actions=cell_actions,
            global_actions=global_actions,
            global_edges=global_index,
        )
        semantic_changed = source.state.signature != target.state.signature
        example = {
            "seed": int(seed),
            "split": str(split),
            "ordinal": edge.ordinal,
            "edge_id": edge.edge_id,
            "source_state_id": state_id,
            "action": {
                "action_name": edge.action.action_name,
                "action_data": dict(edge.action.action_data),
            },
            "action_key": edge.action.key,
            "archive_context": context.to_dict(),
            "semantic_changed": semantic_changed,
            "novel": edge.novel,
        }
        example["example_id"] = representation_example_checksum(example)
        examples.append(example)

        cell_counts = cell_actions[(edge.source_cell_id, edge.action.key)]
        global_counts = global_actions[edge.action.key]
        for counts in (cell_counts, global_counts):
            counts[0] += 1
            counts[1] += int(semantic_changed)
            counts[2] += int(edge.novel)
        cell_expansions[edge.source_cell_id] += 1
        seen_cells.add(edge.target_cell_id)
        cell_visits[edge.target_cell_id] += 1
    return states, examples


def representation_example_checksum(payload: Mapping[str, Any]) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_qa(
    states: Mapping[str, Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("train", "validation", "all"):
        rows = [
            item for item in examples if split == "all" or item["split"] == split
        ]
        count = len(rows)
        relational = 0
        contexts = set()
        for item in rows:
            state = abstract_state_from_payload(states[str(item["source_state_id"])])
            action = GroundedAction(
                str(item["action"]["action_name"]),
                dict(item["action"].get("action_data", {}) or {}),
            )
            relational += int(any(encode_action_entity_relations(state, action)))
            contexts.add(tuple(sorted(dict(item["archive_context"]).items())))
        output[split] = {
            "examples": count,
            "semantic_changed_prevalence": (
                0.0
                if count == 0
                else sum(bool(item["semantic_changed"]) for item in rows) / count
            ),
            "novelty_prevalence": (
                0.0 if count == 0 else sum(bool(item["novel"]) for item in rows) / count
            ),
            "unique_actions": len({str(item["action_key"]) for item in rows}),
            "unique_states": len({str(item["source_state_id"]) for item in rows}),
            "unique_archive_contexts": len(contexts),
            "relational_feature_coverage": (
                0.0 if count == 0 else relational / count
            ),
        }
    train_states = {
        str(item["source_state_id"])
        for item in examples
        if item["split"] == "train"
    }
    validation_states = {
        str(item["source_state_id"])
        for item in examples
        if item["split"] == "validation"
    }
    output["train_validation_state_overlap"] = len(
        train_states & validation_states
    )
    output["validation_state_overlap_fraction"] = len(
        train_states & validation_states
    ) / max(1, len(validation_states))
    return output


def _dataset_gate(
    qa: Mapping[str, Any],
    protocol: RepresentationProtocol,
) -> bool:
    for split, minimum_examples in (
        ("train", protocol.minimum_training_examples),
        ("validation", protocol.minimum_validation_examples),
    ):
        metrics = dict(qa[split])
        if int(metrics["examples"]) < minimum_examples:
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


def collect_representation_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: Any | None = None,
) -> dict[str, Any]:
    manifest = load_representation_manifest(
        manifest_path,
        verify_code=env_factory is None,
    )
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a collection requires a clean scientific freeze")
    if not manifest.get("firewall", {}).get(
        "representation_collection_authorized", False
    ):
        raise ValueError("T12.4a prospective collection is not authorized")
    protocol = RepresentationProtocol(**dict(manifest["protocol"]))
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
                "format_version": "sage-t12.4a-representation-excursions-v1",
                "game_id": game_id,
                "seed": seed,
                "excursions": [item.to_dict() for item in run.excursions],
            },
            storage_budget=storage,
        )
        seed_states, seed_examples = compile_archive_examples(
            run.archive,
            seed=seed,
            split=_split_for_seed(seed, protocol),
        )
        for state_id, state_payload in seed_states.items():
            previous = states.setdefault(state_id, state_payload)
            if previous != state_payload:
                raise ValueError("T12.4a cross-seed state signature collision")
        examples.extend(seed_examples)
        conditions.append(
            {
                "game_id": game_id,
                "seed": seed,
                "split": _split_for_seed(seed, protocol),
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
        raise ValueError("T12.4a dataset contains duplicate example identities")
    qa = _dataset_qa(states, examples)
    dataset = seal_representation_dataset(
        {
            "protocol_checksum": protocol.checksum,
            "manifest_checksum": manifest["manifest_checksum"],
            "label_contract": {
                "semantic_changed": "abstract_state_signature_delta",
                "novel": "target_cell_unseen_before_action",
                "context_timing": "strictly_pre_action",
            },
            "source_seeds": list(protocol.collection_seeds),
            "states": states,
            "examples": examples,
            "qa": qa,
        }
    )
    dataset_path = destination / "representation_dataset.sealed.json"
    _write_json_once(dataset_path, dataset, storage_budget=storage)
    collection_path = destination / "collection.json"
    _write_json_once(
        collection_path,
        {
            "format_version": "sage-t12.4a-representation-collection-v1",
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
        "PASS_T12_4A_COLLECTION_GATE"
        if passed
        else "FAIL_T12_4A_COLLECTION_GATE"
    )
    report = {
        "format_version": "sage-t12.4a-representation-collection-report-v1",
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
    receipt = representation_phase_receipt(
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
) -> list[tuple[Any, GroundedAction, ArchiveContext, bool, bool, str]]:
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
            )
        )
    return output


def _fit(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    protocol: RepresentationProtocol,
    seed_offset: int,
) -> int:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=1e-4,
    )
    generator = torch.Generator().manual_seed(protocol.torch_seed + seed_offset)
    updates = 0
    for _ in range(protocol.training_epochs):
        order = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], protocol.batch_size):
            indices = order[start : start + protocol.batch_size]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indices])
            loss = F.binary_cross_entropy_with_logits(logits, targets[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
    return updates


def _predict(model: nn.Module, features: torch.Tensor) -> list[tuple[float, float]]:
    model.eval()
    with torch.no_grad():
        values = torch.sigmoid(model(features)).tolist()
    return [(float(item[0]), float(item[1])) for item in values]


def _action_priors(
    training: Sequence[tuple[Any, GroundedAction, ArchiveContext, bool, bool, str]],
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for _, action, _, changed, novel, _ in training:
        row = totals[action.key]
        row[0] += 1
        row[1] += int(changed)
        row[2] += int(novel)
    priors = {
        key: ((row[1] + 1) / (row[0] + 2), (row[2] + 1) / (row[0] + 2))
        for key, row in totals.items()
    }
    count = len(training)
    global_prior = (
        (sum(int(item[3]) for item in training) + 1) / (count + 2),
        (sum(int(item[4]) for item in training) + 1) / (count + 2),
    )
    return priors, global_prior


def _head_brier(
    predictions: Sequence[tuple[float, float]],
    targets: Sequence[tuple[float, float]],
    head: int,
) -> float:
    return brier_score(
        [item[head] for item in predictions],
        [item[head] for item in targets],
    )


def train_representation_experiment(
    *,
    manifest_path: str | Path,
    collection_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = load_representation_manifest(manifest_path)
    collection_receipt = load_representation_receipt(
        collection_receipt_path,
        manifest=manifest,
        require_passed=True,
    )
    if collection_receipt.get("status") != "PASS_T12_4A_COLLECTION_GATE":
        raise ValueError("T12.4a training requires a passed prospective collection")
    protocol = RepresentationProtocol(**dict(manifest["protocol"]))
    dataset_path = _resolve_manifest_path(
        str(collection_receipt["artifacts"]["dataset"]["path"])
    )
    dataset = load_representation_dataset(dataset_path, protocol=protocol)
    if dataset["dataset_checksum"] != collection_receipt["artifacts"]["dataset"].get(
        "dataset_checksum"
    ):
        raise ValueError("T12.4a dataset signature does not match collection receipt")
    training = _examples(dataset, split="train")
    validation = _examples(dataset, split="validation")
    if not training or not validation:
        raise ValueError("T12.4a train/validation split is empty")

    relational = RelationalNoveltyPredictor(
        seed=protocol.torch_seed,
        hidden_dim=protocol.hidden_dim,
    )
    torch.manual_seed(protocol.torch_seed + 1)
    legacy = ChangeNoveltyMLP(hidden_dim=protocol.hidden_dim).cpu()
    targets_train = torch.tensor(
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
        targets_train,
        protocol=protocol,
        seed_offset=0,
    )
    legacy_updates = _fit(
        legacy,
        legacy_train,
        targets_train,
        protocol=protocol,
        seed_offset=1,
    )

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
    predictions = _predict(relational.model, relational_validation)
    legacy_predictions = _predict(legacy, legacy_validation)
    state_shuffle_predictions = _predict(relational.model, state_shuffle)
    context_shuffle_predictions = _predict(relational.model, context_shuffle)
    relation_ablation_predictions = _predict(relational.model, relation_ablation)
    targets = [(float(item[3]), float(item[4])) for item in validation]
    priors, global_prior = _action_priors(training)
    baseline_predictions = [
        priors.get(item[1].key, global_prior) for item in validation
    ]

    action_change_brier = _head_brier(baseline_predictions, targets, 0)
    action_novelty_brier = _head_brier(baseline_predictions, targets, 1)
    change_brier = _head_brier(predictions, targets, 0)
    novelty_brier = _head_brier(predictions, targets, 1)
    relational_mean = 0.5 * (change_brier + novelty_brier)
    legacy_mean = 0.5 * (
        _head_brier(legacy_predictions, targets, 0)
        + _head_brier(legacy_predictions, targets, 1)
    )
    relation_ablation_mean = 0.5 * (
        _head_brier(relation_ablation_predictions, targets, 0)
        + _head_brier(relation_ablation_predictions, targets, 1)
    )
    change_ece = expected_calibration_error(
        [item[0] for item in predictions], [item[0] for item in targets]
    )
    novelty_ece = expected_calibration_error(
        [item[1] for item in predictions], [item[1] for item in targets]
    )
    metrics = {
        "training_examples": len(training),
        "validation_examples": len(validation),
        "relational_parameter_count": relational.parameter_count,
        "legacy_parameter_count": sum(item.numel() for item in legacy.parameters()),
        "relational_optimizer_updates": relational_updates,
        "legacy_optimizer_updates": legacy_updates,
        "action_only_change_brier": action_change_brier,
        "action_only_novelty_brier": action_novelty_brier,
        "relational_change_brier": change_brier,
        "relational_novelty_brier": novelty_brier,
        "change_brier_gain": action_change_brier - change_brier,
        "novelty_brier_gain": action_novelty_brier - novelty_brier,
        "relational_mean_brier": relational_mean,
        "legacy_mean_brier": legacy_mean,
        "legacy_mean_brier_improvement": legacy_mean - relational_mean,
        "state_shuffle_change_degradation": (
            _head_brier(state_shuffle_predictions, targets, 0) - change_brier
        ),
        "context_shuffle_novelty_degradation": (
            _head_brier(context_shuffle_predictions, targets, 1) - novelty_brier
        ),
        "relation_ablation_mean_brier": relation_ablation_mean,
        "relation_ablation_degradation": relation_ablation_mean - relational_mean,
        "change_ece": change_ece,
        "novelty_ece": novelty_ece,
        "maximum_ece": max(change_ece, novelty_ece),
    }
    passed = bool(
        metrics["training_examples"] >= protocol.minimum_training_examples
        and metrics["validation_examples"] >= protocol.minimum_validation_examples
        and metrics["relational_parameter_count"] <= protocol.maximum_parameters
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
        and metrics["maximum_ece"] <= protocol.maximum_ece
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    checkpoint_path = destination / "relational_novelty_predictor.pt"
    buffer = io.BytesIO()
    torch.save(
        {
            "format_version": "sage-t12.4a-relational-novelty-v1",
            "seed": relational.seed,
            "hidden_dim": relational.hidden_dim,
            "state_dict": relational.model.state_dict(),
            "metadata": metrics,
        },
        buffer,
    )
    encoded = buffer.getvalue()
    storage.reserve(len(encoded))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as handle:
        handle.write(encoded)
    predictions_path = destination / "validation_predictions.json"
    _write_json_once(
        predictions_path,
        {
            "format_version": "sage-t12.4a-representation-predictions-v1",
            "rows": [
                {
                    "example_id": item[5],
                    "prediction": list(predictions[index]),
                    "legacy_prediction": list(legacy_predictions[index]),
                    "action_only_prediction": list(baseline_predictions[index]),
                    "target": list(targets[index]),
                }
                for index, item in enumerate(validation)
            ],
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_4A_REPRESENTATION_GATE"
        if passed
        else "FAIL_T12_4A_REPRESENTATION_GATE"
    )
    report = {
        "format_version": "sage-t12.4a-representation-training-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "collection_receipt_checksum": collection_receipt["receipt_checksum"],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "representation_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = representation_phase_receipt(
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
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": _file_sha256(predictions_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "dataset": dict(collection_receipt["artifacts"]["dataset"]),
        },
    )
    _write_json_once(
        destination / "representation_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def representation_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_representation_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_representation_receipt(receipt_path, manifest=manifest)
    )
    freeze_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_FREEZE"
    )
    collection_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_COLLECTION_GATE"
    )
    representation_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4A_REPRESENTATION_GATE"
    )
    return {
        "format_version": "sage-t12.4a-representation-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_4_status": manifest["parent"]["receipt"]["status"],
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
        "next_phase_authorized": (
            freeze_passed or collection_passed or representation_passed
        ),
        "firewall": {
            **dict(manifest["firewall"]),
            "representation_training_authorized": collection_passed,
            "t12_4b_freeze_authorized": representation_passed,
        },
    }


__all__ = [
    "collect_representation_experiment",
    "compile_archive_examples",
    "representation_experiment_status",
    "train_representation_experiment",
]
