"""Preflight and evaluate the SAGE12 V3 action-target effect pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from theory.live_transition_loop import build_observation
from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION

from .action_target_collection import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    load_frozen_manifest,
)
from .action_target_data import (
    EFFECT_LABELS,
    PROJECTION_LADDER,
    ActionTargetTrace,
    feature_row,
    resolve_action_target,
    validate_model_projection,
)
from .constrained_pilot import FrozenQwenEmbedder


PREFLIGHT_FORMAT_VERSION = "sage12-action-target-preflight-v3"
PROJECTION_FREEZE_FORMAT_VERSION = "sage12-action-target-projection-freeze-v3"
RESULT_FORMAT_VERSION = "sage12-action-target-result-v3"
PREDICTION_FORMAT_VERSION = "sage12-action-target-prediction-v3"


class _ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_probability(self, rows: int) -> np.ndarray:
        return np.full(rows, self.probability, dtype=np.float64)


def run_source_train_preflight(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    traces = _load_traces(destination, "source_train")
    if tuple(sorted({row.game_id for row in traces})) != tuple(
        sorted(SOURCE_TRAIN)
    ):
        raise ValueError("V3 preflight requires exactly the source-training games")
    for row in traces:
        validate_model_projection(row, "full")

    action_identity = _identity_probe(
        [_action_only_row(row) for row in traces],
        [row.game_id for row in traces],
    )
    projection_audits: dict[str, Any] = {}
    selected_projection: str | None = None
    maximum_gain = float(
        frozen["gates"]["maximum_identity_gain_over_selected_action"]
    )
    for projection in PROJECTION_LADDER:
        state_rows = [
            feature_row(row, projection, include_action=False) for row in traces
        ]
        combined_rows = [feature_row(row, projection) for row in traces]
        state_identity = _identity_probe(
            state_rows, [row.game_id for row in traces]
        )
        combined_identity = _identity_probe(
            combined_rows, [row.game_id for row in traces]
        )
        conditional_gain = (
            combined_identity["accuracy"] - action_identity["accuracy"]
        )
        projection_audits[projection] = {
            "state_only": state_identity,
            "action_plus_state": combined_identity,
            "conditional_identity_gain": conditional_gain,
            "passes": conditional_gain <= maximum_gain,
        }
        if selected_projection is None and conditional_gain <= maximum_gain:
            selected_projection = projection

    model_selection: dict[str, Any] = {}
    selected_model: str | None = None
    if selected_projection is not None:
        targets, masks = _target_arrays(traces)
        candidates = ("logistic", "gradient_boosting")
        for model_name in candidates:
            probabilities = _logo_probabilities(
                traces,
                projection=selected_projection,
                model_name=model_name,
            )
            predictions = probabilities >= 0.5
            model_selection[model_name] = _multilabel_metrics(
                targets, predictions, probabilities, masks
            )
        selected_model = max(
            candidates,
            key=lambda name: (
                model_selection[name]["macro_f1"],
                name == "logistic",
            ),
        )

    quality = _data_quality(traces)
    payload: dict[str, Any] = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "status": (
            "PASS_SOURCE_TRAIN_PREFLIGHT"
            if selected_projection is not None
            else "LEAKAGE_FAIL"
        ),
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "source_train_rows": len(traces),
        "source_train_games": list(SOURCE_TRAIN),
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
        "action_only_identity": action_identity,
        "projection_audits": projection_audits,
        "selected_projection": selected_projection,
        "model_selection": model_selection,
        "selected_structured_model": selected_model,
        "decision_threshold": 0.5,
        "data_quality": quality,
    }
    payload["preflight_checksum"] = _payload_checksum(payload)
    _write_json_atomic(destination / "source_train_preflight.json", payload)

    freeze = {
        "format_version": PROJECTION_FREEZE_FORMAT_VERSION,
        "status": (
            "FROZEN_BEFORE_SOURCE_VALIDATION"
            if selected_projection is not None
            else "LEAKAGE_FAIL"
        ),
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "source_train_preflight_checksum": payload["preflight_checksum"],
        "selected_projection": selected_projection,
        "selected_structured_model": selected_model,
        "decision_threshold": 0.5,
        "source_validation_metrics_seen": False,
        "world_model_fit_authorized": False,
    }
    freeze["projection_freeze_checksum"] = _payload_checksum(freeze)
    _write_json_atomic(destination / "projection_freeze.json", freeze)
    return payload


def run_evaluation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    projection_freeze = _read_json(destination / "projection_freeze.json")
    _verify_embedded_checksum(
        projection_freeze, "projection_freeze_checksum"
    )
    if projection_freeze.get("status") != "FROZEN_BEFORE_SOURCE_VALIDATION":
        raise RuntimeError("V3 validation is not authorized by the projection freeze")
    train = _load_traces(destination, "source_train")
    validation = _load_traces(destination, "source_validation")
    if tuple(sorted({row.game_id for row in validation})) != tuple(
        sorted(SOURCE_VALIDATION)
    ):
        raise ValueError("V3 evaluation requires exactly source-validation games")
    projection = str(projection_freeze["selected_projection"])
    model_name = str(projection_freeze["selected_structured_model"])
    threshold = float(projection_freeze["decision_threshold"])

    train_targets, train_masks = _target_arrays(train)
    validation_targets, validation_masks = _target_arrays(validation)
    train_rows = [feature_row(row, projection) for row in train]
    validation_rows = [feature_row(row, projection) for row in validation]
    action_train = [_action_only_row(row) for row in train]
    action_validation = [_action_only_row(row) for row in validation]

    structured_models, structured_vectorizer = _fit_models(
        train_rows,
        train_targets,
        train_masks,
        model_name=model_name,
    )
    structured_probabilities = _predict_models(
        structured_models,
        structured_vectorizer,
        validation_rows,
    )
    structured_predictions = structured_probabilities >= threshold

    action_models, action_vectorizer = _fit_models(
        action_train,
        train_targets,
        train_masks,
        model_name="logistic",
    )
    action_probabilities = _predict_models(
        action_models,
        action_vectorizer,
        action_validation,
    )
    action_predictions = action_probabilities >= threshold
    template_predictions = _template_predictions(validation)
    template_probabilities = template_predictions.astype(np.float64)

    shuffled_feature_maps = _shuffle_target_feature_maps(
        validation, projection
    )
    shuffled_rows = [
        _encode_raw_feature_row(item) for item in shuffled_feature_maps
    ]
    shuffle_probabilities = _predict_models(
        structured_models,
        structured_vectorizer,
        shuffled_rows,
    )
    shuffle_predictions = shuffle_probabilities >= threshold

    action_shuffled_rows = _shuffle_action_rows(validation, projection)
    action_shuffle_probabilities = _predict_models(
        structured_models,
        structured_vectorizer,
        action_shuffled_rows,
    )
    action_shuffle_predictions = action_shuffle_probabilities >= threshold

    label_permutation_probabilities = _label_permutation_control(
        train_rows,
        train_targets,
        train_masks,
        validation_rows,
        model_name=model_name,
    )
    label_permutation_predictions = (
        label_permutation_probabilities >= threshold
    )

    metrics = {
        "structured": _multilabel_metrics(
            validation_targets,
            structured_predictions,
            structured_probabilities,
            validation_masks,
        ),
        "action_only": _multilabel_metrics(
            validation_targets,
            action_predictions,
            action_probabilities,
            validation_masks,
        ),
        "deterministic_template": _multilabel_metrics(
            validation_targets,
            template_predictions,
            template_probabilities,
            validation_masks,
        ),
        "target_shuffle": _multilabel_metrics(
            validation_targets,
            shuffle_predictions,
            shuffle_probabilities,
            validation_masks,
        ),
        "action_shuffle": _multilabel_metrics(
            validation_targets,
            action_shuffle_predictions,
            action_shuffle_probabilities,
            validation_masks,
        ),
        "label_permutation": _multilabel_metrics(
            validation_targets,
            label_permutation_predictions,
            label_permutation_probabilities,
            validation_masks,
        ),
    }
    baseline_name = max(
        ("action_only", "deterministic_template"),
        key=lambda name: metrics[name]["macro_f1"],
    )
    baseline_predictions = (
        action_predictions
        if baseline_name == "action_only"
        else template_predictions
    )
    gain = (
        metrics["structured"]["macro_f1"]
        - metrics[baseline_name]["macro_f1"]
    )
    shuffle_degradation = (
        metrics["structured"]["macro_f1"]
        - metrics["target_shuffle"]["macro_f1"]
    )
    bootstrap = _bootstrap_gain(
        validation,
        validation_targets,
        validation_masks,
        structured_predictions,
        baseline_predictions,
        samples=int(frozen["evaluation"]["bootstrap_samples"]),
        seed=int(frozen["training"]["random_state"]),
    )
    per_game = _per_game_metrics(
        validation,
        validation_targets,
        validation_masks,
        structured_predictions,
        structured_probabilities,
        action_predictions,
        action_probabilities,
        template_predictions,
        template_probabilities,
    )

    output_metrics, prediction_rows = _render_and_ground_predictions(
        validation,
        structured_predictions,
        structured_probabilities,
    )

    qwen = _evaluate_qwen(
        frozen=frozen,
        train=train,
        validation=validation,
        projection=projection,
        train_targets=train_targets,
        train_masks=train_masks,
        validation_targets=validation_targets,
        validation_masks=validation_masks,
        shuffled_feature_maps=shuffled_feature_maps,
    )
    metrics["qwen_ablation"] = qwen["metrics"]
    metrics["qwen_target_shuffle"] = qwen["shuffle_metrics"]

    train_quality = _data_quality(train)
    validation_quality = _data_quality(validation)
    gates = _evaluate_gates(
        frozen=frozen,
        train_quality=train_quality,
        validation_quality=validation_quality,
        metrics=metrics,
        baseline_name=baseline_name,
        gain=gain,
        shuffle_degradation=shuffle_degradation,
        bootstrap=bootstrap,
        per_game=per_game,
        output_metrics=output_metrics,
        projection_freeze=projection_freeze,
    )
    passed = all(gates.values())
    result: dict[str, Any] = {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "all_gates_passed": passed,
        "world_model_fit_authorized": passed,
        "qwen_required_for_promotion": False,
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "projection_freeze_checksum": projection_freeze[
            "projection_freeze_checksum"
        ],
        "selected_projection": projection,
        "selected_structured_model": model_name,
        "rows": {
            "source_train": len(train),
            "source_validation": len(validation),
        },
        "train_quality": train_quality,
        "validation_quality": validation_quality,
        "stronger_baseline": baseline_name,
        "primary_macro_f1_gain": gain,
        "target_shuffle_degradation": shuffle_degradation,
        "bootstrap_gain": bootstrap,
        "metrics": metrics,
        "per_game": per_game,
        "output_contract": output_metrics,
        "qwen": {
            key: value
            for key, value in qwen.items()
            if key not in {"metrics", "shuffle_metrics", "probabilities"}
        },
        "gates": gates,
        "runtime": _runtime_metadata(),
        "firewall": {
            "source_only": True,
            "holdout_opened": False,
            "historical_opened": False,
            "ar25_opened": False,
            "validation_tuning": False,
        },
    }
    result["result_checksum"] = _payload_checksum(result)
    _write_json_atomic(destination / "pilot_result.json", result)
    _write_jsonl_atomic(destination / "predictions.jsonl", prediction_rows)
    return result


def _evaluate_qwen(
    *,
    frozen: Mapping[str, Any],
    train: Sequence[ActionTargetTrace],
    validation: Sequence[ActionTargetTrace],
    projection: str,
    train_targets: np.ndarray,
    train_masks: np.ndarray,
    validation_targets: np.ndarray,
    validation_masks: np.ndarray,
    shuffled_feature_maps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_prompts = [_semantic_prompt(row.model_features(projection)) for row in train]
    validation_prompts = [
        _semantic_prompt(row.model_features(projection)) for row in validation
    ]
    shuffled_prompts = [
        _semantic_prompt(row) for row in shuffled_feature_maps
    ]
    all_prompts = list(dict.fromkeys(train_prompts + validation_prompts + shuffled_prompts))
    model = frozen["qwen_ablation"]
    embedder = FrozenQwenEmbedder(
        model_path=str(model["path"]),
        device=str(model["device"]),
        batch_size=int(model["batch_size"]),
        maximum_input_tokens=int(model["maximum_input_tokens"]),
    )
    vectors, timing = embedder.encode(all_prompts)
    vector_by_prompt = {
        prompt: vectors[index] for index, prompt in enumerate(all_prompts)
    }
    train_matrix = np.stack([vector_by_prompt[item] for item in train_prompts])
    validation_matrix = np.stack(
        [vector_by_prompt[item] for item in validation_prompts]
    )
    shuffled_matrix = np.stack(
        [vector_by_prompt[item] for item in shuffled_prompts]
    )
    models = _fit_matrix_models(
        train_matrix,
        train_targets,
        train_masks,
        model_name="logistic",
    )
    probabilities = _predict_matrix_models(models, validation_matrix)
    shuffle_probabilities = _predict_matrix_models(models, shuffled_matrix)
    predictions = probabilities >= 0.5
    shuffle_predictions = shuffle_probabilities >= 0.5
    return {
        "metrics": _multilabel_metrics(
            validation_targets,
            predictions,
            probabilities,
            validation_masks,
        ),
        "shuffle_metrics": _multilabel_metrics(
            validation_targets,
            shuffle_predictions,
            shuffle_probabilities,
            validation_masks,
        ),
        "probabilities": probabilities,
        "embedding": timing,
        "unique_prompts": len(all_prompts),
        "prompt_checksum": hashlib.sha256(
            "\n".join(all_prompts).encode("utf-8")
        ).hexdigest(),
        "embedding_checksum": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "model_name": model["name"],
        "model_weights_sha256": model["weights_sha256"],
    }


def _semantic_prompt(features: Mapping[str, Any]) -> str:
    lines = [
        "Predict abstract one-step effects for this action-target relation.",
    ]
    for key, value in sorted(features.items()):
        lines.append(f"{key}={value}")
    lines.append("effects=" + ",".join(EFFECT_LABELS))
    return "\n".join(lines)


def _fit_models(
    rows: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    masks: np.ndarray,
    *,
    model_name: str,
) -> tuple[list[Any], Any]:
    from sklearn.feature_extraction import DictVectorizer

    vectorizer = DictVectorizer(sparse=False)
    matrix = np.asarray(vectorizer.fit_transform(rows), dtype=np.float64)
    models = _fit_matrix_models(matrix, targets, masks, model_name=model_name)
    return models, vectorizer


def _fit_matrix_models(
    matrix: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    *,
    model_name: str,
) -> list[Any]:
    models = []
    for index, _label in enumerate(EFFECT_LABELS):
        eligible = masks[:, index]
        labels = targets[eligible, index]
        features = matrix[eligible]
        if labels.size == 0:
            models.append(_ConstantModel(0.0))
            continue
        unique = np.unique(labels)
        if unique.size < 2:
            models.append(_ConstantModel(float(unique[0])))
            continue
        if model_name == "logistic":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=12,
                solver="liblinear",
            )
            model.fit(features, labels)
        elif model_name == "gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.utils.class_weight import compute_sample_weight

            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=3,
                max_iter=100,
                random_state=12,
                l2_regularization=1.0,
            )
            weights = compute_sample_weight("balanced", labels)
            model.fit(features, labels, sample_weight=weights)
        else:
            raise ValueError(f"unknown V3 model: {model_name}")
        models.append(model)
    return models


def _predict_models(
    models: Sequence[Any],
    vectorizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    matrix = np.asarray(vectorizer.transform(rows), dtype=np.float64)
    return _predict_matrix_models(models, matrix)


def _predict_matrix_models(
    models: Sequence[Any], matrix: np.ndarray
) -> np.ndarray:
    columns = []
    for model in models:
        if isinstance(model, _ConstantModel):
            column = model.predict_probability(matrix.shape[0])
        else:
            probabilities = model.predict_proba(matrix)
            classes = list(model.classes_)
            column = probabilities[:, classes.index(1)]
        columns.append(np.asarray(column, dtype=np.float64))
    return np.stack(columns, axis=1)


def _logo_probabilities(
    traces: Sequence[ActionTargetTrace],
    *,
    projection: str,
    model_name: str,
) -> np.ndarray:
    targets, masks = _target_arrays(traces)
    rows = [feature_row(row, projection) for row in traces]
    probabilities = np.zeros_like(targets, dtype=np.float64)
    games = np.asarray([row.game_id for row in traces])
    for game in sorted(set(games)):
        train_indices = games != game
        test_indices = games == game
        fold_rows = [row for row, keep in zip(rows, train_indices) if keep]
        held_rows = [row for row, keep in zip(rows, test_indices) if keep]
        models, vectorizer = _fit_models(
            fold_rows,
            targets[train_indices],
            masks[train_indices],
            model_name=model_name,
        )
        probabilities[test_indices] = _predict_models(
            models, vectorizer, held_rows
        )
    return probabilities


def _identity_probe(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, float]:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.multiclass import OneVsRestClassifier

    matrix = DictVectorizer(sparse=True).fit_transform(rows)
    label_array = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=12,
            solver="liblinear",
        )
    )
    scores = cross_val_score(model, matrix, label_array, cv=splitter)
    majority = max(Counter(labels).values()) / len(labels)
    return {
        "accuracy": float(np.mean(scores)),
        "fold_std": float(np.std(scores)),
        "majority_accuracy": float(majority),
        "gain_over_majority": float(np.mean(scores) - majority),
    }


def _target_arrays(
    traces: Sequence[ActionTargetTrace],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [
            [int(row.effects.labels[label]) for label in EFFECT_LABELS]
            for row in traces
        ],
        dtype=np.int8,
    )
    masks = np.asarray(
        [
            [bool(row.effects.applicable[label]) for label in EFFECT_LABELS]
            for row in traces
        ],
        dtype=bool,
    )
    return targets, masks


def _multilabel_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    per_label: dict[str, Any] = {}
    f1_values = []
    recall_values = []
    ap_values = []
    ece_values = []
    for index, label in enumerate(EFFECT_LABELS):
        eligible = masks[:, index]
        y_true = targets[eligible, index]
        y_pred = predictions[eligible, index]
        y_prob = probabilities[eligible, index]
        if y_true.size == 0:
            row = {
                "applicable": 0,
                "positives": 0,
                "negatives": 0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "average_precision": 0.0,
                "ece": 0.0,
            }
        else:
            ap = (
                float(average_precision_score(y_true, y_prob))
                if np.unique(y_true).size > 1
                else float(np.mean(y_true))
            )
            row = {
                "applicable": int(y_true.size),
                "positives": int(np.sum(y_true)),
                "negatives": int(y_true.size - np.sum(y_true)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "average_precision": ap,
                "ece": _ece(y_true, y_prob),
            }
        per_label[label] = row
        f1_values.append(row["f1"])
        recall_values.append(row["recall"])
        ap_values.append(row["average_precision"])
        ece_values.append(row["ece"])
    return {
        "macro_f1": float(statistics.fmean(f1_values)),
        "macro_recall": float(statistics.fmean(recall_values)),
        "macro_average_precision": float(statistics.fmean(ap_values)),
        "macro_ece": float(statistics.fmean(ece_values)),
        "per_label": per_label,
    }


def _ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    if targets.size == 0:
        return 0.0
    total = targets.size
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            mask = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if not np.any(mask):
            continue
        value += (
            np.sum(mask)
            / total
            * abs(float(np.mean(targets[mask])) - float(np.mean(probabilities[mask])))
        )
    return float(value)


def _template_predictions(
    traces: Sequence[ActionTargetTrace],
) -> np.ndarray:
    rows = []
    for trace in traces:
        anchor = trace.anchor
        rows.append(
            [
                anchor.action_family == "move" and anchor.path_status == "open",
                anchor.kind == "clicked_empty",
                anchor.kind == "clicked_object",
                anchor.kind == "move_destination" and anchor.occupied,
            ]
        )
    return np.asarray(rows, dtype=bool)


def _shuffle_target_rows(
    traces: Sequence[ActionTargetTrace], projection: str
) -> list[dict[str, Any]]:
    return [
        _encode_raw_feature_row(item)
        for item in _shuffle_target_feature_maps(traces, projection)
    ]


def _shuffle_target_feature_maps(
    traces: Sequence[ActionTargetTrace], projection: str
) -> list[dict[str, Any]]:
    raw = [row.model_features(projection) for row in traces]
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, trace in enumerate(traces):
        groups[
            (trace.game_id, trace.selected_action_name, trace.anchor.kind)
        ].append(index)
    shuffled = [dict(item) for item in raw]
    action_keys = {
        "selected_action_name",
        "action_family",
        "requested_direction",
        "anchor_kind",
    }
    for key, indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        offset = 1 + int(
            hashlib.sha256(str(key).encode()).hexdigest()[:8], 16
        ) % (len(indices) - 1)
        rotated = indices[offset:] + indices[:offset]
        for target_index, source_index in zip(indices, rotated):
            for feature_key in list(shuffled[target_index]):
                if feature_key not in action_keys:
                    shuffled[target_index].pop(feature_key)
            for feature_key, value in raw[source_index].items():
                if feature_key not in action_keys:
                    shuffled[target_index][feature_key] = value
    return shuffled


def _shuffle_action_rows(
    traces: Sequence[ActionTargetTrace], projection: str
) -> list[dict[str, Any]]:
    raw = [row.model_features(projection) for row in traces]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, trace in enumerate(traces):
        groups[trace.game_id].append(index)
    shuffled = [dict(item) for item in raw]
    action_keys = (
        "selected_action_name",
        "action_family",
        "requested_direction",
    )
    for game, indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        rng = random.Random(f"sage12-v3-action-shuffle:{game}")
        sources = list(indices)
        rng.shuffle(sources)
        for target_index, source_index in zip(indices, sources):
            for key in action_keys:
                shuffled[target_index][key] = raw[source_index][key]
    return [_encode_raw_feature_row(item) for item in shuffled]


def _encode_raw_feature_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    encoded = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            encoded[key] = int(value)
        elif isinstance(value, (int, float)):
            encoded[key] = value
        else:
            encoded[f"{key}:{value}"] = 1
    return encoded


def _label_permutation_control(
    train_rows: Sequence[Mapping[str, Any]],
    train_targets: np.ndarray,
    train_masks: np.ndarray,
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
) -> np.ndarray:
    rng = np.random.default_rng(12)
    permuted = np.array(train_targets, copy=True)
    for index in range(len(EFFECT_LABELS)):
        eligible = np.flatnonzero(train_masks[:, index])
        values = np.array(permuted[eligible, index], copy=True)
        rng.shuffle(values)
        permuted[eligible, index] = values
    models, vectorizer = _fit_models(
        train_rows,
        permuted,
        train_masks,
        model_name=model_name,
    )
    return _predict_models(models, vectorizer, validation_rows)


def _bootstrap_gain(
    traces: Sequence[ActionTargetTrace],
    targets: np.ndarray,
    masks: np.ndarray,
    structured: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups: dict[str, np.ndarray] = {}
    game_ids = np.asarray([row.game_id for row in traces])
    for game in sorted(set(game_ids)):
        groups[game] = np.flatnonzero(game_ids == game)
    values = []
    unit_prob = structured.astype(np.float64)
    baseline_prob = baseline.astype(np.float64)
    for _ in range(samples):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()]
        )
        left = _multilabel_metrics(
            targets[sampled],
            structured[sampled],
            unit_prob[sampled],
            masks[sampled],
        )["macro_f1"]
        right = _multilabel_metrics(
            targets[sampled],
            baseline[sampled],
            baseline_prob[sampled],
            masks[sampled],
        )["macro_f1"]
        values.append(left - right)
    return {
        "samples": int(samples),
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _per_game_metrics(
    traces: Sequence[ActionTargetTrace],
    targets: np.ndarray,
    masks: np.ndarray,
    structured_predictions: np.ndarray,
    structured_probabilities: np.ndarray,
    action_predictions: np.ndarray,
    action_probabilities: np.ndarray,
    template_predictions: np.ndarray,
    template_probabilities: np.ndarray,
) -> dict[str, Any]:
    games = np.asarray([row.game_id for row in traces])
    payload = {}
    for game in SOURCE_VALIDATION:
        selected = games == game
        structured = _multilabel_metrics(
            targets[selected],
            structured_predictions[selected],
            structured_probabilities[selected],
            masks[selected],
        )
        action = _multilabel_metrics(
            targets[selected],
            action_predictions[selected],
            action_probabilities[selected],
            masks[selected],
        )
        template = _multilabel_metrics(
            targets[selected],
            template_predictions[selected],
            template_probabilities[selected],
            masks[selected],
        )
        baseline_name, baseline = max(
            (("action_only", action), ("deterministic_template", template)),
            key=lambda item: item[1]["macro_f1"],
        )
        payload[game] = {
            "rows": int(np.sum(selected)),
            "structured": structured,
            "action_only": action,
            "deterministic_template": template,
            "stronger_baseline": baseline_name,
            "gain": structured["macro_f1"] - baseline["macro_f1"],
        }
    return payload


def _render_and_ground_predictions(
    traces: Sequence[ActionTargetTrace],
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    valid_json = 0
    support_zero = 0
    grounded = 0
    emitted = 0
    rows = []
    for row_index, trace in enumerate(traces):
        hypotheses = []
        for label_index, label in enumerate(EFFECT_LABELS):
            if not predictions[row_index, label_index]:
                continue
            emitted += 1
            hypotheses.append(
                {
                    "hypothesis_id": (
                        f"v3_{trace.trace_digest[:10]}_{label}"
                    ),
                    "action_name": trace.selected_action_name,
                    "action_data": dict(trace.selected_action_data),
                    "anchor_kind": trace.anchor.kind,
                    "effect": label,
                    "confidence": float(probabilities[row_index, label_index]),
                    "support": 0,
                }
            )
        raw = json.dumps({"hypotheses": hypotheses}, sort_keys=True)
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict) and isinstance(
                decoded.get("hypotheses"), list
            ):
                valid_json += len(hypotheses)
        except json.JSONDecodeError:
            decoded = {"hypotheses": []}
        for hypothesis in decoded["hypotheses"]:
            if hypothesis.get("support") == 0:
                support_zero += 1
            if _hypothesis_grounded(trace, hypothesis):
                grounded += 1
        rows.append(
            {
                "format_version": PREDICTION_FORMAT_VERSION,
                "game_id": trace.game_id,
                "trace_digest": trace.trace_digest,
                "targets": {
                    label: bool(trace.effects.labels[label])
                    if trace.effects.applicable[label]
                    else None
                    for label in EFFECT_LABELS
                },
                "probabilities": {
                    label: float(probabilities[row_index, index])
                    for index, label in enumerate(EFFECT_LABELS)
                },
                "hypotheses_json": raw,
            }
        )
    denominator = max(1, emitted)
    return {
        "emitted_hypotheses": emitted,
        "strict_json_validity": valid_json / denominator,
        "support_zero_rate": support_zero / denominator,
        "grounded_hypothesis_rate": grounded / denominator,
    }, rows


def _hypothesis_grounded(
    trace: ActionTargetTrace, hypothesis: Mapping[str, Any]
) -> bool:
    if str(hypothesis.get("action_name")) not in trace.available_action_names:
        return False
    if dict(hypothesis.get("action_data", {})) != dict(trace.selected_action_data):
        return False
    observation = build_observation(
        trace.frame_before,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_before,
        levels_completed=trace.levels_completed_before,
        infer_players=True,
    )
    anchor = resolve_action_target(
        observation,
        trace.selected_action_name,
        trace.selected_action_data,
    )
    return str(hypothesis.get("anchor_kind")) == anchor.kind


def _data_quality(traces: Sequence[ActionTargetTrace]) -> dict[str, Any]:
    per_label = {}
    for label in EFFECT_LABELS:
        applicable = [row for row in traces if row.effects.applicable[label]]
        positives = sum(row.effects.labels[label] for row in applicable)
        games_with_positive = len(
            {
                row.game_id
                for row in applicable
                if row.effects.labels[label]
            }
        )
        per_label[label] = {
            "applicable": len(applicable),
            "positives": int(positives),
            "negatives": int(len(applicable) - positives),
            "games_with_positive": games_with_positive,
        }
    ambiguous_rows = [
        row for row in traces if bool(row.effects.ambiguity_reasons)
    ]
    per_game = {}
    for game in sorted({row.game_id for row in traces}):
        rows = [row for row in traces if row.game_id == game]
        clean = sum(not row.effects.ambiguity_reasons for row in rows)
        per_game[game] = {
            "rows": len(rows),
            "non_ambiguous_rate": clean / len(rows),
        }
    return {
        "rows": len(traces),
        "exact_repeat_keys": len({row.exact_repeat_key() for row in traces}),
        "duplicate_rows": len(traces)
        - len({row.exact_repeat_key() for row in traces}),
        "non_ambiguous_rate": 1.0 - len(ambiguous_rows) / len(traces),
        "ambiguity_reason_counts": dict(
            Counter(
                reason
                for row in traces
                for reason in row.effects.ambiguity_reasons
            )
        ),
        "per_label": per_label,
        "per_game": per_game,
    }


def _evaluate_gates(
    *,
    frozen: Mapping[str, Any],
    train_quality: Mapping[str, Any],
    validation_quality: Mapping[str, Any],
    metrics: Mapping[str, Any],
    baseline_name: str,
    gain: float,
    shuffle_degradation: float,
    bootstrap: Mapping[str, float],
    per_game: Mapping[str, Any],
    output_metrics: Mapping[str, float],
    projection_freeze: Mapping[str, Any],
) -> dict[str, bool]:
    gates = frozen["gates"]

    def label_capacity(
        quality: Mapping[str, Any], positive: int, negative: int, games: int = 0
    ) -> bool:
        return all(
            int(quality["per_label"][label]["positives"]) >= positive
            and int(quality["per_label"][label]["negatives"]) >= negative
            and int(quality["per_label"][label]["games_with_positive"]) >= games
            for label in EFFECT_LABELS
        )

    return {
        "source_train_label_capacity": label_capacity(
            train_quality,
            int(gates["minimum_source_train_positives_per_label"]),
            int(gates["minimum_source_train_negatives_per_label"]),
        ),
        "source_validation_label_capacity": label_capacity(
            validation_quality,
            int(gates["minimum_source_validation_positives_per_label"]),
            int(gates["minimum_source_validation_negatives_per_label"]),
            int(gates["minimum_validation_games_with_positive_per_label"]),
        ),
        "no_exact_duplicate_rows": (
            train_quality["duplicate_rows"] == 0
            and validation_quality["duplicate_rows"] == 0
        ),
        "minimum_global_non_ambiguous_rate": (
            train_quality["non_ambiguous_rate"]
            >= float(gates["minimum_global_non_ambiguous_rate"])
            and validation_quality["non_ambiguous_rate"]
            >= float(gates["minimum_global_non_ambiguous_rate"])
        ),
        "minimum_per_game_non_ambiguous_rate": all(
            item["non_ambiguous_rate"]
            >= float(gates["minimum_per_game_non_ambiguous_rate"])
            for quality in (train_quality, validation_quality)
            for item in quality["per_game"].values()
        ),
        "strict_json_validity": (
            output_metrics["strict_json_validity"]
            >= float(gates["minimum_strict_json_validity"])
        ),
        "support_zero_rate": (
            output_metrics["support_zero_rate"]
            >= float(gates["minimum_support_zero_rate"])
        ),
        "grounded_hypothesis_rate": (
            output_metrics["grounded_hypothesis_rate"]
            >= float(gates["minimum_grounded_hypothesis_rate"])
        ),
        "primary_macro_f1_gain": (
            gain >= float(gates["minimum_primary_macro_f1_gain"])
        ),
        "bootstrap_lower_bound_positive": bootstrap["lower_95"] > 0.0,
        "target_shuffle_degradation": (
            shuffle_degradation
            >= float(gates["minimum_target_shuffle_degradation"])
        ),
        "every_validation_game_nonnegative": all(
            item["gain"] >= 0.0 for item in per_game.values()
        ),
        "calibration_ece": (
            metrics["structured"]["macro_ece"]
            <= float(gates["maximum_macro_ece"])
        ),
        "game_identity_leakage": all(
            item["conditional_identity_gain"]
            <= float(gates["maximum_identity_gain_over_selected_action"])
            for name, item in _read_json(
                Path(DEFAULT_OUTPUT_DIR) / "source_train_preflight.json"
            )["projection_audits"].items()
            if name == projection_freeze["selected_projection"]
        ),
    }


def _action_only_row(trace: ActionTargetTrace) -> dict[str, int]:
    return {f"selected_action_name:{trace.selected_action_name}": 1}


def _runtime_metadata() -> dict[str, Any]:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": np.__version__,
    }
    try:
        import sklearn

        payload["scikit_learn"] = sklearn.__version__
    except ImportError:
        payload["scikit_learn"] = None
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        payload["torch"] = None
    return payload


def _load_traces(directory: Path, split: str) -> list[ActionTargetTrace]:
    games = SOURCE_TRAIN if split == "source_train" else SOURCE_VALIDATION
    traces = []
    for game in games:
        path = directory / "shards" / f"{game}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing V3 shard: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trace = ActionTargetTrace.from_dict(json.loads(line))
                if trace.source_split != split or trace.game_id != game:
                    raise ValueError(f"V3 shard provenance mismatch: {path}")
                traces.append(trace)
    return traces


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_embedded_checksum(
    payload: Mapping[str, Any], field: str
) -> None:
    expected = str(payload[field])
    check = dict(payload)
    check.pop(field, None)
    actual = _payload_checksum(check)
    if actual != expected:
        raise ValueError(f"{field} mismatch: {actual} != {expected}")


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(
    path: Path, payloads: Iterable[Mapping[str, Any]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "evaluate"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--frozen-manifest",
        default=str(DEFAULT_FROZEN_MANIFEST_PATH),
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.command == "preflight":
        payload = run_source_train_preflight(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
        )
    else:
        payload = run_evaluation(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wall_seconds={time.perf_counter() - started:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREFLIGHT_FORMAT_VERSION",
    "PROJECTION_FREEZE_FORMAT_VERSION",
    "RESULT_FORMAT_VERSION",
    "run_evaluation",
    "run_source_train_preflight",
]
