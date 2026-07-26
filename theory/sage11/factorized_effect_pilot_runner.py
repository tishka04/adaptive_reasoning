"""Run the pre-registered SAGE.11 factorized effect pilot v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from .effect_pilot_runner import (
    DEFAULT_MANIFEST_PATH,
    iter_source_rows,
)
from .pilot import effect_macro_f1, majority_predictions
from .source_dataset_runner import verify_source_dataset
from .splits import SOURCE_TRAIN, SOURCE_VALIDATION


PILOT_V2_FORMAT_VERSION = "sage11-factorized-effect-pilot-v2"
DEFAULT_V2_RESULT_PATH = (
    Path("diagnostics") / "sage" / "sage11_factorized_effect_pilot_v2.json"
)
PILOT_V2_MINIMUM_IMPROVEMENT = 0.10
RANDOM_STATE = 11

ACTION_NAMES: Tuple[str, ...] = tuple(
    f"ACTION{index}"
    for index in range(1, 7)
)
CHANGED_BUCKETS: Tuple[str, ...] = (
    "zero",
    "one",
    "few",
    "some",
    "many",
)
BOOLEAN_VALUES: Tuple[str, ...] = ("False", "True")
CORE_HEADS: Tuple[str, ...] = ("changed_cells", "player_moved")
AUDIT_HEADS: Tuple[str, ...] = ("level_complete", "game_over")

ACTION_FEATURE_NAMES: Tuple[str, ...] = (
    *(f"current_action:{name}" for name in ACTION_NAMES),
    "current_argument:has_xy",
    "current_argument:on_boundary",
    "current_argument:on_corner",
    "current_argument:on_diagonal",
)
ARGUMENT_FEATURE_NAMES: Tuple[str, ...] = ACTION_FEATURE_NAMES[-4:]
CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    "context:exact_continuity",
    "reset_step:zero",
    "reset_step:one_to_three",
    "reset_step:four_to_fifteen",
    "reset_step:sixteen_plus",
    "state_visit:first",
    "state_visit:second",
    "state_visit:third_or_fourth",
    "state_visit:fifth_plus",
    "state_recency:new",
    "state_recency:one",
    "state_recency:two_to_four",
    "state_recency:five_to_sixteen",
    "state_recency:seventeen_plus",
    *(f"previous_action:{name}" for name in ACTION_NAMES),
    "previous_action:same_as_current",
    *(f"previous_changed:{bucket}" for bucket in CHANGED_BUCKETS),
    "previous_effect:player_moved",
    "previous_effect:level_complete",
    "previous_effect:game_over",
    "relative_target:has_xy",
    "relative_target:same_target",
    "relative_target:same_row",
    "relative_target:same_column",
    "relative_target:dx_negative",
    "relative_target:dx_zero",
    "relative_target:dx_positive",
    "relative_target:dy_negative",
    "relative_target:dy_zero",
    "relative_target:dy_positive",
    "relative_target:distance_zero",
    "relative_target:distance_one_to_four",
    "relative_target:distance_five_to_sixteen",
    "relative_target:distance_seventeen_plus",
    *(f"atom_delta:{bucket}" for bucket in CHANGED_BUCKETS),
)

HEAD_VALUE_NAMES: Mapping[str, Tuple[str, ...]] = {
    "changed_cells": CHANGED_BUCKETS,
    "player_moved": BOOLEAN_VALUES,
    "level_complete": BOOLEAN_VALUES,
    "game_over": BOOLEAN_VALUES,
}


@dataclass(frozen=True)
class FactorizedPilotDataset:
    """Model matrix, factor labels, and streaming-context coverage."""

    features: np.ndarray
    labels: Mapping[str, np.ndarray]
    train_mask: np.ndarray
    actions: np.ndarray
    games: np.ndarray
    feature_names: Tuple[str, ...]
    atom_vocabulary: Tuple[str, ...]
    action_feature_indices: Tuple[int, ...]
    argument_feature_indices: Tuple[int, ...]
    exact_continuity: np.ndarray
    revisited_state: np.ndarray
    has_xy: np.ndarray
    manifest_checksum: str


def _effect_value(
    atoms: Sequence[str],
    *,
    kind: str,
    predicate: str,
) -> str:
    prefix = f"{kind}:{predicate}("
    for atom in atoms:
        if atom.startswith(prefix) and atom.endswith(")"):
            return atom[len(prefix):-1]
    raise ValueError(f"missing effect atom {kind}:{predicate}")


def _factor_labels(atoms: Sequence[str]) -> Dict[str, int]:
    values = {
        "changed_cells": _effect_value(
            atoms,
            kind="effect",
            predicate="changed_cells",
        ),
        "player_moved": _effect_value(
            atoms,
            kind="effect",
            predicate="player_moved",
        ),
        "level_complete": _effect_value(
            atoms,
            kind="progress",
            predicate="level_complete",
        ),
        "game_over": _effect_value(
            atoms,
            kind="risk",
            predicate="game_over",
        ),
    }
    return {
        head: HEAD_VALUE_NAMES[head].index(value)
        for head, value in values.items()
    }


def _coordinate_pair(
    action_data: Mapping[str, Any] | None,
) -> Tuple[int, int] | None:
    data = dict(action_data or {})
    if "x" not in data or "y" not in data:
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (TypeError, ValueError):
        return None


def _count_bucket(value: int) -> str:
    count = max(0, int(value))
    if count == 0:
        return "zero"
    if count == 1:
        return "one"
    if count <= 4:
        return "few"
    if count <= 15:
        return "some"
    return "many"


def _reset_step_bucket(step_index: int) -> str:
    step = max(0, int(step_index))
    if step == 0:
        return "zero"
    if step <= 3:
        return "one_to_three"
    if step <= 15:
        return "four_to_fifteen"
    return "sixteen_plus"


def _visit_bucket(previous_visits: int) -> str:
    if previous_visits <= 0:
        return "first"
    if previous_visits == 1:
        return "second"
    if previous_visits <= 3:
        return "third_or_fourth"
    return "fifth_plus"


def _recency_bucket(distance: int | None) -> str:
    if distance is None:
        return "new"
    if distance <= 1:
        return "one"
    if distance <= 4:
        return "two_to_four"
    if distance <= 16:
        return "five_to_sixteen"
    return "seventeen_plus"


def _distance_bucket(distance: int) -> str:
    if distance <= 0:
        return "zero"
    if distance <= 4:
        return "one_to_four"
    if distance <= 16:
        return "five_to_sixteen"
    return "seventeen_plus"


def _direction(value: int) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


def load_factorized_pilot_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> FactorizedPilotDataset:
    """Encode frozen source rows using only pre-action available context."""
    path = Path(manifest_path)
    verify_source_dataset(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    train_atoms: set[str] = set()
    total_rows = 0
    for row in iter_source_rows(path, payload):
        total_rows += 1
        if row["source_split"] == "source_train":
            train_atoms.update(str(atom) for atom in row["atoms_before"])

    atom_vocabulary = tuple(sorted(train_atoms))
    atom_features = tuple(
        f"current_atom:{atom}"
        for atom in atom_vocabulary
    )
    feature_names = (
        ACTION_FEATURE_NAMES
        + atom_features
        + CONTEXT_FEATURE_NAMES
    )
    feature_to_index = {
        name: index
        for index, name in enumerate(feature_names)
    }
    action_feature_indices = tuple(
        feature_to_index[name]
        for name in ACTION_FEATURE_NAMES
    )
    argument_feature_indices = tuple(
        feature_to_index[name]
        for name in ARGUMENT_FEATURE_NAMES
    )

    features = np.zeros(
        (total_rows, len(feature_names)),
        dtype=np.float32,
    )
    labels = {
        head: np.empty(total_rows, dtype=np.int64)
        for head in CORE_HEADS + AUDIT_HEADS
    }
    train_mask = np.empty(total_rows, dtype=bool)
    actions = np.empty(total_rows, dtype="<U16")
    games = np.empty(total_rows, dtype="<U16")
    exact_continuity = np.zeros(total_rows, dtype=bool)
    revisited_state = np.zeros(total_rows, dtype=bool)
    has_xy = np.zeros(total_rows, dtype=bool)

    previous_rows: Dict[Tuple[str, int, int], Mapping[str, Any]] = {}
    state_visits: Dict[Tuple[str, int, int], Counter[str]] = defaultdict(
        Counter
    )
    state_last_steps: Dict[
        Tuple[str, int, int],
        Dict[str, int],
    ] = defaultdict(dict)

    for index, row in enumerate(iter_source_rows(path, payload)):
        game = str(row["game_id"])
        action = str(row["action_name"])
        action_data = dict(row.get("action_data", {}) or {})
        split = str(row["source_split"])
        is_train = split == "source_train"
        if not is_train and split != "source_validation":
            raise ValueError(f"unexpected source split {split}")

        if action not in ACTION_NAMES:
            raise ValueError(f"unsupported action {action}")
        features[
            index,
            feature_to_index[f"current_action:{action}"],
        ] = 1.0
        coordinates = _coordinate_pair(action_data)
        if coordinates is not None:
            x, y = coordinates
            has_xy[index] = True
            features[
                index,
                feature_to_index["current_argument:has_xy"],
            ] = 1.0
            on_x_boundary = x in {0, 63}
            on_y_boundary = y in {0, 63}
            features[
                index,
                feature_to_index["current_argument:on_boundary"],
            ] = float(on_x_boundary or on_y_boundary)
            features[
                index,
                feature_to_index["current_argument:on_corner"],
            ] = float(on_x_boundary and on_y_boundary)
            features[
                index,
                feature_to_index["current_argument:on_diagonal"],
            ] = float(x == y or x + y == 63)

        atoms_before = tuple(str(atom) for atom in row["atoms_before"])
        for atom in atoms_before:
            atom_name = f"current_atom:{atom}"
            atom_index = feature_to_index.get(atom_name)
            if atom_index is not None:
                features[index, atom_index] = 1.0

        step_index = int(row["step_index"])
        features[
            index,
            feature_to_index[
                f"reset_step:{_reset_step_bucket(step_index)}"
            ],
        ] = 1.0

        sequence_key = (
            game,
            int(row["seed"]),
            int(row["reset_index"]),
        )
        state_digest = str(row["state_digest_before"])
        previous_visits = state_visits[sequence_key][state_digest]
        revisited_state[index] = previous_visits > 0
        features[
            index,
            feature_to_index[
                f"state_visit:{_visit_bucket(previous_visits)}"
            ],
        ] = 1.0
        last_seen_step = state_last_steps[sequence_key].get(state_digest)
        recency = (
            None
            if last_seen_step is None
            else max(0, step_index - last_seen_step)
        )
        features[
            index,
            feature_to_index[
                f"state_recency:{_recency_bucket(recency)}"
            ],
        ] = 1.0

        previous = previous_rows.get(sequence_key)
        contiguous = bool(
            previous
            and step_index == int(previous["step_index"]) + 1
            and state_digest == str(previous["state_digest_after"])
        )
        exact_continuity[index] = contiguous
        if contiguous and previous is not None:
            features[
                index,
                feature_to_index["context:exact_continuity"],
            ] = 1.0
            previous_action = str(previous["action_name"])
            features[
                index,
                feature_to_index[f"previous_action:{previous_action}"],
            ] = 1.0
            features[
                index,
                feature_to_index["previous_action:same_as_current"],
            ] = float(previous_action == action)

            previous_factors = _factor_labels(
                tuple(str(atom) for atom in previous["effect_atoms"])
            )
            previous_changed = CHANGED_BUCKETS[
                previous_factors["changed_cells"]
            ]
            features[
                index,
                feature_to_index[
                    f"previous_changed:{previous_changed}"
                ],
            ] = 1.0
            features[
                index,
                feature_to_index["previous_effect:player_moved"],
            ] = float(previous_factors["player_moved"])
            features[
                index,
                feature_to_index["previous_effect:level_complete"],
            ] = float(previous_factors["level_complete"])
            features[
                index,
                feature_to_index["previous_effect:game_over"],
            ] = float(previous_factors["game_over"])

            previous_coordinates = _coordinate_pair(
                dict(previous.get("action_data", {}) or {})
            )
            if coordinates is not None and previous_coordinates is not None:
                x, y = coordinates
                previous_x, previous_y = previous_coordinates
                dx = x - previous_x
                dy = y - previous_y
                features[
                    index,
                    feature_to_index["relative_target:has_xy"],
                ] = 1.0
                features[
                    index,
                    feature_to_index["relative_target:same_target"],
                ] = float(dx == 0 and dy == 0)
                features[
                    index,
                    feature_to_index["relative_target:same_row"],
                ] = float(dy == 0)
                features[
                    index,
                    feature_to_index["relative_target:same_column"],
                ] = float(dx == 0)
                features[
                    index,
                    feature_to_index[
                        f"relative_target:dx_{_direction(dx)}"
                    ],
                ] = 1.0
                features[
                    index,
                    feature_to_index[
                        f"relative_target:dy_{_direction(dy)}"
                    ],
                ] = 1.0
                distance = abs(dx) + abs(dy)
                features[
                    index,
                    feature_to_index[
                        "relative_target:distance_"
                        f"{_distance_bucket(distance)}"
                    ],
                ] = 1.0

            atoms_after = set(
                str(atom)
                for atom in previous["atoms_after"]
            )
            atom_delta = _count_bucket(
                len(set(atoms_before).symmetric_difference(atoms_after))
            )
            features[
                index,
                feature_to_index[f"atom_delta:{atom_delta}"],
            ] = 1.0

        factor_values = _factor_labels(
            tuple(str(atom) for atom in row["effect_atoms"])
        )
        for head, target in labels.items():
            target[index] = factor_values[head]
        train_mask[index] = is_train
        actions[index] = action
        games[index] = game

        state_visits[sequence_key][state_digest] += 1
        state_last_steps[sequence_key][state_digest] = step_index
        previous_rows[sequence_key] = row

    if set(games[train_mask]) != set(SOURCE_TRAIN):
        raise ValueError("v2 pilot source-training games do not match registry")
    if set(games[~train_mask]) != set(SOURCE_VALIDATION):
        raise ValueError(
            "v2 pilot source-validation games do not match registry"
        )
    if total_rows != int(payload["total_transitions"]):
        raise ValueError("v2 pilot row count does not match manifest")

    return FactorizedPilotDataset(
        features=features,
        labels=labels,
        train_mask=train_mask,
        actions=actions,
        games=games,
        feature_names=feature_names,
        atom_vocabulary=atom_vocabulary,
        action_feature_indices=action_feature_indices,
        argument_feature_indices=argument_feature_indices,
        exact_continuity=exact_continuity,
        revisited_state=revisited_state,
        has_xy=has_xy,
        manifest_checksum=str(payload["manifest_checksum"]),
    )


def make_factor_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> HistGradientBoostingClassifier:
    """Return the frozen balanced classifier for one factor head."""
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=4,
        max_iter=100,
        early_stopping=False,
        class_weight="balanced",
        random_state=int(random_state),
    )


def _label_support(
    labels: np.ndarray,
    head: str,
) -> Dict[str, int]:
    counts = Counter(int(value) for value in labels)
    return {
        name: int(counts.get(index, 0))
        for index, name in enumerate(HEAD_VALUE_NAMES[head])
    }


def _head_metrics(
    *,
    truth: np.ndarray,
    head: str,
    majority: np.ndarray,
    action_only: np.ndarray | None,
    full: np.ndarray,
    action_shuffled: np.ndarray,
    argument_shuffled: np.ndarray,
) -> Dict[str, Any]:
    majority_f1 = effect_macro_f1(truth, majority)
    full_f1 = effect_macro_f1(truth, full)
    action_only_f1 = (
        None
        if action_only is None
        else effect_macro_f1(truth, action_only)
    )
    return {
        "rows": int(len(truth)),
        "label_support": _label_support(truth, head),
        "per_action_majority_macro_f1": majority_f1,
        "action_only_macro_f1": action_only_f1,
        "full_macro_f1": full_f1,
        "full_minus_majority": full_f1 - majority_f1,
        "full_minus_action_only": (
            None
            if action_only_f1 is None
            else full_f1 - action_only_f1
        ),
        "action_shuffled_macro_f1": effect_macro_f1(
            truth,
            action_shuffled,
        ),
        "argument_shuffled_macro_f1": effect_macro_f1(
            truth,
            argument_shuffled,
        ),
    }


def _composite_metrics(
    head_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    majority = float(np.mean([
        head_metrics[head]["per_action_majority_macro_f1"]
        for head in CORE_HEADS
    ]))
    action_only = float(np.mean([
        head_metrics[head]["action_only_macro_f1"]
        for head in CORE_HEADS
    ]))
    full = float(np.mean([
        head_metrics[head]["full_macro_f1"]
        for head in CORE_HEADS
    ]))
    action_shuffled = float(np.mean([
        head_metrics[head]["action_shuffled_macro_f1"]
        for head in CORE_HEADS
    ]))
    argument_shuffled = float(np.mean([
        head_metrics[head]["argument_shuffled_macro_f1"]
        for head in CORE_HEADS
    ]))
    return {
        "per_action_majority_macro_f1": majority,
        "action_only_macro_f1": action_only,
        "full_macro_f1": full,
        "full_minus_majority": full - majority,
        "full_minus_action_only": full - action_only,
        "action_shuffle_degradation": full - action_shuffled,
        "argument_shuffle_degradation": full - argument_shuffled,
    }


def _coverage_block(
    dataset: FactorizedPilotDataset,
    mask: np.ndarray,
) -> Dict[str, Any]:
    rows = int(mask.sum())
    return {
        "rows": rows,
        "exact_contiguous_predecessor_rows": int(
            dataset.exact_continuity[mask].sum()
        ),
        "exact_contiguous_predecessor_rate": float(
            dataset.exact_continuity[mask].mean()
        ),
        "revisited_state_rows": int(
            dataset.revisited_state[mask].sum()
        ),
        "revisited_state_rate": float(
            dataset.revisited_state[mask].mean()
        ),
        "xy_argument_rows": int(dataset.has_xy[mask].sum()),
        "xy_argument_rate": float(dataset.has_xy[mask].mean()),
    }


def _hardware_metadata(feature_count: int) -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": (
            str(torch.cuda.get_device_name(0))
            if cuda_available
            else None
        ),
        "training_device": "cpu",
        "device_decision": (
            "The frozen balanced scikit-learn histogram gradient boosters "
            f"use {feature_count} dense features and have no CUDA backend. "
            "Changing estimator to use the GPU would change the "
            "pre-registered method rather than accelerate it effectively."
        ),
    }


def evaluate_factorized_gate(
    overall_heads: Mapping[str, Mapping[str, Any]],
    overall_composite: Mapping[str, float],
    per_game: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply the three frozen v2 gate conditions."""
    improvement_pass = (
        overall_composite["full_minus_action_only"]
        >= PILOT_V2_MINIMUM_IMPROVEMENT
    )
    head_nonnegative = {
        head: overall_heads[head]["full_minus_action_only"] >= 0.0
        for head in CORE_HEADS
    }
    game_nonnegative = {
        game: (
            per_game[game]["composite"]["full_minus_action_only"] >= 0.0
        )
        for game in SOURCE_VALIDATION
    }
    return {
        "go": bool(
            improvement_pass
            and all(head_nonnegative.values())
            and all(game_nonnegative.values())
        ),
        "minimum_overall_full_minus_action_only": (
            PILOT_V2_MINIMUM_IMPROVEMENT
        ),
        "overall_improvement_pass": improvement_pass,
        "core_heads_nonnegative": head_nonnegative,
        "validation_games_nonnegative": game_nonnegative,
    }


def run_factorized_effect_pilot(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    result_path: str | Path = DEFAULT_V2_RESULT_PATH,
    *,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute one fixed source-train/source-validation v2 pilot."""
    started = time.perf_counter()
    dataset = load_factorized_pilot_dataset(manifest_path)
    loaded = time.perf_counter()

    train = dataset.train_mask
    validation = ~train
    validation_features = dataset.features[validation]
    validation_games = dataset.games[validation]
    validation_actions = dataset.actions[validation]
    action_columns = np.asarray(
        dataset.action_feature_indices,
        dtype=np.int64,
    )
    argument_columns = np.asarray(
        dataset.argument_feature_indices,
        dtype=np.int64,
    )

    generator = np.random.default_rng(int(random_state))
    action_shuffled_features = validation_features.copy()
    for game in SOURCE_VALIDATION:
        indices = np.flatnonzero(validation_games == game)
        permutation = generator.permutation(indices)
        action_shuffled_features[np.ix_(indices, action_columns)] = (
            validation_features[np.ix_(permutation, action_columns)]
        )

    argument_shuffled_features = validation_features.copy()
    for game in SOURCE_VALIDATION:
        for action in ACTION_NAMES:
            indices = np.flatnonzero(
                (validation_games == game)
                & (validation_actions == action)
            )
            if len(indices) < 2:
                continue
            permutation = generator.permutation(indices)
            argument_shuffled_features[
                np.ix_(indices, argument_columns)
            ] = validation_features[
                np.ix_(permutation, argument_columns)
            ]

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for head in CORE_HEADS:
        target = dataset.labels[head]
        action_model = make_factor_classifier(random_state=random_state)
        action_model.fit(
            dataset.features[train][:, action_columns],
            target[train],
        )
        full_model = make_factor_classifier(random_state=random_state)
        full_model.fit(dataset.features[train], target[train])
        predictions[head] = {
            "majority": majority_predictions(
                target,
                train,
                groups=dataset.actions,
            ),
            "action_only": action_model.predict(
                validation_features[:, action_columns]
            ),
            "full": full_model.predict(validation_features),
            "action_shuffled": full_model.predict(
                action_shuffled_features
            ),
            "argument_shuffled": full_model.predict(
                argument_shuffled_features
            ),
        }

    audit_estimator_modes = {}
    for head in AUDIT_HEADS:
        target = dataset.labels[head]
        training_classes = np.unique(target[train])
        if len(training_classes) == 1:
            full_prediction = np.full(
                validation.sum(),
                int(training_classes[0]),
                dtype=np.int64,
            )
            action_shuffled_prediction = full_prediction.copy()
            argument_shuffled_prediction = full_prediction.copy()
            audit_estimator_modes[head] = "constant_single_training_class"
        else:
            full_model = make_factor_classifier(
                random_state=random_state
            )
            full_model.fit(dataset.features[train], target[train])
            full_prediction = full_model.predict(validation_features)
            action_shuffled_prediction = full_model.predict(
                action_shuffled_features
            )
            argument_shuffled_prediction = full_model.predict(
                argument_shuffled_features
            )
            audit_estimator_modes[head] = "balanced_hist_gradient_booster"
        predictions[head] = {
            "majority": majority_predictions(
                target,
                train,
                groups=dataset.actions,
            ),
            "full": full_prediction,
            "action_shuffled": action_shuffled_prediction,
            "argument_shuffled": argument_shuffled_prediction,
        }
    fitted = time.perf_counter()

    def metrics_for_mask(
        subset: np.ndarray,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        core_metrics = {}
        for head in CORE_HEADS:
            target = dataset.labels[head][validation][subset]
            values = predictions[head]
            core_metrics[head] = _head_metrics(
                truth=target,
                head=head,
                majority=values["majority"][subset],
                action_only=values["action_only"][subset],
                full=values["full"][subset],
                action_shuffled=values["action_shuffled"][subset],
                argument_shuffled=values["argument_shuffled"][subset],
            )
        return core_metrics, _composite_metrics(core_metrics)

    overall_subset = np.ones(validation.sum(), dtype=bool)
    overall_heads, overall_composite = metrics_for_mask(overall_subset)
    per_game: Dict[str, Any] = {}
    for game in SOURCE_VALIDATION:
        subset = validation_games == game
        head_metrics, composite = metrics_for_mask(subset)
        per_game[game] = {
            "heads": head_metrics,
            "composite": composite,
            "coverage": _coverage_block(
                dataset,
                validation & (dataset.games == game),
            ),
        }

    audit_metrics = {}
    for head in AUDIT_HEADS:
        target = dataset.labels[head][validation]
        values = predictions[head]
        audit_metrics[head] = _head_metrics(
            truth=target,
            head=head,
            majority=values["majority"],
            action_only=None,
            full=values["full"],
            action_shuffled=values["action_shuffled"],
            argument_shuffled=values["argument_shuffled"],
        )
    evaluated = time.perf_counter()

    gate = evaluate_factorized_gate(
        overall_heads,
        overall_composite,
        per_game,
    )
    go = bool(gate["go"])

    payload: Dict[str, Any] = {
        "format_version": PILOT_V2_FORMAT_VERSION,
        "run_date": date.today().isoformat(),
        "decision": {
            "go": go,
            "next_step": (
                "implement_the_v2_input_interface_before_model_training"
                if go
                else "collect_richer_relational_state_or_revise_features"
            ),
            "gate": gate,
        },
        "dataset": {
            "manifest_path": Path(manifest_path).as_posix(),
            "manifest_checksum": dataset.manifest_checksum,
            "rows": int(len(dataset.train_mask)),
            "source_train_rows": int(train.sum()),
            "source_validation_rows": int(validation.sum()),
            "source_train_games": list(SOURCE_TRAIN),
            "source_validation_games": list(SOURCE_VALIDATION),
            "atom_vocabulary_size": len(dataset.atom_vocabulary),
            "action_only_features": len(
                dataset.action_feature_indices
            ),
            "full_features": len(dataset.feature_names),
            "raw_grids_available": False,
            "raw_coordinates_used": False,
        },
        "features": {
            "names": list(dataset.feature_names),
            "action_only_names": list(ACTION_FEATURE_NAMES),
            "argument_names": list(ARGUMENT_FEATURE_NAMES),
            "excluded": [
                "game identity",
                "policy arm",
                "raw x and y",
                "state digest bytes or prefixes",
                "current-row post-action atoms and effects",
                "historical and holdout rows",
            ],
        },
        "targets": {
            "core_heads": list(CORE_HEADS),
            "audit_heads": list(AUDIT_HEADS),
            "audit_estimator_modes": audit_estimator_modes,
            "value_multiset_delta": (
                "excluded from scoring because its cardinality duplicates "
                "the current changed-cell set"
            ),
            "training_support": {
                head: _label_support(dataset.labels[head][train], head)
                for head in CORE_HEADS + AUDIT_HEADS
            },
            "validation_support": {
                head: _label_support(
                    dataset.labels[head][validation],
                    head,
                )
                for head in CORE_HEADS + AUDIT_HEADS
            },
        },
        "classifier": {
            "name": "sklearn.ensemble.HistGradientBoostingClassifier",
            "learning_rate": 0.08,
            "max_depth": 4,
            "max_iter": 100,
            "early_stopping": False,
            "class_weight": "balanced",
            "random_state": int(random_state),
            "hyperparameter_searches": 0,
        },
        "metrics": {
            "overall": {
                "heads": overall_heads,
                "composite": overall_composite,
            },
            "per_game": per_game,
            "audit_heads": audit_metrics,
        },
        "coverage": {
            "source_train": _coverage_block(dataset, train),
            "source_validation": _coverage_block(dataset, validation),
        },
        "controls": {
            "action_shuffle": (
                "all current action-only features permuted within each "
                "validation game without retraining"
            ),
            "argument_shuffle": (
                "direct argument predicates permuted within each "
                "validation game/action stratum without retraining"
            ),
            "random_state": int(random_state),
            "controls_are_gate_conditions": False,
        },
        "hardware": _hardware_metadata(len(dataset.feature_names)),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "timing_seconds": {
            "verify_load_and_encode": round(loaded - started, 3),
            "fit_all_heads": round(fitted - loaded, 3),
            "evaluate_and_serialize": round(evaluated - fitted, 3),
            "total": round(evaluated - started, 3),
        },
        "reproduction": {
            "command": (
                "ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe "
                "-m theory.sage11.factorized_effect_pilot_runner"
            ),
            "protocol": (
                "reports/SAGE11_EFFECT_PILOT_V2_PROTOCOL.md"
            ),
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["result_checksum"] = hashlib.sha256(canonical).hexdigest()

    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, output)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen SAGE.11 factorized effect pilot v2.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_V2_RESULT_PATH,
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_factorized_effect_pilot(
        args.manifest,
        args.output,
        random_state=args.random_state,
    )
    print(json.dumps({
        "decision": result["decision"],
        "overall": result["metrics"]["overall"],
        "per_game_composite": {
            game: values["composite"]
            for game, values in result["metrics"]["per_game"].items()
        },
        "result_checksum": result["result_checksum"],
        "timing_seconds": result["timing_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_V2_RESULT_PATH",
    "FactorizedPilotDataset",
    "evaluate_factorized_gate",
    "load_factorized_pilot_dataset",
    "make_factor_classifier",
    "run_factorized_effect_pilot",
]
