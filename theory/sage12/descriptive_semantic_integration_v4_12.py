"""SAGE12 V4.12 descriptive semantics and conditional architecture pilot.

V4.12 reuses the frozen V4.11 counterfactual panels but removes their
underpowered scalar progress target.  A small object-relative student instead
compares eight directly observed effects, distils relation residuals onto a
root-only anchor under strict leave-one-game-out evaluation, and opens the
unchanged V4.7 world-model/trajectory/EBM stack only if that semantic gate
passes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN

from .compiler import SLOT_EFFECTS, SlotAnnotation
from .counterfactual_semantic_panels_v4_11 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V411_DIR,
    _assemble_training_data,
    _bootstrap_identity_increment,
    _bootstrap_panel_difference,
    _calibrated,
    _centered_residual,
    _fit_model,
    _game_balanced_shifts,
    _identity_metric,
    _logit,
    _panel_brier_rows,
    _predict_model,
    _reverse_neighbor_tensors,
    _select_alpha_and_shifts,
    _shuffled_records,
    _sigmoid,
    load_teacher_panels,
)
from .integration_pilot import load_complete_roots
from .integration_pilot_v4_7 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V47_DIR,
    DEFAULT_V43_DIR,
    SlotExample,
    _action_only_choice,
    _decision_row,
    _fit_nested_world,
    _nodes,
    _paired_bootstrap,
    _select_path,
    _select_primary_baseline,
    _sequence_only_choice,
    _summarize,
    _train_ebm,
    _trajectory_features,
    _world_metrics,
    load_slot_examples,
)
from .object_relative_student_v4_9 import (
    _action_only_probabilities,
    _select_device,
    tensorize_records,
)
from .semantic_adapter_v4_8 import _completion_capture
from .semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    SemanticTeacherRecord,
    _checksum,
    _file_sha256,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)

FORMAT_VERSION = "sage12-descriptive-semantic-integration-v4.12"
MANIFEST_VERSION = "sage12-descriptive-semantic-manifest-v4.12"
SEMANTIC_RESULT_VERSION = "sage12-descriptive-semantic-result-v4.12"
INTEGRATION_RESULT_VERSION = "sage12-descriptive-integration-result-v4.12"
PREDICTION_VERSION = "sage12-descriptive-logo-prediction-v4.12"
SLOT_EXPORT_VERSION = "sage12-descriptive-slot-annotations-v4.12"

DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage12" / "descriptive_semantic_integration_v4_12"
)

ACTIVE_EFFECTS = (
    "changed",
    "moved",
    "target_removed",
    "target_moved",
    "local_change",
    "contact_lost",
    "productive",
    "risk",
)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 5_120


def _source_fingerprints(
    v411_dir: Path,
    v43_dir: Path,
    v47_dir: Path,
) -> dict[str, Any]:
    paths = (
        v411_dir / "frozen_manifest.json",
        v411_dir / "teacher_panels.jsonl",
        v411_dir / "teacher_qa.json",
        v411_dir / "student_result.json",
        v43_dir / "frozen_manifest.json",
        v43_dir / "source_train_collection_manifest.json",
        v47_dir / "frozen_manifest.json",
        v47_dir / "result.json",
    )
    return {
        path.as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    }


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    v47_dir: str | Path = DEFAULT_V47_DIR,
) -> dict[str, Any]:
    """Freeze both gates before fitting any V4.12 model."""

    destination = Path(output_dir)
    qa = _read_json(Path(v411_dir) / "teacher_qa.json")
    eligible = tuple(str(effect) for effect in qa["eligible_effects"])
    if eligible != ACTIVE_EFFECTS:
        raise ValueError(
            f"V4.11 eligible-effect drift: expected {ACTIVE_EFFECTS}, got {eligible}"
        )
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "source_games": list(SOURCE_TRAIN),
        "active_effects": list(ACTIVE_EFFECTS),
        "source_fingerprints": _source_fingerprints(
            Path(v411_dir), Path(v43_dir), Path(v47_dir)
        ),
        "teacher_contract": {
            "reuses_v4_11_panels": True,
            "expected_panels": 1_056,
            "expected_immediate_arms": 3_914,
            "expected_fresh_comparisons": 5_529,
            "requires_v4_11_collection_and_firewall_checks": True,
            "ignores_v4_11_progress_capacity": True,
            "scalar_progress_target_used": False,
        },
        "training": {
            "seed": SEED,
            "hash_buckets": 2_048,
            "embedding_width": 32,
            "hidden_width": 96,
            "epochs": 30,
            "samples_per_game_per_epoch": 256,
            "maximum_pairs_per_epoch": 4_096,
            "learning_rate": 0.0015,
            "weight_decay": 0.0001,
            "effect_pair_weight": 0.50,
            "progress_pair_weight": 0.0,
            "tie_consistency_weight": 0.0,
            "alpha_grid": list(ALPHA_GRID),
            "calibration": "inner_game_balanced_logit_shift",
            "outer_split": "leave_one_source_train_game_out",
        },
        "semantic_gate": {
            "primary_rows": "fresh_v4_11_panel_arms_only",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 51_200,
            "effect_pair_gain_over_root_ci_lower_strictly_positive": True,
            "effect_pair_gain_over_action_ci_lower_strictly_positive": True,
            "relation_shuffle_pair_degradation_ci_lower_strictly_positive": True,
            "active_macro_brier_gain_over_root_ci_lower_strictly_positive": True,
            "relation_shuffle_brier_degradation_ci_lower_strictly_positive": True,
            "ece_not_worse_than_root_only": True,
            "nonnegative_games_minimum": 6,
            "identity_increment_ci_upper_maximum": 0.02,
            "neighbor_permutation_max_probability_delta": 1e-6,
            "arm_swap_max_complement_error": 1e-6,
        },
        "integration_gate": {
            "runs_only_if_semantic_gate_passes": True,
            "reuses_v4_7_world_and_ebm_hyperparameters": True,
            "trajectory_depth": 3,
            "bootstrap_samples": 1_000,
            "descriptive_over_primary_ci_lower_strictly_positive": True,
            "descriptive_over_structured_mean_strictly_positive": True,
            "descriptive_over_root_semantics_mean_strictly_positive": True,
            "relation_shuffle_degradation_mean_strictly_positive": True,
            "nonnegative_games_minimum": 6,
            "completion_selected_minimum_when_available": 1,
            "heuristic_weights": {
                "predicted_return": 1.0,
                "success": 3.0,
                "failure": -4.0,
                "productive": 0.05,
                "entropy": -0.10,
                "uncertainty": -0.10,
                "contradiction": -0.50,
            },
        },
        "authority_promoted": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir) / "frozen_manifest.json")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.12 manifest")
    expected = str(manifest["manifest_checksum"])
    payload = dict(manifest)
    payload.pop("manifest_checksum")
    if _checksum(payload) != expected:
        raise ValueError("V4.12 manifest checksum mismatch")
    if tuple(manifest["source_games"]) != SOURCE_TRAIN:
        raise ValueError("V4.12 source split drift")
    if tuple(manifest["active_effects"]) != ACTIVE_EFFECTS:
        raise ValueError("V4.12 active-effect drift")
    return manifest


def _validate_v411_source(v411_dir: Path, manifest: Mapping[str, Any]) -> None:
    qa = _read_json(v411_dir / "teacher_qa.json")
    checks = qa["checks"]
    required = ("collection_ready", "minimum_panels_each_game", "action_aligned_firewall")
    if not all(bool(checks.get(key)) for key in required):
        raise ValueError("V4.11 collection/firewall prerequisite failed")
    if tuple(qa["eligible_effects"]) != ACTIVE_EFFECTS:
        raise ValueError("V4.11 eligible-effect list changed")
    contract = manifest["teacher_contract"]
    expected = {
        "panels": int(contract["expected_panels"]),
        "arms": int(contract["expected_immediate_arms"]),
        "comparisons": int(contract["expected_fresh_comparisons"]),
    }
    actual = {
        "panels": int(qa["panels"]),
        "arms": int(qa["arms"]),
        "comparisons": int(qa["comparisons"]),
    }
    if actual != expected:
        raise ValueError(f"V4.11 capacity artifact drift: {actual}")


def _effect_pair_rows(
    records: Sequence[SemanticTeacherRecord],
    comparisons: Sequence[Any],
    variants: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    for comparison in comparisons:
        if not comparison.fresh:
            continue
        left_record = records[comparison.left]
        right_record = records[comparison.right]
        for effect in ACTIVE_EFFECTS:
            if not (
                left_record.applicable[effect]
                and right_record.applicable[effect]
                and left_record.labels[effect] != right_record.labels[effect]
            ):
                continue
            effect_index = SEMANTIC_EFFECTS.index(effect)
            target = float(left_record.labels[effect])
            payload: dict[str, float | str] = {
                "game_id": comparison.game_id,
                "panel_id": comparison.panel_id,
                "effect": effect,
            }
            for name, matrix in variants.items():
                delta = _logit(float(matrix[comparison.left, effect_index])) - _logit(
                    float(matrix[comparison.right, effect_index])
                )
                probability = float(_sigmoid(np.asarray([delta]))[0])
                payload[name] = float(
                    -target * math.log(max(probability, 1e-8))
                    - (1.0 - target) * math.log(max(1.0 - probability, 1e-8))
                )
            key = (
                f"{comparison.panel_id}:{comparison.left}:"
                f"{comparison.right}:{effect}"
            )
            rows[key] = payload
    return rows


def _pair_metrics(
    rows: Mapping[str, Mapping[str, float | str]],
    variants: Sequence[str],
) -> dict[str, Any]:
    result = {}
    for variant in variants:
        values = [float(row[variant]) for row in rows.values()]
        per_game = {}
        for game in SOURCE_TRAIN:
            selected = [
                float(row[variant])
                for row in rows.values()
                if row["game_id"] == game
            ]
            per_game[game] = {
                "effect_pairs": len(selected),
                "log_loss": float(np.mean(selected)) if selected else None,
            }
        result[variant] = {
            "effect_pairs": len(values),
            "log_loss": float(np.mean(values)) if values else 0.0,
            "per_game": per_game,
        }
    return result


def _active_brier_metrics(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    lookup = {int(index): local for local, index in enumerate(indices)}
    per_effect = {}
    briers = []
    for effect in ACTIVE_EFFECTS:
        effect_index = SEMANTIC_EFFECTS.index(effect)
        selected = [
            int(index) for index in indices if records[int(index)].applicable[effect]
        ]
        targets = np.asarray(
            [records[index].labels[effect] for index in selected], dtype=np.float64
        )
        predicted = np.asarray(
            [probabilities[lookup[index], effect_index] for index in selected],
            dtype=np.float64,
        )
        brier = float(np.mean((predicted - targets) ** 2))
        briers.append(brier)
        per_effect[effect] = {
            "applicable": len(selected),
            "positives": int(targets.sum()),
            "brier": brier,
        }
    return {"macro_brier": float(np.mean(briers)), "per_effect": per_effect}


def _active_per_game_brier(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    lookup = {int(index): local for local, index in enumerate(indices)}
    result = {}
    for game in SOURCE_TRAIN:
        values = []
        for effect in ACTIVE_EFFECTS:
            effect_index = SEMANTIC_EFFECTS.index(effect)
            selected = [
                int(index)
                for index in indices
                if records[int(index)].game_id == game
                and records[int(index)].applicable[effect]
            ]
            if not selected:
                continue
            targets = np.asarray(
                [records[index].labels[effect] for index in selected],
                dtype=np.float64,
            )
            predicted = np.asarray(
                [probabilities[lookup[index], effect_index] for index in selected],
                dtype=np.float64,
            )
            values.append(float(np.mean((predicted - targets) ** 2)))
        result[game] = float(np.mean(values))
    return result


def _active_ece(
    records: Sequence[SemanticTeacherRecord],
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    lookup = {int(index): local for local, index in enumerate(indices)}
    values = []
    for effect in ACTIVE_EFFECTS:
        effect_index = SEMANTIC_EFFECTS.index(effect)
        selected = [
            int(index) for index in indices if records[int(index)].applicable[effect]
        ]
        targets = np.asarray(
            [records[index].labels[effect] for index in selected], dtype=np.float64
        )
        predicted = np.asarray(
            [probabilities[lookup[index], effect_index] for index in selected],
            dtype=np.float64,
        )
        effect_ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            upper = lower + 0.1
            mask = (predicted >= lower) & (
                predicted <= upper if upper >= 1.0 else predicted < upper
            )
            if mask.any():
                effect_ece += float(mask.mean()) * abs(
                    float(predicted[mask].mean()) - float(targets[mask].mean())
                )
        values.append(effect_ece)
    return float(np.mean(values))


def evaluate_semantics(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Train and evaluate the eight-effect comparator under outer LOGO."""

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    _validate_v411_source(Path(v411_dir), manifest)
    panels = load_teacher_panels(v411_dir)
    records, comparisons, groups, fresh_set = _assemble_training_data(panels)
    parameters = manifest["training"]
    selected_device = _select_device(device)
    tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=16,
        mode="full",
    )
    root_tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=16,
        mode="root_only",
    )
    shuffled_tensors = tensorize_records(
        _shuffled_records(records),
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=16,
        mode="full",
    )
    reversed_tensors = _reverse_neighbor_tensors(tensors)

    count = len(records)
    effect_count = len(SEMANTIC_EFFECTS)
    full_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    root_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    action_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    shuffled_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    reversed_probabilities = np.zeros((count, effect_count), dtype=np.float64)
    fold_rows = []

    for fold_index, held_out_game in enumerate(SOURCE_TRAIN):
        train_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id != held_out_game
            ],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id == held_out_game
            ],
            dtype=np.int64,
        )
        root_model, root_summary = _fit_model(
            records,
            root_tensors,
            train_indices=train_indices,
            comparisons=comparisons,
            parameters=parameters,
            device=selected_device,
            seed=SEED + fold_index * 100,
            active_pair_effects=ACTIVE_EFFECTS,
        )
        full_model, full_summary = _fit_model(
            records,
            tensors,
            train_indices=train_indices,
            comparisons=comparisons,
            parameters=parameters,
            device=selected_device,
            seed=SEED + fold_index * 100 + 1,
            active_pair_effects=ACTIVE_EFFECTS,
        )
        train_root, _ = _predict_model(
            root_model, root_tensors, train_indices, device=selected_device
        )
        train_full, _ = _predict_model(
            full_model, tensors, train_indices, device=selected_device
        )
        train_residual = _centered_residual(
            train_full, train_root, train_indices, groups
        )
        alphas, _unused_shifts, alpha_summary = _select_alpha_and_shifts(
            records,
            train_indices,
            train_root,
            train_residual,
            alpha_grid=parameters["alpha_grid"],
        )
        inactive = [
            index
            for index, effect in enumerate(SEMANTIC_EFFECTS)
            if effect not in ACTIVE_EFFECTS
        ]
        alphas[inactive] = 0.0
        shifts = _game_balanced_shifts(
            records,
            train_indices,
            train_root + train_residual * alphas.reshape(1, -1),
        )
        root_shifts = _game_balanced_shifts(records, train_indices, train_root)
        test_root, _ = _predict_model(
            root_model, root_tensors, test_indices, device=selected_device
        )
        test_full, _ = _predict_model(
            full_model, tensors, test_indices, device=selected_device
        )
        test_shuffle, _ = _predict_model(
            full_model, shuffled_tensors, test_indices, device=selected_device
        )
        test_reversed, _ = _predict_model(
            full_model, reversed_tensors, test_indices, device=selected_device
        )
        test_residual = _centered_residual(
            test_full, test_root, test_indices, groups
        )
        shuffle_residual = _centered_residual(
            test_shuffle, test_root, test_indices, groups
        )
        reversed_residual = _centered_residual(
            test_reversed, test_root, test_indices, groups
        )
        full_probabilities[test_indices] = _calibrated(
            test_root, test_residual, alphas, shifts
        )
        shuffled_probabilities[test_indices] = _calibrated(
            test_root, shuffle_residual, alphas, shifts
        )
        reversed_probabilities[test_indices] = _calibrated(
            test_root, reversed_residual, alphas, shifts
        )
        root_probabilities[test_indices] = _sigmoid(
            test_root + root_shifts.reshape(1, -1)
        )
        action_probabilities[test_indices] = _action_only_probabilities(
            records, train_indices, test_indices
        )
        fold_rows.append(
            {
                "held_out_game": held_out_game,
                "root_only": root_summary,
                "full": full_summary,
                "active_alphas": {
                    effect: float(alphas[SEMANTIC_EFFECTS.index(effect)])
                    for effect in ACTIVE_EFFECTS
                },
                "alpha_selection": {
                    effect: alpha_summary[effect] for effect in ACTIVE_EFFECTS
                },
            }
        )

    prediction_rows = []
    for index, record in enumerate(records):
        prediction_rows.append(
            {
                "format_version": PREDICTION_VERSION,
                "trace_digest": record.trace_digest,
                "example_id": record.example_id,
                "game_id": record.game_id,
                "fresh_panel_arm": index in fresh_set,
                "probabilities": {
                    name: {
                        effect: float(matrix[index, effect_index])
                        for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                    }
                    for name, matrix in (
                        ("descriptive_distilled", full_probabilities),
                        ("root_only", root_probabilities),
                        ("action_only", action_probabilities),
                        ("relation_shuffle", shuffled_probabilities),
                    )
                },
            }
        )
    prediction_path = destination / "logo_predictions.jsonl"
    _write_jsonl(prediction_path, prediction_rows)

    variants = {
        "descriptive": full_probabilities,
        "root_only": root_probabilities,
        "action_only": action_probabilities,
        "relation_shuffle": shuffled_probabilities,
    }
    pair_rows = _effect_pair_rows(records, comparisons, variants)
    pair_metrics = _pair_metrics(pair_rows, tuple(variants))
    samples = int(manifest["semantic_gate"]["bootstrap_samples"])
    bootstrap_seed = int(manifest["semantic_gate"]["bootstrap_seed"])
    pair_gain_root = _bootstrap_panel_difference(
        pair_rows,
        left_key="root_only",
        right_key="descriptive",
        samples=samples,
        seed=bootstrap_seed,
    )
    pair_gain_action = _bootstrap_panel_difference(
        pair_rows,
        left_key="action_only",
        right_key="descriptive",
        samples=samples,
        seed=bootstrap_seed + 1,
    )
    shuffle_pair = _bootstrap_panel_difference(
        pair_rows,
        left_key="relation_shuffle",
        right_key="descriptive",
        samples=samples,
        seed=bootstrap_seed + 2,
    )

    by_digest = {record.trace_digest: index for index, record in enumerate(records)}
    fresh_indices = np.asarray(sorted(fresh_set), dtype=np.int64)
    brier_rows = _panel_brier_rows(
        panels,
        by_digest,
        records,
        {
            "descriptive": full_probabilities,
            "root_only": root_probabilities,
            "action_only": action_probabilities,
            "relation_shuffle": shuffled_probabilities,
        },
        ACTIVE_EFFECTS,
    )
    brier_gain = _bootstrap_panel_difference(
        brier_rows,
        left_key="root_only",
        right_key="descriptive",
        samples=samples,
        seed=bootstrap_seed + 3,
    )
    shuffle_brier = _bootstrap_panel_difference(
        brier_rows,
        left_key="relation_shuffle",
        right_key="descriptive",
        samples=samples,
        seed=bootstrap_seed + 4,
    )
    fresh_variants = {
        name: matrix[fresh_indices] for name, matrix in variants.items()
    }
    absolute_metrics = {
        name: _active_brier_metrics(records, fresh_indices, matrix)
        for name, matrix in fresh_variants.items()
    }
    per_game_full = _active_per_game_brier(
        records, fresh_indices, fresh_variants["descriptive"]
    )
    per_game_root = _active_per_game_brier(
        records, fresh_indices, fresh_variants["root_only"]
    )
    pair_nonnegative = sum(
        pair_metrics["descriptive"]["per_game"][game]["log_loss"] is not None
        and pair_metrics["root_only"]["per_game"][game]["log_loss"] is not None
        and pair_metrics["descriptive"]["per_game"][game]["log_loss"]
        <= pair_metrics["root_only"]["per_game"][game]["log_loss"]
        for game in SOURCE_TRAIN
    )
    absolute_nonnegative = sum(
        per_game_full[game] <= per_game_root[game] for game in SOURCE_TRAIN
    )
    active_columns = [SEMANTIC_EFFECTS.index(effect) for effect in ACTIVE_EFFECTS]
    full_identity, full_correct = _identity_metric(
        records,
        fresh_indices,
        fresh_variants["descriptive"][:, active_columns],
        seed=SEED,
    )
    root_identity, root_correct = _identity_metric(
        records,
        fresh_indices,
        fresh_variants["root_only"][:, active_columns],
        seed=SEED,
    )
    identity_increment = _bootstrap_identity_increment(
        records,
        fresh_indices,
        full_correct,
        root_correct,
        samples=samples,
        seed=bootstrap_seed + 5,
    )
    permutation_delta = float(
        np.max(
            np.abs(
                full_probabilities[:, active_columns]
                - reversed_probabilities[:, active_columns]
            )
        )
    )
    arm_swap_error = 0.0
    for comparison in comparisons:
        if not comparison.fresh:
            continue
        for effect in ACTIVE_EFFECTS:
            if not (
                records[comparison.left].applicable[effect]
                and records[comparison.right].applicable[effect]
                and records[comparison.left].labels[effect]
                != records[comparison.right].labels[effect]
            ):
                continue
            column = SEMANTIC_EFFECTS.index(effect)
            delta = _logit(full_probabilities[comparison.left, column]) - _logit(
                full_probabilities[comparison.right, column]
            )
            forward = float(_sigmoid(np.asarray([delta]))[0])
            reverse = float(_sigmoid(np.asarray([-delta]))[0])
            arm_swap_error = max(arm_swap_error, abs(forward + reverse - 1.0))

    full_ece = _active_ece(
        records, fresh_indices, fresh_variants["descriptive"]
    )
    root_ece = _active_ece(records, fresh_indices, fresh_variants["root_only"])
    thresholds = manifest["semantic_gate"]
    checks = {
        "v4_11_collection_and_firewall_ready": True,
        "effect_pair_gain_over_root_ci": pair_gain_root["ci_lower"] > 0.0,
        "effect_pair_gain_over_action_ci": pair_gain_action["ci_lower"] > 0.0,
        "relation_shuffle_pair_degradation_ci": shuffle_pair["ci_lower"] > 0.0,
        "active_macro_brier_gain_over_root_ci": brier_gain["ci_lower"] > 0.0,
        "relation_shuffle_brier_degradation_ci": shuffle_brier["ci_lower"] > 0.0,
        "ece_not_worse_than_root_only": full_ece <= root_ece,
        "pair_nonnegative_games": (
            pair_nonnegative >= int(thresholds["nonnegative_games_minimum"])
        ),
        "absolute_nonnegative_games": (
            absolute_nonnegative >= int(thresholds["nonnegative_games_minimum"])
        ),
        "identity_increment": (
            identity_increment["ci_upper"]
            <= float(thresholds["identity_increment_ci_upper_maximum"])
        ),
        "neighbor_permutation_invariance": (
            permutation_delta
            <= float(thresholds["neighbor_permutation_max_probability_delta"])
        ),
        "arm_swap_antisymmetry": (
            arm_swap_error
            <= float(thresholds["arm_swap_max_complement_error"])
        ),
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "format_version": SEMANTIC_RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "verdict": (
            "READY_FOR_CONDITIONAL_SOURCE_ARCHITECTURE_PILOT"
            if passed
            else "DESCRIPTIVE_SEMANTICS_NOT_SUPPORTED"
        ),
        "semantic_gate_passed": passed,
        "active_effects": list(ACTIVE_EFFECTS),
        "device": selected_device,
        "records": len(records),
        "fresh_panel_arms": len(fresh_indices),
        "fresh_effect_pairs": len(pair_rows),
        "checks": checks,
        "effect_pair_metrics": pair_metrics,
        "effect_pair_gain_over_root": pair_gain_root,
        "effect_pair_gain_over_action": pair_gain_action,
        "relation_shuffle_pair_degradation": shuffle_pair,
        "absolute_metrics": absolute_metrics,
        "active_macro_brier_gain_over_root": brier_gain,
        "relation_shuffle_brier_degradation": shuffle_brier,
        "ece": {"descriptive": full_ece, "root_only": root_ece},
        "nonnegative_games": {
            "effect_pair": pair_nonnegative,
            "absolute_brier": absolute_nonnegative,
        },
        "identity_probe": {
            "descriptive": full_identity,
            "root_only": root_identity,
            "increment_bootstrap": identity_increment,
        },
        "neighbor_permutation_max_probability_delta": permutation_delta,
        "arm_swap_max_complement_error": arm_swap_error,
        "folds": fold_rows,
        "world_model_fitted": False,
        "ebm_fitted": False,
        "authority_promoted": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {
            "logo_predictions": _file_sha256(prediction_path),
            "v4_11_teacher_panels": _file_sha256(
                Path(v411_dir) / "teacher_panels.jsonl"
            ),
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "semantic_result.json", result)
    export_slot_semantics(output_dir=destination)
    return result


def export_slot_semantics(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    predictions = {
        str(row["trace_digest"]): row["probabilities"]
        for row in _read_jsonl(destination / "logo_predictions.jsonl")
    }
    roots = load_complete_roots(v43_dir)
    examples = load_slot_examples(roots)
    by_position = {
        (item.root_key, item.path, item.side): item for item in examples
    }
    rows = []
    for root in roots:
        for path, pair in sorted(root.tree.items()):
            for side, arm in zip("LR", (pair.left, pair.right)):
                item = by_position[(root.root_key, path, side)]
                if arm.trace.trace_digest not in predictions:
                    raise ValueError("V4.3 slot lacks V4.12 LOGO prediction")
                for variant in (
                    "descriptive_distilled",
                    "root_only",
                    "action_only",
                    "relation_shuffle",
                ):
                    probabilities = predictions[arm.trace.trace_digest][variant]
                    rows.append(
                        {
                            "format_version": SLOT_EXPORT_VERSION,
                            "slot_id": item.slot.slot_id,
                            "example_id": item.example_id,
                            "game_id": item.game_id,
                            "trace_digest": arm.trace.trace_digest,
                            "variant": variant,
                            "effect_probabilities": {
                                effect: float(probabilities[effect])
                                for effect in SEMANTIC_EFFECTS
                            },
                            "source": f"descriptive_semantics_{variant}_logo_v4_12",
                            "support": 0,
                        }
                    )
    path = destination / "v4_7_slot_semantics.jsonl"
    _write_jsonl(path, rows)
    summary: dict[str, Any] = {
        "format_version": SLOT_EXPORT_VERSION,
        "slots": len(examples),
        "rows": len(rows),
        "variants": [
            "descriptive_distilled",
            "root_only",
            "action_only",
            "relation_shuffle",
        ],
        "sha256": _file_sha256(path),
    }
    summary["checksum"] = _checksum(summary)
    _write_json(destination / "v4_7_slot_export.json", summary)
    return summary


def _load_slot_inputs(
    destination: Path,
    examples: Sequence[SlotExample],
    *,
    variant: str,
) -> tuple[tuple[SlotExample, ...], dict[str, SlotAnnotation]]:
    rows = {
        str(row["slot_id"]): row
        for row in _read_jsonl(destination / "v4_7_slot_semantics.jsonl")
        if row["variant"] == variant
    }
    transformed = []
    annotations = {}
    for item in examples:
        row = rows.get(item.slot.slot_id)
        if row is None:
            raise ValueError(f"missing {variant} semantics for {item.example_id}")
        probabilities = row["effect_probabilities"]
        slot = replace(
            item.slot,
            semantic_signature={
                **dict(item.slot.semantic_signature),
                **{
                    f"v412.effect.{effect}": float(probabilities[effect])
                    for effect in ACTIVE_EFFECTS
                },
            },
        )
        transformed.append(replace(item, slot=slot))
        annotations[item.slot.slot_id] = SlotAnnotation(
            slot_id=item.slot.slot_id,
            effect_probabilities={
                effect: float(probabilities[effect]) for effect in SLOT_EFFECTS
            },
            source=str(row["source"]),
            support=0,
        )
    return tuple(transformed), annotations


def _oracle_inputs(
    examples: Sequence[SlotExample],
    roots: Sequence[Any],
    *,
    v411_dir: str | Path,
) -> tuple[tuple[SlotExample, ...], dict[str, SlotAnnotation]]:
    panels = load_teacher_panels(v411_dir)
    records, _comparisons, _groups, _fresh = _assemble_training_data(panels)
    by_digest = {record.trace_digest: record for record in records}
    by_position = {
        (item.root_key, item.path, item.side): item for item in examples
    }
    by_slot = {}
    for root in roots:
        for path, pair in root.tree.items():
            for side, arm in zip("LR", (pair.left, pair.right)):
                item = by_position[(root.root_key, path, side)]
                record = by_digest.get(arm.trace.trace_digest)
                if record is None:
                    raise ValueError(f"missing oracle record for {item.example_id}")
                by_slot[item.slot.slot_id] = record
    transformed = []
    annotations = {}
    for item in examples:
        record = by_slot.get(item.slot.slot_id)
        if record is None:
            raise ValueError(f"missing oracle semantics for {item.example_id}")
        slot = replace(
            item.slot,
            semantic_signature={
                **dict(item.slot.semantic_signature),
                **{
                    f"v412.effect.{effect}": float(record.labels[effect])
                    for effect in ACTIVE_EFFECTS
                },
            },
        )
        transformed.append(replace(item, slot=slot))
        annotations[item.slot.slot_id] = item.annotation(
            source="oracle_semantics_v4_12"
        )
    return tuple(transformed), annotations


def _heuristic_path(
    root: Any,
    predictions: Mapping[str, Any],
    by_position: Mapping[tuple[str, str, str], SlotExample],
    weights: Mapping[str, float],
) -> str:
    choices = []
    for bits in itertools.product("LR", repeat=3):
        path = "".join(bits)
        features = _trajectory_features(
            root, path, predictions, by_position, depth=3
        )
        score = (
            float(weights["predicted_return"]) * features[0]
            + float(weights["success"]) * features[1]
            + float(weights["failure"]) * features[2]
            + float(weights["productive"]) * features[3]
            + float(weights["entropy"]) * features[4]
            + float(weights["uncertainty"]) * features[5]
            + float(weights["contradiction"]) * features[7]
        )
        choices.append((score, path))
    return max(choices, key=lambda row: (row[0], row[1]))[1]


def evaluate_integration(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    """Run the complete source-only architecture if the semantic gate passed."""

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    semantic = _read_json(destination / "semantic_result.json")
    if not semantic["semantic_gate_passed"]:
        result: dict[str, Any] = {
            "format_version": INTEGRATION_RESULT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "semantic_result_checksum": semantic["result_checksum"],
            "verdict": "SKIPPED_SEMANTIC_GATE_FAILED",
            "semantic_gate_passed": False,
            "world_model_fitted": False,
            "ebm_fitted": False,
            "authority_promoted": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "historical_opened": False,
            "live_environment_opened": False,
        }
        result["result_checksum"] = _checksum(result)
        _write_json(destination / "integration_result.json", result)
        return result

    roots = load_complete_roots(v43_dir)
    examples = load_slot_examples(roots)
    descriptive, descriptive_annotations = _load_slot_inputs(
        destination, examples, variant="descriptive_distilled"
    )
    root_semantics, root_annotations = _load_slot_inputs(
        destination, examples, variant="root_only"
    )
    shuffled, shuffled_annotations = _load_slot_inputs(
        destination, examples, variant="relation_shuffle"
    )
    oracle, oracle_annotations = _oracle_inputs(
        examples, roots, v411_dir=v411_dir
    )
    variants = {
        "structured": examples,
        "root_semantics": root_semantics,
        "descriptive": descriptive,
        "oracle": oracle,
    }
    annotations = {
        "structured": None,
        "root_semantics": root_annotations,
        "descriptive": descriptive_annotations,
        "oracle": oracle_annotations,
    }
    by_position = {
        (item.root_key, item.path, item.side): item for item in descriptive
    }
    decisions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    held_predictions: dict[str, dict[str, Any]] = {
        name: {} for name in variants
    }
    games = sorted({root.game_id for root in roots})
    weights = manifest["integration_gate"]["heuristic_weights"]

    for fold_index, held_out_game in enumerate(games):
        training_roots = tuple(root for root in roots if root.game_id != held_out_game)
        validation_roots = tuple(root for root in roots if root.game_id == held_out_game)
        models = {}
        oof = {}
        validation_predictions = {}
        for offset, name in enumerate(variants):
            training = tuple(
                item for item in variants[name] if item.game_id != held_out_game
            )
            validation = tuple(
                item for item in variants[name] if item.game_id == held_out_game
            )
            model, nested = _fit_nested_world(
                training,
                annotations=annotations[name],
                use_annotations=name != "structured",
                seed=SEED + fold_index * 100 + offset * 10,
            )
            models[name] = model
            oof[name] = nested
            validation_predictions[name] = model.predict(
                validation, annotations[name]
            )
            held_predictions[name].update(validation_predictions[name])
        training_examples = tuple(
            item for item in descriptive if item.game_id != held_out_game
        )
        ebms = {
            name: _train_ebm(
                training_roots,
                oof[name],
                by_position,
                depth=3,
                seed=SEED + 1_000 + fold_index * 10 + offset,
            )
            for offset, name in enumerate(variants)
        }
        shuffled_validation = tuple(
            item for item in shuffled if item.game_id == held_out_game
        )
        shuffled_predictions = models["descriptive"].predict(
            shuffled_validation, shuffled_annotations
        )
        primary = _select_primary_baseline(training_roots)
        folds.append(
            {
                "held_out_game": held_out_game,
                "training_games": sorted(
                    {root.game_id for root in training_roots}
                ),
                "training_slots": len(training_examples),
                "validation_slots": sum(
                    item.game_id == held_out_game for item in descriptive
                ),
                "primary_baseline": primary,
                "ebm_pairs": {
                    name: model.trained_pairs for name, model in ebms.items()
                },
            }
        )
        for root in validation_roots:
            baseline_paths = {
                "deterministic_left": "L",
                "action_only": _action_only_choice(root, training_roots),
                "action_sequence_only": _sequence_only_choice(
                    root, training_roots
                ),
            }
            for name, path in baseline_paths.items():
                decisions.append(
                    _decision_row(root, method=name, selected_path=path)
                )
            decisions.append(
                _decision_row(
                    root,
                    method="primary_baseline",
                    selected_path=baseline_paths[primary],
                    baseline_method=primary,
                )
            )
            method_variants = (
                ("structured_depth3_ebm", "structured"),
                ("root_semantics_depth3_ebm", "root_semantics"),
                ("descriptive_semantics_depth3_ebm", "descriptive"),
                ("oracle_semantics_depth3_ebm", "oracle"),
            )
            for method, name in method_variants:
                path = _select_path(
                    root,
                    validation_predictions[name],
                    by_position,
                    ebms[name],
                    depth=3,
                )
                decisions.append(
                    _decision_row(root, method=method, selected_path=path)
                )
            decisions.append(
                _decision_row(
                    root,
                    method="descriptive_semantics_depth3_heuristic",
                    selected_path=_heuristic_path(
                        root,
                        validation_predictions["descriptive"],
                        by_position,
                        weights,
                    ),
                )
            )
            decisions.append(
                _decision_row(
                    root,
                    method="descriptive_relation_shuffle_depth3_ebm",
                    selected_path=_select_path(
                        root,
                        shuffled_predictions,
                        by_position,
                        ebms["descriptive"],
                        depth=3,
                    ),
                )
            )

    decisions_path = destination / "decisions.jsonl"
    folds_path = destination / "folds.jsonl"
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(folds_path, folds)
    methods = sorted({str(row["method"]) for row in decisions})
    by_method = {
        method: [row for row in decisions if row["method"] == method]
        for method in methods
    }
    metrics = {method: _summarize(rows) for method, rows in by_method.items()}
    descriptive_rows = by_method["descriptive_semantics_depth3_ebm"]
    comparisons = {
        "descriptive_over_primary": _paired_bootstrap(
            descriptive_rows,
            by_method["primary_baseline"],
            seed=SEED,
        ),
        "descriptive_over_structured": _paired_bootstrap(
            descriptive_rows,
            by_method["structured_depth3_ebm"],
            seed=SEED + 1,
        ),
        "descriptive_over_root_semantics": _paired_bootstrap(
            descriptive_rows,
            by_method["root_semantics_depth3_ebm"],
            seed=SEED + 2,
        ),
        "descriptive_over_relation_shuffle": _paired_bootstrap(
            descriptive_rows,
            by_method["descriptive_relation_shuffle_depth3_ebm"],
            seed=SEED + 3,
        ),
        "descriptive_heuristic_over_primary": _paired_bootstrap(
            by_method["descriptive_semantics_depth3_heuristic"],
            by_method["primary_baseline"],
            seed=SEED + 4,
        ),
        "oracle_over_primary": _paired_bootstrap(
            by_method["oracle_semantics_depth3_ebm"],
            by_method["primary_baseline"],
            seed=SEED + 5,
        ),
    }
    primary_per_game = metrics["primary_baseline"]["per_game"]
    descriptive_per_game = metrics["descriptive_semantics_depth3_ebm"]["per_game"]
    nonnegative_games = sum(
        descriptive_per_game[game]["mean_utility"]
        >= primary_per_game[game]["mean_utility"]
        for game in games
    )
    completion = _completion_capture(roots, decisions)
    selected_completions = completion["selected_by_method"].get(
        "descriptive_semantics_depth3_ebm", 0
    )
    checks = {
        "semantic_gate_passed": True,
        "descriptive_over_primary_ci": (
            comparisons["descriptive_over_primary"]["ci_low"] > 0.0
        ),
        "descriptive_over_structured_mean": (
            comparisons["descriptive_over_structured"]["mean_gain"] > 0.0
        ),
        "descriptive_over_root_semantics_mean": (
            comparisons["descriptive_over_root_semantics"]["mean_gain"] > 0.0
        ),
        "relation_shuffle_degradation_mean": (
            comparisons["descriptive_over_relation_shuffle"]["mean_gain"] > 0.0
        ),
        "nonnegative_games": (
            nonnegative_games
            >= int(manifest["integration_gate"]["nonnegative_games_minimum"])
        ),
        "completion_selected_when_available": (
            completion["opportunities"] == 0 or selected_completions >= 1
        ),
    }
    passed = all(checks.values())
    result = {
        "format_version": INTEGRATION_RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "semantic_result_checksum": semantic["result_checksum"],
        "verdict": (
            "EXPLORATORY_GLOBAL_ARCHITECTURE_SUPPORTED"
            if passed
            else "EXPLORATORY_GLOBAL_ARCHITECTURE_NOT_SUPPORTED"
        ),
        "semantic_gate_passed": True,
        "integration_gate_passed": passed,
        "checks": checks,
        "roots": len(roots),
        "nodes": len(_nodes(examples)),
        "slots": len(examples),
        "metrics": metrics,
        "comparisons": comparisons,
        "nonnegative_games": nonnegative_games,
        "completion_capture": completion,
        "world_model_metrics": {
            name: _world_metrics(variants[name], predictions)
            for name, predictions in held_predictions.items()
        },
        "world_model_fitted": True,
        "ebm_fitted": True,
        "authority_promoted": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {
            "semantic_slot_export": _file_sha256(
                destination / "v4_7_slot_semantics.jsonl"
            ),
            "decisions": _file_sha256(decisions_path),
            "folds": _file_sha256(folds_path),
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "integration_result.json", result)
    return result


def run(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    if not (destination / "frozen_manifest.json").exists():
        freeze_manifest(output_dir=destination)
    semantic = evaluate_semantics(output_dir=destination, device=device)
    integration = evaluate_integration(output_dir=destination)
    return {"semantic": semantic, "integration": integration}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    semantics = subparsers.add_parser("evaluate-semantics")
    semantics.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    semantics.add_argument("--device", default="cuda:0")
    integration = subparsers.add_parser("evaluate-integration")
    integration.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(output_dir=args.output_dir)
    elif args.command == "evaluate-semantics":
        payload = evaluate_semantics(
            output_dir=args.output_dir, device=args.device
        )
    elif args.command == "evaluate-integration":
        payload = evaluate_integration(output_dir=args.output_dir)
    else:
        payload = run(output_dir=args.output_dir, device=args.device)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_EFFECTS",
    "evaluate_integration",
    "evaluate_semantics",
    "export_slot_semantics",
    "freeze_manifest",
    "load_manifest",
    "run",
]
