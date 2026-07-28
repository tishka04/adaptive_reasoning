"""SAGE12 V4.13 unconditional semantic-bottleneck architecture curve.

Every registered semantic condition is run through the complete source-only
V4.7 world-model, depth-three trajectory, EBM and controller stack.  Unlike
V4.12, no intermediate semantic gate can skip the global evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN

from .compiler import SLOT_EFFECTS, SlotAnnotation
from .counterfactual_semantic_panels_v4_11 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V411_DIR,
)
from .descriptive_semantic_integration_v4_12 import (
    ACTIVE_EFFECTS,
    DEFAULT_OUTPUT_DIR as DEFAULT_V412_DIR,
    _heuristic_path,
    _load_slot_inputs,
    _oracle_inputs,
)
from .integration_pilot import load_complete_roots
from .integration_pilot_v4_7 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V47_DIR,
    DEFAULT_V43_DIR,
    SlotExample,
    _action_only_choice,
    _decision_row,
    _fit_nested_world,
    _leaf_value,
    _nodes,
    _paired_bootstrap,
    _select_path,
    _select_primary_baseline,
    _sequence_only_choice,
    _summarize,
    _train_ebm,
    _true_world_predictions,
    _world_metrics,
    load_slot_examples,
)
from .semantic_adapter_v4_8 import _completion_capture
from .semantic_teacher_v4_9 import (
    _checksum,
    _file_sha256,
    _read_json,
    _write_json,
    _write_jsonl,
)

FORMAT_VERSION = "sage12-semantic-bottleneck-curve-v4.13"
MANIFEST_VERSION = "sage12-semantic-bottleneck-manifest-v4.13"
RESULT_VERSION = "sage12-semantic-bottleneck-result-v4.13"
DECISION_VERSION = "sage12-semantic-bottleneck-decision-v4.13"

DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "semantic_bottleneck_curve_v4_13"
SEED = 5_130
NOISE_LEVELS = {
    "oracle_100": 0.00,
    "oracle_90": 0.10,
    "oracle_75": 0.25,
    "oracle_50": 0.50,
}
CORRUPTED_EFFECTS = tuple(dict.fromkeys(SLOT_EFFECTS + ACTIVE_EFFECTS))


def _source_fingerprints(
    v412_dir: Path,
    v43_dir: Path,
    v47_dir: Path,
) -> dict[str, Any]:
    paths = (
        v412_dir / "frozen_manifest.json",
        v412_dir / "semantic_result.json",
        v412_dir / "integration_result.json",
        v412_dir / "logo_predictions.jsonl",
        v412_dir / "v4_7_slot_semantics.jsonl",
        v43_dir / "frozen_manifest.json",
        v43_dir / "source_train_collection_manifest.json",
        v47_dir / "frozen_manifest.json",
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
    v412_dir: str | Path = DEFAULT_V412_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    v47_dir: str | Path = DEFAULT_V47_DIR,
) -> dict[str, Any]:
    """Freeze the unconditional component ladder before observing V4.13."""

    v412 = _read_json(Path(v412_dir) / "semantic_result.json")
    if tuple(v412["active_effects"]) != ACTIVE_EFFECTS:
        raise ValueError("V4.12 active-effect drift")
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "source_games": list(SOURCE_TRAIN),
        "source_fingerprints": _source_fingerprints(
            Path(v412_dir), Path(v43_dir), Path(v47_dir)
        ),
        "evaluation": {
            "seed": SEED,
            "outer_split": "leave_one_source_train_game_out",
            "trajectory_depth": 3,
            "bootstrap_samples": 1_000,
            "all_conditions_run_unconditionally": True,
            "reuse_v4_7_world_and_ebm_hyperparameters": True,
            "reuse_v4_3_candidate_complete_trees": True,
            "future_tree_topology_is_non_deployable": True,
            "live_win_rate_claimed": False,
        },
        "semantic_conditions": {
            "structured_no_teacher": {
                "kind": "no_teacher_semantics",
                "fit_world_model": True,
            },
            "learned_v4_12": {
                "kind": "held_game_v4_12_probabilities",
                "fit_world_model": True,
            },
            **{
                name: {
                    "kind": "deterministically_flipped_oracle_bits",
                    "flip_probability": rate,
                    "fit_world_model": True,
                }
                for name, rate in NOISE_LEVELS.items()
            },
            "learned_root_only_stress": {
                "kind": "predict_with_v4_12_root_anchor",
                "fit_world_model": False,
            },
            "learned_relation_shuffle_stress": {
                "kind": "predict_with_v4_12_relation_shuffle",
                "fit_world_model": False,
            },
            "learned_root_reuse_stress": {
                "kind": "depth_three_reuses_current_root_slots",
                "fit_world_model": False,
            },
            "oracle_root_reuse_stress": {
                "kind": "depth_three_reuses_current_root_slots",
                "fit_world_model": False,
            },
            "true_world_oracle": {
                "kind": "executed_effects_and_utility",
                "fit_world_model": False,
            },
            "oracle_energy": {
                "kind": "maximum_executed_leaf_return",
                "fit_world_model": False,
            },
        },
        "corruption": {
            "seed": SEED,
            "effects": list(CORRUPTED_EFFECTS),
            "deterministic_hash": "sha256(seed|slot_id|effect)",
            "probabilities_after_flip": [0.0, 1.0],
            "applied_to_training_and_held_game_slots": True,
            "strict_logo_world_fitting": True,
            "world_and_ebm_seeds_shared_across_oracle_curve": True,
        },
        "diagnostic_decisions": {
            "true_world_over_primary_ci_lower_strictly_positive": True,
            "oracle_semantics_over_primary_ci_lower_strictly_positive": True,
            "oracle_semantics_over_structured_mean_strictly_positive": True,
            "learned_over_primary_ci_lower_strictly_positive": True,
            "completion_selected_minimum_when_available": 1,
            "corruption_spearman_minimum": 0.80,
            "nonnegative_games_minimum": 6,
        },
        "heuristic_weights": {
            "predicted_return": 1.0,
            "success": 3.0,
            "failure": -4.0,
            "productive": 0.05,
            "entropy": -0.10,
            "uncertainty": -0.10,
            "contradiction": -0.50,
        },
        "authority_promoted": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(Path(output_dir) / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir) / "frozen_manifest.json")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.13 manifest")
    expected = str(manifest["manifest_checksum"])
    payload = dict(manifest)
    payload.pop("manifest_checksum")
    if _checksum(payload) != expected:
        raise ValueError("V4.13 manifest checksum mismatch")
    if tuple(manifest["source_games"]) != SOURCE_TRAIN:
        raise ValueError("V4.13 source split drift")
    return manifest


def _hash_uniform(slot_id: str, effect: str, *, seed: int) -> float:
    digest = hashlib.sha256(
        f"{seed}|{slot_id}|{effect}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _oracle_effect(
    item: SlotExample,
    annotation: SlotAnnotation,
    effect: str,
) -> float:
    if effect in ACTIVE_EFFECTS:
        return float(item.slot.semantic_signature[f"v412.effect.{effect}"])
    return float(annotation.effect_probabilities[effect])


def _corrupt_oracle(
    examples: Sequence[SlotExample],
    annotations: Mapping[str, SlotAnnotation],
    *,
    flip_probability: float,
    seed: int,
    source: str,
) -> tuple[tuple[SlotExample, ...], dict[str, SlotAnnotation], dict[str, Any]]:
    """Deterministically flip oracle bits without exposing the flip key."""

    transformed = []
    output_annotations = {}
    correct = 0
    total = 0
    per_effect = {}
    for item in examples:
        annotation = annotations[item.slot.slot_id]
        values = {}
        for effect in CORRUPTED_EFFECTS:
            truth = _oracle_effect(item, annotation, effect)
            flipped = _hash_uniform(item.slot.slot_id, effect, seed=seed) < float(
                flip_probability
            )
            values[effect] = 1.0 - truth if flipped else truth
            correct += int(values[effect] == truth)
            total += 1
            bucket = per_effect.setdefault(effect, {"correct": 0, "rows": 0})
            bucket["correct"] += int(values[effect] == truth)
            bucket["rows"] += 1
        slot = replace(
            item.slot,
            semantic_signature={
                **dict(item.slot.semantic_signature),
                **{
                    f"v412.effect.{effect}": float(values[effect])
                    for effect in ACTIVE_EFFECTS
                },
            },
        )
        transformed.append(replace(item, slot=slot))
        output_annotations[item.slot.slot_id] = SlotAnnotation(
            slot_id=item.slot.slot_id,
            effect_probabilities={
                effect: float(values[effect]) for effect in SLOT_EFFECTS
            },
            source=source,
            support=0,
        )
    accuracy = correct / total if total else 0.0
    return (
        tuple(transformed),
        output_annotations,
        {
            "requested_flip_probability": float(flip_probability),
            "observed_bit_accuracy": float(accuracy),
            "bits": total,
            "per_effect": {
                effect: {
                    **row,
                    "accuracy": row["correct"] / row["rows"],
                }
                for effect, row in per_effect.items()
            },
        },
    )


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def evaluate(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v412_dir: str | Path = DEFAULT_V412_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    """Execute every semantic and oracle condition through the global stack."""

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    roots = load_complete_roots(v43_dir)
    original = load_slot_examples(roots)
    learned, learned_annotations = _load_slot_inputs(
        Path(v412_dir), original, variant="descriptive_distilled"
    )
    learned_root, learned_root_annotations = _load_slot_inputs(
        Path(v412_dir), original, variant="root_only"
    )
    learned_shuffle, learned_shuffle_annotations = _load_slot_inputs(
        Path(v412_dir), original, variant="relation_shuffle"
    )
    oracle, oracle_annotations = _oracle_inputs(
        original, roots, v411_dir=v411_dir
    )

    variants: dict[str, tuple[SlotExample, ...]] = {
        "structured": original,
        "learned_v4_12": learned,
        "oracle_100": oracle,
    }
    annotations: dict[str, Mapping[str, SlotAnnotation] | None] = {
        "structured": None,
        "learned_v4_12": learned_annotations,
        "oracle_100": oracle_annotations,
    }
    corruption = {
        "oracle_100": {
            "requested_flip_probability": 0.0,
            "observed_bit_accuracy": 1.0,
            "bits": len(oracle) * len(CORRUPTED_EFFECTS),
        }
    }
    for name, rate in NOISE_LEVELS.items():
        if rate == 0.0:
            continue
        rows, row_annotations, summary = _corrupt_oracle(
            oracle,
            oracle_annotations,
            flip_probability=rate,
            seed=SEED,
            source=f"{name}_v4_13",
        )
        variants[name] = rows
        annotations[name] = row_annotations
        corruption[name] = summary

    fitted_names = (
        "structured",
        "learned_v4_12",
        "oracle_100",
        "oracle_90",
        "oracle_75",
        "oracle_50",
    )
    by_position = {
        (item.root_key, item.path, item.side): item for item in learned
    }
    games = sorted({root.game_id for root in roots})
    decisions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    held_predictions: dict[str, dict[str, Any]] = {
        name: {} for name in fitted_names
    }
    heuristic_weights = manifest["heuristic_weights"]

    for fold_index, held_out_game in enumerate(games):
        training_roots = tuple(root for root in roots if root.game_id != held_out_game)
        validation_roots = tuple(root for root in roots if root.game_id == held_out_game)
        models = {}
        oof = {}
        validation_predictions = {}
        for name in fitted_names:
            training = tuple(
                item for item in variants[name] if item.game_id != held_out_game
            )
            validation = tuple(
                item for item in variants[name] if item.game_id == held_out_game
            )
            model_seed = (
                SEED + fold_index * 100
                if name == "structured"
                else SEED + fold_index * 100 + 10
                if name == "learned_v4_12"
                else SEED + fold_index * 100 + 20
            )
            model, nested = _fit_nested_world(
                training,
                annotations=annotations[name],
                use_annotations=name != "structured",
                seed=model_seed,
            )
            models[name] = model
            oof[name] = nested
            validation_predictions[name] = model.predict(
                validation, annotations[name]
            )
            held_predictions[name].update(validation_predictions[name])

        training_original = tuple(
            item for item in original if item.game_id != held_out_game
        )
        validation_original = tuple(
            item for item in original if item.game_id == held_out_game
        )
        true_training = _true_world_predictions(training_original)
        true_validation = _true_world_predictions(validation_original)
        ebms = {
            name: _train_ebm(
                training_roots,
                oof[name],
                by_position,
                depth=3,
                seed=(
                    SEED + 1_000 + fold_index
                    if name == "structured"
                    else SEED + 2_000 + fold_index
                    if name == "learned_v4_12"
                    else SEED + 3_000 + fold_index
                ),
            )
            for name in fitted_names
        }
        ebms["learned_root_reuse"] = _train_ebm(
            training_roots,
            oof["learned_v4_12"],
            by_position,
            depth=3,
            seed=SEED + 4_000 + fold_index,
            root_reuse=True,
        )
        ebms["oracle_root_reuse"] = _train_ebm(
            training_roots,
            oof["oracle_100"],
            by_position,
            depth=3,
            seed=SEED + 5_000 + fold_index,
            root_reuse=True,
        )
        ebms["true_world"] = _train_ebm(
            training_roots,
            true_training,
            by_position,
            depth=3,
            seed=SEED + 6_000 + fold_index,
        )
        root_validation = tuple(
            item for item in learned_root if item.game_id == held_out_game
        )
        shuffle_validation = tuple(
            item for item in learned_shuffle if item.game_id == held_out_game
        )
        root_predictions = models["learned_v4_12"].predict(
            root_validation, learned_root_annotations
        )
        shuffle_predictions = models["learned_v4_12"].predict(
            shuffle_validation, learned_shuffle_annotations
        )
        primary = _select_primary_baseline(training_roots)
        folds.append(
            {
                "format_version": FORMAT_VERSION,
                "held_out_game": held_out_game,
                "training_games": sorted(
                    {root.game_id for root in training_roots}
                ),
                "training_slots": len(training_original),
                "validation_slots": len(validation_original),
                "primary_baseline": primary,
                "fitted_conditions": list(fitted_names),
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
            for name in fitted_names:
                decisions.append(
                    _decision_row(
                        root,
                        method=f"{name}_depth3_ebm",
                        selected_path=_select_path(
                            root,
                            validation_predictions[name],
                            by_position,
                            ebms[name],
                            depth=3,
                        ),
                    )
                )
            decisions.extend(
                (
                    _decision_row(
                        root,
                        method="learned_root_only_stress_depth3_ebm",
                        selected_path=_select_path(
                            root,
                            root_predictions,
                            by_position,
                            ebms["learned_v4_12"],
                            depth=3,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="learned_relation_shuffle_stress_depth3_ebm",
                        selected_path=_select_path(
                            root,
                            shuffle_predictions,
                            by_position,
                            ebms["learned_v4_12"],
                            depth=3,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="learned_root_reuse_stress_depth3_ebm",
                        selected_path=_select_path(
                            root,
                            validation_predictions["learned_v4_12"],
                            by_position,
                            ebms["learned_root_reuse"],
                            depth=3,
                            root_reuse=True,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="oracle_root_reuse_stress_depth3_ebm",
                        selected_path=_select_path(
                            root,
                            validation_predictions["oracle_100"],
                            by_position,
                            ebms["oracle_root_reuse"],
                            depth=3,
                            root_reuse=True,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="learned_v4_12_depth3_heuristic",
                        selected_path=_heuristic_path(
                            root,
                            validation_predictions["learned_v4_12"],
                            by_position,
                            heuristic_weights,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="oracle_100_depth3_heuristic",
                        selected_path=_heuristic_path(
                            root,
                            validation_predictions["oracle_100"],
                            by_position,
                            heuristic_weights,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="true_world_depth3_ebm",
                        selected_path=_select_path(
                            root,
                            true_validation,
                            by_position,
                            ebms["true_world"],
                            depth=3,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="oracle_energy",
                        selected_path=max(
                            (
                                "".join(bits)
                                for bits in itertools.product("LR", repeat=3)
                            ),
                            key=lambda path: (_leaf_value(root, path), path),
                        ),
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
    primary_rows = by_method["primary_baseline"]

    comparison_methods = {
        "structured": "structured_depth3_ebm",
        "learned_v4_12": "learned_v4_12_depth3_ebm",
        "oracle_100": "oracle_100_depth3_ebm",
        "oracle_90": "oracle_90_depth3_ebm",
        "oracle_75": "oracle_75_depth3_ebm",
        "oracle_50": "oracle_50_depth3_ebm",
        "true_world": "true_world_depth3_ebm",
        "oracle_energy": "oracle_energy",
    }
    comparisons = {
        f"{name}_over_primary": _paired_bootstrap(
            by_method[method],
            primary_rows,
            seed=SEED + index,
        )
        for index, (name, method) in enumerate(comparison_methods.items())
    }
    comparisons.update(
        {
            "oracle_semantics_over_structured": _paired_bootstrap(
                by_method["oracle_100_depth3_ebm"],
                by_method["structured_depth3_ebm"],
                seed=SEED + 100,
            ),
            "learned_over_structured": _paired_bootstrap(
                by_method["learned_v4_12_depth3_ebm"],
                by_method["structured_depth3_ebm"],
                seed=SEED + 101,
            ),
            "learned_over_root_only_stress": _paired_bootstrap(
                by_method["learned_v4_12_depth3_ebm"],
                by_method["learned_root_only_stress_depth3_ebm"],
                seed=SEED + 102,
            ),
            "learned_over_relation_shuffle_stress": _paired_bootstrap(
                by_method["learned_v4_12_depth3_ebm"],
                by_method["learned_relation_shuffle_stress_depth3_ebm"],
                seed=SEED + 103,
            ),
            "learned_full_over_root_reuse": _paired_bootstrap(
                by_method["learned_v4_12_depth3_ebm"],
                by_method["learned_root_reuse_stress_depth3_ebm"],
                seed=SEED + 104,
            ),
            "oracle_full_over_root_reuse": _paired_bootstrap(
                by_method["oracle_100_depth3_ebm"],
                by_method["oracle_root_reuse_stress_depth3_ebm"],
                seed=SEED + 105,
            ),
            "oracle_root_reuse_over_primary": _paired_bootstrap(
                by_method["oracle_root_reuse_stress_depth3_ebm"],
                primary_rows,
                seed=SEED + 106,
            ),
            "learned_root_reuse_over_primary": _paired_bootstrap(
                by_method["learned_root_reuse_stress_depth3_ebm"],
                primary_rows,
                seed=SEED + 107,
            ),
        }
    )

    completion = _completion_capture(roots, decisions)
    threshold = manifest["diagnostic_decisions"]
    oracle_completion = completion["selected_by_method"].get(
        "oracle_100_depth3_ebm", 0
    )
    learned_completion = completion["selected_by_method"].get(
        "learned_v4_12_depth3_ebm", 0
    )
    true_world_completion = completion["selected_by_method"].get(
        "true_world_depth3_ebm", 0
    )
    required_completion = min(
        int(threshold["completion_selected_minimum_when_available"]),
        int(completion["opportunities"]),
    )
    primary_per_game = metrics["primary_baseline"]["per_game"]

    def nonnegative_games(method: str) -> int:
        rows = metrics[method]["per_game"]
        return sum(
            rows[game]["mean_utility"]
            >= primary_per_game[game]["mean_utility"]
            for game in games
        )

    curve = []
    for name in NOISE_LEVELS:
        method = f"{name}_depth3_ebm"
        curve.append(
            {
                "condition": name,
                "requested_semantic_accuracy": 1.0 - NOISE_LEVELS[name],
                "observed_semantic_accuracy": corruption[name][
                    "observed_bit_accuracy"
                ],
                "mean_utility": metrics[method]["mean_utility"],
                "mean_leaf_utility": metrics[method]["mean_leaf_utility"],
                "completion_selected": completion["selected_by_method"].get(
                    method, 0
                ),
                "over_primary": comparisons[f"{name}_over_primary"],
                "nonnegative_games": nonnegative_games(method),
            }
        )
    curve_spearman = _spearman(
        [row["observed_semantic_accuracy"] for row in curve],
        [row["mean_utility"] for row in curve],
    )
    supported_curve_rows = [
        row
        for row in curve
        if row["over_primary"]["ci_low"] > 0.0
        and row["completion_selected"] >= required_completion
    ]
    minimum_supported_accuracy = (
        min(row["observed_semantic_accuracy"] for row in supported_curve_rows)
        if supported_curve_rows
        else None
    )
    true_world_supported = (
        comparisons["true_world_over_primary"]["ci_low"] > 0.0
        and true_world_completion >= required_completion
    )
    oracle_semantics_supported = (
        comparisons["oracle_100_over_primary"]["ci_low"] > 0.0
        and comparisons["oracle_semantics_over_structured"]["mean_gain"] > 0.0
        and oracle_completion >= required_completion
    )
    learned_supported = (
        comparisons["learned_v4_12_over_primary"]["ci_low"] > 0.0
        and learned_completion >= required_completion
        and nonnegative_games("learned_v4_12_depth3_ebm")
        >= int(threshold["nonnegative_games_minimum"])
    )
    root_reuse_supported = (
        comparisons["oracle_root_reuse_over_primary"]["ci_low"] > 0.0
    )
    if not true_world_supported:
        verdict = "TRUE_WORLD_EBM_CONTROLLER_NOT_SUPPORTED"
    elif not oracle_semantics_supported:
        verdict = "SEMANTIC_WORLD_MODEL_BOTTLENECK"
    elif learned_supported:
        verdict = "LEARNED_GLOBAL_CHAIN_SUPPORTED"
    else:
        verdict = "SEMANTIC_PREDICTOR_BOTTLENECK"

    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "verdict": verdict,
        "all_conditions_executed": True,
        "component_diagnosis": {
            "true_world_ebm_controller_supported": true_world_supported,
            "oracle_semantic_world_chain_supported": oracle_semantics_supported,
            "learned_semantic_chain_supported": learned_supported,
            "oracle_root_reuse_stress_supported": root_reuse_supported,
        },
        "diagnostic_checks": {
            "true_world_over_primary_ci": (
                comparisons["true_world_over_primary"]["ci_low"] > 0.0
            ),
            "true_world_completion_selected": (
                true_world_completion >= required_completion
            ),
            "oracle_semantics_over_primary_ci": (
                comparisons["oracle_100_over_primary"]["ci_low"] > 0.0
            ),
            "oracle_semantics_over_structured_mean": (
                comparisons["oracle_semantics_over_structured"]["mean_gain"] > 0.0
            ),
            "oracle_completion_selected": (
                oracle_completion >= required_completion
            ),
            "learned_over_primary_ci": (
                comparisons["learned_v4_12_over_primary"]["ci_low"] > 0.0
            ),
            "learned_completion_selected": (
                learned_completion >= required_completion
            ),
            "learned_nonnegative_games": (
                nonnegative_games("learned_v4_12_depth3_ebm")
                >= int(threshold["nonnegative_games_minimum"])
            ),
            "corruption_monotonicity": (
                curve_spearman
                >= float(threshold["corruption_spearman_minimum"])
            ),
        },
        "roots": len(roots),
        "nodes": len(_nodes(original)),
        "slots": len(original),
        "games": games,
        "metrics": metrics,
        "comparisons": comparisons,
        "semantic_corruption": corruption,
        "semantic_accuracy_curve": {
            "rows": curve,
            "utility_spearman": curve_spearman,
            "minimum_supported_observed_accuracy": minimum_supported_accuracy,
        },
        "completion_capture": completion,
        "nonnegative_games": {
            name: nonnegative_games(method)
            for name, method in comparison_methods.items()
            if method in metrics
        },
        "world_model_metrics": {
            name: _world_metrics(variants[name], predictions)
            for name, predictions in held_predictions.items()
        },
        "topology_boundary": {
            "candidate_complete_future_slots_used": True,
            "root_reuse_stress_is_not_a_deployable_rollout": True,
            "live_win_rate_claimed": False,
            "reason": (
                "V4.7 consumes future V4.3 slot descriptors; a real win-rate "
                "claim requires a learned deployable state-transition rollout."
            ),
        },
        "world_model_fitted": True,
        "ebm_fitted": True,
        "authority_promoted": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {
            "decisions": _file_sha256(decisions_path),
            "folds": _file_sha256(folds_path),
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def run(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if not (destination / "frozen_manifest.json").exists():
        freeze_manifest(output_dir=destination)
    return evaluate(output_dir=destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(output_dir=args.output_dir)
    elif args.command == "evaluate":
        payload = evaluate(output_dir=args.output_dir)
    else:
        payload = run(output_dir=args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORRUPTED_EFFECTS",
    "NOISE_LEVELS",
    "evaluate",
    "freeze_manifest",
    "load_manifest",
    "run",
]
