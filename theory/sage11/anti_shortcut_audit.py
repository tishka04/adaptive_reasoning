"""Source-train-only leave-one-game-out anti-shortcut audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from .effect_pilot_runner import DEFAULT_MANIFEST_PATH
from .pilot import effect_macro_f1, majority_predictions
from .splits import SOURCE_TRAIN
from .streaming_dataset import load_source_train_streaming_dataset
from .streaming_features import CORE_FACTOR_HEADS


AUDIT_FORMAT_VERSION = "sage11-anti-shortcut-logo-v1"
DEFAULT_AUDIT_RESULT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage11_source_train_anti_shortcut_logo.json"
)
RANDOM_STATE = 11
MINIMUM_CHANGED_IMPROVEMENT = 0.10
MINIMUM_ACTION_SHUFFLE_DEGRADATION = 0.10
MINIMUM_NONNEGATIVE_FOLDS = 9
MINIMUM_ALLOWED_FOLD_DELTA = -0.05
MAXIMUM_SIGNATURE_PURITY = 0.80
MAXIMUM_SIGNATURE_ABLATION_DROP = 0.02


def make_audit_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=4,
        max_iter=100,
        early_stopping=False,
        class_weight="balanced",
        random_state=int(random_state),
    )


def _conditional_action_shuffle(
    features: np.ndarray,
    *,
    signature_columns: Sequence[int],
    action_dependent_columns: Sequence[int],
    random_state: int,
) -> np.ndarray:
    shuffled = features.copy()
    signature = np.packbits(
        features[:, tuple(signature_columns)].astype(np.uint8),
        axis=1,
    )
    groups: Dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(signature):
        groups[row.tobytes()].append(index)
    generator = np.random.default_rng(int(random_state))
    action_columns = np.asarray(
        tuple(action_dependent_columns),
        dtype=np.int64,
    )
    for indices_list in groups.values():
        if len(indices_list) < 2:
            continue
        indices = np.asarray(indices_list, dtype=np.int64)
        permutation = generator.permutation(indices)
        shuffled[np.ix_(indices, action_columns)] = features[
            np.ix_(permutation, action_columns)
        ]
    return shuffled


def _signature_identity_metrics(
    features: np.ndarray,
    games: np.ndarray,
    signature_columns: Sequence[int],
) -> Dict[str, Any]:
    packed = np.packbits(
        features[:, tuple(signature_columns)].astype(np.uint8),
        axis=1,
    )
    counts: Dict[bytes, Counter[str]] = defaultdict(Counter)
    for row, game in zip(packed, games):
        counts[row.tobytes()][str(game)] += 1
    correct = sum(max(values.values()) for values in counts.values())
    exclusive_rows = sum(
        sum(values.values())
        for values in counts.values()
        if len(values) == 1
    )
    shared = sum(1 for values in counts.values() if len(values) > 1)
    return {
        "signature_count": len(counts),
        "shared_signature_count": shared,
        "exclusive_signature_count": len(counts) - shared,
        "row_weighted_majority_game_accuracy": correct / len(games),
        "rows_in_game_exclusive_signatures": exclusive_rows,
        "game_exclusive_row_rate": exclusive_rows / len(games),
    }


def _head_metric(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    scores = {
        name: effect_macro_f1(truth, prediction)
        for name, prediction in predictions.items()
    }
    best_baseline = max(scores["action_only"], scores["state_only"])
    return {
        **scores,
        "best_action_or_state_baseline": best_baseline,
        "full_minus_best_baseline": scores["full"] - best_baseline,
        "conditional_action_shuffle_degradation": (
            scores["full"] - scores["conditional_action_shuffled"]
        ),
        "signature_ablation_drop": (
            scores["full"] - scores["full_without_signature"]
        ),
    }


def _composite(
    heads: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    metric_names = (
        "per_action_majority",
        "action_only",
        "state_only",
        "full",
        "full_without_signature",
        "conditional_action_shuffled",
    )
    values = {
        name: float(np.mean([
            heads[head][name]
            for head in CORE_FACTOR_HEADS
        ]))
        for name in metric_names
    }
    best_baseline = max(values["action_only"], values["state_only"])
    return {
        **values,
        "best_action_or_state_baseline": best_baseline,
        "full_minus_best_baseline": values["full"] - best_baseline,
        "conditional_action_shuffle_degradation": (
            values["full"] - values["conditional_action_shuffled"]
        ),
        "signature_ablation_drop": (
            values["full"] - values["full_without_signature"]
        ),
    }


def evaluate_anti_shortcut_gate(
    aggregate_heads: Mapping[str, Mapping[str, float]],
    aggregate_composite: Mapping[str, float],
    fold_changed_deltas: Mapping[str, float],
    signature_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply every frozen anti-shortcut condition."""
    changed_improvement = aggregate_heads["changed_cells"][
        "full_minus_best_baseline"
    ]
    changed_pass = changed_improvement >= MINIMUM_CHANGED_IMPROVEMENT
    shuffle_degradation = aggregate_composite[
        "conditional_action_shuffle_degradation"
    ]
    shuffle_pass = (
        shuffle_degradation >= MINIMUM_ACTION_SHUFFLE_DEGRADATION
    )
    nonnegative_folds = sum(
        delta >= 0.0
        for delta in fold_changed_deltas.values()
    )
    worst_fold_delta = min(fold_changed_deltas.values())
    fold_pass = (
        nonnegative_folds >= MINIMUM_NONNEGATIVE_FOLDS
        and worst_fold_delta >= MINIMUM_ALLOWED_FOLD_DELTA
    )
    signature_purity = float(
        signature_identity["row_weighted_majority_game_accuracy"]
    )
    changed_signature_drop = aggregate_heads["changed_cells"][
        "signature_ablation_drop"
    ]
    composite_signature_drop = aggregate_composite[
        "signature_ablation_drop"
    ]
    signature_reliance = (
        signature_purity > MAXIMUM_SIGNATURE_PURITY
        and (
            changed_signature_drop > MAXIMUM_SIGNATURE_ABLATION_DROP
            or composite_signature_drop
            > MAXIMUM_SIGNATURE_ABLATION_DROP
        )
    )
    signature_pass = not signature_reliance
    return {
        "passed": bool(
            changed_pass
            and shuffle_pass
            and fold_pass
            and signature_pass
        ),
        "changed_cells_improvement_at_least_0_10": changed_pass,
        "composite_action_shuffle_degradation_at_least_0_10": (
            shuffle_pass
        ),
        "fold_robustness": fold_pass,
        "no_fixed_signature_shortcut_reliance": signature_pass,
        "observed": {
            "changed_cells_full_minus_best_baseline": changed_improvement,
            "composite_action_shuffle_degradation": shuffle_degradation,
            "nonnegative_changed_folds": nonnegative_folds,
            "worst_changed_fold_delta": worst_fold_delta,
            "signature_game_purity": signature_purity,
            "changed_signature_ablation_drop": changed_signature_drop,
            "composite_signature_ablation_drop": (
                composite_signature_drop
            ),
        },
        "thresholds": {
            "minimum_changed_improvement": MINIMUM_CHANGED_IMPROVEMENT,
            "minimum_action_shuffle_degradation": (
                MINIMUM_ACTION_SHUFFLE_DEGRADATION
            ),
            "minimum_nonnegative_folds": MINIMUM_NONNEGATIVE_FOLDS,
            "minimum_allowed_fold_delta": MINIMUM_ALLOWED_FOLD_DELTA,
            "maximum_signature_purity_without_reliance_test": (
                MAXIMUM_SIGNATURE_PURITY
            ),
            "maximum_signature_ablation_drop": (
                MAXIMUM_SIGNATURE_ABLATION_DROP
            ),
        },
    }


def run_anti_shortcut_audit(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    result_path: str | Path = DEFAULT_AUDIT_RESULT_PATH,
    *,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Run one source-train-only LOGO audit with no source-val reads."""
    started = time.perf_counter()
    dataset = load_source_train_streaming_dataset(manifest_path)
    loaded = time.perf_counter()
    schema = dataset.schema
    games = dataset.games
    features = dataset.features

    action_columns = np.asarray(
        schema.action_feature_indices,
        dtype=np.int64,
    )
    state_columns = np.asarray(
        schema.state_only_feature_indices,
        dtype=np.int64,
    )
    action_dependent_columns = np.asarray(
        schema.action_dependent_feature_indices,
        dtype=np.int64,
    )
    signature_columns = np.asarray(
        schema.game_signature_feature_indices,
        dtype=np.int64,
    )
    without_signature_columns = np.asarray(
        tuple(
            index
            for index in range(schema.feature_count)
            if index not in set(signature_columns.tolist())
        ),
        dtype=np.int64,
    )

    prediction_names = (
        "per_action_majority",
        "action_only",
        "state_only",
        "full",
        "full_without_signature",
        "conditional_action_shuffled",
    )
    out_of_fold = {
        head: {
            name: np.empty(len(games), dtype=np.int64)
            for name in prediction_names
        }
        for head in CORE_FACTOR_HEADS
    }
    per_game: Dict[str, Any] = {}

    for fold_index, held_out_game in enumerate(SOURCE_TRAIN):
        train = games != held_out_game
        test = games == held_out_game
        test_indices = np.flatnonzero(test)
        test_features = features[test]
        shuffled_features = _conditional_action_shuffle(
            test_features,
            signature_columns=signature_columns,
            action_dependent_columns=action_dependent_columns,
            random_state=int(random_state) + fold_index,
        )
        fold_predictions: Dict[str, Dict[str, np.ndarray]] = {}
        for head in CORE_FACTOR_HEADS:
            target = dataset.labels[head]
            majority = majority_predictions(
                target,
                train,
                groups=dataset.actions,
            )
            models = {}
            for name, columns in {
                "action_only": action_columns,
                "state_only": state_columns,
                "full": np.arange(
                    schema.feature_count,
                    dtype=np.int64,
                ),
                "full_without_signature": without_signature_columns,
            }.items():
                model = make_audit_classifier(random_state=random_state)
                model.fit(
                    features[train][:, columns],
                    target[train],
                )
                models[name] = model
            predictions = {
                "per_action_majority": majority,
                "action_only": models["action_only"].predict(
                    test_features[:, action_columns]
                ),
                "state_only": models["state_only"].predict(
                    test_features[:, state_columns]
                ),
                "full": models["full"].predict(test_features),
                "full_without_signature": (
                    models["full_without_signature"].predict(
                        test_features[:, without_signature_columns]
                    )
                ),
                "conditional_action_shuffled": models["full"].predict(
                    shuffled_features
                ),
            }
            fold_predictions[head] = predictions
            for name, prediction in predictions.items():
                out_of_fold[head][name][test_indices] = prediction
        fold_heads = {
            head: _head_metric(
                dataset.labels[head][test],
                fold_predictions[head],
            )
            for head in CORE_FACTOR_HEADS
        }
        per_game[held_out_game] = {
            "rows": int(test.sum()),
            "heads": fold_heads,
            "composite": _composite(fold_heads),
        }
    fitted = time.perf_counter()

    aggregate_heads = {
        head: _head_metric(
            dataset.labels[head],
            out_of_fold[head],
        )
        for head in CORE_FACTOR_HEADS
    }
    aggregate_composite = _composite(aggregate_heads)
    signature_identity = _signature_identity_metrics(
        features,
        games,
        signature_columns,
    )
    fold_changed_deltas = {
        game: float(
            per_game[game]["heads"]["changed_cells"][
                "full_minus_best_baseline"
            ]
        )
        for game in SOURCE_TRAIN
    }
    gate = evaluate_anti_shortcut_gate(
        aggregate_heads,
        aggregate_composite,
        fold_changed_deltas,
        signature_identity,
    )
    evaluated = time.perf_counter()

    payload: Dict[str, Any] = {
        "format_version": AUDIT_FORMAT_VERSION,
        "run_date": date.today().isoformat(),
        "decision": {
            "passed": bool(gate["passed"]),
            "next_step": (
                "gpu_train_factorized_world_model"
                if gate["passed"]
                else (
                    "collect_smaller_object_relational_pilot_corpus"
                )
            ),
            "gate": gate,
        },
        "firewall": {
            "manifest_path": Path(manifest_path).as_posix(),
            "manifest_checksum": dataset.manifest_checksum,
            "source_train_rows": int(len(games)),
            "source_train_games": list(SOURCE_TRAIN),
            "leave_one_game_out_folds": len(SOURCE_TRAIN),
            "source_validation_shards_opened": False,
            "historical_shards_opened": False,
            "holdout_shards_opened": False,
        },
        "streaming_schema": schema.to_dict(),
        "feature_views": {
            "action_only_columns": len(action_columns),
            "state_only_columns": len(state_columns),
            "full_columns": schema.feature_count,
            "full_without_signature_columns": len(
                without_signature_columns
            ),
            "action_dependent_columns": len(
                action_dependent_columns
            ),
            "fixed_signature_columns": len(signature_columns),
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
            "aggregate_out_of_fold": {
                "heads": aggregate_heads,
                "composite": aggregate_composite,
            },
            "per_held_out_game": per_game,
            "fixed_signature_identity": signature_identity,
        },
        "conditional_shuffle": {
            "method": (
                "permute all current-action-dependent columns within each "
                "held-out game's exact availability/object-role signature"
            ),
            "fold_random_states": {
                game: int(random_state) + index
                for index, game in enumerate(SOURCE_TRAIN)
            },
        },
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": (
                str(torch.cuda.get_device_name(0))
                if torch.cuda.is_available()
                else None
            ),
            "training_device": "cpu",
            "device_decision": (
                "The frozen scikit-learn audit estimators have no CUDA "
                "backend; changing estimator would violate pre-registration."
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "timing_seconds": {
            "verify_load_and_encode": round(loaded - started, 3),
            "fit_logo_models": round(fitted - loaded, 3),
            "evaluate_and_serialize": round(evaluated - fitted, 3),
            "total": round(evaluated - started, 3),
        },
        "reproduction": {
            "command": (
                "ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe "
                "-m theory.sage11.anti_shortcut_audit"
            ),
            "protocol": (
                "reports/SAGE11_ANTI_SHORTCUT_AUDIT_PROTOCOL.md"
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
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the source-train-only SAGE.11 shortcut audit.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AUDIT_RESULT_PATH,
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_anti_shortcut_audit(
        args.manifest,
        args.output,
        random_state=args.random_state,
    )
    print(json.dumps({
        "decision": result["decision"],
        "aggregate": result["metrics"]["aggregate_out_of_fold"],
        "fixed_signature_identity": (
            result["metrics"]["fixed_signature_identity"]
        ),
        "result_checksum": result["result_checksum"],
        "timing_seconds": result["timing_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_AUDIT_RESULT_PATH",
    "evaluate_anti_shortcut_gate",
    "make_audit_classifier",
    "run_anti_shortcut_audit",
]
