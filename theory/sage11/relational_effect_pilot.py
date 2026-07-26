"""LOGO changed-effect pilot for the smaller object-relational corpus."""

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

from .pilot import effect_macro_f1, majority_predictions
from .relational_dataset import (
    iter_relational_shard,
    verify_relational_manifest,
)
from .relational_features import RELATIONAL_FEATURE_SCHEMA
from .relational_pilot_collection import (
    DEFAULT_RELATIONAL_MANIFEST_PATH,
)
from .splits import SOURCE_TRAIN
from .streaming_features import (
    CORE_FACTOR_HEADS,
    encode_transition_rows,
)


RELATIONAL_PILOT_FORMAT_VERSION = "sage11-relational-effect-logo-v1"
DEFAULT_RELATIONAL_RESULT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage11_relational_effect_pilot.json"
)
RANDOM_STATE = 11
MINIMUM_CHANGED_IMPROVEMENT = 0.10
MINIMUM_ACTION_SHUFFLE_DEGRADATION = 0.10
MINIMUM_RELATIONAL_CHANGED_CONTRIBUTION = 0.05
MINIMUM_NONNEGATIVE_FOLDS = 9
MINIMUM_ALLOWED_FOLD_DELTA = -0.05


def make_relational_classifier(
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


def evaluate_relational_pilot_gate(
    aggregate_heads: Mapping[str, Mapping[str, float]],
    aggregate_composite: Mapping[str, float],
    fold_changed_deltas: Mapping[str, float],
) -> Dict[str, Any]:
    changed = aggregate_heads["changed_cells"]
    changed_improvement = changed["full_minus_best_baseline"]
    shuffle_degradation = aggregate_composite[
        "conditional_action_shuffle_degradation"
    ]
    relational_contribution = changed[
        "full_minus_without_relations"
    ]
    nonnegative_folds = sum(
        delta >= 0.0
        for delta in fold_changed_deltas.values()
    )
    worst_fold = min(fold_changed_deltas.values())
    conditions = {
        "changed_cells_improvement_at_least_0_10": (
            changed_improvement >= MINIMUM_CHANGED_IMPROVEMENT
        ),
        "composite_action_shuffle_degradation_at_least_0_10": (
            shuffle_degradation
            >= MINIMUM_ACTION_SHUFFLE_DEGRADATION
        ),
        "relational_changed_contribution_at_least_0_05": (
            relational_contribution
            >= MINIMUM_RELATIONAL_CHANGED_CONTRIBUTION
        ),
        "fold_robustness": (
            nonnegative_folds >= MINIMUM_NONNEGATIVE_FOLDS
            and worst_fold >= MINIMUM_ALLOWED_FOLD_DELTA
        ),
    }
    return {
        "passed": all(conditions.values()),
        **conditions,
        "observed": {
            "changed_cells_full_minus_best_baseline": (
                changed_improvement
            ),
            "composite_action_shuffle_degradation": (
                shuffle_degradation
            ),
            "changed_cells_full_minus_without_relations": (
                relational_contribution
            ),
            "nonnegative_changed_folds": nonnegative_folds,
            "worst_changed_fold_delta": worst_fold,
        },
        "thresholds": {
            "minimum_changed_improvement": MINIMUM_CHANGED_IMPROVEMENT,
            "minimum_action_shuffle_degradation": (
                MINIMUM_ACTION_SHUFFLE_DEGRADATION
            ),
            "minimum_relational_changed_contribution": (
                MINIMUM_RELATIONAL_CHANGED_CONTRIBUTION
            ),
            "minimum_nonnegative_folds": MINIMUM_NONNEGATIVE_FOLDS,
            "minimum_allowed_fold_delta": MINIMUM_ALLOWED_FOLD_DELTA,
        },
    }


def run_relational_effect_pilot(
    manifest_path: str | Path = DEFAULT_RELATIONAL_MANIFEST_PATH,
    result_path: str | Path = DEFAULT_RELATIONAL_RESULT_PATH,
    *,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Run the one frozen small-corpus relational pilot."""
    started = time.perf_counter()
    manifest_source = Path(manifest_path)
    manifest = verify_relational_manifest(manifest_source)
    records = tuple(_iter_manifest_records(manifest_source, manifest))
    rows = tuple(
        record.base_transition.to_dict()
        for record in records
    )
    streaming = encode_transition_rows(
        lambda: iter(rows),
        total_rows=len(rows),
        manifest_checksum=str(manifest["manifest_checksum"]),
    )
    relational = np.asarray(
        [
            record.relational_features_before
            for record in records
        ],
        dtype=np.float32,
    )
    loaded = time.perf_counter()

    streaming_signature = set(
        streaming.schema.game_signature_feature_indices
    )
    kept_streaming = tuple(
        index
        for index in range(streaming.schema.feature_count)
        if index not in streaming_signature
    )
    streaming_remap = {
        old: new
        for new, old in enumerate(kept_streaming)
    }
    streaming_without_signature = streaming.features[
        :,
        kept_streaming,
    ]
    features = np.concatenate(
        (streaming_without_signature, relational),
        axis=1,
    )
    relational_offset = len(kept_streaming)
    action_columns = np.asarray(
        [
            streaming_remap[index]
            for index in streaming.schema.action_feature_indices
        ],
        dtype=np.int64,
    )
    streaming_state_columns = [
        streaming_remap[index]
        for index in streaming.schema.state_only_feature_indices
        if index in streaming_remap
    ]
    relational_state_columns = [
        relational_offset + index
        for index in RELATIONAL_FEATURE_SCHEMA.state_feature_indices
    ]
    state_columns = np.asarray(
        streaming_state_columns + relational_state_columns,
        dtype=np.int64,
    )
    without_relations_columns = np.arange(
        relational_offset,
        dtype=np.int64,
    )
    action_dependent_columns = np.asarray(
        [
            streaming_remap[index]
            for index
            in streaming.schema.action_dependent_feature_indices
            if index in streaming_remap
        ]
        + [
            relational_offset + index
            for index
            in RELATIONAL_FEATURE_SCHEMA.action_dependent_feature_indices
        ],
        dtype=np.int64,
    )
    relational_signature_columns = np.asarray(
        relational_state_columns,
        dtype=np.int64,
    )

    prediction_names = (
        "per_action_majority",
        "action_only",
        "state_only",
        "full_without_relations",
        "full",
        "conditional_action_shuffled",
    )
    out_of_fold = {
        head: {
            name: np.empty(len(records), dtype=np.int64)
            for name in prediction_names
        }
        for head in CORE_FACTOR_HEADS
    }
    per_game: Dict[str, Any] = {}
    for fold_index, held_out_game in enumerate(SOURCE_TRAIN):
        train = streaming.games != held_out_game
        test = streaming.games == held_out_game
        test_indices = np.flatnonzero(test)
        test_features = features[test]
        shuffled = _conditional_action_shuffle(
            test_features,
            signature_columns=relational_signature_columns,
            action_dependent_columns=action_dependent_columns,
            random_state=int(random_state) + fold_index,
        )
        fold_predictions: Dict[str, Dict[str, np.ndarray]] = {}
        for head in CORE_FACTOR_HEADS:
            target = streaming.labels[head]
            models = {}
            for name, columns in {
                "action_only": action_columns,
                "state_only": state_columns,
                "full_without_relations": without_relations_columns,
                "full": np.arange(features.shape[1], dtype=np.int64),
            }.items():
                model = make_relational_classifier(
                    random_state=random_state
                )
                model.fit(features[train][:, columns], target[train])
                models[name] = model
            predictions = {
                "per_action_majority": majority_predictions(
                    target,
                    train,
                    groups=streaming.actions,
                ),
                "action_only": models["action_only"].predict(
                    test_features[:, action_columns]
                ),
                "state_only": models["state_only"].predict(
                    test_features[:, state_columns]
                ),
                "full_without_relations": models[
                    "full_without_relations"
                ].predict(
                    test_features[:, without_relations_columns]
                ),
                "full": models["full"].predict(test_features),
                "conditional_action_shuffled": models["full"].predict(
                    shuffled
                ),
            }
            fold_predictions[head] = predictions
            for name, prediction in predictions.items():
                out_of_fold[head][name][test_indices] = prediction
        fold_heads = {
            head: _head_metric(
                streaming.labels[head][test],
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
            streaming.labels[head],
            out_of_fold[head],
        )
        for head in CORE_FACTOR_HEADS
    }
    aggregate_composite = _composite(aggregate_heads)
    relational_identity = _signature_identity_metrics(
        features,
        streaming.games,
        relational_signature_columns,
    )
    fold_changed_deltas = {
        game: float(
            per_game[game]["heads"]["changed_cells"][
                "full_minus_best_baseline"
            ]
        )
        for game in SOURCE_TRAIN
    }
    gate = evaluate_relational_pilot_gate(
        aggregate_heads,
        aggregate_composite,
        fold_changed_deltas,
    )
    evaluated = time.perf_counter()
    payload: Dict[str, Any] = {
        "format_version": RELATIONAL_PILOT_FORMAT_VERSION,
        "run_date": date.today().isoformat(),
        "decision": {
            "passed": bool(gate["passed"]),
            "next_step": (
                "implement_relational_world_model_interface_and_gpu_train"
                if gate["passed"]
                else "stop_world_model_track_and_revisit_representation"
            ),
            "gate": gate,
        },
        "firewall": {
            "manifest_path": manifest_source.as_posix(),
            "manifest_checksum": manifest["manifest_checksum"],
            "source_train_rows": len(records),
            "source_train_games": list(SOURCE_TRAIN),
            "leave_one_game_out_folds": len(SOURCE_TRAIN),
            "source_validation_shards_opened": False,
            "historical_shards_opened": False,
            "holdout_shards_opened": False,
        },
        "features": {
            "streaming_schema": streaming.schema.to_dict(),
            "relational_schema": RELATIONAL_FEATURE_SCHEMA.to_dict(),
            "fixed_game_signature_atoms_removed": True,
            "action_only_columns": len(action_columns),
            "state_only_columns": len(state_columns),
            "full_without_relations_columns": len(
                without_relations_columns
            ),
            "full_columns": features.shape[1],
            "action_dependent_columns": len(
                action_dependent_columns
            ),
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
            "relational_state_signature_identity": (
                relational_identity
            ),
        },
        "conditional_shuffle": {
            "method": (
                "permute all current-action-dependent streaming and "
                "object-relative columns within each held-out game's exact "
                "22-bit relational state signature"
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
                "The frozen scikit-learn pilot estimator has no CUDA "
                "backend; GPU use begins only after a pass."
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
                "-m theory.sage11.relational_effect_pilot"
            ),
            "protocol": (
                "reports/SAGE11_RELATIONAL_PILOT_PROTOCOL.md"
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
        "full_minus_without_relations": (
            scores["full"] - scores["full_without_relations"]
        ),
        "conditional_action_shuffle_degradation": (
            scores["full"] - scores["conditional_action_shuffled"]
        ),
    }


def _composite(
    heads: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    metric_names = (
        "per_action_majority",
        "action_only",
        "state_only",
        "full_without_relations",
        "full",
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
        "full_minus_without_relations": (
            values["full"] - values["full_without_relations"]
        ),
        "conditional_action_shuffle_degradation": (
            values["full"] - values["conditional_action_shuffled"]
        ),
    }


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
    return {
        "signature_count": len(counts),
        "shared_signature_count": sum(
            1 for values in counts.values() if len(values) > 1
        ),
        "row_weighted_majority_game_accuracy": correct / len(games),
        "game_exclusive_row_rate": exclusive_rows / len(games),
    }


def _iter_manifest_records(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Any:
    by_game = {
        str(shard["game_id"]): dict(shard)
        for shard in manifest["shards"]
    }
    if set(by_game) != set(SOURCE_TRAIN):
        raise ValueError("relational manifest source games mismatch")
    for game in SOURCE_TRAIN:
        shard = by_game[game]
        path = _resolve_manifest_path(
            str(shard["path"]),
            manifest_path,
        )
        yield from iter_relational_shard(path)


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in manifest_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve relational shard {raw_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen small-corpus relational LOGO pilot.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RELATIONAL_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RELATIONAL_RESULT_PATH,
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_relational_effect_pilot(
        args.manifest,
        args.output,
        random_state=args.random_state,
    )
    print(json.dumps({
        "decision": result["decision"],
        "aggregate": result["metrics"]["aggregate_out_of_fold"],
        "result_checksum": result["result_checksum"],
        "timing_seconds": result["timing_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RELATIONAL_RESULT_PATH",
    "evaluate_relational_pilot_gate",
    "make_relational_classifier",
    "run_relational_effect_pilot",
]
