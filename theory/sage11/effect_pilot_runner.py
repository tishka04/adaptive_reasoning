"""Run the pre-registered cheap effect classifier on the source corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np
import sklearn
import torch

from .atoms import action_features
from .pilot import (
    EFFECT_PILOT_MINIMUM_IMPROVEMENT,
    effect_macro_f1,
    majority_predictions,
    make_effect_classifier,
)
from .source_dataset_runner import verify_source_dataset
from .splits import SOURCE_TRAIN, SOURCE_VALIDATION


PILOT_FORMAT_VERSION = "sage11-effect-pilot-v1"
DEFAULT_MANIFEST_PATH = (
    Path("training") / "sage11" / "source_dataset_v2" / "manifest.json"
)
DEFAULT_RESULT_PATH = (
    Path("diagnostics") / "sage" / "sage11_effect_predictability_pilot.json"
)
ACTION_FEATURE_NAMES: Tuple[str, ...] = (
    "action_index",
    "has_xy",
    "bounded_x",
    "bounded_y",
    "has_action_data",
    "bias",
)


@dataclass(frozen=True)
class EffectPilotDataset:
    """Compact train/validation matrix with provenance metadata."""

    features: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    actions: np.ndarray
    games: np.ndarray
    atom_vocabulary: Tuple[str, ...]
    effect_vocabulary: Tuple[Tuple[str, ...], ...]
    manifest_checksum: str
    unseen_validation_atom_rows: int
    unseen_validation_effect_rows: int

    @property
    def action_feature_offset(self) -> int:
        return len(self.atom_vocabulary)


def _resolve_shard_path(
    raw_path: str,
    manifest_path: Path,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in manifest_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve source shard {raw_path}")


def _iter_rows(
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    for shard in manifest_payload["shards"]:
        shard_path = _resolve_shard_path(
            str(shard["path"]),
            manifest_path,
        )
        with shard_path.open(encoding="utf-8") as handle:
            for line in handle:
                yield dict(json.loads(line))


def load_effect_pilot_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> EffectPilotDataset:
    """Verify and encode source rows without fitting anything on validation."""
    path = Path(manifest_path)
    verify_source_dataset(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    train_atoms: set[str] = set()
    train_effects: set[Tuple[str, ...]] = set()
    total_rows = 0
    for row in _iter_rows(path, payload):
        total_rows += 1
        if row["source_split"] == "source_train":
            train_atoms.update(str(atom) for atom in row["atoms_before"])
            train_effects.add(tuple(str(atom) for atom in row["effect_atoms"]))

    atom_vocabulary = tuple(sorted(train_atoms))
    effect_vocabulary = tuple(sorted(train_effects))
    atom_to_id = {
        atom: index
        for index, atom in enumerate(atom_vocabulary)
    }
    effect_to_id = {
        effect: index
        for index, effect in enumerate(effect_vocabulary)
    }
    unseen_effect_id = len(effect_vocabulary)

    feature_count = len(atom_vocabulary) + len(ACTION_FEATURE_NAMES)
    features = np.zeros((total_rows, feature_count), dtype=np.float32)
    labels = np.empty(total_rows, dtype=np.int64)
    train_mask = np.empty(total_rows, dtype=bool)
    actions = np.empty(total_rows, dtype="<U16")
    games = np.empty(total_rows, dtype="<U16")
    unseen_atom_rows = 0
    unseen_effect_rows = 0

    for index, row in enumerate(_iter_rows(path, payload)):
        split = str(row["source_split"])
        is_train = split == "source_train"
        if not is_train and split != "source_validation":
            raise ValueError(f"unexpected source split {split}")
        atoms = tuple(str(atom) for atom in row["atoms_before"])
        unknown_atoms = set(atoms).difference(atom_to_id)
        if unknown_atoms and not is_train:
            unseen_atom_rows += 1
        for atom in atoms:
            atom_id = atom_to_id.get(atom)
            if atom_id is not None:
                features[index, atom_id] = 1.0

        action_name = str(row["action_name"])
        features[index, len(atom_vocabulary):] = action_features(
            action_name,
            dict(row.get("action_data", {}) or {}),
        )
        effect = tuple(str(atom) for atom in row["effect_atoms"])
        effect_id = effect_to_id.get(effect, unseen_effect_id)
        if effect_id == unseen_effect_id and not is_train:
            unseen_effect_rows += 1
        labels[index] = effect_id
        train_mask[index] = is_train
        actions[index] = action_name
        games[index] = str(row["game_id"])

    expected_rows = int(payload["total_transitions"])
    if total_rows != expected_rows:
        raise ValueError(
            f"manifest declares {expected_rows} rows but encoded {total_rows}"
        )
    if set(games[train_mask]) != set(SOURCE_TRAIN):
        raise ValueError("pilot source-training games do not match registry")
    if set(games[~train_mask]) != set(SOURCE_VALIDATION):
        raise ValueError("pilot source-validation games do not match registry")

    return EffectPilotDataset(
        features=features,
        labels=labels,
        train_mask=train_mask,
        actions=actions,
        games=games,
        atom_vocabulary=atom_vocabulary,
        effect_vocabulary=effect_vocabulary,
        manifest_checksum=str(payload["manifest_checksum"]),
        unseen_validation_atom_rows=unseen_atom_rows,
        unseen_validation_effect_rows=unseen_effect_rows,
    )


def _metric_block(
    truth: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    shuffled_prediction: np.ndarray,
) -> Dict[str, Any]:
    baseline_f1 = effect_macro_f1(truth, baseline)
    classifier_f1 = effect_macro_f1(truth, prediction)
    shuffled_f1 = effect_macro_f1(truth, shuffled_prediction)
    improvement = classifier_f1 - baseline_f1
    return {
        "rows": int(len(truth)),
        "validation_effect_classes": int(len(np.unique(truth))),
        "per_action_majority_macro_f1": baseline_f1,
        "classifier_macro_f1": classifier_f1,
        "absolute_improvement": improvement,
        "action_shuffled_macro_f1": shuffled_f1,
        "action_shuffle_degradation": classifier_f1 - shuffled_f1,
        "meets_effect_gate": (
            improvement >= EFFECT_PILOT_MINIMUM_IMPROVEMENT
        ),
    }


def _hardware_metadata() -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = (
        str(torch.cuda.get_device_name(0))
        if cuda_available
        else None
    )
    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "training_device": "cpu",
        "device_decision": (
            "The fixed scikit-learn histogram gradient booster has no CUDA "
            "backend. With only 25 dense features, a GPU implementation or "
            "new dependency would add transfer/setup overhead and change the "
            "pre-registered estimator, so CPU is the effective path."
        ),
    }


def run_source_effect_pilot(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    result_path: str | Path = DEFAULT_RESULT_PATH,
    *,
    random_state: int = 11,
) -> Dict[str, Any]:
    """Fit once on source-train and publish source-validation results."""
    started = time.perf_counter()
    dataset = load_effect_pilot_dataset(manifest_path)
    loaded = time.perf_counter()

    classifier = make_effect_classifier(random_state=random_state)
    classifier.fit(
        dataset.features[dataset.train_mask],
        dataset.labels[dataset.train_mask],
    )
    fitted = time.perf_counter()

    validation_mask = ~dataset.train_mask
    validation_features = dataset.features[validation_mask]
    truth = dataset.labels[validation_mask]
    validation_games = dataset.games[validation_mask]
    baseline = majority_predictions(
        dataset.labels,
        dataset.train_mask,
        groups=dataset.actions,
    )
    prediction = classifier.predict(validation_features)

    shuffled_features = validation_features.copy()
    generator = np.random.default_rng(int(random_state))
    action_slice = slice(dataset.action_feature_offset, None)
    for game in SOURCE_VALIDATION:
        game_indices = np.flatnonzero(validation_games == game)
        permutation = generator.permutation(game_indices)
        shuffled_features[game_indices, action_slice] = (
            validation_features[permutation, action_slice]
        )
    shuffled_prediction = classifier.predict(shuffled_features)
    evaluated = time.perf_counter()

    overall = _metric_block(
        truth,
        baseline,
        prediction,
        shuffled_prediction,
    )
    per_game = {}
    for game in SOURCE_VALIDATION:
        game_mask = validation_games == game
        per_game[game] = _metric_block(
            truth[game_mask],
            baseline[game_mask],
            prediction[game_mask],
            shuffled_prediction[game_mask],
        )

    go = bool(overall["meets_effect_gate"])
    payload: Dict[str, Any] = {
        "format_version": PILOT_FORMAT_VERSION,
        "run_date": date.today().isoformat(),
        "decision": {
            "go": go,
            "next_step": (
                "world_model_training_is_permitted"
                if go
                else "stop_and_revisit_effect_labels_or_state_features"
            ),
            "gate": (
                "validation effect macro-F1 must exceed the train-only "
                "per-action majority baseline by at least 0.10"
            ),
        },
        "dataset": {
            "manifest_path": Path(manifest_path).as_posix(),
            "manifest_checksum": dataset.manifest_checksum,
            "rows": int(len(dataset.labels)),
            "source_train_rows": int(dataset.train_mask.sum()),
            "source_validation_rows": int(validation_mask.sum()),
            "source_train_games": list(SOURCE_TRAIN),
            "source_validation_games": list(SOURCE_VALIDATION),
            "state_atom_features": len(dataset.atom_vocabulary),
            "action_features": len(ACTION_FEATURE_NAMES),
            "total_features": int(dataset.features.shape[1]),
            "training_effect_classes": len(dataset.effect_vocabulary),
            "validation_effect_classes": int(len(np.unique(truth))),
            "unseen_validation_atom_rows": (
                dataset.unseen_validation_atom_rows
            ),
            "unseen_validation_effect_rows": (
                dataset.unseen_validation_effect_rows
            ),
        },
        "features": {
            "state": (
                "train-fitted binary presence of pre-action typed "
                "object/state/action-availability atoms"
            ),
            "state_atom_vocabulary": list(dataset.atom_vocabulary),
            "action": list(ACTION_FEATURE_NAMES),
            "excluded": [
                "game identity",
                "policy arm",
                "post-action atoms",
                "effect labels",
                "holdout and historical rows",
            ],
        },
        "classifier": {
            "name": "sklearn.ensemble.HistGradientBoostingClassifier",
            "learning_rate": 0.08,
            "max_depth": 4,
            "max_iter": 100,
            "early_stopping": False,
            "random_state": int(random_state),
            "tuning_runs": 0,
        },
        "baseline": {
            "name": "train-only per-action majority",
            "unseen_action_fallback": "global source-training majority",
        },
        "action_shuffle_control": {
            "method": (
                "shuffle all six action features within each validation "
                "game without retraining"
            ),
            "random_state": int(random_state),
        },
        "metrics": {
            "overall": overall,
            "per_game": per_game,
        },
        "hardware": _hardware_metadata(),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "timing_seconds": {
            "load_and_encode": round(loaded - started, 3),
            "fit": round(fitted - loaded, 3),
            "evaluate": round(evaluated - fitted, 3),
            "total": round(evaluated - started, 3),
        },
        "reproduction": {
            "command": (
                "ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe "
                "-m theory.sage11.effect_pilot_runner"
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
        description="Run the SAGE.11 source-only effect pilot.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT_PATH,
    )
    parser.add_argument("--random-state", type=int, default=11)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_source_effect_pilot(
        args.manifest,
        args.output,
        random_state=args.random_state,
    )
    print(json.dumps({
        "decision": result["decision"],
        "metrics": result["metrics"],
        "result_checksum": result["result_checksum"],
        "timing_seconds": result["timing_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_RESULT_PATH",
    "EffectPilotDataset",
    "load_effect_pilot_dataset",
    "run_source_effect_pilot",
]
