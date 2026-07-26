"""Cheap effect-predictability pilot required before graph-model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Hashable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score


EFFECT_PILOT_MINIMUM_IMPROVEMENT = 0.10


@dataclass(frozen=True)
class EffectPilotResult:
    samples: int
    classes: int
    majority_macro_f1: float
    classifier_macro_f1: float
    improvement: float
    go: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_effect_classifier(
    *,
    random_state: int = 11,
) -> HistGradientBoostingClassifier:
    """Build the single fixed cheap classifier used by the pilot."""
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=4,
        max_iter=100,
        early_stopping=False,
        random_state=int(random_state),
    )


def majority_predictions(
    effect_labels: Sequence[int],
    train_mask: Sequence[bool],
    *,
    groups: Sequence[Hashable] | None = None,
) -> np.ndarray:
    """Predict each validation row with its train-only group majority.

    The source-corpus pilot passes action names as groups.  An unseen
    validation group falls back to the global source-training majority.
    """
    y = np.asarray(effect_labels, dtype=np.int64)
    train = np.asarray(train_mask, dtype=bool)
    if y.ndim != 1 or train.shape != y.shape:
        raise ValueError("majority baseline requires aligned one-dimensional rows")
    if not train.any() or train.all():
        raise ValueError("majority baseline requires train and validation rows")
    if np.any(y < 0):
        raise ValueError("effect labels must be non-negative integers")

    if groups is None:
        group_values: list[Hashable] = ["all"] * len(y)
    else:
        group_values = list(groups)
        if len(group_values) != len(y):
            raise ValueError("majority baseline groups have incorrect length")

    global_majority = int(np.argmax(np.bincount(y[train])))
    grouped_labels: Dict[Hashable, list[int]] = {}
    for label, group, is_train in zip(y, group_values, train):
        if is_train:
            grouped_labels.setdefault(group, []).append(int(label))
    grouped_majority = {
        group: int(np.argmax(np.bincount(labels)))
        for group, labels in grouped_labels.items()
    }
    return np.asarray(
        [
            grouped_majority.get(group, global_majority)
            for group, is_train in zip(group_values, train)
            if not is_train
        ],
        dtype=np.int64,
    )


def effect_macro_f1(
    truth: Sequence[int],
    prediction: Sequence[int],
) -> float:
    """Return validation macro-F1 with absent classes excluded."""
    return float(f1_score(
        truth,
        prediction,
        average="macro",
        zero_division=0,
    ))


def run_effect_predictability_pilot(
    features: Sequence[Sequence[float]],
    effect_labels: Sequence[int],
    *,
    train_mask: Sequence[bool] | None = None,
    baseline_groups: Sequence[Hashable] | None = None,
    minimum_improvement: float = EFFECT_PILOT_MINIMUM_IMPROVEMENT,
    random_state: int = 11,
) -> EffectPilotResult:
    """Estimate the learnable ceiling before paying for the graph model.

    The split mask must be pre-registered by source game/run.  A deterministic
    tail split is provided only for small procedural tests.
    """
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(effect_labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 20:
        raise ValueError("effect pilot requires >=20 aligned feature rows")
    if train_mask is None:
        split = max(1, int(0.8 * len(y)))
        train = np.arange(len(y)) < split
    else:
        train = np.asarray(train_mask, dtype=bool)
        if train.shape != y.shape:
            raise ValueError("pilot train mask has incorrect shape")
    validation = ~train
    if not train.any() or not validation.any():
        raise ValueError("pilot requires non-empty train and validation rows")
    baseline = majority_predictions(
        y,
        train,
        groups=baseline_groups,
    )
    majority_f1 = effect_macro_f1(y[validation], baseline)
    classifier = make_effect_classifier(random_state=random_state)
    classifier.fit(x[train], y[train])
    prediction = classifier.predict(x[validation])
    classifier_f1 = effect_macro_f1(y[validation], prediction)
    improvement = classifier_f1 - majority_f1
    go = improvement >= float(minimum_improvement)
    return EffectPilotResult(
        samples=len(y),
        classes=len(np.unique(y[validation])),
        majority_macro_f1=majority_f1,
        classifier_macro_f1=classifier_f1,
        improvement=improvement,
        go=go,
        reason=(
            "effect structure is predictably above the per-action majority"
            if go
            else "majority baseline is near the measured effect ceiling; "
            "revisit labels/features before graph-model training"
        ),
    )


__all__ = [
    "EFFECT_PILOT_MINIMUM_IMPROVEMENT",
    "EffectPilotResult",
    "effect_macro_f1",
    "majority_predictions",
    "make_effect_classifier",
    "run_effect_predictability_pilot",
]
