"""Cheap effect-predictability pilot required before graph-model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score


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


def run_effect_predictability_pilot(
    features: Sequence[Sequence[float]],
    effect_labels: Sequence[int],
    *,
    train_mask: Sequence[bool] | None = None,
    minimum_improvement: float = 0.05,
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
    counts = np.bincount(y[train])
    majority_class = int(np.argmax(counts))
    baseline = np.full(validation.sum(), majority_class, dtype=np.int64)
    labels = np.unique(y)
    majority_f1 = float(f1_score(
        y[validation],
        baseline,
        labels=labels,
        average="macro",
        zero_division=0,
    ))
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=4,
        max_iter=100,
        random_state=int(random_state),
    )
    classifier.fit(x[train], y[train])
    prediction = classifier.predict(x[validation])
    classifier_f1 = float(f1_score(
        y[validation],
        prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    ))
    improvement = classifier_f1 - majority_f1
    go = improvement >= float(minimum_improvement)
    return EffectPilotResult(
        samples=len(y),
        classes=len(labels),
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


__all__ = ["EffectPilotResult", "run_effect_predictability_pilot"]
