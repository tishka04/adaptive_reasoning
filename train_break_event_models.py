"""Train rare-event break expert models.

The expert is intentionally separate from the general student:

    break_classifier -> P(BIG_BREAK | state, local_geometry, history, action)
    break_regressor  -> E[delta_largest_component_size | true BIG_BREAK rows]

Training oversamples only the training split. Evaluation stays on the natural
distribution and reports ranking metrics suited to rare events.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import game_splits
from abstraction_dataset_io import (
    largest_component_input_feature_names,
    largest_component_input_matrix,
    load_dataset,
)
from build_break_event_dataset import BREAK_CLASSES

DEFAULT_DATASET = "training/break_event_dataset.natural.jsonl"
DEFAULT_CLASSIFIER_OUT = "models/break_classifier.joblib"
DEFAULT_REGRESSOR_OUT = "models/break_regressor.joblib"
TARGET = "delta_largest_component_size"


def _can_stratify(labels: Sequence[str]) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= 2


def _oversample_train_indices(
    train_indices: np.ndarray,
    labels: Sequence[str],
    *,
    seed: int,
    target_per_class: Optional[int],
) -> np.ndarray:
    rng = random.Random(seed)
    grouped: Dict[str, List[int]] = {label: [] for label in BREAK_CLASSES}
    for idx in train_indices.tolist():
        grouped.setdefault(str(labels[idx]), []).append(int(idx))
    target = target_per_class or max((len(items) for items in grouped.values()), default=0)
    sampled: List[int] = []
    for label in BREAK_CLASSES:
        items = grouped.get(label, [])
        if not items:
            continue
        sampled.extend(items)
        needed = max(0, target - len(items))
        sampled.extend(rng.choice(items) for _ in range(needed))
    rng.shuffle(sampled)
    return np.array(sampled, dtype=np.int64)


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if np.unique(y_true).size < 2:
        return None
    return round(float(roc_auc_score(y_true, scores)), 4)


def _safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if np.unique(y_true).size < 2:
        return None
    return round(float(average_precision_score(y_true, scores)), 4)


def _top_k_metrics(y_true: np.ndarray, scores: np.ndarray, deltas: np.ndarray, ks: Sequence[int]) -> Dict[str, Any]:
    order = np.argsort(scores)[::-1]
    positives = int(np.sum(y_true))
    out: Dict[str, Any] = {}
    for raw_k in ks:
        k = int(min(max(1, raw_k), len(y_true)))
        top = order[:k]
        hits = int(np.sum(y_true[top]))
        out[str(raw_k)] = {
            "k_effective": k,
            "hits": hits,
            "precision_at_k": round(float(hits / k), 4),
            "recall_at_k": round(float(hits / positives), 4) if positives else None,
            "top_k_hit": bool(hits > 0),
            "mean_true_delta": round(float(np.mean(deltas[top])), 4),
            "min_true_delta": round(float(np.min(deltas[top])), 4),
        }
    return out


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    if np.unique(y_true).size < 2:
        return {"threshold": None, "precision": None, "recall": None, "f1": None}
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return {"threshold": None, "precision": None, "recall": None, "f1": None}
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    threshold = float(thresholds[best])
    pred = scores >= threshold
    return {
        "threshold": round(threshold, 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1[best]), 4),
    }


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    if y_true.size == 0:
        return {"count": 0, "mae": None, "bias": None, "target_mean": None, "pred_mean": None}
    return {
        "count": int(y_true.size),
        "mae": round(float(mean_absolute_error(y_true, pred)), 4),
        "bias": round(float(np.mean(pred - y_true)), 4),
        "target_mean": round(float(np.mean(y_true)), 4),
        "pred_mean": round(float(np.mean(pred)), 4),
    }


def train(
    *,
    dataset_path: str,
    classifier_out: str,
    regressor_out: str,
    train_games: str,
    sources: Optional[List[str]],
    test_size: float,
    n_estimators: int,
    seed: int,
    target_per_class: Optional[int],
    top_ks: Sequence[int],
    quiet: bool,
) -> Dict[str, Any]:
    data = load_dataset(dataset_path)
    if train_games and train_games.lower() != "all":
        data = data.filter_games(game_splits.resolve(train_games, full_ids=True))
    data = data.filter_sources(sources)
    rows = data.rows
    if len(rows) < 20:
        raise SystemExit(f"Not enough rows ({len(rows)}) to train break expert.")

    x = largest_component_input_matrix(rows)
    feature_names = largest_component_input_feature_names()
    labels = np.array([str(row.get("break_class", "OTHER_CHANGE")) for row in rows])
    y_big = np.array([label == "BIG_BREAK" for label in labels], dtype=np.int32)
    deltas = np.array([float(row.get(TARGET, 0.0)) for row in rows], dtype=np.float32)
    indices = np.arange(len(rows))
    stratify = labels if _can_stratify(labels.tolist()) else None
    tr, te = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    tr_balanced = _oversample_train_indices(
        tr,
        labels.tolist(),
        seed=seed,
        target_per_class=target_per_class,
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    classifier.fit(x[tr_balanced], y_big[tr_balanced])
    class_index = list(classifier.classes_).index(1) if 1 in classifier.classes_ else 0
    proba = classifier.predict_proba(x[te])[:, class_index]

    big_train = tr[y_big[tr] == 1]
    if big_train.size < 2:
        raise SystemExit(f"Not enough BIG_BREAK training rows ({big_train.size}) for regressor.")
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    regressor.fit(x[big_train], deltas[big_train])
    big_test = te[y_big[te] == 1]
    reg_pred = regressor.predict(x[big_test]) if big_test.size else np.array([], dtype=np.float32)

    classifier_metrics = {
        "roc_auc": _safe_roc_auc(y_big[te], proba),
        "pr_auc": _safe_pr_auc(y_big[te], proba),
        "positive_rate_test": round(float(np.mean(y_big[te])), 4),
        "top_k": _top_k_metrics(y_big[te], proba, deltas[te], top_ks),
        "best_f1_threshold": _best_f1_threshold(y_big[te], proba),
    }
    regressor_metrics = _regression_metrics(deltas[big_test], reg_pred)
    metrics: Dict[str, Any] = {
        "dataset": dataset_path,
        "train_games": train_games,
        "n_rows": int(len(rows)),
        "n_train_natural": int(len(tr)),
        "n_train_oversampled": int(len(tr_balanced)),
        "n_test_natural": int(len(te)),
        "feature_count": int(x.shape[1]),
        "class_counts_natural": {label: int(np.sum(labels == label)) for label in BREAK_CLASSES},
        "class_counts_train_natural": {label: int(np.sum(labels[tr] == label)) for label in BREAK_CLASSES},
        "class_counts_train_oversampled": {
            label: int(np.sum(labels[tr_balanced] == label)) for label in BREAK_CLASSES
        },
        "class_counts_test": {label: int(np.sum(labels[te] == label)) for label in BREAK_CLASSES},
        "classifier": classifier_metrics,
        "regressor_on_true_big_breaks": regressor_metrics,
    }

    classifier_path = Path(classifier_out)
    regressor_path = Path(regressor_out)
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": classifier,
            "input_feature_names": feature_names,
            "target": "is_big_break",
            "break_classes": BREAK_CLASSES,
        },
        classifier_path,
    )
    dump(
        {
            "model": regressor,
            "input_feature_names": feature_names,
            "target": TARGET,
            "condition": "trained_on_true_BIG_BREAK_rows",
        },
        regressor_path,
    )
    metrics_path = classifier_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not quiet:
        print(f"Trained break expert on {len(rows)} natural rows.")
        print(f"train natural={len(tr)} oversampled={len(tr_balanced)} test={len(te)}")
        print(f"class_counts_natural={metrics['class_counts_natural']}")
        cm = metrics["classifier"]
        print(
            f"classifier ROC-AUC={cm['roc_auc']} PR-AUC={cm['pr_auc']} "
            f"test_pos={cm['positive_rate_test']}"
        )
        for k, item in cm["top_k"].items():
            print(
                f"  top@{k}: hits={item['hits']} "
                f"P@k={item['precision_at_k']} R@k={item['recall_at_k']}"
            )
        rm = metrics["regressor_on_true_big_breaks"]
        print(
            f"regressor true-big-break MAE={rm['mae']} "
            f"target_mean={rm['target_mean']} pred_mean={rm['pred_mean']}"
        )
        print(f"Saved classifier -> {classifier_path}")
        print(f"Saved regressor  -> {regressor_path}")
        print(f"Metrics -> {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rare break-event expert models.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--classifier-out", default=DEFAULT_CLASSIFIER_OUT)
    parser.add_argument("--regressor-out", default=DEFAULT_REGRESSOR_OUT)
    parser.add_argument("--train-games", default="public_seen")
    parser.add_argument("--sources", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-per-class", type=int, default=None)
    parser.add_argument("--top-k", default="10,25,50,100,250")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    top_ks = [int(k.strip()) for k in args.top_k.split(",") if k.strip()]
    train(
        dataset_path=args.dataset,
        classifier_out=args.classifier_out,
        regressor_out=args.regressor_out,
        train_games=args.train_games,
        sources=sources,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        seed=args.seed,
        target_per_class=args.target_per_class,
        top_ks=top_ks,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
