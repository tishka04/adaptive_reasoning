"""Train/evaluate a simple cognitive-phase classifier.

Question tested:

    Can we predict the human cognitive phase from ARC state abstractions?

The input is deliberately small: ``state_features`` from the cognitive trace
dataset. Validation is grouped by human episode by default, so temporally
nearby frames do not leak across train/test folds.

Run:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe train_cognitive_phase_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from abstraction_dataset_io import history_matrix, state_matrix
from build_cognitive_trace_dataset import PHASE_LABELS
from cognitive_taxonomy import TAXONOMY_V1, map_cognitive_phase, taxonomy_labels
from extract_state_abstractions import FEATURE_SCHEMA

DEFAULT_DATASET = "training/cognitive_trace_dataset.jsonl"
DEFAULT_MODEL_OUT = "models/cognitive_phase.joblib"


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_games(rows: List[Dict[str, Any]], games: str) -> List[Dict[str, Any]]:
    if not games or games.lower() == "all":
        return rows
    wanted = {part.strip() for part in games.split(",") if part.strip()}
    return [
        row
        for row in rows
        if row.get("game_id") in wanted
        or str(row.get("game_id", "")).split("-", 1)[0] in wanted
    ]


def _input_matrix(rows: List[Dict[str, Any]], *, with_history: bool) -> np.ndarray:
    x_state = state_matrix(rows)
    if not with_history:
        return x_state
    return np.concatenate([x_state, history_matrix(rows)], axis=1)


def _input_feature_names(*, with_history: bool) -> List[str]:
    if not with_history:
        return list(FEATURE_SCHEMA)
    from abstraction_dataset_io import HISTORY_FEATURE_SCHEMA

    return list(FEATURE_SCHEMA) + list(HISTORY_FEATURE_SCHEMA)


def _groups(rows: List[Dict[str, Any]], group_by: str) -> np.ndarray:
    if group_by == "game":
        return np.array([str(row.get("game_id", "unknown")) for row in rows])
    return np.array(
        [
            f"{row.get('game_id', 'unknown')}::{row.get('episode_id', 'unknown')}"
            for row in rows
        ]
    )


def _classifier(seed: int, *, n_estimators: int, min_samples_leaf: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
    )


def _metric_block(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def _mean_std(items: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted(items[0].keys()) if items else []
    return {
        key: {
            "mean": round(float(np.mean([item[key] for item in items])), 4),
            "std": round(float(np.std([item[key] for item in items])), 4),
        }
        for key in keys
    }


def _top_importances(
    model: RandomForestClassifier,
    feature_names: List[str],
    *,
    k: int = 20,
) -> List[Dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    order = np.argsort(importances)[::-1][:k]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 5)}
        for i in order
    ]


def train_and_evaluate(
    *,
    dataset_path: str,
    model_out: str,
    games: str = "all",
    n_splits: int = 5,
    seed: int = 0,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    group_by: str = "episode",
    with_history: bool = False,
    taxonomy: str = TAXONOMY_V1,
    quiet: bool = False,
) -> Dict[str, Any]:
    rows = _filter_games(load_rows(dataset_path), games)
    target_labels = taxonomy_labels(taxonomy)
    mapped_rows: List[Dict[str, Any]] = []
    mapped_targets: List[str] = []
    for row in rows:
        label = map_cognitive_phase(row, taxonomy)
        if label in target_labels:
            mapped_rows.append(row)
            mapped_targets.append(str(label))
    rows = mapped_rows
    if len(rows) < 20:
        raise SystemExit(f"Not enough cognitive rows ({len(rows)}).")

    x = _input_matrix(rows, with_history=with_history)
    y = np.array(mapped_targets)
    groups = _groups(rows, group_by)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise SystemExit(f"Need at least two {group_by} groups for CV.")
    effective_splits = max(2, min(int(n_splits), len(unique_groups)))
    feature_names = _input_feature_names(with_history=with_history)

    splitter = StratifiedGroupKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=seed,
    )

    oof_pred = np.empty(len(rows), dtype=object)
    oof_dummy = np.empty(len(rows), dtype=object)
    fold_metrics: List[Dict[str, float]] = []
    dummy_metrics: List[Dict[str, float]] = []
    folds: List[Dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
        # StratifiedGroupKFold may be too constrained for tiny future subsets.
        # The explicit labels/report below tolerate missing classes in a fold.
        model = _classifier(
            seed + fold_index,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        oof_pred[test_idx] = pred

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(x[train_idx], y[train_idx])
        dummy_pred = dummy.predict(x[test_idx])
        oof_dummy[test_idx] = dummy_pred

        metrics = _metric_block(y[test_idx], pred)
        baseline = _metric_block(y[test_idx], dummy_pred)
        fold_metrics.append(metrics)
        dummy_metrics.append(baseline)
        folds.append(
            {
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_groups": int(len(np.unique(groups[test_idx]))),
                "metrics": metrics,
                "baseline": baseline,
            }
        )

    report = classification_report(
        y,
        oof_pred,
        labels=target_labels,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y, oof_pred, labels=target_labels).tolist()
    label_counts = {label: int(np.sum(y == label)) for label in target_labels}

    final_model = _classifier(
        seed,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
    )
    final_model.fit(x, y)
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": final_model,
            "target_names": list(target_labels),
            "input_feature_names": feature_names,
            "with_history": bool(with_history),
            "taxonomy": taxonomy,
        },
        out_path,
    )

    metrics: Dict[str, Any] = {
        "dataset": dataset_path,
        "taxonomy": taxonomy,
        "target_column": "cognitive_phase" if taxonomy == TAXONOMY_V1 else "cognitive_phase_v2",
        "rows": int(len(rows)),
        "games": sorted({str(row.get("game_id", "")) for row in rows}),
        "group_by": group_by,
        "groups": int(len(unique_groups)),
        "n_splits": int(effective_splits),
        "with_history": bool(with_history),
        "label_counts": label_counts,
        "cv": _mean_std(fold_metrics),
        "baseline": _mean_std(dummy_metrics),
        "oof": _metric_block(y, oof_pred),
        "oof_baseline": _metric_block(y, oof_dummy),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in target_labels
        },
        "confusion_matrix": {
            "labels": list(target_labels),
            "matrix": confusion,
        },
        "folds": folds,
        "top_feature_importances": _top_importances(final_model, feature_names),
        "model_out": str(out_path),
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not quiet:
        print(
            f"Cognitive phase classifier ({taxonomy}): rows={len(rows)} "
            f"groups={len(unique_groups)} folds={effective_splits}"
        )
        print(
            "OOF: "
            f"acc={metrics['oof']['accuracy']:.3f} "
            f"bal_acc={metrics['oof']['balanced_accuracy']:.3f} "
            f"macro_f1={metrics['oof']['macro_f1']:.3f}"
        )
        print(
            "Baseline: "
            f"acc={metrics['oof_baseline']['accuracy']:.3f} "
            f"bal_acc={metrics['oof_baseline']['balanced_accuracy']:.3f} "
            f"macro_f1={metrics['oof_baseline']['macro_f1']:.3f}"
        )
        print("Per-class F1:")
        for label in target_labels:
            item = metrics["per_class"][label]
            print(
                f"  {label:22s} f1={item['f1']:.3f} "
                f"recall={item['recall']:.3f} support={item['support']}"
            )
        print("Top state features:")
        for item in metrics["top_feature_importances"][:10]:
            print(f"  {item['feature']:42s} {item['importance']:.4f}")
        print(f"Saved -> {out_path}")
        print(f"Metrics -> {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validated state -> cognitive_phase classifier."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_MODEL_OUT)
    parser.add_argument("--games", default="all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--taxonomy",
        choices=["v1", "v2"],
        default="v1",
        help="Label taxonomy only; model/features stay unchanged.",
    )
    parser.add_argument(
        "--group-by",
        choices=["episode", "game"],
        default="episode",
        help="CV grouping; episode is the default anti-leakage setting.",
    )
    parser.add_argument(
        "--with-history",
        action="store_true",
        help="Ablation: append short action-history features to state features.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train_and_evaluate(
        dataset_path=args.dataset,
        model_out=args.out,
        games=args.games,
        n_splits=args.folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        group_by=args.group_by,
        with_history=args.with_history,
        taxonomy=args.taxonomy,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
