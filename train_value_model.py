"""Model B - Value, and Model C - Macro Affordance.

Transition-conditioned models use state features, optional short-history
features, and the candidate action one-hot. Missing history is zero-filled.

Model B (the most important one) maps ``state_features`` to:
    progress_score        -> GradientBoostingRegressor
    level_up_probability  -> GradientBoostingClassifier  (target: future_level_up)
    danger_probability    -> GradientBoostingClassifier  (target: future_game_over_soon)

Model C maps ``state_features ⊕ one-hot(action)`` to the macro affordance label
(RandomForestClassifier) so we can inspect per-class errors.

Pure-sklearn / CPU:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe train_value_model.py \\
        --dataset training\\abstraction_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import dump
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import game_splits
from abstraction_dataset_io import (
    input_feature_names,
    load_dataset,
    state_matrix,
    transition_input_matrix,
)
from abstraction_labels import MACRO_LABELS, MACRO_SCORE_NAMES
from extract_state_abstractions import FEATURE_SCHEMA

DEFAULT_DATASET = "training/abstraction_dataset.jsonl"
DEFAULT_VALUE_OUT = "models/value.joblib"
DEFAULT_MACRO_OUT = "models/macro.joblib"
DEFAULT_MACRO_SCORE_OUT = "models/macro_scores.joblib"


def _fit_classifier(x_tr, y_tr, seed: int):
    """Fit a probability classifier, tolerating single-class targets."""

    classes = np.unique(y_tr)
    if classes.size < 2:
        return {"constant": float(classes[0]) if classes.size else 0.0}
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(x_tr, y_tr)
    return clf


def _auc(clf, x_te, y_te) -> Optional[float]:
    if isinstance(clf, dict):
        return None
    if np.unique(y_te).size < 2:
        return None
    proba = clf.predict_proba(x_te)[:, 1]
    return round(float(roc_auc_score(y_te, proba)), 4)


def _future_level_target(row: Dict[str, Any], horizon: int) -> bool:
    key = f"future_level_up_within_{horizon}"
    if key in row:
        return bool(row.get(key))
    steps_to = int(row.get("steps_to_level_up", -1))
    return 0 <= steps_to <= horizon


def _progress_target(rows: List[Dict[str, Any]]) -> tuple[np.ndarray, str]:
    if any("discounted_future_progress" in row for row in rows):
        values = [float(row.get("discounted_future_progress", 0.0)) for row in rows]
        return np.array(values, dtype=np.float32), "discounted_future_progress"
    values = [float(row.get("progress_score", 0.0)) for row in rows]
    return np.array(values, dtype=np.float32), "progress_score"


def _row_macro_scores(row: Dict[str, Any]) -> List[float]:
    scores = row.get("macro_scores")
    if isinstance(scores, dict):
        return [float(scores.get(name, 0.0)) for name in MACRO_SCORE_NAMES]
    return [
        max(0.0, float(row.get("break_progress", 0.0))),
        max(0.0, float(row.get("fragmentation_progress", 0.0))),
        max(
            0.0,
            float(row.get("correspondence_progress", 0.0)),
            float(row.get("auto_levelup_progress", 0.0)),
        ),
        float(row.get("no_op", 0.0)),
        float(row.get("danger", 0.0)),
    ]


def train_value_model(rows: List[Dict[str, Any]], *, test_size: float, seed: int) -> Dict[str, Any]:
    x = state_matrix(rows)
    progress, progress_target_name = _progress_target(rows)
    levelup = np.array([int(bool(r.get("future_level_up", False))) for r in rows])
    levelup5 = np.array([int(_future_level_target(r, 5)) for r in rows])
    levelup10 = np.array([int(_future_level_target(r, 10)) for r in rows])
    danger = np.array([int(bool(r.get("future_game_over_soon", False))) for r in rows])

    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)

    reg = GradientBoostingRegressor(random_state=seed)
    reg.fit(x[tr], progress[tr])
    progress_mae = round(float(mean_absolute_error(progress[te], reg.predict(x[te]))), 4)

    levelup_clf = _fit_classifier(x[tr], levelup[tr], seed)
    levelup5_clf = _fit_classifier(x[tr], levelup5[tr], seed)
    levelup10_clf = _fit_classifier(x[tr], levelup10[tr], seed)
    danger_clf = _fit_classifier(x[tr], danger[tr], seed)

    importances = getattr(reg, "feature_importances_", np.zeros(len(FEATURE_SCHEMA)))
    order = np.argsort(importances)[::-1][:15]
    top_features = [
        {"feature": FEATURE_SCHEMA[i], "importance": round(float(importances[i]), 5)}
        for i in order
    ]

    metrics = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "progress_target": progress_target_name,
        "progress_mae": progress_mae,
        "level_up_auc": _auc(levelup_clf, x[te], levelup[te]),
        "level_up_within_5_auc": _auc(levelup5_clf, x[te], levelup5[te]),
        "level_up_within_10_auc": _auc(levelup10_clf, x[te], levelup10[te]),
        "danger_auc": _auc(danger_clf, x[te], danger[te]),
        "level_up_positive_rate": round(float(levelup.mean()), 4),
        "level_up_within_5_positive_rate": round(float(levelup5.mean()), 4),
        "level_up_within_10_positive_rate": round(float(levelup10.mean()), 4),
        "danger_positive_rate": round(float(danger.mean()), 4),
        "top_feature_importances": top_features,
    }
    bundle = {
        "progress_regressor": reg,
        "level_up_classifier": levelup_clf,
        "level_up_within_5_classifier": levelup5_clf,
        "level_up_within_10_classifier": levelup10_clf,
        "danger_classifier": danger_clf,
        "progress_target": progress_target_name,
        "input_feature_names": list(FEATURE_SCHEMA),
    }
    return {"bundle": bundle, "metrics": metrics}


def train_macro_model(rows: List[Dict[str, Any]], *, test_size: float, seed: int) -> Dict[str, Any]:
    x = transition_input_matrix(rows)
    y = np.array([r.get("macro_label", "UNKNOWN") for r in rows])

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)
    clf = RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=2, class_weight="balanced"
    )
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)

    labels_present = sorted(set(y.tolist()))
    report = classification_report(
        y_te, pred, labels=labels_present, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_te, pred, labels=labels_present).tolist()
    metrics = {
        "n_train": int(len(x_tr)),
        "n_test": int(len(x_te)),
        "labels": labels_present,
        "classification_report": report,
        "confusion_matrix": cm,
        "label_distribution": {
            label: int(np.sum(y == label)) for label in MACRO_LABELS if np.any(y == label)
        },
    }
    bundle = {"model": clf, "labels": labels_present, "input_feature_names": input_feature_names()}
    return {"bundle": bundle, "metrics": metrics}


def train_macro_score_model(rows: List[Dict[str, Any]], *, test_size: float, seed: int) -> Dict[str, Any]:
    x = transition_input_matrix(rows)
    y = np.array([_row_macro_scores(row) for row in rows], dtype=np.float32)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)
    reg = RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    reg.fit(x_tr, y_tr)
    pred = reg.predict(x_te)

    per_target: Dict[str, Any] = {}
    for i, target in enumerate(MACRO_SCORE_NAMES):
        yi, pi = y_te[:, i], pred[:, i]
        per_target[target] = {
            "r2": round(float(r2_score(yi, pi)) if np.std(yi) > 1e-9 else 0.0, 4),
            "mae": round(float(mean_absolute_error(yi, pi)), 4),
            "target_mean": round(float(np.mean(yi)), 4),
            "target_std": round(float(np.std(yi)), 4),
            "positive_rate": round(float(np.mean(yi > 1e-9)), 4),
        }
    metrics = {
        "n_train": int(len(x_tr)),
        "n_test": int(len(x_te)),
        "targets": list(MACRO_SCORE_NAMES),
        "per_target": per_target,
        "mean_r2": round(float(np.mean([m["r2"] for m in per_target.values()])), 4),
    }
    bundle = {
        "model": reg,
        "target_names": list(MACRO_SCORE_NAMES),
        "input_feature_names": input_feature_names(),
    }
    return {"bundle": bundle, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model B (value) and Model C (macro).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--value-out", default=DEFAULT_VALUE_OUT)
    parser.add_argument("--macro-out", default=DEFAULT_MACRO_OUT)
    parser.add_argument("--macro-score-out", default=DEFAULT_MACRO_SCORE_OUT)
    parser.add_argument("--train-games", default="public_seen")
    parser.add_argument("--sources", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.train_games and args.train_games.lower() != "all":
        data = data.filter_games(game_splits.resolve(args.train_games, full_ids=True))
    if args.sources:
        data = data.filter_sources([s.strip() for s in args.sources.split(",")])
    rows = data.rows
    if len(rows) < 20:
        raise SystemExit(f"Not enough rows ({len(rows)}) to train Models B/C.")

    value = train_value_model(rows, test_size=args.test_size, seed=args.seed)
    macro = train_macro_model(rows, test_size=args.test_size, seed=args.seed)
    macro_scores = train_macro_score_model(rows, test_size=args.test_size, seed=args.seed)

    for out, result in (
        (args.value_out, value),
        (args.macro_out, macro),
        (args.macro_score_out, macro_scores),
    ):
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump(result["bundle"], out_path)
        out_path.with_suffix(".metrics.json").write_text(
            json.dumps(result["metrics"], indent=2), encoding="utf-8"
        )

    if not args.quiet:
        vm = value["metrics"]
        print(f"Model B (value): {vm['n_train']} train / {vm['n_test']} test")
        print(f"  progress_target = {vm['progress_target']}")
        print(f"  progress_mae = {vm['progress_mae']}")
        print(f"  level_up_auc = {vm['level_up_auc']} (pos rate {vm['level_up_positive_rate']})")
        print(
            f"  level_up@5_auc = {vm['level_up_within_5_auc']} "
            f"(pos rate {vm['level_up_within_5_positive_rate']})"
        )
        print(
            f"  level_up@10_auc = {vm['level_up_within_10_auc']} "
            f"(pos rate {vm['level_up_within_10_positive_rate']})"
        )
        print(f"  danger_auc   = {vm['danger_auc']} (pos rate {vm['danger_positive_rate']})")
        print("  top value features:")
        for item in vm["top_feature_importances"][:8]:
            print(f"    {item['feature']:40s} {item['importance']:.4f}")
        mm = macro["metrics"]
        print(f"\nModel C (macro): labels={mm['labels']}")
        print(f"  label_distribution = {mm['label_distribution']}")
        acc = mm["classification_report"].get("accuracy")
        print(f"  accuracy = {round(float(acc), 4) if acc is not None else 'n/a'}")
        sm = macro_scores["metrics"]
        print(f"\nModel C-score (macro scores): mean R2={sm['mean_r2']}")
        for target, m in sm["per_target"].items():
            print(
                f"  {target:12s} R2={m['r2']:+.3f} MAE={m['mae']:.3f} "
                f"(pos={m['positive_rate']})"
            )
        print(
            f"Saved value -> {args.value_out}; macro -> {args.macro_out}; "
            f"macro_scores -> {args.macro_score_out}"
        )


if __name__ == "__main__":
    main()
