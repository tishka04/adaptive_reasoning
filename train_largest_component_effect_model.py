"""Specialized local model for delta_largest_component_size.

This experiment deliberately avoids the full 75-feature abstraction vector. It
uses only largest-component local geometry, short action history, and action
one-hot to test whether the weak Model-A R2 on largest-component effects is a
feature-description problem or mostly a rare-event imbalance problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import game_splits
from abstraction_dataset_io import (
    action_onehot,
    history_matrix,
    input_feature_names,
    largest_component_matrix,
    largest_component_input_feature_names,
    largest_component_input_matrix,
    load_dataset,
    state_matrix,
)

DEFAULT_DATASET = "training/abstraction_dataset.jsonl"
DEFAULT_OUT = "models/largest_component_effect.joblib"
TARGET = "delta_largest_component_size"


def _inputs(rows: List[Dict[str, Any]], mode: str) -> tuple[np.ndarray, List[str]]:
    if mode == "local":
        return largest_component_input_matrix(rows), largest_component_input_feature_names()
    if mode == "state_local":
        x = np.concatenate(
            [state_matrix(rows), largest_component_matrix(rows), history_matrix(rows), action_onehot(rows)],
            axis=1,
        )
        names = (
            list(input_feature_names()[:75])
            + [
                name
                for name in largest_component_input_feature_names()
                if not name.startswith("last_action_is_")
                and not name.startswith("prev_action_is_")
                and name not in {"action_repeat_count", "steps_since_state_change"}
                and not name.startswith("is_ACTION")
            ]
            + [
                name
                for name in input_feature_names()[75:]
            ]
        )
        return x, names
    raise ValueError(f"Unknown input mode: {mode}")


def _top_importances(model: RandomForestRegressor, names: List[str], k: int = 15) -> List[Dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    order = np.argsort(importances)[::-1][:k]
    return [
        {"feature": names[i], "importance": round(float(importances[i]), 5)}
        for i in order
    ]


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    if y_true.size == 0:
        return {"count": 0, "r2": None, "mae": None, "target_std": None}
    return {
        "count": int(y_true.size),
        "r2": round(float(r2_score(y_true, pred)) if np.std(y_true) > 1e-9 else 0.0, 4),
        "mae": round(float(mean_absolute_error(y_true, pred)), 4),
        "target_std": round(float(np.std(y_true)), 4),
    }


def _baseline_metric(path: str) -> Optional[float]:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    target = payload.get("per_target", {}).get(TARGET, {})
    value = target.get("r2")
    return float(value) if value is not None else None


def train(
    *,
    dataset_path: str,
    model_out: str,
    train_games: str,
    sources: Optional[List[str]],
    test_size: float,
    n_estimators: int,
    seed: int,
    baseline_metrics: str,
    input_mode: str,
    quiet: bool,
) -> Dict[str, Any]:
    data = load_dataset(dataset_path)
    if train_games and train_games.lower() != "all":
        data = data.filter_games(game_splits.resolve(train_games, full_ids=True))
    data = data.filter_sources(sources)
    rows = data.rows
    if len(rows) < 20:
        raise SystemExit(f"Not enough rows ({len(rows)}) to train largest-component model.")

    x, feature_names = _inputs(rows, input_mode)
    y = np.array(
        [float(row.get("delta_features", {}).get(TARGET, 0.0)) for row in rows],
        dtype=np.float32,
    )
    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x[tr], y[tr])
    pred = model.predict(x[te])
    nonzero_mask = np.abs(y[te]) > 1e-9

    missing_local = sum(1 for row in rows if "largest_component_features" not in row)
    overall = _regression_metrics(y[te], pred)
    nonzero = _regression_metrics(y[te][nonzero_mask], pred[nonzero_mask])
    baseline_r2 = _baseline_metric(baseline_metrics)
    metrics: Dict[str, Any] = {
        "target": TARGET,
        "n_rows": len(rows),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "feature_count": len(feature_names),
        "input_mode": input_mode,
        "local_feature_count": len(largest_component_input_feature_names()) - 16 - 7,
        "missing_local_feature_rows": int(missing_local),
        "overall": overall,
        "nonzero_target": nonzero,
        "nonzero_target_rate": round(float(np.mean(np.abs(y) > 1e-9)), 4),
        "baseline_model_a_r2": baseline_r2,
        "r2_delta_vs_baseline": (
            round(float(overall["r2"] - baseline_r2), 4)
            if baseline_r2 is not None and overall["r2"] is not None
            else None
        ),
        "top_feature_importances": _top_importances(model, feature_names),
    }

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "target_name": TARGET,
            "input_feature_names": feature_names,
        },
        out_path,
    )
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not quiet:
        print(f"Trained largest-component model on {len(rows)} rows ({metrics['n_train']} train).")
        print(
            f"overall R2={overall['r2']:+.4f} MAE={overall['mae']:.4f} "
            f"(std={overall['target_std']})"
        )
        print(
            f"nonzero R2={nonzero['r2']} MAE={nonzero['mae']} "
            f"count={nonzero['count']} rate={metrics['nonzero_target_rate']}"
        )
        if baseline_r2 is not None:
            print(f"baseline Model A R2={baseline_r2:+.4f}; delta={metrics['r2_delta_vs_baseline']:+.4f}")
        print("Top features:")
        for item in metrics["top_feature_importances"][:10]:
            print(f"  {item['feature']:42s} {item['importance']:.4f}")
        print(f"Saved -> {out_path}\nMetrics -> {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train specialized largest-component effect model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--train-games", default="public_seen")
    parser.add_argument("--sources", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-metrics", default="models/action_effect_history.metrics.json")
    parser.add_argument(
        "--input-mode",
        choices=["local", "state_local"],
        default="local",
        help="local = local+history+action only; state_local = 75 state + local + history + action.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    train(
        dataset_path=args.dataset,
        model_out=args.out,
        train_games=args.train_games,
        sources=sources,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        seed=args.seed,
        baseline_metrics=args.baseline_metrics,
        input_mode=args.input_mode,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
