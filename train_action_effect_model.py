"""Model A - Action Effect: predict what an action does to the abstractions.

Current inputs are state features, optional short-history features, and the
candidate action one-hot. Older JSONL rows without history are zero-filled.

Multi-output regressor mapping ``state_features ⊕ one-hot(action)`` to the
expected deltas:

    delta_largest_component_size
    delta_component_count
    delta_top_pair_0_global_correspondence
    delta_fragmentation_ratio

We report per-target R²/MAE and the top feature importances so we can answer
the scientific question: *which abstractions actually predict action effects?*

Pure-sklearn / CPU. Run with either interpreter (no env stepping needed):
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe train_action_effect_model.py \\
        --dataset training\\abstraction_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import game_splits
from abstraction_dataset_io import (
    ACTION_EFFECT_TARGETS,
    action_effect_xy,
    input_feature_names,
    load_dataset,
)

DEFAULT_DATASET = "training/abstraction_dataset.jsonl"
DEFAULT_MODEL_OUT = "models/action_effect.joblib"


def _top_importances(model: RandomForestRegressor, names: List[str], k: int = 15) -> List[Dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    order = np.argsort(importances)[::-1][:k]
    return [
        {"feature": names[i], "importance": round(float(importances[i]), 5)}
        for i in order
    ]


def train(
    *,
    dataset_path: str,
    model_out: str,
    train_games: str,
    sources: List[str] | None,
    test_size: float,
    n_estimators: int,
    seed: int,
    quiet: bool,
) -> Dict[str, Any]:
    data = load_dataset(dataset_path)
    if train_games and train_games.lower() != "all":
        wanted = game_splits.resolve(train_games, full_ids=True)
        data = data.filter_games(wanted)
    data = data.filter_sources(sources)
    rows = data.rows
    if len(rows) < 20:
        raise SystemExit(f"Not enough rows ({len(rows)}) to train Model A.")

    x, y, target_names = action_effect_xy(rows)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)

    metrics: Dict[str, Any] = {"per_target": {}, "n_train": len(x_tr), "n_test": len(x_te)}
    for i, target in enumerate(target_names):
        yi, pi = y_te[:, i], pred[:, i]
        metrics["per_target"][target] = {
            "r2": round(float(r2_score(yi, pi)) if np.std(yi) > 1e-9 else 0.0, 4),
            "mae": round(float(mean_absolute_error(yi, pi)), 4),
            "target_std": round(float(np.std(yi)), 4),
        }
    metrics["mean_r2"] = round(
        float(np.mean([m["r2"] for m in metrics["per_target"].values()])), 4
    )
    feature_names = input_feature_names()
    metrics["top_feature_importances"] = _top_importances(model, feature_names)

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "target_names": target_names,
            "input_feature_names": feature_names,
        },
        out_path,
    )
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not quiet:
        print(f"Trained Model A on {len(rows)} rows ({metrics['n_train']} train).")
        print(f"mean R2 = {metrics['mean_r2']}")
        for target, m in metrics["per_target"].items():
            print(f"  {target:42s} R2={m['r2']:+.3f} MAE={m['mae']:.3f} (std={m['target_std']})")
        print("Top features:")
        for item in metrics["top_feature_importances"][:10]:
            print(f"  {item['feature']:42s} {item['importance']:.4f}")
        print(f"Saved -> {out_path}\nMetrics -> {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model A (action-effect).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_MODEL_OUT)
    parser.add_argument("--train-games", default="public_seen", help="Split alias or comma list (or 'all').")
    parser.add_argument("--sources", default=None, help="Comma list of episode_source to keep (ablation).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
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
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
