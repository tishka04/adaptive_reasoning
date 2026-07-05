"""Build a separate rare-event dataset for largest-component breaks.

The main abstraction dataset stays general-purpose. This derived dataset adds a
three-way break label for rare-event training/evaluation:

    BIG_BREAK     delta_largest_component_size <= -50
    SMALL_CHANGE  abs(delta_largest_component_size) < 5
    OTHER_CHANGE  everything else

By default we write both a natural distribution file and an oversampled training
pool. Model training should evaluate on the natural distribution to avoid
inflated rare-event metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import game_splits
from abstraction_dataset_io import load_dataset

DEFAULT_INPUT = "training/abstraction_dataset.jsonl"
DEFAULT_OUT = "training/break_event_dataset.jsonl"
DEFAULT_NATURAL_OUT = "training/break_event_dataset.natural.jsonl"

BIG_BREAK_THRESHOLD = -50.0
SMALL_CHANGE_ABS_THRESHOLD = 5.0
BREAK_CLASSES = ["BIG_BREAK", "SMALL_CHANGE", "OTHER_CHANGE"]


def break_class(delta_largest_component_size: float) -> str:
    if delta_largest_component_size <= BIG_BREAK_THRESHOLD:
        return "BIG_BREAK"
    if abs(delta_largest_component_size) < SMALL_CHANGE_ABS_THRESHOLD:
        return "SMALL_CHANGE"
    return "OTHER_CHANGE"


def _split_name(game_id: str) -> str:
    return game_splits.split_for_game(game_id) or "unknown"


def _event_row(row: Dict[str, Any], row_id: int) -> Dict[str, Any]:
    delta = float(row.get("delta_features", {}).get("delta_largest_component_size", 0.0))
    label = break_class(delta)
    return {
        "row_id": int(row_id),
        "game_id": row.get("game_id"),
        "split": _split_name(str(row.get("game_id", ""))),
        "level": row.get("level", 0),
        "episode_source": row.get("episode_source"),
        "episode_index": row.get("episode_index", 0),
        "step": row.get("step", 0),
        "history_features": row.get("history_features", {}),
        "action": row.get("action"),
        "action_data": row.get("action_data"),
        "state_features": row.get("state_features", {}),
        "largest_component_features": row.get("largest_component_features", {}),
        "changed_cells": row.get("changed_cells", 0),
        "level_up": bool(row.get("level_up", False)),
        "game_over": bool(row.get("game_over", False)),
        "delta_largest_component_size": delta,
        "break_class": label,
        "is_big_break": label == "BIG_BREAK",
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


def _oversample(rows: List[Dict[str, Any]], *, seed: int, target_per_class: Optional[int]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {label: [] for label in BREAK_CLASSES}
    for row in rows:
        grouped.setdefault(str(row.get("break_class", "OTHER_CHANGE")), []).append(row)
    max_count = max((len(items) for items in grouped.values()), default=0)
    target = target_per_class or max_count
    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    for label in BREAK_CLASSES:
        items = grouped.get(label, [])
        if not items:
            continue
        sampled.extend(items)
        needed = max(0, target - len(items))
        for _ in range(needed):
            clone = dict(rng.choice(items))
            clone["oversampled"] = True
            sampled.append(clone)
    rng.shuffle(sampled)
    return sampled


def build(
    *,
    dataset_path: str,
    out_path: str,
    natural_out_path: str,
    games: str,
    seed: int,
    target_per_class: Optional[int],
    quiet: bool,
) -> Dict[str, Any]:
    data = load_dataset(dataset_path)
    if games and games.lower() != "all":
        data = data.filter_games(game_splits.resolve(games, full_ids=True))
    natural_rows = [_event_row(row, i) for i, row in enumerate(data.rows)]
    oversampled_rows = _oversample(
        natural_rows,
        seed=seed,
        target_per_class=target_per_class,
    )

    natural_path = Path(natural_out_path)
    out = Path(out_path)
    natural_count = _write_jsonl(natural_path, natural_rows)
    oversampled_count = _write_jsonl(out, oversampled_rows)

    natural_counts = Counter(row["break_class"] for row in natural_rows)
    oversampled_counts = Counter(row["break_class"] for row in oversampled_rows)
    summary = {
        "source_dataset": dataset_path,
        "out": str(out),
        "natural_out": str(natural_path),
        "games": games,
        "seed": seed,
        "thresholds": {
            "big_break_lte": BIG_BREAK_THRESHOLD,
            "small_change_abs_lt": SMALL_CHANGE_ABS_THRESHOLD,
        },
        "classes": BREAK_CLASSES,
        "natural_rows": natural_count,
        "oversampled_rows": oversampled_count,
        "natural_counts": {label: int(natural_counts.get(label, 0)) for label in BREAK_CLASSES},
        "oversampled_counts": {label: int(oversampled_counts.get(label, 0)) for label in BREAK_CLASSES},
    }
    Path(out_path).with_suffix(".schema.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if not quiet:
        print(f"Natural rows -> {natural_path} ({natural_count})")
        print(f"Oversampled rows -> {out} ({oversampled_count})")
        print(f"natural_counts: {summary['natural_counts']}")
        print(f"oversampled_counts: {summary['oversampled_counts']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rare break-event dataset.")
    parser.add_argument("--dataset", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--natural-out", default=DEFAULT_NATURAL_OUT)
    parser.add_argument("--games", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Oversampled rows per class. Defaults to the natural majority count.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    build(
        dataset_path=args.dataset,
        out_path=args.out,
        natural_out_path=args.natural_out,
        games=args.games,
        seed=args.seed,
        target_per_class=args.target_per_class,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
