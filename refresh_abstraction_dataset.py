"""Refresh derived annotations in an existing abstraction dataset JSONL.

This is intentionally env-free: it does not step games or change transitions.
It only derives fields that can be recomputed from existing rows:

    history_features
    macro_scores
    future_level_up_within_{5,10,25}
    discounted_future_progress

Use this after changing label/feature annotations when a full rebuild would add
noise or take longer than needed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import abstraction_labels as labels
from abstraction_dataset_io import make_history_features

DEFAULT_DATASET = "training/abstraction_dataset.jsonl"
LEVELUP_HORIZONS = (5, 10, 25)


def _episode_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row.get("game_id", "")),
        str(row.get("episode_source", "")),
        int(row.get("episode_index", -1)),
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _attach_history(rows: List[Dict[str, Any]]) -> None:
    episodes: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        episodes[_episode_key(row)].append(idx)

    for indices in episodes.values():
        action_history: List[str] = []
        action_repeat_count = 0
        steps_since_state_change = 0
        for idx in indices:
            row = rows[idx]
            row["history_features"] = make_history_features(
                action_history,
                action_repeat_count,
                steps_since_state_change,
            )
            action = str(row.get("action", ""))
            action_repeat_count = (
                action_repeat_count + 1
                if action_history and action_history[-1] == action
                else 1
            )
            action_history.append(action)
            changed = int(row.get("changed_cells", 0))
            steps_since_state_change = 0 if changed > 0 else steps_since_state_change + 1


def _attach_future_horizons(rows: List[Dict[str, Any]]) -> None:
    episodes: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        episodes[_episode_key(row)].append(idx)

    for indices in episodes.values():
        n = len(indices)
        for pos, idx in enumerate(indices):
            row = rows[idx]
            steps_to = -1
            for ahead in range(pos, n):
                if rows[indices[ahead]].get("level_up"):
                    steps_to = ahead - pos
                    break
            row["steps_to_level_up"] = int(steps_to)
            for horizon in LEVELUP_HORIZONS:
                row[f"future_level_up_within_{horizon}"] = bool(0 <= steps_to <= horizon)
            row["future_level_up"] = bool(0 <= steps_to <= max(LEVELUP_HORIZONS))
            row["discounted_future_progress"] = round(
                float(1.0 / (1.0 + steps_to)) if steps_to >= 0 else 0.0,
                4,
            )


def _attach_macro_scores(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        state_features = row.get("state_features", {})
        total_cells = int(
            float(state_features.get("grid_height", 64.0))
            * float(state_features.get("grid_width", 64.0))
        )
        delta = row.get("delta_features", {})
        changed_cells = int(row.get("changed_cells", 0))
        game_over = bool(row.get("game_over"))
        auto_levelup_progress = float(row.get("auto_levelup_progress", 0.0))
        row["macro_scores"] = {
            key: round(float(value), 4)
            for key, value in labels.macro_scores(
                delta=delta,
                changed_cells=changed_cells,
                total_cells=total_cells,
                game_over=game_over,
                auto_levelup_progress=auto_levelup_progress,
            ).items()
        }
        row["macro_label"] = labels.macro_label(
            delta=delta,
            changed_cells=changed_cells,
            total_cells=total_cells,
            game_over=game_over,
            auto_levelup_progress=auto_levelup_progress,
        )


def refresh_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _attach_history(rows)
    _attach_future_horizons(rows)
    _attach_macro_scores(rows)
    return rows


def _refresh_schema(dataset_path: Path) -> None:
    schema_path = dataset_path.with_suffix(".schema.json")
    if not schema_path.exists():
        return
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    schema["macro_score_names"] = labels.MACRO_SCORE_NAMES
    schema["history_features"] = [
        "last_action",
        "last_two_actions",
        "action_repeat_count",
        "steps_since_state_change",
    ]
    schema["levelup_horizons"] = list(LEVELUP_HORIZONS)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def refresh_file(dataset_path: Path, out_path: Path) -> Dict[str, Any]:
    rows = refresh_rows(_read_jsonl(dataset_path))
    if dataset_path.resolve() == out_path.resolve():
        tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
        _write_jsonl(tmp_path, rows)
        tmp_path.replace(dataset_path)
        _refresh_schema(dataset_path)
    else:
        _write_jsonl(out_path, rows)
        _refresh_schema(out_path)
    return {
        "rows": len(rows),
        "out": str(out_path),
        "history_rows": sum(1 for row in rows if "history_features" in row),
        "macro_score_rows": sum(1 for row in rows if "macro_scores" in row),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh derived abstraction dataset annotations.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=None)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if args.in_place:
        out_path = dataset_path
    elif args.out:
        out_path = Path(args.out)
    else:
        out_path = dataset_path.with_name(dataset_path.stem + ".refreshed" + dataset_path.suffix)
    summary = refresh_file(dataset_path, out_path)
    print(
        f"Refreshed {summary['rows']} rows -> {summary['out']} "
        f"(history={summary['history_rows']}, macro_scores={summary['macro_score_rows']})"
    )


if __name__ == "__main__":
    main()
