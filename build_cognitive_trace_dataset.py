"""Build a human cognitive-trace dataset from recorded ARC-AGI-3 traces.

The abstraction dataset answers: "what happens if I do this action?".
This dataset targets the next question:

    "Given this state, what is the human trying to discover?"

Each row is one human step with a shared state representation and three
main supervised labels plus rare cognitive event markers:

    - action             -> what the human did next
    - cognitive_phase    -> the recorder intent tag
    - hypothesis         -> the sticky human hypothesis active at that state
    - cognitive_events   -> rare markers such as hypothesis_confirmed

Run with the bundled interpreter:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe build_cognitive_trace_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from abstraction_dataset_io import ACTIONS, make_history_features
from extract_state_abstractions import FEATURE_SCHEMA, extract_state_features
from human_trace.schema import CognitiveEvent, EpisodeRecord, IntentTag, StepRecord, TraceReader
from level7_frontier_recovery import PROJECT_ROOT

DEFAULT_TRACES = PROJECT_ROOT / "human_traces"
DEFAULT_OUT = PROJECT_ROOT / "training" / "cognitive_trace_dataset.jsonl"
NONE_HYPOTHESIS = "__none__"
NONE_COGNITIVE_EVENT = "__none__"
ACTION_ID_TO_NAME = {0: "RESET", **{i: f"ACTION{i}" for i in range(1, 8)}}
PHASE_LABELS = [tag.value for tag in IntentTag.cycle_order()]
COGNITIVE_EVENT_LABELS = CognitiveEvent.labels()
COGNITIVE_EVENT_SET = set(COGNITIVE_EVENT_LABELS)


def _round_features(features: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in features.items()}


def _frame_shape(frame: Sequence[Sequence[int]]) -> List[int]:
    height = len(frame)
    width = len(frame[0]) if height else 0
    return [int(height), int(width)]


def _state_hash(frame: Sequence[Sequence[int]]) -> str:
    arr = np.array(frame, dtype=np.int32)
    digest = hashlib.sha1()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()[:16]


def _strip_status_rows(
    frame: Sequence[Sequence[int]],
    *,
    status_rows: int,
) -> Sequence[Sequence[int]]:
    if status_rows <= 0 or len(frame) <= status_rows:
        return frame
    return frame[:-status_rows]


def _changed_cells(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    *,
    status_rows: int = 0,
) -> int:
    left = np.array(_strip_status_rows(before, status_rows=status_rows), dtype=np.int32)
    right = np.array(_strip_status_rows(after, status_rows=status_rows), dtype=np.int32)
    if left.shape != right.shape:
        return int(max(left.size, right.size))
    return int(np.count_nonzero(left != right))


def _normalise_action_name(value: Any) -> str:
    if isinstance(value, int):
        return ACTION_ID_TO_NAME.get(value, f"ACTION{value}")
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if text.isdigit():
        return ACTION_ID_TO_NAME.get(int(text), f"ACTION{text}")
    return text


def _available_action_names(values: Iterable[Any]) -> List[str]:
    names = [_normalise_action_name(value) for value in values]
    return [name for name in names if name != "RESET"]


_WHITESPACE = re.compile(r"\s+")


def _clean_hypothesis(value: str) -> str:
    return _WHITESPACE.sub(" ", str(value or "").strip())


def _cognitive_phase(value: str) -> str:
    phase = str(value or "").strip()
    return phase if phase in PHASE_LABELS else IntentTag.NONE.value


def _cognitive_events(values: Any) -> List[str]:
    if values is None:
        raw_values: Iterable[Any] = []
    elif isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = values

    out: List[str] = []
    for value in raw_values:
        label = str(value or "").strip()
        if label in COGNITIVE_EVENT_SET and label not in out:
            out.append(label)
    return out


def _build_row(
    step: StepRecord,
    episode: Optional[EpisodeRecord],
    *,
    episode_index: int,
    level_before: int,
    history_features: Dict[str, Any],
    previous_hypothesis: str,
    status_rows: int,
    include_raw_grid: bool,
) -> Dict[str, Any]:
    hypothesis = _clean_hypothesis(step.hypothesis)
    hypothesis_label = hypothesis if hypothesis else NONE_HYPOTHESIS
    action = _normalise_action_name(step.action)
    phase = _cognitive_phase(step.intent)
    cognitive_events = _cognitive_events(getattr(step, "cognitive_events", []))
    cognitive_event_label = (
        "+".join(cognitive_events) if cognitive_events else NONE_COGNITIVE_EVENT
    )
    hypothesis_confirmed = CognitiveEvent.HYPOTHESIS_CONFIRMED.value in cognitive_events
    hypothesis_rejected = CognitiveEvent.HYPOTHESIS_REJECTED.value in cognitive_events
    goal_changed = CognitiveEvent.GOAL_CHANGED.value in cognitive_events
    action_args = dict(step.action_args or {})
    row: Dict[str, Any] = {
        "game_id": step.game_id,
        "episode_id": step.episode_id,
        "episode_index": int(episode_index),
        "step": int(step.step),
        "level_before": int(level_before),
        "level_after": int(step.levels_completed_after),
        "state_hash": _state_hash(step.frame_before),
        "frame_shape": _frame_shape(step.frame_before),
        "state_features": _round_features(extract_state_features(step.frame_before)),
        "history_features": history_features,
        "available_actions": _available_action_names(step.available_actions),
        "action": action,
        "action_args": action_args or None,
        "click_x": int(action_args["x"]) if "x" in action_args else None,
        "click_y": int(action_args["y"]) if "y" in action_args else None,
        "cognitive_phase": phase,
        "phase_index": PHASE_LABELS.index(phase),
        "cognitive_events": cognitive_events,
        "cognitive_event_label": cognitive_event_label,
        "has_cognitive_event": bool(cognitive_events),
        "hypothesis_confirmed": hypothesis_confirmed,
        "hypothesis_rejected": hypothesis_rejected,
        "goal_changed": goal_changed,
        "hypothesis": hypothesis,
        "hypothesis_label": hypothesis_label,
        "hypothesis_key": f"human::{hypothesis[:80]}" if hypothesis else NONE_HYPOTHESIS,
        "has_hypothesis": bool(hypothesis),
        "hypothesis_changed": hypothesis != previous_hypothesis,
        "targets": {
            "action": action,
            "cognitive_phase": phase,
            "hypothesis": hypothesis_label,
            "cognitive_event": cognitive_event_label,
            "hypothesis_confirmed": hypothesis_confirmed,
            "hypothesis_rejected": hypothesis_rejected,
            "goal_changed": goal_changed,
        },
        "next_game_state": str(step.game_state_after),
        "level_up": int(step.levels_completed_after) > int(level_before),
        "game_over": step.game_state_after == "GAME_OVER",
        "win": step.game_state_after == "WIN",
        "changed_cells": _changed_cells(step.frame_before, step.frame_after),
        "core_changed_cells": _changed_cells(
            step.frame_before,
            step.frame_after,
            status_rows=status_rows,
        ),
        "t_ms": int(step.t_ms),
        "episode_final_state": episode.final_state if episode is not None else None,
        "episode_levels_completed": (
            int(episode.levels_completed) if episode is not None else None
        ),
        "episode_game_type_guess": episode.game_type_guess if episode is not None else "",
        "episode_objective_guess": episode.objective_guess if episode is not None else "",
        "episode_discovered_mechanics": (
            list(episode.discovered_mechanics) if episode is not None else []
        ),
        "episode_discovered_mistakes": (
            list(episode.discovered_mistakes) if episode is not None else []
        ),
    }
    if include_raw_grid:
        row["state_grid"] = step.frame_before
    return row


def build_rows(
    *,
    trace_dir: str | Path,
    games: Optional[Sequence[str]] = None,
    include_reset: bool = False,
    include_empty_hypothesis: bool = True,
    status_rows: int = 1,
    include_raw_grid: bool = False,
) -> List[Dict[str, Any]]:
    """Build multitask cognitive rows from human trace JSONL files."""
    reader = TraceReader(trace_dir)
    wanted = {game for game in games or [] if game and game.lower() != "all"}
    game_filter = None if not wanted or len(wanted) != 1 else next(iter(wanted))
    grouped = reader.group_by_episode(game_id=game_filter)

    rows: List[Dict[str, Any]] = []
    episode_index_by_id = {
        episode_id: idx
        for idx, episode_id in enumerate(sorted(grouped.keys()))
    }
    for episode_id in sorted(grouped.keys()):
        bucket = grouped[episode_id]
        episode = bucket.get("episode")
        steps: List[StepRecord] = sorted(
            bucket.get("steps", []),
            key=lambda step: (int(step.t_ms), int(step.step)),
        )
        if not steps:
            continue
        game_id = episode.game_id if isinstance(episode, EpisodeRecord) else steps[0].game_id
        short_id = game_id.split("-", 1)[0]
        if wanted and game_id not in wanted and short_id not in wanted:
            continue

        action_history: List[str] = []
        action_repeat_count = 0
        steps_since_change = 0
        level_before = 0
        previous_hypothesis = ""
        for step in steps:
            action = _normalise_action_name(step.action)
            if action == "RESET":
                level_before = int(step.levels_completed_after)
                action_history = []
                action_repeat_count = 0
                steps_since_change = 0
                previous_hypothesis = _clean_hypothesis(step.hypothesis)
                if not include_reset:
                    continue

            hypothesis = _clean_hypothesis(step.hypothesis)
            if include_empty_hypothesis or hypothesis:
                rows.append(
                    _build_row(
                        step,
                        episode if isinstance(episode, EpisodeRecord) else None,
                        episode_index=episode_index_by_id[episode_id],
                        level_before=level_before,
                        history_features=make_history_features(
                            action_history,
                            action_repeat_count,
                            steps_since_change,
                        ),
                        previous_hypothesis=previous_hypothesis,
                        status_rows=status_rows,
                        include_raw_grid=include_raw_grid,
                    )
                )

            if action != "RESET":
                changed = _changed_cells(
                    step.frame_before,
                    step.frame_after,
                    status_rows=status_rows,
                )
                action_repeat_count = (
                    action_repeat_count + 1
                    if action_history and action_history[-1] == action
                    else 1
                )
                action_history.append(action)
                steps_since_change = 0 if changed > 0 else steps_since_change + 1
            level_before = int(step.levels_completed_after)
            previous_hypothesis = hypothesis
    return rows


def _schema(rows: List[Dict[str, Any]], *, trace_dir: Path) -> Dict[str, Any]:
    action_counts = Counter(row["action"] for row in rows)
    phase_counts = Counter(row["cognitive_phase"] for row in rows)
    cognitive_event_counts = Counter(row["cognitive_event_label"] for row in rows)
    hypothesis_counts = Counter(row["hypothesis_label"] for row in rows)
    game_counts = Counter(row["game_id"] for row in rows)
    return {
        "description": "Human cognitive trace multitask dataset.",
        "trace_dir": str(trace_dir),
        "rows": len(rows),
        "games": dict(sorted(game_counts.items())),
        "feature_schema": list(FEATURE_SCHEMA),
        "action_labels": list(ACTIONS),
        "cognitive_phase_labels": PHASE_LABELS,
        "cognitive_event_labels": COGNITIVE_EVENT_LABELS,
        "none_cognitive_event_label": NONE_COGNITIVE_EVENT,
        "none_hypothesis_label": NONE_HYPOTHESIS,
        "target_columns": [
            "action",
            "cognitive_phase",
            "hypothesis_label",
            "cognitive_event_label",
            "hypothesis_confirmed",
            "hypothesis_rejected",
            "goal_changed",
        ],
        "action_counts": dict(action_counts.most_common()),
        "cognitive_phase_counts": dict(phase_counts.most_common()),
        "cognitive_event_counts": dict(cognitive_event_counts.most_common()),
        "hypothesis_counts_top20": dict(hypothesis_counts.most_common(20)),
        "columns": {
            "state_features": "dict over feature_schema extracted from frame_before",
            "history_features": "last actions and local stall counters before the action",
            "action": "human primitive action label",
            "cognitive_phase": "human intent tag recorded via Tab",
            "cognitive_events": "list of rare cognitive event markers pending before the action",
            "cognitive_event_label": "joined cognitive_events label, or __none__",
            "hypothesis_confirmed": "binary rare-event target",
            "hypothesis_rejected": "binary rare-event target",
            "goal_changed": "binary rare-event target",
            "hypothesis_label": "sticky hypothesis text, or __none__",
            "targets": "labels grouped for multitask training",
        },
    }


def write_dataset(rows: List[Dict[str, Any]], out_path: str | Path, *, trace_dir: Path) -> Dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    schema = _schema(rows, trace_dir=trace_dir)
    out.with_suffix(".schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return schema


def _parse_games(value: str) -> Optional[List[str]]:
    if not value or value.lower() == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a multitask dataset from human cognitive traces."
    )
    parser.add_argument("--traces", default=str(DEFAULT_TRACES))
    parser.add_argument("--games", default="all", help="Comma list, e.g. bp35,cd82; default all.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--include-reset", action="store_true")
    parser.add_argument("--drop-empty-hypothesis", action="store_true")
    parser.add_argument(
        "--status-rows",
        type=int,
        default=1,
        help="Rows at the bottom ignored only for core_changed_cells/history no-op.",
    )
    parser.add_argument("--include-raw-grid", action="store_true")
    args = parser.parse_args()

    trace_dir = Path(args.traces)
    rows = build_rows(
        trace_dir=trace_dir,
        games=_parse_games(args.games),
        include_reset=args.include_reset,
        include_empty_hypothesis=not args.drop_empty_hypothesis,
        status_rows=args.status_rows,
        include_raw_grid=args.include_raw_grid,
    )
    schema = write_dataset(rows, args.out, trace_dir=trace_dir)
    print(
        f"Wrote {schema['rows']} rows across {len(schema['games'])} games -> {args.out}"
    )
    print(f"Schema -> {Path(args.out).with_suffix('.schema.json')}")


if __name__ == "__main__":
    main()
