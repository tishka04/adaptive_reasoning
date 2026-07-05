"""M2.14b offline transition dataset builder for ARC-LeWM."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

from .schema import M2_TRUTH_STATUS


DEFAULT_ARC_LEWM_TRANSITIONS_OUTPUT_PATH = (
    Path("training") / "m2_arc_lewm_transitions.jsonl"
)
DEFAULT_ARC_LEWM_DATASET_MANIFEST_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_dataset_manifest.json"
)
DEFAULT_ARC_LEWM_SOURCE_AUDIT_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_source_audit.json"
)
DATASET_SCHEMA_VERSION = "m2.arc_lewm_transitions.v1"
MANIFEST_SCHEMA_VERSION = "m2.arc_lewm_dataset_manifest.v1"
SOURCE_AUDIT_SCHEMA_VERSION = "m2.arc_lewm_source_audit.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
TRANSITION_TYPES = (
    "no-op",
    "motion",
    "local",
    "global",
    "object-delta",
    "relation-delta",
    "terminal",
    "level-up",
)
TERMINAL_STATES = {"GAME_OVER", "FINISHED", "DONE", "COMPLETED", "WON", "LOST"}


@dataclass(frozen=True)
class SourceRow:
    source_path: str
    line_number: int
    data: Mapping[str, Any]


class SourceAdapter(Protocol):
    source_type: str

    def iter_rows(self) -> Iterable[SourceRow]:
        ...

    def continuity_diagnostics(self) -> Dict[str, Any]:
        ...


class HumanTraceAdapter:
    source_type = "human_traces_steps_jsonl"

    def __init__(self, traces_dir: str | Path):
        self.traces_dir = Path(traces_dir)
        self.source_paths = tuple(sorted(self.traces_dir.glob("*.steps.jsonl")))
        self._continuity: Dict[str, Any] | None = None

    def iter_rows(self) -> Iterable[SourceRow]:
        all_stats = _empty_continuity_stats()
        for path in self.source_paths:
            rows = list(_read_jsonl_rows(path))
            file_stats = _continuity_diagnostics_for_rows(rows)
            _merge_continuity_stats(all_stats, file_stats)
            for line_number, row in enumerate(rows, start=1):
                yield SourceRow(
                    source_path=str(path),
                    line_number=line_number,
                    data=row,
                )
        self._continuity = all_stats

    def continuity_diagnostics(self) -> Dict[str, Any]:
        return dict(self._continuity or _empty_continuity_stats())


def run_arc_lewm_dataset_builder(
    *,
    traces_dir: str | Path = "human_traces",
    split: str = "by_game",
) -> Dict[str, Any]:
    if split != "by_game":
        raise ValueError("M2.14 foundation supports only --split by_game")
    adapter = HumanTraceAdapter(traces_dir)
    transitions: list[Dict[str, Any]] = []
    rejected_rows: list[Dict[str, Any]] = []
    for source_row in adapter.iter_rows():
        transition = transition_from_human_trace_row(source_row)
        if transition is None:
            rejected_rows.append(
                {
                    "source_path": source_row.source_path,
                    "line_number": source_row.line_number,
                    "reason": "missing_grid_t_or_grid_t1",
                    "support": 0,
                    "truth_status": M2_TRUTH_STATUS,
                }
            )
            continue
        transitions.append(transition)
    return build_dataset_payload(
        transitions=transitions,
        rejected_rows=rejected_rows,
        adapter=adapter,
        split=split,
    )


def transition_from_human_trace_row(source_row: SourceRow) -> Dict[str, Any] | None:
    row = source_row.data
    grid_t = _normalize_grid(row.get("frame_before"))
    grid_t1 = _normalize_grid(row.get("frame_after"))
    if grid_t is None or grid_t1 is None:
        return None
    action = normalize_action(row.get("action"))
    action_args = normalize_action_args(row.get("action_args"))
    available_actions = [
        normalize_action(value) for value in row.get("available_actions", []) or []
    ]
    terminal_t1 = _is_terminal(row)
    level_delta = _level_delta(row)
    transition_tags = transition_type_tags(grid_t, grid_t1, terminal_t1, level_delta)
    return {
        "game_id": str(row.get("game_id", "")),
        "episode_id": str(row.get("episode_id", "")),
        "step": int(row.get("step", 0) or 0),
        "grid_t": grid_t,
        "grid_t1": grid_t1,
        "action": action,
        "action_args": action_args,
        "available_actions_t": available_actions,
        "terminal_t1": terminal_t1,
        "level_delta": level_delta,
        "transition_type": transition_tags[0],
        "transition_tags": transition_tags,
        "hud": {
            "available": False,
            "source": "not_extracted_in_m2_14_foundation",
        },
        "source": {
            "source_type": HumanTraceAdapter.source_type,
            "source_path": source_row.source_path,
            "line_number": source_row.line_number,
        },
        "candidate_only_metadata": candidate_only_metadata(),
    }


def build_dataset_payload(
    *,
    transitions: Sequence[Mapping[str, Any]],
    rejected_rows: Sequence[Mapping[str, Any]],
    adapter: HumanTraceAdapter,
    split: str,
) -> Dict[str, Any]:
    games = sorted({str(row.get("game_id", "")) for row in transitions if row.get("game_id")})
    split_info = by_game_split(games)
    per_game_counts = Counter(str(row.get("game_id", "")) for row in transitions)
    histogram = {key: 0 for key in TRANSITION_TYPES}
    for row in transitions:
        for tag in row.get("transition_tags", []) or []:
            if tag in histogram:
                histogram[tag] += 1
    continuity = adapter.continuity_diagnostics()
    alignment_policy = {
        "transition_alignment_policy": "row_local_frame_before_action_frame_after",
        "continuity_checks_enabled": True,
        "off_by_one_pairing_used": False,
        "continuity_mismatches": int(continuity.get("continuity_mismatches", 0)),
        "continuity_checked_pairs": int(continuity.get("continuity_checked_pairs", 0)),
        "continuity_skipped_pairs": int(continuity.get("continuity_skipped_pairs", 0)),
    }
    manifest = {
        "config": {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "source_adapter": adapter.source_type,
            "source_paths": [str(path) for path in adapter.source_paths],
            "split_mode": split,
        },
        "summary": {
            "transitions_written": len(transitions),
            "rows_rejected_missing_grid_t1": len(rejected_rows),
            "games_total": len(games),
            "split_mode": split,
            "train_games": len(split_info["train_games"]),
            "val_games": len(split_info["val_games"]),
            "support": 0,
            "truth_status": M2_TRUTH_STATUS,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "trace_support_counted_as_proof": False,
            "world_model_prediction_counted_as_evidence": False,
            "world_model_score_counted_as_support": False,
        },
        "per_game_counts": dict(sorted(per_game_counts.items())),
        "transition_type_histogram": histogram,
        "alignment_policy": alignment_policy,
        "continuity_checks": continuity,
        "split": {
            "mode": split,
            "train_games": split_info["train_games"],
            "val_games": split_info["val_games"],
        },
        "rejected_rows": list(rejected_rows[:20]),
        "candidate_only_metadata": candidate_only_metadata(),
    }
    return {
        "transitions": [dict(row) for row in transitions],
        "manifest": manifest,
        "source_audit": build_source_audit(),
    }


def write_arc_lewm_dataset_artifacts(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path = DEFAULT_ARC_LEWM_TRANSITIONS_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_ARC_LEWM_DATASET_MANIFEST_OUTPUT_PATH,
    source_audit_path: str | Path = DEFAULT_ARC_LEWM_SOURCE_AUDIT_OUTPUT_PATH,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row, sort_keys=True)
        for row in payload.get("transitions", []) or []
    ]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = Path(source_audit_path)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps(payload["source_audit"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_source_audit() -> Dict[str, Any]:
    return {
        "config": {"schema_version": SOURCE_AUDIT_SCHEMA_VERSION},
        "accepted_source_types": ["human_traces_steps_jsonl"],
        "rejected_source_types": [
            "m1_observation_dataset_jsonl",
            "m3_experiment_logs",
            "p1_policy_logs",
            "p3_policy_logs",
        ],
        "rejection_reason": "raw_grid_t_grid_t1_not_verified_for_foundation_pass",
        "silent_ingestion_performed": False,
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def candidate_only_metadata() -> Dict[str, Any]:
    return {
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "trace_support_counted_as_proof": False,
        "world_model_prediction_counted_as_evidence": False,
        "world_model_score_counted_as_support": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def normalize_action(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"ACTION{value}"
    text = str(value).strip()
    if not text:
        return ""
    if text.upper() == "RESET":
        return "RESET"
    upper = text.upper()
    if upper.startswith("ACTION"):
        return upper
    if text.isdigit():
        return f"ACTION{text}"
    return upper


def normalize_action_args(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    args: Dict[str, Any] = {}
    for key, item in value.items():
        if key in {"x", "y"}:
            args[str(key)] = _optional_int(item, default=0)
        else:
            args[str(key)] = item
    if "x" in args or "y" in args:
        args.setdefault("x", 0)
        args.setdefault("y", 0)
    return args


def by_game_split(games: Sequence[str]) -> Dict[str, list[str]]:
    ordered = sorted(dict.fromkeys(str(game) for game in games if str(game)))
    if len(ordered) <= 1:
        return {"train_games": ordered, "val_games": []}
    val_count = 2 if len(ordered) >= 6 else 1
    return {
        "train_games": ordered[:-val_count],
        "val_games": ordered[-val_count:],
    }


def transition_type_tags(
    grid_t: Sequence[Sequence[int]],
    grid_t1: Sequence[Sequence[int]],
    terminal_t1: bool,
    level_delta: int,
) -> list[str]:
    tags: list[str] = []
    changed = _changed_pixels(grid_t, grid_t1)
    same_shape = _grid_shape(grid_t) == _grid_shape(grid_t1)
    if changed == 0 and same_shape:
        tags.append("no-op")
    elif same_shape and changed <= 4:
        tags.append("motion")
    elif same_shape and changed <= 64:
        tags.append("local")
    else:
        tags.append("global")
    if _color_counts(grid_t) != _color_counts(grid_t1):
        tags.append("object-delta")
    if _adjacent_contact_count(grid_t) != _adjacent_contact_count(grid_t1):
        tags.append("relation-delta")
    if terminal_t1:
        tags.append("terminal")
    if level_delta > 0:
        tags.append("level-up")
    return tags


def _read_jsonl_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, Mapping):
                yield data


def _normalize_grid(value: Any) -> list[list[int]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return None
        rows.append([int(cell) for cell in row])
    if not rows:
        return None
    return rows


def _is_terminal(row: Mapping[str, Any]) -> bool:
    if bool(row.get("terminal_t1", False)) or bool(row.get("terminal", False)):
        return True
    if bool(row.get("done", False)) or bool(row.get("truncated", False)):
        return True
    state = str(row.get("game_state_after", "")).upper()
    return state in TERMINAL_STATES


def _level_delta(row: Mapping[str, Any]) -> int:
    if row.get("level_delta") is not None:
        return _optional_int(row.get("level_delta"), default=0)
    before = row.get("levels_completed_before")
    after = row.get("levels_completed_after")
    if before is not None and after is not None:
        return _optional_int(after, default=0) - _optional_int(before, default=0)
    return 0


def _optional_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _grid_shape(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    return len(grid), max((len(row) for row in grid), default=0)


def _changed_pixels(
    grid_t: Sequence[Sequence[int]],
    grid_t1: Sequence[Sequence[int]],
) -> int:
    height = max(len(grid_t), len(grid_t1))
    width = max(
        max((len(row) for row in grid_t), default=0),
        max((len(row) for row in grid_t1), default=0),
    )
    changed = 0
    for y in range(height):
        for x in range(width):
            before = grid_t[y][x] if y < len(grid_t) and x < len(grid_t[y]) else None
            after = grid_t1[y][x] if y < len(grid_t1) and x < len(grid_t1[y]) else None
            if before != after:
                changed += 1
    return changed


def _color_counts(grid: Sequence[Sequence[int]]) -> Dict[int, int]:
    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(int(cell) for cell in row)
    return dict(counts)


def _adjacent_contact_count(grid: Sequence[Sequence[int]]) -> int:
    count = 0
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if int(value) == 0:
                continue
            if x + 1 < len(row) and int(row[x + 1]) != 0 and row[x + 1] != value:
                count += 1
            if y + 1 < len(grid) and x < len(grid[y + 1]):
                below = grid[y + 1][x]
                if int(below) != 0 and below != value:
                    count += 1
    return count


def _empty_continuity_stats() -> Dict[str, Any]:
    return {
        "continuity_checks_enabled": True,
        "continuity_checked_pairs": 0,
        "continuity_skipped_pairs": 0,
        "continuity_mismatches": 0,
        "skipped_reset_boundaries": 0,
        "skipped_episode_boundaries": 0,
        "skipped_terminal_or_truncated_boundaries": 0,
        "mismatch_examples": [],
    }


def _continuity_diagnostics_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stats = _empty_continuity_stats()
    for previous, current in zip(rows, rows[1:]):
        skip_reason = _continuity_skip_reason(previous, current)
        if skip_reason:
            stats["continuity_skipped_pairs"] += 1
            if skip_reason == "reset":
                stats["skipped_reset_boundaries"] += 1
            elif skip_reason == "episode":
                stats["skipped_episode_boundaries"] += 1
            else:
                stats["skipped_terminal_or_truncated_boundaries"] += 1
            continue
        stats["continuity_checked_pairs"] += 1
        if previous.get("frame_after") != current.get("frame_before"):
            stats["continuity_mismatches"] += 1
            if len(stats["mismatch_examples"]) < 5:
                stats["mismatch_examples"].append(
                    {
                        "game_id": str(previous.get("game_id", "")),
                        "episode_id": str(previous.get("episode_id", "")),
                        "previous_step": previous.get("step"),
                        "current_step": current.get("step"),
                    }
                )
    return stats


def _continuity_skip_reason(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> str:
    if previous.get("episode_id") != current.get("episode_id"):
        return "episode"
    if normalize_action(previous.get("action")) == "RESET" or normalize_action(current.get("action")) == "RESET":
        return "reset"
    if _is_terminal(previous) or bool(previous.get("truncated", False)):
        return "terminal"
    return ""


def _merge_continuity_stats(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key == "mismatch_examples":
            target[key].extend(value)
            target[key] = target[key][:5]
        elif isinstance(value, bool):
            target[key] = bool(target.get(key, False) or value)
        elif isinstance(value, int):
            target[key] += value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reward-free M2.14b ARC-LeWM transitions from human traces.",
    )
    parser.add_argument("--traces", type=Path, default=Path("human_traces"))
    parser.add_argument("--out", type=Path, default=DEFAULT_ARC_LEWM_TRANSITIONS_OUTPUT_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ARC_LEWM_DATASET_MANIFEST_OUTPUT_PATH,
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=DEFAULT_ARC_LEWM_SOURCE_AUDIT_OUTPUT_PATH,
    )
    parser.add_argument("--split", choices=("by_game",), default="by_game")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_arc_lewm_dataset_builder(traces_dir=args.traces, split=args.split)
    write_arc_lewm_dataset_artifacts(
        payload,
        output_path=args.out,
        manifest_path=args.manifest,
        source_audit_path=args.source_audit,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "manifest_path": str(args.manifest),
                "source_audit_path": str(args.source_audit),
                "summary": payload["manifest"]["summary"],
                "alignment_policy": payload["manifest"]["alignment_policy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
