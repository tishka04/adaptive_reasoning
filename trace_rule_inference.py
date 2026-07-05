"""Infer a global task program from pre-frontier human trace segments.

This diagnostic intentionally changes the question from local repair
("how do we pass level 7?") to rule inference ("what did the human learn
before reaching level 7?"). It segments successful level transitions in
the human traces, extracts cross-level invariants, and emits a deterministic
TaskProgram draft without overwriting the existing hand/LLM program.

Example:
    python trace_rule_inference.py --game ar25 --max-level 6
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from human_trace.loader import load_traces
from human_trace.schema import EpisodeRecord, StepRecord
from human_trace.task_program import (
    ActionRole,
    Constraint,
    Entity,
    HypothesisRevision,
    SubgoalTest,
    TaskProgram,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACES_DIR = PROJECT_ROOT / "human_traces"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "rule_inference"
DEFAULT_PROGRAM_DIR = PROJECT_ROOT / "task_programs"


@dataclass
class EpisodeBundle:
    """Trace episode plus its step records."""

    episode: Optional[EpisodeRecord]
    steps: List[StepRecord]

    @property
    def episode_id(self) -> str:
        if self.episode is not None:
            return self.episode.episode_id
        if self.steps:
            return self.steps[0].episode_id
        return "unknown"

    @property
    def levels_completed(self) -> int:
        if self.episode is not None:
            return int(self.episode.levels_completed)
        return max((int(step.levels_completed_after) for step in self.steps), default=0)

    @property
    def final_state(self) -> str:
        if self.episode is not None:
            return self.episode.final_state
        return self.steps[-1].game_state_after if self.steps else "UNKNOWN"


@dataclass
class LevelSegment:
    """One successful RESET-to-level-up prefix inside an episode."""

    episode_id: str
    level_number: int
    trace_start_step: int
    trace_end_step: int
    actions: List[str]
    intents: List[str]
    state_after: str
    start_grid: List[List[int]]
    end_grid: List[List[int]]
    level_up_before_grid: List[List[int]]
    level_up_after_grid: List[List[int]]

    @property
    def level_up_action(self) -> str:
        return self.actions[-1] if self.actions else "NONE"

    def to_report(self) -> Dict[str, Any]:
        whole_change = change_report(self.start_grid, self.end_grid)
        level_up_change = change_report(self.level_up_before_grid, self.level_up_after_grid)
        return {
            "episode_id": self.episode_id,
            "level_number": self.level_number,
            "trace_start_step": self.trace_start_step,
            "trace_end_step": self.trace_end_step,
            "n_actions": len(self.actions),
            "action_counts": dict(Counter(self.actions)),
            "intent_counts": dict(Counter(self.intents)),
            "last_actions": self.actions[-8:],
            "level_up_action": self.level_up_action,
            "state_after": self.state_after,
            "whole_segment_change": whole_change,
            "level_up_transition_change": level_up_change,
        }


def _grid_hash(grid: Sequence[Sequence[int]]) -> str:
    arr = np.array(grid, dtype=np.int32)
    return __import__("hashlib").sha1(arr.tobytes()).hexdigest()[:16]


def _changed_cells(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> int:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return int(left.size + right.size)
    return int(np.count_nonzero(left != right))


def _histogram_delta(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> int:
    left = np.array(before, dtype=np.int32).ravel()
    right = np.array(after, dtype=np.int32).ravel()
    values = set(int(v) for v in left) | set(int(v) for v in right)
    return int(
        sum(
            abs(int(np.count_nonzero(left == value)) - int(np.count_nonzero(right == value)))
            for value in values
        )
    )


def _diff_bbox(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> Optional[Dict[str, int]]:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return None
    coords = np.argwhere(left != right)
    if coords.size == 0:
        return None
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    return {
        "min_y": int(min_y),
        "min_x": int(min_x),
        "max_y": int(max_y),
        "max_x": int(max_x),
        "height": int(max_y - min_y + 1),
        "width": int(max_x - min_x + 1),
        "changed_cells": int(coords.shape[0]),
    }


def _changed_value_report(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    *,
    limit: int = 16,
) -> Dict[str, Any]:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return {"shape_mismatch": True}
    mask = left != right
    transitions: Dict[str, int] = {}
    for old, new in zip(left[mask].tolist(), right[mask].tolist()):
        key = f"{int(old)}->{int(new)}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "before_values": sorted(int(v) for v in np.unique(left[mask])),
        "after_values": sorted(int(v) for v in np.unique(right[mask])),
        "transitions": dict(
            sorted(transitions.items(), key=lambda item: item[1], reverse=True)[:limit]
        ),
    }


def _connected_components(grid: Sequence[Sequence[int]], *, limit: int = 256) -> List[Tuple[int, int, int, int, int, int]]:
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2:
        return []
    height, width = arr.shape
    seen = np.zeros(arr.shape, dtype=bool)
    out: List[Tuple[int, int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            color = int(arr[y, x])
            if color == 0 or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if seen[ny, nx] or int(arr[ny, nx]) != color:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            out.append((color, len(cells), min(ys), min(xs), max(ys), max(xs)))
    out.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return out[:limit]


def _component_delta(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> int:
    return len(set(_connected_components(before)) ^ set(_connected_components(after)))


def change_report(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Compact structural diff between two primary grids."""

    arr = np.array(before, dtype=np.int32)
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])] if arr.ndim == 2 else list(arr.shape),
        "before_hash": _grid_hash(before),
        "after_hash": _grid_hash(after),
        "changed_cells": _changed_cells(before, after),
        "histogram_delta": _histogram_delta(before, after),
        "component_delta": _component_delta(before, after),
        "bounding_box_of_diff": _diff_bbox(before, after),
        "changed_values": _changed_value_report(before, after),
    }


def _non_reset_steps(steps: Sequence[StepRecord]) -> List[StepRecord]:
    return [step for step in steps if step.action != "RESET"]


def build_level_segments(
    bundle: EpisodeBundle,
    *,
    max_level: int,
) -> List[LevelSegment]:
    """Return successful level segments up to the requested completed level."""

    play_steps = _non_reset_steps(bundle.steps)
    if not play_steps:
        return []

    segments: List[LevelSegment] = []
    current_completed = 0
    start_index = 0

    for index, step in enumerate(play_steps):
        next_completed = int(step.levels_completed_after)
        if next_completed <= current_completed:
            continue
        if next_completed > current_completed + 1:
            # A jump means this trace starts mid-run or is truncated. It
            # cannot prove a RESET-to-level-up causal prefix for skipped levels.
            current_completed = next_completed
            start_index = index + 1
            if current_completed >= max_level:
                break
            continue

        segment_steps = play_steps[start_index : index + 1]
        if not segment_steps:
            current_completed = next_completed
            start_index = index + 1
            continue

        segments.append(
            LevelSegment(
                episode_id=bundle.episode_id,
                level_number=current_completed + 1,
                trace_start_step=int(segment_steps[0].step),
                trace_end_step=int(segment_steps[-1].step),
                actions=[item.action for item in segment_steps],
                intents=[item.intent for item in segment_steps],
                state_after=segment_steps[-1].game_state_after,
                start_grid=segment_steps[0].frame_before,
                end_grid=segment_steps[-1].frame_after,
                level_up_before_grid=segment_steps[-1].frame_before,
                level_up_after_grid=segment_steps[-1].frame_after,
            )
        )

        current_completed = next_completed
        start_index = index + 1
        if current_completed >= max_level:
            break

    return [segment for segment in segments if segment.level_number <= max_level]


def _covers_level_prefix(bundle: EpisodeBundle, *, max_level: int) -> bool:
    observed = {segment.level_number for segment in build_level_segments(bundle, max_level=max_level)}
    return set(range(1, max_level + 1)).issubset(observed)


def _resolve_game_entries(traces_dir: Path, requested_game: str) -> Tuple[str, List[EpisodeBundle]]:
    corpus = load_traces(traces_dir)
    if requested_game in corpus.by_game:
        full_game_id = requested_game
        entries = corpus.by_game[requested_game]
    else:
        full_game_id = ""
        entries = []
        for game_id, game_entries in sorted(corpus.by_game.items()):
            if game_id == requested_game or game_id.startswith(requested_game + "-"):
                full_game_id = game_id
                entries = game_entries
                break
        if not entries:
            for game_id, game_entries in sorted(corpus.by_game.items()):
                if game_id.startswith(requested_game):
                    full_game_id = game_id
                    entries = game_entries
                    break
    if not entries:
        raise ValueError(f"No human traces matched game {requested_game!r} under {traces_dir}")

    bundles = [
        EpisodeBundle(
            episode=bucket.get("episode"),  # type: ignore[arg-type]
            steps=list(bucket.get("steps") or []),  # type: ignore[arg-type]
        )
        for bucket in entries
    ]
    return full_game_id, bundles


def _select_bundles(
    bundles: Sequence[EpisodeBundle],
    *,
    max_level: int,
    episode_id: Optional[str],
) -> List[EpisodeBundle]:
    if episode_id:
        selected = [bundle for bundle in bundles if bundle.episode_id == episode_id]
        if not selected:
            raise ValueError(f"No episode matched {episode_id!r}")
        return selected

    strong = [
        bundle
        for bundle in bundles
        if bundle.levels_completed >= max_level and _covers_level_prefix(bundle, max_level=max_level)
    ]
    if strong:
        return strong

    scored = [
        (
            len({segment.level_number for segment in build_level_segments(bundle, max_level=max_level)}),
            bundle.levels_completed,
            len(bundle.steps),
            bundle,
        )
        for bundle in bundles
    ]
    if not scored:
        return []
    best_score = max(score[:3] for score in scored)
    return [bundle for *score, bundle in scored if tuple(score) == best_score]


def _episode_evidence(bundles: Sequence[EpisodeBundle]) -> Dict[str, Any]:
    objectives = Counter()
    game_types = Counter()
    mechanics = Counter()
    mistakes = Counter()
    for bundle in bundles:
        ep = bundle.episode
        if ep is None:
            continue
        if ep.objective_guess:
            objectives[ep.objective_guess] += 1
        if ep.game_type_guess:
            game_types[ep.game_type_guess] += 1
        mechanics.update(ep.discovered_mechanics)
        mistakes.update(ep.discovered_mistakes)
    return {
        "objective_guesses": dict(objectives.most_common()),
        "game_type_guesses": dict(game_types.most_common()),
        "discovered_mechanics": dict(mechanics.most_common()),
        "discovered_mistakes": dict(mistakes.most_common()),
    }


def _load_level7_diagnostic(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("action3_precondition_matrix") or {}
    center = matrix.get("center") or payload.get("what_changed_after_ACTION6_center") or {}
    return {
        "path": str(path),
        "action3_safe_distinct_bbox_count": int(matrix.get("ACTION3_safe_distinct_bbox_count", 0)),
        "fatal_distinct_bbox_count": len(matrix.get("fatal_distinct_bbox") or []),
        "safe_no_distinct_bbox_count": len(matrix.get("safe_no_distinct_bbox") or []),
        "candidate_count": int(matrix.get("candidate_count", 0) or 0),
        "center": center,
    }


def infer_cross_level_invariants(
    segments: Sequence[LevelSegment],
    evidence: Dict[str, Any],
    level7_diagnostic: Dict[str, Any],
) -> Dict[str, Any]:
    action_counts = Counter()
    level_up_actions = Counter()
    last3_sequences = Counter()
    intent_counts = Counter()
    transition_counts = Counter()
    segment_lengths: List[int] = []
    changed_cells: List[int] = []
    component_deltas: List[int] = []

    for segment in segments:
        report = segment.to_report()
        action_counts.update(segment.actions)
        level_up_actions[segment.level_up_action] += 1
        last3_sequences[tuple(segment.actions[-3:])] += 1
        intent_counts.update(segment.intents)
        segment_lengths.append(len(segment.actions))
        changed_cells.append(int(report["whole_segment_change"]["changed_cells"]))
        component_deltas.append(int(report["whole_segment_change"]["component_delta"]))
        transition_counts.update(report["whole_segment_change"]["changed_values"]["transitions"])

    mechanics = evidence.get("discovered_mechanics") or {}
    mistakes = evidence.get("discovered_mistakes") or {}
    objective_guesses = evidence.get("objective_guesses") or {}
    hypotheses: List[str] = []
    if any("match" in item or "shape" in item for item in objective_guesses):
        hypotheses.append("task is a correspondence puzzle over colored shapes")
    if any("gray_shape" in item and "control" in item for item in mechanics):
        hypotheses.append("gray shape acts as a controller for other shapes")
    if any("dotes" in item or "doted" in item for item in mechanics):
        hypotheses.append("dotted lines behave like cursor/mirror constraints")
    if any("go_out" in item or "exit_grid" in item for item in mechanics):
        hypotheses.append("some shapes may legally leave the grid")
    if any("all_shapes" in item for item in mistakes):
        hypotheses.append("only a subset of shape matches is required")
    if level7_diagnostic.get("action3_safe_distinct_bbox_count") == 0 and level7_diagnostic:
        hypotheses.append("ACTION3 distinct geometry at level 7 is a contradiction signal")

    def avg(values: Sequence[int]) -> float:
        return round(float(sum(values) / max(1, len(values))), 3)

    return {
        "levels_observed": sorted({segment.level_number for segment in segments}),
        "n_success_segments": len(segments),
        "action_counts": dict(action_counts.most_common()),
        "level_up_actions": dict(level_up_actions.most_common()),
        "common_last3_sequences": [
            {"actions": list(actions), "count": count}
            for actions, count in last3_sequences.most_common(8)
        ],
        "intent_counts": dict(intent_counts.most_common()),
        "avg_actions_per_level": avg(segment_lengths),
        "avg_changed_cells_per_level": avg(changed_cells),
        "avg_component_delta_per_level": avg(component_deltas),
        "top_value_transitions": dict(transition_counts.most_common(16)),
        "rule_hypotheses": hypotheses,
        "raw_human_evidence": {
            "objective_guesses": objective_guesses,
            "discovered_mechanics": mechanics,
            "discovered_mistakes": mistakes,
        },
    }


def build_task_program(
    *,
    game_id: str,
    selected_bundles: Sequence[EpisodeBundle],
    invariants: Dict[str, Any],
    level7_diagnostic: Dict[str, Any],
) -> TaskProgram:
    """Create a deterministic TaskProgram from extracted invariants."""

    action_counts = invariants.get("action_counts") or {}
    source_episodes = [bundle.episode_id for bundle in selected_bundles][:16]
    roles: List[ActionRole] = []

    movement_actions = [name for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if action_counts.get(name, 0) > 0]
    for action in movement_actions:
        roles.append(
            ActionRole(
                action=action,
                role="movement",
                evidence="used repeatedly inside successful pre-level-7 segments",
                confidence=0.68,
            )
        )

    roles.append(
        ActionRole(
            action="ACTION5",
            role="control_switch",
            evidence="human mechanics identify gray_shape as controlling shapes",
            confidence=0.78,
            changes_semantics_of=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        )
    )
    roles.append(
        ActionRole(
            action="ACTION6",
            role="click_select",
            evidence="level7 diagnostics show coordinate clicks trigger geometric toggles",
            confidence=0.62,
        )
    )
    roles.append(
        ActionRole(
            action="ACTION7",
            role="undo",
            evidence="frontier diagnostics show safe reversible cycling after transformations",
            confidence=0.58,
        )
    )

    anti_patterns = [
        "do not assume all shapes must match",
        "do not optimize local ACTION7 cycles as progress",
        "avoid ACTION2 danger suffix at the level-7 frontier",
    ]
    if level7_diagnostic.get("action3_safe_distinct_bbox_count") == 0 and level7_diagnostic:
        anti_patterns.append("treat ACTION3 distinct bbox at level 7 as contradiction/failure")

    level_hypotheses = list(invariants.get("rule_hypotheses") or [])[:8]
    if not level_hypotheses:
        level_hypotheses = [
            "levels 1-6 are solved by applying one correspondence rule",
            "level 7 likely requires the same global rule, not wider local search",
        ]
    elif "level 7 likely requires the same global rule, not wider local search" not in level_hypotheses:
        level_hypotheses.append("level 7 likely requires the same global rule, not wider local search")

    subgoals = [
        SubgoalTest(
            id="probe_control_switch",
            description="confirm ACTION5 switches which shape movement controls",
            verification="controlled object or movement effect changes",
            max_actions=10,
            expected_signal="role_switch",
            prefer_actions=["ACTION5"],
        ),
        SubgoalTest(
            id="align_shape_correspondence",
            description="move controlled shapes until yellow/purple correspondence improves",
            verification="level advances or overlap/correspondence score improves",
            max_actions=80,
            expected_signal="level_advance",
            prefer_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
        ),
        SubgoalTest(
            id="probe_mask_toggle",
            description="click geometric bbox/center candidates only after movement setup",
            verification="persistent structured grid change without GAME_OVER",
            max_actions=20,
            preconditions=["movement alignment has stalled", "candidate target is geometrically motivated"],
            expected_signal="click_triggered_change",
            click_target_xy=(31, 31),
            prefer_actions=["ACTION6"],
        ),
        SubgoalTest(
            id="validate_without_contradiction",
            description="test ACTION3 only when current geometry is aligned enough",
            verification="ACTION3 does not create a distinct fatal bbox",
            max_actions=12,
            preconditions=["avoid known ACTION3 fatal-distinct pattern"],
            expected_signal="grid_change",
            prefer_actions=["ACTION3", "ACTION7"],
        ),
    ]

    revisions = [
        HypothesisRevision(
            was="all visible shapes must match",
            now="only a constrained subset of shape correspondences matters",
            cause="human discovered mistake: all_shapes_do_not_must_match",
        )
    ]

    confidence = min(
        0.86,
        0.56
        + 0.025 * len(invariants.get("levels_observed") or [])
        + 0.035 * len((invariants.get("raw_human_evidence") or {}).get("discovered_mechanics") or {})
        + (0.05 if level7_diagnostic else 0.0),
    )

    return TaskProgram(
        game_id=game_id,
        goal_family="correspondence",
        macro_goal=(
            "infer and apply the global colored-shape correspondence rule; "
            "use local search only to test rule-consistent transformations"
        ),
        level_hypotheses=level_hypotheses[:10],
        entities=[
            Entity(name="gray_shape", role="controller", notes="changes which shape motion controls"),
            Entity(name="yellow_shape", role="movable", notes="can sometimes leave the grid"),
            Entity(name="purple_shape", role="reference", notes="correspondence/match target"),
            Entity(name="dotted_lines", role="cursor", notes="act as cursor or mirror constraints"),
            Entity(name="action6_bbox_target", role="click_target", notes="bbox/center click can toggle masks"),
        ],
        win_condition_hypotheses=[
            "advance level when required colored-shape correspondence is satisfied",
            "structured grid changes are useful only if they preserve the global rule",
        ],
        constraints=[
            Constraint(
                type="cursor_mediated_control",
                description="dotted lines/cursors mediate which shape is affected",
                applies_to=["dotted_lines", "gray_shape"],
            ),
            Constraint(
                type="only_subset_matters",
                description="not every visible shape must be matched",
                applies_to=["yellow_shape", "purple_shape"],
            ),
            Constraint(
                type="can_exit_grid",
                description="yellow shapes can legally leave the grid",
                applies_to=["yellow_shape"],
            ),
            Constraint(
                type="click_requires_setup",
                description="ACTION6 clicks should be tested after geometry is prepared",
                applies_to=["action6_bbox_target"],
            ),
            Constraint(
                type="movement_limited",
                description="sidebar/trace evidence suggests remaining moves matter",
                applies_to=[],
            ),
        ],
        anti_patterns=anti_patterns,
        action_roles=roles[:12],
        subgoal_tests=subgoals,
        belief_revisions=revisions,
        confidence=round(confidence, 3),
        source_episodes=source_episodes,
        compiler_model="deterministic_trace_rule_inference",
        compiled_at=datetime.now(timezone.utc).isoformat(),
    )


def infer_rule_report(
    *,
    game_id: str,
    bundles: Sequence[EpisodeBundle],
    selected_bundles: Sequence[EpisodeBundle],
    max_level: int,
    level7_diagnostic: Dict[str, Any],
) -> Tuple[Dict[str, Any], TaskProgram]:
    segments: List[LevelSegment] = []
    for bundle in selected_bundles:
        segments.extend(build_level_segments(bundle, max_level=max_level))

    evidence = _episode_evidence(selected_bundles)
    invariants = infer_cross_level_invariants(segments, evidence, level7_diagnostic)
    program = build_task_program(
        game_id=game_id,
        selected_bundles=selected_bundles,
        invariants=invariants,
        level7_diagnostic=level7_diagnostic,
    )
    report = {
        "game_id": game_id,
        "objective": "infer global rule from human levels before frontier",
        "max_level_analyzed": max_level,
        "episodes_available": [
            {
                "episode_id": bundle.episode_id,
                "levels_completed": bundle.levels_completed,
                "final_state": bundle.final_state,
                "n_steps": len(bundle.steps),
            }
            for bundle in bundles
        ],
        "episodes_selected": [bundle.episode_id for bundle in selected_bundles],
        "level_segments": [segment.to_report() for segment in segments],
        "cross_level_invariants": invariants,
        "level7_frontier_context": level7_diagnostic,
        "task_program": json.loads(program.to_json()),
    }
    return report, program


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _default_level7_path(game_id: str) -> Path:
    root = PROJECT_ROOT / "diagnostics" / "level_7_frontier_recovery"
    matches = sorted(root.glob(f"{game_id}.*.level7.json"))
    if matches:
        return matches[0]
    return root / f"{game_id}.level7.json"


def _print_summary(report: Dict[str, Any], task_program_path: Optional[Path], report_path: Path) -> None:
    invariants = report["cross_level_invariants"]
    print("=" * 88)
    print("Trace rule inference")
    print("=" * 88)
    print(f"game:        {report['game_id']}")
    print(f"levels:      1..{report['max_level_analyzed']}")
    print(f"episodes:    {', '.join(report['episodes_selected'])}")
    print(f"segments:    {invariants['n_success_segments']}")
    print(f"actions:     {invariants['action_counts']}")
    print(f"level-ups:   {invariants['level_up_actions']}")
    print("hypotheses:")
    for hypothesis in invariants["rule_hypotheses"]:
        print(f"             - {hypothesis}")
    if task_program_path is not None:
        print(f"program:     {task_program_path}")
    print(f"json:        {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25", help="Game id or prefix, e.g. ar25")
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--max-level", type=int, default=6)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--program-dir", type=Path, default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--level7-diagnostic", type=Path, default=None)
    parser.add_argument(
        "--no-write-task-program",
        action="store_false",
        dest="write_task_program",
        help="Only write the diagnostic report.",
    )
    parser.set_defaults(write_task_program=True)
    args = parser.parse_args()

    full_game_id, bundles = _resolve_game_entries(args.traces, args.game)
    selected_bundles = _select_bundles(
        bundles,
        max_level=max(1, int(args.max_level)),
        episode_id=args.episode_id,
    )
    level7_path = args.level7_diagnostic or _default_level7_path(full_game_id)
    level7_diagnostic = _load_level7_diagnostic(level7_path)

    report, program = infer_rule_report(
        game_id=full_game_id,
        bundles=bundles,
        selected_bundles=selected_bundles,
        max_level=max(1, int(args.max_level)),
        level7_diagnostic=level7_diagnostic,
    )

    report_path = args.report_dir / f"{full_game_id}.global_rule_inference.json"
    _write_json(report_path, report)

    task_program_path: Optional[Path] = None
    if args.write_task_program:
        task_program_path = args.program_dir / f"{full_game_id}.global_inferred.json"
        program.save(task_program_path)

    _print_summary(report, task_program_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
