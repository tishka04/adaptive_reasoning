"""Live TransitionRecord belief loop.

This is the A2 bridge from the consolidated plan: take real before/action/after
observations, build v3 ``TransitionRecord`` objects, update ``GameTheory`` from
the observed effect, and run v3 relational rule verification. It intentionally
does not plan or optimize game score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from v3.mechanics.action_profiler import ActionProfiler
from v3.mechanics.rule_engine import RuleEngine
from v3.perception.affordance_mapper import build_local_contexts, map_affordances
from v3.perception.frame_diff import compute_frame_diff
from v3.perception.object_extractor import (
    extract_objects,
    generate_player_hypotheses,
)
from v3.schemas import (
    FrameDiff,
    GameObservation,
    PlayerHypothesis,
    PrimitiveAction,
    Rule,
    TransitionRecord,
)

from .correspondence_hypothesis import (
    CorrespondenceObservation,
    load_task_program_correspondence_hypotheses,
    normalize_pair_colors,
)
from .epistemic_metrics import MechanicsOracle, score_beliefs
from .mechanic_hypothesis import GameTheory, ObservedEffect
from .revision import verify_relational_rules
from .role_hypotheses import load_task_program_semantic_hypotheses


@dataclass
class LiveTransitionUpdate:
    """What changed after one live transition was ingested."""

    record: TransitionRecord
    effect: ObservedEffect
    rules: List[Rule] = field(default_factory=list)
    high_confidence_rules: List[Rule] = field(default_factory=list)

    @property
    def action(self) -> str:
        return self.record.action.name


def build_observation(
    grid: Any,
    *,
    available_actions: Sequence[Any] | None = None,
    game_state: str = "NOT_FINISHED",
    levels_completed: int = 0,
    background_value: int | None = None,
    infer_players: bool = True,
    prev_player_hypotheses: Sequence[PlayerHypothesis] | None = None,
    danger_map: np.ndarray | None = None,
    known_lethal_values: set[int] | None = None,
    known_collectible_values: set[int] | None = None,
) -> GameObservation:
    """Convert a raw grid into a v3 structured observation."""
    raw_grid = _as_grid(grid)
    bg = _infer_background_value(raw_grid) if background_value is None else background_value
    objects = extract_objects(raw_grid, background_value=int(bg))
    previous = list(prev_player_hypotheses or [])
    player_hypotheses = (
        generate_player_hypotheses(raw_grid, objects, prev_hypotheses=previous)
        if infer_players
        else []
    )
    player_pos = player_hypotheses[0].position if player_hypotheses else None
    affordances = map_affordances(
        objects,
        raw_grid,
        player_pos=player_pos,
        danger_map=danger_map,
        known_lethal_values=known_lethal_values,
        known_collectible_values=known_collectible_values,
    )
    local_contexts = (
        build_local_contexts(raw_grid, [player_pos], danger_map=danger_map)
        if player_pos is not None
        else []
    )
    return GameObservation(
        raw_grid=raw_grid,
        grid_hash=hash(raw_grid.tobytes()),
        game_state=str(game_state),
        levels_completed=int(levels_completed),
        available_actions=_normalize_actions(available_actions or []),
        objects=objects,
        player_candidates=player_hypotheses,
        affordances=affordances,
        danger_map=danger_map,
        local_contexts=local_contexts,
    )


def build_transition_record(
    *,
    action: Any,
    grid_before: Any,
    grid_after: Any,
    available_actions: Sequence[Any] | None = None,
    game_state_before: str = "NOT_FINISHED",
    game_state_after: str = "NOT_FINISHED",
    levels_completed_before: int = 0,
    levels_completed_after: int = 0,
    action_args: dict[str, Any] | None = None,
    timestamp: int = 0,
    background_value: int | None = None,
    infer_players: bool = True,
) -> TransitionRecord:
    """Build a v3 TransitionRecord from raw before/after frames."""
    before = build_observation(
        grid_before,
        available_actions=available_actions,
        game_state=game_state_before,
        levels_completed=levels_completed_before,
        background_value=background_value,
        infer_players=infer_players,
    )
    after = build_observation(
        grid_after,
        available_actions=available_actions,
        game_state=game_state_after,
        levels_completed=levels_completed_after,
        background_value=background_value,
        infer_players=infer_players,
        prev_player_hypotheses=before.player_candidates,
    )
    diff = compute_frame_diff(
        before.raw_grid,
        after.raw_grid,
        before.objects,
        after.objects,
        before.best_player.position if before.best_player else None,
        after.best_player.position if after.best_player else None,
        game_state_after,
        levels_completed_before,
        levels_completed_after,
    )
    after.frame_diff = diff
    return TransitionRecord(
        action=_primitive_action(action, action_args),
        obs_before=before,
        obs_after=after,
        diff=diff,
        timestamp=int(timestamp),
    )


class LiveTransitionBeliefLoop:
    """Belief loop backed by real TransitionRecord observations."""

    def __init__(
        self,
        game_id: str = "",
        *,
        available_actions: Sequence[Any] | None = None,
        theory: GameTheory | None = None,
        profiler: ActionProfiler | None = None,
        rule_engine: RuleEngine | None = None,
        background_value: int | None = None,
        infer_players: bool = True,
        verify_every: int = 1,
        correspondence_pair_colors: tuple[int, int] | None = None,
    ) -> None:
        self.theory = theory or GameTheory(game_id)
        self.profiler = profiler or ActionProfiler()
        self.rule_engine = rule_engine or RuleEngine()
        self.background_value = background_value
        self.infer_players = infer_players
        self.verify_every = max(1, int(verify_every))
        self.correspondence_pair_colors = (
            normalize_pair_colors(correspondence_pair_colors)
            if correspondence_pair_colors is not None
            else None
        )
        self.transition_count = 0
        self.last_rules: List[Rule] = []
        actions = _normalize_actions(available_actions or [])
        if actions:
            self.theory.seed_actions(actions)

    def seed_task_program(self, path: Path) -> None:
        """Seed human-facing role/goal hypotheses from a task program."""
        roles, families = load_task_program_semantic_hypotheses(path)
        correspondence = (
            load_task_program_correspondence_hypotheses(
                path,
                pair_colors=self.correspondence_pair_colors or (10, 11),
            )
            if self.correspondence_pair_colors is not None
            else []
        )
        self.theory.add_semantic_hypotheses(roles, families, correspondence)

    def observe_record(
        self,
        record: TransitionRecord,
        *,
        was_experiment: bool = True,
    ) -> LiveTransitionUpdate:
        """Ingest one real v3 transition and revise mechanic/rule beliefs."""
        self.transition_count += 1
        self.profiler.record_transition(record)
        effect = ObservedEffect.from_frame_diff(record.diff)
        self.theory.observe(
            record.action.name,
            effect,
            was_experiment=was_experiment,
        )
        if self.correspondence_pair_colors is not None:
            relation_signal = CorrespondenceObservation.from_transition(
                record,
                pair_colors=self.correspondence_pair_colors,
            )
            self.theory.observe_correspondence(
                relation_signal,
                was_experiment=was_experiment,
            )

        if self.transition_count % self.verify_every == 0:
            self.last_rules = verify_relational_rules(
                self.rule_engine,
                self.profiler,
                self.profiler.transitions,
            )

        return LiveTransitionUpdate(
            record=record,
            effect=effect,
            rules=list(self.last_rules),
            high_confidence_rules=self.rule_engine.high_confidence_rules(),
        )

    def observe_grids(
        self,
        *,
        action: Any,
        grid_before: Any,
        grid_after: Any,
        available_actions: Sequence[Any] | None = None,
        game_state_before: str = "NOT_FINISHED",
        game_state_after: str = "NOT_FINISHED",
        levels_completed_before: int = 0,
        levels_completed_after: int = 0,
        action_args: dict[str, Any] | None = None,
        timestamp: int | None = None,
        was_experiment: bool = True,
    ) -> LiveTransitionUpdate:
        """Build and ingest a TransitionRecord from raw grids."""
        record = build_transition_record(
            action=action,
            action_args=action_args,
            grid_before=grid_before,
            grid_after=grid_after,
            available_actions=available_actions or self.theory.actions(),
            game_state_before=game_state_before,
            game_state_after=game_state_after,
            levels_completed_before=levels_completed_before,
            levels_completed_after=levels_completed_after,
            timestamp=self.transition_count if timestamp is None else timestamp,
            background_value=self.background_value,
            infer_players=self.infer_players,
        )
        return self.observe_record(record, was_experiment=was_experiment)

    def score(self, oracle: MechanicsOracle, *, experiment_actions: int | None = None):
        """Score the current belief ledger against an epistemic oracle."""
        return score_beliefs(
            self.theory.to_ledger(),
            oracle,
            experiment_actions=experiment_actions or self.transition_count,
        )


def run_trace_file(
    trace_path: Path,
    *,
    game_id: str = "",
    task_program_path: Path | None = None,
    max_steps: int | None = None,
    background_value: int | None = None,
    infer_players: bool = True,
    verify_every: int = 1,
    correspondence_pair_colors: tuple[int, int] | None = None,
) -> LiveTransitionBeliefLoop:
    """Replay a JSONL trace as live before/action/after transitions."""
    loop = LiveTransitionBeliefLoop(
        game_id=game_id,
        background_value=background_value,
        infer_players=infer_players,
        verify_every=verify_every,
        correspondence_pair_colors=correspondence_pair_colors,
    )
    if task_program_path is not None:
        loop.seed_task_program(task_program_path)

    steps_seen = 0
    previous_levels = 0
    with open(Path(trace_path), "r", encoding="utf-8") as handle:
        for line in handle:
            if max_steps is not None and steps_seen >= max_steps:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            action = item.get("action")
            if str(action).upper() == "RESET":
                continue
            before = item.get("frame_before")
            after = item.get("frame_after")
            if before is None or after is None:
                continue
            available = item.get("available_actions") or []
            loop.theory.seed_actions(_normalize_actions(available))
            loop.observe_grids(
                action=action,
                action_args=item.get("action_args"),
                grid_before=before,
                grid_after=after,
                available_actions=available,
                game_state_after=item.get("game_state_after", "NOT_FINISHED"),
                levels_completed_before=previous_levels,
                levels_completed_after=int(item.get("levels_completed_after", 0) or 0),
                timestamp=int(item.get("step", steps_seen) or 0),
                was_experiment=True,
            )
            previous_levels = int(item.get("levels_completed_after", previous_levels) or 0)
            steps_seen += 1
    return loop


def _as_grid(grid: Any) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D grid, got shape {arr.shape}")
    return arr


def _infer_background_value(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize_actions(actions: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for action in actions:
        name = _normalize_action_name(action)
        if name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_action_name(action: Any) -> str:
    if isinstance(action, (int, np.integer)):
        return f"ACTION{int(action)}"
    raw = str(action or "").strip().upper()
    if raw.isdigit():
        return f"ACTION{raw}"
    return raw


def _primitive_action(action: Any, action_args: dict[str, Any] | None = None) -> PrimitiveAction:
    args = dict(action_args or {})
    return PrimitiveAction(
        name=_normalize_action_name(action),
        x=_optional_int(args.get("x")),
        y=_optional_int(args.get("y")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
