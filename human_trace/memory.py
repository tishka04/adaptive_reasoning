"""First-class human trace memory for runtime trajectory guidance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .integration import HumanPriorPack
from .schema import EpisodeRecord, StepRecord

_GOAL_KEYWORDS: Dict[str, str] = {
    "click": "click_puzzle",
    "toggle": "click_puzzle",
    "activate": "click_puzzle",
    "sequence": "sequence_puzzle",
    "order": "sequence_puzzle",
    "navigate": "navigate_exit",
    "exit": "navigate_exit",
    "reach": "navigate_exit",
    "collect": "collection",
    "gather": "collection",
    "push": "push_puzzle",
    "slot": "push_puzzle",
    "transform": "transform_puzzle",
    "apply": "transform_puzzle",
    "move": "navigate_puzzle",
    "match": "correspondence",
    "correspond": "correspondence",
}


def _canonicalise_goal(human_guess: str) -> str:
    text = (human_guess or "").lower().strip()
    if not text:
        return "unknown"
    for keyword, canon in _GOAL_KEYWORDS.items():
        if keyword in text:
            return canon
    return text if text else "unknown"


@dataclass
class TraceSchema:
    """Abstract schema distilled from a human trace."""

    goal_family: str
    abstract_steps: List[str] = field(default_factory=list)
    preferred_actions: List[str] = field(default_factory=list)
    target_objects: List[str] = field(default_factory=list)
    relevant_colors: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class TraceSegment:
    """Distilled trace fragment with a concrete state/action alignment."""

    game_id: str
    episode_id: str
    kind: str
    frames: List[List[List[int]]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    action_data: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    start_index: int = 0
    end_index: int = 0
    start_level: int = 0
    end_level: int = 0
    trace_levels_completed: int = 0
    final_state: str = "NOT_FINISHED"
    score: float = 0.0


@dataclass
class RecoveryFrontier:
    """Last known-safe state before a human trace enters a fatal suffix."""

    game_id: str
    episode_id: str
    frame: List[List[int]]
    level: int
    trace_index: int
    danger_suffix: TraceSegment
    avoid_actions: List[str] = field(default_factory=list)


@dataclass
class HumanTrace:
    """One reusable human trajectory prior."""

    game_id: str
    episode_id: str
    frames: List[List[List[int]]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    action_data: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    inferred_goal: str = ""
    goal_family: str = "unknown"
    final_state: str = ""
    levels_completed: int = 0
    state_embeddings: List[Any] = field(default_factory=list)
    action_effects: List[Dict[str, Any]] = field(default_factory=list)
    success_prefixes: List[TraceSegment] = field(default_factory=list)
    danger_suffixes: List[TraceSegment] = field(default_factory=list)
    recovery_frontiers: List[RecoveryFrontier] = field(default_factory=list)
    abstract_schema: TraceSchema = field(default_factory=lambda: TraceSchema("unknown"))
    score: float = 0.0


class HumanTraceMemory:
    """Runtime retrieval layer over human traces and their schemas."""

    def __init__(self) -> None:
        self.traces: List[HumanTrace] = []
        self.raw_episodes: List[HumanTrace] = []
        self.success_prefixes: List[TraceSegment] = []
        self.danger_suffixes: List[TraceSegment] = []
        self.recovery_frontiers: List[RecoveryFrontier] = []

    def add_trace(self, trace: HumanTrace) -> None:
        self.traces.append(trace)
        self.raw_episodes.append(trace)
        self.success_prefixes.extend(trace.success_prefixes)
        self.danger_suffixes.extend(trace.danger_suffixes)
        self.recovery_frontiers.extend(trace.recovery_frontiers)
        self.traces.sort(
            key=lambda item: (
                float(item.score),
                1 if item.final_state == "WIN" else 0,
                int(item.levels_completed),
                len(item.actions),
            ),
            reverse=True,
        )
        self.success_prefixes.sort(
            key=lambda item: (
                float(item.score),
                1 if item.final_state == "WIN" else 0,
                int(item.trace_levels_completed),
                int(item.end_level),
                -len(item.actions),
            ),
            reverse=True,
        )
        self.danger_suffixes.sort(
            key=lambda item: (
                int(item.start_level),
                float(item.score),
                len(item.actions),
            ),
            reverse=True,
        )

    def retrieve(
        self,
        observation: Any,
        goal_family: str,
        k: int = 5,
    ) -> List[HumanTrace]:
        """Retrieve the best-matching human traces for the current state."""
        query_colors = {
            str(obj.get("color", "")).strip()
            for obj in getattr(observation, "objects", []) or []
            if str(obj.get("color", "")).strip()
        }
        scored: List[tuple[float, HumanTrace]] = []
        for trace in self.traces:
            score = float(trace.score)
            if trace.goal_family == goal_family:
                score += 1.5
            elif goal_family == "unknown":
                score += 0.3
            overlap = len(query_colors & set(trace.abstract_schema.relevant_colors))
            score += 0.2 * overlap
            if getattr(observation, "player_info", None) and "identify controllable object" in trace.abstract_schema.abstract_steps:
                score += 0.15
            scored.append((score, trace))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [trace for _, trace in scored[: max(1, k)]]

    def extract_schema(self, trace: HumanTrace) -> TraceSchema:
        """Return the reusable schema already attached to a trace."""
        return trace.abstract_schema

    def aligned_trace_at(
        self,
        current_grid: Any,
        *,
        available_actions: Optional[Sequence[str]] = None,
        min_levels_completed: int = 1,
    ) -> Optional[tuple[HumanTrace, int]]:
        """Return the best raw trace/index whose pre-action frame matches exactly."""
        available = set(available_actions or [])
        for trace in self.traces:
            if trace.final_state != "WIN" and trace.levels_completed < min_levels_completed:
                continue
            for idx, action in enumerate(trace.actions):
                if available and action not in available:
                    continue
                if idx >= len(trace.frames):
                    continue
                if self._grid_equal(current_grid, trace.frames[idx]):
                    return trace, idx
        return None

    def aligned_success_prefix_at(
        self,
        current_grid: Any,
        *,
        available_actions: Optional[Sequence[str]] = None,
        current_level: Optional[int] = None,
    ) -> Optional[tuple[TraceSegment, int]]:
        """Return the best level-up segment aligned to the current state."""
        return self._aligned_segment_at(
            self.success_prefixes,
            current_grid,
            available_actions=available_actions,
            current_level=current_level,
        )

    def recovery_frontier_at(
        self,
        current_grid: Any,
    ) -> Optional[RecoveryFrontier]:
        """Return a known safe frontier before a fatal suffix, if aligned."""
        for frontier in self.recovery_frontiers:
            if self._grid_equal(current_grid, frontier.frame):
                return frontier
        return None

    def _aligned_segment_at(
        self,
        segments: Sequence[TraceSegment],
        current_grid: Any,
        *,
        available_actions: Optional[Sequence[str]] = None,
        current_level: Optional[int] = None,
    ) -> Optional[tuple[TraceSegment, int]]:
        available = set(available_actions or [])
        for segment in segments:
            if current_level is not None and int(segment.start_level) != int(current_level):
                continue
            for idx, action in enumerate(segment.actions):
                if available and action not in available:
                    continue
                if idx >= len(segment.frames):
                    continue
                if self._grid_equal(current_grid, segment.frames[idx]):
                    return segment, idx
        return None

    @staticmethod
    def _grid_equal(a: Any, b: Any) -> bool:
        if hasattr(a, "tolist"):
            a = a.tolist()
        if hasattr(b, "tolist"):
            b = b.tolist()
        return a == b

    @classmethod
    def from_prior_pack(cls, pack: HumanPriorPack) -> "HumanTraceMemory":
        """Build a reusable runtime memory from a compiled human prior pack."""
        memory = cls()
        steps_by_episode: Dict[str, List[StepRecord]] = {}
        for step in pack.steps:
            steps_by_episode.setdefault(step.episode_id, []).append(step)

        for episode in pack.episodes:
            ep_steps = steps_by_episode.get(episode.episode_id, [])
            if not ep_steps:
                continue
            trace = cls._build_trace(pack.game_id, episode, ep_steps)
            memory.add_trace(trace)
        return memory

    @classmethod
    def _build_trace(
        cls,
        game_id: str,
        episode: EpisodeRecord,
        steps: Sequence[StepRecord],
    ) -> HumanTrace:
        play_steps = [step for step in steps if step.action != "RESET"]
        if not play_steps:
            play_steps = list(steps)
        goal_family = _canonicalise_goal(episode.game_type_guess or episode.objective_guess)
        preferred_actions = [
            action
            for action, _count in Counter(
                step.action for step in play_steps if step.action != "RESET"
            ).most_common(4)
        ]
        colors: List[str] = []
        for step in play_steps:
            for frame in (step.frame_before, step.frame_after):
                for row in frame:
                    for cell in row:
                        if cell == 0:
                            continue
                        color_name = f"color_{cell}"
                        if color_name not in colors:
                            colors.append(color_name)
        target_objects = [
            text.strip()[:80]
            for text in episode.discovered_mechanics[:3]
            if text.strip()
        ]
        schema = TraceSchema(
            goal_family=goal_family,
            abstract_steps=cls._abstract_steps(episode, steps),
            preferred_actions=preferred_actions,
            target_objects=target_objects,
            relevant_colors=colors[:4],
            evidence={
                "objective_guess": episode.objective_guess,
                "mechanics": list(episode.discovered_mechanics[:5]),
                "mistakes": list(episode.discovered_mistakes[:3]),
            },
            score=1.0 if episode.final_state == "WIN" else min(0.85, 0.2 * episode.levels_completed + 0.1),
        )
        action_effects = [
            {
                "action": step.action,
                "changed": step.frame_before != step.frame_after,
                "game_state_after": step.game_state_after,
                "intent": step.intent,
            }
            for step in steps
        ]
        success_prefixes, danger_suffixes, recovery_frontiers = cls._distill_segments(
            game_id=game_id,
            episode=episode,
            play_steps=play_steps,
            base_score=schema.score,
        )
        return HumanTrace(
            game_id=game_id,
            episode_id=episode.episode_id,
            frames=[step.frame_before for step in play_steps] + [play_steps[-1].frame_after],
            actions=[step.action for step in play_steps if step.action != "RESET"],
            action_data=[
                dict(step.action_args) if step.action_args else None
                for step in play_steps
                if step.action != "RESET"
            ],
            inferred_goal=episode.objective_guess or episode.game_type_guess,
            goal_family=goal_family,
            final_state=episode.final_state,
            levels_completed=int(episode.levels_completed),
            action_effects=action_effects,
            success_prefixes=success_prefixes,
            danger_suffixes=danger_suffixes,
            recovery_frontiers=recovery_frontiers,
            abstract_schema=schema,
            score=schema.score,
        )

    @classmethod
    def _distill_segments(
        cls,
        *,
        game_id: str,
        episode: EpisodeRecord,
        play_steps: Sequence[StepRecord],
        base_score: float,
    ) -> tuple[List[TraceSegment], List[TraceSegment], List[RecoveryFrontier]]:
        success_prefixes: List[TraceSegment] = []
        danger_suffixes: List[TraceSegment] = []
        recovery_frontiers: List[RecoveryFrontier] = []
        if not play_steps:
            return success_prefixes, danger_suffixes, recovery_frontiers

        current_level = 0
        segment_start = 0
        for idx, step in enumerate(play_steps):
            next_level = int(step.levels_completed_after)
            if next_level <= current_level:
                continue
            segment_steps = list(play_steps[segment_start : idx + 1])
            if segment_steps:
                success_prefixes.append(
                    cls._build_segment(
                        game_id=game_id,
                        episode=episode,
                        kind="success_prefix",
                        segment_steps=segment_steps,
                        start_index=segment_start,
                        end_index=idx,
                        start_level=current_level,
                        end_level=next_level,
                        score=base_score,
                    )
                )
            current_level = next_level
            segment_start = idx + 1

        for idx, step in enumerate(play_steps):
            if step.game_state_after != "GAME_OVER":
                continue
            suffix_start = max(segment_start, idx - 4)
            suffix_steps = list(play_steps[suffix_start : idx + 1])
            danger = cls._build_segment(
                game_id=game_id,
                episode=episode,
                kind="danger_suffix",
                segment_steps=suffix_steps,
                start_index=suffix_start,
                end_index=idx,
                start_level=max(0, int(step.levels_completed_after)),
                end_level=max(0, int(step.levels_completed_after)),
                score=base_score,
            )
            danger_suffixes.append(danger)
            recovery_frontiers.append(
                RecoveryFrontier(
                    game_id=game_id,
                    episode_id=episode.episode_id,
                    frame=step.frame_before,
                    level=int(step.levels_completed_after),
                    trace_index=idx,
                    danger_suffix=danger,
                    avoid_actions=[step.action],
                )
            )

        return success_prefixes, danger_suffixes, recovery_frontiers

    @staticmethod
    def _build_segment(
        *,
        game_id: str,
        episode: EpisodeRecord,
        kind: str,
        segment_steps: Sequence[StepRecord],
        start_index: int,
        end_index: int,
        start_level: int,
        end_level: int,
        score: float,
    ) -> TraceSegment:
        return TraceSegment(
            game_id=game_id,
            episode_id=episode.episode_id,
            kind=kind,
            frames=[step.frame_before for step in segment_steps] + [segment_steps[-1].frame_after],
            actions=[step.action for step in segment_steps],
            action_data=[
                dict(step.action_args) if step.action_args else None
                for step in segment_steps
            ],
            start_index=int(start_index),
            end_index=int(end_index),
            start_level=int(start_level),
            end_level=int(end_level),
            trace_levels_completed=int(episode.levels_completed),
            final_state=segment_steps[-1].game_state_after,
            score=float(score),
        )

    @staticmethod
    def _abstract_steps(
        episode: EpisodeRecord,
        steps: Sequence[StepRecord],
    ) -> List[str]:
        intents = [step.intent for step in steps if step.intent and step.intent != "none"]
        action_counts = Counter(step.action for step in steps if step.action != "RESET")
        dominant_actions = [action for action, _count in action_counts.most_common(3)]
        abstract: List[str] = []
        if any(intent in {"probe_object", "test_interaction"} for intent in intents):
            abstract.append("identify controllable object")
        if any(intent in {"test_move", "reach_target"} for intent in intents):
            abstract.append("move toward target")
        if "ACTION6" in dominant_actions:
            abstract.append("interact when adjacent")
        if "ACTION5" in dominant_actions:
            abstract.append("switch control then continue")
        if episode.levels_completed > 0 or episode.final_state == "WIN":
            abstract.append("repeat successful burst")
        if not abstract:
            abstract.append("explore unknown affordance")
        return abstract[:4]
