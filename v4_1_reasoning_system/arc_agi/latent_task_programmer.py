"""Runtime task-program synthesis from scored trajectory futures.

The offline TaskProgram JSON is a prior.  This module builds a small,
runtime TaskProgram-like surface from the trajectory scorer itself:

    scored futures -> latent context -> preferred action program

It intentionally stays deterministic and lightweight.  The goal is not
to solve the game in this module; it gives the sampler/actioner a fresh
program that is grounded in what the scorer currently believes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .goal_pursuit import GoalContext
from .trajectory_sampler import SampledTrajectory, TrajectorySampler


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class TrajectoryLatentContext:
    """Compact latent summary exposed by the trajectory scoring agent."""

    objective_id: str
    goal_family: str
    current_level: int
    selected_source: str
    selected_score: float
    selected_energy: float
    preferred_actions: List[str]
    avoid_actions: List[str]
    source_scores: Dict[str, float] = field(default_factory=dict)
    action_scores: Dict[str, float] = field(default_factory=dict)
    top_sequences: List[List[str]] = field(default_factory=list)
    preferred_sequences: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0
    override_static_program: bool = False


@dataclass
class LatentTaskProgram:
    """TaskProgram-shaped runtime program consumed through subgoal metadata."""

    objective_id: str
    description: str
    preferred_actions: List[str]
    avoid_actions: List[str]
    confidence: float
    expected_signal: str
    source: str = "trajectory_scorer"
    preferred_sequences: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def metadata_patch(self) -> Dict[str, Any]:
        """Return metadata keys understood by GoalContext and Actioner."""
        sequences = [
            list(seq)
            for seq in self.preferred_sequences
            if seq
        ]
        patch = {
            "latent_task_program": True,
            "latent_task_program_id": self.objective_id,
            "latent_task_program_source": self.source,
            "latent_task_program_confidence": float(self.confidence),
            "latent_task_program_description": self.description,
            "latent_preferred_actions": list(self.preferred_actions),
            "latent_preferred_sequences": sequences,
            "latent_avoid_actions": list(self.avoid_actions),
        }
        if sequences:
            patch["latent_macro_actions"] = list(sequences[0])
        patch.update(self.metadata)
        return patch


class LatentTaskProgramGenerator:
    """Generate a runtime task program from scored trajectory candidates."""

    def build(
        self,
        *,
        goal_context: GoalContext,
        trajectories: Sequence[SampledTrajectory],
        selected: SampledTrajectory,
        top_indices: Sequence[int],
    ) -> Optional[LatentTaskProgram]:
        if not trajectories or selected is None:
            return None

        top = [
            trajectories[idx]
            for idx in top_indices
            if 0 <= int(idx) < len(trajectories)
        ]
        if selected not in top:
            top.insert(0, selected)
        top = top[:5]
        if not top:
            return None

        context = self._latent_context(goal_context, trajectories, selected, top)
        if not context.preferred_actions:
            return None

        expected_signal = str(
            (goal_context.metadata or {}).get("expected_signal", "")
            or (goal_context.progress_signals[0] if goal_context.progress_signals else "observable_progress")
        )
        action_phrase = " -> ".join(context.preferred_actions[:4])
        macro_phrase = (
            " -> ".join(context.preferred_sequences[0])
            if context.preferred_sequences else action_phrase
        )
        description = (
            f"[latent-program] macro {macro_phrase}; prefer {action_phrase}; "
            f"selected={context.selected_source} score={context.selected_score:.3f}"
        )
        if context.override_static_program:
            description += "; overrides weak static prior"

        return LatentTaskProgram(
            objective_id=f"latent::{context.objective_id}"[:80],
            description=description[:200],
            preferred_actions=context.preferred_actions,
            avoid_actions=context.avoid_actions,
            confidence=context.confidence,
            expected_signal=expected_signal[:80],
            preferred_sequences=context.preferred_sequences,
            metadata={
                "trajectory_latent_context": {
                    "selected_source": context.selected_source,
                    "selected_score": round(context.selected_score, 3),
                    "selected_energy": round(context.selected_energy, 3),
                    "source_scores": {
                        key: round(value, 3)
                        for key, value in context.source_scores.items()
                    },
                    "action_scores": {
                        key: round(value, 3)
                        for key, value in context.action_scores.items()
                    },
                    "top_sequences": context.top_sequences,
                    "preferred_sequences": context.preferred_sequences,
                    "override_static_program": context.override_static_program,
                },
                "latent_override_static_program": context.override_static_program,
            },
        )

    def _latent_context(
        self,
        goal_context: GoalContext,
        trajectories: Sequence[SampledTrajectory],
        selected: SampledTrajectory,
        top: Sequence[SampledTrajectory],
    ) -> TrajectoryLatentContext:
        metadata = goal_context.metadata or {}
        current_level = int(metadata.get("current_level", 0) or 0)
        top_scores = [float(traj.score) for traj in top]
        min_top = min(top_scores) if top_scores else float(selected.score)
        source_scores: Dict[str, float] = {}
        source_counts: Dict[str, int] = {}
        for traj in trajectories:
            source = str(traj.source)
            source_scores[source] = max(
                source_scores.get(source, -1e9),
                float(traj.score),
            )
            source_counts[source] = source_counts.get(source, 0) + 1

        best_prior = source_scores.get("prior", -1e9)
        best_non_prior = max(
            [
                score for source, score in source_scores.items()
                if source != "prior"
            ] or [-1e9]
        )
        static_program = bool(metadata.get("from_task_program"))
        override_static = static_program and best_non_prior > best_prior + 0.08

        action_scores: Dict[str, float] = {}
        top_sequences: List[List[str]] = []
        sequence_scores: Dict[tuple[str, ...], float] = {}
        for traj in top:
            actions = TrajectorySampler._normalize_action_sequence(traj.actions)
            if not actions:
                continue
            top_sequences.append(actions[:4])
            quality = max(0.04, float(traj.score) - min_top + 0.04)
            quality *= max(0.10, 1.0 - min(1.0, float(traj.risk)))
            if override_static and traj.source == "prior":
                quality *= 0.35
            if override_static and traj.source != "prior":
                quality *= 1.25
            sequence = tuple(actions[:4])
            if len(sequence) >= 2:
                if traj is selected:
                    quality *= 1.08
                sequence_scores[sequence] = max(
                    sequence_scores.get(sequence, 0.0),
                    quality,
                )
            for idx, action in enumerate(actions[:4]):
                decay = 1.0 / (1.0 + 0.7 * idx)
                if idx == 0:
                    decay += 0.25
                action_scores[action] = action_scores.get(action, 0.0) + quality * decay

        selected_actions = TrajectorySampler._normalize_action_sequence(selected.actions)
        preferred = self._rank_actions(action_scores, selected_actions)
        avoid = self._avoid_actions(trajectories, preferred)
        preferred_sequences = [
            list(seq)
            for seq, _score in sorted(
                sequence_scores.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:4]
        ]

        best_score = float(selected.score)
        source_margin = (
            best_non_prior - best_prior
            if override_static else
            max(0.0, best_score - sorted(top_scores)[0])
        )
        top_concentration = (
            max(action_scores.values()) / max(sum(action_scores.values()), 1e-6)
            if action_scores else 0.0
        )
        confidence = _clamp(
            0.25
            + 0.20 * min(1.0, max(0.0, best_score + 0.5))
            + 0.25 * min(1.0, max(0.0, source_margin + 0.1))
            + 0.30 * top_concentration
        )

        return TrajectoryLatentContext(
            objective_id=str(goal_context.objective_id or "objective"),
            goal_family=str(goal_context.goal_family or "unknown"),
            current_level=current_level,
            selected_source=str(selected.source),
            selected_score=best_score,
            selected_energy=float(selected.energy),
            preferred_actions=preferred,
            avoid_actions=avoid,
            source_scores=source_scores,
            action_scores=action_scores,
            top_sequences=top_sequences,
            preferred_sequences=preferred_sequences,
            confidence=confidence,
            override_static_program=override_static,
        )

    @staticmethod
    def _rank_actions(
        action_scores: Dict[str, float],
        selected_actions: Sequence[str],
    ) -> List[str]:
        selected = TrajectorySampler._normalize_actions(selected_actions)
        ranked = sorted(
            action_scores,
            key=lambda action: (-float(action_scores[action]), action),
        )
        preferred: List[str] = []
        for action in selected[:3] + ranked:
            if action and action not in preferred:
                preferred.append(action)
            if len(preferred) >= 5:
                break
        return preferred

    @staticmethod
    def _avoid_actions(
        trajectories: Sequence[SampledTrajectory],
        preferred: Sequence[str],
    ) -> List[str]:
        preferred_set = set(preferred)
        risk_by_action: Dict[str, float] = {}
        low_score_by_action: Dict[str, float] = {}
        if not trajectories:
            return []
        scores = [float(traj.score) for traj in trajectories]
        score_floor = min(scores)
        score_mid = sorted(scores)[len(scores) // 2]
        for traj in trajectories:
            actions = TrajectorySampler._normalize_actions(traj.actions)
            if not actions:
                continue
            first = actions[0]
            risk_by_action[first] = max(risk_by_action.get(first, 0.0), float(traj.risk))
            if float(traj.score) <= score_mid:
                low_score_by_action[first] = max(
                    low_score_by_action.get(first, 0.0),
                    score_mid - float(traj.score) + max(0.0, -score_floor),
                )
        avoid = [
            action for action, risk in risk_by_action.items()
            if action not in preferred_set and risk >= 0.65
        ]
        avoid.extend(
            action for action, penalty in low_score_by_action.items()
            if action not in preferred_set and action not in avoid and penalty >= 0.4
        )
        return avoid[:3]
