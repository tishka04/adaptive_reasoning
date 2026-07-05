"""
Minimal trajectory sampler for the v4_1 ARC-AGI-3 agent.

The sampler intentionally keeps a very small surface area:
  - infer a lightweight goal-conditioned action bias
  - sample K short trajectories from a heuristic/random mixture
  - let downstream scoring pick the best first action to execute

This V0 implementation deliberately avoids level-specific guards,
task-program pattern hacks, mutation replay, and long continuation logic.
Those can be reintroduced behind measured stages later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .associative_memory import AssociativeMemory
from .dynamics_inducer import DynamicsInducer
from .game_memory import GameMemory
from .goal_pursuit import GoalContext
from .hypothesis_models import ActionHypothesis, BeliefState, HypothesisTrajectory
from .state_describer import GameObservation
from .strategy_generator import GameStrategy, StrategyType
from .trajectory_memory import TrajectoryMemory


@dataclass
class SampledTrajectory:
    """A short candidate future proposed for scoring."""

    actions: List[str]
    action_data: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    predicted_latents: List[Any] = field(default_factory=list)
    predicted_observations: Optional[List[np.ndarray]] = None
    goal_context: Optional[GoalContext] = None
    source: str = "random"
    novelty: float = 0.0
    risk: float = 0.0
    energy: float = 0.0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_strategy(self) -> GameStrategy:
        """Compatibility shim for legacy callers around the strategy API."""
        desc_goal = self.goal_context.objective_id if self.goal_context else "trajectory_goal"
        click_targets = [
            data
            for action, data in zip(self.actions, self.action_data)
            if action == "ACTION6" and data is not None
        ]
        metadata = {
            "trajectory_source": self.source,
            "goal_context": self.goal_context.objective_id if self.goal_context else None,
            **self.metadata,
        }
        if click_targets:
            metadata["click_targets"] = click_targets
        return GameStrategy(
            strategy_type=StrategyType.SEQUENCE_ACTIONS,
            description=f"{self.source} trajectory toward {desc_goal}",
            action_plan=list(self.actions),
            rationale=f"Sampled {len(self.actions)}-step future from {self.source}",
            confidence=max(0.05, min(0.95, self.score if self.score else 0.5)),
            metadata=metadata,
        )


@dataclass
class ContinuationMode:
    """Kept as a simple container so later stages can reuse the interface."""

    source_trajectory: SampledTrajectory
    remaining_actions: List[str]
    remaining_action_data: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    current_index: int = 1
    adaptation_rules: Dict[str, Any] = field(default_factory=dict)
    contradiction_score: float = 0.0
    max_commit_steps: int = 3
    committed_steps: int = 0
    origin_level: int = 0

    def active(self) -> bool:
        return (
            bool(self.remaining_actions)
            and self.committed_steps < self.max_commit_steps
            and self.contradiction_score < 1.0
        )

    def as_trajectory(
        self,
        goal_context: GoalContext,
        available_actions: Sequence[str],
        horizon: int,
    ) -> Optional[SampledTrajectory]:
        if not self.active():
            return None
        preferred = TrajectorySampler._preferred_actions(goal_context, available_actions)
        actions = TrajectorySampler._adapt_actions(
            self.remaining_actions,
            available_actions,
            preferred,
            horizon,
        )
        if not actions:
            return None
        action_data = list(self.remaining_action_data[: len(actions)])
        while len(action_data) < len(actions):
            action_data.append(None)
        return SampledTrajectory(
            actions=actions,
            action_data=action_data,
            goal_context=goal_context,
            source="continuation",
            metadata={
                "continuation_bonus": 0.1,
                "continuation_contradiction": self.contradiction_score,
                "continuation_origin_source": self.source_trajectory.source,
            },
        )

    def on_feedback(
        self,
        *,
        progress_delta: float,
        prediction_match: float,
        levels_delta: int,
        game_over: bool,
    ) -> None:
        if game_over:
            self.contradiction_score = 1.0
            return
        if levels_delta > 0 or progress_delta >= 0.15 or prediction_match >= 0.7:
            self.contradiction_score = max(0.0, self.contradiction_score - 0.25)
        else:
            self.contradiction_score = min(1.0, self.contradiction_score + 0.25)
        self.committed_steps = min(self.max_commit_steps, self.committed_steps + 1)

    def carry_into_level(self, new_level: int) -> None:
        self.origin_level = int(new_level)
        self.contradiction_score = max(0.0, self.contradiction_score - 0.2)
        self.adaptation_rules["carry_across_level"] = True


class TrajectorySampler:
    """Small, stageable sampler over short action prefixes."""

    DEFAULT_V0_WEIGHTS = {
        "heuristic": 0.6,
        "random": 0.4,
    }
    DEFAULT_V1_WEIGHTS = {
        "prior": 0.45,
        "heuristic": 0.35,
        "random": 0.20,
    }

    def __init__(
        self,
        *,
        stage: str = "v0",
        planner_mode: str = "prior",
        enable_continuation: bool = False,
    ) -> None:
        self.stage = str(stage).lower()
        self.planner_mode = str(planner_mode).lower()
        self.enable_continuation = bool(enable_continuation)
        self.dynamics_inducer = DynamicsInducer()

    def sample(
        self,
        observation: GameObservation,
        goal_context: GoalContext,
        memory: GameMemory,
        assoc_memory: AssociativeMemory,
        trajectory_memory: TrajectoryMemory,
        available_actions: List[str],
        *,
        human_traces: Optional[Sequence[Any]] = None,
        continuation: Optional[ContinuationMode] = None,
        action_counter: int,
        k: int = 12,
        horizon: int = 4,
        state_embedding: Optional[np.ndarray] = None,
    ) -> List[SampledTrajectory]:
        """Sample K candidate trajectories from a minimal heuristic/random mix."""
        del action_counter, state_embedding

        horizon = max(1, min(int(horizon), 5))
        k = max(1, min(int(k), 32))
        available = self._normalize_actions(available_actions)
        if not available:
            return []

        if self.planner_mode == "hypothesis":
            return self._sample_hypothesis_mode(
                observation=observation,
                goal_context=goal_context,
                memory=memory,
                trajectory_memory=trajectory_memory,
                available_actions=available,
                k=k,
                horizon=horizon,
            )

        weights = self._weights()
        allocation = self._allocate_counts(weights, k)
        trajectories: List[SampledTrajectory] = []

        if self.enable_continuation and continuation is not None:
            trajectories.extend(
                self._sample_continuation(
                    goal_context=goal_context,
                    available_actions=available,
                    continuation=continuation,
                    horizon=horizon,
                    count=1,
                )
            )

        if self._uses_prior_guidance():
            trajectories.extend(
                self._sample_prior(
                    goal_context=goal_context,
                    available_actions=available,
                    count=allocation.get("prior", 0),
                    horizon=horizon,
                    human_traces=human_traces or [],
                )
            )
        trajectories.extend(
            self._sample_heuristic(
                observation=observation,
                goal_context=goal_context,
                assoc_memory=assoc_memory,
                available_actions=available,
                count=allocation.get("heuristic", 0),
                horizon=horizon,
            )
        )
        trajectories.extend(
            self._sample_random(
                goal_context=goal_context,
                available_actions=available,
                count=allocation.get("random", 0),
                horizon=horizon,
            )
        )
        return self._finalize(
            trajectories=trajectories,
            goal_context=goal_context,
            available_actions=available,
            trajectory_memory=trajectory_memory,
            k=k,
            horizon=horizon,
        )

    def _sample_heuristic(
        self,
        *,
        observation: GameObservation,
        goal_context: GoalContext,
        assoc_memory: AssociativeMemory,
        available_actions: Sequence[str],
        count: int,
        horizon: int,
    ) -> List[SampledTrajectory]:
        preferred = self._preferred_actions(goal_context, available_actions)
        first_pool = preferred or list(available_actions)
        click_targets = list(goal_context.click_targets or [])
        recent_actions = self._normalize_actions(
            (goal_context.metadata or {}).get("recent_actions", [])
        )

        out: List[SampledTrajectory] = []
        for idx in range(max(0, count)):
            first_action = first_pool[idx % len(first_pool)]
            actions: List[str] = []
            action_data: List[Optional[Dict[str, Any]]] = []

            for step_idx in range(horizon):
                if step_idx == 0:
                    action = first_action
                    data = self._default_action_data(action, click_targets)
                else:
                    action, data = self._next_heuristic_action(
                        observation=observation,
                        assoc_memory=assoc_memory,
                        available_actions=available_actions,
                        preferred_actions=preferred,
                        recent_actions=recent_actions + actions,
                        step_idx=step_idx + idx,
                    )
                    if data is None:
                        data = self._default_action_data(action, click_targets)
                actions.append(action)
                action_data.append(data)

            preferred_fraction = self._preferred_fraction(actions, preferred)
            out.append(
                SampledTrajectory(
                    actions=actions,
                    action_data=action_data,
                    goal_context=goal_context,
                    source="heuristic",
                    metadata={
                        "preferred_fraction": preferred_fraction,
                    },
                )
            )
        return out

    def _sample_hypothesis_mode(
        self,
        *,
        observation: GameObservation,
        goal_context: GoalContext,
        memory: GameMemory,
        trajectory_memory: TrajectoryMemory,
        available_actions: Sequence[str],
        k: int,
        horizon: int,
    ) -> List[SampledTrajectory]:
        belief = self.dynamics_inducer.induce(
            memory=memory,
            observation=observation,
            available_actions=available_actions,
        )
        ranked = self._rank_hypotheses(belief, goal_context)
        trajectories: List[SampledTrajectory] = []

        count = max(1, int(k * 0.75))
        for idx in range(count):
            if not ranked:
                break
            lead = ranked[idx % len(ranked)]
            sampled = self._compose_hypothesis_trajectory(
                lead=lead,
                ranked=ranked,
                belief=belief,
                goal_context=goal_context,
                available_actions=available_actions,
                horizon=horizon,
                offset=idx,
            )
            action_data = [
                self._default_action_data(action, goal_context.click_targets or [])
                for action in sampled.actions
            ]
            metadata = sampled.metadata(belief)
            metadata.update(
                self._hypothesis_experiment_metadata(
                    sampled.lead_hypothesis,
                    goal_context,
                )
            )
            trajectories.append(
                SampledTrajectory(
                    actions=sampled.actions,
                    action_data=action_data,
                    goal_context=goal_context,
                    source="hypothesis",
                    metadata=metadata,
                )
            )

        trajectories.extend(
            self._sample_random(
                goal_context=goal_context,
                available_actions=available_actions,
                count=max(1, k - len(trajectories)),
                horizon=horizon,
            )
        )
        return self._finalize(
            trajectories=trajectories,
            goal_context=goal_context,
            available_actions=available_actions,
            trajectory_memory=trajectory_memory,
            k=k,
            horizon=horizon,
        )

    def _compose_hypothesis_trajectory(
        self,
        *,
        lead: ActionHypothesis,
        ranked: Sequence[ActionHypothesis],
        belief: BeliefState,
        goal_context: GoalContext,
        available_actions: Sequence[str],
        horizon: int,
        offset: int,
    ) -> HypothesisTrajectory:
        del belief
        actions = [lead.action_name]
        support = self._support_hypotheses_for(lead, ranked)
        fill_pool = [hyp.action_name for hyp in support]
        if not fill_pool:
            fill_pool = [hyp.action_name for hyp in ranked if hyp.action_name != lead.action_name]
        if not fill_pool:
            fill_pool = list(available_actions)

        while len(actions) < horizon and fill_pool:
            actions.append(fill_pool[(len(actions) + offset) % len(fill_pool)])
        return HypothesisTrajectory(
            actions=actions[:horizon],
            lead_hypothesis=lead,
            support_hypotheses=support,
            goal_alignment=self._goal_alignment(lead, goal_context),
        )

    def _rank_hypotheses(
        self,
        belief: BeliefState,
        goal_context: GoalContext,
    ) -> List[ActionHypothesis]:
        return sorted(
            belief.hypotheses,
            key=lambda hyp: self._hypothesis_score(hyp, goal_context),
            reverse=True,
        )

    def _hypothesis_score(
        self,
        hypothesis: ActionHypothesis,
        goal_context: GoalContext,
    ) -> float:
        alignment = self._goal_alignment(hypothesis, goal_context)
        recent_penalty = self._recent_action_penalty(hypothesis.action_name, goal_context)
        support_saturation = self._support_saturation(hypothesis.support)
        kind_bonus = 0.0
        if hypothesis.kind in {"unknown", "control_switch", "click_activation"}:
            kind_bonus = 0.08
        return (
            0.35 * float(hypothesis.information_gain)
            + 0.25 * float(hypothesis.confidence)
            + 0.30 * alignment
            + kind_bonus
            - 0.40 * float(hypothesis.risk)
            - 0.55 * recent_penalty
            - 0.20 * support_saturation
        )

    @staticmethod
    def _support_hypotheses_for(
        lead: ActionHypothesis,
        ranked: Sequence[ActionHypothesis],
    ) -> List[ActionHypothesis]:
        if lead.kind == "control_switch":
            preferred_kinds = {"movement", "click_activation", "unknown"}
        elif lead.kind == "click_activation":
            preferred_kinds = {"movement", "control_switch", "unknown"}
        elif lead.kind == "movement":
            preferred_kinds = {"movement", "control_switch", "click_activation", "unknown"}
        else:
            preferred_kinds = {"movement", "control_switch", "click_activation", "unknown"}
        return [
            hyp for hyp in ranked
            if hyp.action_name != lead.action_name and hyp.kind in preferred_kinds
        ][:4]

    @staticmethod
    def _goal_alignment(
        hypothesis: ActionHypothesis,
        goal_context: GoalContext,
    ) -> float:
        family = str(goal_context.goal_family or "").lower()
        expected = {
            str(signal).lower()
            for signal in list(goal_context.progress_signals)
            + [str((goal_context.metadata or {}).get("expected_signal", ""))]
            if str(signal)
        }
        kind = hypothesis.kind
        if kind == "control_switch":
            return 0.85 if family == "correspondence" or "role_switch" in expected else 0.45
        if kind == "movement":
            return 0.80 if family in {"navigation", "navigate_exit", "navigate_puzzle", "correspondence"} else 0.55
        if kind == "click_activation":
            return 0.85 if family == "click_puzzle" or "click_triggered_change" in expected else 0.45
        if kind == "unknown":
            return 0.60
        if kind == "hazard":
            return 0.05
        if kind == "noop":
            return 0.15
        return 0.40

    def _hypothesis_experiment_metadata(
        self,
        hypothesis: ActionHypothesis,
        goal_context: GoalContext,
    ) -> Dict[str, float]:
        recent_penalty = self._recent_action_penalty(hypothesis.action_name, goal_context)
        support_saturation = self._support_saturation(hypothesis.support)
        coverage_bonus = 1.0 / (1.0 + max(0.0, float(hypothesis.support)))
        experiment_score = self._hypothesis_experiment_score(
            hypothesis,
            goal_context,
            recent_penalty=recent_penalty,
            support_saturation=support_saturation,
            coverage_bonus=coverage_bonus,
        )
        return {
            "hypothesis_experiment_score": experiment_score,
            "hypothesis_recent_penalty": recent_penalty,
            "hypothesis_support_saturation": support_saturation,
            "hypothesis_coverage_bonus": coverage_bonus,
        }

    def _hypothesis_experiment_score(
        self,
        hypothesis: ActionHypothesis,
        goal_context: GoalContext,
        *,
        recent_penalty: Optional[float] = None,
        support_saturation: Optional[float] = None,
        coverage_bonus: Optional[float] = None,
    ) -> float:
        alignment = self._goal_alignment(hypothesis, goal_context)
        recent = (
            self._recent_action_penalty(hypothesis.action_name, goal_context)
            if recent_penalty is None else float(recent_penalty)
        )
        saturated = (
            self._support_saturation(hypothesis.support)
            if support_saturation is None else float(support_saturation)
        )
        coverage = (
            1.0 / (1.0 + max(0.0, float(hypothesis.support)))
            if coverage_bonus is None else float(coverage_bonus)
        )
        kind_bonus = 0.0
        if hypothesis.kind in {"unknown", "control_switch", "click_activation"}:
            kind_bonus = 0.12
        raw = (
            0.34 * float(hypothesis.information_gain)
            + 0.22 * alignment
            + 0.16 * coverage
            + 0.12 * float(hypothesis.confidence)
            + kind_bonus
            - 0.30 * float(hypothesis.risk)
            - 0.42 * recent
            - 0.22 * saturated
        )
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _recent_action_penalty(action_name: str, goal_context: GoalContext) -> float:
        recent = TrajectorySampler._normalize_actions(
            (goal_context.metadata or {}).get("recent_actions", [])
        )
        if not recent:
            return 0.0
        window = recent[-6:]
        if not window:
            return 0.0
        freq = window.count(action_name) / len(window)
        last_bonus = 0.35 if window[-1] == action_name else 0.0
        streak = 0
        for action in reversed(window):
            if action != action_name:
                break
            streak += 1
        streak_bonus = min(0.35, 0.12 * max(0, streak - 1))
        return max(0.0, min(1.0, 0.65 * freq + last_bonus + streak_bonus))

    @staticmethod
    def _support_saturation(support: int) -> float:
        return max(0.0, min(1.0, (max(0, int(support)) - 2) / 8.0))

    def _sample_prior(
        self,
        *,
        goal_context: GoalContext,
        available_actions: Sequence[str],
        count: int,
        horizon: int,
        human_traces: Sequence[Any],
    ) -> List[SampledTrajectory]:
        preferred = self._preferred_actions(goal_context, available_actions)
        click_targets = list(goal_context.click_targets or [])
        seeds = self._collect_prior_seeds(
            goal_context=goal_context,
            available_actions=available_actions,
            human_traces=human_traces,
        )
        if not seeds:
            return []

        out: List[SampledTrajectory] = []
        for idx in range(max(0, count)):
            seed = seeds[idx % len(seeds)]
            actions = self._compose_prior_actions(
                seed_actions=seed["actions"],
                available_actions=available_actions,
                preferred_actions=preferred,
                horizon=horizon,
                offset=idx,
                rotate=bool(seed.get("rotate", True)),
            )
            if not actions:
                continue
            action_data = [
                self._default_action_data(action, click_targets)
                for action in actions
            ]
            schema_actions = list(seed.get("schema_preferred", []))
            human_actions = list(seed.get("human_actions", []))
            source = str(seed.get("source", "prior"))
            metadata = {
                "preferred_fraction": self._compatibility(actions, preferred),
                "human_compatibility": max(
                    self._compatibility(actions, human_actions),
                    self._compatibility(actions, schema_actions),
                ),
                "prior_kind": seed.get("kind", "task_program"),
                "prior_strength": float(seed.get("strength", 0.0)),
            }
            if seed.get("latent_macro_sequence"):
                metadata.update({
                    "latent_macro_sequence": list(seed["latent_macro_sequence"]),
                    "latent_program_confidence": float(seed.get("strength", 0.0)),
                })
            out.append(
                SampledTrajectory(
                    actions=actions,
                    action_data=action_data,
                    goal_context=goal_context,
                    source=source,
                    metadata=metadata,
                )
            )
        return out

    def _sample_random(
        self,
        *,
        goal_context: GoalContext,
        available_actions: Sequence[str],
        count: int,
        horizon: int,
    ) -> List[SampledTrajectory]:
        click_targets = list(goal_context.click_targets or [])
        out: List[SampledTrajectory] = []
        for _ in range(max(0, count)):
            actions: List[str] = []
            action_data: List[Optional[Dict[str, Any]]] = []
            for _step in range(horizon):
                action = random.choice(list(available_actions))
                actions.append(action)
                action_data.append(self._default_action_data(action, click_targets))
            out.append(
                SampledTrajectory(
                    actions=actions,
                    action_data=action_data,
                    goal_context=goal_context,
                    source="random",
                    metadata={"preferred_fraction": 0.0},
                )
            )
        return out

    def _sample_continuation(
        self,
        *,
        goal_context: GoalContext,
        available_actions: Sequence[str],
        continuation: ContinuationMode,
        horizon: int,
        count: int,
    ) -> List[SampledTrajectory]:
        out: List[SampledTrajectory] = []
        for _ in range(max(0, count)):
            projected = continuation.as_trajectory(
                goal_context=goal_context,
                available_actions=available_actions,
                horizon=horizon,
            )
            if projected is not None:
                out.append(projected)
        return out

    def _collect_prior_seeds(
        self,
        *,
        goal_context: GoalContext,
        available_actions: Sequence[str],
        human_traces: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        preferred = self._preferred_actions(goal_context, available_actions)
        seeds: List[Dict[str, Any]] = []

        for sequence in self._latent_sequences(goal_context, available_actions):
            seeds.append({
                "kind": "latent_program",
                "source": "latent_program",
                "actions": list(sequence),
                "schema_preferred": list(preferred),
                "human_actions": [],
                "strength": float(
                    (goal_context.metadata or {}).get(
                        "latent_task_program_confidence",
                        0.5,
                    )
                    or 0.5
                ),
                "rotate": False,
                "latent_macro_sequence": list(sequence),
            })

        if preferred:
            seeds.append({
                "kind": "task_program",
                "actions": list(preferred),
                "schema_preferred": [],
                "human_actions": [],
                "strength": 1.0,
            })

        for trace in human_traces:
            schema = getattr(trace, "abstract_schema", trace)
            trace_actions = self._normalize_actions(getattr(trace, "actions", []))
            schema_preferred = self._normalize_actions(
                getattr(schema, "preferred_actions", [])
            )
            seed_actions = trace_actions or schema_preferred
            if not seed_actions:
                continue
            seeds.append({
                "kind": "human_trace",
                "actions": list(seed_actions),
                "schema_preferred": list(schema_preferred),
                "human_actions": list(trace_actions),
                "strength": float(getattr(trace, "score", getattr(schema, "score", 0.0)) or 0.0),
            })
        return seeds

    @staticmethod
    def _latent_sequences(
        goal_context: GoalContext,
        available_actions: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        metadata = goal_context.metadata or {}
        raw_sequences = metadata.get("latent_preferred_sequences", [])
        if not raw_sequences:
            return []
        if isinstance(raw_sequences, str):
            raw_sequences = [raw_sequences]

        out: List[List[str]] = []
        seen: set[tuple[str, ...]] = set()
        for raw in list(raw_sequences)[:6]:
            if isinstance(raw, str):
                raw = raw.replace("->", ",").replace(">", ",").split(",")
            sequence = TrajectorySampler._normalize_action_sequence(
                raw,
                available_actions=available_actions,
                max_len=5,
            )
            if len(sequence) < 2:
                continue
            sig = tuple(sequence)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(sequence)
            if len(out) >= 4:
                break
        return out

    def _next_heuristic_action(
        self,
        *,
        observation: GameObservation,
        assoc_memory: AssociativeMemory,
        available_actions: Sequence[str],
        preferred_actions: Sequence[str],
        recent_actions: Sequence[str],
        step_idx: int,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        fallback_pool = list(preferred_actions) + [
            action for action in available_actions if action not in preferred_actions
        ]
        if not fallback_pool:
            fallback_pool = list(available_actions)

        suggested_action, suggested_data = self._assoc_suggestion(
            observation=observation,
            assoc_memory=assoc_memory,
            available_actions=available_actions,
            recent_actions=recent_actions,
        )
        if suggested_action in preferred_actions:
            return suggested_action, suggested_data
        if not preferred_actions and suggested_action in available_actions:
            return suggested_action, suggested_data
        return fallback_pool[step_idx % len(fallback_pool)], None

    def _assoc_suggestion(
        self,
        *,
        observation: GameObservation,
        assoc_memory: AssociativeMemory,
        available_actions: Sequence[str],
        recent_actions: Sequence[str],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        if assoc_memory is None or not hasattr(assoc_memory, "retrieve_action"):
            return "", None

        available_ints = [self._action_to_int(action) for action in available_actions]
        available_ints = [action for action in available_ints if action > 0]
        recent_ints = [self._action_to_int(action) for action in recent_actions]
        recent_ints = [action for action in recent_ints if action > 0]
        if not available_ints:
            return "", None

        try:
            suggestion, action_data = assoc_memory.retrieve_action(
                observation.raw_grid,
                available_ints,
                recent_actions=recent_ints,
                temperature=0.7,
            )
        except Exception:
            return "", None

        action_name = self._normalize_action(suggestion)
        if action_name not in available_actions:
            return "", None
        return action_name, action_data

    def _compose_prior_actions(
        self,
        *,
        seed_actions: Sequence[str],
        available_actions: Sequence[str],
        preferred_actions: Sequence[str],
        horizon: int,
        offset: int,
        rotate: bool = True,
    ) -> List[str]:
        seed = self._normalize_action_sequence(seed_actions)
        if rotate and seed and len(seed) > 1:
            rotate = offset % len(seed)
            seed = seed[rotate:] + seed[:rotate]
        filler_pool = list(preferred_actions) or list(seed) or list(available_actions)
        actions = self._adapt_actions(
            seed,
            available_actions,
            filler_pool,
            horizon,
        )
        while len(actions) < horizon and filler_pool:
            actions.append(filler_pool[(len(actions) + offset) % len(filler_pool)])
        return actions[:horizon]

    def _finalize(
        self,
        *,
        trajectories: Sequence[SampledTrajectory],
        goal_context: GoalContext,
        available_actions: Sequence[str],
        trajectory_memory: TrajectoryMemory,
        k: int,
        horizon: int,
    ) -> List[SampledTrajectory]:
        deduped: List[SampledTrajectory] = []
        seen: set[tuple[str, ...]] = set()

        for traj in trajectories:
            sig = tuple(traj.actions)
            if not sig or sig in seen:
                continue
            if (
                traj.source != "random"
                and trajectory_memory is not None
                and trajectory_memory.is_failed_prefix(traj.actions)
            ):
                continue
            seen.add(sig)
            deduped.append(traj)

        while len(deduped) < k:
            candidate = self._sample_random(
                goal_context=goal_context,
                available_actions=available_actions,
                count=1,
                horizon=horizon,
            )[0]
            sig = tuple(candidate.actions)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(candidate)
        return deduped[:k]

    def _weights(self) -> Dict[str, float]:
        if self._uses_prior_guidance():
            return dict(self.DEFAULT_V1_WEIGHTS)
        return dict(self.DEFAULT_V0_WEIGHTS)

    def _uses_prior_guidance(self) -> bool:
        return self.stage in {"v1", "v2", "v3"}

    @staticmethod
    def _allocate_counts(weights: Dict[str, float], k: int) -> Dict[str, int]:
        if k <= 0:
            return {name: 0 for name in weights}
        names = list(weights.keys())
        raw = {name: weights[name] * k for name in names}
        counts = {name: int(raw[name]) for name in names}
        assigned = sum(counts.values())
        remainders = sorted(
            names,
            key=lambda name: (raw[name] - counts[name], weights[name]),
            reverse=True,
        )
        for name in remainders:
            if assigned >= k:
                break
            counts[name] += 1
            assigned += 1
        if all(value == 0 for value in counts.values()):
            counts[names[0]] = k
        return counts

    @staticmethod
    def _preferred_actions(
        goal_context: GoalContext,
        available_actions: Sequence[str],
    ) -> List[str]:
        available = set(TrajectorySampler._normalize_actions(available_actions))
        preferred = []
        for action in goal_context.preferred_actions:
            action_name = TrajectorySampler._normalize_action(action)
            if action_name and action_name in available and action_name not in preferred:
                preferred.append(action_name)
        return preferred

    @staticmethod
    def _adapt_actions(
        actions: Sequence[str],
        available_actions: Sequence[str],
        preferred_actions: Sequence[str],
        horizon: int,
    ) -> List[str]:
        available = TrajectorySampler._normalize_actions(available_actions)
        preferred = list(preferred_actions) or list(available)
        out: List[str] = []
        for idx, action in enumerate(actions):
            if len(out) >= horizon:
                break
            action_name = TrajectorySampler._normalize_action(action)
            if action_name in available:
                out.append(action_name)
                continue
            if preferred:
                out.append(preferred[idx % len(preferred)])
        return out[:horizon]

    @staticmethod
    def _preferred_fraction(actions: Sequence[str], preferred_actions: Sequence[str]) -> float:
        preferred = set(preferred_actions)
        if not actions or not preferred:
            return 0.0
        return sum(1 for action in actions if action in preferred) / len(actions)

    @staticmethod
    def _compatibility(actions: Sequence[str], reference_actions: Sequence[str]) -> float:
        reference = set(TrajectorySampler._normalize_actions(reference_actions))
        if not actions or not reference:
            return 0.0
        return sum(1 for action in actions if action in reference) / len(actions)

    @staticmethod
    def _default_action_data(
        action_name: str,
        click_targets: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if action_name == "ACTION6" and click_targets:
            return dict(click_targets[0])
        return None

    @staticmethod
    def _normalize_actions(actions: Sequence[Any]) -> List[str]:
        out: List[str] = []
        for action in actions:
            action_name = TrajectorySampler._normalize_action(action)
            if action_name and action_name not in out:
                out.append(action_name)
        return out

    @staticmethod
    def _normalize_action_sequence(
        actions: Sequence[Any],
        *,
        available_actions: Optional[Sequence[str]] = None,
        max_len: Optional[int] = None,
    ) -> List[str]:
        available = None
        if available_actions is not None:
            available = set(TrajectorySampler._normalize_actions(available_actions))
        out: List[str] = []
        for action in list(actions) if actions is not None else []:
            action_name = TrajectorySampler._normalize_action(action)
            if not action_name:
                continue
            if available is not None and action_name not in available:
                continue
            out.append(action_name)
            if max_len is not None and len(out) >= max_len:
                break
        return out

    @staticmethod
    def _normalize_action(action: Any) -> str:
        if isinstance(action, str):
            action_name = action.strip().upper()
            if action_name.startswith("ACTION"):
                return action_name
            if action_name.isdigit():
                return f"ACTION{int(action_name)}"
            return action_name
        if isinstance(action, (int, np.integer)):
            if int(action) <= 0:
                return ""
            return f"ACTION{int(action)}"
        return ""

    @staticmethod
    def _action_to_int(action: Any) -> int:
        action_name = TrajectorySampler._normalize_action(action)
        if not action_name.startswith("ACTION"):
            return 0
        try:
            return int(action_name.replace("ACTION", ""))
        except ValueError:
            return 0
