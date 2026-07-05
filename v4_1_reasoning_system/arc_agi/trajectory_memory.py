"""
Trajectory memory for the sampling-based ARC-AGI-3 agent.

Stores full trajectory attempts plus short reusable fragments. The memory
acts as a low-trust proposal prior: helpful for sampling, never authoritative.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .goal_pursuit import GoalContext, TrajectoryOutcome


@dataclass
class TrajectoryRecord:
    """Compact memory item used to seed future sampling."""

    goal_family: str
    objective_id: str
    actions: List[str]
    action_data: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    source: str = "agent"
    score: float = 0.0
    progress_delta: float = 0.0
    prediction_match: float = 0.0
    success: bool = False
    state_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        emb = None
        if self.state_embedding is not None:
            emb = np.asarray(self.state_embedding, dtype=np.float32).tolist()
        return {
            "goal_family": self.goal_family,
            "objective_id": self.objective_id,
            "actions": list(self.actions),
            "action_data": list(self.action_data),
            "source": self.source,
            "score": float(self.score),
            "progress_delta": float(self.progress_delta),
            "prediction_match": float(self.prediction_match),
            "success": bool(self.success),
            "state_embedding": emb,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryRecord":
        emb = data.get("state_embedding")
        state_embedding = None
        if emb is not None:
            state_embedding = np.asarray(emb, dtype=np.float32)
        return cls(
            goal_family=str(data.get("goal_family", "unknown")),
            objective_id=str(data.get("objective_id", "unknown")),
            actions=[str(a) for a in data.get("actions", [])],
            action_data=list(data.get("action_data", [])),
            source=str(data.get("source", "agent")),
            score=float(data.get("score", 0.0)),
            progress_delta=float(data.get("progress_delta", 0.0)),
            prediction_match=float(data.get("prediction_match", 0.0)),
            success=bool(data.get("success", False)),
            state_embedding=state_embedding,
            metadata=dict(data.get("metadata", {})),
        )


class TrajectoryMemory:
    """Stores and retrieves compact trajectory priors."""

    HUMAN_BASE_WEIGHT = 0.35
    HUMAN_COLD_START_WEIGHT = 0.45
    HUMAN_LATE_WEIGHT = 0.20
    HUMAN_PRIOR_MIN_TRUST = 0.05
    HUMAN_PRIOR_MAX_TRUST = 0.70

    def __init__(
        self,
        cross_game: Optional[Any] = None,
        *,
        max_records: int = 256,
        max_fragments: int = 256,
    ) -> None:
        self.max_records = max_records
        self.max_fragments = max_fragments
        self.records: List[TrajectoryRecord] = []
        self.fragments: List[TrajectoryRecord] = []
        self.failed_prefixes: set[tuple[str, ...]] = set()
        self.human_prior_trust: float = self.HUMAN_BASE_WEIGHT
        if cross_game is not None:
            self.seed_from_cross_game(cross_game)

    # ------------------------------------------------------------------
    # Seeding and persistence helpers
    # ------------------------------------------------------------------
    def seed_from_cross_game(self, cross_game: Any) -> None:
        """Import compact trajectory priors from cross-game memory."""
        priors = getattr(cross_game, "trajectory_priors", {}) or {}
        for goal_family, items in priors.items():
            for item in items:
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                item.setdefault("goal_family", goal_family)
                record = TrajectoryRecord.from_dict(item)
                self._append_record(record, into_fragments=False)
                self._append_record(record, into_fragments=True)

    def _append_record(self, record: TrajectoryRecord, *, into_fragments: bool) -> None:
        bucket = self.fragments if into_fragments else self.records
        bucket.append(record)
        bucket.sort(
            key=lambda r: (
                -float(r.progress_delta),
                -float(r.prediction_match),
                -float(r.score),
                -int(r.success),
            )
        )
        cap = self.max_fragments if into_fragments else self.max_records
        if len(bucket) > cap:
            del bucket[cap:]

    # ------------------------------------------------------------------
    # Proposal priors
    # ------------------------------------------------------------------
    def human_prior_weight(self, action_counter: int) -> float:
        """Action-count schedule for human/task-program proposal weight."""
        if action_counter <= 20:
            frac = max(0.0, min(1.0, action_counter / 20.0))
            base = self.HUMAN_COLD_START_WEIGHT - 0.10 * frac
        elif action_counter <= 50:
            frac = (action_counter - 20) / 30.0
            base = self.HUMAN_BASE_WEIGHT - 0.15 * frac
        else:
            base = self.HUMAN_LATE_WEIGHT
        trust_scale = self.human_prior_trust / max(self.HUMAN_BASE_WEIGHT, 1e-6)
        return max(0.0, min(self.HUMAN_COLD_START_WEIGHT, base * trust_scale))

    def update_prior_trust(
        self,
        source: str,
        *,
        prediction_match: float,
        progress_delta: float,
    ) -> None:
        """Bayesian-like trust update for human-guided priors."""
        if source not in {"human", "human_trace", "task_program"}:
            return
        contradicted = prediction_match < 0.35 and progress_delta <= 0.05
        if contradicted:
            self.human_prior_trust = max(
                self.HUMAN_PRIOR_MIN_TRUST,
                self.human_prior_trust * 0.5,
            )
            return
        if prediction_match >= 0.70 or progress_delta >= 0.15:
            self.human_prior_trust = min(
                self.HUMAN_PRIOR_MAX_TRUST,
                self.human_prior_trust + 0.05,
            )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def store(
        self,
        outcome: TrajectoryOutcome,
        *,
        state_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Store a full attempted trajectory and derive reusable fragments."""
        goal_ctx = outcome.goal_context
        actions = [step.get("action", "") for step in outcome.prefix_executed if step.get("action")]
        action_data = [step.get("action_data") for step in outcome.prefix_executed if step.get("action")]
        if not actions:
            return

        success = bool(outcome.levels_delta > 0 or outcome.progress_delta >= 0.15)
        record = TrajectoryRecord(
            goal_family=goal_ctx.goal_family,
            objective_id=goal_ctx.objective_id,
            actions=actions,
            action_data=action_data,
            source=outcome.source,
            score=float(outcome.progress_delta + 0.25 * outcome.prediction_match),
            progress_delta=float(outcome.progress_delta),
            prediction_match=float(outcome.prediction_match),
            success=success,
            state_embedding=None if state_embedding is None else np.asarray(state_embedding, dtype=np.float32),
            metadata={
                "game_over": bool(outcome.game_over),
                "levels_delta": int(outcome.levels_delta),
            },
        )
        self._append_record(record, into_fragments=False)

        # Store short reusable fragments so new games can recombine them.
        max_len = min(5, len(actions))
        for frag_len in range(2, max_len + 1):
            fragment = TrajectoryRecord(
                goal_family=goal_ctx.goal_family,
                objective_id=goal_ctx.objective_id,
                actions=list(actions[:frag_len]),
                action_data=list(action_data[:frag_len]),
                source=outcome.source,
                score=record.score,
                progress_delta=record.progress_delta,
                prediction_match=record.prediction_match,
                success=record.success,
                state_embedding=record.state_embedding,
                metadata={"fragment": True, **record.metadata},
            )
            self._append_record(fragment, into_fragments=True)

    # ------------------------------------------------------------------
    # Retrieval and mutation
    # ------------------------------------------------------------------
    def retrieve_similar(
        self,
        goal_family: str,
        state_embedding: Optional[np.ndarray],
        top_k: int = 3,
    ) -> List[TrajectoryRecord]:
        """Retrieve similar prior trajectories for the current goal context."""
        candidates = [r for r in self.fragments if r.goal_family == goal_family]
        if not candidates:
            candidates = list(self.fragments)
        if not candidates:
            return []

        query = None if state_embedding is None else np.asarray(state_embedding, dtype=np.float32)
        scored: List[tuple[float, TrajectoryRecord]] = []
        for rec in candidates:
            sim = self._embedding_similarity(query, rec.state_embedding)
            quality = 0.5 * float(rec.progress_delta) + 0.3 * float(rec.prediction_match)
            if rec.success:
                quality += 0.2
            scored.append((sim + quality, rec))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def mutate(
        self,
        seed_trajectory: Sequence[str],
        available_actions: Sequence[str],
        *,
        n: int = 2,
        preferred_actions: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """Generate local variations around a known-good trajectory."""
        seed = [a for a in seed_trajectory if a]
        if not seed:
            return []
        available = [a for a in available_actions if a]
        if not available:
            return []
        preferred = [a for a in (preferred_actions or []) if a in available]
        pool = preferred + [a for a in available if a not in preferred]
        out: List[List[str]] = []
        for _ in range(max(n, 0)):
            mutated = list(seed)
            op = random.choice(("swap", "replace", "append", "truncate"))
            if op == "swap" and len(mutated) >= 2:
                i = random.randrange(len(mutated))
                j = random.randrange(len(mutated))
                mutated[i], mutated[j] = mutated[j], mutated[i]
            elif op == "replace":
                idx = random.randrange(len(mutated))
                mutated[idx] = random.choice(pool)
            elif op == "append" and len(mutated) < 5:
                mutated.append(random.choice(pool))
            elif op == "truncate" and len(mutated) > 2:
                mutated = mutated[: random.randint(2, len(mutated))]
            out.append(mutated)
        return out

    def remember_failed_prefix(self, actions: Sequence[str]) -> None:
        """Persist a failed prefix so exact resampling becomes less attractive."""
        failed = [a for a in actions if a]
        if len(failed) < 2:
            return
        self.failed_prefixes.add(tuple(failed))
        record = TrajectoryRecord(
            goal_family="unknown",
            objective_id="failed_prefix",
            actions=failed,
            source="failure",
            score=-0.5,
            progress_delta=0.0,
            prediction_match=0.0,
            success=False,
            metadata={"failed_prefix": True},
        )
        self._append_record(record, into_fragments=True)

    def is_failed_prefix(self, actions: Sequence[str]) -> bool:
        return tuple(a for a in actions if a) in self.failed_prefixes

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def export_record(self, outcome: TrajectoryOutcome) -> Dict[str, Any]:
        """Convert an outcome to a compact cross-game payload."""
        return {
            "goal_family": outcome.goal_context.goal_family,
            "objective_id": outcome.goal_context.objective_id,
            "actions": [step.get("action", "") for step in outcome.prefix_executed],
            "action_data": [step.get("action_data") for step in outcome.prefix_executed],
            "source": outcome.source,
            "score": float(outcome.progress_delta + 0.25 * outcome.prediction_match),
            "progress_delta": float(outcome.progress_delta),
            "prediction_match": float(outcome.prediction_match),
            "success": bool(outcome.levels_delta > 0 or outcome.progress_delta >= 0.15),
            "metadata": {"levels_delta": int(outcome.levels_delta), "game_over": bool(outcome.game_over)},
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "records": len(self.records),
            "fragments": len(self.fragments),
            "failed_prefixes": len(self.failed_prefixes),
            "human_prior_trust": round(float(self.human_prior_trust), 3),
        }

    @staticmethod
    def _embedding_similarity(
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
    ) -> float:
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            n = min(a.size, b.size)
            if n == 0:
                return 0.0
            a = a.reshape(-1)[:n]
            b = b.reshape(-1)[:n]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        return max(-1.0, min(1.0, float(np.dot(a, b) / denom)))
