"""V3 Game Memory — per-game mechanical knowledge store.

Integrates operators, rules, macros, solved trajectories, and failure
patterns into a single per-game memory that resets between games.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..schemas import (
    FailurePattern,
    GameObservation,
    MacroAction,
    Operator,
    Rule,
    SolvedTrajectory,
)
from ..mechanics.action_profiler import ActionProfiler, ActionStats
from ..mechanics.operator_inducer import OperatorInducer
from ..mechanics.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


@dataclass
class GameMemoryV3:
    """Per-game memory combining all mechanic knowledge."""

    # Core systems (references, owned by the agent)
    profiler: ActionProfiler = field(default_factory=ActionProfiler)
    inducer: OperatorInducer = field(default_factory=OperatorInducer)
    rules: RuleEngine = field(default_factory=RuleEngine)

    # Solved trajectories (per level)
    solved_trajectories: Dict[int, SolvedTrajectory] = field(default_factory=dict)

    # Failure attractors
    failure_patterns: List[FailurePattern] = field(default_factory=list)

    # State visit tracking
    state_visit_counts: Dict[int, int] = field(default_factory=dict)

    # Macro library
    macros: Dict[str, MacroAction] = field(default_factory=dict)

    # Game-level stats
    total_actions: int = 0
    total_levels_completed: int = 0
    max_level_reached: int = 0

    # Known dangerous values/positions
    lethal_values: Set[int] = field(default_factory=set)
    lethal_positions: Set[tuple] = field(default_factory=set)

    def record_state_visit(self, grid_hash: int) -> bool:
        """Record a state visit. Returns True if novel."""
        count = self.state_visit_counts.get(grid_hash, 0)
        self.state_visit_counts[grid_hash] = count + 1
        return count == 0

    def novelty_ratio(self) -> float:
        """Fraction of visits that were to novel states."""
        if self.total_actions == 0:
            return 1.0
        return len(self.state_visit_counts) / max(self.total_actions, 1)

    def record_failure(
        self,
        motif_hash: str,
        operator_trace: List[str],
        failure_type: str,
    ) -> None:
        """Record a failure pattern for avoidance."""
        for f in self.failure_patterns:
            if f.motif_hash == motif_hash:
                f.count += 1
                return
        self.failure_patterns.append(FailurePattern(
            motif_hash=motif_hash,
            operator_trace=operator_trace,
            failure_type=failure_type,
        ))
        # Cap
        if len(self.failure_patterns) > 50:
            self.failure_patterns.sort(key=lambda f: f.count, reverse=True)
            self.failure_patterns = self.failure_patterns[:50]

    def record_solution(self, trajectory: SolvedTrajectory) -> None:
        """Record a solved level trajectory."""
        existing = self.solved_trajectories.get(trajectory.level_index)
        if existing is None or trajectory.action_count < existing.action_count:
            self.solved_trajectories[trajectory.level_index] = trajectory
            self.total_levels_completed = len(self.solved_trajectories)
            self.max_level_reached = max(
                self.max_level_reached,
                trajectory.level_index + 1,
            )

    def is_known_lethal(self, value: int) -> bool:
        return value in self.lethal_values

    def mark_lethal(self, value: int) -> None:
        self.lethal_values.add(value)

    def knowledge_level(self) -> float:
        """Overall knowledge metric (0-1).

        Focused on *demonstrated* knowledge, not just induction:
          20% action coverage
          25% operator predictive accuracy (validated)
          20% operator control success (produced state progress)
          20% rule validity (rules with support > contradictions)
          15% level progress
        """
        # Action coverage
        total_actions = len(self.profiler.stats)
        covered = sum(
            1 for s in self.profiler.stats.values()
            if s.total_tries >= 2
        )
        coverage = covered / max(total_actions, 1) if total_actions > 0 else 0.0

        # Operator predictive accuracy (requires validation, not just induction)
        pred_accuracy = self.inducer.operator_predictive_accuracy()

        # Operator control success (did using operators produce state changes?)
        control_success = self.inducer.operator_control_success()

        # Rule validity (fraction of rules where support > contradictions)
        if self.rules.rules:
            valid_rules = sum(
                1 for r in self.rules.rules.values()
                if r.support > r.contradictions and r.support >= 2
            )
            rule_validity = valid_rules / len(self.rules.rules)
        else:
            rule_validity = 0.0

        # Level progress
        level = 1.0 if self.max_level_reached > 0 else 0.0

        return (
            0.20 * min(1.0, coverage)
            + 0.25 * pred_accuracy
            + 0.20 * control_success
            + 0.20 * rule_validity
            + 0.15 * level
        )

    def summary(self) -> str:
        lines = [
            f"GameMemoryV3: {self.total_actions} actions, "
            f"knowledge={self.knowledge_level():.2f}",
            f"  Levels solved: {self.total_levels_completed}",
            f"  States visited: {len(self.state_visit_counts)}",
            f"  Operators: {len(self.inducer.operators)}",
            f"  Rules: {len(self.rules.rules)}",
            f"  Macros: {len(self.macros)}",
            f"  Failures: {len(self.failure_patterns)}",
        ]
        return "\n".join(lines)
