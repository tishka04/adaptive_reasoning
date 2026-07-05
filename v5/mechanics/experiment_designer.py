"""Causal experiment designer.

Instead of 'informed probing', the agent asks:
  "What is the most uncertainty-reducing experiment I can perform
   in the next 1-3 actions?"

Each candidate action is scored for expected information gain about
hidden game mechanics, not just immediate progress.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    ExperimentPlan,
    GameObservation,
    OperatorCall,
    PlannedAction,
    PrimitiveAction,
)
from .action_profiler import ActionProfiler
from .operator_inducer import OperatorInducer

logger = logging.getLogger(__name__)

# How many effect-type hypotheses we track per action
EFFECT_TYPES = [
    "movement",
    "interaction",
    "noop",
    "lethal",
    "global_transform",
    "conditional",
    "unknown",
]


class ExperimentDesigner:
    """Design targeted experiments to maximally reduce mechanic uncertainty.

    Maintains a posterior over effect types for each action, then chooses
    experiments that maximise expected entropy reduction per action spent.
    """

    def __init__(self) -> None:
        # action_name → {effect_type: count}
        self.posteriors: Dict[str, Dict[str, float]] = {}
        self._experiment_count: int = 0

    # ─── Posterior maintenance ──────────────────────────────────

    def update_posterior(
        self,
        action_name: str,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
    ) -> None:
        """Rebuild posterior for one action from profiler stats + operators."""
        stats = profiler.get_stats(action_name)
        if stats is None:
            # Uniform prior
            self.posteriors[action_name] = {
                t: 1.0 / len(EFFECT_TYPES) for t in EFFECT_TYPES
            }
            return

        counts: Dict[str, float] = {t: 0.1 for t in EFFECT_TYPES}  # smoothing

        n = stats.total_tries
        if n == 0:
            self.posteriors[action_name] = {
                t: 1.0 / len(EFFECT_TYPES) for t in EFFECT_TYPES
            }
            return

        # Evidence from profiler
        disp = profiler.dominant_displacement(action_name)
        if disp is not None:
            counts["movement"] += n * 0.6
        if stats.change_rate < 0.15:
            counts["noop"] += n * 0.7
        if stats.death_rate > 0.3:
            counts["lethal"] += n * stats.death_rate
        if stats.change_rate > 0.5 and disp is None:
            counts["interaction"] += n * 0.3
            counts["global_transform"] += n * 0.2

        # Evidence from induced operators
        for op in inducer.operators.values():
            if op.primitive_action != action_name:
                continue
            kind_str = op.kind.value
            if kind_str in counts:
                counts[kind_str] += op.support * 0.5

        # Normalise
        total = sum(counts.values())
        self.posteriors[action_name] = {
            t: c / total for t, c in counts.items()
        }

    def update_all(
        self,
        available: List[str],
        profiler: ActionProfiler,
        inducer: OperatorInducer,
    ) -> None:
        """Update posteriors for all available actions."""
        for a in available:
            self.update_posterior(a, profiler, inducer)

    # ─── Entropy and info gain ──────────────────────────────────

    def _entropy(self, dist: Dict[str, float]) -> float:
        """Shannon entropy of a discrete distribution."""
        h = 0.0
        for p in dist.values():
            if p > 1e-10:
                h -= p * math.log2(p)
        return h

    def info_gain_score(
        self,
        action_name: str,
        profiler: ActionProfiler,
    ) -> float:
        """Estimated information gain from trying this action once more.

        Higher = more uncertainty to resolve = better experiment.
        Penalised by death risk and action cost.
        """
        dist = self.posteriors.get(action_name)
        if dist is None:
            return 1.0  # unknown → maximum curiosity

        entropy = self._entropy(dist)
        stats = profiler.get_stats(action_name)
        death_risk = stats.death_rate if stats else 0.0

        # Diminishing returns: less info gain if already well-profiled
        tries = stats.total_tries if stats else 0
        novelty_decay = 1.0 / (1.0 + tries * 0.15)

        return entropy * novelty_decay - 0.5 * death_risk

    # ─── Experiment design ──────────────────────────────────────

    def design(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        max_steps: int = 3,
    ) -> ExperimentPlan:
        """Design the best uncertainty-reducing experiment.

        Returns a plan of 1-3 actions targeting the most informative
        mechanic test.
        """
        self._experiment_count += 1
        available = obs.available_actions
        self.update_all(available, profiler, inducer)

        # Score each action by info gain
        scores: List[Tuple[str, float]] = []
        for a in available:
            if a == "RESET":
                continue
            score = self.info_gain_score(a, profiler)
            scores.append((a, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            return ExperimentPlan(
                objective="fallback random",
                candidate_hypotheses=[],
                planned_actions=[PlannedAction(
                    primitive=PrimitiveAction(name=random.choice(available)),
                    purpose="random fallback",
                )],
                expected_info_gain=0.0,
            )

        # Build plan: try the top-K most informative actions
        plan_actions: List[PlannedAction] = []
        hypotheses: List[str] = []

        for action_name, score in scores[:max_steps]:
            dist = self.posteriors.get(action_name, {})
            # Top competing hypotheses
            top_types = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:2]
            hyp_desc = (
                f"{action_name}: {top_types[0][0]}({top_types[0][1]:.2f}) "
                f"vs {top_types[1][0]}({top_types[1][1]:.2f})"
                if len(top_types) >= 2 else f"{action_name}: uncertain"
            )
            hypotheses.append(hyp_desc)

            plan_actions.append(PlannedAction(
                primitive=PrimitiveAction(name=action_name),
                purpose=f"test {hyp_desc}",
            ))

        objective = f"disambiguate {len(plan_actions)} actions"
        total_gain = sum(s for _, s in scores[:max_steps])

        logger.info(
            f"Experiment #{self._experiment_count}: {objective} "
            f"(gain={total_gain:.2f})"
        )

        return ExperimentPlan(
            objective=objective,
            candidate_hypotheses=hypotheses,
            planned_actions=plan_actions,
            expected_info_gain=total_gain,
        )

    def should_experiment(
        self,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        available: List[str],
        confidence_threshold: float = 0.6,
    ) -> bool:
        """Decide whether the agent should run an experiment now.

        Returns True when:
        - Best operator confidence is low
        - Many actions are under-profiled
        - Entropy across posteriors is high
        """
        # Low operator confidence
        if inducer.best_operator_confidence() < confidence_threshold:
            return True

        # Under-profiled actions
        coverage = profiler.action_coverage(available, min_tries=3)
        if coverage < 0.6:
            return True

        # High average entropy
        if self.posteriors:
            avg_entropy = sum(
                self._entropy(d)
                for d in self.posteriors.values()
            ) / len(self.posteriors)
            if avg_entropy > 1.5:  # high uncertainty
                return True

        return False
