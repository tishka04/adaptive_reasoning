"""Discriminating experiment designer (step A2).

The v3 ExperimentDesigner scores the information gain about a SINGLE action's
effect type. The shift this module makes: choose the action whose competing,
still-plausible hypotheses DISAGREE the most about the next observation — i.e.
the experiment that best separates two live theories ("if H1, the grid
transforms; if H2, nothing happens").

Output is a concrete action to try plus the competing hypotheses it tests, so
the loop can attribute the resulting confirm/refute to a deliberate experiment
(feeding theory.epistemic_metrics.experiment_efficiency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .epistemic_metrics import HypothesisStatus
from .mechanic_hypothesis import GameTheory, MechanicHypothesis


@dataclass
class ExperimentChoice:
    """A designed experiment: which action to try and what it discriminates."""

    action: str
    expected_divergence: float
    competing_keys: List[str] = field(default_factory=list)
    rationale: str = ""


class DiscriminatingExperimentDesigner:
    """Pick the action that best separates competing, plausible hypotheses."""

    def __init__(
        self,
        experiment_penalty: float = 0.15,
        lethal_penalty: float = 0.6,
    ) -> None:
        self.experiment_penalty = experiment_penalty
        self.lethal_penalty = lethal_penalty

    # ── plausibility of an unresolved hypothesis ────────────────
    @staticmethod
    def _plausibility(hyp: MechanicHypothesis, prior: float) -> float:
        # Early on confidence is 0; the decaying prior keeps bootstrap probing
        # alive so every action is tried before the theory commits.
        return max(hyp.confidence, prior)

    def score_action(self, theory: GameTheory, action: str) -> float:
        unresolved = theory.unresolved_for_action(action)
        if not unresolved:
            return float("-inf")  # nothing left to learn here

        spent = max((h.experiments_spent for h in theory.for_action(action)),
                    default=0)
        total_ev = max((h.total_evidence for h in theory.for_action(action)),
                       default=0)
        prior = 1.0 / (1.0 + total_ev)

        ranked = sorted(
            unresolved,
            key=lambda h: self._plausibility(h, prior),
            reverse=True,
        )
        top = ranked[:2]
        if len(top) >= 2:
            divergence = sum(self._plausibility(h, prior) for h in top)
        else:
            # Only one live hypothesis: still worth a confirming probe, but less.
            divergence = 0.5 * self._plausibility(top[0], prior)

        score = divergence - self.experiment_penalty * spent
        if any(
            h.status == HypothesisStatus.CONFIRMED and h.kind == "lethal"
            for h in theory.for_action(action)
        ):
            score -= self.lethal_penalty
        return score

    def design(
        self,
        theory: GameTheory,
        available_actions: List[str],
    ) -> Optional[ExperimentChoice]:
        """Return the most discriminating experiment, or None if all resolved."""
        scored: Dict[str, float] = {}
        for action in available_actions:
            action = str(action).upper()
            if action == "RESET":
                continue
            if action not in theory.actions():
                theory.seed_actions([action])
            scored[action] = self.score_action(theory, action)

        scored = {a: s for a, s in scored.items() if s > float("-inf")}
        if not scored:
            return None

        best_action = max(scored, key=lambda a: scored[a])
        best_score = scored[best_action]

        unresolved = theory.unresolved_for_action(best_action)
        total_ev = max((h.total_evidence for h in theory.for_action(best_action)),
                       default=0)
        prior = 1.0 / (1.0 + total_ev)
        competing = sorted(
            unresolved,
            key=lambda h: self._plausibility(h, prior),
            reverse=True,
        )[:2]
        rationale = (
            f"separate "
            + " vs ".join(h.kind for h in competing)
            + f" on {best_action}"
        )
        return ExperimentChoice(
            action=best_action,
            expected_divergence=round(best_score, 4),
            competing_keys=[h.key for h in competing],
            rationale=rationale,
        )
