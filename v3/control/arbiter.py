"""Bayesian arbiter — selects the best specialist mind proposal.

Performance-based, not LLM-based.  Scores proposals by:
  expected progress, info gain, mind reliability, operator confidence,
  action cost, and risk.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..schemas import MindProposal
from .specialist_minds import SpecialistMind

logger = logging.getLogger(__name__)


class Arbiter:
    """Select the best proposal from competing specialist minds."""

    def __init__(
        self,
        w_progress: float = 0.30,
        w_info_gain: float = 0.20,
        w_accuracy: float = 0.20,
        w_confidence: float = 0.15,
        w_cost: float = 0.05,
        w_risk: float = 0.10,
    ) -> None:
        self.w_progress = w_progress
        self.w_info_gain = w_info_gain
        self.w_accuracy = w_accuracy
        self.w_confidence = w_confidence
        self.w_cost = w_cost
        self.w_risk = w_risk

        self._selection_count: int = 0
        self._mind_selections: Dict[str, int] = {}

    def score_proposal(
        self,
        proposal: MindProposal,
        mind: SpecialistMind,
    ) -> float:
        """Compute composite score for a proposal."""
        score = (
            self.w_progress * proposal.expected_progress
            + self.w_info_gain * proposal.expected_info_gain
            + self.w_accuracy * mind.recent_accuracy
            + self.w_confidence * proposal.confidence
            - self.w_cost * min(proposal.estimated_cost / 10.0, 1.0)
            - self.w_risk * proposal.estimated_risk
        )
        return score

    def select(
        self,
        proposals: List[MindProposal],
        minds: Dict[str, SpecialistMind],
    ) -> Optional[MindProposal]:
        """Select the best proposal.

        Args:
            proposals: Proposals from each mind (may include None-filtered).
            minds: Dict mapping mind_name → SpecialistMind instance.

        Returns:
            Best proposal, or None if all proposals are empty.
        """
        if not proposals:
            return None

        self._selection_count += 1

        scored: List[tuple[float, MindProposal]] = []
        for p in proposals:
            mind = minds.get(p.mind_name)
            if mind is None:
                continue
            s = self.score_proposal(p, mind)
            scored.append((s, p))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]

        # Track selections
        self._mind_selections[best.mind_name] = (
            self._mind_selections.get(best.mind_name, 0) + 1
        )

        if len(scored) >= 2:
            runner_score, runner = scored[1]
            logger.info(
                f"Arbiter #{self._selection_count}: "
                f"{best.mind_name}({best_score:.3f}) > "
                f"{runner.mind_name}({runner_score:.3f}) "
                f"→ {best.objective[:60]}"
            )
        else:
            logger.info(
                f"Arbiter #{self._selection_count}: "
                f"{best.mind_name}({best_score:.3f}) → {best.objective[:60]}"
            )

        return best

    def update_mind_outcomes(
        self,
        mind_name: str,
        minds: Dict[str, SpecialistMind],
        predicted_ok: bool,
        realised_progress: float,
    ) -> None:
        """Feedback loop: update mind reliability after execution."""
        mind = minds.get(mind_name)
        if mind is not None:
            mind.record_outcome(predicted_ok, realised_progress)

    def selection_summary(self) -> str:
        """Summary of which minds have been selected how often."""
        if not self._mind_selections:
            return "  (no selections yet)"
        lines = []
        for name, count in sorted(
            self._mind_selections.items(), key=lambda x: x[1], reverse=True
        ):
            pct = 100 * count / self._selection_count
            lines.append(f"  {name}: {count} ({pct:.0f}%)")
        return "\n".join(lines)
