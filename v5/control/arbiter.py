"""Bayesian arbiter — selects the best specialist mind proposal.

Performance-based, not LLM-based.  Scores proposals by:
  expected progress, info gain, mind reliability, operator confidence,
  action cost, and risk.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional

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
        self._prior_bonus_fn: Optional[Callable[[MindProposal], float]] = None
        self._prior_band: float = 0.0
        self.prior_reorders: int = 0
        self.prior_promotions: int = 0

    def set_prior(
        self,
        bonus_fn: Optional[Callable[[MindProposal], float]],
        band: float,
    ) -> None:
        """Enable a promote-only tie-breaker within a structural score band."""
        self._prior_bonus_fn = bonus_fn
        self._prior_band = max(0.0, float(band))

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
        structural_best_score, structural_best = scored[0]
        best_score, best = structural_best_score, structural_best

        if self._prior_bonus_fn is not None and len(scored) >= 2:
            adjusted: List[tuple[float, float, MindProposal]] = []
            band_floor = structural_best_score - self._prior_band
            for structural_score, proposal in scored:
                bonus = 0.0
                if structural_score >= band_floor:
                    try:
                        bonus = float(self._prior_bonus_fn(proposal))
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("Learned prior failed for proposal: %s", exc)
                    if not math.isfinite(bonus) or bonus < 0.0:
                        bonus = 0.0
                adjusted.append(
                    (structural_score + bonus, structural_score, proposal)
                )

            adjusted_score, best_score, best = max(
                adjusted,
                key=lambda item: item[0],
            )
            if best is not structural_best:
                self.prior_reorders += 1
                self.prior_promotions += 1
                logger.info(
                    "Arbiter prior promoted %s over %s (%.3f vs %.3f adjusted)",
                    best.mind_name,
                    structural_best.mind_name,
                    adjusted_score,
                    structural_best_score,
                )

        # Track selections
        self._mind_selections[best.mind_name] = (
            self._mind_selections.get(best.mind_name, 0) + 1
        )

        if best is not structural_best:
            logger.info(
                f"Arbiter #{self._selection_count}: "
                f"{best.mind_name} selected via prior over "
                f"{structural_best.mind_name} "
                f"({best_score:.3f} vs {structural_best_score:.3f} structural) "
                f"-> {best.objective[:60]}"
            )
        elif len(scored) >= 2:
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
