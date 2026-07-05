"""Solution shortener — compress solved trajectories for action efficiency.

ARC-AGI-3 scoring depends on action count relative to humans.
Once a level is solved, immediately try to compress:
  - Remove redundant actions one by one
  - Remove contiguous spans
  - Replace repeated patterns with macros
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from ..schemas import PrimitiveAction, SolvedTrajectory

logger = logging.getLogger(__name__)


class SolutionShortener:
    """Compress solved trajectories into shorter equivalent solutions."""

    def __init__(self, max_iterations: int = 50) -> None:
        self.max_iterations = max_iterations
        self._compressions: int = 0

    def shorten(
        self,
        trajectory: List[PrimitiveAction],
        replay_fn: Callable[[List[PrimitiveAction]], bool],
    ) -> List[PrimitiveAction]:
        """Iteratively remove actions until no more can be removed.

        Args:
            trajectory: The solved action sequence.
            replay_fn: Function that replays a sequence and returns True
                       if it still solves the level.

        Returns:
            Shortest equivalent trajectory found.
        """
        self._compressions += 1
        best = list(trajectory)
        original_len = len(best)

        if original_len <= 1:
            return best

        logger.info(
            f"Shortener #{self._compressions}: starting with "
            f"{original_len} actions"
        )

        # ── Phase 1: single-action removal ──
        improved = True
        iterations = 0
        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1
            for i in range(len(best)):
                candidate = best[:i] + best[i + 1:]
                if not candidate:
                    continue
                if replay_fn(candidate):
                    best = candidate
                    improved = True
                    break  # restart from beginning

        # ── Phase 2: contiguous span removal ──
        for span_len in [3, 2]:
            improved = True
            while improved and iterations < self.max_iterations:
                improved = False
                iterations += 1
                for i in range(len(best) - span_len + 1):
                    candidate = best[:i] + best[i + span_len:]
                    if not candidate:
                        continue
                    if replay_fn(candidate):
                        best = candidate
                        improved = True
                        break

        # ── Phase 3: repeated pattern compression ──
        best = self._compress_repeats(best, replay_fn)

        savings = original_len - len(best)
        if savings > 0:
            logger.info(
                f"Shortener: {original_len} → {len(best)} "
                f"({savings} removed, {savings/original_len*100:.0f}%)"
            )
        else:
            logger.info("Shortener: no compression possible")

        return best

    def _compress_repeats(
        self,
        trajectory: List[PrimitiveAction],
        replay_fn: Callable[[List[PrimitiveAction]], bool],
    ) -> List[PrimitiveAction]:
        """Find and remove repeated subsequences."""
        best = list(trajectory)

        # Look for consecutive identical actions
        i = 0
        while i < len(best) - 1:
            if best[i].name == best[i + 1].name:
                # Try removing one of the pair
                candidate = best[:i] + best[i + 1:]
                if replay_fn(candidate):
                    best = candidate
                    continue  # don't advance, check same position again
            i += 1

        return best

    def shorten_trajectory(
        self,
        solved: SolvedTrajectory,
        replay_fn: Callable[[List[PrimitiveAction]], bool],
    ) -> SolvedTrajectory:
        """Shorten a SolvedTrajectory in-place and return updated version."""
        shortened = self.shorten(solved.primitive_actions, replay_fn)
        return SolvedTrajectory(
            level_index=solved.level_index,
            primitive_actions=shortened,
            operator_trace=solved.operator_trace,
            action_count=len(shortened),
            solved=True,
        )
