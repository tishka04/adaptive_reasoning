"""Bounded learned priors for V5 structural proposals.

The adapter keeps model-specific feature extraction out of the V5 control
layer. It scores primitive actions once per agent step and exposes only a
small, promote-only bonus to the arbiter.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)


class LearnedPriors:
    """Cache learned action scores and expose guarded, bounded bonuses."""

    def __init__(
        self,
        scorer: Any,
        *,
        band: float = 0.10,
        w_break: float = 0.10,
        w_progress: float = 0.0,
        max_bonus: Optional[float] = None,
        danger_threshold: float = 0.25,
        noop_threshold: float = 0.70,
    ) -> None:
        self.scorer = scorer
        self.band = max(0.0, float(band))
        self.w_break = max(0.0, float(w_break))
        self.w_progress = max(0.0, float(w_progress))
        self.max_bonus = max(
            0.0,
            float(self.band if max_bonus is None else max_bonus),
        )
        self.danger_threshold = float(danger_threshold)
        self.noop_threshold = float(noop_threshold)

        self._scores: Dict[str, Any] = {}
        self.states_scored = 0
        self.actions_scored = 0
        self.danger_guards = 0
        self.noop_guards = 0
        self.scoring_failures = 0

    @classmethod
    def from_paths(
        cls,
        action_effect: str,
        value: str,
        break_classifier: Optional[str] = None,
        macro_scores: Optional[str] = None,
        **cfg: Any,
    ) -> Optional["LearnedPriors"]:
        """Load model artifacts without making V5 startup depend on them."""
        adapter_keys = {
            "band",
            "w_break",
            "w_progress",
            "max_bonus",
            "danger_threshold",
            "noop_threshold",
        }
        adapter_cfg = {key: cfg.pop(key) for key in list(cfg) if key in adapter_keys}
        try:
            # These imports pull in numpy/joblib and the abstraction stack, so
            # keep them behind the feature flag and model-loading boundary.
            from learned_scoring import LearnedScorer

            scorer = LearnedScorer(
                str(action_effect),
                str(value),
                break_classifier_path=(
                    str(break_classifier) if break_classifier else None
                ),
                macro_scores_path=str(macro_scores) if macro_scores else None,
                **cfg,
            )
            return cls(scorer, **adapter_cfg)
        except Exception as exc:
            logger.warning("Learned priors disabled: model loading failed: %s", exc)
            return None

    def begin_step(
        self,
        grid: Any,
        available_actions: Iterable[str],
        history_features: Optional[Mapping[str, Any]],
    ) -> None:
        """Score every available primitive action once for the current step."""
        self._scores = {}
        actions = list(dict.fromkeys(str(a) for a in available_actions))
        if not actions:
            return

        try:
            from extract_state_abstractions import (
                extract_state_features,
                largest_component_local_features,
            )

            state_features = extract_state_features(grid)
            base_local = largest_component_local_features(grid)
            local_by_action = {action: base_local for action in actions}
            if "ACTION6" in local_by_action:
                shape = getattr(grid, "shape", None)
                if shape is not None and len(shape) >= 2:
                    height, width = int(shape[0]), int(shape[1])
                else:
                    height = len(grid)
                    width = len(grid[0]) if height else 0
                cursor = (
                    (max(1, height) - 1) / 2.0,
                    (max(1, width) - 1) / 2.0,
                )
                local_by_action["ACTION6"] = largest_component_local_features(
                    grid,
                    cursor=cursor,
                )

            scores = self.scorer.score_actions(
                state_features,
                actions,
                history_features,
                local_by_action,
            )
            self._scores = {str(score.action): score for score in scores}
            self.states_scored += 1
            self.actions_scored += len(self._scores)
        except Exception as exc:
            self.scoring_failures += 1
            logger.warning("Learned prior scoring failed for this step: %s", exc)

    def bonus(self, action_name: str) -> float:
        """Return a guarded, promote-only bonus for one primitive action."""
        score = self._scores.get(str(action_name))
        if score is None:
            return 0.0

        danger = float(getattr(score, "predicted_danger", 0.0))
        if not math.isfinite(danger) or danger > self.danger_threshold:
            self.danger_guards += 1
            return 0.0

        macro_scores = getattr(score, "predicted_macro_scores", {}) or {}
        no_op_probability = float(macro_scores.get("explore", 0.0))
        if not math.isfinite(no_op_probability) or no_op_probability > self.noop_threshold:
            self.noop_guards += 1
            return 0.0

        break_probability = float(
            getattr(score, "predicted_break_probability", 0.0)
        )
        progress = float(getattr(score, "predicted_progress", 0.0))
        if not math.isfinite(break_probability):
            break_probability = 0.0
        if not math.isfinite(progress):
            progress = 0.0
        raw = (
            self.w_break * max(0.0, break_probability)
            + self.w_progress * max(0.0, progress)
        )
        return min(self.max_bonus, max(0.0, float(raw)))

    def diagnostics(self) -> Dict[str, int]:
        return {
            "states_scored": self.states_scored,
            "actions_scored": self.actions_scored,
            "danger_guards": self.danger_guards,
            "noop_guards": self.noop_guards,
            "scoring_failures": self.scoring_failures,
        }
