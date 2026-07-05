"""Reactive controller — fast, cheap decisions without search.

Handles:
  - Deterministic use of locked operators
  - Immediate danger escape
  - Local greedy movement to target
  - Replay of known macros
  - Stuck detection and recovery

Not every step needs full search.  The reactive controller saves compute
and avoids overplanning.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from ..schemas import (
    GameObservation,
    MacroAction,
    Operator,
    OperatorCall,
    OperatorKind,
)
from ..mechanics.operator_inducer import OperatorInducer
from ..mechanics.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

STUCK_THRESHOLD = 4  # consecutive no-effect actions before unstick


class ReactiveController:
    """Fast reactive layer — acts without search."""

    def __init__(self, inducer: OperatorInducer, rules: RuleEngine) -> None:
        self.inducer = inducer
        self.rules = rules
        self._no_effect_streak: int = 0
        self._last_actions: List[str] = []

    def should_react(self, obs: GameObservation) -> bool:
        """Return True if reactive control is appropriate (no search needed)."""
        # Always react if in danger
        if self.rules.predict_danger(obs):
            return True

        # React if we have high-confidence operators
        applicable = self.inducer.get_applicable(obs)
        confident = [op for op in applicable if op.confidence >= 0.7]
        if confident:
            return True

        return False

    def act(
        self,
        obs: GameObservation,
        target: Optional[Dict] = None,
        macros: Optional[Dict[str, MacroAction]] = None,
    ) -> Optional[OperatorCall]:
        """Produce a reactive action.

        Returns None if reactive control has no useful suggestion.
        """
        # ── 1. Stuck recovery ──
        if self._no_effect_streak >= STUCK_THRESHOLD:
            return self._unstick(obs)

        # ── 2. Danger escape ──
        danger_rules = self.rules.predict_danger(obs)
        if danger_rules:
            return self._escape_danger(obs)

        # ── 3. Greedy movement toward target ──
        if target and "position" in target:
            move = self._greedy_move(obs, target["position"])
            if move is not None:
                return move

        # ── 4. Use best confident operator ──
        applicable = self.inducer.get_applicable(obs)
        confident = [op for op in applicable if op.confidence >= 0.7]
        if confident:
            best = max(confident, key=lambda o: o.confidence)
            return OperatorCall(
                operator_id=best.operator_id,
                args=best.parameters.copy(),
            )

        return None

    def record_effect(self, had_effect: bool) -> None:
        """Track whether last action had an effect (for stuck detection)."""
        if had_effect:
            self._no_effect_streak = 0
        else:
            self._no_effect_streak += 1

    def _escape_danger(self, obs: GameObservation) -> Optional[OperatorCall]:
        """Move away from danger using safest operator."""
        move_ops = self.inducer.get_movement_ops()
        safe = [op for op in move_ops
                if op.risk_estimate < 0.2
                and op.preconditions_met(obs)]
        if not safe:
            return None
        best = max(safe, key=lambda o: o.confidence)
        logger.info(f"Reactive: escaping danger via {best.operator_id}")
        return OperatorCall(operator_id=best.operator_id, args={})

    def _greedy_move(
        self,
        obs: GameObservation,
        target: tuple,
    ) -> Optional[OperatorCall]:
        """Greedily move one step toward target."""
        player = obs.best_player
        if player is None:
            return None

        tr, tc = target
        pr, pc = player.position
        move_ops = self.inducer.get_movement_ops()

        best_op: Optional[Operator] = None
        best_improvement: float = -999.0

        for op in move_ops:
            if not op.preconditions_met(obs):
                continue
            dy = op.parameters.get("dy", 0)
            dx = op.parameters.get("dx", 0)
            nr, nc = pr + dy, pc + dx
            new_dist = abs(tr - nr) + abs(tc - nc)
            old_dist = abs(tr - pr) + abs(tc - pc)
            improvement = (old_dist - new_dist) * op.confidence
            if improvement > best_improvement:
                best_improvement = improvement
                best_op = op

        if best_op is not None and best_improvement > 0:
            return OperatorCall(
                operator_id=best_op.operator_id,
                args={"target": target},
            )
        return None

    def _unstick(self, obs: GameObservation) -> OperatorCall:
        """Random action to escape stuck state."""
        self._no_effect_streak = 0
        available = obs.available_actions
        non_reset = [a for a in available if a != "RESET"]
        action = random.choice(non_reset) if non_reset else "ACTION1"
        logger.info(f"Reactive: unstick → random {action}")
        return OperatorCall(
            operator_id=f"unstick_{action}",
            args={"action": action},
        )
