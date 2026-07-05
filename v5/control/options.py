"""Parameterized options — multi-step behaviors built on validated operators.

Options bridge the gap between single-step operators and full solutions.
They are the compositionality layer that turns local operator knowledge
into reusable multi-step behavior.

Options:
  approach(target)            — navigate toward a target
  click_all(value)            — click all objects of a given value
  sweep(direction)            — systematically move in a direction
  repeat_until_change(op)     — keep applying an operator until grid changes
  avoid_and_reach(hazard, target) — navigate while avoiding hazards
  explore_unseen()            — move to least-visited grid region
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    GameObservation,
    ObjectInfo,
    Operator,
    OperatorCall,
    OperatorKind,
)
from ..mechanics.operator_inducer import OperatorInducer
from ..mechanics.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


class Option:
    """A parameterized multi-step behaviour built on operators."""

    name: str = "base"

    def generate_plan(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        **kwargs,
    ) -> List[OperatorCall]:
        """Generate a concrete operator-call sequence for this option."""
        return []

    def is_applicable(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
    ) -> bool:
        """Can this option fire in the current state?"""
        return False


# ─── Approach ────────────────────────────────────────────────────────

class ApproachOption(Option):
    """Navigate toward a specific target position using movement operators."""

    name = "approach"

    def is_applicable(self, obs: GameObservation, inducer: OperatorInducer) -> bool:
        return (obs.best_player is not None
                and len(inducer.get_movement_ops()) >= 2)

    def generate_plan(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        target: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> List[OperatorCall]:
        player = obs.best_player
        if player is None:
            return []

        move_ops = inducer.get_movement_ops()
        if not move_ops:
            return []

        # Auto-select target if not provided: nearest salient object
        if target is None:
            target = self._find_best_target(obs, player)
        if target is None:
            return []

        tr, tc = target
        pr, pc = player.position

        plan: List[OperatorCall] = []
        for _ in range(min(12, abs(tr - pr) + abs(tc - pc))):
            best_op = self._best_move(move_ops, pr, pc, tr, tc)
            if best_op is None:
                break
            plan.append(OperatorCall(
                operator_id=best_op.operator_id,
                args={"target": (tr, tc), "option": "approach"},
            ))
            dy = best_op.parameters.get("dy", 0)
            dx = best_op.parameters.get("dx", 0)
            pr += dy
            pc += dx
            if pr == tr and pc == tc:
                break

        return plan

    def _find_best_target(
        self, obs: GameObservation, player,
    ) -> Optional[Tuple[int, int]]:
        pr, pc = player.position
        best = None
        best_dist = float("inf")
        for obj in obs.objects:
            if obj.value == player.value:
                continue
            if obj.area > 30:
                continue
            r, c = int(round(obj.center[0])), int(round(obj.center[1]))
            dist = abs(r - pr) + abs(c - pc)
            if 1 < dist < best_dist:
                best = (r, c)
                best_dist = dist
        return best

    def _best_move(
        self, ops: List[Operator], pr, pc, tr, tc,
    ) -> Optional[Operator]:
        best_op = None
        best_score = -999.0
        for op in ops:
            dy = op.parameters.get("dy", 0)
            dx = op.parameters.get("dx", 0)
            new_dist = abs(tr - (pr + dy)) + abs(tc - (pc + dx))
            old_dist = abs(tr - pr) + abs(tc - pc)
            improvement = old_dist - new_dist
            score = improvement * op.confidence - op.risk_estimate * 2
            if score > best_score:
                best_score = score
                best_op = op
        return best_op


# ─── ClickAll ────────────────────────────────────────────────────────

class ClickAllOption(Option):
    """Click all objects of a given value using click operators."""

    name = "click_all"

    def is_applicable(self, obs: GameObservation, inducer: OperatorInducer) -> bool:
        return len(inducer.get_by_kind(OperatorKind.CLICK)) > 0

    def generate_plan(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        target_value: Optional[int] = None,
        **kwargs,
    ) -> List[OperatorCall]:
        click_ops = inducer.get_by_kind(OperatorKind.CLICK)
        if not click_ops:
            return []

        plan: List[OperatorCall] = []

        # If target_value given, use matching click op
        for op in click_ops:
            tv = op.parameters.get("target_value")
            if target_value is not None and tv != target_value:
                continue
            if tv is None:
                continue

            matching = [o for o in obs.objects if o.value == tv]
            for obj in matching:
                r = int(round(obj.center[0]))
                c = int(round(obj.center[1]))
                plan.append(OperatorCall(
                    operator_id=op.operator_id,
                    args={"x": c, "y": r, "target_value": tv,
                           "option": "click_all"},
                ))

        return plan[:15]


# ─── RepeatUntilChange ───────────────────────────────────────────────

class RepeatUntilChangeOption(Option):
    """Repeat a validated operator until the grid changes meaningfully."""

    name = "repeat_until_change"

    def is_applicable(self, obs: GameObservation, inducer: OperatorInducer) -> bool:
        return inducer.num_validated(min_uses=2, min_accuracy=0.4) >= 1

    def generate_plan(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        operator_id: Optional[str] = None,
        max_repeats: int = 8,
        **kwargs,
    ) -> List[OperatorCall]:
        # Pick operator to repeat
        if operator_id and operator_id in inducer.operators:
            op = inducer.operators[operator_id]
        else:
            # Pick best validated non-noop, non-lethal
            candidates = [
                o for o in inducer.operators.values()
                if o.kind not in (OperatorKind.NOOP, OperatorKind.LETHAL)
                and o.confidence >= 0.5
            ]
            if not candidates:
                return []
            op = max(candidates, key=lambda o: o.confidence)

        return [
            OperatorCall(
                operator_id=op.operator_id,
                args={"option": "repeat_until_change", "repeat_idx": i},
            )
            for i in range(max_repeats)
        ]


# ─── Sweep ───────────────────────────────────────────────────────────

class SweepOption(Option):
    """Systematically move in one direction to explore or reach an edge."""

    name = "sweep"

    def is_applicable(self, obs: GameObservation, inducer: OperatorInducer) -> bool:
        return len(inducer.get_movement_ops()) >= 1

    def generate_plan(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        direction: Optional[str] = None,
        max_steps: int = 10,
        **kwargs,
    ) -> List[OperatorCall]:
        move_ops = inducer.get_movement_ops()
        if not move_ops:
            return []

        # Pick direction
        if direction:
            matching = [
                op for op in move_ops
                if op.parameters.get("direction") == direction
            ]
        else:
            # Pick the most confident movement operator
            matching = sorted(move_ops, key=lambda o: o.confidence, reverse=True)

        if not matching:
            return []

        op = matching[0]
        return [
            OperatorCall(
                operator_id=op.operator_id,
                args={"option": "sweep", "direction": op.parameters.get("direction", "?")},
            )
            for _ in range(max_steps)
        ]


# ─── AvoidAndReach ───────────────────────────────────────────────────

class AvoidAndReachOption(Option):
    """Navigate toward a target while avoiding lethal values."""

    name = "avoid_and_reach"

    def is_applicable(self, obs: GameObservation, inducer: OperatorInducer) -> bool:
        return (obs.best_player is not None
                and len(inducer.get_movement_ops()) >= 2)

    def generate_plan(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        target: Optional[Tuple[int, int]] = None,
        lethal_values: Optional[set] = None,
        **kwargs,
    ) -> List[OperatorCall]:
        player = obs.best_player
        if player is None:
            return []

        move_ops = inducer.get_movement_ops()
        if len(move_ops) < 2:
            return []

        if target is None:
            # Find nearest non-lethal, non-player object
            approach = ApproachOption()
            target = approach._find_best_target(obs, player)
        if target is None:
            return []

        tr, tc = target
        pr, pc = player.position
        grid = obs.raw_grid
        H, W = grid.shape
        lethal = lethal_values or set()

        plan: List[OperatorCall] = []
        for _ in range(min(15, abs(tr - pr) + abs(tc - pc) + 3)):
            # Score moves: reduce distance but avoid lethal
            best_op = None
            best_score = -999.0
            for op in move_ops:
                dy = op.parameters.get("dy", 0)
                dx = op.parameters.get("dx", 0)
                nr, nc = pr + dy, pc + dx
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cell_val = int(grid[nr, nc])
                if cell_val in lethal:
                    continue  # skip lethal cells

                new_dist = abs(tr - nr) + abs(tc - nc)
                old_dist = abs(tr - pr) + abs(tc - pc)
                improvement = old_dist - new_dist
                score = improvement * op.confidence - op.risk_estimate * 3
                if score > best_score:
                    best_score = score
                    best_op = op

            if best_op is None:
                break
            plan.append(OperatorCall(
                operator_id=best_op.operator_id,
                args={"option": "avoid_and_reach", "target": (tr, tc)},
            ))
            dy = best_op.parameters.get("dy", 0)
            dx = best_op.parameters.get("dx", 0)
            pr += dy
            pc += dx
            if pr == tr and pc == tc:
                break

        return plan


# ─── ExploreUnseen ───────────────────────────────────────────────────

class ExploreUnseenOption(Option):
    """Move toward the least-visited region of the grid."""

    name = "explore_unseen"

    def is_applicable(self, obs: GameObservation, inducer: OperatorInducer) -> bool:
        return (obs.best_player is not None
                and len(inducer.get_movement_ops()) >= 2)

    def generate_plan(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        **kwargs,
    ) -> List[OperatorCall]:
        player = obs.best_player
        if player is None:
            return []

        move_ops = inducer.get_movement_ops()
        if not move_ops:
            return []

        # Pick random direction with highest confidence
        op = random.choice(move_ops)
        return [
            OperatorCall(
                operator_id=op.operator_id,
                args={"option": "explore_unseen"},
            )
            for _ in range(6)
        ]


# ─── Factory ─────────────────────────────────────────────────────────

def create_default_options() -> List[Option]:
    """Create the default set of options."""
    return [
        ApproachOption(),
        ClickAllOption(),
        RepeatUntilChangeOption(),
        SweepOption(),
        AvoidAndReachOption(),
        ExploreUnseenOption(),
    ]
