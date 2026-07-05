"""Operator induction from observed transitions.

Transforms raw action-effect traces into reusable Operator schemas:
  movement, no-op, lethal, click, push, toggle, global transform.

The inducer runs periodically and merges/updates operators as evidence
accumulates.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from ..schemas import (
    CellFree,
    ChangedCells,
    Effect,
    FrameDiff,
    GameObservation,
    GameOverEffect,
    GlobalGridChange,
    LevelCompleteEffect,
    NoEffect,
    ObjectExists,
    Operator,
    OperatorKind,
    PlayerDisplacement,
    PlayerExists,
    Predicate,
    RemovesObject,
    TransitionRecord,
)
from .action_profiler import ActionProfiler, ActionStats

logger = logging.getLogger(__name__)

# Thresholds
MIN_SUPPORT_FOR_OPERATOR = 3
CONTRADICTION_TOLERANCE = 0.25   # max contradiction_rate before demotion
MOVEMENT_THRESHOLD = 0.6         # fraction of tries showing displacement
NOOP_THRESHOLD = 0.85            # fraction of tries with zero change
LETHAL_THRESHOLD = 0.5           # fraction of tries causing game-over
CLICK_CHANGE_THRESHOLD = 0.5     # fraction of click tries causing change
GLOBAL_CHANGE_MIN_CELLS = 5      # cells changed to count as global (scaled by grid size)


class OperatorInducer:
    """Infer reusable operators from profiled action statistics."""

    def __init__(self) -> None:
        self.operators: Dict[str, Operator] = {}
        self._induction_count: int = 0
        # Validation tracking: operator_id → {used, correct, progress}
        self._validation: Dict[str, Dict] = {}

    def induce(
        self,
        profiler: ActionProfiler,
        recent_transitions: List[TransitionRecord],
    ) -> Dict[str, Operator]:
        """Run a full induction pass over current profiler stats.

        Returns the updated operator library.
        """
        self._induction_count += 1
        new_ops: List[Operator] = []

        new_ops.extend(self._induce_movement_ops(profiler))
        new_ops.extend(self._induce_noop_ops(profiler))
        new_ops.extend(self._induce_lethal_ops(profiler))
        new_ops.extend(self._induce_click_ops(profiler, recent_transitions))
        new_ops.extend(self._induce_global_transform_ops(profiler, recent_transitions))

        # Merge into library
        for op in new_ops:
            existing = self.operators.get(op.operator_id)
            if existing is not None:
                # Update rather than replace
                existing.support = op.support
                existing.contradictions = op.contradictions
                existing.confidence = op.confidence
                existing.risk_estimate = op.risk_estimate
            else:
                self.operators[op.operator_id] = op

        # Prune low-confidence operators with enough evidence
        to_remove = []
        for oid, op in self.operators.items():
            total = op.support + op.contradictions
            if total >= MIN_SUPPORT_FOR_OPERATOR * 2:
                if op.contradictions / max(total, 1) > CONTRADICTION_TOLERANCE * 2:
                    to_remove.append(oid)
        for oid in to_remove:
            logger.info(f"Pruning unreliable operator: {oid}")
            del self.operators[oid]

        logger.info(
            f"Operator induction #{self._induction_count}: "
            f"{len(self.operators)} active operators"
        )
        return self.operators

    # ─── Movement operators ─────────────────────────────────────

    def _induce_movement_ops(self, profiler: ActionProfiler) -> List[Operator]:
        ops: List[Operator] = []

        for action_name, stats in profiler.stats.items():
            if stats.total_tries < MIN_SUPPORT_FOR_OPERATOR:
                continue

            disp = profiler.dominant_displacement(action_name)
            if disp is None:
                continue

            dy, dx = disp
            # Count how many transitions actually match this displacement
            support = 0
            contradictions = 0
            for ctx_key, ctx_stats in stats.context_buckets.items():
                if ctx_stats.tries < 1:
                    continue
                cdy, cdx = ctx_stats.mean_disp
                if (round(cdy) == dy and round(cdx) == dx):
                    support += ctx_stats.tries
                else:
                    contradictions += ctx_stats.tries - ctx_stats.changes
                    # Blocked movement in that context

            frac = support / max(stats.total_tries, 1)
            if frac < MOVEMENT_THRESHOLD:
                continue

            # Determine direction name
            if dy < 0:
                dir_name = "up"
            elif dy > 0:
                dir_name = "down"
            elif dx < 0:
                dir_name = "left"
            else:
                dir_name = "right"

            op_id = f"move_{dir_name}_{action_name}"
            preconditions: List[Predicate] = [PlayerExists()]
            # Add CellFree precondition for the target cell
            preconditions.append(CellFree(relative=(dy, dx)))

            effects: List[Effect] = [PlayerDisplacement(dy=dy, dx=dx)]

            op = Operator(
                operator_id=op_id,
                kind=OperatorKind.MOVE,
                parameters={"direction": dir_name, "dy": dy, "dx": dx},
                preconditions=preconditions,
                expected_effects=effects,
                primitive_action=action_name,
                support=support,
                contradictions=contradictions,
                risk_estimate=stats.death_rate,
                cost_estimate=1.0,
            )
            op.update_confidence()
            ops.append(op)

        return ops

    # ─── No-op operators ────────────────────────────────────────

    def _induce_noop_ops(self, profiler: ActionProfiler) -> List[Operator]:
        ops: List[Operator] = []

        for action_name, stats in profiler.stats.items():
            if stats.total_tries < MIN_SUPPORT_FOR_OPERATOR:
                continue

            noop_frac = 1.0 - stats.change_rate
            if noop_frac < NOOP_THRESHOLD:
                continue

            op_id = f"noop_{action_name}"
            op = Operator(
                operator_id=op_id,
                kind=OperatorKind.NOOP,
                parameters={},
                preconditions=[],
                expected_effects=[NoEffect()],
                primitive_action=action_name,
                support=int(noop_frac * stats.total_tries),
                contradictions=stats.total_changes,
                risk_estimate=stats.death_rate,
                cost_estimate=1.0,
            )
            op.update_confidence()
            ops.append(op)

        return ops

    # ─── Lethal operators ───────────────────────────────────────

    def _induce_lethal_ops(self, profiler: ActionProfiler) -> List[Operator]:
        ops: List[Operator] = []

        for action_name, stats in profiler.stats.items():
            if stats.total_tries < 2:
                continue
            if stats.death_rate < LETHAL_THRESHOLD:
                continue

            op_id = f"lethal_{action_name}"
            op = Operator(
                operator_id=op_id,
                kind=OperatorKind.LETHAL,
                parameters={},
                preconditions=[],
                expected_effects=[GameOverEffect()],
                primitive_action=action_name,
                support=stats.deaths,
                contradictions=stats.total_tries - stats.deaths,
                risk_estimate=stats.death_rate,
                cost_estimate=1.0,
            )
            op.update_confidence()
            ops.append(op)

        return ops

    # ─── Click operators ────────────────────────────────────────

    def _induce_click_ops(
        self,
        profiler: ActionProfiler,
        recent: List[TransitionRecord],
    ) -> List[Operator]:
        ops: List[Operator] = []

        # Look for ACTION6 or similar click-like actions
        for action_name, stats in profiler.stats.items():
            if stats.total_tries < MIN_SUPPORT_FOR_OPERATOR:
                continue
            # Click actions typically have coordinates
            # and high change rate when hitting an object
            if stats.change_rate < CLICK_CHANGE_THRESHOLD:
                continue
            # Skip if already identified as movement
            disp = profiler.dominant_displacement(action_name)
            if disp is not None:
                continue

            # Check if the action typically involves coordinates
            # by scanning recent transitions
            has_coords = False
            value_effects: Dict[int, int] = {}  # target_value → count
            for t in recent:
                if t.action.name != action_name:
                    continue
                if t.action.x is not None:
                    has_coords = True
                    # What object was at the click position?
                    r, c = t.action.y or 0, t.action.x or 0
                    grid = t.obs_before.raw_grid
                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                        val = int(grid[r, c])
                        if val != 0:
                            value_effects[val] = value_effects.get(val, 0) + 1

            if not has_coords:
                continue

            # Create click operators per target value
            for target_val, count in value_effects.items():
                if count < 2:
                    continue
                op_id = f"click_v{target_val}_{action_name}"
                op = Operator(
                    operator_id=op_id,
                    kind=OperatorKind.CLICK,
                    parameters={"target_value": target_val},
                    preconditions=[ObjectExists(value=target_val)],
                    expected_effects=[ChangedCells(), RemovesObject(value=target_val)],
                    primitive_action=action_name,
                    support=count,
                    contradictions=max(0, stats.total_tries - count),
                    risk_estimate=stats.death_rate,
                    cost_estimate=1.0,
                )
                op.update_confidence()
                ops.append(op)

        return ops

    # ─── Global transform operators ─────────────────────────────

    def _induce_global_transform_ops(
        self,
        profiler: ActionProfiler,
        recent: List[TransitionRecord],
    ) -> List[Operator]:
        ops: List[Operator] = []

        for action_name, stats in profiler.stats.items():
            if stats.total_tries < MIN_SUPPORT_FOR_OPERATOR:
                continue

            # Scale threshold by grid size — 5 is for 10×10, scale proportionally
            grid_cells = 4096  # default for 64×64
            if recent:
                for t in recent:
                    if t.obs_before is not None:
                        g = t.obs_before.raw_grid
                        grid_cells = g.shape[0] * g.shape[1]
                        break
            # At least 2% of cells must change to count as "global"
            scaled_min = max(GLOBAL_CHANGE_MIN_CELLS, int(grid_cells * 0.02))

            # Count transitions with many cells changed
            big_changes = 0
            total_with_change = 0
            for t in recent:
                if t.action.name != action_name:
                    continue
                if t.diff.num_changed > 0:
                    total_with_change += 1
                    if t.diff.num_changed >= scaled_min:
                        big_changes += 1

            if total_with_change < 2 or big_changes < 2:
                continue

            frac = big_changes / max(total_with_change, 1)
            if frac < 0.5:
                continue

            # Skip if already identified as movement or noop
            disp = profiler.dominant_displacement(action_name)
            if disp is not None:
                continue
            if stats.change_rate < 0.3:
                continue

            op_id = f"global_transform_{action_name}"
            op = Operator(
                operator_id=op_id,
                kind=OperatorKind.GLOBAL_TRANSFORM,
                parameters={},
                preconditions=[],
                expected_effects=[GlobalGridChange()],
                primitive_action=action_name,
                support=big_changes,
                contradictions=total_with_change - big_changes,
                risk_estimate=stats.death_rate,
                cost_estimate=1.0,
            )
            op.update_confidence()
            ops.append(op)

        return ops

    # ─── Queries ────────────────────────────────────────────────

    def get_applicable(self, obs: GameObservation) -> List[Operator]:
        """Return operators whose preconditions are met in current state."""
        return [
            op for op in self.operators.values()
            if op.preconditions_met(obs) and op.kind != OperatorKind.LETHAL
        ]

    def get_movement_ops(self) -> List[Operator]:
        return [op for op in self.operators.values()
                if op.kind == OperatorKind.MOVE]

    def get_by_kind(self, kind: OperatorKind) -> List[Operator]:
        return [op for op in self.operators.values() if op.kind == kind]

    def best_operator_confidence(self) -> float:
        if not self.operators:
            return 0.0
        return max(op.confidence for op in self.operators.values())

    def num_locked(self) -> int:
        """Count operators with confidence >= 0.7."""
        return sum(1 for op in self.operators.values() if op.confidence >= 0.7)

    # ─── Validation tracking ─────────────────────────────────

    def record_validation(
        self, operator_id: str, predicted_ok: bool, had_progress: bool,
    ) -> None:
        """Record whether an operator's prediction matched reality."""
        v = self._validation.setdefault(operator_id, {
            "used": 0, "correct": 0, "progress": 0,
        })
        v["used"] += 1
        if predicted_ok:
            v["correct"] += 1
        if had_progress:
            v["progress"] += 1

    def operator_predictive_accuracy(self) -> float:
        """Average predictive accuracy across validated operators."""
        if not self._validation:
            return 0.0
        accs = []
        for v in self._validation.values():
            if v["used"] >= 2:
                accs.append(v["correct"] / v["used"])
        return sum(accs) / len(accs) if accs else 0.0

    def operator_control_success(self) -> float:
        """Fraction of validated operator uses that produced state progress."""
        total_used = sum(v["used"] for v in self._validation.values())
        total_progress = sum(v["progress"] for v in self._validation.values())
        if total_used == 0:
            return 0.0
        return total_progress / total_used

    def num_validated(self, min_uses: int = 3, min_accuracy: float = 0.5) -> int:
        """Count operators that have been used enough and predict well."""
        count = 0
        for oid, v in self._validation.items():
            if v["used"] >= min_uses and v["correct"] / max(v["used"], 1) >= min_accuracy:
                count += 1
        return count

    def best_validated_confidence(self) -> float:
        """Best confidence among operators that have been validated."""
        best = 0.0
        for oid, v in self._validation.items():
            if v["used"] >= 2 and v["correct"] / max(v["used"], 1) >= 0.5:
                op = self.operators.get(oid)
                if op and op.confidence > best:
                    best = op.confidence
        return best

    def record_outcome(
        self, operator_id: str, diff: FrameDiff, success: bool
    ) -> None:
        """Update operator stats after execution."""
        op = self.operators.get(operator_id)
        if op is None:
            return
        if success:
            op.support += 1
        else:
            op.contradictions += 1
        op.update_confidence()

    def summary(self) -> str:
        lines = []
        for op in sorted(self.operators.values(),
                         key=lambda o: o.confidence, reverse=True):
            lines.append(f"  {op}")
        return "\n".join(lines) if lines else "  (no operators)"
