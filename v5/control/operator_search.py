"""Beam search over operators and macros.

Searches over *compressed causal actions* (operators, options, macros) —
NOT over raw button presses.  This dramatically reduces branching factor.

Beam scoring:
  35% predicted_goal_progress
  20% rule_consistency
  20% operator_confidence
  15% novelty_bonus
  10% cost
  20% danger
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from ..schemas import (
    GameObservation,
    MacroAction,
    Operator,
    OperatorCall,
    OperatorKind,
    SearchNode,
)
from ..mechanics.operator_inducer import OperatorInducer
from ..mechanics.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_BEAM_WIDTH = 4
DEFAULT_MAX_DEPTH = 5


class OperatorSearcher:
    """Beam search over operators and macros."""

    def __init__(
        self,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        self.beam_width = beam_width
        self.max_depth = max_depth
        self._search_count: int = 0

    def adaptive_depth(self, inducer: OperatorInducer) -> int:
        """Scale search depth based on validated operator quality."""
        validated = inducer.num_validated(min_uses=2, min_accuracy=0.4)
        ctrl_success = inducer.operator_control_success()
        if validated >= 4 and ctrl_success > 0.4:
            return min(12, self.max_depth * 2)
        elif validated >= 2:
            return min(8, int(self.max_depth * 1.5))
        return self.max_depth

    def search(
        self,
        obs: GameObservation,
        inducer: OperatorInducer,
        rules: RuleEngine,
        target: Optional[Dict] = None,
        macros: Optional[List[MacroAction]] = None,
        visited_hashes: Optional[Set[int]] = None,
    ) -> List[OperatorCall]:
        """Run beam search and return best operator sequence.

        Args:
            obs: Current observation.
            inducer: Operator library.
            rules: Rule engine for consistency checking.
            target: Optional target info {"position": (r,c), "value": int}.
            macros: Available macro actions.
            visited_hashes: State hashes already visited (for novelty).

        Returns:
            Best sequence of OperatorCalls found.
        """
        self._search_count += 1
        visited = visited_hashes or set()

        # Root node
        root = SearchNode(
            state_hash=obs.grid_hash,
            predicted_state_summary=self._state_summary(obs),
            operator_trace=[],
            cumulative_cost=0.0,
            heuristic_value=0.0,
            depth=0,
        )

        beam = [root]
        effective_depth = self.adaptive_depth(inducer)

        for depth in range(effective_depth):
            expanded: List[SearchNode] = []

            for node in beam:
                # Get applicable operators
                applicable = inducer.get_applicable(obs)

                # Also include macros
                macro_ops = self._macro_to_operators(macros or [])

                candidates = applicable + macro_ops

                for op in candidates:
                    child = self._expand(
                        node, op, obs, rules, target, visited
                    )
                    if child is not None:
                        expanded.append(child)

            if not expanded:
                break

            # Select top-K by score
            expanded.sort(key=lambda n: n.total_score, reverse=True)
            beam = expanded[:self.beam_width]

        # Return best trace
        if not beam:
            return []

        best = max(beam, key=lambda n: n.total_score)

        logger.info(
            f"Search #{self._search_count}: depth={best.depth}, "
            f"score={best.total_score:.3f}, "
            f"trace={[c.operator_id for c in best.operator_trace]}"
        )

        return best.operator_trace

    def _expand(
        self,
        parent: SearchNode,
        op: Operator,
        obs: GameObservation,
        rules: RuleEngine,
        target: Optional[Dict],
        visited: Set[int],
    ) -> Optional[SearchNode]:
        """Expand a search node by applying one operator."""
        # Skip lethal operators
        if op.kind == OperatorKind.LETHAL:
            return None
        # Skip noops (waste of actions)
        if op.kind == OperatorKind.NOOP:
            return None

        call = OperatorCall(
            operator_id=op.operator_id,
            args=op.parameters.copy(),
        )
        new_trace = parent.operator_trace + [call]

        # Predict state after this operator (simple heuristic)
        predicted = dict(parent.predicted_state_summary)
        if op.kind == OperatorKind.MOVE:
            dy = op.parameters.get("dy", 0)
            dx = op.parameters.get("dx", 0)
            pr = predicted.get("player_r", 0) + dy
            pc = predicted.get("player_c", 0) + dx
            predicted["player_r"] = pr
            predicted["player_c"] = pc

        # Heuristic scoring
        score = self._score_node(
            new_trace, op, predicted, rules, target, visited, parent.depth + 1
        )

        new_hash = hash((parent.state_hash, op.operator_id))

        return SearchNode(
            state_hash=new_hash,
            predicted_state_summary=predicted,
            operator_trace=new_trace,
            cumulative_cost=parent.cumulative_cost + op.cost_estimate,
            heuristic_value=score,
            depth=parent.depth + 1,
        )

    def _score_node(
        self,
        trace: List[OperatorCall],
        last_op: Operator,
        predicted: Dict,
        rules: RuleEngine,
        target: Optional[Dict],
        visited: Set[int],
        depth: int,
    ) -> float:
        """Score a search node."""
        # Goal progress (distance reduction if target known)
        goal_progress = 0.0
        if target and "position" in target:
            tr, tc = target["position"]
            pr = predicted.get("player_r", 0)
            pc = predicted.get("player_c", 0)
            dist = abs(tr - pr) + abs(tc - pc)
            goal_progress = max(0.0, 1.0 - dist / 20.0)

        # Rule consistency: no predicted death rules violated
        rule_consistency = 1.0
        danger_rules = rules.predict_danger
        # Simple: penalise if last operator is risky
        if last_op.risk_estimate > 0.3:
            rule_consistency -= last_op.risk_estimate

        # Operator confidence
        op_conf = last_op.confidence

        # Novelty: prefer unexplored states
        node_hash = hash(tuple(c.operator_id for c in trace))
        novelty = 0.3 if node_hash not in visited else 0.0

        # Depth penalty (prefer shorter plans)
        depth_penalty = depth * 0.05

        score = (
            0.35 * goal_progress
            + 0.20 * rule_consistency
            + 0.20 * op_conf
            + 0.15 * novelty
            - 0.10 * depth_penalty
        )

        return score

    def _state_summary(self, obs: GameObservation) -> Dict:
        """Extract a minimal state summary for search."""
        summary = {
            "grid_hash": obs.grid_hash,
            "levels": obs.levels_completed,
            "num_objects": len(obs.objects),
        }
        player = obs.best_player
        if player:
            summary["player_r"] = player.position[0]
            summary["player_c"] = player.position[1]
            summary["player_val"] = player.value
        return summary

    def _macro_to_operators(self, macros: List[MacroAction]) -> List[Operator]:
        """Wrap macros as pseudo-operators for search expansion."""
        ops: List[Operator] = []
        for m in macros:
            if m.success_rate < 0.3:
                continue
            op = Operator(
                operator_id=f"macro_{m.macro_id}",
                kind=OperatorKind.SEQUENCE,
                parameters={"macro_id": m.macro_id, "steps": len(m.steps)},
                confidence=m.success_rate,
                support=m.times_succeeded,
                cost_estimate=m.avg_cost,
                risk_estimate=1.0 - m.success_rate,
            )
            ops.append(op)
        return ops
