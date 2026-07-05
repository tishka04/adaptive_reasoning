"""Specialist minds — competing inductive biases.

Each mind embodies a different worldview about what kind of game this is
and proposes operator-level plans accordingly.  The arbiter selects or
mixes proposals based on recent prediction accuracy.

Implemented minds:
  NavigatorMind  — assumes spatial movement toward goals
  ClickMind      — assumes object selection / click interaction
  SequenceMind   — assumes exact short action programs matter
  PhysicsMind    — assumes pushing, collisions, projectiles
  TransformMind  — assumes interactions trigger global grid changes
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    GameObservation,
    MindProposal,
    Operator,
    OperatorCall,
    OperatorKind,
)
from ..mechanics.action_profiler import ActionProfiler
from ..mechanics.operator_inducer import OperatorInducer
from ..mechanics.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


class SpecialistMind(ABC):
    """Base class for all specialist minds."""

    name: str = "base"

    # Tracking for arbiter calibration
    proposals_made: int = 0
    predictions_correct: int = 0
    progress_realised: float = 0.0

    @abstractmethod
    def propose(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        rules: RuleEngine,
    ) -> Optional[MindProposal]:
        """Generate a proposal for the current state.

        Returns None if this mind has no useful suggestion.
        """
        ...

    @property
    def recent_accuracy(self) -> float:
        if self.proposals_made == 0:
            return 0.5  # neutral prior
        return self.predictions_correct / self.proposals_made

    def record_outcome(self, predicted_ok: bool, progress: float) -> None:
        self.proposals_made += 1
        if predicted_ok:
            self.predictions_correct += 1
        self.progress_realised += progress


# =====================================================================
# NavigatorMind
# =====================================================================

class NavigatorMind(SpecialistMind):
    """Assumes: controllable player, spatial goals, movement operators."""

    name = "navigator"

    def propose(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        rules: RuleEngine,
    ) -> Optional[MindProposal]:
        player = obs.best_player
        if player is None or player.confidence < 0.3:
            return None

        move_ops = inducer.get_movement_ops()
        if not move_ops:
            return None

        # Find nearest interesting target
        target = self._find_target(obs, player)
        if target is None:
            return None

        tr, tc = target
        pr, pc = player.position

        # Plan: sequence of movement operators toward target
        plan: List[OperatorCall] = []
        for _ in range(min(8, abs(tr - pr) + abs(tc - pc))):
            best_op = self._best_move_toward(move_ops, pr, pc, tr, tc, obs)
            if best_op is None:
                break
            plan.append(OperatorCall(
                operator_id=best_op.operator_id,
                args={"target": (tr, tc)},
            ))
            # Simulate step
            dy = best_op.parameters.get("dy", 0)
            dx = best_op.parameters.get("dx", 0)
            pr += dy
            pc += dx
            if pr == tr and pc == tc:
                break

        if not plan:
            return None

        dist = abs(tr - player.position[0]) + abs(tc - player.position[1])
        confidence = min(0.9, max(op.confidence for op in move_ops))

        return MindProposal(
            mind_name=self.name,
            objective=f"navigate to ({tr},{tc})",
            candidate_plan=plan,
            confidence=confidence,
            expected_progress=min(1.0, len(plan) / max(dist, 1)),
            expected_info_gain=0.1,
            estimated_cost=float(len(plan)),
            estimated_risk=self._path_risk(plan, inducer),
            justification={"target": (tr, tc), "distance": dist},
        )

    def _find_target(
        self, obs: GameObservation, player
    ) -> Optional[Tuple[int, int]]:
        """Find nearest salient non-player object."""
        pr, pc = player.position
        best: Optional[Tuple[int, int]] = None
        best_dist = float("inf")

        for obj in obs.objects:
            if obj.value == player.value:
                continue
            if obj.area > 20:
                continue
            r, c = int(round(obj.center[0])), int(round(obj.center[1]))
            dist = abs(r - pr) + abs(c - pc)
            if 1 < dist < best_dist:
                best = (r, c)
                best_dist = dist

        return best

    def _best_move_toward(
        self, ops: List[Operator], pr, pc, tr, tc, obs: GameObservation
    ) -> Optional[Operator]:
        """Pick the movement operator that best reduces distance."""
        best_op = None
        best_score = -999.0
        for op in ops:
            dy = op.parameters.get("dy", 0)
            dx = op.parameters.get("dx", 0)
            nr, nc = pr + dy, pc + dx
            new_dist = abs(tr - nr) + abs(tc - nc)
            old_dist = abs(tr - pr) + abs(tc - pc)
            improvement = old_dist - new_dist
            score = improvement * op.confidence - op.risk_estimate
            if score > best_score:
                best_score = score
                best_op = op
        return best_op

    def _path_risk(self, plan: List[OperatorCall], inducer: OperatorInducer) -> float:
        total_risk = 0.0
        for call in plan:
            op = inducer.operators.get(call.operator_id)
            if op:
                total_risk += op.risk_estimate
        return min(1.0, total_risk)


# =====================================================================
# ClickMind
# =====================================================================

class ClickMind(SpecialistMind):
    """Assumes: click/interaction-based gameplay with object selection."""

    name = "click"

    def propose(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        rules: RuleEngine,
    ) -> Optional[MindProposal]:
        click_ops = inducer.get_by_kind(OperatorKind.CLICK)
        if not click_ops:
            # Check if any action looks like it might be click-based
            # (has coords, moderate change rate)
            candidates = []
            for a, s in profiler.stats.items():
                if s.total_tries >= 2 and s.change_rate > 0.3:
                    disp = profiler.dominant_displacement(a)
                    if disp is None:  # not movement
                        candidates.append(a)
            if not candidates:
                return None
            # Propose raw click exploration
            action = random.choice(candidates)
            target_objs = [o for o in obs.objects if o.area <= 10]
            if not target_objs:
                return None
            target = random.choice(target_objs)
            r, c = int(round(target.center[0])), int(round(target.center[1]))
            return MindProposal(
                mind_name=self.name,
                objective=f"click object v{target.value} at ({r},{c})",
                candidate_plan=[OperatorCall(
                    operator_id=f"raw_click_{action}",
                    args={"action": action, "x": c, "y": r,
                           "target_value": target.value},
                )],
                confidence=0.3,
                expected_progress=0.2,
                expected_info_gain=0.5,
                estimated_cost=1.0,
                estimated_risk=0.1,
                justification={"target_object": target.object_id},
            )

        # Use induced click operators
        plan: List[OperatorCall] = []
        targets_hit: List[int] = []
        for op in click_ops:
            if op.confidence < 0.3:
                continue
            tv = op.parameters.get("target_value")
            if tv is not None:
                matching = [o for o in obs.objects if o.value == tv]
                for obj in matching[:3]:
                    r = int(round(obj.center[0]))
                    c = int(round(obj.center[1]))
                    plan.append(OperatorCall(
                        operator_id=op.operator_id,
                        args={"x": c, "y": r, "target_value": tv},
                    ))
                    targets_hit.append(tv)

        if not plan:
            return None

        confidence = max(op.confidence for op in click_ops if op.confidence >= 0.3)

        return MindProposal(
            mind_name=self.name,
            objective=f"click {len(plan)} targets (values: {set(targets_hit)})",
            candidate_plan=plan[:10],
            confidence=confidence,
            expected_progress=0.3 * len(plan),
            expected_info_gain=0.2,
            estimated_cost=float(len(plan)),
            estimated_risk=max(op.risk_estimate for op in click_ops),
            justification={"target_values": list(set(targets_hit))},
        )


# =====================================================================
# SequenceMind
# =====================================================================

class SequenceMind(SpecialistMind):
    """Assumes: exact short action programs are crucial.

    Strategies:
      - Replay known successful sequences
      - Generate action permutations (AB, BA, AAB, ABA, etc.)
      - Toggle sequences (repeat A→B→A→B)
      - Systematic probing of untried combinations
    """

    name = "sequence"

    def __init__(self) -> None:
        super().__init__()
        self._known_sequences: List[List[str]] = []
        self._tried_combos: set = set()

    def propose(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        rules: RuleEngine,
    ) -> Optional[MindProposal]:
        # Strategy 1: Replay known successful sequences
        if self._known_sequences:
            seq = self._known_sequences[0]
            plan = [OperatorCall(
                operator_id=f"seq_step_{i}",
                args={"action": a},
            ) for i, a in enumerate(seq)]
            return MindProposal(
                mind_name=self.name,
                objective=f"replay known sequence ({len(seq)} steps)",
                candidate_plan=plan,
                confidence=0.6,
                expected_progress=0.5,
                expected_info_gain=0.1,
                estimated_cost=float(len(seq)),
                estimated_risk=0.1,
                justification={"sequence": seq},
            )

        available = obs.available_actions
        non_reset = [a for a in available if a != "RESET"]
        if not non_reset:
            return None

        # Strategy 2: Generate action pair permutations (untried combos)
        # Find most effective individual actions
        effective = []
        for a in non_reset:
            s = profiler.stats.get(a)
            if s and s.total_tries >= 2 and s.change_rate > 0.3:
                effective.append((s.change_rate, a))
        effective.sort(reverse=True)
        top_actions = [a for _, a in effective[:4]]

        if len(top_actions) >= 2:
            # Generate short untried combos
            for i, a1 in enumerate(top_actions):
                for a2 in top_actions:
                    combo_key = f"{a1}_{a2}"
                    if combo_key not in self._tried_combos:
                        self._tried_combos.add(combo_key)
                        plan = [
                            OperatorCall(operator_id=f"seq_0", args={"action": a1}),
                            OperatorCall(operator_id=f"seq_1", args={"action": a2}),
                            OperatorCall(operator_id=f"seq_2", args={"action": a1}),
                        ]
                        return MindProposal(
                            mind_name=self.name,
                            objective=f"sequence: {a1}→{a2}→{a1}",
                            candidate_plan=plan,
                            confidence=0.35,
                            expected_progress=0.3,
                            expected_info_gain=0.4,
                            estimated_cost=3.0,
                            estimated_risk=0.15,
                            justification={"strategy": "permutation",
                                           "combo": [a1, a2, a1]},
                        )

        # Strategy 3: Systematic probe (try each action once)
        plan = [OperatorCall(
            operator_id=f"probe_{a}",
            args={"action": a},
        ) for a in non_reset[:5]]

        return MindProposal(
            mind_name=self.name,
            objective="systematic probe sequence",
            candidate_plan=plan,
            confidence=0.25,
            expected_progress=0.1,
            expected_info_gain=0.5,
            estimated_cost=float(len(plan)),
            estimated_risk=0.15,
            justification={"strategy": "systematic_probe"},
        )

    def learn_sequence(self, actions: List[str], success: bool) -> None:
        """Record a successful action sequence for future replay."""
        if success and len(actions) >= 2:
            self._known_sequences.insert(0, actions)
            self._known_sequences = self._known_sequences[:5]


# =====================================================================
# PhysicsMind
# =====================================================================

class PhysicsMind(SpecialistMind):
    """Assumes: pushing, collisions, projectiles, hazard avoidance."""

    name = "physics"

    def propose(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        rules: RuleEngine,
    ) -> Optional[MindProposal]:
        player = obs.best_player
        if player is None:
            return None

        # Check for push-like operators or rules about blocking
        move_ops = inducer.get_movement_ops()
        blocking_rules = rules.get_blocking_rules()

        if not move_ops and not blocking_rules:
            return None

        # Danger avoidance: check if we're near a hazard
        danger_rules = rules.predict_danger(obs)
        if danger_rules:
            # Find safest movement
            safe_ops = [
                op for op in move_ops
                if op.risk_estimate < 0.2 and op.confidence > 0.5
            ]
            if safe_ops:
                best = max(safe_ops, key=lambda o: o.confidence)
                return MindProposal(
                    mind_name=self.name,
                    objective="escape hazard zone",
                    candidate_plan=[OperatorCall(
                        operator_id=best.operator_id,
                        args={"reason": "hazard_escape"},
                    )],
                    confidence=best.confidence * 0.8,
                    expected_progress=0.1,
                    expected_info_gain=0.1,
                    estimated_cost=1.0,
                    estimated_risk=best.risk_estimate,
                    justification={
                        "danger_rules": [r.rule_id for r in danger_rules],
                    },
                )

        # Push hypothesis: try moving into adjacent objects
        push_targets = []
        pr, pc = player.position
        grid = obs.raw_grid
        H, W = grid.shape
        for op in move_ops:
            dy = op.parameters.get("dy", 0)
            dx = op.parameters.get("dx", 0)
            nr, nc = pr + dy, pc + dx
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) != 0:
                push_targets.append((op, nr, nc, int(grid[nr, nc])))

        if push_targets:
            op, nr, nc, val = push_targets[0]
            return MindProposal(
                mind_name=self.name,
                objective=f"push object v{val} at ({nr},{nc})",
                candidate_plan=[OperatorCall(
                    operator_id=op.operator_id,
                    args={"reason": "push_test", "target_value": val},
                )],
                confidence=0.3,
                expected_progress=0.15,
                expected_info_gain=0.4,
                estimated_cost=1.0,
                estimated_risk=op.risk_estimate,
                justification={"push_target": (nr, nc, val)},
            )

        return None


# =====================================================================
# TransformMind
# =====================================================================

class TransformMind(SpecialistMind):
    """Assumes: actions trigger global grid changes or toggles."""

    name = "transform"

    def propose(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        rules: RuleEngine,
    ) -> Optional[MindProposal]:
        transform_ops = inducer.get_by_kind(OperatorKind.GLOBAL_TRANSFORM)
        if not transform_ops:
            return None

        # Propose using the most confident transform operator
        best = max(transform_ops, key=lambda o: o.confidence)
        if best.confidence < 0.3:
            return None

        return MindProposal(
            mind_name=self.name,
            objective=f"trigger transform via {best.primitive_action}",
            candidate_plan=[OperatorCall(
                operator_id=best.operator_id,
                args={},
            )],
            confidence=best.confidence,
            expected_progress=0.25,
            expected_info_gain=0.3,
            estimated_cost=1.0,
            estimated_risk=best.risk_estimate,
            justification={"operator": best.operator_id},
        )


# =====================================================================
# ClosureMind
# =====================================================================

class ClosureMind(SpecialistMind):
    """Hypothesis-driven completion specialist.

    Activates when terminal-proxy conditions suggest the level CAN be finished.
    Generates specific end-condition hypotheses and tests them, rather than
    just repeating whatever other minds do.

    Trigger conditions (any):
      - Few small objects remain (≤5)
      - High structural progress but local plateauing
      - High-confidence ops + many actions without completion
      - Object class nearly exhausted

    Strategies:
      A. Eliminate remaining — click/interact with every remaining small object
      B. Reach unique regions — navigate to unexplored/unique areas
      C. Transform sequence — apply all validated transforms
      D. Terminal probes — try untested actions in the current state
    """

    name = "closure"

    def propose(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        inducer: OperatorInducer,
        rules: RuleEngine,
    ) -> Optional[MindProposal]:
        # Compute activation signals
        validated = inducer.num_validated(min_uses=2, min_accuracy=0.4)
        small_objects = [o for o in obs.objects if o.area <= 15 and o.value != 0]
        n_small = len(small_objects)

        # Activation conditions: at least one must hold
        few_targets = n_small <= 5 and n_small > 0
        has_mechanics = validated >= 2
        high_confidence = inducer.best_validated_confidence() >= 0.7

        if not (few_targets or (has_mechanics and high_confidence)):
            return None

        # Strategy A: Eliminate remaining small objects (highest priority when few remain)
        if few_targets:
            plan = self._eliminate_remaining(obs, small_objects, inducer)
            if plan:
                return MindProposal(
                    mind_name=self.name,
                    objective=f"closure: eliminate {n_small} remaining targets",
                    candidate_plan=plan,
                    confidence=0.7 if has_mechanics else 0.4,
                    expected_progress=0.6,
                    expected_info_gain=0.2,
                    estimated_cost=float(len(plan)),
                    estimated_risk=0.1,
                    justification={"strategy": "eliminate_remaining",
                                   "remaining": n_small},
                )

        # Strategy B: Navigate to unique/furthest objects
        player = obs.best_player
        move_ops = inducer.get_movement_ops()
        if player and len(move_ops) >= 2:
            plan = self._reach_and_interact(obs, player, move_ops, inducer)
            if plan:
                return MindProposal(
                    mind_name=self.name,
                    objective=f"closure: reach+interact ({len(plan)} steps)",
                    candidate_plan=plan,
                    confidence=min(0.7, max(op.confidence for op in move_ops)),
                    expected_progress=0.4,
                    expected_info_gain=0.2,
                    estimated_cost=float(len(plan)),
                    estimated_risk=0.15,
                    justification={"strategy": "reach_and_interact"},
                )

        # Strategy C: Apply all validated transforms in sequence
        transform_ops = [
            op for op in inducer.operators.values()
            if op.kind == OperatorKind.GLOBAL_TRANSFORM
            and op.confidence >= 0.4
        ]
        if transform_ops:
            plan = [
                OperatorCall(operator_id=op.operator_id, args={"option": "closure"})
                for op in sorted(transform_ops, key=lambda o: -o.confidence)
            ]
            return MindProposal(
                mind_name=self.name,
                objective=f"closure: apply {len(plan)} transforms",
                candidate_plan=plan[:5],
                confidence=0.4,
                expected_progress=0.3,
                expected_info_gain=0.3,
                estimated_cost=float(len(plan)),
                estimated_risk=0.1,
                justification={"strategy": "transform_sequence"},
            )

        # Strategy D: Terminal probes — try each untested action
        untried = profiler.least_tried_actions(obs.available_actions, top_k=3)
        if untried:
            plan = [
                OperatorCall(
                    operator_id=f"closure_probe_{a}",
                    args={"action": a, "option": "terminal_probe"},
                )
                for a in untried
            ]
            return MindProposal(
                mind_name=self.name,
                objective=f"closure: probe {len(untried)} untested actions",
                candidate_plan=plan,
                confidence=0.25,
                expected_progress=0.2,
                expected_info_gain=0.5,
                estimated_cost=float(len(plan)),
                estimated_risk=0.2,
                justification={"strategy": "terminal_probe"},
            )

        return None

    def _eliminate_remaining(
        self,
        obs: GameObservation,
        small_objects: list,
        inducer: OperatorInducer,
    ) -> List[OperatorCall]:
        """Build a plan to interact with every remaining small object."""
        plan: List[OperatorCall] = []
        click_ops = inducer.get_by_kind(OperatorKind.CLICK)
        player = obs.best_player

        # First: use click operators on matching objects
        for op in click_ops:
            tv = op.parameters.get("target_value")
            if tv is None or op.confidence < 0.3:
                continue
            matching = [o for o in small_objects if o.value == tv]
            for obj in matching:
                r = int(round(obj.center[0]))
                c = int(round(obj.center[1]))
                plan.append(OperatorCall(
                    operator_id=op.operator_id,
                    args={"x": c, "y": r, "target_value": tv,
                           "option": "closure_click"},
                ))

        # For objects not covered by click ops, navigate then interact
        clicked_values = {op.parameters.get("target_value") for op in click_ops}
        move_ops = inducer.get_movement_ops()
        uncovered = [o for o in small_objects if o.value not in clicked_values]

        if player and len(move_ops) >= 2 and uncovered:
            pr, pc = player.position
            for obj in uncovered[:3]:
                r, c = int(round(obj.center[0])), int(round(obj.center[1]))
                # Navigate toward target
                for _ in range(min(8, abs(r - pr) + abs(c - pc))):
                    best_op = None
                    best_score = -999.0
                    for op in move_ops:
                        dy = op.parameters.get("dy", 0)
                        dx = op.parameters.get("dx", 0)
                        new_d = abs(r - (pr + dy)) + abs(c - (pc + dx))
                        old_d = abs(r - pr) + abs(c - pc)
                        improvement = old_d - new_d
                        score = improvement * op.confidence
                        if score > best_score:
                            best_score = score
                            best_op = op
                    if best_op is None or best_score <= 0:
                        break
                    plan.append(OperatorCall(
                        operator_id=best_op.operator_id,
                        args={"option": "closure_nav"},
                    ))
                    pr += best_op.parameters.get("dy", 0)
                    pc += best_op.parameters.get("dx", 0)

                # Try any interaction operator
                interact_ops = [
                    op for op in inducer.operators.values()
                    if op.kind in (OperatorKind.CLICK, OperatorKind.INTERACT,
                                   OperatorKind.GLOBAL_TRANSFORM)
                    and op.confidence >= 0.3
                ]
                if interact_ops:
                    best = max(interact_ops, key=lambda o: o.confidence)
                    plan.append(OperatorCall(
                        operator_id=best.operator_id,
                        args={"option": "closure_interact",
                               "x": c, "y": r, "target_value": obj.value},
                    ))

        return plan[:15]

    def _reach_and_interact(
        self,
        obs: GameObservation,
        player,
        move_ops: List[Operator],
        inducer: OperatorInducer,
    ) -> List[OperatorCall]:
        """Navigate to interesting objects, then interact."""
        pr, pc = player.position

        # Find furthest small object (closure targets what's left)
        targets = []
        for obj in obs.objects:
            if obj.value == player.value:
                continue
            if obj.area > 20:
                continue
            r, c = int(round(obj.center[0])), int(round(obj.center[1]))
            dist = abs(r - pr) + abs(c - pc)
            if dist >= 1:
                targets.append((dist, r, c, obj))

        if not targets:
            return []

        targets.sort(reverse=True)  # furthest first
        plan: List[OperatorCall] = []
        cur_r, cur_c = pr, pc
        for _, tr, tc, obj in targets[:3]:
            for _ in range(min(10, abs(tr - cur_r) + abs(tc - cur_c))):
                best_op = None
                best_score = -999.0
                for op in move_ops:
                    dy = op.parameters.get("dy", 0)
                    dx = op.parameters.get("dx", 0)
                    new_d = abs(tr - (cur_r + dy)) + abs(tc - (cur_c + dx))
                    old_d = abs(tr - cur_r) + abs(tc - cur_c)
                    improvement = old_d - new_d
                    score = improvement * op.confidence
                    if score > best_score:
                        best_score = score
                        best_op = op
                if best_op is None or best_score <= 0:
                    break
                plan.append(OperatorCall(
                    operator_id=best_op.operator_id,
                    args={"option": "closure_nav"},
                ))
                cur_r += best_op.parameters.get("dy", 0)
                cur_c += best_op.parameters.get("dx", 0)
                if cur_r == tr and cur_c == tc:
                    break

            interact_ops = [
                op for op in inducer.operators.values()
                if op.kind in (OperatorKind.CLICK, OperatorKind.INTERACT,
                               OperatorKind.GLOBAL_TRANSFORM)
                and op.confidence >= 0.3
            ]
            if interact_ops:
                best = max(interact_ops, key=lambda o: o.confidence)
                plan.append(OperatorCall(
                    operator_id=best.operator_id,
                    args={"option": "closure_interact",
                           "x": tc, "y": tr, "target_value": obj.value},
                ))

        return plan[:15]


# =====================================================================
# Factory
# =====================================================================

def create_default_minds() -> List[SpecialistMind]:
    """Create the default set of specialist minds."""
    return [
        NavigatorMind(),
        ClickMind(),
        SequenceMind(),
        PhysicsMind(),
        TransformMind(),
        ClosureMind(),
    ]
