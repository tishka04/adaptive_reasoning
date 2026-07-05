"""
Goal Pursuit Controller — hypothesis-testing loop for ARC-AGI-3.

Manages a **bank of plausible game objectives**, each with measurable
progress signals.  Strategies are judged as interventions in service of
a specific goal, not in the abstract.

Control hierarchy:
  outer loop  — rotate through goal hypotheses
  inner loop  — retry multiple strategies per goal (up to MAX_ATTEMPTS)
  execution   — actioner runs the strategy, progress is measured

The GoalProgressManager is the executive layer between the reasoning
loop and the actioner.  It decides:
  - which goal to pursue
  - when to switch strategies
  - when to abandon a goal
  - when to regenerate the entire goal bank
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
class ObjectiveStatus(Enum):
    """Lifecycle status of a goal hypothesis."""
    ACTIVE = "active"           # currently being pursued
    SUSPENDED = "suspended"     # deprioritised after failures, may revisit
    REJECTED = "rejected"       # enough evidence it's wrong
    CONFIRMED = "confirmed"     # achieved / strong evidence it's right


@dataclass
class ProgressSignal:
    """One measurable indicator of progress toward a goal."""
    name: str                    # e.g. "player_distance_to_green_decreasing"
    description: str             # NL for LLM context
    direction: str = "increase"  # "increase" = higher is better, "decrease" = lower
    weight: float = 1.0          # relative importance in composite score


@dataclass
class GameObjective:
    """A first-class game objective hypothesis.

    The LLM produces several of these.  Each carries its own notion of
    what progress looks like, enabling goal-relative evaluation.
    """
    id: str                                     # e.g. "obj_navigate_green"
    description: str                            # NL: "Navigate player to the green cell"
    success_condition: str                      # NL: "Player position overlaps green cell"
    progress_signals: List[ProgressSignal] = field(default_factory=list)
    anti_signals: List[str] = field(default_factory=list)  # things that indicate wrong direction
    confidence: float = 0.5                     # LLM's initial confidence
    status: ObjectiveStatus = ObjectiveStatus.ACTIVE

    # ── Tracking (updated during pursuit) ──
    attempts: int = 0                           # strategy cycles tried
    best_progress: float = 0.0                  # best progress score achieved
    total_progress_delta: float = 0.0           # cumulative progress across attempts
    strategy_history: List[StrategyOutcome] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in (ObjectiveStatus.REJECTED, ObjectiveStatus.CONFIRMED)

    @property
    def avg_progress(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_progress_delta / self.attempts

    def to_prompt(self) -> str:
        """Render for LLM context."""
        lines = [
            f"**Objective [{self.id}]** ({self.status.value}, "
            f"confidence={self.confidence:.0%}, attempts={self.attempts})",
            f"  Description: {self.description}",
            f"  Success: {self.success_condition}",
        ]
        if self.progress_signals:
            sigs = ", ".join(s.name for s in self.progress_signals)
            lines.append(f"  Progress signals: {sigs}")
        if self.anti_signals:
            lines.append(f"  Anti-signals: {', '.join(self.anti_signals)}")
        if self.strategy_history:
            last = self.strategy_history[-1]
            lines.append(
                f"  Last attempt: {last.strategy_description[:60]} "
                f"→ progress={last.progress_score:.2f} ({last.terminal_status})"
            )
        return "\n".join(lines)


@dataclass
class GoalContext:
    """Compact planning context used by the trajectory sampler."""

    goal_family: str
    objective_id: str
    progress_signals: List[str] = field(default_factory=list)
    anti_signals: List[str] = field(default_factory=list)
    source_confidence: float = 0.5
    human_prior_weight: float = 0.0
    preferred_actions: List[str] = field(default_factory=list)
    click_targets: List[Dict[str, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyOutcome:
    """Full record of one goal-conditioned strategy attempt.

    Stored in memory for retrieval and cross-game transfer.
    """
    goal_id: str
    goal_description: str
    strategy_description: str
    strategy_type: str = "unknown"              # StrategyType.value
    action_sequence: List[Tuple[int, Optional[dict]]] = field(default_factory=list)
    initial_state_hash: int = 0                  # grid hash at start
    predicted_outcome: Optional[str] = None      # NL predicted by JEPA
    observed_outcome: Optional[str] = None       # NL actual result
    progress_score: float = 0.0                  # composite [0, 1]
    progress_components: Dict[str, float] = field(default_factory=dict)
    terminal_status: str = "pending"             # success / partial / failure / game_over
    actions_taken: int = 0
    time_spent: float = 0.0
    levels_before: int = 0
    levels_after: int = 0
    game_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_desc": self.goal_description[:80],
            "strategy": self.strategy_description[:80],
            "strategy_type": self.strategy_type,
            "progress": self.progress_score,
            "status": self.terminal_status,
            "actions": self.actions_taken,
            "time": round(self.time_spent, 2),
        }


@dataclass
class TrajectoryOutcome:
    """Observed result of executing the first prefix of a sampled trajectory."""

    prefix_executed: List[Dict[str, Any]] = field(default_factory=list)
    observed_after: Optional[np.ndarray] = None
    progress_delta: float = 0.0
    prediction_match: float = 0.0
    game_over: bool = False
    levels_delta: int = 0
    source: str = "random"
    goal_context: GoalContext = field(
        default_factory=lambda: GoalContext("unknown", "unknown")
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Progress thresholds
# ------------------------------------------------------------------
SUCCESS_THRESHOLD = 0.70        # strategy likely achieved the objective
PARTIAL_THRESHOLD = 0.15        # some meaningful movement toward goal
SIGNIFICANT_THRESHOLD = 0.10    # after N attempts, was anything meaningful?
MAX_ATTEMPTS_PER_GOAL = 3       # inner loop budget (kept tight so switching happens)


# ------------------------------------------------------------------
# Goal Progress Manager
# ------------------------------------------------------------------
class GoalProgressManager:
    """Executive controller for goal-directed behaviour.

    Responsibilities:
      - Maintain the goal bank (multiple hypotheses)
      - Track per-goal attempts and progress
      - Compute goal-relative progress scores
      - Decide: refine / switch strategy / switch goal / regenerate bank
      - Expose summaries to memory and the main loop
    """

    def __init__(self, max_goals: int = 6, max_attempts: int = MAX_ATTEMPTS_PER_GOAL):
        self.max_goals = max_goals
        self.max_attempts = max_attempts

        # ── Goal bank ──
        self._goals: List[GameObjective] = []
        self._active_goal_idx: int = -1          # index into _goals
        self._bank_generation: int = 0           # how many times the bank was (re)generated

        # ── Strategy tracking ──
        self._current_outcome: Optional[StrategyOutcome] = None
        self._all_outcomes: List[StrategyOutcome] = []
        self._trajectory_outcomes: List[TrajectoryOutcome] = []

        # ── Snapshot for progress measurement ──
        self._snapshot_grid: Optional[np.ndarray] = None
        self._snapshot_levels: int = 0
        self._snapshot_states: int = 0
        self._snapshot_time: float = 0.0

    # ==================================================================
    # Goal bank management
    # ==================================================================
    @property
    def has_goals(self) -> bool:
        return len(self._goals) > 0

    @property
    def active_goal(self) -> Optional[GameObjective]:
        if 0 <= self._active_goal_idx < len(self._goals):
            return self._goals[self._active_goal_idx]
        return None

    @property
    def goals(self) -> List[GameObjective]:
        return self._goals

    @property
    def bank_generation(self) -> int:
        return self._bank_generation

    def set_goal_bank(self, objectives: List[GameObjective]) -> None:
        """Replace the goal bank with new hypotheses from the LLM."""
        self._goals = objectives[:self.max_goals]
        self._bank_generation += 1
        # Activate highest-confidence goal
        if self._goals:
            best_idx = max(range(len(self._goals)),
                           key=lambda i: self._goals[i].confidence)
            self._active_goal_idx = best_idx
            self._goals[best_idx].status = ObjectiveStatus.ACTIVE
            logger.info(
                f"Goal bank gen {self._bank_generation}: "
                f"{len(self._goals)} objectives, "
                f"active='{self._goals[best_idx].description[:60]}'"
            )
        else:
            self._active_goal_idx = -1

    def select_next_goal(self) -> Optional[GameObjective]:
        """Switch to the next best non-terminal goal.

        Returns the newly activated goal, or None if all exhausted.
        """
        # Suspend current goal
        cur = self.active_goal
        if cur is not None and not cur.is_terminal:
            cur.status = ObjectiveStatus.SUSPENDED

        # Find best candidate: highest confidence among non-terminal
        candidates = [
            (i, g) for i, g in enumerate(self._goals)
            if not g.is_terminal
        ]
        if not candidates:
            self._active_goal_idx = -1
            return None

        # Prefer: untried goals first, then highest confidence
        candidates.sort(key=lambda ig: (-int(ig[1].attempts == 0), -ig[1].confidence))
        best_idx, best = candidates[0]
        self._active_goal_idx = best_idx
        best.status = ObjectiveStatus.ACTIVE
        logger.info(f"Switched to goal '{best.description[:60]}' (attempts={best.attempts})")
        return best

    def all_goals_exhausted(self) -> bool:
        """True if every goal is terminal or has exceeded max attempts with no progress."""
        for g in self._goals:
            if g.is_terminal:
                continue
            if g.attempts < self.max_attempts:
                return False
            if g.best_progress >= SIGNIFICANT_THRESHOLD:
                return False  # had some progress, worth revisiting
        return True

    # ==================================================================
    # Strategy execution lifecycle
    # ==================================================================
    def begin_strategy(
        self,
        strategy_description: str,
        strategy_type: str,
        grid: Optional[np.ndarray],
        levels_completed: int,
        states_visited: int,
        game_id: str = "",
    ) -> None:
        """Call before executing a strategy. Snapshots the state."""
        goal = self.active_goal
        if goal is None:
            return

        self._snapshot_grid = grid.copy() if grid is not None else None
        self._snapshot_levels = levels_completed
        self._snapshot_states = states_visited
        self._snapshot_time = time.time()

        self._current_outcome = StrategyOutcome(
            goal_id=goal.id,
            goal_description=goal.description,
            strategy_description=strategy_description,
            strategy_type=strategy_type,
            initial_state_hash=hash(grid.tobytes()) if grid is not None else 0,
            levels_before=levels_completed,
            game_id=game_id,
        )

    def end_strategy(
        self,
        grid_after: Optional[np.ndarray],
        levels_completed: int,
        states_visited: int,
        game_over: bool,
        action_sequence: Optional[List[Tuple[int, Optional[dict]]]] = None,
    ) -> StrategyOutcome:
        """Call after strategy execution. Measures progress and returns full outcome."""
        goal = self.active_goal
        outcome = self._current_outcome
        if outcome is None or goal is None:
            # No active tracking — return dummy
            return StrategyOutcome(
                goal_id="none",
                goal_description="",
                strategy_description="",
                strategy_type="unknown",
            )

        elapsed = time.time() - self._snapshot_time
        outcome.time_spent = elapsed
        outcome.levels_after = levels_completed
        outcome.actions_taken = len(action_sequence) if action_sequence else 0
        if action_sequence:
            outcome.action_sequence = action_sequence

        # ── Compute progress ──
        progress, components = self._measure_progress(
            goal=goal,
            grid_before=self._snapshot_grid,
            grid_after=grid_after,
            levels_before=self._snapshot_levels,
            levels_after=levels_completed,
            states_before=self._snapshot_states,
            states_after=states_visited,
            game_over=game_over,
        )
        outcome.progress_score = progress
        outcome.progress_components = components

        # ── Classify terminal status ──
        if game_over:
            outcome.terminal_status = "game_over"
        elif progress >= SUCCESS_THRESHOLD:
            outcome.terminal_status = "success"
        elif progress >= PARTIAL_THRESHOLD:
            outcome.terminal_status = "partial"
        else:
            outcome.terminal_status = "failure"

        # ── Update goal tracking ──
        goal.attempts += 1
        goal.total_progress_delta += progress
        if progress > goal.best_progress:
            goal.best_progress = progress
        goal.strategy_history.append(outcome)

        # ── Store globally ──
        self._all_outcomes.append(outcome)
        self._current_outcome = None

        logger.info(
            f"Strategy outcome [{goal.id}]: "
            f"progress={progress:.2f} ({outcome.terminal_status}), "
            f"components={components}"
        )
        return outcome

    def measure_goal_context_progress(
        self,
        goal_context: GoalContext,
        *,
        grid_before: Optional[np.ndarray],
        grid_after: Optional[np.ndarray],
        levels_before: int,
        levels_after: int,
        states_before: int,
        states_after: int,
        game_over: bool,
    ) -> tuple[float, Dict[str, float]]:
        """Measure progress for a sampler-facing GoalContext."""
        signals = [
            ProgressSignal(name=name, description=name)
            for name in goal_context.progress_signals
        ]
        goal = GameObjective(
            id=goal_context.objective_id,
            description=goal_context.objective_id,
            success_condition=goal_context.objective_id,
            progress_signals=signals,
            anti_signals=list(goal_context.anti_signals),
            confidence=goal_context.source_confidence,
        )
        return self._measure_progress(
            goal=goal,
            grid_before=grid_before,
            grid_after=grid_after,
            levels_before=levels_before,
            levels_after=levels_after,
            states_before=states_before,
            states_after=states_after,
            game_over=game_over,
        )

    def record_trajectory_outcome(self, outcome: TrajectoryOutcome) -> None:
        """Store trajectory-level credit assignment records."""
        self._trajectory_outcomes.append(outcome)
        if len(self._trajectory_outcomes) > 200:
            self._trajectory_outcomes = self._trajectory_outcomes[-200:]

    # ==================================================================
    # Progress measurement
    # ==================================================================
    def _measure_progress(
        self,
        goal: GameObjective,
        grid_before: Optional[np.ndarray],
        grid_after: Optional[np.ndarray],
        levels_before: int,
        levels_after: int,
        states_before: int,
        states_after: int,
        game_over: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute goal-relative progress as a weighted composite score.

        Weights are calibrated so that:
          - Level completion alone → ~0.90 (clear success)
          - Good state change + novelty → ~0.20-0.35 (crosses partial threshold)
          - State change alone → ~0.10-0.20
          - Goal heuristic can boost partial → success range

        Returns (composite_score, component_dict).
        """
        components: Dict[str, float] = {}

        # ── 1. Level completion (strongest signal) ──
        level_delta = levels_after - levels_before
        p_level = min(1.0, level_delta) if level_delta > 0 else 0.0
        components["level"] = p_level

        # ── 2. State change evidence ──
        p_state = 0.0
        if grid_before is not None and grid_after is not None:
            diff_cells = int(np.sum(grid_before != grid_after))
            total_cells = max(1, grid_before.size)
            change_frac = diff_cells / total_cells
            if change_frac > 0.001:
                # Scale: 1% change → 0.3, 5% → 0.7, 10%+ → 1.0
                p_state = min(1.0, change_frac * 12)
                # Penalise massive changes (>50%) — likely a reset
                if change_frac > 0.5:
                    p_state = 0.15
        components["state_change"] = p_state

        # ── 3. State novelty ──
        new_states = states_after - states_before
        # 1 new state → 0.35, 3 → 0.75, 5+ → 1.0
        p_novelty = min(1.0, new_states / 3.0) if new_states > 0 else 0.0
        components["novelty"] = p_novelty

        # ── 4. Danger / failure penalty ──
        p_danger = 1.0 if game_over else 0.0
        components["danger"] = p_danger

        # ── 5. Player movement (if player is trackable) ──
        p_player = self._player_progress(grid_before, grid_after)
        components["player_movement"] = p_player

        # ── 6. Goal-specific heuristic ──
        p_goal = self._goal_heuristic(goal, grid_before, grid_after)
        components["goal_heuristic"] = p_goal

        # ── Weighted composite ──
        # Brutally simple: level completion dominates.
        # ARC rewards clear breakthroughs, not smooth progress.
        if p_level > 0:
            # Level completion = near-certain success
            composite = 0.90 + 0.10 * p_state
        elif p_danger > 0:
            # Game over = clear failure, small credit for state change
            composite = 0.05 * p_state
        else:
            # No level, no death: did something new happen?
            # Binary-ish: either the grid changed meaningfully or it didn't
            composite = (
                0.30 * p_state       # did the grid change?
                + 0.15 * p_novelty   # did we see a new state?
                + 0.15 * p_player    # did the player move?
                + 0.15 * p_goal      # goal-specific heuristic
            )

        composite = max(0.0, min(1.0, composite))

        return composite, components

    def _player_progress(
        self,
        grid_before: Optional[np.ndarray],
        grid_after: Optional[np.ndarray],
    ) -> float:
        """Detect if a likely player object moved.

        Finds the smallest non-zero region (likely player) and checks
        if its centroid shifted.  Returns [0, 1].
        """
        if grid_before is None or grid_after is None:
            return 0.0

        # Find unique nonzero values and their positions
        vals_before = set(np.unique(grid_before)) - {0}
        vals_after = set(np.unique(grid_after)) - {0}
        common_vals = vals_before & vals_after

        if not common_vals:
            return 0.0

        best_movement = 0.0
        for v in common_vals:
            pos_b = np.argwhere(grid_before == v)
            pos_a = np.argwhere(grid_after == v)
            if len(pos_b) == 0 or len(pos_a) == 0:
                continue
            # Only track small objects (likely player, not background)
            if len(pos_b) > 20 or len(pos_a) > 20:
                continue
            cent_b = pos_b.mean(axis=0)
            cent_a = pos_a.mean(axis=0)
            dist = float(np.abs(cent_b - cent_a).sum())
            if dist > 0.5:
                # Any movement is good; scale by distance
                movement = min(1.0, dist / 5.0)
                best_movement = max(best_movement, movement)

        return best_movement

    def _goal_heuristic(
        self,
        goal: GameObjective,
        grid_before: Optional[np.ndarray],
        grid_after: Optional[np.ndarray],
    ) -> float:
        """Goal-specific progress heuristic based on the objective's signals.

        Interprets signal names using keyword matching.  Returns [0, 1].
        """
        if not goal.progress_signals or grid_before is None or grid_after is None:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0
        for sig in goal.progress_signals:
            s = self._evaluate_signal(sig, grid_before, grid_after)
            if s is not None:
                weighted_sum += s * sig.weight
                weight_total += sig.weight

        if weight_total == 0:
            return 0.0
        return max(0.0, min(1.0, weighted_sum / weight_total))

    def _evaluate_signal(
        self,
        signal: ProgressSignal,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
    ) -> Optional[float]:
        """Evaluate one progress signal.  Returns [0, 1] or None."""
        name = signal.name.lower()

        # ── Player distance to target ──
        if "player" in name and "distance" in name and "decreas" in name:
            return self._distance_progress(grid_before, grid_after)

        # ── Player position changed ──
        if "player" in name and ("position" in name or "moved" in name):
            return self._player_progress(grid_before, grid_after)

        # ── Player visits / reaches positions ──
        if "player" in name and ("visit" in name or "reach" in name):
            return self._player_progress(grid_before, grid_after)

        # ── Color-specific count decrease (e.g. "color3_count_decreasing") ──
        if "count" in name and ("decreas" in name or "fewer" in name):
            return self._color_count_change(name, grid_before, grid_after, decrease=True)

        # ── Color-specific count increase ──
        if "count" in name and ("increas" in name or "more" in name):
            return self._color_count_change(name, grid_before, grid_after, decrease=False)

        # ── Grid cell change ──
        if "grid" in name and "change" in name or "cell" in name and "change" in name:
            diff = int(np.sum(grid_before != grid_after))
            if diff == 0:
                return 0.0
            # 1 cell → 0.2, 5 cells → 0.6, 10+ → 1.0
            return min(1.0, diff / 8.0)

        # ── Cumulative grid change (progressive transformation) ──
        if "cumulative" in name and "change" in name:
            diff = int(np.sum(grid_before != grid_after))
            return min(1.0, diff / 6.0) if diff > 0 else 0.0

        # ── Unique states increasing ──
        if "unique" in name and "state" in name or "states" in name and "increas" in name:
            # We don't have direct state count here — use grid uniqueness as proxy
            unique_before = len(np.unique(grid_before))
            unique_after = len(np.unique(grid_after))
            diff = int(np.sum(grid_before != grid_after))
            if diff > 0:
                return min(1.0, diff / 5.0)
            return 0.0

        # ── Object positions changed ──
        if "object" in name and ("position" in name or "moved" in name):
            return self._object_movement(grid_before, grid_after, large=True)

        # ── General distance decrease ──
        if "distance" in name and "decreas" in name:
            return self._distance_progress(grid_before, grid_after)

        # ── New regions / reachable ──
        if "region" in name or "reachable" in name:
            unique_before = len(np.unique(grid_before))
            unique_after = len(np.unique(grid_after))
            if unique_after > unique_before:
                return min(1.0, (unique_after - unique_before) / 3)
            return 0.0

        # ── Survival duration ──
        if "survival" in name or "alive" in name:
            # Can't measure directly; proxy via grid not identical (still alive)
            if not np.array_equal(grid_before, grid_after):
                return 0.5
            return 0.1  # still alive but stuck

        # ── Fallback: any grid change is a weak positive signal ──
        diff = int(np.sum(grid_before != grid_after))
        if diff > 0:
            return min(1.0, diff / 10.0)
        return 0.0

    def _color_count_change(
        self,
        signal_name: str,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
        decrease: bool,
    ) -> float:
        """Measure count change for a specific color extracted from signal name."""
        import re
        # Try to extract color value from signal name (e.g. "color3_count_decreasing")
        m = re.search(r"color\s*(\d+)", signal_name)
        if m:
            target_val = int(m.group(1))
            count_before = int(np.sum(grid_before == target_val))
            count_after = int(np.sum(grid_after == target_val))
        else:
            # Generic: count all nonzero cells
            count_before = int(np.count_nonzero(grid_before))
            count_after = int(np.count_nonzero(grid_after))

        if decrease:
            delta = count_before - count_after
        else:
            delta = count_after - count_before

        if delta <= 0:
            return 0.0
        # 1 cell change → 0.3, 3 → 0.7, 5+ → 1.0
        return min(1.0, delta / max(count_before, 1) * 5 + 0.1)

    def _object_movement(
        self,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
        large: bool = True,
    ) -> float:
        """Detect if non-player objects changed position."""
        vals_before = set(np.unique(grid_before)) - {0}
        vals_after = set(np.unique(grid_after)) - {0}
        common = vals_before & vals_after

        max_move = 0.0
        for v in common:
            pos_b = np.argwhere(grid_before == v)
            pos_a = np.argwhere(grid_after == v)
            # Skip very small regions (likely player) unless large=False
            if large and (len(pos_b) < 2 or len(pos_a) < 2):
                continue
            if len(pos_b) == 0 or len(pos_a) == 0:
                continue
            cent_b = pos_b.mean(axis=0)
            cent_a = pos_a.mean(axis=0)
            dist = float(np.abs(cent_b - cent_a).sum())
            if dist > 0.5:
                max_move = max(max_move, min(1.0, dist / 5.0))

        return max_move

    def _distance_progress(
        self, grid_before: np.ndarray, grid_after: np.ndarray
    ) -> float:
        """Detect if a small object (player) moved closer to other objects."""
        vals = set(np.unique(grid_before)) - {0}
        if len(vals) < 2:
            return 0.0

        # Find the smallest value region (likely player)
        sizes = {}
        for v in vals:
            sizes[v] = int(np.sum(grid_before == v))
        player_val = min(sizes, key=sizes.get)
        other_vals = [v for v in vals if v != player_val]

        # Player centroid before/after
        pp_b = np.argwhere(grid_before == player_val)
        pp_a = np.argwhere(grid_after == player_val)
        if len(pp_b) == 0 or len(pp_a) == 0:
            return 0.0

        pcent_b = pp_b.mean(axis=0)
        pcent_a = pp_a.mean(axis=0)

        # Did player move?
        player_moved = float(np.abs(pcent_b - pcent_a).sum())
        if player_moved < 0.5:
            return 0.0

        # Check distance to nearest non-player object
        best_improvement = 0.0
        for ov in other_vals:
            op = np.argwhere(grid_before == ov)
            if len(op) == 0:
                continue
            ocent = op.mean(axis=0)
            dist_before = float(np.abs(pcent_b - ocent).sum())
            dist_after = float(np.abs(pcent_a - ocent).sum())
            if dist_before > 0 and dist_after < dist_before:
                improvement = (dist_before - dist_after) / dist_before
                best_improvement = max(best_improvement, improvement)

        # Scale: 10% closer → 0.3, 30% closer → 0.7, 50%+ → 1.0
        return min(1.0, best_improvement * 2.5)

    # ==================================================================
    # Decision logic
    # ==================================================================
    def should_switch_strategy(self, outcome: StrategyOutcome) -> bool:
        """Should we try a different strategy for the same goal?"""
        return outcome.terminal_status in ("failure", "game_over")

    def should_switch_goal(self) -> bool:
        """Should we abandon the current goal and try another?"""
        goal = self.active_goal
        if goal is None:
            return True
        if goal.attempts >= self.max_attempts:
            if goal.best_progress < SIGNIFICANT_THRESHOLD:
                logger.info(
                    f"Goal '{goal.id}' exhausted ({goal.attempts} attempts, "
                    f"best_progress={goal.best_progress:.2f}) → switching"
                )
                goal.status = ObjectiveStatus.REJECTED
                return True
        return False

    def should_regenerate_bank(self) -> bool:
        """Should we ask the LLM for entirely new goal hypotheses?"""
        return self.all_goals_exhausted()

    def decide_next_action(self, outcome: Optional[StrategyOutcome] = None) -> str:
        """High-level decision after a strategy completes.

        Returns one of:
          "continue"       — keep refining same goal+strategy
          "new_strategy"   — try different strategy, same goal
          "new_goal"       — switch to another goal in the bank
          "regenerate"     — all goals exhausted, need new bank
        """
        if outcome is not None and outcome.terminal_status == "success":
            return "continue"

        if self.should_regenerate_bank():
            return "regenerate"

        if self.should_switch_goal():
            return "new_goal"

        if outcome is not None and self.should_switch_strategy(outcome):
            return "new_strategy"

        # Partial progress — try refining
        if outcome is not None and outcome.progress_score >= PARTIAL_THRESHOLD:
            return "continue"

        return "new_strategy"

    # ==================================================================
    # Context for strategy generation
    # ==================================================================
    def strategy_context(self) -> Dict[str, Any]:
        """Return context for goal-conditioned strategy generation.

        This tells the strategy generator:
          - What goal we're pursuing
          - What we've already tried and how it went
          - What to avoid
        """
        goal = self.active_goal
        if goal is None:
            return {}

        failed_strategies = [
            so.strategy_description
            for so in goal.strategy_history
            if so.terminal_status in ("failure", "game_over")
        ]
        partial_strategies = [
            (so.strategy_description, so.progress_score)
            for so in goal.strategy_history
            if so.terminal_status == "partial"
        ]

        return {
            "goal_id": goal.id,
            "goal_description": goal.description,
            "success_condition": goal.success_condition,
            "progress_signals": [s.name for s in goal.progress_signals],
            "anti_signals": goal.anti_signals,
            "attempts": goal.attempts,
            "best_progress": goal.best_progress,
            "failed_strategies": failed_strategies[-5:],  # last 5
            "partial_strategies": partial_strategies[-3:],
        }

    # ==================================================================
    # Stats and serialization
    # ==================================================================
    def stats(self) -> Dict[str, Any]:
        goal = self.active_goal
        return {
            "bank_size": len(self._goals),
            "bank_generation": self._bank_generation,
            "active_goal": goal.id if goal else None,
            "active_goal_desc": goal.description[:60] if goal else None,
            "active_goal_attempts": goal.attempts if goal else 0,
            "active_goal_best_progress": goal.best_progress if goal else 0.0,
            "total_outcomes": len(self._all_outcomes),
            "trajectory_outcomes": len(self._trajectory_outcomes),
            "goals_rejected": sum(1 for g in self._goals if g.status == ObjectiveStatus.REJECTED),
            "goals_confirmed": sum(1 for g in self._goals if g.status == ObjectiveStatus.CONFIRMED),
        }

    def goals_for_prompt(self) -> str:
        """Render goal bank for LLM context."""
        if not self._goals:
            return "No goals generated yet."
        lines = [f"## Goal Bank (generation {self._bank_generation})"]
        for g in self._goals:
            lines.append(g.to_prompt())
        return "\n\n".join(lines)
