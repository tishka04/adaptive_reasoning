"""
Strategy router for ARC-AGI-3 — adapted from v4_1 EBM router.

Selects among high-level game strategies:
  - systematic_explore: try each action to learn what it does
  - directed_explore:   move toward unvisited areas
  - goal_seek:          navigate toward a detected objective
  - pattern_exploit:    repeat a learned successful pattern
  - random_explore:     random actions to break out of loops
  - retry:              after GAME_OVER, try a different approach

Uses rule-based scoring first (Phase 2); can be upgraded to learned
EBM scoring later (Phase 4+).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .game_memory import GameMemory
from .grid_analyzer import GridAnalyzer


class Strategy(Enum):
    SYSTEMATIC_EXPLORE = "systematic_explore"
    DIRECTED_EXPLORE = "directed_explore"
    CLICK_EXPLORE = "click_explore"
    GOAL_SEEK = "goal_seek"
    PATTERN_EXPLOIT = "pattern_exploit"
    RANDOM_EXPLORE = "random_explore"
    RETRY = "retry"


@dataclass
class StrategyCandidate:
    """A candidate strategy with metadata (analogous to ReasoningCandidate)."""
    strategy: Strategy
    score: float = 0.0
    action: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None
    reason: str = ""


class StrategyRouter:
    """
    Selects the best game strategy given current state.

    Analogous to v4_1 EBM router + candidate generator, adapted for
    interactive game environments.
    """

    # Clockwise turn order for wall-following
    _CW_DIRS = ["move_up", "move_right", "move_down", "move_left"]

    def __init__(
        self,
        explore_budget: int = 15,
        max_repeat_before_random: int = 5,
        seed: Optional[int] = None,
    ):
        self.explore_budget = explore_budget
        self.max_repeat = max_repeat_before_random
        self._rng = random.Random(seed)
        self._last_actions: List[str] = []
        self._stuck_counter: int = 0
        self._explore_action_idx: int = 0
        self._retry_count: int = 0

        # Wall-following state
        self._current_dir_idx: int = 0       # index into _CW_DIRS
        self._steps_in_dir: int = 0          # consecutive steps in current direction
        self._dir_action_map: Dict[str, str] = {}  # semantic → action name
        self._interact_cooldown: int = 0     # wait N actions before interacting again
        self._last_effect_changed: bool = True

        # Click exploration state
        self._click_targets: List[Tuple[int, int]] = []  # (y, x) positions to try clicking
        self._click_target_idx: int = 0
        self._clicked_positions: set = set()  # positions already clicked
        self._movement_ineffective: bool = False  # True if movement doesn't help

    # ------------------------------------------------------------------
    # Main routing
    # ------------------------------------------------------------------
    def select_strategy(
        self,
        memory: GameMemory,
        current_grid: np.ndarray,
        game_state: str,
        levels_completed: int,
        available_actions: List[str],
        action_counter: int,
    ) -> StrategyCandidate:
        """Select the best strategy given current game state and knowledge."""

        candidates = self._generate_candidates(
            memory, current_grid, game_state, levels_completed,
            available_actions, action_counter,
        )

        if not candidates:
            # Fallback: random action
            action = self._pick_random_action(available_actions)
            return StrategyCandidate(
                strategy=Strategy.RANDOM_EXPLORE,
                score=0.0,
                action=action,
                reason="No viable candidates, falling back to random",
            )

        # Score and select best candidate
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]

        # Update tracking
        if best.action:
            self._last_actions.append(best.action)
            if len(self._last_actions) > 20:
                self._last_actions = self._last_actions[-20:]

        return best

    # ------------------------------------------------------------------
    # Candidate generation (analogous to v4_1 CandidateGenerator)
    # ------------------------------------------------------------------
    def _generate_candidates(
        self,
        memory: GameMemory,
        current_grid: np.ndarray,
        game_state: str,
        levels_completed: int,
        available_actions: List[str],
        action_counter: int,
    ) -> List[StrategyCandidate]:
        """Generate and score strategy candidates."""
        candidates: List[StrategyCandidate] = []

        # Remove RESET from normal action pool (handled separately)
        normal_actions = [a for a in available_actions if a != "RESET"]

        # 1. Handle special states
        if game_state in ("NOT_PLAYED", "GAME_OVER"):
            candidates.append(StrategyCandidate(
                strategy=Strategy.RETRY,
                score=100.0,
                action="RESET",
                reason=f"Game state is {game_state}, must RESET",
            ))
            self._retry_count += 1
            return candidates

        # 2. Systematic exploration (early game)
        if action_counter < self.explore_budget:
            cand = self._systematic_explore_candidate(
                memory, normal_actions, action_counter
            )
            if cand:
                candidates.append(cand)

        # 3. Click exploration — high priority for click-heavy games
        if "ACTION6" in normal_actions:
            click_cand = self._click_explore_candidate(
                memory, current_grid, normal_actions, action_counter
            )
            if click_cand:
                candidates.append(click_cand)

        # 4. Directed exploration (wall-following)
        if memory.has_learned_movement():
            cand = self._directed_explore_candidate(memory, current_grid, normal_actions)
            if cand:
                candidates.append(cand)

        # 5. Try interact periodically
        interact_cand = self._interact_candidate(memory, current_grid, normal_actions)
        if interact_cand:
            candidates.append(interact_cand)

        # 6. Goal-seeking (if we can detect objectives)
        goal_cand = self._goal_seek_candidate(memory, current_grid, normal_actions)
        if goal_cand:
            candidates.append(goal_cand)

        # 6. Pattern exploitation (repeat successful sequences)
        pattern_cand = self._pattern_exploit_candidate(memory, normal_actions)
        if pattern_cand:
            candidates.append(pattern_cand)

        # 7. Random exploration (break loops)
        if self._is_stuck(memory):
            # On stuck: force a turn in the wall-follower
            self._current_dir_idx = (self._current_dir_idx + 2) % 4  # reverse
            self._steps_in_dir = 0
            candidates.append(StrategyCandidate(
                strategy=Strategy.RANDOM_EXPLORE,
                score=80.0,
                action=self._pick_random_action(normal_actions),
                reason="Stuck in loop, trying random action",
            ))

        # 8. Always add a random option as fallback
        if normal_actions:
            candidates.append(StrategyCandidate(
                strategy=Strategy.RANDOM_EXPLORE,
                score=10.0,
                action=self._rng.choice(normal_actions),
                reason="Random fallback",
            ))

        return candidates

    def _systematic_explore_candidate(
        self,
        memory: GameMemory,
        actions: List[str],
        action_counter: int,
    ) -> Optional[StrategyCandidate]:
        """Try each action systematically to learn effects."""
        unexplored = memory.get_unexplored_actions(actions)
        if unexplored:
            action = unexplored[0]
            return StrategyCandidate(
                strategy=Strategy.SYSTEMATIC_EXPLORE,
                score=90.0 - action_counter,  # Priority decreases over time
                action=action,
                reason=f"Exploring untried action: {action}",
            )

        # If all actions tried, but some only once, try them again
        under_explored = [
            a for a in actions
            if a in memory.action_profiles
            and memory.action_profiles[a].times_tried < 3
        ]
        if under_explored:
            action = under_explored[self._explore_action_idx % len(under_explored)]
            self._explore_action_idx += 1
            return StrategyCandidate(
                strategy=Strategy.SYSTEMATIC_EXPLORE,
                score=60.0 - action_counter,
                action=action,
                reason=f"Re-exploring under-tested action: {action}",
            )

        return None

    def _click_explore_candidate(
        self,
        memory: GameMemory,
        current_grid: np.ndarray,
        actions: List[str],
        action_counter: int,
    ) -> Optional[StrategyCandidate]:
        """Systematically click on objects in the grid.

        Many ARC-AGI-3 games require clicking (ACTION6) on specific objects.
        This strategy detects objects and clicks on each one.
        """
        if "ACTION6" not in actions:
            return None

        # Detect if movement is ineffective (few unique states despite many actions)
        if action_counter > 20 and memory.total_actions > 15:
            movement_actions_tried = sum(
                1 for p in memory.action_profiles.values()
                if p.is_movement_action() and p.times_tried > 3
            )
            if memory.max_level_reached == 0 and len(memory._visited_hashes) < 10:
                self._movement_ineffective = True

        # Build click targets from detected objects (refresh periodically)
        if not self._click_targets or action_counter % 30 == 0:
            self._rebuild_click_targets(current_grid, memory)

        # Score: higher when movement is ineffective, lower otherwise
        base_score = 75.0 if self._movement_ineffective else 45.0

        # After exploration phase, try clicking more aggressively
        if action_counter > self.explore_budget and action_counter % 3 == 0:
            target = self._next_click_target(current_grid)
            if target is not None:
                y, x = target
                return StrategyCandidate(
                    strategy=Strategy.CLICK_EXPLORE,
                    score=base_score,
                    action="ACTION6",
                    action_data={"x": int(x), "y": int(y)},
                    reason=f"Click-explore at ({y},{x})",
                )

        # For movement-ineffective games, click every other action
        if self._movement_ineffective and action_counter % 2 == 0:
            target = self._next_click_target(current_grid)
            if target is not None:
                y, x = target
                return StrategyCandidate(
                    strategy=Strategy.CLICK_EXPLORE,
                    score=base_score + 10,
                    action="ACTION6",
                    action_data={"x": int(x), "y": int(y)},
                    reason=f"Click-explore (no-move game) at ({y},{x})",
                )

        return None

    def _rebuild_click_targets(
        self, grid: np.ndarray, memory: GameMemory
    ) -> None:
        """Build a list of click targets from grid objects."""
        h, w = grid.shape
        priority_targets: List[Tuple[int, int]] = []  # effective-value objects first
        normal_targets: List[Tuple[int, int]] = []

        effective_vals = memory.get_effective_click_values()

        # 1. Centers of detected objects (non-background)
        objects = GridAnalyzer.find_objects(grid, ignore_values={0}, min_size=1)
        for obj in objects:
            cy, cx = obj.center
            pos = (int(round(cy)), int(round(cx)))
            if pos not in self._clicked_positions:
                if obj.value in effective_vals:
                    priority_targets.append(pos)
                else:
                    normal_targets.append(pos)

            # Also add individual cells of small objects (for precise click games)
            if obj.size <= 10:
                for cell_y, cell_x in list(obj.cells)[:5]:
                    cpos = (cell_y, cell_x)
                    if cpos not in self._clicked_positions:
                        if obj.value in effective_vals:
                            priority_targets.append(cpos)
                        else:
                            normal_targets.append(cpos)

        # 2. If player is known, also click positions near player
        if memory.player.identified and memory.player.position is not None:
            py, px = memory.player.position
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0),
                           (-2, 0), (2, 0), (0, -2), (0, 2)]:
                ny, nx = int(py) + dy, int(px) + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in self._clicked_positions:
                    normal_targets.append((ny, nx))

        # 3. Grid scan positions (denser) for broad coverage
        if len(priority_targets) + len(normal_targets) < 8:
            step = max(h // 5, 1)
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if (y, x) not in self._clicked_positions:
                        normal_targets.append((y, x))

        # Priority targets first, then normal targets (both shuffled)
        self._rng.shuffle(priority_targets)
        self._rng.shuffle(normal_targets)
        self._click_targets = priority_targets + normal_targets
        self._click_target_idx = 0

    def _next_click_target(
        self, grid: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Get the next click target position."""
        h, w = grid.shape
        while self._click_target_idx < len(self._click_targets):
            y, x = self._click_targets[self._click_target_idx]
            self._click_target_idx += 1
            if 0 <= y < h and 0 <= x < w:
                self._clicked_positions.add((y, x))
                return (y, x)

        # If exhausted, reset targets for next rebuild
        self._click_targets = []
        self._click_target_idx = 0
        return None

    def _build_dir_action_map(self, memory: GameMemory) -> None:
        """Build mapping from semantic directions to action names."""
        semantics = memory.infer_action_semantics()
        self._dir_action_map = {}
        for act_name, sem in semantics.items():
            if sem in ("move_up", "move_down", "move_left", "move_right"):
                self._dir_action_map[sem] = act_name

    def _directed_explore_candidate(
        self,
        memory: GameMemory,
        current_grid: np.ndarray,
        actions: List[str],
    ) -> Optional[StrategyCandidate]:
        """Wall-following exploration: persist in a direction until blocked."""
        if not memory.has_learned_movement():
            return None

        # Rebuild direction map if needed
        if not self._dir_action_map:
            self._build_dir_action_map(memory)
        if not self._dir_action_map:
            return None

        # Check if last action had no effect (hit a wall)
        last_blocked = False
        if memory.action_history:
            last_eff = memory.action_history[-1]
            if not last_eff.anything_changed and not last_eff.player_moved:
                last_blocked = True

        # If blocked or taken many steps, turn clockwise
        if last_blocked or self._steps_in_dir > 8:
            self._current_dir_idx = (self._current_dir_idx + 1) % 4
            self._steps_in_dir = 0

        # Get the action for current direction
        target_dir = self._CW_DIRS[self._current_dir_idx]
        action = self._dir_action_map.get(target_dir)

        # If this direction isn't available, try next
        attempts = 0
        while (action is None or action not in actions) and attempts < 4:
            self._current_dir_idx = (self._current_dir_idx + 1) % 4
            target_dir = self._CW_DIRS[self._current_dir_idx]
            action = self._dir_action_map.get(target_dir)
            attempts += 1

        if action is None or action not in actions:
            return None

        self._steps_in_dir += 1

        return StrategyCandidate(
            strategy=Strategy.DIRECTED_EXPLORE,
            score=55.0,
            action=action,
            reason=f"Wall-follow: {target_dir} (step {self._steps_in_dir})",
        )

    def _goal_seek_candidate(
        self,
        memory: GameMemory,
        current_grid: np.ndarray,
        actions: List[str],
    ) -> Optional[StrategyCandidate]:
        """Navigate toward detected objectives, cycling through them."""
        if not memory.player.identified or memory.player.position is None:
            return None

        objects = GridAnalyzer.find_objects(current_grid, min_size=2)
        if not objects:
            return None

        player_val = memory.player.value
        py, px = memory.player.position

        # Find non-player objects as potential goals
        potential_goals = [
            o for o in objects
            if o.value != player_val
            and o.value != 0
            and o.size < 100
        ]

        if not potential_goals:
            return None

        # Sort by distance; cycle through them using a rotating index
        potential_goals.sort(key=lambda o: (
            (o.center[0] - py) ** 2 + (o.center[1] - px) ** 2
        ))

        # Pick target: cycle through goals to try reaching different ones
        if not hasattr(self, '_goal_idx'):
            self._goal_idx = 0
            self._goal_reached_set: set = set()

        # Skip already-reached goals (player within 2 cells)
        target = None
        for i in range(len(potential_goals)):
            idx = (self._goal_idx + i) % len(potential_goals)
            g = potential_goals[idx]
            dist_sq = (g.center[0] - py) ** 2 + (g.center[1] - px) ** 2
            goal_key = (int(round(g.center[0])), int(round(g.center[1])), g.value)
            if dist_sq < 4:
                # We've reached this goal — mark it and advance
                self._goal_reached_set.add(goal_key)
                self._goal_idx = (idx + 1) % max(len(potential_goals), 1)
                continue
            target = g
            break

        if target is None:
            # All goals reached — reset and try again
            self._goal_reached_set.clear()
            self._goal_idx = 0
            return None

        # Navigate toward the target
        goal_y, goal_x = target.center
        dy = goal_y - py
        dx = goal_x - px

        best_action = None
        best_alignment = -float("inf")

        for act_name in memory.get_movement_actions():
            if act_name not in actions:
                continue
            act_dir = memory.get_action_direction(act_name)
            if act_dir is None:
                continue
            ady, adx = act_dir
            alignment = dy * ady + dx * adx
            if alignment > best_alignment:
                best_alignment = alignment
                best_action = act_name

        if best_action and best_alignment > 0:
            return StrategyCandidate(
                strategy=Strategy.GOAL_SEEK,
                score=72.0,
                action=best_action,
                reason=f"Goal-seek: object (val={target.value}) at {target.center}",
            )

        # Can't align — advance to next goal
        self._goal_idx = (self._goal_idx + 1) % max(len(potential_goals), 1)
        return None

    def _interact_candidate(
        self,
        memory: GameMemory,
        current_grid: np.ndarray,
        actions: List[str],
    ) -> Optional[StrategyCandidate]:
        """Periodically try ACTION5 (interact) or ACTION6 (click)."""
        if self._interact_cooldown > 0:
            self._interact_cooldown -= 1
            return None

        # Try ACTION5 every ~15 actions if the grid has objects near player
        if memory.total_actions > 0 and memory.total_actions % 15 == 0:
            if "ACTION5" in actions:
                self._interact_cooldown = 10
                return StrategyCandidate(
                    strategy=Strategy.SYSTEMATIC_EXPLORE,
                    score=62.0,
                    action="ACTION5",
                    reason="Periodic interact attempt",
                )

        # Try ACTION6 (click) on nearby objects every ~25 actions
        if (
            memory.total_actions > 10
            and memory.total_actions % 25 == 0
            and "ACTION6" in actions
            and memory.player.identified
            and memory.player.position is not None
        ):
            # Click near the player's position
            py, px = memory.player.position
            # Try clicking slightly ahead in current movement direction
            self._interact_cooldown = 10
            return StrategyCandidate(
                strategy=Strategy.SYSTEMATIC_EXPLORE,
                score=58.0,
                action="ACTION6",
                action_data={"x": int(px), "y": int(py)},
                reason="Click-interact near player",
            )

        return None

    def _pattern_exploit_candidate(
        self,
        memory: GameMemory,
        actions: List[str],
    ) -> Optional[StrategyCandidate]:
        """Repeat a pattern that previously led to progress."""
        # Look for recent action sequences that led to level changes
        for level, seq in memory.level_action_sequences.items():
            if level < memory.current_level and len(seq) > 0:
                # This sequence completed a level — try repeating parts
                # Use the last few actions before level completion
                useful = [a for a in seq[-5:] if a in actions and a != "RESET"]
                if useful:
                    action = useful[memory.total_actions % len(useful)]
                    return StrategyCandidate(
                        strategy=Strategy.PATTERN_EXPLOIT,
                        score=65.0,
                        action=action,
                        reason=f"Repeating pattern from level {level}",
                    )

        return None

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------
    def _is_stuck(self, memory: GameMemory) -> bool:
        """Detect if the agent is stuck in a loop."""
        if len(self._last_actions) < self.max_repeat:
            return False

        # Check if the same action was repeated too many times
        recent = self._last_actions[-self.max_repeat:]
        if len(set(recent)) == 1:
            return True

        # Check if alternating between two actions
        if len(self._last_actions) >= 6:
            last6 = self._last_actions[-6:]
            if last6[0] == last6[2] == last6[4] and last6[1] == last6[3] == last6[5]:
                return True

        # Check if the grid hasn't changed (via memory)
        if memory.action_history:
            recent_effects = memory.action_history[-self.max_repeat:]
            if all(not e.anything_changed for e in recent_effects):
                return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pick_random_action(self, actions: List[str]) -> str:
        """Pick a random action, preferring non-RESET."""
        non_reset = [a for a in actions if a != "RESET"]
        if non_reset:
            return self._rng.choice(non_reset)
        return "RESET"

    def on_game_over(self) -> None:
        """Update state after game over."""
        self._stuck_counter = 0
        self._retry_count += 1

    def on_level_complete(self) -> None:
        """Update state after level completion."""
        self._stuck_counter = 0
