"""
Actioner — converts a selected strategy + active subgoal into concrete actions.

Given:
  - The current GameStrategy (selected by EBM scoring)
  - The active SubGoal from the goal hierarchy
  - Current observation and game memory

Produces:
  - A single concrete action name (ACTION1-7, RESET)
  - Optional action_data (e.g. click coordinates for ACTION6)

The Actioner is *stateful* — it tracks progress within the current
subgoal (e.g. which target it's navigating toward, which click
position it should try next) and resets when the subgoal changes.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state_describer import GameObservation
from .strategy_generator import GameStrategy, StrategyType
from .goal_decomposer import SubGoal, SubGoalStatus
from .game_memory import GameMemory

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Output of a single Actioner step."""
    action: str
    action_data: Optional[Dict[str, Any]] = None
    reason: str = ""


class Actioner:
    """
    Translates high-level strategies into low-level game actions.

    Uses memory's learned action semantics (which action moves in which
    direction, which actions interact, etc.) to implement strategies:

      - **Navigate**: pick the movement action whose direction best aligns
        with the vector toward the subgoal's target.
      - **Interact**: use ACTION5 or ACTION6 when near a target object.
      - **Click**: use ACTION6 with specific (x, y) coordinates from the
        subgoal / strategy metadata.
      - **Explore**: cycle through untried actions or visit unvisited regions.
      - **Sequence**: execute the strategy's action_plan in order.
      - **Undo**: use ACTION7 or RESET to backtrack.
    """

    def __init__(self) -> None:
        # Per-subgoal state (reset when subgoal changes)
        self._current_subgoal_id: Optional[int] = None
        self._step_in_subgoal: int = 0
        self._nav_target_idx: int = 0       # rotating index for multi-target nav
        self._click_target_idx: int = 0     # rotating index for click targets
        self._plan_idx: int = 0             # index into strategy.action_plan
        self._last_strategy_token: Optional[int] = None
        self._last_action: Optional[str] = None
        self._stuck_counter: int = 0        # counts consecutive no-effect actions
        self._explore_idx: int = 0          # index for exploration cycling
        self._blocked_actions: set = set()  # actions that had no effect recently
        self._clicked_positions: set = set()  # tracks (y, x) we already clicked
        self._gameover_positions: set = set()  # tracks (y, x) that caused game-over
        self._last_click_pos: Optional[tuple] = None  # last ACTION6 click position

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def act(
        self,
        strategy: GameStrategy,
        subgoal: Optional[SubGoal],
        observation: GameObservation,
        memory: GameMemory,
        available_actions: List[str],
    ) -> ActionResult:
        """
        Produce the next concrete action.

        Args:
            strategy:  the EBM-selected strategy for the current subgoal
            subgoal:   the active subgoal (may be None during exploration)
            observation: current game observation
            memory:    accumulated game knowledge
            available_actions: actions the environment currently allows

        Returns:
            ActionResult with action name and optional data.
        """
        # Reset internal state when subgoal changes
        sg_id = subgoal.id if subgoal else -1
        if sg_id != self._current_subgoal_id:
            self._on_subgoal_change(sg_id)

        strategy_token = id(strategy)
        if strategy_token != self._last_strategy_token:
            # Receding-horizon planning replans every step and should
            # always execute the first action of the newly selected
            # prefix, not continue at an old plan index.
            self._plan_idx = 0
            self._last_strategy_token = strategy_token

        self._step_in_subgoal += 1

        # Dispatch to the appropriate handler based on strategy type
        result = self._dispatch(strategy, subgoal, observation, memory, available_actions)

        # Validate the chosen action
        if result.action not in available_actions and result.action != "RESET":
            result.action = available_actions[0] if available_actions else "ACTION1"

        # Track click position for game-over avoidance
        if result.action == "ACTION6" and result.action_data:
            self._last_click_pos = (
                result.action_data.get("y", 0), result.action_data.get("x", 0)
            )
        else:
            self._last_click_pos = None

        self._last_action = result.action
        return result

    # ------------------------------------------------------------------
    # Preferred-action bias (Phase 2 — from TaskProgram subgoal metadata)
    # ------------------------------------------------------------------
    # When a subgoal comes from a compiled human-trace TaskProgram, its
    # metadata may carry `prefer_actions` — a short list of ACTION
    # names the human used effectively for an analogous subgoal. We
    # narrow the candidate pool to those actions *when available*, and
    # fall back to the full pool when the intersection is empty.
    @staticmethod
    def _preferred_actions(subgoal: Optional[SubGoal]) -> Optional[List[str]]:
        if subgoal is None:
            return None
        pref = subgoal.metadata.get("prefer_actions")
        if not pref:
            return None
        # Normalise — accept lowercase, numbers-only, etc.
        out: List[str] = []
        for p in pref:
            s = str(p).strip().upper().replace(" ", "")
            if s.isdigit():
                s = f"ACTION{s}"
            if s.startswith("ACTION") and s not in out:
                out.append(s)
        return out or None

    @staticmethod
    def _is_probe_subgoal(subgoal: Optional[SubGoal]) -> bool:
        """True when the subgoal is an identify/probe-style test.

        Probes use tight budgets (typically max_actions ≤ 10) to
        confirm a specific action's role. For them we apply a HARD
        filter on prefer_actions (concentrate primitive discovery).
        Everything else (achieve/solve/explore/navigate) gets a SOFT
        reorder so sibling-discovered primitives stay reachable.
        """
        if subgoal is None:
            return False
        # Primary signal: the original TaskProgram test id (string),
        # preserved by goal_decomposer in metadata["task_program_id"].
        meta = getattr(subgoal, "metadata", {}) or {}
        if meta.get("probe"):
            return True
        tp_id = str(meta.get("task_program_id", "")).lower()
        if tp_id.startswith("probe") or tp_id.startswith("identify"):
            return True
        if tp_id.startswith("achieve") or tp_id.startswith("solve") or tp_id.startswith("win"):
            return False
        # Fallback heuristic: small budget + prefer_actions present
        # → treat as probe even if id doesn't match.
        max_act = getattr(subgoal, "max_actions", 999)
        return (
            max_act is not None
            and max_act <= 10
            and bool(meta.get("prefer_actions"))
        )

    def _biased_available(
        self, available: List[str], subgoal: Optional[SubGoal]
    ) -> List[str]:
        """Bias `available` toward the subgoal's preferred actions.

        Two modes:
          - PROBE subgoals: HARD filter — return only preferred
            actions (with safe fallback to full list if the
            intersection is empty). Concentrates primitive discovery.
          - ACHIEVE / other subgoals: SOFT reorder — preferred first,
            but every available action is still present, so sibling
            primitives (e.g. a control-switch action) stay reachable.
        """
        pref = self._preferred_actions(subgoal)
        if not pref:
            return available
        preferred_first = [a for a in available if a in pref]
        rest = [a for a in available if a not in pref]
        if self._is_probe_subgoal(subgoal):
            # Hard filter, with empty-intersection fallback.
            return preferred_first if preferred_first else available
        # Soft reorder.
        return preferred_first + rest

    def on_game_over(self) -> None:
        """Record that the last click position caused a game-over."""
        if self._last_click_pos is not None:
            self._gameover_positions.add(self._last_click_pos)
        # Reset click tracker on game-over so we re-explore with avoidance
        self._clicked_positions.clear()

    def on_action_effect(self, changed: bool) -> None:
        """Called after an action to track stuck detection."""
        if changed:
            self._stuck_counter = 0
            self._blocked_actions.clear()
        else:
            self._stuck_counter += 1
            if self._last_action:
                self._blocked_actions.add(self._last_action)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _dispatch(
        self,
        strategy: GameStrategy,
        subgoal: Optional[SubGoal],
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
    ) -> ActionResult:

        st = strategy.strategy_type

        # If stuck for many actions, try something different
        if self._stuck_counter >= 3:
            self._stuck_counter = 0
            return self._unstick(obs, memory, available, subgoal)

        if st == StrategyType.NAVIGATE_TO_GOAL:
            return self._act_navigate(strategy, subgoal, obs, memory, available)
        elif st == StrategyType.COLLECT_ITEMS:
            return self._act_navigate(strategy, subgoal, obs, memory, available)
        elif st == StrategyType.CLICK_OBJECTS:
            return self._act_click(strategy, subgoal, obs, memory, available)
        elif st == StrategyType.EXPLORE_SYSTEMATICALLY:
            return self._act_explore(strategy, subgoal, obs, memory, available)
        elif st == StrategyType.UNDO_AND_RETRY:
            return self._act_undo(strategy, subgoal, obs, memory, available)
        elif st in (StrategyType.SOLVE_PUZZLE, StrategyType.SEQUENCE_ACTIONS):
            return self._act_sequence(strategy, subgoal, obs, memory, available)
        elif st == StrategyType.AVOID_HAZARDS:
            return self._act_navigate(strategy, subgoal, obs, memory, available)
        else:
            return self._act_sequence(strategy, subgoal, obs, memory, available)

    # ------------------------------------------------------------------
    # Navigate / Collect
    # ------------------------------------------------------------------
    def _act_navigate(
        self,
        strategy: GameStrategy,
        subgoal: Optional[SubGoal],
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
    ) -> ActionResult:
        """Move the player toward the subgoal's target position."""

        # Determine target coordinates
        ty, tx = self._get_target(subgoal, obs)

        # If no player or no target, fall through to plan cycling
        if obs.player_info is None or (ty is None):
            return self._act_sequence(strategy, subgoal, obs, memory, available)

        py, px = obs.player_info["y"], obs.player_info["x"]

        # Check if we've arrived (within ~3 cells — targets are approximate)
        dist_sq = (ty - py) ** 2 + (tx - px) ** 2
        if dist_sq <= 9:
            # Limit interaction attempts per target — try interact twice then move on
            interact_count = getattr(self, "_interact_attempts", 0)
            if interact_count < 2:
                self._interact_attempts = interact_count + 1
                return self._interact_at_target(obs, memory, available, ty, tx)
            # Exhausted interaction → advance to next target
            self._interact_attempts = 0
            self._nav_target_idx += 1
            ty2, tx2 = self._get_target(subgoal, obs)
            if ty2 is not None:
                dy, dx = ty2 - py, tx2 - px
                best = self._best_movement_toward(dy, dx, memory, available, subgoal)
                if best:
                    return ActionResult(action=best, reason=f"Moving to next target ({ty2:.0f},{tx2:.0f})")
            return self._act_sequence(strategy, subgoal, obs, memory, available)

        # Rotate target if stuck on same one for too long
        if self._step_in_subgoal > 0 and self._step_in_subgoal % 15 == 0:
            self._nav_target_idx += 1
            ty, tx = self._get_target(subgoal, obs)
            if ty is None:
                return self._act_sequence(strategy, subgoal, obs, memory, available)
            dy, dx = ty - py, tx - px

        # Pick the movement action that best aligns toward target
        dy, dx = ty - py, tx - px
        best_action = self._best_movement_toward(dy, dx, memory, available, subgoal)

        if best_action:
            return ActionResult(
                action=best_action,
                reason=f"Moving toward ({ty:.0f},{tx:.0f}), d=({dy:.0f},{dx:.0f})",
            )

        # No directional info yet → cycle through known movement actions or action_plan
        move_actions = memory.get_movement_actions()
        if not move_actions:
            # Use strategy's action_plan as candidate movements
            move_actions = [a for a in strategy.action_plan if a in available]
        if move_actions:
            action = move_actions[self._step_in_subgoal % len(move_actions)]
            return ActionResult(
                action=action,
                reason=f"Cycling movement toward ({ty:.0f},{tx:.0f})",
            )

        # Absolute fallback: cycle any non-interact action
        fallback = [a for a in available if a not in ("ACTION5", "ACTION6", "ACTION7")]
        if fallback:
            action = fallback[self._step_in_subgoal % len(fallback)]
            return ActionResult(action=action, reason="Trying directional actions")

        return self._act_sequence(strategy, subgoal, obs, memory, available)

    def _get_target(
        self,
        subgoal: Optional[SubGoal],
        obs: GameObservation,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract target (y, x) from subgoal metadata or obs objects."""
        if subgoal and subgoal.metadata.get("target_y") is not None:
            return subgoal.metadata["target_y"], subgoal.metadata["target_x"]

        # Fall back to nearest non-player object
        objects = [
            o for o in (obs.objects or [])
            if not o.get("is_player") and o.get("value", 0) != 0
        ]
        if not objects or obs.player_info is None:
            return None, None

        py, px = obs.player_info["y"], obs.player_info["x"]
        idx = self._nav_target_idx % len(objects)
        obj = objects[idx]
        return obj["center_y"], obj["center_x"]

    def _best_movement_toward(
        self,
        dy: float,
        dx: float,
        memory: GameMemory,
        available: List[str],
        subgoal: Optional[SubGoal] = None,
    ) -> Optional[str]:
        """Find the movement action with best dot-product alignment.

        Skips actions that recently had no effect (blocked by wall).
        With prefer_actions:
          - PROBE subgoal: hard-restrict to preferred (return None if
            no preferred movement aligns — caller falls back to
            cycling on the already-filtered `available` set).
          - ACHIEVE / other subgoal: two-pass. Try preferred first;
            fall back to all available movement actions if none
            of the preferred ones has positive alignment.
        """
        preferred = self._preferred_actions(subgoal)
        is_probe = self._is_probe_subgoal(subgoal)

        def _pick(restrict_to_preferred: bool) -> Optional[str]:
            best_action = None
            best_alignment = -float("inf")
            for act_name in memory.get_movement_actions():
                if act_name not in available:
                    continue
                if act_name in self._blocked_actions:
                    continue  # skip wall-blocked actions
                if restrict_to_preferred and preferred and act_name not in preferred:
                    continue
                act_dir = memory.get_action_direction(act_name)
                if act_dir is None:
                    continue
                ady, adx = act_dir
                alignment = dy * ady + dx * adx
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_action = act_name
            return best_action if best_action and best_alignment > 0 else None

        if preferred:
            pick = _pick(restrict_to_preferred=True)
            if pick is not None:
                return pick
            if is_probe:
                # Do NOT widen to all movement actions for probes —
                # doing so would defeat the hard-filter concentration.
                return None
        # Achieve / non-probe subgoals (and no-preferred case) widen.
        return _pick(restrict_to_preferred=False)

    def _interact_at_target(
        self,
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
        ty: float,
        tx: float,
    ) -> ActionResult:
        """Try interaction at a target.

        Many games collect items by walking over them, so we try
        movement toward the item first, then ACTION5/ACTION6."""
        py, px = (0.0, 0.0)
        if obs.player_info:
            py, px = obs.player_info["y"], obs.player_info["x"]

        interact_count = getattr(self, "_interact_attempts", 0)

        # First attempt: try to walk directly onto/through the target
        if interact_count == 0:
            dy, dx = ty - py, tx - px
            best = self._best_movement_toward(dy, dx, memory, available)
            # NB: no subgoal threading here — _interact_at_target is
            # called once we've already arrived at the target, so any
            # remaining movement is fine-grained and shouldn't be
            # restricted by prefer_actions.
            if best:
                return ActionResult(
                    action=best,
                    reason=f"Walking onto target ({ty:.0f},{tx:.0f})",
                )

        # Second attempt: try ACTION5 (interact)
        if "ACTION5" in available:
            return ActionResult(
                action="ACTION5",
                reason=f"At target ({ty:.0f},{tx:.0f}), trying interact",
            )
        # Third attempt: try ACTION6 (click on target)
        if "ACTION6" in available:
            return ActionResult(
                action="ACTION6",
                action_data={"x": int(round(tx)), "y": int(round(ty))},
                reason=f"At target ({ty:.0f},{tx:.0f}), trying click",
            )
        # Fallback: random movement
        move_actions = memory.get_movement_actions()
        if move_actions:
            action = move_actions[self._step_in_subgoal % len(move_actions)]
            return ActionResult(action=action, reason="Target reached, trying move")
        return ActionResult(action=available[0], reason="No interact action available")

    # ------------------------------------------------------------------
    # Click
    # ------------------------------------------------------------------
    def _act_click(
        self,
        strategy: GameStrategy,
        subgoal: Optional[SubGoal],
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
    ) -> ActionResult:
        """Click on objects specified by subgoal or strategy metadata."""
        if "ACTION6" not in available:
            return self._act_explore(strategy, subgoal, obs, memory, available)

        # Get click targets from subgoal metadata, strategy metadata, or objects
        targets = []
        if subgoal and subgoal.metadata.get("click_targets"):
            targets = subgoal.metadata["click_targets"]
        elif strategy.metadata.get("click_targets"):
            targets = strategy.metadata["click_targets"]
        else:
            # Build targets from non-player objects
            objects = [
                o for o in (obs.objects or [])
                if not o.get("is_player") and o.get("value", 0) != 0
            ]
            targets = [
                {"x": int(o["center_x"]), "y": int(o["center_y"])}
                for o in objects
            ]

        if not targets:
            return self._act_explore(strategy, subgoal, obs, memory, available)

        idx = self._click_target_idx % len(targets)
        self._click_target_idx += 1
        t = targets[idx]

        return ActionResult(
            action="ACTION6",
            action_data={"x": t["x"], "y": t["y"]},
            reason=f"Clicking at ({t['y']}, {t['x']})",
        )

    # ------------------------------------------------------------------
    # Explore
    # ------------------------------------------------------------------
    def _act_explore(
        self,
        strategy: GameStrategy,
        subgoal: Optional[SubGoal],
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
    ) -> ActionResult:
        """Systematically try actions to discover mechanics."""
        import random as _rand

        # Phase 2: if the subgoal comes from a TaskProgram and lists
        # prefer_actions, narrow exploration to those actions first.
        biased = self._biased_available(available, subgoal)
        available = biased

        # For click-only games (only ACTION6 available), do smart clicking
        non_click = [a for a in available if a != "ACTION6"]
        if not non_click and "ACTION6" in available:
            return self._explore_click(obs, memory)

        # Prioritise untried actions
        unexplored = memory.get_unexplored_actions(available)
        if unexplored:
            action = unexplored[0]
            action_data = None
            if action == "ACTION6":
                return self._explore_click(obs, memory)
            return ActionResult(action=action, reason=f"Exploring untried action {action}")

        # Cycle through all available actions, each tried 2x
        n = len(available)
        cycle_idx = self._explore_idx // 2
        action_idx = cycle_idx % n
        action = available[action_idx]
        self._explore_idx += 1

        action_data = None
        if action == "ACTION6":
            return self._explore_click(obs, memory)

        return ActionResult(
            action=action,
            action_data=action_data,
            reason=f"Systematic explore: {action} (cycle {self._explore_idx})",
        )

    def _explore_click(
        self, obs: GameObservation, memory: GameMemory
    ) -> ActionResult:
        """Novelty-driven click exploration.

        Priority order:
          1. Object centers (one pass only — skip already-clicked)
          2. Sampled non-zero cells (skip already-clicked)
          3. Systematic grid scan (covers remaining area)
        Always avoids positions that caused game-over."""
        import numpy as np

        grid = obs.raw_grid
        h, w = (64, 64) if grid is None else grid.shape

        # Build candidate positions: objects first, then non-zero cells, then scan
        candidates = []

        # 1. Object center positions
        if obs.objects:
            for obj in obs.objects:
                pos = (int(obj["center_y"]), int(obj["center_x"]))
                if pos not in self._clicked_positions and pos not in self._gameover_positions:
                    candidates.append((pos, f"Click object at {pos}"))

        # 2. Non-zero grid cells (sampled every Nth to cover diverse area)
        if grid is not None and not candidates:
            nz = list(zip(*np.nonzero(grid))) if grid.any() else []
            if nz:
                step = max(1, len(nz) // 60)  # sample ~60 positions
                for i in range(0, len(nz), step):
                    pos = (int(nz[i][0]), int(nz[i][1]))
                    if pos not in self._clicked_positions and pos not in self._gameover_positions:
                        candidates.append((pos, f"Click cell at {pos}"))

        # 3. Grid scan (16x16 = 256 positions)
        if not candidates:
            for scan_idx in range(256):
                row, col = divmod(scan_idx, 16)
                cy = int((row + 0.5) * h / 16)
                cx = int((col + 0.5) * w / 16)
                pos = (cy, cx)
                if pos not in self._clicked_positions and pos not in self._gameover_positions:
                    candidates.append((pos, f"Grid scan {pos}"))

        # Pick the next unclicked position
        if candidates:
            pos, reason = candidates[0]
            self._clicked_positions.add(pos)
            return ActionResult(
                action="ACTION6",
                action_data={"x": pos[1], "y": pos[0]},
                reason=reason,
            )

        # All positions exhausted — reset tracker and try again with fresh grid
        self._clicked_positions.clear()
        cy = (self._explore_idx * 7) % h  # pseudo-random pattern
        cx = (self._explore_idx * 13) % w
        self._explore_idx += 1
        return ActionResult(
            action="ACTION6",
            action_data={"x": cx, "y": cy},
            reason=f"Exhausted positions, pseudo-random ({cy},{cx})",
        )

    # ------------------------------------------------------------------
    # Undo / Retry
    # ------------------------------------------------------------------
    def _act_undo(
        self,
        strategy: GameStrategy,
        subgoal: Optional[SubGoal],
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
    ) -> ActionResult:
        """Undo recent actions via ACTION7 or try different approaches."""
        step = self._step_in_subgoal

        # First few steps: undo with ACTION7
        if step <= 3 and "ACTION7" in available:
            return ActionResult(action="ACTION7", reason="Undoing recent action")

        # Then try actions we haven't tried much
        safe = memory.get_safe_actions()
        candidates = [a for a in safe if a in available and a != self._last_action]
        if candidates:
            action = candidates[step % len(candidates)]
            return ActionResult(action=action, reason=f"Retrying with {action}")

        return self._act_explore(strategy, subgoal, obs, memory, available)

    # ------------------------------------------------------------------
    # Sequence (follow action_plan)
    # ------------------------------------------------------------------
    def _act_sequence(
        self,
        strategy: GameStrategy,
        subgoal: Optional[SubGoal],
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
    ) -> ActionResult:
        """Execute the strategy's action_plan in order."""
        plan = strategy.action_plan
        if not plan:
            return self._act_explore(strategy, subgoal, obs, memory, available)

        idx = self._plan_idx % len(plan)
        self._plan_idx += 1
        action = plan[idx]

        if action not in available and action != "RESET":
            action = available[0] if available else "ACTION1"

        action_data = None
        if action == "ACTION6" and strategy.metadata.get("click_targets"):
            targets = strategy.metadata["click_targets"]
            click_idx = (self._plan_idx - 1) % max(len(targets), 1)
            if click_idx < len(targets):
                action_data = targets[click_idx]

        return ActionResult(
            action=action,
            action_data=action_data,
            reason=f"Executing plan step {idx}/{len(plan)}: {action}",
        )

    # ------------------------------------------------------------------
    # Unstick
    # ------------------------------------------------------------------
    def _unstick(
        self,
        obs: GameObservation,
        memory: GameMemory,
        available: List[str],
        subgoal: Optional[SubGoal] = None,
    ) -> ActionResult:
        """Break out of a stuck loop by trying something unexpected."""
        # Phase 2: respect subgoal prefer_actions if present.
        available = self._biased_available(available, subgoal)
        # Try a random action we haven't used recently
        candidates = [a for a in available if a != self._last_action]
        if not candidates:
            candidates = available

        action = random.choice(candidates)
        action_data = None
        if action == "ACTION6" and obs.raw_grid is not None:
            h, w = obs.raw_grid.shape
            # Random position
            action_data = {"x": random.randint(0, w - 1), "y": random.randint(0, h - 1)}

        return ActionResult(
            action=action,
            action_data=action_data,
            reason=f"Unsticking: random {action}",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _on_subgoal_change(self, new_id: int) -> None:
        """Reset per-subgoal state."""
        self._current_subgoal_id = new_id
        self._step_in_subgoal = 0
        self._nav_target_idx = 0
        self._click_target_idx = 0
        self._plan_idx = 0
        self._last_strategy_token = None
        self._stuck_counter = 0
        self._explore_idx = 0
        self._blocked_actions.clear()
        self._interact_attempts = 0
