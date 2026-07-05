"""
Game-specific memory for ARC-AGI-3 environments.

Tracks:
  - Action effects: what each action does to the grid
  - Player state: position, identity across frames
  - Level knowledge: discovered rules per level
  - Cross-level patterns: rules that generalize across levels
  - Visited states: for loop detection and exploration efficiency

Adapted from v4_1 episodic + structural memory for game domains.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .grid_analyzer import FrameDiff, GridAnalyzer


@dataclass
class ActionEffect:
    """Records the observed effect of a single action."""
    action_name: str
    anything_changed: bool
    num_cells_changed: int
    player_moved: bool
    player_displacement: Optional[Tuple[float, float]] = None
    game_state_after: str = "NOT_FINISHED"
    level_changed: bool = False
    grid_hash_before: int = 0
    grid_hash_after: int = 0
    changed_values_before: List[int] = field(default_factory=list)
    changed_values_after: List[int] = field(default_factory=list)
    moved_values: List[int] = field(default_factory=list)
    candidate_control_value: Optional[int] = None


@dataclass
class ActionProfile:
    """Aggregated statistics about what an action does."""
    action_name: str
    times_tried: int = 0
    times_changed_grid: int = 0
    times_moved_player: int = 0
    avg_displacement: Tuple[float, float] = (0.0, 0.0)
    displacements: List[Tuple[float, float]] = field(default_factory=list)
    times_caused_game_over: int = 0
    times_caused_win: int = 0
    times_no_effect: int = 0

    @property
    def change_rate(self) -> float:
        return self.times_changed_grid / max(self.times_tried, 1)

    @property
    def move_rate(self) -> float:
        return self.times_moved_player / max(self.times_tried, 1)

    @property
    def dominant_displacement(self) -> Optional[Tuple[float, float]]:
        """Most common displacement direction."""
        if not self.displacements:
            return None
        dy = np.mean([d[0] for d in self.displacements])
        dx = np.mean([d[1] for d in self.displacements])
        return (float(dy), float(dx))

    def is_movement_action(self, threshold: float = 0.2) -> bool:
        return self.move_rate >= threshold


@dataclass
class PlayerState:
    """Tracks the player entity across frames."""
    value: Optional[int] = None
    position: Optional[Tuple[float, float]] = None
    size: Optional[int] = None
    identified: bool = False
    confidence: float = 0.0


class GameMemory:
    """
    Stores and retrieves knowledge accumulated during ARC-AGI-3 gameplay.

    Adapts the v4_1 structural + episodic memory concepts to the game domain.
    """

    def __init__(self, max_visited_states: int = 10000):
        # Action knowledge
        self.action_profiles: Dict[str, ActionProfile] = {}
        self.action_history: List[ActionEffect] = []

        # Player tracking
        self.player: PlayerState = PlayerState()

        # State tracking
        self._visited_hashes: Set[int] = set()
        self._max_visited = max_visited_states
        self._state_visit_counts: Dict[int, int] = defaultdict(int)

        # Level tracking
        self.current_level: int = 0
        self.level_action_sequences: Dict[int, List[str]] = defaultdict(list)
        self.level_attempts: Dict[int, int] = defaultdict(int)

        # Discovered rules (hypothesis → confidence)
        self.hypotheses: Dict[str, float] = {}

        # Action-to-direction mapping (learned from observations)
        self.action_directions: Dict[str, Optional[Tuple[float, float]]] = {}

        # Visual cortex predicted directions (fallback when no observed data)
        self.vc_directions: Dict[str, Tuple[float, float]] = {}

        # Grid analysis cache
        self._prev_grid: Optional[np.ndarray] = None
        self._prev_grid_hash: int = 0

        # Click tracking
        self.click_history: List[Dict] = []  # {pos, value_at_pos, changed, level_changed}
        self.effective_click_values: Set[int] = set()  # grid values that responded to click

        # Strategy statistics
        self.total_actions: int = 0
        self.total_resets: int = 0
        self.total_game_overs: int = 0
        self.max_level_reached: int = 0

    # ------------------------------------------------------------------
    # Record an action and its effects
    # ------------------------------------------------------------------
    def record_action(
        self,
        action_name: str,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
        diff: FrameDiff,
        game_state: str,
        levels_completed: int,
    ) -> ActionEffect:
        """Record the effect of an action and update knowledge."""
        self.total_actions += 1

        # Track level progression
        previous_level = self.current_level
        level_changed = levels_completed > previous_level
        if level_changed:
            self.current_level = levels_completed
            self.max_level_reached = max(self.max_level_reached, levels_completed)

        # Detect player movement
        player_moved = False
        player_disp = None
        if diff.moved_objects:
            # If we know the player value, find their movement
            if self.player.identified and self.player.value is not None:
                for mo in diff.moved_objects:
                    if mo["value"] == self.player.value:
                        player_moved = True
                        player_disp = mo["displacement"]
                        self.player.position = mo["to_center"]
                        break
            else:
                # Use smallest moving object as candidate player
                candidates = [
                    mo for mo in diff.moved_objects
                    if int(mo.get("value", 0)) != 0
                ]
                if candidates:
                    smallest = min(candidates, key=lambda m: m["num_cells"])
                    player_moved = True
                    player_disp = smallest["displacement"]
                    # Update player hypothesis
                    self._update_player_hypothesis(smallest)

        # Create effect record
        h_before = GridAnalyzer.grid_hash(grid_before)
        h_after = GridAnalyzer.grid_hash(grid_after)
        effect = ActionEffect(
            action_name=action_name,
            anything_changed=diff.anything_changed,
            num_cells_changed=diff.num_changes,
            player_moved=player_moved,
            player_displacement=player_disp,
            game_state_after=game_state,
            level_changed=level_changed,
            grid_hash_before=h_before,
            grid_hash_after=h_after,
            changed_values_before=sorted(int(value) for value in diff.disappeared.keys()),
            changed_values_after=sorted(int(value) for value in diff.appeared.keys()),
            moved_values=sorted({
                int(mo.get("value", -1))
                for mo in diff.moved_objects
                if int(mo.get("value", -1)) != 0
            }),
            candidate_control_value=(
                int(self.player.value)
                if self.player.value is not None else None
            ),
        )
        self.action_history.append(effect)
        self.level_action_sequences[levels_completed].append(action_name)

        # Update action profile
        self._update_action_profile(effect)

        # Update visited states
        self._visited_hashes.add(h_after)
        self._state_visit_counts[h_after] += 1

        # Track game overs / resets
        if game_state == "GAME_OVER":
            self.total_game_overs += 1
        if action_name == "RESET":
            self.total_resets += 1

        # Update action direction mapping
        if player_disp and action_name not in ("RESET", "ACTION6"):
            self.action_directions[action_name] = player_disp

        # Cache current grid
        self._prev_grid = grid_after.copy()
        self._prev_grid_hash = h_after

        return effect

    def _update_action_profile(self, effect: ActionEffect) -> None:
        """Update aggregated statistics for an action."""
        name = effect.action_name
        if name not in self.action_profiles:
            self.action_profiles[name] = ActionProfile(action_name=name)

        profile = self.action_profiles[name]
        profile.times_tried += 1

        if effect.anything_changed:
            profile.times_changed_grid += 1
        else:
            profile.times_no_effect += 1

        if effect.player_moved and effect.player_displacement:
            profile.times_moved_player += 1
            profile.displacements.append(effect.player_displacement)
            # Running average
            all_dy = [d[0] for d in profile.displacements]
            all_dx = [d[1] for d in profile.displacements]
            profile.avg_displacement = (float(np.mean(all_dy)), float(np.mean(all_dx)))

        if effect.game_state_after == "GAME_OVER":
            profile.times_caused_game_over += 1
        elif effect.game_state_after == "WIN":
            profile.times_caused_win += 1

    def _update_player_hypothesis(self, moved_obj: Dict[str, Any]) -> None:
        """Update hypothesis about which object is the player."""
        val = moved_obj["value"]
        if not self.player.identified:
            if self.player.value == val:
                self.player.confidence += 0.2
                if self.player.confidence >= 0.6:
                    self.player.identified = True
            else:
                self.player.value = val
                self.player.confidence = 0.3
            self.player.position = moved_obj["to_center"]
            self.player.size = moved_obj["num_cells"]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def is_state_visited(self, grid: np.ndarray) -> bool:
        """Check if we've seen this grid state before."""
        return GridAnalyzer.grid_hash(grid) in self._visited_hashes

    def state_visit_count(self, grid: np.ndarray) -> int:
        """How many times we've visited this state."""
        return self._state_visit_counts.get(GridAnalyzer.grid_hash(grid), 0)

    def record_click(
        self,
        pos: Tuple[int, int],
        grid_before: np.ndarray,
        changed: bool,
        level_changed: bool,
    ) -> None:
        """Record the result of a click action at a specific position."""
        y, x = pos
        h, w = grid_before.shape
        val = int(grid_before[y, x]) if 0 <= y < h and 0 <= x < w else -1
        entry = {
            "pos": pos, "value_at_pos": val,
            "changed": changed, "level_changed": level_changed,
        }
        self.click_history.append(entry)
        if changed and val >= 0:
            self.effective_click_values.add(val)

    def get_effective_click_values(self) -> Set[int]:
        """Return grid values that produced a change when clicked."""
        return self.effective_click_values

    def get_movement_actions(self) -> List[str]:
        """Return actions that consistently move the player."""
        explicit = [
            name for name, prof in self.action_profiles.items()
            if prof.is_movement_action() and name not in ("RESET", "ACTION6")
        ]
        if explicit:
            return explicit

        # Fallback: actions that consistently change the grid are likely
        # movement actions (ACTION1-4 in most games are directional)
        directional_candidates = [
            name for name, prof in self.action_profiles.items()
            if name not in ("RESET", "ACTION5", "ACTION6", "ACTION7")
            and prof.times_tried >= 2
            and prof.change_rate >= 0.3
        ]
        return directional_candidates

    def get_action_direction(self, action_name: str) -> Optional[Tuple[float, float]]:
        """Get the typical displacement direction for an action.

        Priority: profile dominant_displacement > observed > VC predicted.
        """
        profile = self.action_profiles.get(action_name)
        if profile and profile.dominant_displacement is not None:
            return profile.dominant_displacement
        # Fallback: direct observation of player displacement
        observed = self.action_directions.get(action_name)
        if observed is not None:
            return observed
        # Tertiary fallback: visual cortex prediction
        return self.vc_directions.get(action_name)

    def get_unexplored_actions(self, available_actions: List[str]) -> List[str]:
        """Return actions that haven't been tried yet."""
        return [
            a for a in available_actions
            if a not in self.action_profiles or self.action_profiles[a].times_tried == 0
        ]

    def get_safe_actions(self) -> List[str]:
        """Return actions that have never caused GAME_OVER."""
        return [
            name for name, prof in self.action_profiles.items()
            if prof.times_caused_game_over == 0 and name != "RESET"
        ]

    def has_learned_movement(self) -> bool:
        """Check if we've identified movement actions."""
        return len(self.get_movement_actions()) >= 2

    def get_exploration_score(self) -> float:
        """How well explored is the current game (0-1)."""
        if self.total_actions == 0:
            return 0.0
        n_actions_tried = sum(
            1 for p in self.action_profiles.values() if p.times_tried > 0
        )
        actions_score = n_actions_tried / 7.0  # max 7 actions
        states_score = min(len(self._visited_hashes) / 50.0, 1.0)
        movement_score = 1.0 if self.has_learned_movement() else 0.0
        return (actions_score + states_score + movement_score) / 3.0

    # ------------------------------------------------------------------
    # Mechanism locking: freeze consistent actions for exploitation
    # ------------------------------------------------------------------
    def get_locked_mechanisms(self) -> Dict[str, str]:
        """Return actions with consistent, well-understood behavior.

        An action is 'locked' when:
          - Tried >= 5 times AND
          - change_rate variance is low (consistently does or doesn't change)
          - If it moves the player, displacement is consistent

        Returns: {action_name: semantic_label} for locked actions.
        """
        locked = {}
        for name, p in self.action_profiles.items():
            if name == "RESET" or p.times_tried < 5:
                continue
            cr = p.change_rate
            # Consistently changes (>80%) or consistently doesn't (<15%)
            if cr >= 0.80:
                if p.is_movement_action(threshold=0.3) and p.dominant_displacement:
                    dy, dx = p.dominant_displacement
                    if abs(dy) > abs(dx):
                        locked[name] = "move_up" if dy < 0 else "move_down"
                    else:
                        locked[name] = "move_left" if dx < 0 else "move_right"
                else:
                    locked[name] = "interact"
            elif cr <= 0.15:
                locked[name] = "no_effect"
        return locked

    def score_action(self, action_name: str) -> float:
        """Score an action by learned effect profile (higher = more useful).

        This is the primary action-selection signal once exploration is done.
        score = change_rate - death_rate + novelty_bonus + consistency_bonus
        """
        if action_name not in self.action_profiles:
            return 0.5  # untried: moderate score to encourage trying

        p = self.action_profiles[action_name]
        if p.times_tried == 0:
            return 0.5

        cr = p.change_rate
        dr = p.times_caused_game_over / max(p.times_tried, 1)

        # Consistency bonus: actions that reliably do something
        consistency = 0.0
        if p.times_tried >= 3:
            if cr >= 0.8 or cr <= 0.1:
                consistency = 0.2  # we understand this action well

        # Win bonus: actions that have led to level completion
        win_bonus = 0.3 if p.times_caused_win > 0 else 0.0

        return cr - dr * 0.8 + consistency + win_bonus

    def rank_actions(self, available: List[str]) -> List[str]:
        """Rank available actions by score (best first)."""
        return sorted(available, key=lambda a: -self.score_action(a))

    # ------------------------------------------------------------------
    # Level management
    # ------------------------------------------------------------------
    def on_level_change(self, new_level: int) -> None:
        """Called when a new level starts."""
        self.current_level = new_level
        self.level_attempts[new_level] = self.level_attempts.get(new_level, 0) + 1

    def on_game_over(self) -> None:
        """Called when GAME_OVER occurs."""
        self.total_game_overs += 1
        self.level_attempts[self.current_level] += 1

    def on_reset(self) -> None:
        """Called when the game is reset."""
        self.total_resets += 1

    # ------------------------------------------------------------------
    # Hypothesis management
    # ------------------------------------------------------------------
    def add_hypothesis(self, key: str, confidence: float = 0.5) -> None:
        """Add or update a hypothesis about game mechanics."""
        self.hypotheses[key] = max(self.hypotheses.get(key, 0.0), confidence)

    def get_hypothesis(self, key: str) -> float:
        """Get confidence in a hypothesis (0 if unknown)."""
        return self.hypotheses.get(key, 0.0)

    def infer_action_semantics(self) -> Dict[str, str]:
        """Infer what each action likely does based on observations."""
        semantics: Dict[str, str] = {}

        for name, profile in self.action_profiles.items():
            if name == "RESET":
                semantics[name] = "reset_game"
                continue

            disp = profile.dominant_displacement
            if disp and profile.is_movement_action():
                dy, dx = disp
                if abs(dy) > abs(dx):
                    semantics[name] = "move_up" if dy < 0 else "move_down"
                else:
                    semantics[name] = "move_left" if dx < 0 else "move_right"
            elif profile.change_rate > 0.5:
                semantics[name] = "interact"
            else:
                semantics[name] = "unknown"

        return semantics

    # ------------------------------------------------------------------
    # Summary for logging / debugging
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a summary of accumulated game knowledge."""
        return {
            "total_actions": self.total_actions,
            "total_resets": self.total_resets,
            "total_game_overs": self.total_game_overs,
            "max_level": self.max_level_reached,
            "states_visited": len(self._visited_hashes),
            "exploration_score": self.get_exploration_score(),
            "player_identified": self.player.identified,
            "player_value": self.player.value,
            "movement_actions": self.get_movement_actions(),
            "action_semantics": self.infer_action_semantics(),
            "hypotheses": dict(self.hypotheses),
        }
