"""Three-tier progress tracking + branch killing.

Replaces the flat "repeated progress" counter with three distinct signals:

  LP (Local Progress)   — agent can act, operators work locally
  SP (Structural Progress) — game state genuinely changed in a useful way
  TP (Terminal Progress)   — evidence that level completion is getting closer

Also provides a BranchKiller that detects sterile loops and signals
when the current behavioral branch should be abandoned.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Three-tier progress signals
# =====================================================================

@dataclass
class ProgressSignals:
    """Rolling progress counters (reset per branch/attempt)."""

    # Local progress — agent is acting, operators predict correctly
    safe_moves: int = 0
    operator_predictions_matched: int = 0
    clicks_executed: int = 0
    local_transforms: int = 0

    # Structural progress — game state changed meaningfully
    new_regions_accessed: int = 0       # novel grid hashes
    object_class_first_removal: int = 0 # first time a VALUE's count drops
    new_affordances_unlocked: int = 0   # new interaction possibilities
    blocking_rules_validated: int = 0   # rules about what stops movement
    novel_state_motifs: int = 0         # states never seen before

    # Terminal progress — evidence completion is nearer
    remaining_target_objects: int = 999 # count of "target" objects
    target_classes_exhausted: int = 0   # object classes fully removed
    closure_rule_plausible: int = 0     # #rules suggesting completion
    successful_prefix_replayed: int = 0 # matched previous solution start
    unique_region_reachable: int = 0    # new unique region accessible

    @property
    def local_score(self) -> float:
        """0–1 score for local progress."""
        raw = (
            min(self.safe_moves, 20) / 20.0 * 0.3
            + min(self.operator_predictions_matched, 10) / 10.0 * 0.4
            + min(self.clicks_executed + self.local_transforms, 5) / 5.0 * 0.3
        )
        return min(1.0, raw)

    @property
    def structural_score(self) -> float:
        """0–1 score for structural progress."""
        raw = (
            min(self.novel_state_motifs, 10) / 10.0 * 0.3
            + min(self.object_class_first_removal, 3) / 3.0 * 0.3
            + min(self.new_regions_accessed, 5) / 5.0 * 0.2
            + min(self.blocking_rules_validated, 2) / 2.0 * 0.2
        )
        return min(1.0, raw)

    @property
    def terminal_score(self) -> float:
        """0–1 score for terminal progress (closer to level end)."""
        # Fewer remaining targets = better
        target_signal = max(0.0, 1.0 - self.remaining_target_objects / 20.0)
        raw = (
            target_signal * 0.3
            + min(self.target_classes_exhausted, 3) / 3.0 * 0.3
            + min(self.closure_rule_plausible, 2) / 2.0 * 0.2
            + min(self.unique_region_reachable, 2) / 2.0 * 0.2
        )
        return min(1.0, raw)


# =====================================================================
# Branch Killer
# =====================================================================

@dataclass
class BranchState:
    """Rolling window state for a behavioral branch."""
    window_size: int = 40
    # Rolling counters within the window
    new_validated_ops: int = 0
    new_validated_rules: int = 0
    terminal_progress_delta: float = 0.0
    repeated_state_hashes: int = 0
    repeated_diff_signatures: int = 0
    repeated_macro_ids: int = 0
    window_actions: int = 0


class ProgressTracker:
    """Tracks three-tier progress and detects sterile branches.

    Usage:
        tracker = ProgressTracker()
        tracker.on_action(obs, diff, ...)  # called each step
        if tracker.should_kill_branch():
            # force restart or mode change
        lp, sp, tp = tracker.scores()
    """

    # Branch killing thresholds
    KILL_WINDOW: int = 40
    MIN_NOVEL_STATES: int = 3     # minimum unique states in window
    MAX_REPEATED_HASHES: int = 3  # repeated state hashes before kill
    MAX_REPEATED_DIFFS: int = 5   # repeated diff signatures before kill
    TERMINAL_STALL_LIMIT: int = 50  # actions with no terminal improvement

    def __init__(self) -> None:
        self.signals = ProgressSignals()

        # Rolling window for branch killing
        self._recent_hashes: deque = deque(maxlen=self.KILL_WINDOW)
        self._recent_diffs: deque = deque(maxlen=self.KILL_WINDOW)
        self._recent_macros: deque = deque(maxlen=self.KILL_WINDOW)

        # Tracking for structural signals
        self._seen_hashes: Set[int] = set()
        self._seen_object_classes_removed: Set[int] = set()
        self._prev_object_class_counts: Dict[int, int] = {}
        self._smallest_target_count: int = 999

        # Terminal tracking
        self._prev_terminal_score: float = 0.0
        self._actions_since_terminal_improvement: int = 0

        # Validated ops/rules tracking (snapshot at branch start)
        self._prev_validated_ops: int = 0
        self._prev_validated_rules: int = 0

        # Per-branch counters
        self._branch_actions: int = 0
        self._branch_id: int = 0
        self._num_updates: int = 0

    def on_action(
        self,
        grid_hash: int,
        diff_signature: Optional[str],
        macro_id: Optional[str],
        is_noop: bool,
        game_over: bool,
        player_moved: bool,
        num_changed: int,
        objects: list,
        current_validated_ops: int,
        current_validated_rules: int,
        operator_predicted_ok: bool,
        is_click: bool,
        is_transform: bool,
    ) -> None:
        """Record one action's outcome for progress tracking."""
        self._branch_actions += 1
        self._num_updates += 1

        # ── Local progress ──
        if player_moved and not game_over and not is_noop:
            self.signals.safe_moves += 1
        if operator_predicted_ok:
            self.signals.operator_predictions_matched += 1
        if is_click:
            self.signals.clicks_executed += 1
        if is_transform and num_changed > 3:
            self.signals.local_transforms += 1

        # ── Structural progress ──
        is_novel = grid_hash not in self._seen_hashes
        if is_novel:
            self._seen_hashes.add(grid_hash)
            self.signals.novel_state_motifs += 1
            self.signals.new_regions_accessed += 1

        # Track object class removals
        current_class_counts: Dict[int, int] = {}
        for obj in objects:
            v = obj.value if hasattr(obj, 'value') else 0
            current_class_counts[v] = current_class_counts.get(v, 0) + 1

        if self._prev_object_class_counts:
            for cls, prev_count in self._prev_object_class_counts.items():
                cur_count = current_class_counts.get(cls, 0)
                if cur_count < prev_count and cls not in self._seen_object_classes_removed:
                    self._seen_object_classes_removed.add(cls)
                    self.signals.object_class_first_removal += 1
                if cur_count == 0 and prev_count > 0:
                    self.signals.target_classes_exhausted += 1
        self._prev_object_class_counts = current_class_counts

        # New validated ops/rules since branch start
        new_ops = current_validated_ops - self._prev_validated_ops
        new_rules = current_validated_rules - self._prev_validated_rules
        if new_ops > 0 or new_rules > 0:
            self.signals.blocking_rules_validated += max(0, new_ops) + max(0, new_rules)

        # ── Terminal progress ──
        # Remaining target objects = non-background, small objects
        target_count = sum(
            1 for obj in objects
            if hasattr(obj, 'area') and obj.area <= 15
            and hasattr(obj, 'value') and obj.value != 0
        )
        self.signals.remaining_target_objects = target_count
        if target_count < self._smallest_target_count:
            self._smallest_target_count = target_count

        # Track terminal improvement
        new_ts = self.signals.terminal_score
        if new_ts > self._prev_terminal_score + 0.01:
            self._prev_terminal_score = new_ts
            self._actions_since_terminal_improvement = 0
        else:
            self._actions_since_terminal_improvement += 1

        # ── Branch killing window ──
        self._recent_hashes.append(grid_hash)
        self._recent_diffs.append(diff_signature or "noop")
        self._recent_macros.append(macro_id or "none")

    @property
    def num_updates(self) -> int:
        """How many times ``on_action`` has been called this game."""
        return self._num_updates

    def branch_diagnostics(self) -> Dict[str, int | float]:
        """Expose the current branch-killer signals for debugging."""
        hash_counts: Dict[int, int] = {}
        for h in self._recent_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1

        diff_counts: Dict[str, int] = {}
        for d in self._recent_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1

        max_hash_repeat = max(hash_counts.values()) if hash_counts else 0
        max_diff_repeat = max(diff_counts.values()) if diff_counts else 0
        unique_in_window = len(set(self._recent_hashes))

        return {
            "branch_id": self._branch_id,
            "branch_actions": self._branch_actions,
            "window_actions": len(self._recent_hashes),
            "unique_states_in_window": unique_in_window,
            "max_hash_repeat": max_hash_repeat,
            "max_diff_repeat": max_diff_repeat,
            "actions_since_terminal_improvement": self._actions_since_terminal_improvement,
        }

    def should_kill_branch(self) -> bool:
        """Check if the current branch is sterile and should be killed."""
        if self._branch_actions < self.KILL_WINDOW:
            return False

        # Count repeated hashes in window
        hash_counts: Dict[int, int] = {}
        for h in self._recent_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1
        max_hash_repeat = max(hash_counts.values()) if hash_counts else 0

        # Count repeated diff signatures in window
        diff_counts: Dict[str, int] = {}
        for d in self._recent_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        max_diff_repeat = max(diff_counts.values()) if diff_counts else 0

        # Unique states in window
        unique_in_window = len(set(self._recent_hashes))

        # Kill conditions (any one is sufficient)
        # 1. Same state hash dominates the window
        if max_hash_repeat >= self.KILL_WINDOW * 0.5:
            return True

        # 2. Same diff signature dominates (doing the same thing repeatedly)
        if max_diff_repeat >= self.KILL_WINDOW * 0.6:
            return True

        # 3. Very few unique states (cycling)
        if unique_in_window < self.MIN_NOVEL_STATES:
            return True

        # 4. No terminal improvement for too long
        if (self._actions_since_terminal_improvement >= self.TERMINAL_STALL_LIMIT
                and self.signals.structural_score < 0.2):
            return True

        return False

    def start_new_branch(
        self,
        current_validated_ops: int = 0,
        current_validated_rules: int = 0,
    ) -> None:
        """Reset branch-local tracking for a new attempt."""
        self._branch_id += 1
        self._branch_actions = 0
        self._recent_hashes.clear()
        self._recent_diffs.clear()
        self._recent_macros.clear()
        self._prev_validated_ops = current_validated_ops
        self._prev_validated_rules = current_validated_rules
        self._actions_since_terminal_improvement = 0
        self._prev_terminal_score = self.signals.terminal_score
        # Keep cumulative signals — they're game-level, not branch-level

    def scores(self) -> Tuple[float, float, float]:
        """Return (local_score, structural_score, terminal_score)."""
        return (
            self.signals.local_score,
            self.signals.structural_score,
            self.signals.terminal_score,
        )

    def summary(self) -> Dict[str, any]:
        """Return a summary dict for diagnostics."""
        lp, sp, tp = self.scores()
        branch_diag = self.branch_diagnostics()
        return {
            "local_progress": round(lp, 2),
            "structural_progress": round(sp, 2),
            "terminal_progress": round(tp, 2),
            "safe_moves": self.signals.safe_moves,
            "novel_states": self.signals.novel_state_motifs,
            "class_removals": self.signals.object_class_first_removal,
            "classes_exhausted": self.signals.target_classes_exhausted,
            "remaining_targets": self.signals.remaining_target_objects,
            "progress_updates": self._num_updates,
            "branch_id": branch_diag["branch_id"],
            "branch_actions": branch_diag["branch_actions"],
            "unique_states_in_window": branch_diag["unique_states_in_window"],
            "max_hash_repeat": branch_diag["max_hash_repeat"],
            "max_diff_repeat": branch_diag["max_diff_repeat"],
            "actions_since_terminal_improvement": branch_diag["actions_since_terminal_improvement"],
        }
