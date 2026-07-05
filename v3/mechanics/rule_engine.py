"""Symbolic rule proposer and verifier.

Infers compact causal rules from repeated operator outcomes:
  "if player enters red cell → game over"
  "if all green objects removed → level complete"
  "if adjacent to wall, movement blocked"

Rules are proposed from transition patterns and verified against new
evidence.  High-confidence rules are used to prune search and predict
outcomes without neural models.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    AdjacentToValue,
    ChangedCells,
    Effect,
    FrameDiff,
    GameObservation,
    GameOverEffect,
    GlobalGridChange,
    LevelCompleteEffect,
    NoEffect,
    ObjectCount,
    ObjectExists,
    PlayerDisplacement,
    PlayerExists,
    Predicate,
    RemovesObject,
    Rule,
    TransitionRecord,
)
from .action_profiler import ActionProfiler

logger = logging.getLogger(__name__)

# Thresholds
MIN_SUPPORT_FOR_RULE = 3
MAX_RULES = 50
CONTRADICTION_DEMOTE_RATE = 0.3


class RuleEngine:
    """Propose, verify, and maintain symbolic causal rules."""

    def __init__(self) -> None:
        self.rules: Dict[str, Rule] = {}
        self._proposal_count: int = 0

    def propose_and_verify(
        self,
        profiler: ActionProfiler,
        transitions: List[TransitionRecord],
    ) -> Dict[str, Rule]:
        """Run a full proposal + verification pass."""
        self._proposal_count += 1

        # Propose new rules from transition patterns
        candidates = []
        candidates.extend(self._propose_death_rules(transitions))
        candidates.extend(self._propose_completion_rules(transitions))
        candidates.extend(self._propose_blocking_rules(profiler, transitions))
        candidates.extend(self._propose_removal_rules(transitions))
        candidates.extend(self._propose_transform_rules(transitions))

        # Merge into library
        for rule in candidates:
            existing = self.rules.get(rule.rule_id)
            if existing is not None:
                existing.support = rule.support
                existing.contradictions = rule.contradictions
                existing.confidence = rule.support / max(
                    rule.support + rule.contradictions, 1
                )
            else:
                rule.confidence = rule.support / max(
                    rule.support + rule.contradictions, 1
                )
                self.rules[rule.rule_id] = rule

        # Verify existing rules against recent transitions
        self._verify_against_recent(transitions[-20:] if len(transitions) > 20 else transitions)

        # Prune low-confidence rules
        to_remove = []
        for rid, r in self.rules.items():
            total = r.support + r.contradictions
            if total >= MIN_SUPPORT_FOR_RULE * 2:
                if r.confidence < 0.3:
                    to_remove.append(rid)
        for rid in to_remove:
            logger.info(f"Pruning rule: {rid}")
            del self.rules[rid]

        # Cap total rules
        if len(self.rules) > MAX_RULES:
            by_conf = sorted(self.rules.values(), key=lambda r: r.confidence)
            for r in by_conf[:len(self.rules) - MAX_RULES]:
                del self.rules[r.rule_id]

        logger.info(
            f"Rule engine #{self._proposal_count}: {len(self.rules)} rules"
        )
        return self.rules

    # ─── Rule proposal strategies ───────────────────────────────

    def _propose_death_rules(
        self, transitions: List[TransitionRecord]
    ) -> List[Rule]:
        """Propose rules about what causes game-over."""
        rules: List[Rule] = []
        death_transitions = [t for t in transitions if t.diff.game_over]

        # Check: did player move onto a specific value before dying?
        value_death_counts: Dict[int, int] = {}
        value_total_visits: Dict[int, int] = {}
        for t in transitions:
            if t.diff.player_displacement and t.obs_before.best_player:
                p = t.obs_before.best_player
                dy, dx = t.diff.player_displacement
                nr, nc = p.position[0] + dy, p.position[1] + dx
                grid = t.obs_after.raw_grid
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    val = int(grid[nr, nc])
                    if val != 0:
                        value_total_visits[val] = value_total_visits.get(val, 0) + 1
                        if t.diff.game_over:
                            value_death_counts[val] = value_death_counts.get(val, 0) + 1

        for val, deaths in value_death_counts.items():
            visits = value_total_visits.get(val, deaths)
            if deaths >= 2 and deaths / max(visits, 1) > 0.5:
                rule_id = f"death_on_value_{val}"
                rules.append(Rule(
                    rule_id=rule_id,
                    conditions=[PlayerExists(), AdjacentToValue(val)],
                    operator_kind="move",
                    effects=[GameOverEffect()],
                    support=deaths,
                    contradictions=max(0, visits - deaths),
                ))

        return rules

    def _propose_completion_rules(
        self, transitions: List[TransitionRecord]
    ) -> List[Rule]:
        """Propose rules about what causes level completion."""
        rules: List[Rule] = []
        win_transitions = [t for t in transitions if t.diff.level_complete]

        if len(win_transitions) < 1:
            return rules

        # Check: did a specific object value disappear right before win?
        for t in win_transitions:
            if t.diff.removed_objects:
                # Look at what objects existed before but not after
                for obj in t.obs_before.objects:
                    if obj.object_id in t.diff.removed_objects:
                        rule_id = f"win_on_remove_v{obj.value}"
                        rules.append(Rule(
                            rule_id=rule_id,
                            conditions=[ObjectCount(obj.value, "==", 1)],
                            operator_kind="interact",
                            effects=[LevelCompleteEffect()],
                            support=1,
                            contradictions=0,
                        ))

        return rules

    def _propose_blocking_rules(
        self,
        profiler: ActionProfiler,
        transitions: List[TransitionRecord],
    ) -> List[Rule]:
        """Propose rules about blocked movement."""
        rules: List[Rule] = []

        # For movement actions, check when they DON'T displace
        for action_name, stats in profiler.stats.items():
            disp = profiler.dominant_displacement(action_name)
            if disp is None:
                continue

            dy, dx = disp
            blocked_values: Dict[int, int] = {}
            blocked_total = 0
            for t in transitions:
                if t.action.name != action_name:
                    continue
                if t.diff.player_displacement is None and t.obs_before.best_player:
                    # Movement was blocked
                    p = t.obs_before.best_player
                    nr, nc = p.position[0] + dy, p.position[1] + dx
                    grid = t.obs_before.raw_grid
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        val = int(grid[nr, nc])
                        if val != 0:
                            blocked_values[val] = blocked_values.get(val, 0) + 1
                            blocked_total += 1

            for val, count in blocked_values.items():
                if count >= 2:
                    dir_name = ("up" if dy < 0 else "down" if dy > 0
                                else "left" if dx < 0 else "right")
                    rule_id = f"blocked_{dir_name}_by_v{val}"
                    rules.append(Rule(
                        rule_id=rule_id,
                        conditions=[PlayerExists(), AdjacentToValue(val)],
                        operator_kind="move",
                        effects=[NoEffect()],
                        support=count,
                        contradictions=0,
                    ))

        return rules

    def _propose_removal_rules(
        self, transitions: List[TransitionRecord]
    ) -> List[Rule]:
        """Propose rules about object removal."""
        rules: List[Rule] = []
        removal_counts: Dict[Tuple[str, int], int] = {}  # (action, value) → count

        for t in transitions:
            if not t.diff.removed_objects:
                continue
            for obj in t.obs_before.objects:
                if obj.object_id in t.diff.removed_objects:
                    key = (t.action.name, obj.value)
                    removal_counts[key] = removal_counts.get(key, 0) + 1

        for (action_name, val), count in removal_counts.items():
            if count >= 2:
                rule_id = f"remove_v{val}_with_{action_name}"
                rules.append(Rule(
                    rule_id=rule_id,
                    conditions=[ObjectExists(val)],
                    operator_kind="interact",
                    effects=[RemovesObject(val)],
                    support=count,
                    contradictions=0,
                ))

        return rules

    # ─── Verification ───────────────────────────────────────────

    def _propose_transform_rules(
        self, transitions: List[TransitionRecord]
    ) -> List[Rule]:
        """Propose rules about repeated large grid transformations."""
        rules: List[Rule] = []
        totals: Dict[str, int] = {}
        transforms: Dict[str, int] = {}

        for t in transitions:
            action_name = t.action.name
            totals[action_name] = totals.get(action_name, 0) + 1
            if t.diff.num_changed >= 5 and not t.diff.game_over:
                transforms[action_name] = transforms.get(action_name, 0) + 1

        for action_name, support in transforms.items():
            total = totals.get(action_name, support)
            if support >= MIN_SUPPORT_FOR_RULE and support / max(total, 1) >= 0.6:
                rule_id = f"global_transform_with_{action_name}"
                rules.append(Rule(
                    rule_id=rule_id,
                    conditions=[ObjectExists()],
                    operator_kind="global_transform",
                    effects=[GlobalGridChange(min_cells=5)],
                    support=support,
                    contradictions=max(0, total - support),
                ))

        return rules

    def _verify_against_recent(
        self, recent: List[TransitionRecord]
    ) -> None:
        """Check existing rules against recent transitions."""
        for r in self.rules.values():
            for t in recent:
                if not r.matches_context(t.obs_before):
                    continue
                # Check if effects match
                matched = False
                for e in r.effects:
                    if hasattr(e, "matches") and e.matches(t.diff):
                        matched = True
                        break
                if matched:
                    r.support += 1
                else:
                    r.contradictions += 1

                r.confidence = r.support / max(r.support + r.contradictions, 1)

    # ─── Queries ────────────────────────────────────────────────

    def predict_danger(self, obs: GameObservation) -> List[Rule]:
        """Return death rules whose conditions are met."""
        return [
            r for r in self.rules.values()
            if any(isinstance(e, GameOverEffect) for e in r.effects)
            and r.matches_context(obs)
            and r.confidence > 0.5
        ]

    def predict_win(self, obs: GameObservation) -> List[Rule]:
        """Return completion rules whose conditions are met."""
        return [
            r for r in self.rules.values()
            if any(isinstance(e, LevelCompleteEffect) for e in r.effects)
            and r.matches_context(obs)
            and r.confidence > 0.4
        ]

    def get_blocking_rules(self) -> List[Rule]:
        """Return rules about blocked movement."""
        return [
            r for r in self.rules.values()
            if any(isinstance(e, NoEffect) for e in r.effects)
            and r.confidence > 0.5
        ]

    def high_confidence_rules(self, threshold: float = 0.7) -> List[Rule]:
        return [r for r in self.rules.values() if r.confidence >= threshold]

    def summary(self) -> str:
        lines = []
        for r in sorted(self.rules.values(),
                        key=lambda r: r.confidence, reverse=True):
            lines.append(f"  {r.rule_id}: conf={r.confidence:.2f} "
                         f"(+{r.support}/-{r.contradictions})")
        return "\n".join(lines) if lines else "  (no rules)"
