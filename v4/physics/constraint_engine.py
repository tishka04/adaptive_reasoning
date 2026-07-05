"""Constraint and rule induction for V4."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..schemas import Effect, ObservationV4, Predicate, Rule, TransitionRecord


class ConstraintEngine:
    """Induce local rules and constraints from observed transitions."""

    def __init__(self) -> None:
        self.rules: dict[str, Rule] = {}
        self._support: dict[str, int] = defaultdict(int)
        self._contradictions: dict[str, int] = defaultdict(int)

    def update(
        self,
        obs: ObservationV4,
        transition: TransitionRecord,
        operators,
    ) -> list[Rule]:
        before_obs = transition.metadata.get("obs_before")
        if before_obs is None:
            return list(self.rules.values())

        action = transition.action
        diff = transition.diff
        rules_added: list[Rule] = []

        if diff.game_over and before_obs.best_player is not None and diff.player_displacement:
            player = before_obs.best_player
            nr = player.position[0] + diff.player_displacement[0]
            nc = player.position[1] + diff.player_displacement[1]
            grid = obs.raw_grid
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                value = int(grid[nr, nc])
                if value != 0:
                    rule, meaningful = self._upsert(
                        Rule(
                            rule_id=f"death_on_{value}",
                            family="death",
                            conditions=[Predicate("adjacent_value", {"value": value})],
                            effect=Effect("game_over", {"value": value}),
                            ontology_tags=["avatar_world", "token_world"],
                        ),
                        success=True,
                    )
                    if meaningful:
                        rules_added.append(rule)

        if action.name.startswith("ACTION") and diff.is_noop and before_obs.best_player is not None:
            for operator in operators.get_movement_ops():
                if operator.primitive_action != action.name:
                    continue
                dy = int(operator.parameters.get("dy", 0))
                dx = int(operator.parameters.get("dx", 0))
                nr = before_obs.best_player.position[0] + dy
                nc = before_obs.best_player.position[1] + dx
                grid = before_obs.raw_grid
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    value = int(grid[nr, nc])
                    if value != 0:
                        rule, meaningful = self._upsert(
                            Rule(
                                rule_id=f"blocked_by_{value}",
                                family="blocking",
                                conditions=[Predicate("adjacent_value", {"value": value})],
                                effect=Effect("blocked", {"value": value}),
                                ontology_tags=["avatar_world", "physics"],
                            ),
                            success=True,
                        )
                        if meaningful:
                            rules_added.append(rule)

        if transition.action.x is not None:
            click_value = transition.metadata.get("click_value_before")
            if click_value not in (None, 0):
                success = diff.num_changed > 0
                rule, meaningful = self._upsert(
                    Rule(
                        rule_id=f"click_requires_{click_value}",
                        family="adjacency_requirement",
                        conditions=[Predicate("object_exists", {"value": int(click_value)})],
                        effect=Effect("click_effect", {"value": int(click_value)}),
                        ontology_tags=["click_world"],
                    ),
                    success=success,
                )
                if meaningful:
                    rules_added.append(rule)

        removed_values = transition.metadata.get("removed_values", {})
        for value, count in removed_values.items():
            if value == 0 or count <= 0:
                continue
            rule, meaningful = self._upsert(
                Rule(
                    rule_id=f"removes_{value}_{action.name}",
                    family="removal",
                    conditions=[Predicate("action", {"name": action.name})],
                    effect=Effect("removes_object", {"value": int(value)}),
                    ontology_tags=["token_world"],
                ),
                success=True,
            )
            if meaningful:
                rules_added.append(rule)

        if transition.metadata.get("phase_shift"):
            rule, meaningful = self._upsert(
                Rule(
                    rule_id=f"phase_shift_{action.name}",
                    family="phase",
                    conditions=[Predicate("action", {"name": action.name})],
                    effect=Effect("phase_shift", {}),
                    ontology_tags=["phase_world", "transform_world"],
                ),
                success=True,
            )
            if meaningful:
                rules_added.append(rule)

        return [rule for rule in rules_added if rule is not None]

    def _upsert(self, rule: Rule, success: bool) -> tuple[Rule, bool]:
        existing = self.rules.get(rule.rule_id)
        if existing is None and not success:
            self._contradictions[rule.rule_id] += 1
            rule.contradictions = self._contradictions[rule.rule_id]
            rule.confidence = 0.0
            rule.survival_score = self._survival_score(rule)
            return rule, False
        previous_confidence = existing.confidence if existing is not None else 0.0
        created = existing is None
        if existing is None:
            existing = rule
            self.rules[rule.rule_id] = existing

        if success:
            self._support[rule.rule_id] += 1
        else:
            self._contradictions[rule.rule_id] += 1

        support = self._support[rule.rule_id]
        contradictions = self._contradictions[rule.rule_id]
        existing.support = support
        existing.contradictions = contradictions
        total = support + contradictions
        existing.confidence = (support / max(total, 1)) * min(1.0, support / 6.0)
        existing.survival_score = self._survival_score(existing)
        meaningful = (created and success) or (previous_confidence < 0.35 <= existing.confidence)
        return existing, meaningful

    def prune(self, max_active: int = 6) -> list[str]:
        if len(self.rules) <= max_active:
            for rule in self.rules.values():
                rule.survival_score = self._survival_score(rule)
            return []

        scored = sorted(
            ((self._survival_score(rule), rule_id, rule) for rule_id, rule in self.rules.items()),
            key=lambda item: item[0],
            reverse=True,
        )
        keep_ids: list[str] = []
        for family in ("death", "blocking", "removal", "phase", "adjacency_requirement"):
            family_items = [item for item in scored if item[2].family == family]
            if family_items:
                keep_ids.append(family_items[0][1])
        for _, rule_id, _ in scored:
            if rule_id not in keep_ids:
                keep_ids.append(rule_id)
            if len(keep_ids) >= max_active:
                break
        removed = [rule_id for rule_id in list(self.rules) if rule_id not in keep_ids]
        for rule_id in removed:
            self.rules.pop(rule_id, None)
        return removed

    def _survival_score(self, rule: Rule) -> float:
        contradiction_rate = rule.contradictions / max(rule.support + rule.contradictions, 1)
        score = (
            0.45 * rule.confidence
            + 0.20 * min(rule.support / 5.0, 1.0)
            - 0.45 * contradiction_rate
        )
        if rule.family in {"death", "blocking"}:
            score += 0.05
        rule.survival_score = score
        return score
