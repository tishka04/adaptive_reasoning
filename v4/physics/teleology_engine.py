"""Teleological law induction for V4."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..schemas import Effect, ObservationV4, Predicate, Rule


class TeleologyEngine:
    """Infer soft laws about what seems to buy terminal progress."""

    def __init__(self) -> None:
        self._scores: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "support": 0.0,
                "weight": 0.0,
                "structural": 0.0,
                "terminal": 0.0,
                "contradictions": 0.0,
            }
        )
        self._speculative: dict[str, Rule] = {}
        self._validated: dict[str, Rule] = {}

    def update(self, obs: ObservationV4, memory: Any) -> None:
        transition = memory.fast.last_transition
        if transition is None:
            return

        sp_gain = float(transition.metadata.get("sp_delta", 0.0))
        tp_gain = float(transition.metadata.get("tp_delta", 0.0))
        removed_values: dict[int, int] = transition.metadata.get("removed_values", {})
        for value, count in removed_values.items():
            if value == 0 or count <= 0:
                continue
            self._bump(
                key=f"exhaust_class_{value}",
                family="teleology",
                conditions=[Predicate("object_exists", {"value": int(value)})],
                effect=Effect("terminal_progress", {"kind": "class_exhaustion", "value": int(value)}),
                structural_gain=sp_gain,
                terminal_gain=tp_gain,
                ontology_tags=["token_world", "closure"],
                memory=memory,
            )
            if sp_gain + tp_gain <= 0.01:
                self._contradict(f"exhaust_class_{value}")

        if obs.topology.unlocked_regions:
            self._bump(
                key="reach_new_region",
                family="teleology",
                conditions=[Predicate("topology_unlock", {})],
                effect=Effect("terminal_progress", {"kind": "region_unlock"}),
                structural_gain=sp_gain,
                terminal_gain=tp_gain,
                ontology_tags=["avatar_world", "field_world"],
                memory=memory,
            )
            if sp_gain + tp_gain <= 0.01:
                self._contradict("reach_new_region")

        if transition.diff.num_changed >= 8:
            self._bump(
                key="transform_before_completion",
                family="teleology",
                conditions=[Predicate("large_change", {})],
                effect=Effect("terminal_progress", {"kind": "transform"}),
                structural_gain=sp_gain,
                terminal_gain=tp_gain,
                ontology_tags=["transform_world"],
                memory=memory,
            )
            if sp_gain + tp_gain <= 0.01:
                self._contradict("transform_before_completion")

        if transition.action.x is not None and transition.diff.num_changed > 0:
            self._bump(
                key="unique_object_interaction",
                family="teleology",
                conditions=[Predicate("click_effective", {})],
                effect=Effect("terminal_progress", {"kind": "object_interaction"}),
                structural_gain=sp_gain,
                terminal_gain=tp_gain,
                ontology_tags=["click_world"],
                memory=memory,
            )
            if sp_gain + tp_gain <= 0.01:
                self._contradict("unique_object_interaction")

        if transition.metadata.get("prefix_replayed"):
            self._bump(
                key="successful_prefix_replay",
                family="teleology",
                conditions=[Predicate("ritual_replay", {})],
                effect=Effect("terminal_progress", {"kind": "prefix"}),
                structural_gain=sp_gain,
                terminal_gain=max(tp_gain, 0.02),
                ontology_tags=["closure", "sequence"],
                memory=memory,
            )
            if sp_gain + tp_gain <= 0.01:
                self._contradict("successful_prefix_replay")

    def _bump(
        self,
        key: str,
        family: str,
        conditions: list[Predicate],
        effect: Effect,
        structural_gain: float,
        terminal_gain: float,
        ontology_tags: list[str],
        memory: Any,
    ) -> None:
        entry = self._scores[key]
        entry["support"] += 1.0
        entry["structural"] += max(0.0, structural_gain)
        entry["terminal"] += max(0.0, terminal_gain)
        gain = 0.40 * max(0.0, structural_gain) + 0.60 * max(0.0, terminal_gain)
        entry["weight"] += gain
        support = int(entry["support"])
        avg_weight = entry["weight"] / max(entry["support"], 1.0)
        avg_terminal = entry["terminal"] / max(entry["support"], 1.0)
        contradictions = int(entry["contradictions"])
        contradiction_rate = contradictions / max(support + contradictions, 1)
        confidence = max(0.0, min(1.0, avg_weight * 2.0 - 0.4 * contradiction_rate))
        stage = (
            "validated"
            if support >= 3 and (avg_terminal >= 0.03 or avg_weight >= 0.08)
            else "speculative"
        )
        rule = Rule(
            rule_id=key,
            family=family,
            conditions=conditions,
            effect=effect,
            confidence=confidence,
            support=support,
            contradictions=contradictions,
            ontology_tags=ontology_tags,
            stage=stage,
            survival_score=self._survival_score(confidence, support, contradictions, stage),
        )
        if getattr(memory.game, "learning", None) is not None:
            rule = memory.game.learning.teleology_validator.refine(rule, memory)
            stage = rule.stage
        if stage == "validated":
            self._validated[key] = rule
            self._speculative.pop(key, None)
        else:
            self._speculative[key] = rule
            self._validated.pop(key, None)

    def _contradict(self, key: str) -> None:
        self._scores[key]["contradictions"] += 1.0

    def hypotheses(self) -> list[Rule]:
        return sorted(
            self._validated.values(),
            key=lambda rule: (rule.survival_score, rule.confidence, rule.support),
            reverse=True,
        )[:3]

    def speculative_hypotheses(self) -> list[Rule]:
        return sorted(
            self._speculative.values(),
            key=lambda rule: (rule.survival_score, rule.confidence, rule.support),
            reverse=True,
        )[:4]

    def evidence_hits(self, transition, obs) -> tuple[int, int]:
        validated_hits = sum(1 for rule in self.hypotheses() if self._rule_matches(rule, transition, obs))
        speculative_hits = sum(1 for rule in self.speculative_hypotheses() if self._rule_matches(rule, transition, obs))
        return validated_hits, speculative_hits

    def prune(self, max_validated: int = 3, max_speculative: int = 4) -> None:
        self._validated = {
            rule.rule_id: rule
            for rule in sorted(
                self._validated.values(),
                key=lambda item: item.survival_score,
                reverse=True,
            )[:max_validated]
        }
        self._speculative = {
            rule.rule_id: rule
            for rule in sorted(
                self._speculative.values(),
                key=lambda item: item.survival_score,
                reverse=True,
            )[:max_speculative]
        }

    def _rule_matches(self, rule: Rule, transition, obs) -> bool:
        if rule.rule_id.startswith("exhaust_class_"):
            value = int(rule.effect.args.get("value", -1))
            return transition.metadata.get("removed_values", {}).get(value, 0) > 0
        if rule.rule_id == "reach_new_region":
            return bool(obs.topology.unlocked_regions)
        if rule.rule_id == "transform_before_completion":
            return transition.diff.num_changed >= 8
        if rule.rule_id == "unique_object_interaction":
            return transition.action.x is not None and transition.diff.num_changed > 0
        if rule.rule_id == "successful_prefix_replay":
            return bool(transition.metadata.get("prefix_replayed"))
        return False

    def _survival_score(self, confidence: float, support: int, contradictions: int, stage: str) -> float:
        contradiction_rate = contradictions / max(support + contradictions, 1)
        stage_bonus = 0.10 if stage == "validated" else 0.0
        return (
            0.45 * confidence
            + 0.25 * min(support / 4.0, 1.0)
            + stage_bonus
            - 0.40 * contradiction_rate
        )
