"""Relational mechanic hypotheses and the competing-theory store (step A1).

This is the missing layer identified in CONSOLIDATION_PLAN.md: explicit,
*predictive* hypotheses about game mechanics that are tested against
observations and then confirmed / refuted / revised.

Unlike the v4_1 ``ActionHypothesis`` (a local belief re-converted into an
action score), a ``MechanicHypothesis`` carries a falsifiable PREDICTION over
the next observed effect. The disagreement between competing hypotheses is what
the DiscriminatingExperimentDesigner (A2) exploits, and the confirm/refute
outcome is what ``theory.epistemic_metrics`` scores.

To stay light and unit-testable, this module depends only on a small
``ObservedEffect`` (adaptable from a v3 ``FrameDiff``), not on perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .epistemic_metrics import (
    HypothesisRecord,
    HypothesisStatus,
    mechanic_key,
    normalize_operator_kind,
)
from .correspondence_hypothesis import (
    CorrespondenceHypothesis,
    CorrespondenceObservation,
    CorrespondenceRule,
)
from .precondition_hypothesis import (
    PreconditionHypothesis,
    PreconditionObservation,
)
from .promoted_relational_rule import PromotedRelationalRule
from .role_hypotheses import ActionRoleHypothesis, GoalFamilyHypothesis

# A change of >= this many cells is treated as a global/transform-like effect
# (matches the FALSE-noop threshold used by theory.ar25_oracle).
GLOBAL_MIN_CELLS = 50

# Evidence / decision policy (mirrors v3 BeliefDebugger thresholds).
MIN_EVIDENCE = 4
CONFIRM_CONFIDENCE = 0.70
REFUTE_CONTRA_RATIO = 0.60

# Candidate effect classes seeded as competing hypotheses per action.
CANDIDATE_KINDS = ("move", "global_transform", "noop", "click", "lethal")


@dataclass
class ObservedEffect:
    """Minimal, perception-free summary of one action's outcome."""

    num_changed: int = 0
    player_moved: bool = False
    game_over: bool = False
    level_complete: bool = False

    @property
    def is_noop(self) -> bool:
        return (
            self.num_changed == 0
            and not self.game_over
            and not self.level_complete
        )

    @classmethod
    def from_frame_diff(cls, diff: Any) -> "ObservedEffect":
        """Adapt a v3 ``FrameDiff`` (duck-typed) into an ObservedEffect."""
        return cls(
            num_changed=int(getattr(diff, "num_changed", 0) or 0),
            player_moved=getattr(diff, "player_displacement", None) is not None,
            game_over=bool(getattr(diff, "game_over", False)),
            level_complete=bool(getattr(diff, "level_complete", False)),
        )


def predicted_signal_holds(kind: str, effect: ObservedEffect) -> Optional[bool]:
    """Does the effect-class prediction hold for this observed effect?

    Returns True/False, or None when the class makes no falsifiable claim
    about this observation (so it is neither supported nor contradicted).
    """
    k = normalize_operator_kind(kind)
    if k == "move":
        return effect.player_moved
    if k == "global_transform":
        return effect.num_changed >= GLOBAL_MIN_CELLS
    if k == "noop":
        return effect.is_noop
    if k in ("click", "interact"):
        return 0 < effect.num_changed < GLOBAL_MIN_CELLS and not effect.player_moved
    if k == "lethal":
        return effect.game_over
    if k in ("win", "level_complete"):
        return effect.level_complete
    return None


@dataclass
class MechanicHypothesis:
    """A falsifiable claim: 'ACTION<n> has effect-class <kind>'."""

    action: str
    kind: str
    statement: str = ""
    support: int = 0
    contradictions: int = 0
    experiments_spent: int = 0
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    def __post_init__(self) -> None:
        self.action = str(self.action).upper()
        self.kind = normalize_operator_kind(self.kind)
        if not self.statement:
            self.statement = f"{self.action} has effect-class '{self.kind}'"

    @property
    def key(self) -> str:
        return mechanic_key(self.action, self.kind)

    @property
    def total_evidence(self) -> int:
        return self.support + self.contradictions

    @property
    def confidence(self) -> float:
        """support/total scaled by evidence sufficiency (v3 Operator style)."""
        total = self.total_evidence
        if total == 0:
            return 0.0
        return (self.support / total) * min(1.0, total / 8.0)

    def predicts(self, effect: ObservedEffect) -> Optional[bool]:
        return predicted_signal_holds(self.kind, effect)

    def observe(self, effect: ObservedEffect, *, was_experiment: bool = False) -> None:
        """Update evidence from one observation of this action's outcome."""
        if was_experiment:
            self.experiments_spent += 1
        holds = self.predicts(effect)
        if holds is None:
            return
        if holds:
            self.support += 1
        else:
            self.contradictions += 1
        self._recompute_status()

    def _recompute_status(self) -> None:
        total = self.total_evidence
        if total < MIN_EVIDENCE:
            self.status = HypothesisStatus.UNRESOLVED
            return
        contra_ratio = self.contradictions / max(1, total)
        if self.confidence >= CONFIRM_CONFIDENCE:
            self.status = HypothesisStatus.CONFIRMED
        elif contra_ratio >= REFUTE_CONTRA_RATIO:
            self.status = HypothesisStatus.REFUTED
        else:
            self.status = HypothesisStatus.UNRESOLVED

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=self.statement,
            status=self.status,
            support=self.support,
            contradictions=self.contradictions,
            experiments_spent=self.experiments_spent,
        )


class GameTheory:
    """A set of competing mechanic hypotheses, grouped by action.

    For each action we entertain several effect-class hypotheses at once
    (the 'competing theories'). Observations update all of them; the metric
    judges how well the agent converges on the true ones.
    """

    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        # action -> {kind -> MechanicHypothesis}
        self._by_action: Dict[str, Dict[str, MechanicHypothesis]] = {}
        # Human-facing semantics kept separate from effect-class hypotheses so
        # the experiment designer remains focused on low-level mechanics.
        self._roles_by_action: Dict[str, Dict[str, ActionRoleHypothesis]] = {}
        self._goal_families: Dict[str, GoalFamilyHypothesis] = {}
        self._correspondence_by_action: Dict[
            str,
            Dict[str, CorrespondenceHypothesis],
        ] = {}
        self._preconditions_by_target: Dict[
            str,
            Dict[str, PreconditionHypothesis],
        ] = {}
        # Generic live predictions enter this store only after an explicit
        # independent-context promotion gate.  Refuted rules remain auditable.
        self._promoted_relational_rules: Dict[str, PromotedRelationalRule] = {}

    # ── construction ────────────────────────────────────────────
    def seed_actions(
        self,
        actions: List[str],
        kinds: Optional[List[str]] = None,
    ) -> None:
        """Create competing hypotheses for each (action, candidate-kind)."""
        kinds = kinds or list(CANDIDATE_KINDS)
        for action in actions:
            action = str(action).upper()
            if action in ("RESET",):
                continue
            slot = self._by_action.setdefault(action, {})
            for kind in kinds:
                nk = normalize_operator_kind(kind)
                slot.setdefault(nk, MechanicHypothesis(action=action, kind=nk))

    def add_action_role(self, hypothesis: ActionRoleHypothesis) -> None:
        """Add or merge a human-facing action-role hypothesis."""
        action = str(hypothesis.action).upper()
        role = hypothesis.role
        slot = self._roles_by_action.setdefault(action, {})
        existing = slot.get(role)
        if existing is None:
            slot[role] = hypothesis
            return
        existing.evidence_for.extend(hypothesis.evidence_for)
        existing.evidence_against.extend(hypothesis.evidence_against)
        existing.prior_confidence = max(
            existing.prior_confidence,
            hypothesis.prior_confidence,
        )
        existing._recompute_status()

    def add_goal_family(self, hypothesis: GoalFamilyHypothesis) -> None:
        """Add or merge a goal-family hypothesis."""
        existing = self._goal_families.get(hypothesis.family)
        if existing is None:
            self._goal_families[hypothesis.family] = hypothesis
            return
        existing.evidence_for.extend(hypothesis.evidence_for)
        existing.evidence_against.extend(hypothesis.evidence_against)
        existing.prior_confidence = max(
            existing.prior_confidence,
            hypothesis.prior_confidence,
        )
        existing._recompute_status()

    def add_semantic_hypotheses(
        self,
        action_roles: List[ActionRoleHypothesis] | None = None,
        goal_families: List[GoalFamilyHypothesis] | None = None,
        correspondence: List[CorrespondenceHypothesis] | None = None,
        preconditions: List[PreconditionHypothesis] | None = None,
    ) -> None:
        """Add action-role and goal-family hypotheses to the theory store."""
        for hypothesis in action_roles or []:
            self.add_action_role(hypothesis)
        for hypothesis in goal_families or []:
            self.add_goal_family(hypothesis)
        for hypothesis in correspondence or []:
            self.add_correspondence(hypothesis)
        for hypothesis in preconditions or []:
            self.add_precondition(hypothesis)

    def add_correspondence(self, hypothesis: CorrespondenceHypothesis) -> None:
        """Add or merge a correspondence hypothesis."""
        action = str(hypothesis.action).upper()
        slot = self._correspondence_by_action.setdefault(action, {})
        existing = slot.get(hypothesis.key)
        if existing is None:
            slot[hypothesis.key] = hypothesis
            return
        existing.support += hypothesis.support
        existing.contradictions += hypothesis.contradictions
        existing.experiments_spent += hypothesis.experiments_spent
        existing.cumulative_match_delta += hypothesis.cumulative_match_delta
        existing.cumulative_global_delta += hypothesis.cumulative_global_delta
        existing._recompute_status()

    def add_precondition(self, hypothesis: PreconditionHypothesis) -> None:
        """Add or merge a rule-precondition hypothesis."""
        slot = self._preconditions_by_target.setdefault(hypothesis.target_rule, {})
        existing = slot.get(hypothesis.predicate)
        if existing is None:
            slot[hypothesis.predicate] = hypothesis
            return
        existing.evidence_for.extend(hypothesis.evidence_for)
        existing.evidence_against.extend(hypothesis.evidence_against)
        existing.experiments_spent += hypothesis.experiments_spent
        existing._recompute_status()

    def add_promoted_relational_rule(
        self,
        rule: PromotedRelationalRule,
    ) -> PromotedRelationalRule:
        """Register one independently confirmed generic relational rule."""
        existing = self._promoted_relational_rules.get(rule.key)
        if existing is None:
            self._promoted_relational_rules[rule.key] = rule
            return rule
        return existing

    # ── learning ────────────────────────────────────────────────
    def observe(
        self,
        action: str,
        effect: ObservedEffect,
        *,
        was_experiment: bool = False,
    ) -> None:
        """Feed one observed transition to every hypothesis about ``action``."""
        action = str(action).upper()
        slot = self._by_action.get(action)
        if slot is None:
            self.seed_actions([action])
            slot = self._by_action[action]
        for hyp in slot.values():
            hyp.observe(effect, was_experiment=was_experiment)
        for hyp in self._roles_by_action.get(action, {}).values():
            hyp.observe(effect, was_experiment=was_experiment)
        for hyp in self._goal_families.values():
            hyp.observe(effect, was_experiment=was_experiment)

    def observe_correspondence(
        self,
        observation: CorrespondenceObservation,
        *,
        was_experiment: bool = False,
    ) -> None:
        """Feed one relation-level transition signal to correspondence hypotheses."""
        action = str(observation.action).upper()
        for hyp in self._correspondence_by_action.get(action, {}).values():
            hyp.observe(observation, was_experiment=was_experiment)

    def observe_precondition(
        self,
        observation: PreconditionObservation,
        *,
        was_experiment: bool = False,
    ) -> None:
        """Feed one rule-applicability observation to precondition hypotheses."""
        for hyp in self._preconditions_by_target.get(observation.target_rule, {}).values():
            hyp.observe(observation, was_experiment=was_experiment)

    # ── queries ─────────────────────────────────────────────────
    def hypotheses(self) -> List[MechanicHypothesis]:
        return [h for slot in self._by_action.values() for h in slot.values()]

    def action_role_hypotheses(self) -> List[ActionRoleHypothesis]:
        return [
            h for slot in self._roles_by_action.values()
            for h in slot.values()
        ]

    def goal_family_hypotheses(self) -> List[GoalFamilyHypothesis]:
        return list(self._goal_families.values())

    def correspondence_hypotheses(self) -> List[CorrespondenceHypothesis]:
        return [
            h for slot in self._correspondence_by_action.values()
            for h in slot.values()
        ]

    def correspondence_rules(self) -> List[CorrespondenceRule]:
        rules: List[CorrespondenceRule] = []
        for hypothesis in self.correspondence_hypotheses():
            rule = hypothesis.to_rule()
            if rule is not None:
                rules.append(rule)
        return rules

    def precondition_hypotheses(self) -> List[PreconditionHypothesis]:
        return [
            h for slot in self._preconditions_by_target.values()
            for h in slot.values()
        ]

    def preconditions_for_rule(self, target_rule: str) -> List[PreconditionHypothesis]:
        return list(self._preconditions_by_target.get(str(target_rule), {}).values())

    def promoted_relational_hypotheses(self) -> List[PromotedRelationalRule]:
        """Return promoted rules at every revision status for audit/scoring."""
        return list(self._promoted_relational_rules.values())

    def promoted_relational_rules(self) -> List[PromotedRelationalRule]:
        """Return only currently confirmed rules available to option planning."""
        return [
            rule
            for rule in self.promoted_relational_hypotheses()
            if rule.status == HypothesisStatus.CONFIRMED
        ]

    def promoted_relational_rule(
        self,
        key: str,
    ) -> PromotedRelationalRule | None:
        return self._promoted_relational_rules.get(str(key))

    def for_action(self, action: str) -> List[MechanicHypothesis]:
        return list(self._by_action.get(str(action).upper(), {}).values())

    def unresolved_for_action(self, action: str) -> List[MechanicHypothesis]:
        return [
            h for h in self.for_action(action)
            if h.status == HypothesisStatus.UNRESOLVED
        ]

    def dominant(self, action: str) -> Optional[MechanicHypothesis]:
        """Highest-confidence confirmed hypothesis for an action (if any)."""
        confirmed = [
            h for h in self.for_action(action)
            if h.status == HypothesisStatus.CONFIRMED
        ]
        if not confirmed:
            return None
        return max(confirmed, key=lambda h: h.confidence)

    def actions(self) -> List[str]:
        return list(self._by_action.keys())

    # ── output for scoring ──────────────────────────────────────
    def to_ledger(self) -> List[HypothesisRecord]:
        """Emit a HypothesisRecord ledger for theory.epistemic_metrics."""
        records = [h.to_record() for h in self.hypotheses()]
        records.extend(h.to_record() for h in self.action_role_hypotheses())
        records.extend(h.to_record() for h in self.goal_family_hypotheses())
        records.extend(h.to_record() for h in self.correspondence_hypotheses())
        records.extend(h.to_record() for h in self.precondition_hypotheses())
        records.extend(h.to_record() for h in self.promoted_relational_hypotheses())
        return records

    def summary(self) -> Dict[str, Any]:
        confirmed = [h for h in self.hypotheses()
                     if h.status == HypothesisStatus.CONFIRMED]
        refuted = [h for h in self.hypotheses()
                   if h.status == HypothesisStatus.REFUTED]
        semantic_confirmed = [
            h for h in (
                self.action_role_hypotheses()
                + self.goal_family_hypotheses()
                + self.correspondence_hypotheses()
                + self.precondition_hypotheses()
                + self.promoted_relational_hypotheses()
            )
            if h.status == HypothesisStatus.CONFIRMED
        ]
        return {
            "actions": len(self._by_action),
            "hypotheses": len(self.hypotheses()),
            "action_role_hypotheses": len(self.action_role_hypotheses()),
            "goal_family_hypotheses": len(self.goal_family_hypotheses()),
            "correspondence_hypotheses": len(self.correspondence_hypotheses()),
            "precondition_hypotheses": len(self.precondition_hypotheses()),
            "promoted_relational_hypotheses": len(
                self.promoted_relational_hypotheses()
            ),
            "promoted_relational_rules": [
                rule.key for rule in self.promoted_relational_rules()
            ],
            "preconditions_confirmed": [
                h.key for h in self.precondition_hypotheses()
                if h.status == HypothesisStatus.CONFIRMED
            ],
            "correspondence_rules": [rule.key for rule in self.correspondence_rules()],
            "confirmed": [h.key for h in confirmed],
            "refuted": [h.key for h in refuted],
            "semantic_confirmed": [h.key for h in semantic_confirmed],
        }
