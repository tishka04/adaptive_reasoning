"""Online structural-break detection and contextual theory revision.

SAGE.9q treats a learned theory as conditionally valid, not universal.  A
regime is identified only from the live structural layout.  A break requires
both a novel/unconfirmed structural context and repeated failures of actions
that the active theory predicted would reduce its own violations.

The detector never knows a game id, level number, target coordinate, or answer
sequence.  When a break is supported it suspends the old rule only for that
structural signature, creates explicit alternative rule families, and allows
them to become policies only after a terminal transition.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

from .online_terminal_relational_stencil import StencilTheoryAssessment


RuleTuple = Tuple[Tuple[str, bool], ...]


@dataclass
class StructuralRevisionHypothesis:
    """One explicit relation family awaiting an online terminal test."""

    hypothesis_id: str
    structural_signature: str
    rules: RuleTuple
    status: str = "candidate"
    actions: int = 0
    terminal_confirmations: int = 0
    terminal_refutations: int = 0
    nonterminal_refutations: int = 0
    discriminating_actions: int = 0
    cumulative_disagreement: float = 0.0
    generation_index: int = 0

    def rule_map(self) -> Dict[str, bool]:
        return dict(self.rules)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "structural_signature": self.structural_signature,
            "rules": {
                marker: (
                    "equal_to_center"
                    if desired_equal
                    else "different_from_center"
                )
                for marker, desired_equal in self.rules
            },
            "status": self.status,
            "actions": self.actions,
            "terminal_confirmations": self.terminal_confirmations,
            "terminal_refutations": self.terminal_refutations,
            "nonterminal_refutations": self.nonterminal_refutations,
            "discriminating_actions": self.discriminating_actions,
            "cumulative_disagreement": self.cumulative_disagreement,
            "generation_index": self.generation_index,
        }


@dataclass
class StructuralRegimeMemory:
    """Conditional validity and revision state for one visual regime."""

    structural_signature: str
    structural_family_signature: str = ""
    status: str = "provisional"
    observations: int = 0
    terminal_successes: int = 0
    prediction_residuals: int = 0
    consecutive_residuals: int = 0
    peak_consecutive_residuals: int = 0
    successful_predictions: int = 0
    no_effect_residuals: int = 0
    consecutive_no_effect_residuals: int = 0
    terminal_condition_residuals: int = 0
    break_detections: int = 0
    hypotheses: Dict[str, StructuralRevisionHypothesis] = field(
        default_factory=dict
    )
    active_hypothesis_id: str = ""
    confirmed_hypothesis_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structural_signature": self.structural_signature,
            "structural_family_signature": (
                self.structural_family_signature
            ),
            "status": self.status,
            "observations": self.observations,
            "terminal_successes": self.terminal_successes,
            "prediction_residuals": self.prediction_residuals,
            "consecutive_residuals": self.consecutive_residuals,
            "peak_consecutive_residuals": (
                self.peak_consecutive_residuals
            ),
            "successful_predictions": self.successful_predictions,
            "no_effect_residuals": self.no_effect_residuals,
            "consecutive_no_effect_residuals": (
                self.consecutive_no_effect_residuals
            ),
            "terminal_condition_residuals": (
                self.terminal_condition_residuals
            ),
            "break_detections": self.break_detections,
            "active_hypothesis_id": self.active_hypothesis_id,
            "confirmed_hypothesis_id": self.confirmed_hypothesis_id,
            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis in self.hypotheses.values()
            ],
        }


@dataclass
class StructuralRegimeFamilyMemory:
    """A scale/count/position-invariant family of structural regimes."""

    family_signature: str
    member_regimes: set[str] = field(default_factory=set)
    rule_votes: Counter[RuleTuple] = field(default_factory=Counter)
    confirmed_rules: RuleTuple = ()
    source_regime_signature: str = ""
    transferred_regimes: set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_signature": self.family_signature,
            "member_regimes": sorted(self.member_regimes),
            "confirmed_rules": {
                marker: (
                    "equal_to_center"
                    if desired_equal
                    else "different_from_center"
                )
                for marker, desired_equal in self.confirmed_rules
            },
            "source_regime_signature": self.source_regime_signature,
            "transferred_regimes": sorted(self.transferred_regimes),
            "rule_votes": [
                {
                    "rules": {
                        marker: (
                            "equal_to_center"
                            if desired_equal
                            else "different_from_center"
                        )
                        for marker, desired_equal in rules
                    },
                    "votes": int(votes),
                }
                for rules, votes in self.rule_votes.items()
            ],
        }


@dataclass
class StructuralTheoryProgram:
    """A terminally grounded theory coupled to its relational policy."""

    theory_id: str
    kind: str
    rules: RuleTuple
    family_signature: str = ""
    source_regime_signature: str = ""
    terminal_confirmations: int = 0
    activations: int = 0
    reactivations: int = 0
    policy_actions: int = 0
    transferred_policy_actions: int = 0
    regime_signatures: set[str] = field(default_factory=set)

    def rule_map(self) -> Dict[str, bool]:
        return dict(self.rules)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "kind": self.kind,
            "rules": {
                marker: (
                    "equal_to_center"
                    if desired_equal
                    else "different_from_center"
                )
                for marker, desired_equal in self.rules
            },
            "family_signature": self.family_signature,
            "source_regime_signature": self.source_regime_signature,
            "terminal_confirmations": self.terminal_confirmations,
            "activations": self.activations,
            "reactivations": self.reactivations,
            "policy_actions": self.policy_actions,
            "transferred_policy_actions": (
                self.transferred_policy_actions
            ),
            "regime_signatures": sorted(self.regime_signatures),
        }


@dataclass(frozen=True)
class StructuralPolicyResolution:
    """The theory-policy program authorized for one live regime."""

    theory_id: str
    rules: RuleTuple
    source: str
    structural_signature: str
    structural_family_signature: str
    transferred: bool = False
    reactivated: bool = False

    def rule_map(self) -> Dict[str, bool]:
        return dict(self.rules)


@dataclass(frozen=True)
class StructuralBreakObservation:
    """Auditable result of one theory prediction check."""

    structural_signature: str
    prediction_checked: bool
    prediction_failed: bool
    terminal_condition_failed: bool
    actual_reduction: int
    consecutive_residuals: int
    break_detected: bool
    old_theory_suspended: bool


class OnlineStructuralBreakDetector:
    """Detect repeated local theory failure and open contextual revision."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        condition_memory_by_regime: bool = True,
        minimum_consecutive_residuals: int = 3,
        minimum_terminal_confirmations: int = 2,
        max_revision_hypotheses: int = 3,
        max_actions_per_hypothesis: int = 48,
        enable_active_hypothesis_arbitration: bool = True,
        enable_regime_abstraction: bool = True,
        enable_hierarchical_theory_composition: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.condition_memory_by_regime = bool(
            condition_memory_by_regime
        )
        self.minimum_consecutive_residuals = max(
            2,
            int(minimum_consecutive_residuals),
        )
        self.minimum_terminal_confirmations = max(
            2,
            int(minimum_terminal_confirmations),
        )
        self.max_revision_hypotheses = max(
            1,
            int(max_revision_hypotheses),
        )
        self.max_actions_per_hypothesis = max(
            1,
            int(max_actions_per_hypothesis),
        )
        self.enable_active_hypothesis_arbitration = bool(
            enable_active_hypothesis_arbitration
        )
        self.enable_regime_abstraction = bool(
            enable_regime_abstraction
        )
        self.enable_hierarchical_theory_composition = bool(
            enable_hierarchical_theory_composition
        )
        self._regimes: Dict[str, StructuralRegimeMemory] = {}
        self._families: Dict[str, StructuralRegimeFamilyMemory] = {}
        self._theory_programs: Dict[str, StructuralTheoryProgram] = {}
        self._active_theory_id = ""
        self._states_assessed = 0
        self._topology_novelties = 0
        self._prediction_checks = 0
        self._prediction_residuals = 0
        self._successful_predictions = 0
        self._terminal_condition_residuals = 0
        self._breaks_detected = 0
        self._old_theory_suspensions = 0
        self._old_theory_blocks = 0
        self._hypotheses_generated = 0
        self._revision_actions = 0
        self._revision_confirmations = 0
        self._revision_refutations = 0
        self._contextual_policy_actions = 0
        self._arbitration_decisions = 0
        self._discriminating_experiments = 0
        self._cumulative_disagreement = 0.0
        self._unactionable_hypotheses_refuted = 0
        self._family_transfers = 0
        self._family_transfer_actions = 0
        self._family_rule_conflicts = 0
        self._theory_switches = 0
        self._theory_reactivations = 0

    def note_state(
        self,
        assessment: StencilTheoryAssessment,
    ) -> StructuralRegimeMemory | None:
        """Register a structural context without assigning goal semantics."""
        if (
            not self.enabled
            or not assessment.applicable
            or not assessment.structural_signature
        ):
            return None
        self._states_assessed += 1
        signature = assessment.structural_signature
        regime = self._regimes.get(signature)
        if regime is None:
            regime = StructuralRegimeMemory(
                structural_signature=signature,
                structural_family_signature=(
                    assessment.structural_family_signature
                ),
            )
            self._regimes[signature] = regime
            self._topology_novelties += 1
        elif (
            not regime.structural_family_signature
            and assessment.structural_family_signature
        ):
            regime.structural_family_signature = (
                assessment.structural_family_signature
            )
        if (
            self.enable_regime_abstraction
            and regime.structural_family_signature
        ):
            family = self._families.setdefault(
                regime.structural_family_signature,
                StructuralRegimeFamilyMemory(
                    family_signature=(
                        regime.structural_family_signature
                    ),
                ),
            )
            family.member_regimes.add(signature)
        regime.observations += 1
        return regime

    def observe_base_prediction(
        self,
        *,
        before: StencilTheoryAssessment,
        after: StencilTheoryAssessment,
        predicted_reduction: int,
        action_no_effect: bool,
        terminal_success: bool,
        base_rules: Mapping[str, bool],
    ) -> StructuralBreakObservation:
        """Update drift evidence from one action selected by the old theory."""
        regime = self.note_state(before)
        if regime is None:
            return StructuralBreakObservation(
                structural_signature="",
                prediction_checked=False,
                prediction_failed=False,
                terminal_condition_failed=False,
                actual_reduction=0,
                consecutive_residuals=0,
                break_detected=False,
                old_theory_suspended=False,
            )
        signature = before.structural_signature
        if terminal_success:
            regime.status = "compatible"
            regime.terminal_successes += 1
            regime.consecutive_residuals = 0
            regime.consecutive_no_effect_residuals = 0
            regime.terminal_condition_residuals = 0
            return StructuralBreakObservation(
                structural_signature=signature,
                prediction_checked=False,
                prediction_failed=False,
                terminal_condition_failed=False,
                actual_reduction=max(0, int(predicted_reduction)),
                consecutive_residuals=0,
                break_detected=False,
                old_theory_suspended=False,
            )

        same_regime = bool(
            after.applicable
            and after.structural_signature == signature
        )
        if not same_regime or int(predicted_reduction) <= 0:
            return StructuralBreakObservation(
                structural_signature=signature,
                prediction_checked=False,
                prediction_failed=False,
                terminal_condition_failed=False,
                actual_reduction=0,
                consecutive_residuals=regime.consecutive_residuals,
                break_detected=False,
                old_theory_suspended=self.is_suspended(signature),
            )

        self._prediction_checks += 1
        actual_reduction = (
            int(before.total_violations)
            - int(after.total_violations)
        )
        prediction_failed = actual_reduction <= 0
        terminal_condition_failed = bool(
            not prediction_failed
            and int(before.total_violations) > 0
            and int(after.total_violations) == 0
        )
        break_detected = False
        if prediction_failed:
            self._prediction_residuals += 1
            regime.prediction_residuals += 1
            regime.consecutive_residuals += 1
            if action_no_effect:
                regime.no_effect_residuals += 1
                regime.consecutive_no_effect_residuals += 1
            else:
                regime.consecutive_no_effect_residuals = 0
            regime.peak_consecutive_residuals = max(
                regime.peak_consecutive_residuals,
                regime.consecutive_residuals,
            )
            if (
                regime.status in {"provisional", "compatible"}
                and regime.consecutive_residuals
                >= self.minimum_consecutive_residuals
                and (
                    regime.status == "provisional"
                    or regime.consecutive_no_effect_residuals
                    >= self.minimum_consecutive_residuals
                )
            ):
                regime.status = "broken"
                regime.break_detections += 1
                self._breaks_detected += 1
                break_detected = True
                self._compile_revision_hypotheses(
                    regime,
                    base_rules,
                )
                if self.condition_memory_by_regime:
                    self._old_theory_suspensions += 1
        else:
            self._successful_predictions += 1
            regime.successful_predictions += 1
            regime.consecutive_residuals = 0
            regime.consecutive_no_effect_residuals = 0
            # A new topology is not evidence of a new theory by itself.
            # Once the old theory has made repeated correct local
            # predictions here, classify the regime as compatible.  Any
            # later break must then also exhibit repeated action no-effects,
            # preventing a short noisy burst from erasing useful knowledge.
            if (
                regime.status == "provisional"
                and regime.successful_predictions >= 2
            ):
                regime.status = "compatible"
            if terminal_condition_failed:
                regime.terminal_condition_residuals += 1
                self._terminal_condition_residuals += 1
                if (
                    regime.status in {"provisional", "compatible"}
                    and regime.terminal_condition_residuals
                    >= self.minimum_consecutive_residuals
                ):
                    regime.status = "broken"
                    regime.break_detections += 1
                    self._breaks_detected += 1
                    break_detected = True
                    self._compile_revision_hypotheses(
                        regime,
                        base_rules,
                    )
                    if self.condition_memory_by_regime:
                        self._old_theory_suspensions += 1

        return StructuralBreakObservation(
            structural_signature=signature,
            prediction_checked=True,
            prediction_failed=prediction_failed,
            terminal_condition_failed=terminal_condition_failed,
            actual_reduction=actual_reduction,
            consecutive_residuals=regime.consecutive_residuals,
            break_detected=break_detected,
            old_theory_suspended=self.is_suspended(signature),
        )

    def is_suspended(self, structural_signature: str) -> bool:
        """Whether the old theory is blocked in exactly this regime."""
        regime = self._regimes.get(str(structural_signature))
        family_rules = self._family_rules(regime)
        suspended = bool(
            self.enabled
            and self.condition_memory_by_regime
            and regime is not None
            and regime.status == "broken"
            and not regime.confirmed_hypothesis_id
            and not family_rules
        )
        return suspended

    def note_old_theory_block(self) -> None:
        self._old_theory_blocks += 1

    def revision_hypothesis(
        self,
        structural_signature: str,
    ) -> StructuralRevisionHypothesis | None:
        """Return the highest-value bounded candidate for this regime."""
        hypotheses = self.revision_hypotheses(structural_signature)
        return hypotheses[0] if hypotheses else None

    def revision_hypotheses(
        self,
        structural_signature: str,
    ) -> Tuple[StructuralRevisionHypothesis, ...]:
        """Rank live candidates by evidence, cost, and experiment coverage."""
        regime = self._regimes.get(str(structural_signature))
        if (
            regime is None
            or not self.condition_memory_by_regime
            or regime.status != "broken"
        ):
            return ()
        candidates = [
            hypothesis
            for hypothesis in regime.hypotheses.values()
            if (
                hypothesis.status == "candidate"
                and hypothesis.actions
                < self.max_actions_per_hypothesis
            )
        ]
        if not self.enable_active_hypothesis_arbitration:
            return tuple(candidates)
        candidates.sort(
            key=lambda hypothesis: (
                -hypothesis.terminal_confirmations,
                hypothesis.actions,
                hypothesis.discriminating_actions,
                -hypothesis.cumulative_disagreement,
                hypothesis.generation_index,
            )
        )
        return tuple(candidates)

    def committed_revision_hypothesis(
        self,
        structural_signature: str,
    ) -> StructuralRevisionHypothesis | None:
        """Return the experiment that must finish before re-arbitration."""
        regime = self._regimes.get(str(structural_signature))
        if regime is None or not regime.active_hypothesis_id:
            return None
        hypothesis = regime.hypotheses.get(
            regime.active_hypothesis_id
        )
        if (
            hypothesis is None
            or hypothesis.status != "candidate"
            or hypothesis.actions >= self.max_actions_per_hypothesis
        ):
            regime.active_hypothesis_id = ""
            return None
        return hypothesis

    def revision_hypothesis_rules(
        self,
        structural_signature: str,
    ) -> Dict[str, Dict[str, bool]]:
        """Return every live candidate for one discriminating decision."""
        return {
            hypothesis.hypothesis_id: hypothesis.rule_map()
            for hypothesis in self.revision_hypotheses(
                structural_signature
            )
        }

    def confirmed_revision_rules(
        self,
        structural_signature: str,
    ) -> Dict[str, bool]:
        regime = self._regimes.get(str(structural_signature))
        if regime is None:
            return {}
        if regime.confirmed_hypothesis_id:
            hypothesis = regime.hypotheses.get(
                regime.confirmed_hypothesis_id
            )
            if hypothesis is not None:
                return hypothesis.rule_map()
        return dict(self._family_rules(regime))

    def note_revision_action(
        self,
        hypothesis_id: str,
        *,
        discriminating: bool = False,
        disagreement_score: float = 0.0,
    ) -> None:
        hypothesis = self._find_hypothesis(hypothesis_id)
        if hypothesis is None:
            return
        hypothesis.actions += 1
        self._revision_actions += 1
        if discriminating:
            regime = self._regimes.get(hypothesis.structural_signature)
            if regime is not None:
                regime.active_hypothesis_id = hypothesis.hypothesis_id
            hypothesis.discriminating_actions += 1
            hypothesis.cumulative_disagreement += float(
                disagreement_score
            )
            self._arbitration_decisions += 1
            self._discriminating_experiments += 1
            self._cumulative_disagreement += float(
                disagreement_score
            )

    def refute_unactionable_hypothesis(
        self,
        hypothesis_id: str,
    ) -> bool:
        """Reject a rule that cannot propose any resolving intervention."""
        hypothesis = self._find_hypothesis(hypothesis_id)
        if hypothesis is None or hypothesis.status != "candidate":
            return False
        hypothesis.status = "refuted"
        hypothesis.nonterminal_refutations += 1
        regime = self._regimes.get(hypothesis.structural_signature)
        if (
            regime is not None
            and regime.active_hypothesis_id == hypothesis.hypothesis_id
        ):
            regime.active_hypothesis_id = ""
        self._revision_refutations += 1
        self._unactionable_hypotheses_refuted += 1
        return True

    def note_contextual_policy_action(self) -> None:
        self._contextual_policy_actions += 1

    def observe_revision_outcome(
        self,
        *,
        hypothesis_id: str,
        after: StencilTheoryAssessment,
        terminal_success: bool,
        game_over: bool,
    ) -> str:
        """Promote only terminal success; refute completed nonterminal tests."""
        hypothesis = self._find_hypothesis(hypothesis_id)
        if hypothesis is None or hypothesis.status != "candidate":
            return ""
        regime = self._regimes.get(hypothesis.structural_signature)
        if regime is None:
            return ""
        if terminal_success:
            hypothesis.terminal_confirmations += 1
            regime.terminal_successes += 1
            if (
                hypothesis.terminal_confirmations
                < self.minimum_terminal_confirmations
            ):
                return "nominated"
            hypothesis.status = "confirmed"
            regime.active_hypothesis_id = ""
            regime.confirmed_hypothesis_id = hypothesis.hypothesis_id
            regime.status = "revised"
            self._revision_confirmations += 1
            self._register_confirmed_revision(regime, hypothesis)
            return "confirmed"
        if game_over:
            hypothesis.status = "refuted"
            regime.active_hypothesis_id = ""
            hypothesis.terminal_refutations += 1
            self._revision_refutations += 1
            return "refuted"
        if (
            after.applicable
            and after.structural_signature
            == hypothesis.structural_signature
            and (
                after.total_violations == 0
                or after.improving_actions == 0
            )
        ):
            hypothesis.status = "refuted"
            regime.active_hypothesis_id = ""
            hypothesis.nonterminal_refutations += 1
            self._revision_refutations += 1
            if after.total_violations > 0:
                self._unactionable_hypotheses_refuted += 1
            return "refuted"
        if hypothesis.actions >= self.max_actions_per_hypothesis:
            hypothesis.status = "inconclusive"
            regime.active_hypothesis_id = ""
            return "inconclusive"
        return "pending"

    def resolve_policy(
        self,
        *,
        assessment: StencilTheoryAssessment,
        base_rules: Mapping[str, bool],
    ) -> StructuralPolicyResolution | None:
        """Resolve R1 or a contextual Rn as one hierarchical program."""
        regime = self.note_state(assessment)
        if regime is None:
            return None
        rules: RuleTuple = ()
        source = ""
        transferred = False
        kind = "base"
        source_regime = regime.structural_signature

        if regime.confirmed_hypothesis_id:
            hypothesis = regime.hypotheses.get(
                regime.confirmed_hypothesis_id
            )
            if hypothesis is not None:
                rules = hypothesis.rules
                source = "exact_revision"
                kind = "revision"
        if not rules:
            family_rules = self._family_rules(regime)
            if family_rules:
                rules = family_rules
                source = "family_revision"
                kind = "revision"
                family = self._families.get(
                    regime.structural_family_signature
                )
                if family is not None:
                    source_regime = family.source_regime_signature
                    transferred = bool(
                        regime.structural_signature
                        != family.source_regime_signature
                    )
                    if (
                        transferred
                        and regime.structural_signature
                        not in family.transferred_regimes
                    ):
                        family.transferred_regimes.add(
                            regime.structural_signature
                        )
                        self._family_transfers += 1
                if regime.status == "broken":
                    regime.status = "transferred"
        if not rules:
            if self.is_suspended(regime.structural_signature):
                return None
            rules = _rule_tuple(base_rules)
            source = "base"
            kind = "base"
        if not rules:
            return None

        program = self._ensure_theory_program(
            kind=kind,
            rules=rules,
            family_signature=regime.structural_family_signature,
            source_regime_signature=source_regime,
        )
        program.regime_signatures.add(regime.structural_signature)
        reactivated = self._activate_theory_program(program)
        return StructuralPolicyResolution(
            theory_id=program.theory_id,
            rules=program.rules,
            source=source,
            structural_signature=regime.structural_signature,
            structural_family_signature=(
                regime.structural_family_signature
            ),
            transferred=transferred,
            reactivated=reactivated,
        )

    def note_policy_action(
        self,
        theory_id: str,
        *,
        transferred: bool = False,
    ) -> None:
        program = self._theory_programs.get(str(theory_id))
        if program is None:
            return
        program.policy_actions += 1
        if transferred:
            program.transferred_policy_actions += 1
            self._family_transfer_actions += 1
        if program.kind == "revision":
            self._contextual_policy_actions += 1

    def theory_rules(self, theory_id: str) -> Dict[str, bool]:
        program = self._theory_programs.get(str(theory_id))
        return {} if program is None else program.rule_map()

    def summary(self) -> Dict[str, Any]:
        hypotheses = [
            hypothesis
            for regime in self._regimes.values()
            for hypothesis in regime.hypotheses.values()
        ]
        return {
            "enabled": self.enabled,
            "condition_memory_by_regime": (
                self.condition_memory_by_regime
            ),
            "minimum_consecutive_residuals": (
                self.minimum_consecutive_residuals
            ),
            "minimum_terminal_confirmations": (
                self.minimum_terminal_confirmations
            ),
            "active_hypothesis_arbitration_enabled": (
                self.enable_active_hypothesis_arbitration
            ),
            "regime_abstraction_enabled": (
                self.enable_regime_abstraction
            ),
            "hierarchical_theory_composition_enabled": (
                self.enable_hierarchical_theory_composition
            ),
            "regimes": len(self._regimes),
            "states_assessed": self._states_assessed,
            "topology_novelties": self._topology_novelties,
            "prediction_checks": self._prediction_checks,
            "prediction_residuals": self._prediction_residuals,
            "successful_predictions": self._successful_predictions,
            "terminal_condition_residuals": (
                self._terminal_condition_residuals
            ),
            "peak_consecutive_residuals": max(
                (
                    regime.peak_consecutive_residuals
                    for regime in self._regimes.values()
                ),
                default=0,
            ),
            "breaks_detected": self._breaks_detected,
            "old_theory_suspensions": self._old_theory_suspensions,
            "old_theory_blocks": self._old_theory_blocks,
            "hypotheses_generated": self._hypotheses_generated,
            "revision_actions": self._revision_actions,
            "revision_confirmations": self._revision_confirmations,
            "revision_refutations": self._revision_refutations,
            "contextual_policy_actions": self._contextual_policy_actions,
            "arbitration_decisions": self._arbitration_decisions,
            "discriminating_experiments": (
                self._discriminating_experiments
            ),
            "cumulative_disagreement": self._cumulative_disagreement,
            "unactionable_hypotheses_refuted": (
                self._unactionable_hypotheses_refuted
            ),
            "regime_families": len(self._families),
            "family_transfers": self._family_transfers,
            "family_transfer_actions": self._family_transfer_actions,
            "family_rule_conflicts": self._family_rule_conflicts,
            "theory_programs": len(self._theory_programs),
            "theory_switches": self._theory_switches,
            "theory_reactivations": self._theory_reactivations,
            "active_theory_id": self._active_theory_id,
            "candidate_hypotheses": sum(
                hypothesis.status == "candidate"
                for hypothesis in hypotheses
            ),
            "confirmed_hypotheses": sum(
                hypothesis.status == "confirmed"
                for hypothesis in hypotheses
            ),
            "refuted_hypotheses": sum(
                hypothesis.status == "refuted"
                for hypothesis in hypotheses
            ),
            "regime_memory": [
                regime.to_dict()
                for regime in self._regimes.values()
            ],
            "regime_family_memory": [
                family.to_dict()
                for family in self._families.values()
            ],
            "theory_program_hierarchy": [
                program.to_dict()
                for program in self._theory_programs.values()
            ],
        }

    def _register_confirmed_revision(
        self,
        regime: StructuralRegimeMemory,
        hypothesis: StructuralRevisionHypothesis,
    ) -> None:
        family_signature = regime.structural_family_signature
        if self.enable_regime_abstraction and family_signature:
            family = self._families.setdefault(
                family_signature,
                StructuralRegimeFamilyMemory(
                    family_signature=family_signature,
                ),
            )
            family.member_regimes.add(regime.structural_signature)
            family.rule_votes[hypothesis.rules] += (
                hypothesis.terminal_confirmations
            )
            winner, _ = max(
                family.rule_votes.items(),
                key=lambda item: (item[1], item[0]),
            )
            if family.confirmed_rules and family.confirmed_rules != winner:
                self._family_rule_conflicts += 1
            family.confirmed_rules = winner
            if not family.source_regime_signature:
                family.source_regime_signature = (
                    regime.structural_signature
                )
        program = self._ensure_theory_program(
            kind="revision",
            rules=hypothesis.rules,
            family_signature=family_signature,
            source_regime_signature=regime.structural_signature,
        )
        program.terminal_confirmations += (
            hypothesis.terminal_confirmations
        )
        program.regime_signatures.add(regime.structural_signature)

    def _family_rules(
        self,
        regime: StructuralRegimeMemory | None,
    ) -> RuleTuple:
        if (
            not self.enable_regime_abstraction
            or regime is None
            or not regime.structural_family_signature
        ):
            return ()
        family = self._families.get(
            regime.structural_family_signature
        )
        return () if family is None else family.confirmed_rules

    def _ensure_theory_program(
        self,
        *,
        kind: str,
        rules: RuleTuple,
        family_signature: str,
        source_regime_signature: str,
    ) -> StructuralTheoryProgram:
        normalized_rules = _rule_tuple(dict(rules))
        identity_scope = (
            family_signature if kind == "revision" else "global"
        )
        theory_id = _theory_id(kind, normalized_rules, identity_scope)
        program = self._theory_programs.get(theory_id)
        if program is None:
            program = StructuralTheoryProgram(
                theory_id=theory_id,
                kind=kind,
                rules=normalized_rules,
                family_signature=(
                    family_signature if kind == "revision" else ""
                ),
                source_regime_signature=source_regime_signature,
            )
            self._theory_programs[theory_id] = program
        return program

    def _activate_theory_program(
        self,
        program: StructuralTheoryProgram,
    ) -> bool:
        if not self.enable_hierarchical_theory_composition:
            return False
        if self._active_theory_id == program.theory_id:
            return False
        reactivated = bool(program.activations > 0)
        if self._active_theory_id:
            self._theory_switches += 1
        program.activations += 1
        if reactivated:
            program.reactivations += 1
            self._theory_reactivations += 1
        self._active_theory_id = program.theory_id
        return reactivated

    def _compile_revision_hypotheses(
        self,
        regime: StructuralRegimeMemory,
        base_rules: Mapping[str, bool],
    ) -> None:
        markers = tuple(sorted(str(marker) for marker in base_rules))
        if not markers:
            return
        base = _rule_tuple(base_rules)
        alternatives = []
        for values in itertools.product((False, True), repeat=len(markers)):
            rules = tuple(zip(markers, values))
            if rules == base:
                continue
            alternatives.append(rules)
        alternatives.sort(
            key=lambda rules: (
                -sum(
                    desired != dict(base)[marker]
                    for marker, desired in rules
                ),
                rules,
            )
        )
        for rules in alternatives[: self.max_revision_hypotheses]:
            hypothesis_id = _hypothesis_id(
                regime.structural_signature,
                rules,
            )
            if hypothesis_id in regime.hypotheses:
                continue
            regime.hypotheses[hypothesis_id] = (
                StructuralRevisionHypothesis(
                    hypothesis_id=hypothesis_id,
                    structural_signature=(
                        regime.structural_signature
                    ),
                    rules=rules,
                    generation_index=self._hypotheses_generated,
                )
            )
            self._hypotheses_generated += 1

    def _find_hypothesis(
        self,
        hypothesis_id: str,
    ) -> StructuralRevisionHypothesis | None:
        wanted = str(hypothesis_id)
        for regime in self._regimes.values():
            hypothesis = regime.hypotheses.get(wanted)
            if hypothesis is not None:
                return hypothesis
        return None

    def hypothesis_rules(
        self,
        hypothesis_id: str,
    ) -> Dict[str, bool]:
        hypothesis = self._find_hypothesis(hypothesis_id)
        return {} if hypothesis is None else hypothesis.rule_map()


def _hypothesis_id(
    structural_signature: str,
    rules: RuleTuple,
) -> str:
    payload = f"{structural_signature}:{rules}"
    return "revision-" + hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()[:12]


def _rule_tuple(rules: Mapping[str, bool]) -> RuleTuple:
    return tuple(
        sorted(
            (str(marker), bool(desired_equal))
            for marker, desired_equal in rules.items()
        )
    )


def _theory_id(
    kind: str,
    rules: RuleTuple,
    identity_scope: str,
) -> str:
    payload = f"{kind}:{identity_scope}:{rules}"
    return "theory-" + hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()[:12]


__all__ = [
    "OnlineStructuralBreakDetector",
    "StructuralBreakObservation",
    "StructuralPolicyResolution",
    "StructuralRegimeFamilyMemory",
    "StructuralRegimeMemory",
    "StructuralRevisionHypothesis",
    "StructuralTheoryProgram",
]
