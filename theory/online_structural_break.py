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
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

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
        }


@dataclass
class StructuralRegimeMemory:
    """Conditional validity and revision state for one visual regime."""

    structural_signature: str
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
    confirmed_hypothesis_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structural_signature": self.structural_signature,
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
            "confirmed_hypothesis_id": self.confirmed_hypothesis_id,
            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis in self.hypotheses.values()
            ],
        }


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
        self._regimes: Dict[str, StructuralRegimeMemory] = {}
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
            regime = StructuralRegimeMemory(signature)
            self._regimes[signature] = regime
            self._topology_novelties += 1
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
        suspended = bool(
            self.enabled
            and self.condition_memory_by_regime
            and regime is not None
            and regime.status == "broken"
            and not regime.confirmed_hypothesis_id
        )
        return suspended

    def note_old_theory_block(self) -> None:
        self._old_theory_blocks += 1

    def revision_hypothesis(
        self,
        structural_signature: str,
    ) -> StructuralRevisionHypothesis | None:
        """Return the next bounded candidate for this broken regime."""
        regime = self._regimes.get(str(structural_signature))
        if (
            regime is None
            or not self.condition_memory_by_regime
            or regime.status != "broken"
        ):
            return None
        for hypothesis in regime.hypotheses.values():
            if (
                hypothesis.status == "candidate"
                and hypothesis.actions
                < self.max_actions_per_hypothesis
            ):
                return hypothesis
        return None

    def confirmed_revision_rules(
        self,
        structural_signature: str,
    ) -> Dict[str, bool]:
        regime = self._regimes.get(str(structural_signature))
        if regime is None or not regime.confirmed_hypothesis_id:
            return {}
        hypothesis = regime.hypotheses.get(
            regime.confirmed_hypothesis_id
        )
        return {} if hypothesis is None else hypothesis.rule_map()

    def note_revision_action(
        self,
        hypothesis_id: str,
    ) -> None:
        hypothesis = self._find_hypothesis(hypothesis_id)
        if hypothesis is None:
            return
        hypothesis.actions += 1
        self._revision_actions += 1

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
            regime.confirmed_hypothesis_id = hypothesis.hypothesis_id
            regime.status = "revised"
            self._revision_confirmations += 1
            return "confirmed"
        if game_over:
            hypothesis.status = "refuted"
            hypothesis.terminal_refutations += 1
            self._revision_refutations += 1
            return "refuted"
        if (
            after.applicable
            and after.structural_signature
            == hypothesis.structural_signature
            and after.total_violations == 0
        ):
            hypothesis.status = "refuted"
            hypothesis.nonterminal_refutations += 1
            self._revision_refutations += 1
            return "refuted"
        if hypothesis.actions >= self.max_actions_per_hypothesis:
            hypothesis.status = "inconclusive"
            return "inconclusive"
        return "pending"

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
        }

    def _compile_revision_hypotheses(
        self,
        regime: StructuralRegimeMemory,
        base_rules: Mapping[str, bool],
    ) -> None:
        markers = tuple(sorted(str(marker) for marker in base_rules))
        if not markers:
            return
        base = tuple(
            (marker, bool(base_rules[marker]))
            for marker in markers
        )
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


__all__ = [
    "OnlineStructuralBreakDetector",
    "StructuralBreakObservation",
    "StructuralRegimeMemory",
    "StructuralRevisionHypothesis",
]
