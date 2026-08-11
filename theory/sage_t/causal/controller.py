"""Fail-closed target controller backed only by the causal-program posterior."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from theory.sage_t.compiler import compile_observation, compile_transition_record
from theory.sage_t.contracts import normalized_action_candidates
from theory.sage_t.controller import SageTConfig, SageTMode
from v3.schemas import GameObservation, TransitionRecord

from .adapters import (
    CausalProgramProposal,
    CausalProposalCoordinator,
    causal_state_from_abstract,
    grounded_action_from_legacy,
    transition_evidence_from_observed,
)
from .contracts import ActionProgram, CausalProgram, GroundedAction
from .decision import CausalDecision
from .posterior import CausalPosterior
from .runtime import CausalRuntime


@dataclass(frozen=True)
class CausalSageTArbitration:
    action_name: str
    action_data: Mapping[str, Any]
    applied: bool
    requested_mode: str
    effective_mode: str
    reason: str
    decision: CausalDecision | None = None
    posterior: Mapping[str, Any] = field(default_factory=dict)


class CausalSageTController:
    """Drop-in controller target with no legacy belief store.

    It can be passed through the existing ``sage_t_controller`` injection point.
    Default/shadow behavior never changes the historical action.
    """

    def __init__(
        self,
        *,
        programs: Sequence[CausalProgram] = (),
        config: SageTConfig | None = None,
        runtime: CausalRuntime | None = None,
        memory_path: str | Path | None = None,
    ) -> None:
        self.config = config or SageTConfig()
        self.runtime = runtime or CausalRuntime(memory_path=memory_path)
        if programs:
            self.runtime.seed(tuple(programs))
        self._interventions_this_reset = 0
        self._intervened_contexts: set[str] = set()
        self._decisions = 0
        self._shadow_decisions = 0
        self._interventions = 0
        self._vetoes = 0
        self._trace_path = Path(self.config.trace_path) if self.config.trace_path else None
        self._trace_errors = 0
        self._proposal_coordinator = CausalProposalCoordinator()

    @property
    def effective_mode(self) -> SageTMode:
        return self.config.effective_mode

    @property
    def posterior(self) -> CausalPosterior:
        """Compatibility view used by ``UnifiedCognitiveController``."""
        return self.runtime.posterior

    def seed_programs(self, programs: Sequence[CausalProgram]) -> None:
        self.runtime.seed(tuple(programs))

    def admit_proposals(
        self,
        proposals: Sequence[CausalProgramProposal],
        *,
        action_catalog: Sequence[str],
    ) -> int:
        return self._proposal_coordinator.propose_into(
            posterior=self.runtime.posterior,
            proposals=proposals,
            action_catalog=action_catalog,
        )

    def decide(
        self,
        *,
        symbolic_action_name: str,
        symbolic_action_data: Mapping[str, Any] | None,
        observation: GameObservation,
        legal_actions: Sequence[Any],
        mechanic_theory: Any | None = None,
        goal_hypotheses: Sequence[Any] = (),
        route_memory: Any | None = None,
        danger_veto: Callable[[Any], bool] | None = None,
        protected_route: bool = False,
        causal_candidates: Sequence[ActionProgram] = (),
    ) -> CausalSageTArbitration:
        del mechanic_theory, goal_hypotheses, route_memory
        symbolic_name = str(symbolic_action_name).strip().upper()
        symbolic_data = dict(symbolic_action_data or {})
        if self.effective_mode is SageTMode.OFF:
            return self._fallback(symbolic_name, symbolic_data, "off")
        if not self.runtime.posterior.particles:
            return self._fallback(symbolic_name, symbolic_data, "empty_causal_posterior")
        try:
            abstract = compile_observation(observation)
            state = causal_state_from_abstract(abstract)
            normalized = normalized_action_candidates(legal_actions)
            programs = []
            baseline = GroundedAction(symbolic_name, symbolic_data)
            if protected_route:
                programs.append(ActionProgram((baseline,), source="protected_route"))
            programs.extend(causal_candidates)
            for action in normalized:
                grounded = grounded_action_from_legacy(action)
                source = "exact_route" if protected_route and grounded.key == baseline.key else "generic"
                programs.append(ActionProgram((grounded,), source=source))
            decision = self.runtime.decide(
                state,
                programs,
                danger_veto=(
                    None
                    if danger_veto is None
                    else lambda program: danger_veto(program.actions[0])
                ),
            )
        except (ArithmeticError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return self._fallback(
                symbolic_name,
                symbolic_data,
                f"causal_pipeline_error:{type(exc).__name__}",
            )
        self._decisions += 1
        chosen = decision.chosen
        applied = False
        reason = decision.reason
        output_name = symbolic_name
        output_data = symbolic_data
        if chosen is None:
            reason = decision.reason
        elif protected_route:
            reason = "protected_route"
        elif self.effective_mode is SageTMode.SHADOW:
            reason = "causal_shadow"
            self._shadow_decisions += 1
        else:
            action = chosen.action_program.actions[0]
            differs = action.action_name != symbolic_name or dict(action.action_data) != symbolic_data
            if self.effective_mode is SageTMode.BOUNDED:
                context = state.abstract_signature
                if chosen.terminal_risk > self.config.bounded_maximum_terminal_risk:
                    reason = "bounded_risk_veto"
                    self._vetoes += 1
                elif context in self._intervened_contexts:
                    reason = "bounded_context_budget"
                elif self._interventions_this_reset >= self.config.bounded_maximum_interventions_per_reset:
                    reason = "bounded_reset_budget"
                elif differs:
                    self._intervened_contexts.add(context)
                    self._interventions_this_reset += 1
                    applied = True
                    reason = "bounded_causal_override"
            elif differs:
                applied = True
                reason = "active_causal_override"
            if applied:
                output_name = action.action_name
                output_data = dict(action.action_data)
                self._interventions += 1
        arbitration = CausalSageTArbitration(
            action_name=output_name,
            action_data=output_data,
            applied=applied,
            requested_mode=self.config.requested_mode.value,
            effective_mode=self.effective_mode.value,
            reason=reason,
            decision=decision,
            posterior=self.runtime.posterior.snapshot(maximum_particles=None),
        )
        self._record(
            {
                "kind": "causal_decision",
                "reason": reason,
                "applied": applied,
                "protected_route": bool(protected_route),
                "baseline": {"name": symbolic_name, "data": symbolic_data},
                "selected": {"name": output_name, "data": output_data},
                "posterior": arbitration.posterior,
            }
        )
        return arbitration

    def observe_transition(self, record: TransitionRecord) -> None:
        if self.effective_mode is SageTMode.OFF or not self.runtime.posterior.particles:
            return
        try:
            observed = compile_transition_record(record)
            evidence = transition_evidence_from_observed(
                observed,
                game_id=str(getattr(record, "game_id", "")),
            )
            update = self.runtime.observe(evidence)
        except (ArithmeticError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._record(
                {
                    "kind": "causal_observation_rejected",
                    "reason": type(exc).__name__,
                }
            )
            return
        self._record(
            {
                "kind": "causal_observation",
                "evidence_id": evidence.evidence_id,
                "entropy_before": update.entropy_before,
                "entropy_after": update.entropy_after,
                "posterior": self.runtime.posterior.snapshot(maximum_particles=None),
            }
        )

    def start_branch(self, *, regime_index: int | None = None) -> None:
        del regime_index
        self._interventions_this_reset = 0
        self._intervened_contexts.clear()

    def note_level_change(self) -> None:
        self.start_branch()

    def summary(self) -> Mapping[str, Any]:
        return {
            "architecture": "sage-t-causal-program-v1",
            "requested_mode": self.config.requested_mode.value,
            "effective_mode": self.effective_mode.value,
            "decisions": self._decisions,
            "shadow_decisions": self._shadow_decisions,
            "interventions": self._interventions,
            "vetoes": self._vetoes,
            "posterior": self.runtime.posterior.snapshot(),
            "trace_errors": self._trace_errors,
        }

    def _fallback(
        self, action_name: str, action_data: Mapping[str, Any], reason: str
    ) -> CausalSageTArbitration:
        return CausalSageTArbitration(
            action_name=action_name,
            action_data=dict(action_data),
            applied=False,
            requested_mode=self.config.requested_mode.value,
            effective_mode=self.effective_mode.value,
            reason=reason,
            posterior=self.runtime.posterior.snapshot(),
        )

    def _record(self, payload: Mapping[str, Any]) -> None:
        if self._trace_path is None:
            return
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(payload), sort_keys=True, default=str))
                handle.write("\n")
        except (OSError, TypeError, ValueError):
            self._trace_errors += 1


__all__ = ["CausalSageTArbitration", "CausalSageTController"]
