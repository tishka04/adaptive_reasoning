"""Fail-closed SAGE.T orchestration for replay, shadow and active modes."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from v3.schemas import GameObservation, TransitionRecord

from .compiler import compile_observation, compile_transition_record
from .consolidation import ConsolidationRegistry
from .contracts import ActionCandidate, normalized_action_candidates
from .decision import (
    BayesianDecision,
    CounterfactualDecisionEngine,
    SequenceAssessment,
)
from .executor import ProgramExecutor
from .posterior import ProgramPosterior
from .synthesis import (
    DeterministicFragmentProposer,
    FragmentProposal,
    ProgramAssembler,
)


class SageTMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    BOUNDED = "bounded"
    ACTIVE = "active"


@dataclass(frozen=True)
class SageTConfig:
    mode: str | SageTMode = SageTMode.OFF.value
    counterfactual_gate_passed: bool = False
    active_gate_passed: bool = False
    maximum_programs: int = 64
    maximum_sequences: int = 64
    maximum_particles_per_decision: int = 16
    ordinary_horizon: int = 3
    bounded_maximum_interventions_per_reset: int = 5
    bounded_maximum_terminal_risk: float = 0.05
    bounded_minimum_intervention_support: int = 0
    bounded_minimum_top_probability: float = 0.0
    trace_path: str = ""

    def __post_init__(self) -> None:
        SageTMode(_mode_value(self.mode))
        if not 1 <= int(self.maximum_programs) <= 64:
            raise ValueError("SAGE.T supports between 1 and 64 programs")
        if not 1 <= int(self.maximum_sequences) <= 64:
            raise ValueError("SAGE.T supports between 1 and 64 sequences")
        if not 1 <= int(self.maximum_particles_per_decision) <= 16:
            raise ValueError("SAGE.T supports between 1 and 16 decision particles")
        if not 1 <= int(self.ordinary_horizon) <= 3:
            raise ValueError("ordinary SAGE.T rollouts use horizon 1-3")
        if int(self.bounded_maximum_interventions_per_reset) < 0:
            raise ValueError("bounded intervention budget cannot be negative")
        if int(self.bounded_minimum_intervention_support) < 0:
            raise ValueError("bounded intervention support cannot be negative")
        if not 0.0 <= float(self.bounded_maximum_terminal_risk) <= 0.05:
            raise ValueError("bounded SAGE.T risk must be in [0, 0.05]")
        if not 0.0 <= float(self.bounded_minimum_top_probability) <= 1.0:
            raise ValueError("bounded top probability must be in [0, 1]")

    @property
    def requested_mode(self) -> SageTMode:
        return SageTMode(_mode_value(self.mode))

    @property
    def effective_mode(self) -> SageTMode:
        requested = self.requested_mode
        if requested in {SageTMode.OFF, SageTMode.SHADOW}:
            return requested
        if requested is SageTMode.BOUNDED:
            return (
                SageTMode.BOUNDED
                if self.counterfactual_gate_passed
                else SageTMode.SHADOW
            )
        if self.active_gate_passed and self.counterfactual_gate_passed:
            return SageTMode.ACTIVE
        if self.counterfactual_gate_passed:
            return SageTMode.BOUNDED
        return SageTMode.SHADOW


@dataclass(frozen=True)
class SageTArbitration:
    action_name: str
    action_data: Mapping[str, Any]
    applied: bool
    requested_mode: str
    effective_mode: str
    reason: str
    decision: BayesianDecision | None = None
    posterior: Mapping[str, Any] = field(default_factory=dict)


class SageTTraceWriter:
    """Append-only JSONL audit sink; failures never affect action selection."""

    def __init__(self, path: str | Path = "") -> None:
        self.path = Path(path) if str(path) else None
        self.errors = 0

    def write(self, record: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str))
                handle.write("\n")
        except (OSError, TypeError, ValueError):
            # Audit I/O must fail closed: retain the old controller decision.
            self.errors += 1


class SageTController:
    """Own the executable-program posterior and its authority gates."""

    def __init__(
        self,
        *,
        config: SageTConfig | None = None,
        executor: ProgramExecutor | None = None,
        proposer: DeterministicFragmentProposer | None = None,
        assembler: ProgramAssembler | None = None,
        posterior: ProgramPosterior | None = None,
        decision_engine: CounterfactualDecisionEngine | None = None,
        consolidation: ConsolidationRegistry | None = None,
    ) -> None:
        self.config = config or SageTConfig()
        self.executor = executor or ProgramExecutor()
        self.proposer = proposer or DeterministicFragmentProposer()
        self.assembler = assembler or ProgramAssembler(
            maximum_programs=self.config.maximum_programs
        )
        self.posterior = posterior or ProgramPosterior(
            executor=self.executor,
            maximum_particles=self.config.maximum_programs,
        )
        self.decision_engine = decision_engine or CounterfactualDecisionEngine(
            executor=self.executor,
            maximum_sequences=self.config.maximum_sequences,
            maximum_particles=self.config.maximum_particles_per_decision,
            ordinary_horizon=self.config.ordinary_horizon,
        )
        self.trace_writer = SageTTraceWriter(self.config.trace_path)
        self.consolidation = consolidation or ConsolidationRegistry()
        self._transitions = []
        self._latest_proposal = FragmentProposal(())
        self._available_action_names: tuple[str, ...] = ()
        self._needs_reassembly = True
        self._branch_index = 0
        self._regime_index = 0
        self._interventions_this_reset = 0
        self._intervened_contexts: set[str] = set()
        self._observed_contexts: set[str] = set()
        self._decisions = 0
        self._shadow_decisions = 0
        self._interventions = 0
        self._vetoes = 0
        self._audit = deque(maxlen=128)

    @property
    def effective_mode(self) -> SageTMode:
        return self.config.effective_mode

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
        danger_veto: Callable[[ActionCandidate], bool] | None = None,
        protected_route: bool = False,
    ) -> SageTArbitration:
        mode = self.effective_mode
        symbolic_name = str(symbolic_action_name).strip().upper()
        symbolic_data = dict(symbolic_action_data or {})
        if mode is SageTMode.OFF:
            return SageTArbitration(
                action_name=symbolic_name,
                action_data=symbolic_data,
                applied=False,
                requested_mode=self.config.requested_mode.value,
                effective_mode=mode.value,
                reason="off",
            )
        try:
            candidates = normalized_action_candidates(legal_actions)
        except (TypeError, ValueError):
            return self._fallback(
                symbolic_name,
                symbolic_data,
                reason="invalid_legal_candidates",
            )
        if not candidates:
            return self._fallback(
                symbolic_name,
                symbolic_data,
                reason="no_legal_candidates",
            )
        try:
            state = compile_observation(
                observation,
                regime_index=self._regime_index,
            )
            self._ensure_programs(
                state=state,
                candidates=candidates,
                mechanic_theory=mechanic_theory,
                goal_hypotheses=goal_hypotheses,
                route_memory=route_memory,
            )
        except (
            ArithmeticError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return self._fallback(
                symbolic_name,
                symbolic_data,
                reason="program_pipeline_error",
            )
        if not self.posterior.particles:
            return self._fallback(
                symbolic_name,
                symbolic_data,
                reason="no_complete_program",
            )
        before = self.posterior.snapshot(maximum_programs=None)
        try:
            decision = self.decision_engine.decide(
                self.posterior,
                state,
                candidates,
                memory_macros=self._latest_proposal.plan_sequences,
                danger_veto=danger_veto,
            )
        except (
            ArithmeticError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return self._fallback(
                symbolic_name,
                symbolic_data,
                reason="counterfactual_execution_error",
            )
        self._decisions += 1
        chosen = decision.chosen
        reason = decision.reason
        applied = False
        output_name = symbolic_name
        output_data = symbolic_data
        context = state.signature
        if chosen is None:
            reason = decision.reason
        elif protected_route:
            reason = "protected_route_veto"
            self._vetoes += 1
        elif chosen.veto:
            reason = chosen.veto
            self._vetoes += 1
        elif mode is SageTMode.SHADOW:
            reason = "shadow"
            self._shadow_decisions += 1
        elif mode is SageTMode.BOUNDED:
            reason, applied = self._bounded_authority(
                chosen,
                context=context,
                symbolic_name=symbolic_name,
                symbolic_data=symbolic_data,
            )
        elif mode is SageTMode.ACTIVE:
            applied = not _same_action(
                chosen.first_action,
                symbolic_name,
                symbolic_data,
            )
            reason = "active_override" if applied else "active_agreement"
        if applied and chosen is not None:
            output_name = chosen.first_action.action_name
            output_data = dict(chosen.first_action.action_data)
            self._interventions += 1
        arbitration = SageTArbitration(
            action_name=output_name,
            action_data=output_data,
            applied=applied,
            requested_mode=self.config.requested_mode.value,
            effective_mode=mode.value,
            reason=reason,
            decision=decision,
            posterior=before,
        )
        self._record_decision(
            arbitration,
            symbolic_name=symbolic_name,
            symbolic_data=symbolic_data,
        )
        return arbitration

    def observe_transition(self, record: TransitionRecord) -> None:
        if self.effective_mode is SageTMode.OFF:
            return
        try:
            evidence = compile_transition_record(
                record,
                regime_index=self._regime_index,
            )
        except (
            ArithmeticError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            self._record(
                {
                    "kind": "observation_rejected",
                    "reason": "uncompilable_transition",
                }
            )
            return
        before = self.posterior.snapshot(maximum_programs=None)
        self._transitions.append(evidence)
        self._observed_contexts.add(evidence.state_before.signature)
        self.posterior.observe(evidence)
        self._needs_reassembly = True
        self._record(
            {
                "kind": "observation",
                "action": evidence.action.key,
                "events": evidence.events,
                "posterior_before": before,
                "posterior_after": self.posterior.snapshot(maximum_programs=None),
                "surprise": (
                    None
                    if not self.posterior.particles
                    else -max(
                        particle.latest_log_likelihood
                        for particle in self.posterior.particles
                    )
                ),
            }
        )

    def start_branch(self, *, regime_index: int | None = None) -> None:
        if self.effective_mode is SageTMode.OFF:
            return
        self._branch_index += 1
        if regime_index is not None:
            self._regime_index = int(regime_index)
        self._interventions_this_reset = 0
        self._intervened_contexts.clear()
        self.posterior.start_branch(regime_index=self._regime_index)
        self._record(
            {
                "kind": "branch",
                "branch_index": self._branch_index,
                "regime_index": self._regime_index,
            }
        )

    def note_level_change(self) -> None:
        if self.effective_mode is SageTMode.OFF:
            return
        self._regime_index += 1
        self.start_branch(regime_index=self._regime_index)

    def summary(self) -> Mapping[str, Any]:
        return {
            "requested_mode": self.config.requested_mode.value,
            "effective_mode": self.effective_mode.value,
            "counterfactual_gate_passed": self.config.counterfactual_gate_passed,
            "active_gate_passed": self.config.active_gate_passed,
            "decisions": self._decisions,
            "shadow_decisions": self._shadow_decisions,
            "interventions": self._interventions,
            "vetoes": self._vetoes,
            "interventions_this_reset": self._interventions_this_reset,
            "transitions": len(self._transitions),
            "posterior": self.posterior.snapshot(),
            "executor": self.executor.summary(),
            "consolidation": self.consolidation.snapshot(),
            "trace_errors": self.trace_writer.errors,
            "recent_audit": tuple(self._audit)[-8:],
        }

    def _ensure_programs(
        self,
        *,
        state: Any,
        candidates: Sequence[ActionCandidate],
        mechanic_theory: Any | None,
        goal_hypotheses: Sequence[Any],
        route_memory: Any | None,
    ) -> None:
        available_names = tuple(
            sorted({candidate.action_name for candidate in candidates})
        )
        if (
            not self._needs_reassembly
            and available_names == self._available_action_names
        ):
            return
        proposal = self.proposer.propose(
            available_actions=available_names,
            transitions=tuple(self._transitions),
            mechanic_theory=mechanic_theory,
            goal_hypotheses=goal_hypotheses,
            route_memory=route_memory,
        )
        programs = self.assembler.assemble(
            proposal.fragments,
            available_actions=available_names,
        )
        if self.posterior.particles and available_names == self._available_action_names:
            self.posterior.add_programs(programs, initial_state=state)
        else:
            self.posterior.seed(programs, initial_state=state)
        self._latest_proposal = proposal
        self._available_action_names = available_names
        self._needs_reassembly = False

    def _bounded_authority(
        self,
        chosen: SequenceAssessment,
        *,
        context: str,
        symbolic_name: str,
        symbolic_data: Mapping[str, Any],
    ) -> tuple[str, bool]:
        if _same_action(
            chosen.first_action,
            symbolic_name,
            symbolic_data,
        ):
            return "bounded_agreement", False
        if chosen.terminal_risk > self.config.bounded_maximum_terminal_risk + 1e-9:
            self._vetoes += 1
            return "bounded_risk_veto", False
        if context in self._observed_contexts:
            return "bounded_known_context", False
        if context in self._intervened_contexts:
            return "bounded_context_budget", False
        if (
            self._interventions_this_reset
            >= self.config.bounded_maximum_interventions_per_reset
        ):
            return "bounded_reset_budget", False
        self._intervened_contexts.add(context)
        self._interventions_this_reset += 1
        return "bounded_override", True

    def _fallback(
        self,
        action_name: str,
        action_data: Mapping[str, Any],
        *,
        reason: str,
    ) -> SageTArbitration:
        arbitration = SageTArbitration(
            action_name=action_name,
            action_data=dict(action_data),
            applied=False,
            requested_mode=self.config.requested_mode.value,
            effective_mode=self.effective_mode.value,
            reason=reason,
            posterior=self.posterior.snapshot(),
        )
        self._decisions += 1
        self._record_decision(
            arbitration,
            symbolic_name=action_name,
            symbolic_data=action_data,
        )
        return arbitration

    def _record_decision(
        self,
        arbitration: SageTArbitration,
        *,
        symbolic_name: str,
        symbolic_data: Mapping[str, Any],
    ) -> None:
        decision = arbitration.decision
        assessments = () if decision is None else decision.assessments
        self._record(
            {
                "kind": "decision",
                "requested_mode": arbitration.requested_mode,
                "effective_mode": arbitration.effective_mode,
                "reason": arbitration.reason,
                "applied": arbitration.applied,
                "symbolic_action": {
                    "name": symbolic_name,
                    "data": dict(symbolic_data),
                },
                "action": {
                    "name": arbitration.action_name,
                    "data": dict(arbitration.action_data),
                },
                "posterior_before": arbitration.posterior,
                "posterior_after": self.posterior.snapshot(maximum_programs=None),
                "sequences": [_assessment_record(item) for item in assessments],
            }
        )

    def _record(self, record: Mapping[str, Any]) -> None:
        public = dict(record)
        self._audit.append(public)
        self.trace_writer.write(public)


def _same_action(
    candidate: ActionCandidate,
    action_name: str,
    action_data: Mapping[str, Any],
) -> bool:
    return candidate.action_name == str(action_name).strip().upper() and dict(
        candidate.action_data
    ) == dict(action_data)


def _mode_value(value: str | SageTMode) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _assessment_record(item: SequenceAssessment) -> Mapping[str, Any]:
    return {
        "sequence": tuple(action.key for action in item.candidate.actions),
        "source": item.candidate.source,
        "utility": item.utility,
        "expected_goal": item.expected_goal,
        "expected_progress": item.expected_progress,
        "terminal_risk": item.terminal_risk,
        "information_gain": item.information_gain,
        "beta": item.beta,
        "residual_mass": item.residual_mass,
        "veto": item.veto,
        "disagreement": {
            "observational": item.disagreement.observational,
            "causal": item.disagreement.causal,
            "teleological": item.disagreement.teleological,
            "planning": item.disagreement.planning,
        },
        "program_predictions": [
            {
                "program_hash": prediction.program_hash,
                "probability": prediction.probability,
                "packets": [
                    {
                        "known_channels": sorted(packet.known_channels),
                        "unknown_channels": sorted(packet.unknown_channels),
                        "signature": packet.full_signature,
                    }
                    for packet in prediction.rollout.packets
                ],
            }
            for prediction in item.predictions
        ],
    }


__all__ = [
    "SageTArbitration",
    "SageTConfig",
    "SageTController",
    "SageTMode",
    "SageTTraceWriter",
]
