"""Proposal-only adapters from legacy SAGE components into SAGE.T11."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from theory.sage_t.contracts import AbstractState, ObservedTransition

from .contracts import (
    ActionInterventionSpec,
    ActionProgram,
    BindingSpec,
    CausalProgram,
    CausalState,
    CausalVariableSpec,
    GoalSpec,
    GroundedAction,
    MechanismSpec,
    ObservationModelSpec,
    ParentRef,
    StructuredDelta,
    TransitionEvidence,
    ValueDistribution,
    causal_program_from_dict,
)
from .mechanisms import MechanismRegistry, NeuralMechanism, ObservationLikelihoodModel
from .posterior import CausalPosterior


@dataclass(frozen=True)
class CausalProgramProposal:
    source: str
    program: CausalProgram | None = None
    family: str = ""
    payload: Mapping[str, Any] | None = None
    support: int = 0

    def __post_init__(self) -> None:
        if self.support != 0:
            raise ValueError("proposal adapters never manufacture evidence support")


@dataclass(frozen=True)
class Sage9pRelationProposal:
    proposal_id: str
    action_name: str
    source_variable: str
    target_variable: str
    relation_variable: str
    completion_variable: str = "level.complete"
    failure_variable: str = "level.failed"
    source_role: str = "source"
    target_role: str = "target"
    source_entity: str = ""
    target_entity: str = ""


class Sage9pProgramAdapter:
    """Compile a confirmed relational stencil as a complete rival program."""

    def propose(self, relation: Sage9pRelationProposal) -> CausalProgramProposal:
        variables = (
            CausalVariableSpec(relation.source_variable, "attribute"),
            CausalVariableSpec(relation.target_variable, "attribute"),
            CausalVariableSpec(relation.relation_variable, "relation", (False, True)),
            CausalVariableSpec(relation.completion_variable, "terminal", (False, True)),
            CausalVariableSpec(relation.failure_variable, "terminal", (False, True)),
        )
        mechanisms = (
            MechanismSpec(
                f"{relation.proposal_id}_source_persistence",
                relation.source_variable,
                (ParentRef(relation.source_variable),),
                "identity",
            ),
            MechanismSpec(
                f"{relation.proposal_id}_target_persistence",
                relation.target_variable,
                (ParentRef(relation.target_variable),),
                "identity",
            ),
            MechanismSpec(
                f"{relation.proposal_id}_alignment",
                relation.relation_variable,
                (
                    ParentRef(relation.source_variable, "next"),
                    ParentRef(relation.target_variable, "next"),
                ),
                "align_relation",
            ),
            MechanismSpec(
                f"{relation.proposal_id}_completion",
                relation.completion_variable,
                (ParentRef(relation.relation_variable, "next"),),
                "all_predicate",
            ),
            MechanismSpec(
                f"{relation.proposal_id}_failure",
                relation.failure_variable,
                (ParentRef(relation.failure_variable),),
                "identity",
            ),
        )
        program = CausalProgram(
            program_id=relation.proposal_id,
            bindings=BindingSpec(
                {
                    relation.source_role: relation.source_entity,
                    relation.target_role: relation.target_entity,
                }
            ),
            variables=variables,
            mechanisms=mechanisms,
            action_model=(ActionInterventionSpec(relation.action_name),),
            goal=GoalSpec(
                f"{relation.completion_variable} == true",
                (f"{relation.relation_variable} == true",),
                f"{relation.failure_variable} == true",
            ),
            description_length=float(len(variables) + len(mechanisms)),
            provenance=("proposal:sage9p",),
        )
        return CausalProgramProposal(source="sage9p", program=program)

    def propose_from_learner(self, learner: Any) -> tuple[CausalProgramProposal, ...]:
        rules = dict(learner.selection_rules())
        return tuple(
            CausalProgramProposal(
                source="sage9p",
                family="relational_stencil",
                payload={
                    "marker": str(marker),
                    "desired_relation": (
                        "equal_to_center" if desired_equal else "different_from_center"
                    ),
                    "requires_binding": True,
                },
                support=0,
            )
            for marker, desired_equal in sorted(rules.items())
        )


class M2ProgramAdapter:
    def propose(self, payload: Mapping[str, Any]) -> CausalProgramProposal:
        return CausalProgramProposal(
            source="m2_or_llm",
            program=causal_program_from_dict(payload),
        )


class CausalFamilyProposalAdapter:
    """Demote T10.3.12f family beliefs to non-authoritative proposals."""

    def propose(self, family: str, payload: Mapping[str, Any]) -> CausalProgramProposal:
        return CausalProgramProposal(
            source="t10_3_12f_family",
            family=str(family),
            payload=dict(payload),
            support=0,
        )


class ArcLeWMProposalAdapter:
    def register_module(
        self,
        registry: MechanismRegistry,
        *,
        module_id: str,
        module: NeuralMechanism,
    ) -> None:
        registry.register_neural(module_id, module)

    def register_observation_model(
        self,
        registry: MechanismRegistry,
        *,
        module_id: str,
        model: ObservationLikelihoodModel,
    ) -> None:
        registry.register_observation_likelihood(module_id, model)

    def observation_model(self, *, module_id: str) -> ObservationModelSpec:
        return ObservationModelSpec(
            channels=("variables", "objects", "relations", "patch", "terminal", "goal", "progress"),
            neural_module_id=module_id,
        )


class RouteReplayProposalAdapter:
    def action_program(
        self,
        actions: Sequence[Any],
        *,
        exact: bool = True,
        progressive: bool = False,
    ) -> ActionProgram:
        if exact and not progressive:
            source = "exact_route"
        elif exact:
            source = "progressive_route"
        else:
            source = "frontier"
        return ActionProgram(
            tuple(grounded_action_from_legacy(action) for action in actions),
            source=source,
        )


class CausalProposalCoordinator:
    """The only admission path from proposal sources into the posterior."""

    def propose_into(
        self,
        *,
        posterior: CausalPosterior,
        proposals: Sequence[CausalProgramProposal],
        action_catalog: Sequence[str],
    ) -> int:
        programs = []
        for proposal in proposals:
            if proposal.support != 0:
                raise ValueError(
                    "proposal support must remain zero before observation"
                )
            if proposal.program is None:
                continue
            posterior.executor.compile(
                proposal.program,
                action_catalog=tuple(str(action) for action in action_catalog),
            )
            programs.append(proposal.program)
        return posterior.add_programs(programs)


def causal_state_from_abstract(
    state: AbstractState,
    *,
    observation_hash: str = "",
    confidence: float = 1.0,
) -> CausalState:
    variables: dict[str, ValueDistribution] = {}
    for fact in state.true_facts:
        variables[_fact_variable(fact)] = ValueDistribution.deterministic(True)
    for fact in state.false_facts:
        variables[_fact_variable(fact)] = ValueDistribution.deterministic(False)
    for key, value in state.counters:
        variables[f"counter.{_safe_token(key)}"] = ValueDistribution.deterministic(value)
    for key, value in state.registers:
        variables[f"register.{_safe_token(key)}"] = ValueDistribution.deterministic(value)
    for key, value in state.topology:
        variables[f"topology.{_safe_token(key)}"] = ValueDistribution.deterministic(value)
    variables["regime.index"] = ValueDistribution.deterministic(state.regime_index)
    entities = tuple(str(entity.entity_id) for entity in state.entities)
    relations = tuple(
        sorted(
            _fact_variable(fact)
            for fact in state.true_facts
            if len(_fact_arguments(fact)) >= 2
        )
    )
    return CausalState(
        variables=variables,
        entities=entities,
        relations=relations,
        observation_hash=observation_hash,
        confidence=confidence,
    )


def transition_evidence_from_observed(
    observed: ObservedTransition,
    *,
    game_id: str = "",
    prefix_hash: str = "",
) -> TransitionEvidence:
    before = causal_state_from_abstract(
        observed.state_before,
        observation_hash=prefix_hash,
    )
    after = causal_state_from_abstract(observed.state_after)
    changes = {
        key: value
        for key, value in after.variables.items()
        if value.total_variation(before.value(key)) > 1e-12
    }
    event_payload = json.dumps(
        {
            "before": before.abstract_signature,
            "action": observed.action.key,
            "after": after.abstract_signature,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_id = "e_" + hashlib.sha256(event_payload.encode("utf-8")).hexdigest()[:24]
    terminal_probability = observed.observation.terminal_probability
    goal_probability = observed.observation.goal_probability
    progress = observed.observation.progress_mean or 0.0
    return TransitionEvidence(
        evidence_id=evidence_id,
        state_before=before,
        action=grounded_action_from_legacy(observed.action),
        state_after=after,
        observed_delta=StructuredDelta(
            variable_changes=changes,
            affected_objects=tuple(observed.observation.object_deltas),
            relation_changes=tuple(observed.observation.relation_deltas),
            progress=min(1.0, max(0.0, float(progress))),
        ),
        terminal=bool(terminal_probability is not None and terminal_probability >= 0.5),
        success=(None if goal_probability is None else goal_probability >= 0.5),
        level_change=int(bool(goal_probability is not None and goal_probability >= 0.5)),
        prefix_hash=prefix_hash,
        game_id=game_id,
    )


def grounded_action_from_legacy(action: Any) -> GroundedAction:
    name = str(getattr(action, "action_name", getattr(action, "name", action)))
    data = getattr(action, "action_data", getattr(action, "action_args", {})) or {}
    return GroundedAction(name, dict(data))


def _fact_variable(fact: Any) -> str:
    arguments = ".".join(_safe_token(item) for item in _fact_arguments(fact))
    suffix = f".{arguments}" if arguments else ""
    return f"fact.{_safe_token(fact.predicate)}{suffix}"


def _fact_arguments(fact: Any) -> tuple[Any, ...]:
    return tuple(getattr(fact, "arguments", getattr(fact, "terms", ())) or ())


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    if not token:
        token = "unknown"
    if not token[0].isalpha():
        token = "v_" + token
    return token[:80]


__all__ = [
    "ArcLeWMProposalAdapter", "CausalFamilyProposalAdapter", "CausalProgramProposal",
    "CausalProposalCoordinator",
    "M2ProgramAdapter", "RouteReplayProposalAdapter", "Sage9pProgramAdapter",
    "Sage9pRelationProposal", "causal_state_from_abstract",
    "grounded_action_from_legacy", "transition_evidence_from_observed",
]
