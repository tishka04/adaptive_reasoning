"""Validation and grounding boundary between proposals and executable options."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .hypotheses import SemanticHypothesis, predicate_key
from .scene_graph import SceneGraph


@dataclass(frozen=True)
class CompiledSemanticOption:
    option_id: str
    hypothesis_id: str
    action_name: str
    action_data: Mapping[str, Any]
    bindings: Mapping[str, str]
    preconditions: Tuple[str, ...]
    asserted_effects: Tuple[str, ...]
    retracted_effects: Tuple[str, ...]
    confidence: float
    source: str

    @property
    def action_key(self) -> str:
        return _action_key(self.action_name, self.action_data)


@dataclass(frozen=True)
class CompilationResult:
    options: Tuple[CompiledSemanticOption, ...]
    rejected: Tuple[str, ...]


class HypothesisCompiler:
    """Ground typed roles and reject unbound, illegal, or impossible proposals."""

    def __init__(self, *, maximum_bindings_per_hypothesis: int = 16) -> None:
        self.maximum_bindings_per_hypothesis = max(
            1, int(maximum_bindings_per_hypothesis)
        )

    def compile(
        self,
        hypotheses: Sequence[SemanticHypothesis],
        *,
        graph: SceneGraph,
        legal_candidates: Sequence[Any],
    ) -> CompilationResult:
        legal_specs = tuple(_candidate_spec(item) for item in legal_candidates)
        options = []
        rejected = []
        for hypothesis in hypotheses:
            matching_actions = tuple(
                (name, action_data)
                for name, action_data in legal_specs
                if name == hypothesis.action_name
                and (
                    not hypothesis.action_data
                    or _action_key(name, action_data)
                    == _action_key(
                        hypothesis.action_name,
                        hypothesis.action_data,
                    )
                )
            )
            if not matching_actions:
                rejected.append(
                    f"{hypothesis.hypothesis_id}:illegal_action:"
                    + _action_key(
                        hypothesis.action_name,
                        hypothesis.action_data,
                    )
                )
                continue
            roles = sorted(
                {
                    ref.role
                    for predicate in hypothesis.preconditions
                    for ref in predicate.roles
                }
                | {
                    ref.role
                    for effect in hypothesis.effects
                    for ref in effect.predicate.roles
                }
            )
            candidates_by_role = {
                role: graph.entities_for_role(role) for role in roles
            }
            missing = [
                role
                for role, entities in candidates_by_role.items()
                if not entities
            ]
            if missing:
                rejected.append(
                    f"{hypothesis.hypothesis_id}:unbound_roles:"
                    + ",".join(missing)
                )
                continue
            compiled_for_hypothesis = 0
            for action_name, action_data in matching_actions:
                action_key = _action_key(action_name, action_data)
                combinations = (
                    itertools.product(
                        *(candidates_by_role[role] for role in roles)
                    )
                    if roles
                    else ((),)
                )
                for entities in combinations:
                    if (
                        compiled_for_hypothesis
                        >= self.maximum_bindings_per_hypothesis
                    ):
                        break
                    bindings = {
                        role: entity.entity_id
                        for role, entity in zip(roles, entities)
                    }
                    if len(bindings.values()) != len(set(bindings.values())):
                        continue
                    preconditions = tuple(
                        predicate_key(predicate, bindings)
                        for predicate in hypothesis.preconditions
                    )
                    if not set(preconditions).issubset(
                        graph.state_predicates
                    ):
                        continue
                    asserted = tuple(
                        predicate_key(effect.predicate, bindings)
                        for effect in hypothesis.effects
                        if effect.operation == "assert"
                    )
                    retracted = tuple(
                        predicate_key(effect.predicate, bindings)
                        for effect in hypothesis.effects
                        if effect.operation == "retract"
                    )
                    canonical = json.dumps(
                        {
                            "hypothesis": hypothesis.hypothesis_id,
                            "action": action_key,
                            "bindings": bindings,
                            "asserted": asserted,
                            "retracted": retracted,
                        },
                        sort_keys=True,
                    )
                    option_id = "s12_" + hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest()[:12]
                    options.append(
                        CompiledSemanticOption(
                            option_id=option_id,
                            hypothesis_id=hypothesis.hypothesis_id,
                            action_name=action_name,
                            action_data=dict(action_data),
                            bindings=bindings,
                            preconditions=preconditions,
                            asserted_effects=asserted,
                            retracted_effects=retracted,
                            confidence=hypothesis.confidence,
                            source=hypothesis.source,
                        )
                    )
                    compiled_for_hypothesis += 1
                if (
                    compiled_for_hypothesis
                    >= self.maximum_bindings_per_hypothesis
                ):
                    break
            if compiled_for_hypothesis == 0:
                rejected.append(
                    f"{hypothesis.hypothesis_id}:preconditions_not_grounded"
                )
        return CompilationResult(
            options=tuple(options),
            rejected=tuple(rejected),
        )


def _action_key(action_name: str, action_data: Mapping[str, Any]) -> str:
    normalized = {
        str(key): value
        for key, value in dict(action_data or {}).items()
        if value is not None
    }
    return (
        str(action_name).strip().upper()
        + ":"
        + json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _candidate_spec(candidate: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(candidate, str):
        return str(candidate).strip().upper(), {}
    return (
        str(getattr(candidate, "action_name", "")).strip().upper(),
        dict(getattr(candidate, "action_data", {}) or {}),
    )


__all__ = [
    "CompilationResult",
    "CompiledSemanticOption",
    "HypothesisCompiler",
]
