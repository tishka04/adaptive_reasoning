"""Fail-closed compiler for two-slice SAGE.T causal programs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .contracts import CausalProgram, CausalVariableSpec, MechanismSpec
from .mechanisms import MechanismRegistry


@dataclass(frozen=True)
class CompiledCausalProgram:
    program: CausalProgram
    variables: Mapping[str, CausalVariableSpec]
    mechanisms: Mapping[str, MechanismSpec]
    topological_order: tuple[str, ...]
    descendants: Mapping[str, frozenset[str]]
    available_actions: frozenset[str]
    graph_digest: str

    @property
    def canonical_hash(self) -> str:
        return self.program.canonical_hash


class ProgramCompiler:
    def __init__(self, mechanism_registry: MechanismRegistry) -> None:
        self.mechanism_registry = mechanism_registry

    def compile(
        self,
        program: CausalProgram,
        *,
        action_catalog: Sequence[str],
    ) -> CompiledCausalProgram:
        variables = _unique_by_id(program.variables, "variable_id", "variable")
        mechanisms = _unique_by_id(program.mechanisms, "output_variable", "mechanism output")
        if set(mechanisms) != set(variables):
            missing = sorted(set(variables) - set(mechanisms))
            extra = sorted(set(mechanisms) - set(variables))
            raise ValueError(
                f"every predicted variable needs one mechanism; missing={missing}, extra={extra}"
            )
        catalog = frozenset(str(action) for action in action_catalog)
        declared_actions = {item.action_name for item in program.action_model}
        unavailable = declared_actions - catalog
        if unavailable:
            raise ValueError(f"causal program declares unavailable actions: {sorted(unavailable)}")
        if len(declared_actions) != len(program.action_model):
            raise ValueError("causal program has duplicate action declarations")
        next_edges: dict[str, set[str]] = {variable_id: set() for variable_id in variables}
        for mechanism in program.mechanisms:
            if not self.mechanism_registry.can_resolve(mechanism):
                raise ValueError(f"unresolved mechanism: {mechanism.mechanism_id}")
            required_action = mechanism.parameters.get("action_name")
            if required_action is not None and str(required_action) not in declared_actions:
                raise ValueError(
                    f"mechanism {mechanism.mechanism_id} uses undeclared action {required_action}"
                )
            for parent in mechanism.parent_variables:
                if parent.variable_id not in variables:
                    raise ValueError(
                        f"mechanism {mechanism.mechanism_id} has missing parent {parent.variable_id}"
                    )
                if parent.time_slice == "next":
                    next_edges[parent.variable_id].add(mechanism.output_variable)
            self._validate_domain(mechanism, variables[mechanism.output_variable])
            self._validate_types(mechanism, variables)
        order = _topological_order(next_edges)
        descendants = {
            variable_id: frozenset(_descendants(variable_id, next_edges))
            for variable_id in variables
        }
        graph_payload = {
            "program": program.canonical_hash,
            "next_edges": {key: sorted(value) for key, value in sorted(next_edges.items())},
            "order": order,
        }
        graph_digest = hashlib.sha256(
            json.dumps(graph_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return CompiledCausalProgram(
            program=program,
            variables=MappingProxyType(variables),
            mechanisms=MappingProxyType(mechanisms),
            topological_order=order,
            descendants=MappingProxyType(descendants),
            available_actions=frozenset(declared_actions),
            graph_digest=graph_digest,
        )

    @staticmethod
    def _validate_domain(
        mechanism: MechanismSpec,
        output: CausalVariableSpec,
    ) -> None:
        if not output.domain:
            return
        if mechanism.operator_type.lower() == "set":
            value = mechanism.parameters.get("value")
            if value not in output.domain:
                raise ValueError(
                    f"mechanism {mechanism.mechanism_id} emits value outside {output.variable_id} domain"
                )
        if mechanism.operator_type.lower() == "cycle_attribute":
            values = tuple(mechanism.parameters.get("values", ()))
            if not values or any(value not in output.domain for value in values):
                raise ValueError(
                    f"mechanism {mechanism.mechanism_id} has an incompatible cycle domain"
                )

    @staticmethod
    def _validate_types(
        mechanism: MechanismSpec,
        variables: Mapping[str, CausalVariableSpec],
    ) -> None:
        output = variables[mechanism.output_variable]
        parents = [variables[parent.variable_id] for parent in mechanism.parent_variables]
        operator = mechanism.operator_type.lower()
        if (
            operator in {"identity", "copy_attribute", "cycle_attribute", "swap", "move"}
            and parents
            and parents[0].variable_type != output.variable_type
        ):
            raise ValueError(
                f"mechanism {mechanism.mechanism_id} has incompatible parent/output types"
            )
        if operator == "align_relation":
            if len(parents) != 2 or parents[0].variable_type != parents[1].variable_type:
                raise ValueError(
                    f"mechanism {mechanism.mechanism_id} needs two type-compatible parents"
                )
            if output.variable_type not in {"relation", "boolean", "terminal"}:
                raise ValueError(
                    f"mechanism {mechanism.mechanism_id} needs a boolean-like output"
                )
        if (
            operator
            in {"all_predicate", "any_predicate", "count_threshold", "terminal_predicate"}
            and output.variable_type not in {"relation", "boolean", "terminal"}
        ):
            raise ValueError(
                f"mechanism {mechanism.mechanism_id} needs a boolean-like output"
            )


def _unique_by_id(items: Sequence[object], attribute: str, label: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in items:
        key = str(getattr(item, attribute))
        if key in result:
            raise ValueError(f"duplicate {label}: {key}")
        result[key] = item
    return result


def _topological_order(edges: Mapping[str, set[str]]) -> tuple[str, ...]:
    indegree = {node: 0 for node in edges}
    for children in edges.values():
        for child in children:
            indegree[child] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    result = []
    while ready:
        node = ready.pop(0)
        result.append(node)
        for child in sorted(edges[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(result) != len(edges):
        cyclic = sorted(node for node, degree in indegree.items() if degree > 0)
        raise ValueError(f"forbidden contemporaneous causal cycle: {cyclic}")
    return tuple(result)


def _descendants(root: str, edges: Mapping[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    frontier = list(edges[root])
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(edges[node])
    return seen


__all__ = ["CompiledCausalProgram", "ProgramCompiler"]
