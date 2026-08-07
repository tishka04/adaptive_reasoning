"""Coordinate-free ordinal role induction for T8.6i."""

from __future__ import annotations

from dataclasses import replace

from .contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    JointProgramHypothesis,
    PredictionPacket,
)
from .executor import ProgramExecutor

WESTMOST_TARGET = "westmost_target"
EASTMOST_TARGET = "eastmost_target"
STRUCTURAL_TARGET_ROLES = (WESTMOST_TARGET, EASTMOST_TARGET)


def augment_structural_roles(state: AbstractState) -> AbstractState:
    """Assign unique horizontal extrema using relative order only."""

    targets = [
        entity
        for entity in state.entities
        if entity.has_role("target") and entity.center is not None
    ]
    if len(targets) < 2:
        return state
    columns = [float(entity.center[1]) for entity in targets if entity.center]
    minimum = min(columns)
    maximum = max(columns)
    if minimum == maximum:
        return state
    west = [entity for entity in targets if float(entity.center[1]) == minimum]
    east = [entity for entity in targets if float(entity.center[1]) == maximum]
    assignments: dict[str, str] = {}
    if len(west) == 1:
        assignments[west[0].entity_id] = WESTMOST_TARGET
    if len(east) == 1:
        assignments[east[0].entity_id] = EASTMOST_TARGET
    if not assignments:
        return state
    entities = tuple(
        replace(
            entity,
            roles=tuple(
                sorted(set(entity.roles) | {assignments[entity.entity_id]})
            ),
        )
        if entity.entity_id in assignments
        else entity
        for entity in state.entities
    )
    return AbstractState(
        entities=entities,
        true_facts=state.true_facts,
        false_facts=state.false_facts,
        counters=state.counters,
        registers=state.registers,
        topology=state.topology,
        regime_index=state.regime_index,
    )


def action_target_structural_role(
    state: AbstractState,
    action: ActionCandidate,
) -> str:
    """Return the induced extremal role of an action's grounded target."""

    enriched = augment_structural_roles(state)
    payload = dict(action.action_data)
    explicit = str(
        payload.get(
            "entity_id",
            payload.get("target_id", payload.get("object_id", "")),
        )
    )
    entity: AbstractEntity | None = next(
        (item for item in enriched.entities if item.entity_id == explicit),
        None,
    )
    if entity is None:
        x = payload.get("x", payload.get("col"))
        y = payload.get("y", payload.get("row"))
        try:
            column = float(x)
            row = float(y)
        except (TypeError, ValueError):
            return ""
        targets = [
            item
            for item in enriched.entities
            if item.has_role("target") and item.center is not None
        ]
        if not targets:
            return ""
        entity = min(
            targets,
            key=lambda item: (
                (float(item.center[0]) - row) ** 2
                + (float(item.center[1]) - column) ** 2
            ),
        )
    return next(
        (role for role in STRUCTURAL_TARGET_ROLES if entity.has_role(role)),
        "",
    )


class StructuralRoleProgramExecutor(ProgramExecutor):
    """Canonical executor adapter enriching states with ordinal roles."""

    def step(
        self,
        program: JointProgramHypothesis,
        state: AbstractState,
        action: ActionCandidate,
    ) -> PredictionPacket:
        return super().step(program, augment_structural_roles(state), action)


__all__ = [
    "EASTMOST_TARGET",
    "STRUCTURAL_TARGET_ROLES",
    "WESTMOST_TARGET",
    "StructuralRoleProgramExecutor",
    "action_target_structural_role",
    "augment_structural_roles",
]
