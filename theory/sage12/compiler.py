"""Validation and grounding boundary between proposals and executable options."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from .hypotheses import SemanticHypothesis, predicate_key
from .scene_graph import SceneGraph

SLOT_EFFECTS = (
    "changed",
    "moved",
    "target_created",
    "target_removed",
    "target_moved",
    "level_complete",
    "game_over",
)


@dataclass(frozen=True)
class SemanticActionSlot:
    """Candidate-complete, identity-free semantic view of one legal action."""

    slot_id: str
    action_name: str
    action_data: Mapping[str, Any]
    semantic_signature: Mapping[str, Any]

    @property
    def action_key(self) -> str:
        return _action_key(self.action_name, self.action_data)

    @property
    def semantic_key(self) -> str:
        return json.dumps(
            dict(self.semantic_signature),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


@dataclass(frozen=True)
class SlotAnnotation:
    """Frozen-model effect scores for one slot; never evidence support."""

    slot_id: str
    effect_probabilities: Mapping[str, float]
    source: str = "qwen_constrained_bits"
    support: int = 0

    def __post_init__(self) -> None:
        if self.support != 0:
            raise ValueError("slot annotations must have support=0")
        missing = set(SLOT_EFFECTS) - set(self.effect_probabilities)
        extra = set(self.effect_probabilities) - set(SLOT_EFFECTS)
        if missing or extra:
            raise ValueError(
                "slot annotations require exactly the frozen effect vocabulary"
            )
        if any(
            not 0.0 <= float(value) <= 1.0
            for value in self.effect_probabilities.values()
        ):
            raise ValueError("slot effect probabilities must be in [0, 1]")


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
    semantic_key: str = ""
    effect_probabilities: Mapping[str, float] = field(default_factory=dict)

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

    def compile_slots(
        self,
        slots: Sequence[SemanticActionSlot],
        *,
        annotations: Sequence[SlotAnnotation] = (),
    ) -> CompilationResult:
        """Compile every legal slot, including all-zero model annotations.

        Unlike free-form hypotheses, slots are created from the legal action
        set itself.  An annotation may change scores but can never remove a
        candidate, so candidate coverage is exactly one by construction.
        """
        by_slot = {annotation.slot_id: annotation for annotation in annotations}
        if len(by_slot) != len(annotations):
            raise ValueError("duplicate slot annotation")
        unknown = set(by_slot) - {slot.slot_id for slot in slots}
        if unknown:
            raise ValueError("annotation refers to an unknown semantic slot")
        options = []
        seen_ids: set[str] = set()
        for slot in slots:
            if slot.slot_id in seen_ids:
                raise ValueError("duplicate semantic slot id")
            seen_ids.add(slot.slot_id)
            annotation = by_slot.get(
                slot.slot_id,
                SlotAnnotation(
                    slot_id=slot.slot_id,
                    effect_probabilities={
                        effect: 0.0 for effect in SLOT_EFFECTS
                    },
                    source="unannotated_slot",
                ),
            )
            probabilities = {
                effect: float(annotation.effect_probabilities[effect])
                for effect in SLOT_EFFECTS
            }
            asserted = tuple(
                f"{effect}|-|-|"
                for effect in SLOT_EFFECTS
                if probabilities[effect] >= 0.5
            )
            canonical = json.dumps(
                {
                    "slot_id": slot.slot_id,
                    "action": slot.action_key,
                    "semantic_key": slot.semantic_key,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            options.append(
                CompiledSemanticOption(
                    option_id="s12slot_"
                    + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
                    hypothesis_id=f"slot_annotation:{slot.slot_id}",
                    action_name=slot.action_name,
                    action_data=dict(slot.action_data),
                    bindings={},
                    preconditions=(),
                    asserted_effects=asserted,
                    retracted_effects=(),
                    confidence=(
                        sum(probabilities.values()) / len(probabilities)
                    ),
                    source=annotation.source,
                    semantic_key=slot.semantic_key,
                    effect_probabilities=probabilities,
                )
            )
        return CompilationResult(options=tuple(options), rejected=())


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
    "SLOT_EFFECTS",
    "SemanticActionSlot",
    "SlotAnnotation",
]
