"""Reset-local causal identification and control for SAGE.T10.3.12f.

The transferable object in this module is a *procedure*, never a source-game
action, target, coordinate, colour, entity, frame, or trajectory.  Grounded
candidate identities are used transiently while a reset is live and are
deliberately absent from every serialisable payload.

The controller implements a small falsifiable loop::

    IDENTIFY -> VERIFY -> CONTROL -> REVISE | ABSTAIN

Only an observed level delta is credited as success.  Local effects merely
update the reset-local posterior over four control families.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .contracts import (
    AbstractState,
    ActionCandidate,
    ObservedTransition,
    PredictionPacket,
    normalized_action_candidates,
)
from .relational_program_v10_3_12 import assert_transfer_safe

FORMAT_VERSION = "sage-t10.3.12f-causal-procedure-v1"
PRIOR_FORMAT_VERSION = "sage-t10.3.12f-causal-prior-v1"
COMPILATION_FORMAT_VERSION = "sage-t10.3.12f-source-compilation-v1"

MODEL_FAMILIES = (
    "stable_repeat",
    "relational_successor",
    "state_conditioned_switch",
    "null_or_unsafe",
)
ARMS = (
    "source_closed_loop",
    "uniform_closed_loop",
    "permuted_source_closed_loop",
    "source_open_loop",
)
PHASE_ORDER = ("IDENTIFY", "VERIFY", "CONTROL", "REVISE", "ABSTAIN")

ADMISSIBLE_RELATIONS = frozenset({"contact", "inside", "encloses", "reachable"})
MINIMUM_CORRESPONDENCE_CONFIDENCE = 0.60

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CONCRETE_ACTION = re.compile(r"\bACTION[0-9]+\b", re.IGNORECASE)
_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "game",
        "game_id",
        "source_game",
        "source_game_id",
        "action",
        "action_name",
        "action_data",
        "action_args",
        "candidate_key",
        "x",
        "y",
        "row",
        "column",
        "coordinate",
        "coordinates",
        "colour",
        "color",
        "palette",
        "entity",
        "entity_id",
        "object_id",
        "frame",
        "frame_hash",
        "raw_frame",
        "raw_grid",
        "grid",
        "trajectory",
        "source_trajectory",
        "grounded_path",
        "reset_id",
        "root_id",
    }
)


def canonical_json(value: Any) -> str:
    """Canonical JSON used by all immutable procedure artefacts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def verify_signed(payload: Mapping[str, Any], checksum_field: str) -> None:
    expected = str(payload.get(checksum_field, ""))
    core = {key: value for key, value in payload.items() if key != checksum_field}
    if not expected or sha256_payload(core) != expected:
        raise ValueError(f"invalid {checksum_field}")


def _identifier(value: Any, *, label: str) -> str:
    result = str(value).strip().lower()
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError(f"{label} must be a bounded snake_case identifier")
    return result


def assert_causal_transfer_safe(value: Any, *, path: str = "payload") -> None:
    """Reject grounded information beyond the older T10.3.12 firewall.

    Abstract words such as ``action_family`` and ``source_closed_loop`` are
    allowed.  Concrete action tokens and keys capable of carrying grounded
    observations are not.
    """

    assert_transfer_safe(value, path=path)
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in _FORBIDDEN_EXACT_KEYS:
                raise ValueError(f"forbidden causal transfer field at {path}.{key}")
            assert_causal_transfer_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            assert_causal_transfer_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _CONCRETE_ACTION.search(value):
        raise ValueError(f"concrete action leaked at {path}")


def _normalise_distribution(weights: Mapping[str, float]) -> dict[str, float]:
    unknown = set(weights) - set(MODEL_FAMILIES)
    missing = set(MODEL_FAMILIES) - set(weights)
    if unknown or missing:
        raise ValueError(
            f"prior families drifted; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    values = {family: float(weights[family]) for family in MODEL_FAMILIES}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("all causal prior weights must be finite and positive")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("causal prior has zero mass")
    return {family: values[family] / total for family in MODEL_FAMILIES}


class ProcedurePhase(str, Enum):
    IDENTIFY = "IDENTIFY"
    VERIFY = "VERIFY"
    CONTROL = "CONTROL"
    REVISE = "REVISE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class CausalProcedureSpec:
    """Frozen, transfer-safe declaration of the causal procedure."""

    prior: Mapping[str, float] = field(
        default_factory=lambda: {family: 0.25 for family in MODEL_FAMILIES}
    )
    phase_order: tuple[str, ...] = PHASE_ORDER
    information_weight: float = 1.0
    repeatability_weight: float = 0.75
    composability_weight: float = 0.75
    risk_weight: float = 1.0
    redundancy_weight: float = 0.50
    maximum_hypotheses: int = 8
    maximum_candidates: int = 16
    posterior_threshold: float = 0.80
    posterior_margin: float = 0.20
    verification_contexts: int = 2
    predictive_demotion_threshold: float = 0.10
    stagnation_limit: int = 4
    maximum_revisions: int = 2
    option_horizon: int = 16
    closed_loop_revision: bool = True

    def __post_init__(self) -> None:
        prior = _normalise_distribution(self.prior)
        object.__setattr__(self, "prior", MappingProxyType(prior))
        if tuple(self.phase_order) != PHASE_ORDER:
            raise ValueError("causal phase order drifted")
        if not 1 <= int(self.maximum_hypotheses) <= 8:
            raise ValueError("maximum_hypotheses must be in [1, 8]")
        if not 1 <= int(self.maximum_candidates) <= 16:
            raise ValueError("maximum_candidates must be in [1, 16]")
        if not 0.5 <= float(self.posterior_threshold) < 1.0:
            raise ValueError("posterior_threshold must be in [0.5, 1)")
        if not 0.0 <= float(self.posterior_margin) < 1.0:
            raise ValueError("posterior_margin must be in [0, 1)")
        if int(self.verification_contexts) < 2:
            raise ValueError("verification requires at least two abstract contexts")
        if not 0.0 < float(self.predictive_demotion_threshold) <= 0.10:
            raise ValueError("predictive demotion threshold must be in (0, 0.10]")
        if int(self.stagnation_limit) != 4:
            raise ValueError("T10.3.12f stagnation limit is frozen at four")
        if int(self.maximum_revisions) != 2:
            raise ValueError("T10.3.12f permits exactly two revisions")
        if int(self.option_horizon) != 16:
            raise ValueError("T10.3.12f option horizon is frozen at sixteen")
        for name in (
            "information_weight",
            "repeatability_weight",
            "composability_weight",
            "risk_weight",
            "redundancy_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        assert_causal_transfer_safe(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "prior": {family: self.prior[family] for family in MODEL_FAMILIES},
            "phase_order": list(self.phase_order),
            "weights": {
                "information": self.information_weight,
                "repeatability": self.repeatability_weight,
                "composability": self.composability_weight,
                "risk": self.risk_weight,
                "redundancy": self.redundancy_weight,
            },
            "limits": {
                "hypotheses": self.maximum_hypotheses,
                "candidates": self.maximum_candidates,
                "verification_contexts": self.verification_contexts,
                "stagnation": self.stagnation_limit,
                "revisions": self.maximum_revisions,
                "option_horizon": self.option_horizon,
            },
            "thresholds": {
                "posterior": self.posterior_threshold,
                "margin": self.posterior_margin,
                "predictive_demotion": self.predictive_demotion_threshold,
            },
            "closed_loop_revision": self.closed_loop_revision,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_payload(self.safe_payload)


@dataclass(frozen=True)
class InterventionSignature:
    """Identity-free description of one locally grounded intervention."""

    action_family: str
    target_role: str
    argument_schema: str

    def __post_init__(self) -> None:
        for name in ("action_family", "target_role", "argument_schema"):
            object.__setattr__(self, name, _identifier(getattr(self, name), label=name))
        assert_causal_transfer_safe(self.safe_payload)

    @classmethod
    def from_candidate(
        cls,
        candidate: ActionCandidate,
        *,
        target_role: str = "unbound_role",
    ) -> "InterventionSignature":
        keys = {str(key).strip().lower() for key in candidate.action_data}
        spatial = bool(keys & {"x", "y", "row", "column"})
        if not keys:
            family = "parameterless_operator"
            schema = "no_arguments"
        elif spatial:
            family = "spatial_operator"
            schema = "point_binding" if len(keys) <= 2 else "point_with_modifiers"
        else:
            family = "parameterized_operator"
            schema = "single_argument" if len(keys) == 1 else "multiple_arguments"
        return cls(family, target_role, schema)

    @property
    def key(self) -> str:
        return f"{self.action_family}|{self.target_role}|{self.argument_schema}"

    @property
    def safe_payload(self) -> dict[str, str]:
        return {
            "action_family": self.action_family,
            "target_role": self.target_role,
            "argument_schema": self.argument_schema,
        }


def _candidate_target_entity(
    state: AbstractState | None,
    candidate: ActionCandidate,
) -> Any | None:
    """Resolve a local target for scoring without serialising its identity."""

    if state is None or not state.entities:
        return None
    payload = dict(candidate.action_data)
    explicit = str(
        payload.get(
            "entity_id",
            payload.get("target_id", payload.get("object_id", "")),
        )
    )
    if explicit:
        return next((entity for entity in state.entities if entity.entity_id == explicit), None)
    x = payload.get("x", payload.get("col", payload.get("column")))
    y = payload.get("y", payload.get("row"))
    if x is None or y is None:
        return state.entities[0] if len(state.entities) == 1 else None
    try:
        point = (float(y), float(x))
    except (TypeError, ValueError, OverflowError):
        return None
    positioned = tuple(entity for entity in state.entities if entity.center is not None)
    if not positioned:
        return None
    return min(
        positioned,
        key=lambda entity: (
            math.dist(point, entity.center or point),
            tuple(entity.roles),
            entity.entity_id,
        ),
    )


def candidate_target_role(
    state: AbstractState | None,
    candidate: ActionCandidate,
) -> str:
    """Return the nearest target's most specific abstract role."""

    entity = _candidate_target_entity(state, candidate)
    if entity is None or not entity.roles:
        return "unbound_role"
    generic = {"object", "entity", "target", "candidate"}
    specific = tuple(role for role in entity.roles if role not in generic)
    return min(specific or entity.roles)


def _relation_connects(
    state: AbstractState | None,
    left_entity_id: str,
    right_entity_id: str,
) -> bool:
    if state is None or not left_entity_id or not right_entity_id:
        return False
    for fact in state.true_facts:
        if fact.predicate not in ADMISSIBLE_RELATIONS or len(fact.terms) < 2:
            continue
        if left_entity_id in fact.terms and right_entity_id in fact.terms:
            return True
    return False


def _scope_sample(
    candidates: Sequence[ActionCandidate],
    *,
    maximum: int,
    scope: int,
) -> tuple[ActionCandidate, ...]:
    """Select canonical positions spread across the whole legal candidate set."""

    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.key))
    # Every work scope observes the exact same bounded candidate set.  Scope
    # changes only its cyclic tie-breaking order; it is neither an environment
    # seed nor a way to expose different evidence to an arm.
    selected = ordered[: min(maximum, len(ordered))]
    shift = scope % len(selected) if selected else 0
    return selected[shift:] + selected[:shift]


def _event_relation(raw: str) -> tuple[str, str] | None:
    value = str(raw).strip().lower()
    for direction in ("added", "removed"):
        prefix = f"relation_{direction}:"
        if value.startswith(prefix):
            relation = value[len(prefix) :].split(":", 1)[0]
            if relation in ADMISSIBLE_RELATIONS:
                return direction, relation
    return None


@dataclass(frozen=True)
class CausalOutcome:
    """Quality-filtered, identity-free outcome of a physical intervention."""

    persistent_moves: int = 0
    persistent_transformations: int = 0
    relations_added: tuple[str, ...] = ()
    relations_removed: tuple[str, ...] = ()
    action_space_changed: bool = False
    noop: bool = False
    game_over: bool = False
    level_delta: int = 0
    quality: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "persistent_moves", max(0, min(8, int(self.persistent_moves))))
        object.__setattr__(
            self,
            "persistent_transformations",
            max(0, min(8, int(self.persistent_transformations))),
        )
        added = tuple(sorted({_identifier(item, label="relation") for item in self.relations_added}))
        removed = tuple(
            sorted({_identifier(item, label="relation") for item in self.relations_removed})
        )
        if (set(added) | set(removed)) - ADMISSIBLE_RELATIONS:
            raise ValueError("outcome contains a non-admissible relation")
        if set(added) & set(removed):
            raise ValueError("a relation cannot be simultaneously added and removed")
        object.__setattr__(self, "relations_added", added)
        object.__setattr__(self, "relations_removed", removed)
        object.__setattr__(self, "level_delta", max(0, int(self.level_delta)))
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("outcome quality must be in [0, 1]")
        object.__setattr__(self, "quality", quality)
        assert_causal_transfer_safe(self.safe_payload)

    @classmethod
    def from_observed_transition(
        cls,
        evidence: ObservedTransition,
        *,
        action_space_changed: bool = False,
        correspondence_quality: float = 1.0,
    ) -> "CausalOutcome":
        quality = float(correspondence_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("correspondence quality must be in [0, 1]")
        events = tuple(str(item).strip().lower() for item in evidence.events)
        relation_added: set[str] = set()
        relation_removed: set[str] = set()
        for event in events:
            parsed = _event_relation(event)
            if parsed is None:
                continue
            direction, relation = parsed
            (relation_added if direction == "added" else relation_removed).add(relation)
        before_entities = {entity.entity_id: entity for entity in evidence.state_before.entities}
        after_entities = {entity.entity_id: entity for entity in evidence.state_after.entities}
        persistent_ids = set(before_entities) & set(after_entities)
        state_moves = 0
        state_transformations = 0
        if quality >= MINIMUM_CORRESPONDENCE_CONFIDENCE:
            state_moves = sum(
                before_entities[entity_id].center is not None
                and after_entities[entity_id].center is not None
                and before_entities[entity_id].center != after_entities[entity_id].center
                for entity_id in persistent_ids
            )
            state_transformations = sum(
                (
                    before_entities[entity_id].roles,
                    before_entities[entity_id].attributes,
                )
                != (
                    after_entities[entity_id].roles,
                    after_entities[entity_id].attributes,
                )
                for entity_id in persistent_ids
            )

            def persistent_relations(state: AbstractState) -> set[Any]:
                return {
                    fact
                    for fact in state.true_facts
                    if fact.predicate in ADMISSIBLE_RELATIONS
                    and fact.terms
                    and all(term in persistent_ids for term in fact.terms)
                }

            before_relations = persistent_relations(evidence.state_before)
            after_relations = persistent_relations(evidence.state_after)
            relation_added.update(fact.predicate for fact in after_relations - before_relations)
            relation_removed.update(fact.predicate for fact in before_relations - after_relations)
        else:
            # Low-confidence correspondence cannot support persistent-object
            # or relation claims.  Terminal/noop channels remain observable.
            relation_added.clear()
            relation_removed.clear()

        conflicts = relation_added & relation_removed
        if conflicts:
            raise ValueError(
                "conflicting relation delta rejected: " + ",".join(sorted(conflicts))
            )

        object_events = {
            str(key).split(":", 1)[0].lower(): float(value)
            for key, value in evidence.observation.object_deltas.items()
            if float(value) >= 0.5
        }
        persistent_moves = max(state_moves, int("moved" in object_events or "moved" in events))
        transformations = max(state_transformations, int(
            any(
                token in object_events or token in events
                for token in ("morphology_changed", "component_count_changed")
            )
        ))
        if quality < MINIMUM_CORRESPONDENCE_CONFIDENCE:
            persistent_moves = 0
            transformations = 0
        # Birth/death observations never contribute to persistent-object or
        # relation evidence.  They remain visible only through ``noop=False``.
        level_before = evidence.state_before.counter("levels_completed", 0.0)
        level_after = evidence.state_after.counter("levels_completed", level_before)
        level_delta = max(0, int(round(level_after - level_before)))
        if level_delta == 0 and "level_complete" in events:
            level_delta = 1
        game_over = "game_over" in events or bool(
            evidence.observation.terminal_probability is not None
            and float(evidence.observation.terminal_probability) >= 0.5
        )
        explicit_noop = "no_effect" in events
        productive = bool(
            persistent_moves
            or transformations
            or relation_added
            or relation_removed
            or action_space_changed
            or level_delta
            or game_over
            or any(token in object_events for token in ("created", "removed"))
        )
        return cls(
            persistent_moves=persistent_moves,
            persistent_transformations=transformations,
            relations_added=tuple(relation_added),
            relations_removed=tuple(relation_removed),
            action_space_changed=bool(action_space_changed),
            noop=bool(explicit_noop or not productive),
            game_over=game_over,
            level_delta=level_delta,
            quality=quality,
        )

    @property
    def productive(self) -> bool:
        return bool(
            not self.noop
            and not self.game_over
            and (
                self.persistent_moves
                or self.persistent_transformations
                or self.relations_added
                or self.relations_removed
                or self.action_space_changed
                or self.level_delta
            )
        )

    @property
    def mode(self) -> str:
        if self.level_delta:
            return "level_progress"
        if self.game_over:
            return "unsafe_terminal"
        if self.noop:
            return "noop"
        if self.action_space_changed:
            return "action_space_change"
        if self.relations_added or self.relations_removed:
            return "relation_change"
        if self.persistent_moves:
            return "persistent_motion"
        if self.persistent_transformations:
            return "persistent_transformation"
        return "unresolved_change"

    @property
    def signature(self) -> str:
        return sha256_payload(self.safe_payload)[:20]

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "persistent_moves": self.persistent_moves,
            "persistent_transformations": self.persistent_transformations,
            "relations_added": list(self.relations_added),
            "relations_removed": list(self.relations_removed),
            "action_space_changed": self.action_space_changed,
            "noop": self.noop,
            "game_over": self.game_over,
            "level_delta": self.level_delta,
            "quality": self.quality,
            "mode": self.mode,
        }


def causal_outcome_from_mt_transition(
    transition: Any,
    *,
    action_space_changed: bool = False,
    level_delta: int = 0,
    game_over: bool = False,
) -> CausalOutcome:
    """Compile effects only through confident persistent one-to-one matches."""

    correspondences = tuple(
        row
        for row in transition.correspondences
        if row.kind == "persist"
        and len(row.before_ids) == 1
        and len(row.after_ids) == 1
        and float(row.confidence) >= MINIMUM_CORRESPONDENCE_CONFIDENCE
    )
    before_ids = [row.before_ids[0] for row in correspondences]
    after_ids = [row.after_ids[0] for row in correspondences]
    if len(before_ids) != len(set(before_ids)) or len(after_ids) != len(set(after_ids)):
        correspondences = ()
    mapping = {row.before_ids[0]: row.after_ids[0] for row in correspondences}
    reverse = {after_id: before_id for before_id, after_id in mapping.items()}
    before_nodes = {node.node_id: node for node in transition.graph_before.nodes}
    after_nodes = {node.node_id: node for node in transition.graph_after.nodes}
    moves = 0
    transformations = 0
    for before_id, after_id in mapping.items():
        left = before_nodes[before_id]
        right = after_nodes[after_id]
        moves += int(math.dist(left.center, right.center) > 0.75)
        transformations += int(
            (
                left.kind,
                left.roles,
                left.area_bucket,
                left.aspect_bucket,
                left.compactness_bucket,
                left.holes,
                left.boundary_contacts,
            )
            != (
                right.kind,
                right.roles,
                right.area_bucket,
                right.aspect_bucket,
                right.compactness_bucket,
                right.holes,
                right.boundary_contacts,
            )
        )

    before_relations = {
        (relation.kind, relation.subject_id, relation.object_id)
        for relation in transition.graph_before.relations
        if relation.kind in ADMISSIBLE_RELATIONS
        and relation.subject_id in mapping
        and relation.object_id in mapping
    }
    mapped_before = {
        (kind, mapping[subject], mapping[obj])
        for kind, subject, obj in before_relations
    }
    after_relations = {
        (relation.kind, relation.subject_id, relation.object_id)
        for relation in transition.graph_after.relations
        if relation.kind in ADMISSIBLE_RELATIONS
        and relation.subject_id in reverse
        and relation.object_id in reverse
    }
    added = {kind for kind, _, _ in after_relations - mapped_before}
    removed = {kind for kind, _, _ in mapped_before - after_relations}
    confidences = [float(row.confidence) for row in correspondences]
    quality = statistics.fmean(confidences) if confidences else 0.0
    nonpersistent_change = any(
        row.kind in {"birth", "death", "merge", "split"}
        for row in transition.correspondences
    )
    productive_or_observed = bool(
        moves
        or transformations
        or added
        or removed
        or action_space_changed
        or level_delta
        or game_over
        or nonpersistent_change
    )
    return CausalOutcome(
        persistent_moves=moves,
        persistent_transformations=transformations,
        relations_added=tuple(added),
        relations_removed=tuple(removed),
        action_space_changed=action_space_changed,
        noop=not productive_or_observed,
        game_over=game_over,
        level_delta=level_delta,
        quality=quality,
    )


def abstract_context_signature(state: AbstractState | None) -> str:
    """D4-, palette-, candidate-order-, and entity-id-free state context."""

    if state is None:
        payload: Mapping[str, Any] = {"context": "unknown"}
    else:
        role_profiles = Counter(
            tuple(sorted(str(role).lower() for role in entity.roles))
            for entity in state.entities
        )
        predicate_counts = Counter(fact.predicate for fact in state.true_facts)
        payload = {
            "role_profiles": sorted((list(key), value) for key, value in role_profiles.items()),
            "predicates": sorted(predicate_counts.items()),
            "topology_values": sorted(int(value) for _, value in state.topology),
            "counter_values": sorted(round(float(value), 6) for _, value in state.counters),
            "regime": int(state.regime_index),
        }
    return sha256_payload(payload)[:20]


@dataclass
class _FamilyRecord:
    probability: float
    support: int = 0
    contradictions: int = 0
    contexts: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class BeliefUpdate:
    likelihoods: Mapping[str, float]
    posterior: Mapping[str, float]
    predicted_probability: float
    predicted_family: str


class TargetCausalBeliefStore:
    """One reset's causal posterior; grounded keys never leave memory."""

    def __init__(self, prior: Mapping[str, float]) -> None:
        normalised = _normalise_distribution(prior)
        self._initial_prior = dict(normalised)
        self._records = {
            family: _FamilyRecord(probability=normalised[family])
            for family in MODEL_FAMILIES
        }
        self._observations = 0
        self._last_candidate_key = ""
        self._last_intervention: InterventionSignature | None = None
        self._last_outcome: CausalOutcome | None = None
        self._last_context = ""
        self._candidate_uses: Counter[str] = Counter()
        self._candidate_noops: Counter[str] = Counter()
        self._candidate_risks: Counter[str] = Counter()
        self._seen_candidate_keys: set[str] = set()

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def posterior(self) -> Mapping[str, float]:
        return MappingProxyType(
            {family: self._records[family].probability for family in MODEL_FAMILIES}
        )

    @property
    def ranked_families(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                MODEL_FAMILIES,
                key=lambda family: (-self._records[family].probability, family),
            )
        )

    @property
    def top_family(self) -> str:
        return self.ranked_families[0]

    @property
    def margin(self) -> float:
        ranked = self.ranked_families
        return self._records[ranked[0]].probability - self._records[ranked[1]].probability

    def context_count(self, family: str) -> int:
        return len(self._records[family].contexts)

    def candidate_uses(self, local_key: str) -> int:
        return int(self._candidate_uses[local_key])

    def candidate_noop_rate(self, local_key: str) -> float:
        return self._candidate_noops[local_key] / max(1, self._candidate_uses[local_key])

    def candidate_risk_rate(self, local_key: str) -> float:
        return self._candidate_risks[local_key] / max(1, self._candidate_uses[local_key])

    @property
    def seen_candidate_keys(self) -> frozenset[str]:
        return frozenset(self._seen_candidate_keys)

    @property
    def last_candidate_key(self) -> str:
        return self._last_candidate_key

    @property
    def last_intervention(self) -> InterventionSignature | None:
        return self._last_intervention

    def _likelihoods(
        self,
        *,
        intervention: InterventionSignature,
        outcome: CausalOutcome,
        context: str,
        local_candidate_key: str,
    ) -> dict[str, float]:
        if self._last_outcome is None:
            productive = outcome.productive
            return {
                "stable_repeat": 0.58 if productive else 0.16,
                "relational_successor": (
                    0.68 if productive and (outcome.relations_added or outcome.relations_removed) else 0.52 if productive else 0.12
                ),
                "state_conditioned_switch": 0.45 if productive else 0.18,
                "null_or_unsafe": 0.95 if outcome.noop or outcome.game_over else 0.05,
            }

        same_candidate = local_candidate_key == self._last_candidate_key
        same_signature = intervention == self._last_intervention
        same_outcome = outcome.signature == self._last_outcome.signature
        context_changed = context != self._last_context
        productive = outcome.productive
        relation_change = bool(outcome.relations_added or outcome.relations_removed)

        if outcome.noop or outcome.game_over:
            repeat = 0.05
        elif same_candidate and same_outcome and productive:
            repeat = 0.94
        elif same_signature and same_outcome and productive:
            repeat = 0.78
        elif not same_candidate and productive:
            repeat = 0.08
        else:
            repeat = 0.20

        if productive and not same_candidate and relation_change:
            successor = 0.94
        elif productive and not same_candidate:
            successor = 0.72
        elif productive and same_candidate:
            successor = 0.08
        else:
            successor = 0.10

        if productive and context_changed and not same_signature:
            switching = 0.92
        elif productive and context_changed and not same_candidate:
            switching = 0.72
        elif productive and not same_candidate:
            switching = 0.42
        else:
            switching = 0.16

        unsafe = 0.97 if outcome.noop or outcome.game_over else 0.05
        return {
            "stable_repeat": repeat,
            "relational_successor": successor,
            "state_conditioned_switch": switching,
            "null_or_unsafe": unsafe,
        }

    def update(
        self,
        *,
        intervention: InterventionSignature,
        outcome: CausalOutcome,
        context: str,
        local_candidate_key: str,
        predicted_family: str,
    ) -> BeliefUpdate:
        if predicted_family not in MODEL_FAMILIES:
            raise ValueError(f"unknown predicted family: {predicted_family}")
        likelihoods = self._likelihoods(
            intervention=intervention,
            outcome=outcome,
            context=context,
            local_candidate_key=local_candidate_key,
        )
        weighted = {
            family: max(1e-9, self._records[family].probability * likelihoods[family])
            for family in MODEL_FAMILIES
        }
        total = sum(weighted.values())
        for family in MODEL_FAMILIES:
            record = self._records[family]
            record.probability = weighted[family] / total
            if likelihoods[family] >= 0.50:
                record.support += 1
                if outcome.quality >= MINIMUM_CORRESPONDENCE_CONFIDENCE:
                    record.contexts.add(context)
            elif likelihoods[family] <= 0.10:
                record.contradictions += 1

        self._observations += 1
        self._candidate_uses[local_candidate_key] += 1
        self._candidate_noops[local_candidate_key] += int(outcome.noop)
        self._candidate_risks[local_candidate_key] += int(outcome.game_over)
        self._seen_candidate_keys.add(local_candidate_key)
        self._last_candidate_key = local_candidate_key
        self._last_intervention = intervention
        self._last_outcome = outcome
        self._last_context = context
        posterior = {family: self._records[family].probability for family in MODEL_FAMILIES}
        return BeliefUpdate(
            likelihoods=MappingProxyType(likelihoods),
            posterior=MappingProxyType(posterior),
            predicted_probability=float(likelihoods[predicted_family]),
            predicted_family=predicted_family,
        )

    def demote(self, family: str) -> None:
        if family not in MODEL_FAMILIES:
            raise ValueError(f"unknown family: {family}")
        values = {name: self._records[name].probability for name in MODEL_FAMILIES}
        values[family] = min(values[family], 1e-4)
        values = _normalise_distribution(values)
        for name in MODEL_FAMILIES:
            self._records[name].probability = values[name]

    def snapshot(self) -> dict[str, Any]:
        """Return aggregates only; no local key or context digest is exposed."""

        payload = {
            "format_version": FORMAT_VERSION,
            "observations": self._observations,
            "posterior": {
                family: self._records[family].probability for family in MODEL_FAMILIES
            },
            "families": {
                family: {
                    "support": self._records[family].support,
                    "contradictions": self._records[family].contradictions,
                    "distinct_contexts": len(self._records[family].contexts),
                }
                for family in MODEL_FAMILIES
            },
            "distinct_interventions": len(self._seen_candidate_keys),
        }
        assert_causal_transfer_safe(payload)
        return payload


class CausalProcedurePrior:
    """Signed, transfer-safe prior over procedure families."""

    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        kind: str = "source_informed",
    ) -> None:
        if payload is not None:
            if payload.get("format_version") != PRIOR_FORMAT_VERSION:
                raise ValueError("causal prior format drifted")
            verify_signed(payload, "prior_checksum")
            assert_causal_transfer_safe(payload)
            weights = payload.get("weights", {})
            kind = str(payload.get("kind", ""))
        if weights is None:
            weights = {family: 0.25 for family in MODEL_FAMILIES}
        self._weights = _normalise_distribution(weights)
        self.kind = _identifier(kind, label="prior kind")
        if self.kind not in {"source_informed", "uniform", "permuted_source"}:
            raise ValueError(f"unsupported causal prior kind: {self.kind}")
        assert_causal_transfer_safe(self.safe_payload)

    @property
    def weights(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._weights))

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": PRIOR_FORMAT_VERSION,
            "kind": self.kind,
            "weights": {family: self._weights[family] for family in MODEL_FAMILIES},
        }

    @property
    def prior_checksum(self) -> str:
        return sha256_payload(self.safe_payload)

    def snapshot(self) -> dict[str, Any]:
        return signed(self.safe_payload, "prior_checksum")


CausalPrior = CausalProcedurePrior


def uniform_prior() -> CausalProcedurePrior:
    return CausalProcedurePrior(
        weights={family: 0.25 for family in MODEL_FAMILIES},
        kind="uniform",
    )


def permuted_prior(prior: CausalProcedurePrior) -> CausalProcedurePrior:
    """Rotate by two families, preserving mass, norm, and entropy exactly."""

    values = prior.weights
    return CausalProcedurePrior(
        weights={
            family: values[MODEL_FAMILIES[(index + 2) % len(MODEL_FAMILIES)]]
            for index, family in enumerate(MODEL_FAMILIES)
        },
        kind="permuted_source",
    )


@dataclass(frozen=True)
class SourceProcedureProjection:
    """Generic root-group projection used only by the offline compiler."""

    source_slot: str
    group_index: int
    inferred_family: str
    outcome_mode: str
    correspondence_confidence: float = 1.0
    persistent_one_to_one: bool = True
    ambiguous: bool = False
    relation_conflict: bool = False
    birth_or_death_relation: bool = False
    level_delta: int = 0
    terminal_chain_link: bool = False

    def __post_init__(self) -> None:
        slot = _identifier(self.source_slot, label="source slot")
        if slot not in {"source_a", "source_b"}:
            raise ValueError("source slot must be source_a or source_b")
        object.__setattr__(self, "source_slot", slot)
        if int(self.group_index) < 0:
            raise ValueError("group_index must be non-negative")
        object.__setattr__(self, "group_index", int(self.group_index))
        if self.inferred_family not in MODEL_FAMILIES:
            raise ValueError(f"unsupported inferred family: {self.inferred_family}")
        object.__setattr__(self, "outcome_mode", _identifier(self.outcome_mode, label="outcome mode"))
        confidence = float(self.correspondence_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("correspondence confidence must be in [0, 1]")
        object.__setattr__(self, "correspondence_confidence", confidence)
        object.__setattr__(self, "level_delta", max(0, int(self.level_delta)))
        object.__setattr__(self, "terminal_chain_link", bool(self.terminal_chain_link))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceProcedureProjection":
        return cls(
            source_slot=str(value.get("source_slot", "")),
            group_index=int(value.get("group_index", value.get("root_group", -1))),
            inferred_family=str(value.get("inferred_family", value.get("family", ""))),
            outcome_mode=str(value.get("outcome_mode", value.get("mode", ""))),
            correspondence_confidence=float(value.get("correspondence_confidence", value.get("quality", 0.0))),
            persistent_one_to_one=bool(value.get("persistent_one_to_one", False)),
            ambiguous=bool(value.get("ambiguous", False)),
            relation_conflict=bool(value.get("relation_conflict", False)),
            birth_or_death_relation=bool(value.get("birth_or_death_relation", False)),
            level_delta=int(value.get("level_delta", 0)),
            terminal_chain_link=bool(value.get("terminal_chain_link", False)),
        )

    @property
    def admissible(self) -> bool:
        return bool(
            self.correspondence_confidence >= MINIMUM_CORRESPONDENCE_CONFIDENCE
            and self.persistent_one_to_one
            and not self.ambiguous
            and not self.relation_conflict
            and not self.birth_or_death_relation
        )


class CausalLabelQAError(ValueError):
    pass


class ProcedureNotSourceIdentifiableError(ValueError):
    pass


def _coerce_projections(
    values: Sequence[SourceProcedureProjection | Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[SourceProcedureProjection, ...]:
    if isinstance(values, Mapping):
        nested = values.get("projections", values.get("rows", ()))
        values = tuple(nested) if isinstance(nested, Sequence) else ()
    output = []
    for value in values:
        output.append(
            value if isinstance(value, SourceProcedureProjection) else SourceProcedureProjection.from_mapping(value)
        )
    return tuple(output)


def qa_source_projections(
    projections: Sequence[SourceProcedureProjection | Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, Any]:
    rows = _coerce_projections(projections)
    admissible = tuple(row for row in rows if row.admissible)
    reasons: list[str] = []
    per_source: dict[str, dict[str, Any]] = {}
    for slot in ("source_a", "source_b"):
        selected = tuple(row for row in admissible if row.source_slot == slot)
        modes = Counter(row.outcome_mode for row in selected)
        dominant = max(modes.values(), default=0) / max(1, sum(modes.values()))
        if len(modes) < 2:
            reasons.append(f"{slot}:fewer_than_two_modes")
        if dominant >= 0.95:
            reasons.append(f"{slot}:universal_or_near_universal_mode")
        if not selected:
            reasons.append(f"{slot}:no_admissible_projection")
        per_source[slot] = {
            "admissible_projections": len(selected),
            "root_groups": len({row.group_index for row in selected}),
            "outcome_modes": len(modes),
            "dominant_mode_fraction": dominant,
            "winning_projections": sum(row.level_delta > 0 for row in selected),
            "terminal_chain_link_projections": sum(
                row.terminal_chain_link for row in selected
            ),
        }
    payload = {
        "format_version": COMPILATION_FORMAT_VERSION,
        "passed": not reasons,
        "verdict": "PASS_CAUSAL_LABEL_QA" if not reasons else "CAUSAL_LABEL_QA_MISS",
        "input_projections": len(rows),
        "admissible_projections": len(admissible),
        "rejected_projections": len(rows) - len(admissible),
        "per_source": per_source,
        "source_contribution_fractions": {"source_a": 0.5, "source_b": 0.5},
        "reasons": reasons,
    }
    assert_causal_transfer_safe(payload)
    return payload


qa_source_artifacts = qa_source_projections


def _fit_source_weights(rows: Sequence[SourceProcedureProjection]) -> dict[str, float]:
    components: dict[str, dict[str, float]] = {}
    for slot in ("source_a", "source_b"):
        selected = tuple(row for row in rows if row.source_slot == slot and row.admissible)
        if not selected:
            raise ProcedureNotSourceIdentifiableError(f"{slot} has no admissible evidence")
        counts = Counter(row.inferred_family for row in selected)
        smoothed = {family: float(counts[family]) + 0.10 for family in MODEL_FAMILIES}
        total = sum(smoothed.values())
        observational = {
            family: smoothed[family] / total for family in MODEL_FAMILIES
        }
        terminal_rows = tuple(row for row in selected if row.terminal_chain_link)
        if terminal_rows:
            terminal_counts = Counter(row.inferred_family for row in terminal_rows)
            terminal_total = sum(terminal_counts.values())
            terminal = {
                family: terminal_counts[family] / terminal_total
                for family in MODEL_FAMILIES
            }
            # All signed interventions remain represented, while the causal
            # family recovered from the shortest terminal chain receives an
            # equal, explicit share of that source's contribution.
            components[slot] = {
                family: 0.5 * observational[family] + 0.5 * terminal[family]
                for family in MODEL_FAMILIES
            }
        else:
            # Synthetic/unit callers may exercise the generic compiler without
            # a terminal journal.  The physical runtime separately requires a
            # terminal-chain link from each real source before target access.
            components[slot] = observational
    weights = {
        family: 0.5 * components["source_a"][family] + 0.5 * components["source_b"][family]
        for family in MODEL_FAMILIES
    }
    maximum = max(weights.values())
    if maximum > 0.70:
        blend = (maximum - 0.70) / (maximum - 0.25)
        weights = {
            family: (1.0 - blend) * value + blend * 0.25
            for family, value in weights.items()
        }
    return _normalise_distribution(weights)


@dataclass(frozen=True)
class SourcePriorCompilation:
    prior: CausalProcedurePrior
    uniform: CausalProcedurePrior
    permuted: CausalProcedurePrior
    qa: Mapping[str, Any]
    projection_count: int

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "format_version": COMPILATION_FORMAT_VERSION,
            "prior": self.prior.snapshot(),
            "uniform": self.uniform.snapshot(),
            "permuted": self.permuted.snapshot(),
            "qa": dict(self.qa),
            "projection_count": int(self.projection_count),
            "source_contribution_fractions": {"source_a": 0.5, "source_b": 0.5},
        }
        assert_causal_transfer_safe(payload)
        return signed(payload, "compilation_checksum")


def compile_source_prior(
    projections: Sequence[SourceProcedureProjection | Mapping[str, Any]] | Mapping[str, Any],
) -> SourcePriorCompilation:
    rows = _coerce_projections(projections)
    qa = qa_source_projections(rows)
    if not qa["passed"]:
        raise CausalLabelQAError("CAUSAL_LABEL_QA_MISS: " + ",".join(qa["reasons"]))
    weights = _fit_source_weights(rows)
    if max(weights.values()) - min(weights.values()) <= 1e-9:
        raise ProcedureNotSourceIdentifiableError("source prior is uniform")
    if max(weights.values()) > 0.70 + 1e-12:
        raise ProcedureNotSourceIdentifiableError("source prior exceeds maximum mass")
    source = CausalProcedurePrior(weights=weights, kind="source_informed")
    return SourcePriorCompilation(
        prior=source,
        uniform=uniform_prior(),
        permuted=permuted_prior(source),
        qa=MappingProxyType(qa),
        projection_count=len(rows),
    )


def compile_causal_procedure_prior(
    source_payloads: Sequence[SourceProcedureProjection | Mapping[str, Any]] | Mapping[str, Any],
) -> CausalProcedurePrior:
    """Compatibility entry point returning only the transfer-safe prior."""

    return compile_source_prior(source_payloads).prior


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else math.inf


def evaluate_source_prior(
    prior: CausalProcedurePrior,
    projections: Sequence[SourceProcedureProjection | Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, Any]:
    """Grouped leave-one-root-out source check; never splits individual rows."""

    rows = tuple(row for row in _coerce_projections(projections) if row.admissible)
    expected = CausalProcedurePrior(
        weights=_fit_source_weights(rows),
        kind="source_informed",
    )
    if expected.prior_checksum != prior.prior_checksum:
        raise ValueError("evaluated prior is detached from the supplied projections")
    per_source_raw: dict[str, dict[str, list[float]]] = {
        slot: defaultdict(list) for slot in ("source_a", "source_b")
    }
    terminal_links_available = {
        slot: any(row.terminal_chain_link for row in rows if row.source_slot == slot)
        for slot in ("source_a", "source_b")
    }
    folds = 0
    for slot, group in sorted({(row.source_slot, row.group_index) for row in rows}):
        held = tuple(row for row in rows if row.source_slot == slot and row.group_index == group)
        training = tuple(
            row for row in rows if not (row.source_slot == slot and row.group_index == group)
        )
        if not held or {row.source_slot for row in training} != {"source_a", "source_b"}:
            continue
        fold_prior = CausalProcedurePrior(
            weights=_fit_source_weights(training),
            kind="source_informed",
        )
        wrong = permuted_prior(fold_prior)
        for row in held:
            p = max(1e-12, float(fold_prior.weights[row.inferred_family]))
            q = max(1e-12, float(wrong.weights[row.inferred_family]))
            per_source_raw[slot]["source_log_loss"].append(-math.log(p))
            per_source_raw[slot]["permuted_log_loss"].append(-math.log(q))
            # Identification cost concerns the terminally linked winning
            # family, not the cost of predicting every noop or unsafe probe.
            if row.terminal_chain_link or not terminal_links_available[slot]:
                per_source_raw[slot]["source_cost"].append(1.0 / p)
                per_source_raw[slot]["uniform_cost"].append(4.0)
        folds += 1

    per_source: dict[str, dict[str, Any]] = {}
    improvements: list[float] = []
    cost_reductions: list[float] = []
    for slot in ("source_a", "source_b"):
        raw = per_source_raw[slot]
        source_loss = statistics.fmean(raw["source_log_loss"]) if raw["source_log_loss"] else math.inf
        wrong_loss = statistics.fmean(raw["permuted_log_loss"]) if raw["permuted_log_loss"] else math.inf
        improvement = (
            1.0 - source_loss / wrong_loss
            if math.isfinite(source_loss) and math.isfinite(wrong_loss) and wrong_loss > 0
            else -math.inf
        )
        source_cost = _median(raw["source_cost"])
        uniform_cost = _median(raw["uniform_cost"])
        reduction = (
            1.0 - source_cost / uniform_cost
            if math.isfinite(source_cost) and math.isfinite(uniform_cost) and uniform_cost > 0
            else -math.inf
        )
        improvements.append(improvement)
        cost_reductions.append(reduction)
        per_source[slot] = {
            "held_out_rows": len(raw["source_log_loss"]),
            "source_log_loss": source_loss,
            "permuted_log_loss": wrong_loss,
            "log_loss_improvement_fraction": improvement,
            "source_identification_cost": source_cost,
            "uniform_identification_cost": uniform_cost,
            "identification_cost_reduction_fraction": reduction,
        }
    passed = bool(
        folds >= 2
        and all(value >= 0.05 for value in improvements)
        and max(cost_reductions, default=-math.inf) >= 0.20
        and min(cost_reductions, default=-math.inf) >= -1e-12
    )
    payload = {
        "format_version": COMPILATION_FORMAT_VERSION,
        "passed": passed,
        "verdict": (
            "PASS_SOURCE_PROCEDURE_IDENTIFICATION"
            if passed
            else "PROCEDURE_NOT_SOURCE_IDENTIFIABLE"
        ),
        "grouped_leave_one_root_out": True,
        "folds": folds,
        "per_source": per_source,
        "prior_checksum": prior.prior_checksum,
    }
    assert_causal_transfer_safe(payload)
    return payload


def preflight_prior(prior: CausalProcedurePrior) -> dict[str, Any]:
    snapshot = prior.snapshot()
    restored = CausalProcedurePrior(snapshot)
    wrong = permuted_prior(restored)
    entropy = -sum(value * math.log(value) for value in restored.weights.values())
    wrong_entropy = -sum(value * math.log(value) for value in wrong.weights.values())
    passed = bool(
        restored.prior_checksum == prior.prior_checksum
        and math.isclose(sum(restored.weights.values()), 1.0, abs_tol=1e-12)
        and math.isclose(entropy, wrong_entropy, abs_tol=1e-12)
        and sorted(restored.weights.values()) == sorted(wrong.weights.values())
    )
    return {
        "format_version": FORMAT_VERSION,
        "passed": passed,
        "prior_checksum": prior.prior_checksum,
        "permutation_preserves_entropy": math.isclose(entropy, wrong_entropy, abs_tol=1e-12),
        "physical_actions": 0,
    }


@dataclass(frozen=True)
class ProcedureDecision:
    candidate: ActionCandidate | None
    reason: str
    phase: str
    predicted_family: str
    predicted_probability: float
    intervention: InterventionSignature | None
    program_hash: str
    candidates_inspected: int

    @property
    def abstained(self) -> bool:
        return self.candidate is None

    @property
    def safe_payload(self) -> dict[str, Any]:
        payload = {
            "reason": self.reason,
            "phase": self.phase,
            "predicted_family": self.predicted_family,
            "predicted_probability": self.predicted_probability,
            "intervention": None if self.intervention is None else self.intervention.safe_payload,
            "program_hash": self.program_hash,
            "candidates_inspected": self.candidates_inspected,
            "abstained": self.abstained,
        }
        assert_causal_transfer_safe(payload)
        return payload


@dataclass(frozen=True)
class ProcedureUpdate:
    phase_before: str
    phase_after: str
    predicted_family: str
    predicted_probability: float
    outcome: CausalOutcome
    posterior: Mapping[str, float]
    mismatch: bool
    revised: bool
    abstained: bool
    reason: str

    @property
    def safe_payload(self) -> dict[str, Any]:
        payload = {
            "phase_before": self.phase_before,
            "phase_after": self.phase_after,
            "predicted_family": self.predicted_family,
            "predicted_probability": self.predicted_probability,
            "outcome": self.outcome.safe_payload,
            "posterior": {family: self.posterior[family] for family in MODEL_FAMILIES},
            "mismatch": self.mismatch,
            "revised": self.revised,
            "abstained": self.abstained,
            "reason": self.reason,
        }
        assert_causal_transfer_safe(payload)
        return payload


class CausalProcedureController:
    """Bounded causal experiment designer and reset-local feedback controller."""

    def __init__(
        self,
        arm: str,
        scope: int = 0,
        prior: CausalProcedurePrior | Mapping[str, Any] | None = None,
        spec: CausalProcedureSpec | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported causal procedure arm: {arm}")
        if prior is None:
            if arm != "uniform_closed_loop":
                raise ValueError("source-conditioned arm requires a compiled source prior")
            source_prior = uniform_prior()
        elif isinstance(prior, CausalProcedurePrior):
            source_prior = prior
        elif prior.get("format_version") == PRIOR_FORMAT_VERSION:
            source_prior = CausalProcedurePrior(prior)
        else:
            source_prior = CausalProcedurePrior(weights=prior, kind="source_informed")
        if arm != "uniform_closed_loop" and (
            source_prior.kind != "source_informed"
            or max(source_prior.weights.values()) - min(source_prior.weights.values()) <= 1e-9
        ):
            raise ValueError("source-conditioned arm requires a non-uniform source prior")
        if arm == "uniform_closed_loop":
            selected_prior = uniform_prior()
        elif arm == "permuted_source_closed_loop":
            selected_prior = permuted_prior(source_prior)
        else:
            selected_prior = source_prior
        closed_loop = arm != "source_open_loop"
        if spec is None:
            spec = CausalProcedureSpec(
                prior=selected_prior.weights,
                closed_loop_revision=closed_loop,
            )
        elif bool(spec.closed_loop_revision) != closed_loop:
            raise ValueError("arm and spec revision policy disagree")
        self.arm = arm
        self.scope = int(scope) % 4
        self.prior = selected_prior
        self.spec = spec
        self.store = TargetCausalBeliefStore(spec.prior)
        self._phase = ProcedurePhase.IDENTIFY
        self._pending: ProcedureDecision | None = None
        self._pending_local_key = ""
        self._pending_binding_key = ""
        self._last_binding_key = ""
        self._actions = 0
        self._actions_in_option = 0
        self._revisions = 0
        self._mismatches = 0
        self._stagnation = 0
        self._abstention_reason = ""
        self._locked_family = ""
        self._seen_contexts: set[str] = set()
        self._updates: list[ProcedureUpdate] = []

    @property
    def phase(self) -> ProcedurePhase:
        return self._phase

    def reset(self) -> None:
        """Erase all target-local support; priors and frozen spec remain."""

        self.store = TargetCausalBeliefStore(self.spec.prior)
        self._phase = ProcedurePhase.IDENTIFY
        self._pending = None
        self._pending_local_key = ""
        self._pending_binding_key = ""
        self._last_binding_key = ""
        self._actions = 0
        self._actions_in_option = 0
        self._revisions = 0
        self._mismatches = 0
        self._stagnation = 0
        self._abstention_reason = ""
        self._locked_family = ""
        self._seen_contexts.clear()
        self._updates.clear()

    def _abstain(self, reason: str, *, inspected: int = 0) -> ProcedureDecision:
        self._phase = ProcedurePhase.ABSTAIN
        self._abstention_reason = reason
        decision = ProcedureDecision(
            candidate=None,
            reason=reason,
            phase=self._phase.value,
            predicted_family=self._locked_family or self.store.top_family,
            predicted_probability=float(self.store.posterior[self.store.top_family]),
            intervention=None,
            program_hash=self.spec.spec_hash,
            candidates_inspected=inspected,
        )
        self._pending = None
        return decision

    def _candidate_score(
        self,
        state: AbstractState | None,
        candidate: ActionCandidate,
        signature: InterventionSignature,
        *,
        binding_key: str,
        family: str,
        rank: int,
    ) -> float:
        local_key = candidate.key
        uses = self.store.candidate_uses(local_key)
        novelty = 1.0 / (1.0 + uses)
        score = self.spec.information_weight * novelty
        score -= self.spec.redundancy_weight * min(1.0, uses / 4.0)
        score -= self.spec.risk_weight * self.store.candidate_risk_rate(local_key)
        score -= 0.5 * self.spec.risk_weight * self.store.candidate_noop_rate(local_key)
        if family == "stable_repeat":
            score += self.spec.repeatability_weight * float(local_key == self.store.last_candidate_key)
        elif family == "relational_successor":
            same_abstract_binding = bool(
                self.store.last_intervention is not None
                and signature.action_family == self.store.last_intervention.action_family
                and signature.target_role == self.store.last_intervention.target_role
            )
            distinct_binding = bool(binding_key and binding_key != self._last_binding_key)
            score += self.spec.composability_weight * float(
                same_abstract_binding and distinct_binding
            )
            score += self.spec.composability_weight * float(
                _relation_connects(state, self._last_binding_key, binding_key)
            )
            score += 0.20 * float(local_key not in self.store.seen_candidate_keys)
        elif family == "state_conditioned_switch":
            score += self.spec.composability_weight * float(
                self.store.last_intervention is not None and signature != self.store.last_intervention
            )
        elif family == "null_or_unsafe":
            score -= 0.5
        return score - rank * 1e-9

    def propose(
        self,
        state: AbstractState | None,
        legal_candidates: Sequence[Any] = (),
        *,
        candidates: Sequence[Any] | None = None,
        shape: tuple[int, int] | None = None,
        step_index: int | None = None,
    ) -> ProcedureDecision:
        del shape
        if candidates is not None:
            if legal_candidates:
                raise ValueError("provide legal_candidates or candidates, not both")
            legal_candidates = candidates
        if self._phase is ProcedurePhase.ABSTAIN:
            return self._abstain(self._abstention_reason or "already_abstained")
        if self._pending is not None:
            raise RuntimeError("observe must seal the previous proposal before another propose")
        if self._phase is ProcedurePhase.REVISE:
            self._phase = ProcedurePhase.IDENTIFY
        if self._actions_in_option >= self.spec.option_horizon:
            return self._abstain("option_horizon_exhausted")
        if step_index is not None and int(step_index) >= 48:
            return self._abstain("work_budget_exhausted")
        all_legal = normalized_action_candidates(legal_candidates)
        legal = _scope_sample(
            all_legal,
            maximum=self.spec.maximum_candidates,
            scope=self.scope,
        )
        if not legal:
            return self._abstain("no_legal_candidate")
        family = self._locked_family or self.store.top_family
        if family == "null_or_unsafe" and self._phase in {
            ProcedurePhase.VERIFY,
            ProcedurePhase.CONTROL,
        }:
            return self._abstain("null_or_unsafe_verified", inspected=len(legal))
        scored = []
        for rank, candidate in enumerate(legal):
            binding = _candidate_target_entity(state, candidate)
            binding_key = "" if binding is None else str(binding.entity_id)
            signature = InterventionSignature.from_candidate(
                candidate,
                target_role=candidate_target_role(state, candidate),
            )
            scored.append(
                (
                    self._candidate_score(
                        state,
                        candidate,
                        signature,
                        binding_key=binding_key,
                        family=family,
                        rank=rank,
                    ),
                    candidate,
                    signature,
                    binding_key,
                )
            )
        _, chosen, signature, binding_key = max(scored, key=lambda item: item[0])
        reason = {
            ProcedurePhase.IDENTIFY: "discriminating_intervention",
            ProcedurePhase.VERIFY: "causal_repeatability_verification",
            ProcedurePhase.CONTROL: "causal_control_execution",
        }.get(self._phase, "causal_intervention")
        decision = ProcedureDecision(
            candidate=chosen,
            reason=reason,
            phase=self._phase.value,
            predicted_family=family,
            predicted_probability=float(self.store.posterior[family]),
            intervention=signature,
            program_hash=self.spec.spec_hash,
            candidates_inspected=len(legal),
        )
        self._pending = decision
        self._pending_local_key = chosen.key
        self._pending_binding_key = binding_key
        return decision

    def _revise(self, family: str, reason: str) -> bool:
        del reason
        if not self.spec.closed_loop_revision:
            return False
        if self._revisions >= self.spec.maximum_revisions:
            self._phase = ProcedurePhase.ABSTAIN
            self._abstention_reason = "revision_budget_exhausted"
            return False
        self._revisions += 1
        self.store.demote(family)
        self._locked_family = ""
        self._actions_in_option = 0
        self._phase = ProcedurePhase.REVISE
        return True

    def observe(
        self,
        observed_transition: ObservedTransition | None = None,
        *,
        state_before: AbstractState | None = None,
        state_after: AbstractState | None = None,
        selected: ActionCandidate | None = None,
        level_delta: int = 0,
        game_over: bool = False,
        action_space_changed: bool = False,
        correspondence_quality: float = 1.0,
        outcome: CausalOutcome | None = None,
    ) -> ProcedureUpdate:
        if self._pending is None or self._pending.candidate is None:
            raise RuntimeError("observe requires one unsealed proposal")
        pending = self._pending
        if observed_transition is None:
            chosen = selected or pending.candidate
            before = state_before or AbstractState()
            after = state_after or before.with_updates(
                counters={
                    "levels_completed": before.counter("levels_completed", 0.0)
                    + max(0, int(level_delta))
                }
            )
            events = []
            if level_delta:
                events.extend(("progress", "level_complete"))
            if game_over:
                events.append("game_over")
            observed_transition = ObservedTransition(
                state_before=before,
                action=chosen,
                state_after=after,
                observation=PredictionPacket(
                    progress_mean=float(max(0, int(level_delta))),
                    terminal_probability=float(bool(game_over)),
                    known_channels=frozenset({"progress", "terminal"}),
                    state_after=after,
                ),
                events=tuple(events),
            )
        if observed_transition.action.key != pending.candidate.key:
            raise ValueError("observed action does not match pending proposal")
        if outcome is None:
            outcome = CausalOutcome.from_observed_transition(
                observed_transition,
                action_space_changed=action_space_changed,
                correspondence_quality=correspondence_quality,
            )
        context = abstract_context_signature(observed_transition.state_before)
        self._seen_contexts.add(context)
        phase_before = self._phase.value
        belief = self.store.update(
            intervention=pending.intervention or InterventionSignature.from_candidate(pending.candidate),
            outcome=outcome,
            context=context,
            local_candidate_key=self._pending_local_key,
            predicted_family=pending.predicted_family,
        )
        self._pending = None
        self._pending_local_key = ""
        self._last_binding_key = self._pending_binding_key
        self._pending_binding_key = ""
        self._actions += 1
        self._actions_in_option += 1
        state_repeated = abstract_context_signature(observed_transition.state_after) in self._seen_contexts
        self._stagnation = self._stagnation + 1 if outcome.noop or state_repeated else 0
        mismatch = belief.predicted_probability < self.spec.predictive_demotion_threshold
        if mismatch:
            self._mismatches += 1

        revised = False
        reason = "posterior_updated"
        if outcome.level_delta > 0:
            self._phase = ProcedurePhase.ABSTAIN
            self._abstention_reason = "level_progress_observed"
            reason = "level_progress_observed"
        elif outcome.game_over:
            self._phase = ProcedurePhase.ABSTAIN
            self._abstention_reason = "game_over_observed"
            reason = "game_over_observed"
        elif mismatch:
            revised = self._revise(pending.predicted_family, "predictive_mismatch")
            if revised:
                reason = "predictive_mismatch_revised"
            elif self._phase is ProcedurePhase.ABSTAIN:
                reason = "revision_budget_exhausted"
            else:
                reason = "predictive_mismatch_open_loop_locked"
        elif self._stagnation >= self.spec.stagnation_limit:
            revised = self._revise(pending.predicted_family, "stagnation")
            if revised:
                reason = "stagnation_revised"
                self._stagnation = 0
            else:
                self._phase = ProcedurePhase.ABSTAIN
                self._abstention_reason = "open_loop_stagnation"
                reason = "open_loop_stagnation"
        else:
            top = self.store.top_family
            probability = self.store.posterior[top]
            enough = (
                probability >= self.spec.posterior_threshold
                and self.store.margin >= self.spec.posterior_margin
            )
            if self._phase is ProcedurePhase.IDENTIFY and enough:
                self._phase = ProcedurePhase.VERIFY
                reason = "hypothesis_ready_for_verification"
            elif (
                self._phase is ProcedurePhase.VERIFY
                and enough
                and self.store.context_count(top) >= self.spec.verification_contexts
            ):
                self._phase = ProcedurePhase.CONTROL
                self._locked_family = top
                reason = "hypothesis_verified_for_control"

        update = ProcedureUpdate(
            phase_before=phase_before,
            phase_after=self._phase.value,
            predicted_family=pending.predicted_family,
            predicted_probability=belief.predicted_probability,
            outcome=outcome,
            posterior=belief.posterior,
            mismatch=mismatch,
            revised=revised,
            abstained=self._phase is ProcedurePhase.ABSTAIN,
            reason=reason,
        )
        self._updates.append(update)
        return update

    def summary(self) -> dict[str, Any]:
        payload = {
            "format_version": FORMAT_VERSION,
            "arm": self.arm,
            "phase": self._phase.value,
            "prior_kind": self.prior.kind,
            "prior_checksum": self.prior.prior_checksum,
            "program_hash": self.spec.spec_hash,
            "actions": self._actions,
            "actions_in_option": self._actions_in_option,
            "revisions": self._revisions,
            "mismatches": self._mismatches,
            "stagnation": self._stagnation,
            "distinct_contexts": len(self._seen_contexts),
            "abstention_reason": self._abstention_reason,
            "level_deltas": sum(update.outcome.level_delta for update in self._updates),
            "game_over_actions": sum(update.outcome.game_over for update in self._updates),
            "noop_actions": sum(update.outcome.noop for update in self._updates),
            "belief": self.store.snapshot(),
            "grounded_payload_persisted": False,
            "legacy_fallback_actions": 0,
        }
        assert_causal_transfer_safe(payload)
        return payload


__all__ = [
    "ADMISSIBLE_RELATIONS",
    "ARMS",
    "COMPILATION_FORMAT_VERSION",
    "CausalLabelQAError",
    "CausalOutcome",
    "CausalPrior",
    "CausalProcedureController",
    "CausalProcedurePrior",
    "CausalProcedureSpec",
    "FORMAT_VERSION",
    "InterventionSignature",
    "MINIMUM_CORRESPONDENCE_CONFIDENCE",
    "MODEL_FAMILIES",
    "PHASE_ORDER",
    "PRIOR_FORMAT_VERSION",
    "ProcedureDecision",
    "ProcedureNotSourceIdentifiableError",
    "ProcedurePhase",
    "ProcedureUpdate",
    "SourcePriorCompilation",
    "SourceProcedureProjection",
    "TargetCausalBeliefStore",
    "abstract_context_signature",
    "assert_causal_transfer_safe",
    "candidate_target_role",
    "causal_outcome_from_mt_transition",
    "canonical_json",
    "compile_causal_procedure_prior",
    "compile_source_prior",
    "evaluate_source_prior",
    "permuted_prior",
    "preflight_prior",
    "qa_source_artifacts",
    "qa_source_projections",
    "sha256_payload",
    "signed",
    "uniform_prior",
    "verify_signed",
]
