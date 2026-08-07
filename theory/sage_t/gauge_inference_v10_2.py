"""SAGE.T10.2 joint relational-gauge posterior and decision engine.

This module composes around the frozen SAGE.T contracts.  It deliberately does
not subclass or modify :class:`ProgramPosterior`: one physical outcome is
scored once, frame evidence is averaged, and finite option state belongs to
each complete particle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

from .contracts import (
    AbstractState,
    ActionCandidate,
    JointProgramHypothesis,
    PredictionPacket,
)
from .executor import ProgramExecutor
from .frame_transport_v10_2 import (
    TransportMap,
    TransportOrbitWitness,
)
from .frame_transport_v10_2 import find_transport as resolve_official_transport
from .observer_frames_v10_2 import (
    OBSERVER_FRAME_SPECS,
    ObserverFrameSpec,
    observer_frame_spec,
)
from .posterior import DEFAULT_CHANNEL_WEIGHTS, packet_log_likelihood

FORMAT_VERSION = "sage-t10.2-joint-gauge-hypothesis-v1"
MAXIMUM_GAUGE_CLASSES = 256
MAXIMUM_DECISION_CLASSES = 64
MAP_MASS_THRESHOLD = 0.90
MAP_BAYES_FACTOR_THRESHOLD = 20.0
MAP_STABILITY_TRANSITIONS = 3
NEW_AST_NODE_LOG_PRIOR = -0.05
OPTION_ACTION_MISMATCH_LOG_LIKELIHOOD = -4.0
_GAME_TOKEN = re.compile(r"^[a-z]{2}[0-9]+(?:-[0-9a-f]{6,})?$")
_COORDINATE_TOKEN = re.compile(r"^-?[0-9]+(?:,-?[0-9]+)+$")
_UUID_TOKEN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_COLOR_TOKEN = re.compile(r"^#[0-9a-f]{3}(?:[0-9a-f]{3})?$", re.IGNORECASE)
_RAW_COLOR_SEGMENT = re.compile(
    r"(?:^|[_:.-])(?:black|blue|brown|color|colour|cyan|gray|green|grey|"
    r"magenta|orange|palette|pixel|purple|red|rgb|white|yellow)"
    r"(?:$|[_:.-])",
    re.IGNORECASE,
)
_RAW_IDENTITY_SEGMENT = re.compile(
    r"(?:^|[_:.-])(?:entity_id|game_id|global|global_id|identity|object_id|"
    r"persistent|persistent_id|seed_id|source_game_id|target_object_id|uuid)"
    r"(?:$|[_:.-])",
    re.IGNORECASE,
)
_FORBIDDEN_KEYS = frozenset(
    {
        "game",
        "game_id",
        "seed",
        "seed_id",
        "persistent_id",
        "entity_id",
        "coordinate",
        "coordinates",
        "absolute_x",
        "absolute_y",
        "color",
        "pixel",
        "grid",
    }
)
_FROZEN_FRAME_IDS = frozenset(frame.frame_id for frame in OBSERVER_FRAME_SPECS)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical payload numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize(item) for item in value), key=_canonical_json)
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    payload = getattr(value, "canonical_payload", None)
    if payload is not None:
        return _canonicalize(payload)
    canonical_hash = getattr(value, "canonical_hash", None)
    if canonical_hash is not None:
        return {"canonical_hash": str(canonical_hash)}
    raise TypeError(
        f"{type(value).__name__} has no deterministic canonical representation"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _assert_transferable(value: Any, *, path: str = "hypothesis") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden transferable field at {path}.{key}")
            _assert_transferable(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            _assert_transferable(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        normalized = value.strip().lower()
        if _GAME_TOKEN.fullmatch(normalized) or _COORDINATE_TOKEN.fullmatch(normalized):
            raise ValueError(f"forbidden local identity at {path}")


def _program_token_is_forbidden(value: str) -> bool:
    normalized = str(value).strip().lower()
    if not normalized:
        return False
    return bool(
        _GAME_TOKEN.fullmatch(normalized)
        or _COORDINATE_TOKEN.fullmatch(normalized)
        or _UUID_TOKEN.fullmatch(normalized)
        or _HEX_COLOR_TOKEN.fullmatch(normalized)
        or _RAW_COLOR_SEGMENT.search(normalized)
        or _RAW_IDENTITY_SEGMENT.search(normalized)
        or normalized.startswith("rgb(")
    )


def _assert_transferable_program(
    value: Any,
    *,
    path: str = "hypothesis.world_program",
) -> None:
    """Reject raw appearance and branch identity in transferable programs."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _FORBIDDEN_KEYS or _program_token_is_forbidden(
                normalized_key
            ):
                raise ValueError(
                    f"forbidden transferable program field at {path}.{key}"
                )
            _assert_transferable_program(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            _assert_transferable_program(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _program_token_is_forbidden(value):
        raise ValueError(f"forbidden transferable program token at {path}")


def _component_payload(value: Any) -> Any:
    payload = getattr(value, "canonical_payload", None)
    if payload is not None:
        return _canonicalize(payload)
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    canonical_hash = getattr(value, "canonical_hash", None)
    if canonical_hash is not None:
        return {"canonical_hash": str(canonical_hash)}
    return _canonicalize(value)


def _component_hash(value: Any) -> str:
    canonical_hash = getattr(value, "canonical_hash", None)
    return str(canonical_hash) if canonical_hash is not None else _sha256(value)


def _component_nodes(value: Any) -> int:
    for name in ("node_count", "ast_node_count"):
        count = getattr(value, name, None)
        if count is not None:
            return max(1, int(count))
    return 1


def _frame_id(value: Any) -> str:
    direct = getattr(value, "frame_id", None)
    if direct:
        return str(direct)
    spec = getattr(value, "spec", None) or getattr(value, "frame", None)
    nested = getattr(spec, "frame_id", None)
    if nested:
        return str(nested)
    if isinstance(value, Mapping):
        return str(value.get("frame_id", ""))
    return ""


def _registered_frame_id(value: Any) -> str:
    frame_id = _frame_id(value).strip().lower()
    try:
        registered = observer_frame_spec(frame_id)
    except ValueError as exc:
        raise ValueError("hypothesis frame is outside the four frozen frames") from exc
    if isinstance(value, ObserverFrameSpec):
        observed_payload = value.canonical_payload
    else:
        observed_payload = _component_payload(value)
    if _canonicalize(observed_payload) != _canonicalize(registered.canonical_payload):
        raise ValueError("hypothesis frame payload drifted from its frozen frame spec")
    return frame_id


def _transport_endpoints(value: Any) -> tuple[str, str]:
    source = (
        str(
            getattr(value, "source_frame", None)
            or getattr(value, "source_frame_id", "")
        )
        .strip()
        .lower()
    )
    target = (
        str(
            getattr(value, "target_frame", None)
            or getattr(value, "target_frame_id", "")
        )
        .strip()
        .lower()
    )
    return source, target


def _validate_native_transport_connectivity(
    native_frame: str,
    transports: Sequence[Any],
) -> None:
    """Require every declared transport to belong to the native frame orbit."""

    if not transports:
        return
    native_frame = str(native_frame).strip().lower()
    adjacency: dict[str, set[str]] = {}
    endpoints: set[str] = set()
    for transport in transports:
        source, target = _transport_endpoints(transport)
        if not source or not target:
            raise ValueError("joint gauge transport requires two frame endpoints")
        if source not in _FROZEN_FRAME_IDS or target not in _FROZEN_FRAME_IDS:
            raise ValueError("joint gauge transport escaped the four frozen frames")
        endpoints.update((source, target))
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    visited = {native_frame}
    frontier = [native_frame]
    while frontier:
        frame = frontier.pop()
        for neighbor in adjacency.get(frame, ()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append(neighbor)
    if not endpoints <= visited:
        raise ValueError("joint gauge transports are not connected to the native frame")


def _program_expression_symbols(
    expression: Any,
    *,
    roles: set[str],
    facts: set[str],
) -> None:
    predicate = str(getattr(expression, "predicate", "")).strip().lower()
    if predicate:
        facts.add(predicate)
    role = str(getattr(expression, "role", "")).strip().lower()
    if role:
        roles.add(role)
    for term in tuple(getattr(expression, "terms", ())):
        normalized = str(term).strip().lower()
        if normalized.startswith("$") and len(normalized) > 1:
            roles.add(normalized[1:])
    for child in tuple(getattr(expression, "args", ())):
        _program_expression_symbols(child, roles=roles, facts=facts)


def _required_certified_source_domain(
    hypothesis: JointGaugeHypothesis,
) -> frozenset[str]:
    """Conservatively extract the D/G/A symbols a native witness must cover."""

    program = hypothesis.world_program
    roles = {
        str(role).strip().lower()
        for role in tuple(getattr(program.object_schema, "roles", ()))
        if str(role).strip()
    }
    facts: set[str] = set()
    actions: set[str] = set()
    for binding in tuple(getattr(program, "action_bindings", ())):
        action_name = str(getattr(binding, "action_name", "")).strip().upper()
        if action_name:
            actions.add(action_name)
        target_role = str(getattr(binding, "target_role", "")).strip().lower()
        if target_role:
            roles.add(target_role)
    for rule in tuple(getattr(program, "transition_rules", ())):
        _program_expression_symbols(
            getattr(rule, "condition", None), roles=roles, facts=facts
        )
        for effect in tuple(getattr(rule, "effects", ())):
            predicate = str(getattr(effect, "predicate", "")).strip().lower()
            if predicate:
                facts.add(predicate)
            for term in tuple(getattr(effect, "terms", ())):
                normalized = str(term).strip().lower()
                if normalized.startswith("$") and len(normalized) > 1:
                    roles.add(normalized[1:])
    for owner in (
        getattr(program, "progress_rule", None),
        *tuple(getattr(program, "terminal_rules", ())),
        getattr(program, "goal_rule", None),
    ):
        _program_expression_symbols(
            getattr(owner, "expression", None), roles=roles, facts=facts
        )

    option = hypothesis.option
    transitions = tuple(getattr(option, "transitions", ()))
    for transition in transitions:
        action_schema = str(getattr(transition, "action_schema", "")).strip().upper()
        if action_schema and action_schema != "*":
            actions.add(action_schema)
        predicate = str(getattr(transition, "predicate", "")).strip().lower()
        if predicate:
            facts.add(predicate)
        relation = str(getattr(transition, "relation", "identity")).strip().lower()
        if relation and relation != "identity":
            facts.add(relation)
    initiation = getattr(option, "initiation_condition", None)
    initiation_role = str(getattr(initiation, "role", "")).strip().lower()
    initiation_predicate = str(getattr(initiation, "predicate", "")).strip().lower()
    if initiation_role:
        roles.add(initiation_role)
    if initiation_predicate:
        facts.add(initiation_predicate)
    declared_action_schemas = tuple(getattr(option, "action_schemas", ()))
    if not declared_action_schemas:
        allowed = getattr(option, "allowed_action_schemas", None)
        initial = getattr(
            option,
            "initial_state",
            getattr(option, "initial_state_id", None),
        )
        if callable(allowed) and initial is not None:
            try:
                declared_action_schemas = tuple(allowed(initial))
            except (KeyError, TypeError, ValueError):
                declared_action_schemas = ()
    for action_schema in declared_action_schemas:
        normalized = str(action_schema).strip().upper()
        if normalized and normalized != "*":
            actions.add(normalized)
    return frozenset(
        {
            *(f"role:{role}" for role in roles),
            *(f"fact:{fact}" for fact in facts),
            *(f"action:{action}" for action in actions),
        }
    )


@dataclass(frozen=True)
class JointGaugeHypothesis:
    """A complete ``(D, G, F, Tau, A)`` T10.2 posterior particle."""

    world_program: JointProgramHypothesis
    frame: Any
    transports: tuple[Any, ...]
    option: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "transports", tuple(self.transports))
        native_frame = _registered_frame_id(self.frame)
        if self.option is None:
            raise ValueError("joint gauge hypothesis requires an option automaton")
        if not callable(getattr(self.option, "allowed_action_schemas", None)):
            raise TypeError("option automaton requires allowed_action_schemas(state)")
        if not callable(getattr(self.option, "observe", None)):
            raise TypeError("option automaton requires observe(state, action, events)")
        if not (
            callable(getattr(self.option, "new_execution", None))
            or getattr(self.option, "initial_state", None) is not None
            or getattr(self.option, "initial_state_id", None) is not None
        ):
            raise ValueError("option automaton requires an explicit initial state")
        _assert_transferable_program(self.world_program.canonical_payload)
        if any(
            hasattr(item, "source_projection_hash")
            or hasattr(item, "target_projection_hash")
            for item in self.transports
        ):
            raise ValueError(
                "event-local transport certificates cannot enter a hypothesis; "
                "use a stable structural orbit witness"
            )
        _validate_native_transport_connectivity(native_frame, self.transports)
        _assert_transferable(self.canonical_payload)

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "world_program": self.world_program.canonical_payload,
            "frame": _component_payload(self.frame),
            "transports": [
                _component_payload(item)
                for item in sorted(self.transports, key=_component_hash)
            ],
            "option": _component_payload(self.option),
        }

    @property
    def canonical_hash(self) -> str:
        return _sha256(self.canonical_payload)

    @property
    def gauge_equivalence_key(self) -> str:
        # Only construction-token protected structural witnesses can merge
        # frame copies.  Duck-typed ``certifies_gauge_equivalence`` flags have
        # no authority, and every witness leaving the native frame must cover
        # the D/G/A symbols actually used by this particle.
        native_frame = _registered_frame_id(self.frame)
        required_domain = _required_certified_source_domain(self)
        certified = tuple(
            item for item in self.transports if isinstance(item, TransportOrbitWitness)
        )
        departing = tuple(
            item for item in certified if item.source_frame_id == native_frame
        )
        fully_certified = bool(
            self.transports
            and len(certified) == len(self.transports)
            and departing
            and all(
                required_domain <= item.certified_source_domain for item in departing
            )
        )
        if fully_certified:
            certified_keys = tuple(
                sorted(item.gauge_equivalence_key for item in certified)
            )
            gauge_witness: Any = {
                "certified_transport_orbit": certified_keys,
            }
        else:
            gauge_witness = {
                "uncertified_frame": _component_hash(self.frame),
                "uncertified_transports": [
                    _component_hash(item)
                    for item in sorted(self.transports, key=_component_hash)
                ],
            }
        return _sha256(
            {
                "world_program": self.world_program.canonical_hash,
                "option": _component_hash(self.option),
                "gauge_witness": gauge_witness,
            }
        )

    @property
    def node_count(self) -> int:
        return (
            self.world_program.node_count
            + 1
            + sum(_component_nodes(item) for item in self.transports)
            + _component_nodes(self.option)
        )

    @property
    def log_prior(self) -> float:
        base = (
            -0.05 * self.world_program.node_count
            - 0.25 * self.world_program.local_constant_count
            - float(self.world_program.edit_distance)
        )
        new_nodes = sum(
            _component_nodes(item) for item in self.transports
        ) + _component_nodes(self.option)
        return base + NEW_AST_NODE_LOG_PRIOR * new_nodes


@dataclass(frozen=True)
class GaugeParticle:
    hypothesis: JointGaugeHypothesis
    log_prior: float
    log_weight: float
    state: AbstractState | None = None
    option_state: Any = None
    trace_signature: tuple[str, ...] = ()
    latest_physical_log_likelihood: float = 0.0
    latest_frame_log_likelihood: float = 0.0
    latest_commutativity_penalty: float = 0.0
    latest_option_log_likelihood: float = 0.0
    observations: int = 0

    @property
    def probability(self) -> float:
        return math.exp(self.log_weight)


@dataclass(frozen=True)
class GaugeClass:
    key: str
    particles: tuple[GaugeParticle, ...]
    log_mass: float

    @property
    def probability(self) -> float:
        return math.exp(self.log_mass)

    @property
    def representative(self) -> GaugeParticle:
        return max(
            self.particles,
            key=lambda particle: (
                particle.log_weight,
                particle.hypothesis.canonical_hash,
            ),
        )


@dataclass(frozen=True)
class GaugeUpdate:
    event_id: str
    physical_scored_particles: int
    projection_score_count: int
    classes_before: int
    classes_after: int
    collapsed: bool


@dataclass(frozen=True)
class OptionSequenceSignatureMass:
    """Posterior mass of one gauge-aggregated option-sequence signature."""

    rank: int
    signature: str
    posterior_mass: float
    compatible: bool


@dataclass(frozen=True)
class OptionSequenceRanking:
    """Strict whole-prefix option ranking used by progressing-event gates.

    ``best_compatible_rank`` is deliberately a rank over complete option
    sequence signatures, not over the primitive action chosen at the latest
    step.  Particles that differ only by their observer-frame gauge copy share
    a signature and therefore contribute to the same posterior mass.
    """

    best_compatible_rank: int | None
    compatible_posterior_mass: float
    explicit_posterior_mass: float
    residual_posterior_mass: float
    compatible_signature_count: int
    signature_count: int
    ranked_signatures: tuple[OptionSequenceSignatureMass, ...]


class GaugeProgramPosterior:
    """Log-space posterior over complete frame/transport/option programs."""

    def __init__(
        self,
        *,
        executor: ProgramExecutor | None = None,
        maximum_classes: int = MAXIMUM_GAUGE_CLASSES,
        channel_weights: Mapping[str, float] | None = None,
        unknown_coverage_penalty: float = 0.75,
        commutativity_penalty: float = 1.0,
    ) -> None:
        self.executor = executor or ProgramExecutor()
        self.maximum_classes = max(1, min(MAXIMUM_GAUGE_CLASSES, int(maximum_classes)))
        self.channel_weights = dict(channel_weights or DEFAULT_CHANNEL_WEIGHTS)
        self.unknown_coverage_penalty = max(0.0, float(unknown_coverage_penalty))
        self.commutativity_penalty = max(0.0, float(commutativity_penalty))
        self._particles: list[GaugeParticle] = []
        self._event_ids: list[str] = []
        self._seen_event_ids: set[str] = set()
        self._branch_index = 0
        self._top_class_key = ""
        self._top_class_streak = 0
        self._collapsed = False
        self._last_update: GaugeUpdate | None = None
        self._residual_log_mass = float("-inf")

    @property
    def particles(self) -> tuple[GaugeParticle, ...]:
        return tuple(self._particles)

    @property
    def classes(self) -> tuple[GaugeClass, ...]:
        grouped: dict[str, list[GaugeParticle]] = {}
        for particle in self._particles:
            grouped.setdefault(particle.hypothesis.gauge_equivalence_key, []).append(
                particle
            )
        return tuple(
            sorted(
                (
                    GaugeClass(
                        key=key,
                        particles=tuple(items),
                        log_mass=_logsumexp(item.log_weight for item in items),
                    )
                    for key, items in grouped.items()
                ),
                key=lambda item: (item.log_mass, item.key),
                reverse=True,
            )
        )

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(self._event_ids)

    @property
    def last_update(self) -> GaugeUpdate | None:
        return self._last_update

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def residual_mass(self) -> float:
        """Probability retained for classes pruned outside the explicit bank."""

        return (
            0.0
            if not math.isfinite(self._residual_log_mass)
            else math.exp(self._residual_log_mass)
        )

    def seed(
        self,
        hypotheses: Sequence[JointGaugeHypothesis],
        *,
        initial_states: Mapping[str, AbstractState] | None = None,
    ) -> None:
        # Canonically identical programs can be reached by several synthesis
        # derivations.  Keep the minimum-MDL derivation deterministically;
        # never let input order choose which prior survives deduplication.
        unique: dict[str, JointGaugeHypothesis] = {}
        for item in hypotheses:
            previous = unique.get(item.canonical_hash)
            if (
                previous is None
                or item.log_prior > previous.log_prior
                or (
                    item.log_prior == previous.log_prior
                    and _canonical_json(item) < _canonical_json(previous)
                )
            ):
                unique[item.canonical_hash] = item
        ordered = tuple(unique[key] for key in sorted(unique))
        states = dict(initial_states or {})
        particles = []
        for hypothesis in ordered:
            # A certified gauge copy is another derivation carrying prior
            # mass.  Class marginalization performs the only multiplicity
            # aggregation; cancelling it here would silently erase valid
            # evidence for the equivalence class.
            prior = hypothesis.log_prior
            initial_state = states.get(_frame_id(hypothesis.frame))
            particles.append(
                GaugeParticle(
                    hypothesis=hypothesis,
                    log_prior=prior,
                    log_weight=prior,
                    state=initial_state,
                    option_state=_option_initial_state(
                        hypothesis.option,
                        state=initial_state,
                    ),
                )
            )
        self._residual_log_mass = float("-inf")
        self._particles, omitted = self._prune_classes(self._deduplicate(particles))
        self._residual_log_mass = omitted
        self._normalize()
        self._event_ids.clear()
        self._seen_event_ids.clear()
        self._top_class_key = ""
        self._top_class_streak = 0
        self._collapsed = False
        self._last_update = None

    def observe(self, bundle: Any) -> GaugeUpdate:
        event_id = str(getattr(bundle, "event_id", "")).strip()
        if not event_id:
            raise ValueError("physical event bundle requires event_id")
        if event_id in self._seen_event_ids:
            raise ValueError(f"duplicate physical event_id: {event_id}")
        if bool(getattr(bundle, "reset", False)):
            self.start_branch()
            self._seen_event_ids.add(event_id)
            self._event_ids.append(event_id)
            update = GaugeUpdate(
                event_id, 0, 0, len(self.classes), len(self.classes), False
            )
            self._last_update = update
            return update

        classes_before = len(self.classes)
        common_observation = _bundle_common_observation(bundle)
        action = _bundle_action(bundle)
        projections = _bundle_projections(bundle)
        updated: list[GaugeParticle] = []
        physical_scored = 0
        projection_scores = 0
        for particle in self._particles:
            option_schema_allows = _option_allows(
                particle.hypothesis.option,
                particle.option_state,
                action,
            )
            native_frame = _frame_id(particle.hypothesis.frame)
            native = projections.get(native_frame)
            public_before = _projection_state(native, before=True)
            relation_state = (
                public_before if public_before is not None else particle.state
            )
            option_relation_allows = _option_relation_satisfied(
                particle.hypothesis.option,
                particle.option_state,
                action,
                relation_state,
            )
            option_log_likelihood = (
                0.0
                if option_schema_allows and option_relation_allows
                else OPTION_ACTION_MISMATCH_LOG_LIKELIHOOD
            )
            if public_before is None:
                public_before = particle.state
            if public_before is None:
                physical = -self.unknown_coverage_penalty * sum(
                    self.channel_weights[channel]
                    for channel in ("progress", "terminal", "goal")
                    if channel in common_observation.known_channels
                )
                option_state = _option_observe(
                    particle.hypothesis.option,
                    particle.option_state,
                    action,
                    tuple(getattr(bundle, "events", ())),
                )
                updated.append(
                    replace(
                        particle,
                        log_weight=(
                            particle.log_weight + physical + option_log_likelihood
                        ),
                        latest_physical_log_likelihood=physical,
                        latest_frame_log_likelihood=0.0,
                        latest_commutativity_penalty=0.0,
                        latest_option_log_likelihood=option_log_likelihood,
                        option_state=option_state,
                        observations=particle.observations + 1,
                        trace_signature=particle.trace_signature
                        + (_sha256((event_id, "missing_native_projection")),),
                    )
                )
                physical_scored += 1
                continue

            start = (
                public_before
                if particle.state is None
                else particle.state.merge_observation(public_before)
            )
            prediction = self.executor.step(
                particle.hypothesis.world_program,
                start,
                action,
            )
            physical = packet_log_likelihood(
                _packet_channels(prediction, {"progress", "terminal", "goal"}),
                _packet_channels(
                    common_observation,
                    {"progress", "terminal", "goal"},
                ),
                channel_weights=self.channel_weights,
                unknown_coverage_penalty=self.unknown_coverage_penalty,
            )
            physical_scored += 1

            frame_scores: list[float] = []
            commute_scores: list[float] = []
            for target_frame, projection in projections.items():
                observed = _projection_observation(projection) or common_observation
                requested = (
                    {"objects", "relations", "topology"}
                    & set(observed.known_channels)
                    & _projection_channels(projection)
                )
                if target_frame == native_frame:
                    transported: PredictionPacket | None = prediction
                    transport = None
                else:
                    transport = _find_transport(
                        particle.hypothesis.transports,
                        native_frame,
                        target_frame,
                    )
                    transported = _transport_prediction(prediction, transport)
                if requested:
                    scored_prediction = transported or PredictionPacket()
                    frame_scores.append(
                        packet_log_likelihood(
                            _packet_channels(scored_prediction, requested),
                            _packet_channels(observed, requested),
                            channel_weights=self.channel_weights,
                            unknown_coverage_penalty=self.unknown_coverage_penalty,
                        )
                    )
                    projection_scores += 1
                if target_frame != native_frame:
                    target_state = _projection_state(projection, before=True)
                    if not requested:
                        continue
                    target_action = _transport_action(action, transport)
                    if (
                        transported is None
                        or target_state is None
                        or target_action is None
                    ):
                        commute_scores.append(1.0)
                        continue
                    direct = self.executor.step(
                        particle.hypothesis.world_program,
                        target_state,
                        target_action,
                    )
                    jointly_known = set(transported.known_channels) & set(
                        direct.known_channels
                    )
                    commute_scores.append(
                        1.0
                        if not requested <= jointly_known
                        else _prediction_distance(transported, direct, requested)
                    )

            frame_average = (
                sum(frame_scores) / len(frame_scores) if frame_scores else 0.0
            )
            commute_average = (
                sum(commute_scores) / len(commute_scores) if commute_scores else 0.0
            )
            total = (
                physical
                + frame_average
                + option_log_likelihood
                - (self.commutativity_penalty * commute_average)
            )
            observed_after = _projection_state(native, before=False)
            predicted_after = prediction.state_after or start
            next_state = (
                predicted_after
                if observed_after is None
                else predicted_after.merge_observation(observed_after)
            )
            option_state = _option_observe(
                particle.hypothesis.option,
                particle.option_state,
                action,
                tuple(getattr(bundle, "events", ())),
            )
            trace_item = _sha256(
                {
                    "event_id": event_id,
                    "prediction": prediction.full_signature,
                    "option_state": _stateful_signature(
                        option_state,
                        particle.hypothesis.option,
                    ),
                }
            )
            updated.append(
                replace(
                    particle,
                    log_weight=particle.log_weight + total,
                    state=next_state,
                    option_state=option_state,
                    trace_signature=particle.trace_signature + (trace_item,),
                    latest_physical_log_likelihood=physical,
                    latest_frame_log_likelihood=frame_average,
                    latest_commutativity_penalty=commute_average,
                    latest_option_log_likelihood=option_log_likelihood,
                    observations=particle.observations + 1,
                )
            )

        previous_residual_mass = self.residual_mass
        self._seen_event_ids.add(event_id)
        self._event_ids.append(event_id)
        deduplicated = self._deduplicate(updated)
        explicit_log_mass = _logsumexp(particle.log_weight for particle in deduplicated)
        explicit_budget = max(0.0, 1.0 - previous_residual_mass)
        if deduplicated and not math.isfinite(explicit_log_mass):
            raise ValueError("explicit posterior lost all finite likelihood mass")
        if explicit_budget > 0.0 and deduplicated:
            budget_log = math.log(explicit_budget)
            deduplicated = [
                replace(
                    particle,
                    log_weight=particle.log_weight - explicit_log_mass + budget_log,
                )
                for particle in deduplicated
            ]
        self._particles, omitted = self._prune_classes(deduplicated)
        previous_residual_log_mass = (
            float("-inf")
            if previous_residual_mass <= 0.0
            else math.log(previous_residual_mass)
        )
        self._residual_log_mass = _logsumexp((previous_residual_log_mass, omitted))
        self._normalize()
        collapsed = self._maybe_collapse()
        update = GaugeUpdate(
            event_id=event_id,
            physical_scored_particles=physical_scored,
            projection_score_count=projection_scores,
            classes_before=classes_before,
            classes_after=len(self.classes),
            collapsed=collapsed,
        )
        self._last_update = update
        return update

    def start_branch(self, *, regime_index: int | None = None) -> None:
        self._branch_index += 1
        refreshed = []
        for particle in self._particles:
            # A reset opens a new physical branch.  No entity, fact, counter,
            # register, or topology from the preceding reset is transferable.
            state = (
                None
                if regime_index is None
                else AbstractState(regime_index=int(regime_index))
            )
            refreshed.append(
                replace(
                    particle,
                    state=state,
                    option_state=_option_initial_state(
                        particle.hypothesis.option,
                        state=state,
                    ),
                )
            )
        self._particles = refreshed
        self._top_class_key = ""
        self._top_class_streak = 0
        self._collapsed = False

    @property
    def entropy(self) -> float:
        explicit = -sum(
            item.probability * math.log(max(item.probability, 1e-300))
            for item in self.classes
            if item.probability > 0.0
        )
        residual = self.residual_mass
        return explicit - (
            residual * math.log(max(residual, 1e-300)) if residual > 0.0 else 0.0
        )

    @property
    def normalized_entropy(self) -> float:
        class_count = len(self.classes) + int(self.residual_mass > 0.0)
        if class_count <= 1:
            return 0.0
        return self.entropy / math.log(class_count)

    def top(self, maximum: int = MAXIMUM_DECISION_CLASSES) -> tuple[GaugeClass, ...]:
        return self.classes[: max(0, min(MAXIMUM_DECISION_CLASSES, int(maximum)))]

    def snapshot(self, *, maximum_classes: int = 8) -> Mapping[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "particles": len(self._particles),
            "classes": len(self.classes) + int(self.residual_mass > 0.0),
            "retained_classes": len(self.classes),
            "residual_mass": round(self.residual_mass, 10),
            "events": len(self._event_ids),
            "branch_index": self._branch_index,
            "normalized_entropy": round(self.normalized_entropy, 8),
            "map_streak": self._top_class_streak,
            "collapsed": self._collapsed,
            "top": [
                {
                    "class_key": item.key,
                    "probability": round(item.probability, 10),
                    "variants": len(item.particles),
                    "representative": item.representative.hypothesis.canonical_hash,
                }
                for item in self.top(maximum_classes)
            ],
        }

    def _deduplicate(
        self,
        particles: Sequence[GaugeParticle],
    ) -> list[GaugeParticle]:
        unique: dict[tuple[str, str, str, tuple[str, ...]], GaugeParticle] = {}
        for particle in particles:
            key = (
                particle.hypothesis.canonical_hash,
                "" if particle.state is None else particle.state.signature,
                _stateful_signature(
                    particle.option_state,
                    particle.hypothesis.option,
                ),
                particle.trace_signature,
            )
            previous = unique.get(key)
            if previous is None:
                unique[key] = particle
                continue
            representative = (
                particle if particle.log_weight > previous.log_weight else previous
            )
            unique[key] = replace(
                representative,
                log_prior=_logsumexp((previous.log_prior, particle.log_prior)),
                log_weight=_logsumexp((previous.log_weight, particle.log_weight)),
            )
        return list(unique.values())

    def _prune_classes(
        self,
        particles: Sequence[GaugeParticle],
    ) -> tuple[list[GaugeParticle], float]:
        grouped: dict[str, list[GaugeParticle]] = {}
        for particle in particles:
            grouped.setdefault(particle.hypothesis.gauge_equivalence_key, []).append(
                particle
            )
        ranked = sorted(
            grouped.items(),
            key=lambda item: (_logsumexp(p.log_weight for p in item[1]), item[0]),
            reverse=True,
        )
        retained = ranked[: self.maximum_classes]
        omitted = ranked[self.maximum_classes :]
        omitted_log_mass = _logsumexp(
            _logsumexp(particle.log_weight for particle in items)
            for _, items in omitted
        )
        return (
            [particle for _, items in retained for particle in items],
            omitted_log_mass,
        )

    def _normalize(self) -> None:
        masses = [item.log_weight for item in self._particles]
        if math.isfinite(self._residual_log_mass):
            masses.append(self._residual_log_mass)
        if not masses:
            return
        normalizer = _logsumexp(masses)
        self._particles = [
            replace(item, log_weight=item.log_weight - normalizer)
            for item in self._particles
        ]
        if math.isfinite(self._residual_log_mass):
            self._residual_log_mass -= normalizer

    def _maybe_collapse(self) -> bool:
        classes = self.classes
        if not classes:
            return False
        top = classes[0]
        second_mass = max(
            classes[1].probability if len(classes) > 1 else 0.0,
            self.residual_mass,
        )
        bayes_factor = (
            float("inf") if second_mass <= 0.0 else top.probability / second_mass
        )
        eligible = (
            top.probability >= MAP_MASS_THRESHOLD
            and bayes_factor >= MAP_BAYES_FACTOR_THRESHOLD
        )
        if eligible and top.key == self._top_class_key:
            self._top_class_streak += 1
        elif eligible:
            self._top_class_key = top.key
            self._top_class_streak = 1
        else:
            self._top_class_key = ""
            self._top_class_streak = 0
        if self._top_class_streak < MAP_STABILITY_TRANSITIONS:
            return False
        self._particles = [
            particle
            for particle in self._particles
            if particle.hypothesis.gauge_equivalence_key == top.key
        ]
        self._residual_log_mass = float("-inf")
        self._normalize()
        self._collapsed = True
        return True


@dataclass(frozen=True)
class GaugeActionAssessment:
    action: ActionCandidate
    utility: float
    expected_goal: float
    expected_progress: float
    terminal_risk: float
    information_gain: float
    commutativity_penalty: float
    beta: float
    evaluated_mass: float
    residual_mass: float
    veto: str = ""


@dataclass(frozen=True)
class _GaugeCounterfactualRollout:
    actions: tuple[ActionCandidate, ...]
    packets: tuple[PredictionPacket, ...]
    option_state: Any
    option_state_signature: str
    commutativity_penalty: float = 0.0

    @property
    def goal(self) -> float:
        known = [
            float(packet.goal_probability)
            for packet in self.packets
            if packet.goal_probability is not None
        ]
        return known[-1] if known else 0.0

    @property
    def progress(self) -> float:
        return sum(
            float(packet.progress_mean)
            for packet in self.packets
            if packet.progress_mean is not None
        )

    @property
    def terminal_risk(self) -> float:
        probabilities = [
            max(0.0, min(1.0, float(packet.terminal_probability)))
            for packet in self.packets
            if packet.terminal_probability is not None
        ]
        return 1.0 - math.prod(1.0 - value for value in probabilities)

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            tuple(action.action_name for action in self.actions),
            tuple(packet.full_signature for packet in self.packets),
            self.option_state_signature,
        )


@dataclass(frozen=True)
class GaugeDecision:
    action: ActionCandidate | None
    chosen: GaugeActionAssessment | None
    assessments: tuple[GaugeActionAssessment, ...]
    normalized_entropy: float
    reason: str


class GaugeDecisionEngine:
    """Bounded one-action decision over at most 64 marginalized classes."""

    def __init__(
        self,
        *,
        executor: ProgramExecutor | None = None,
        maximum_classes: int = MAXIMUM_DECISION_CLASSES,
        maximum_option_horizon: int = 16,
    ) -> None:
        self.executor = executor or ProgramExecutor()
        self.maximum_classes = max(
            1,
            min(MAXIMUM_DECISION_CLASSES, int(maximum_classes)),
        )
        self.maximum_option_horizon = max(1, min(16, int(maximum_option_horizon)))

    def decide(
        self,
        posterior: GaugeProgramPosterior,
        frame_states: Mapping[str, AbstractState],
        legal_actions: Sequence[ActionCandidate],
        *,
        danger_veto: Callable[[ActionCandidate], bool] | None = None,
        fallback_action: ActionCandidate | None = None,
    ) -> GaugeDecision:
        actions = tuple({item.key: item for item in legal_actions}.values())
        if not actions:
            return GaugeDecision(
                None, None, (), posterior.normalized_entropy, "no_legal_action"
            )
        classes = posterior.top(self.maximum_classes)
        if not classes:
            return GaugeDecision(
                None, None, (), posterior.normalized_entropy, "empty_posterior"
            )
        beta = 1.0 if posterior.normalized_entropy > 0.5 else 0.25
        assessments = tuple(
            self._assess(
                action,
                classes=classes,
                frame_states=frame_states,
                legal_actions=actions,
                beta=beta,
                danger_veto=danger_veto,
            )
            for action in actions
        )
        admissible = tuple(
            item for item in assessments if not item.veto and item.evaluated_mass > 0.0
        )
        if not admissible:
            safe_fallback = (
                fallback_action
                if fallback_action is not None
                and fallback_action.key in {item.key for item in actions}
                and (danger_veto is None or not danger_veto(fallback_action))
                and _fallback_supported_by_option(
                    fallback_action,
                    classes=classes,
                    frame_states=frame_states,
                )
                else None
            )
            return GaugeDecision(
                safe_fallback,
                None,
                assessments,
                posterior.normalized_entropy,
                "incomplete_projection_fallback" if safe_fallback else "all_vetoed",
            )
        chosen = max(
            admissible,
            key=lambda item: (
                item.utility,
                -item.terminal_risk,
                item.action.key,
            ),
        )
        return GaugeDecision(
            chosen.action,
            chosen,
            tuple(
                sorted(assessments, key=lambda item: (bool(item.veto), -item.utility))
            ),
            posterior.normalized_entropy,
            "selected",
        )

    def _assess(
        self,
        action: ActionCandidate,
        *,
        classes: Sequence[GaugeClass],
        frame_states: Mapping[str, AbstractState],
        legal_actions: Sequence[ActionCandidate],
        beta: float,
        danger_veto: Callable[[ActionCandidate], bool] | None,
    ) -> GaugeActionAssessment:
        top_mass = sum(item.probability for item in classes)
        if danger_veto is not None and danger_veto(action):
            return GaugeActionAssessment(
                action,
                float("-inf"),
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                beta,
                0.0,
                max(0.0, 1.0 - top_mass),
                "observed_danger",
            )

        expected_goal = 0.0
        expected_progress = 0.0
        terminal_risk = 0.0
        commute = 0.0
        expected_length = 0.0
        evaluated_mass = 0.0
        class_signatures: list[tuple[float, Any]] = []
        for gauge_class in classes:
            variant_predictions: list[tuple[float, _GaugeCounterfactualRollout]] = []
            for particle in gauge_class.particles:
                if not _option_allows(
                    particle.hypothesis.option,
                    particle.option_state,
                    action,
                ):
                    continue
                frame = _frame_id(particle.hypothesis.frame)
                observed_state = frame_states.get(frame)
                if observed_state is None:
                    continue
                rollout = self._rollout_particle(
                    particle,
                    action,
                    observed_state=observed_state,
                    frame_states=frame_states,
                    legal_actions=legal_actions,
                    danger_veto=danger_veto,
                )
                if not rollout.packets:
                    continue
                variant_predictions.append((particle.probability, rollout))
                commute += particle.probability * rollout.commutativity_penalty
            class_mass = sum(weight for weight, _ in variant_predictions)
            if class_mass <= 0.0:
                continue
            evaluated_mass += class_mass
            class_signature = tuple(
                sorted(
                    (
                        round(weight / class_mass, 8),
                        rollout.signature,
                    )
                    for weight, rollout in variant_predictions
                )
            )
            class_signatures.append((class_mass, class_signature))
            for weight, rollout in variant_predictions:
                expected_goal += weight * rollout.goal
                expected_progress += weight * rollout.progress
                terminal_risk += weight * rollout.terminal_risk
                expected_length += weight * len(rollout.actions)

        residual_mass = max(0.0, 1.0 - evaluated_mass)
        # Use one action-independent denominator, as in the frozen T7 engine.
        # Otherwise an action supported by a tiny posterior tail is
        # spuriously promoted by conditioning on its own support.
        considered_mass = sum(item.probability for item in classes)
        if considered_mass > 0.0:
            expected_goal /= considered_mass
            expected_progress /= considered_mass
            terminal_risk /= considered_mass
            commute /= considered_mass
            expected_length /= considered_mass
        information_gain = _entropy_with_residual(class_signatures, residual_mass)
        utility = (
            2.0 * expected_goal
            + expected_progress
            + beta * information_gain
            - 5.0 * terminal_risk
            - commute
            - 0.02 * expected_length
        )
        return GaugeActionAssessment(
            action=action,
            utility=utility,
            expected_goal=expected_goal,
            expected_progress=expected_progress,
            terminal_risk=terminal_risk,
            information_gain=information_gain,
            commutativity_penalty=commute,
            beta=beta,
            evaluated_mass=evaluated_mass,
            residual_mass=residual_mass,
        )

    def _rollout_particle(
        self,
        particle: GaugeParticle,
        first_action: ActionCandidate,
        *,
        observed_state: AbstractState,
        frame_states: Mapping[str, AbstractState],
        legal_actions: Sequence[ActionCandidate],
        danger_veto: Callable[[ActionCandidate], bool] | None,
    ) -> _GaugeCounterfactualRollout:
        option = particle.hypothesis.option
        option_state = particle.option_state
        native_frame = _frame_id(particle.hypothesis.frame)
        native_state = (
            observed_state
            if particle.state is None
            else particle.state.merge_observation(observed_state)
        )
        target_states = dict(frame_states)
        actions_by_name: dict[str, list[ActionCandidate]] = {}
        for candidate in legal_actions:
            actions_by_name.setdefault(candidate.action_name, []).append(candidate)
        for candidates in actions_by_name.values():
            candidates.sort(key=lambda item: item.key)

        action = first_action
        actions: list[ActionCandidate] = []
        packets: list[PredictionPacket] = []
        commute_scores: list[float] = []
        option_horizon = max(
            1,
            min(
                self.maximum_option_horizon,
                int(getattr(option, "maximum_horizon", self.maximum_option_horizon)),
            ),
        )
        for step in range(option_horizon):
            if step and danger_veto is not None and danger_veto(action):
                break
            if not _option_allows(option, option_state, action) or not (
                _option_relation_satisfied(
                    option,
                    option_state,
                    action,
                    native_state,
                )
            ):
                break
            packet = self.executor.step(
                particle.hypothesis.world_program,
                native_state,
                action,
            )
            actions.append(action)
            packets.append(packet)

            for target_frame, target_state in tuple(target_states.items()):
                if target_frame == native_frame:
                    continue
                transport = _find_transport(
                    particle.hypothesis.transports,
                    native_frame,
                    target_frame,
                )
                transported = _transport_prediction(packet, transport)
                target_action = _transport_action(action, transport)
                if transported is None or target_action is None:
                    commute_scores.append(1.0)
                    continue
                direct = self.executor.step(
                    particle.hypothesis.world_program,
                    target_state,
                    target_action,
                )
                comparable = (
                    {
                        "objects",
                        "relations",
                        "topology",
                    }
                    & set(transported.known_channels)
                    & set(direct.known_channels)
                )
                if comparable:
                    commute_scores.append(
                        _prediction_distance(transported, direct, comparable)
                    )
                else:
                    commute_scores.append(1.0)
                target_states[target_frame] = direct.state_after or target_state

            native_state = packet.state_after or native_state
            option_state = _option_observe(
                option,
                option_state,
                action,
                _prediction_events(packet),
            )
            action = _next_option_action(
                option,
                option_state,
                actions_by_name,
                previous=action,
            )
            if action is None:
                break
        return _GaugeCounterfactualRollout(
            actions=tuple(actions),
            packets=tuple(packets),
            option_state=option_state,
            option_state_signature=_stateful_signature(option_state, option),
            commutativity_penalty=(
                sum(commute_scores) / len(commute_scores) if commute_scores else 0.0
            ),
        )


def _bundle_action(bundle: Any) -> ActionCandidate:
    action = getattr(bundle, "action", None)
    if not isinstance(action, ActionCandidate):
        raise TypeError("physical event bundle requires one ActionCandidate")
    return action


def _bundle_common_observation(bundle: Any) -> PredictionPacket:
    for name in ("common_observation", "common_outcome", "outcome"):
        value = getattr(bundle, name, None)
        if isinstance(value, PredictionPacket):
            return value
    raise ValueError("physical event bundle requires one common outcome packet")


def _bundle_projections(bundle: Any) -> dict[str, Any]:
    raw = getattr(bundle, "projections", ())
    items = raw.values() if isinstance(raw, Mapping) else raw
    output = {}
    for projection in items:
        frame = _frame_id(projection)
        if not frame:
            raise ValueError("frame projection lacks frame_id")
        if frame in output:
            raise ValueError(f"duplicate projection frame: {frame}")
        output[frame] = projection
    return output


def _projection_state(projection: Any, *, before: bool) -> AbstractState | None:
    if projection is None:
        return None
    names = ("state_before", "before") if before else ("state_after", "after")
    for name in names:
        value = getattr(projection, name, None)
        if isinstance(value, AbstractState):
            return value
    transition = getattr(projection, "transition", None)
    if transition is not None:
        value = getattr(transition, "state_before" if before else "state_after", None)
        if isinstance(value, AbstractState):
            return value
    framed = getattr(projection, "before" if before else "after", None)
    value = getattr(framed, "state", None)
    if isinstance(value, AbstractState):
        return value
    value = getattr(projection, "state", None)
    stage = str(getattr(projection, "stage", ""))
    if isinstance(value, AbstractState) and stage == ("before" if before else "after"):
        return value
    return None


def _projection_observation(projection: Any) -> PredictionPacket | None:
    if projection is None:
        return None
    value = getattr(projection, "observation", None)
    if isinstance(value, PredictionPacket):
        return value
    transition = getattr(projection, "transition", None)
    value = getattr(transition, "observation", None)
    return value if isinstance(value, PredictionPacket) else None


def _projection_action(projection: Any) -> ActionCandidate | None:
    value = getattr(projection, "action", None)
    if isinstance(value, ActionCandidate):
        return value
    transition = getattr(projection, "transition", None)
    value = getattr(transition, "action", None)
    return value if isinstance(value, ActionCandidate) else None


def _projection_channels(projection: Any) -> set[str]:
    raw = getattr(projection, "covered_channels", None)
    if raw is None:
        return {"objects", "relations", "topology"}
    declared = {str(item) for item in raw}
    channels = declared & {"objects", "relations", "topology"}
    if declared & {"entities", "facts"}:
        channels.update({"objects", "relations"})
    if "topology" in declared:
        channels.add("topology")
    return channels


def _packet_channels(packet: PredictionPacket, channels: set[str]) -> PredictionPacket:
    known = frozenset(set(packet.known_channels) & channels)
    return PredictionPacket(
        object_deltas=(packet.object_deltas if "objects" in known else {}),
        relation_deltas=(packet.relation_deltas if "relations" in known else {}),
        topology_deltas=(packet.topology_deltas if "topology" in known else {}),
        progress_mean=(packet.progress_mean if "progress" in known else None),
        progress_distribution=(
            packet.progress_distribution if "progress" in known else {}
        ),
        terminal_probability=(
            packet.terminal_probability if "terminal" in known else None
        ),
        goal_probability=(packet.goal_probability if "goal" in known else None),
        known_channels=known,
        residual=packet.residual,
        state_after=packet.state_after,
    )


def _find_transport(
    transports: Sequence[Any],
    source_frame: str,
    target_frame: str,
) -> TransportMap | None:
    if source_frame == target_frame:
        return None
    # Do not let a direct duck-typed object pre-empt the official resolver.
    # Exact types are intentional: TransportOrbitWitness construction is
    # token-protected, while a subclass could override transport methods after
    # inheriting otherwise valid endpoint fields.
    official = tuple(
        item
        for item in transports
        if type(item) in {TransportMap, TransportOrbitWitness}
    )
    if not official:
        return None
    try:
        resolved = resolve_official_transport(
            official,
            source_frame,
            target_frame,
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if type(resolved) is not TransportMap:
        return None
    if (
        resolved.source_frame_id != str(source_frame).strip().lower()
        or resolved.target_frame_id != str(target_frame).strip().lower()
    ):
        return None
    return resolved


def _transport_prediction(
    packet: PredictionPacket,
    transport: Any | None,
) -> PredictionPacket | None:
    if transport is None:
        return None
    for name in ("transport_prediction", "apply_prediction", "apply_packet"):
        method = getattr(transport, name, None)
        if callable(method):
            value = method(packet)
            return value if isinstance(value, PredictionPacket) else None
    try:
        from .frame_transport_v10_2 import transport_prediction

        value = transport_prediction(packet, transport)
        return value if isinstance(value, PredictionPacket) else None
    except (AttributeError, ImportError, TypeError, ValueError):
        return None


def _transport_action(
    action: ActionCandidate,
    transport: Any | None,
) -> ActionCandidate | None:
    if transport is None:
        return None
    for name in ("transport_action", "apply_action"):
        method = getattr(transport, name, None)
        if callable(method):
            value = method(action)
            return value if isinstance(value, ActionCandidate) else None
    try:
        from .frame_transport_v10_2 import transport_action

        value = transport_action(action, transport)
        return value if isinstance(value, ActionCandidate) else None
    except (AttributeError, ImportError, TypeError, ValueError):
        return None


def _prediction_distance(
    left: PredictionPacket,
    right: PredictionPacket,
    channels: set[str],
) -> float:
    comparable = channels & set(left.known_channels) & set(right.known_channels)
    if not comparable:
        return 0.0
    mismatches = sum(
        left.channel_signature(channel) != right.channel_signature(channel)
        for channel in comparable
    )
    return mismatches / len(comparable)


def rank_option_sequence_signatures(
    posterior: Any,
    prefix: Sequence[
        tuple[ActionCandidate, Sequence[str]]
        | tuple[
            ActionCandidate,
            Sequence[str],
            Mapping[str, AbstractState],
        ]
    ],
) -> OptionSequenceRanking:
    """Rank strict option explanations of one executed reset prefix.

    Every particle's option is replayed from its frozen initial cursor across
    the *whole* action/event prefix.  Posterior masses are then aggregated by
    the resulting option-sequence signature, so observer-frame gauge copies do
    not receive multiple ranks.  Incompatible signatures remain in the ranked
    denominator: a high-mass one-step explanation can therefore outrank the
    lower-mass option sequence that actually explains a progressing prefix.
    """

    normalized_steps: list[
        tuple[
            ActionCandidate,
            tuple[str, ...],
            Mapping[str, AbstractState],
        ]
    ] = []
    for raw_step in prefix:
        if len(raw_step) == 2:
            action, events = raw_step
            frame_states: Mapping[str, AbstractState] = {}
        elif len(raw_step) == 3:
            action, events, raw_frame_states = raw_step
            if not isinstance(raw_frame_states, Mapping):
                raise TypeError("option prefix frame states must be a mapping")
            frame_states = {
                str(frame).strip().lower(): state
                for frame, state in raw_frame_states.items()
                if isinstance(state, AbstractState)
            }
        else:
            raise ValueError("option prefix steps must contain two or three items")
        normalized_action = (
            action
            if isinstance(action, ActionCandidate)
            else ActionCandidate(str(getattr(action, "action_name", action)))
        )
        normalized_events = tuple(
            sorted(
                {str(event).strip().lower() for event in events if str(event).strip()}
            )
        )
        normalized_steps.append((normalized_action, normalized_events, frame_states))
    normalized_prefix = tuple(normalized_steps)
    masses: dict[str, float] = {}
    compatibility: dict[str, bool] = {}
    for particle in getattr(posterior, "particles", ()):
        try:
            mass = float(particle.probability)
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(mass) or mass <= 0.0:
            continue
        hypothesis = getattr(particle, "hypothesis", None)
        option = getattr(hypothesis, "option", None)
        if option is None:
            option = getattr(particle, "option", None)
        if option is None:
            continue
        native_frame = (
            _frame_id(getattr(hypothesis, "frame", None))
            if hypothesis is not None
            else ""
        )
        signature, compatible = _strict_option_sequence_signature(
            option,
            normalized_prefix,
            native_frame=native_frame,
        )
        masses[signature] = masses.get(signature, 0.0) + mass
        compatibility[signature] = compatibility.get(signature, True) and compatible

    ranked_items = sorted(masses.items(), key=lambda item: (-item[1], item[0]))
    ranked = tuple(
        OptionSequenceSignatureMass(
            rank=index,
            signature=signature,
            posterior_mass=mass,
            compatible=compatibility[signature],
        )
        for index, (signature, mass) in enumerate(ranked_items, start=1)
    )
    compatible_items = tuple(item for item in ranked if item.compatible)
    residual = getattr(posterior, "residual_mass", 0.0)
    try:
        residual_mass = float(residual)
    except (TypeError, ValueError):
        residual_mass = 0.0
    if not math.isfinite(residual_mass) or residual_mass < 0.0:
        residual_mass = 0.0
    return OptionSequenceRanking(
        best_compatible_rank=(
            min(item.rank for item in compatible_items) if compatible_items else None
        ),
        compatible_posterior_mass=sum(item.posterior_mass for item in compatible_items),
        explicit_posterior_mass=sum(item.posterior_mass for item in ranked),
        residual_posterior_mass=residual_mass,
        compatible_signature_count=len(compatible_items),
        signature_count=len(ranked),
        ranked_signatures=ranked,
    )


def _strict_option_sequence_signature(
    option: Any,
    prefix: Sequence[
        tuple[
            ActionCandidate,
            Sequence[str],
            Mapping[str, AbstractState],
        ]
    ],
    *,
    native_frame: str,
) -> tuple[str, bool]:
    initial_abstract_state = prefix[0][2].get(native_frame) if prefix else None
    trace: list[dict[str, Any]] = []
    compatible = True
    mismatch_index: int | None = None
    mismatch_reason = ""
    try:
        state = _option_initial_state(
            option,
            state=initial_abstract_state,
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        state = None
        compatible = False
        mismatch_index = 0 if prefix else None
        mismatch_reason = "initiation"
    for index, (action, events, frame_states) in enumerate(prefix):
        if not compatible:
            break
        abstract_state = frame_states.get(native_frame)
        if not _option_can_initiate(option, abstract_state):
            compatible = False
            mismatch_index = index
            mismatch_reason = "initiation"
            break
        if not _option_relation_satisfied(
            option,
            state,
            action,
            abstract_state,
        ):
            compatible = False
            mismatch_index = index
            mismatch_reason = "relation"
            break
        prepared = _option_prepare_observation(option, state, action)
        if prepared is None:
            compatible = False
            mismatch_index = index
            mismatch_reason = "action_schema"
            break
        pending_state, schema = prepared
        method = getattr(option, "observe", None)
        if callable(method):
            observed = _strict_option_observe(
                method,
                pending_state,
                schema,
                action,
                tuple(events),
            )
            if observed is None:
                compatible = False
                mismatch_index = index
                mismatch_reason = "observation"
                break
            state = observed
        else:
            state = pending_state
        trace.append(
            {
                "action_schema": schema,
                "events": tuple(events),
                "state": _stateful_signature(state, option),
            }
        )
    signature = _sha256(
        {
            "option": _component_hash(option),
            "prefix_length": len(prefix),
            "matched_length": len(trace),
            "mismatch_index": mismatch_index,
            "mismatch_reason": mismatch_reason,
            "trace": trace,
        }
    )
    return signature, compatible


def _strict_option_observe(
    method: Callable[..., Any],
    state: Any,
    schema: str,
    action: ActionCandidate,
    events: tuple[str, ...],
) -> Any | None:
    for call in (
        lambda: method(state, schema, events=events),
        lambda: method(state, action, events=events),
        lambda: method(state, schema, events),
        lambda: method(state, action, events),
        lambda: method(state=state, action=schema, events=events),
        lambda: method(state=state, action=action, events=events),
    ):
        try:
            return call()
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
    return None


def _option_initial_state(
    option: Any,
    *,
    state: AbstractState | None = None,
) -> Any:
    new_execution = getattr(option, "new_execution", None)
    if callable(new_execution):
        try:
            return new_execution(state)
        except TypeError:
            try:
                return new_execution(state=state)
            except TypeError:
                if state is None:
                    return new_execution()
                raise
    for name in ("initial_state", "initial_state_id"):
        value = getattr(option, name, None)
        if value is not None:
            return value
    return 0


def _option_can_initiate(
    option: Any,
    state: AbstractState | None,
) -> bool:
    method = getattr(option, "can_initiate", None)
    if callable(method):
        try:
            return bool(method(state))
        except TypeError:
            if state is not None:
                return False
            try:
                return bool(method())
            except (KeyError, RuntimeError, TypeError, ValueError):
                return False
        except (KeyError, RuntimeError, ValueError):
            return False
    condition = getattr(option, "initiation_condition", None)
    satisfied_by = getattr(condition, "satisfied_by", None)
    if callable(satisfied_by):
        try:
            return bool(satisfied_by(state))
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False
    # Compatibility with bounded legacy test doubles: absence of an
    # initiation contract is the historical registered-true behavior.
    return True


def _option_observe(
    option: Any,
    state: Any,
    action: ActionCandidate,
    events: tuple[str, ...],
) -> Any:
    method = getattr(option, "observe", None)
    if not callable(method):
        raise TypeError("option automaton lacks observe")
    prepared = _option_prepare_observation(option, state, action)
    if prepared is None:
        return state
    state, schema = prepared
    for call in (
        lambda: method(state, schema, events=events),
        lambda: method(state, action, events=events),
        lambda: method(state, schema, events),
        lambda: method(state, action, events),
        lambda: method(state=state, action=schema, events=events),
        lambda: method(
            state=state,
            action=action,
            events=events,
        ),
    ):
        try:
            return call()
        except (TypeError, ValueError):
            continue
    raise ValueError("option observation failed for every supported call contract")


def _option_prepare_observation(
    option: Any,
    state: Any,
    action: ActionCandidate,
) -> tuple[Any, str] | None:
    """Create the post-action/pre-observation cursor expected by automata.

    The decision engine issues a real action directly.  Mixed option automata
    normally create this pending cursor in ``execute_one``; the posterior must
    reproduce that bookkeeping before consuming the resulting observation.
    """

    schema = _option_schema_for_action(option, state, action)
    if schema is None:
        return None
    if not (
        hasattr(state, "awaiting_observation")
        and hasattr(state, "pending_action_schema")
        and hasattr(state, "steps")
        and hasattr(state, "state_visits")
    ):
        return state, schema
    if bool(getattr(state, "terminated", False)):
        return None
    if bool(getattr(state, "awaiting_observation", False)):
        pending = str(getattr(state, "pending_action_schema", ""))
        return (state, pending) if pending else None
    horizon = max(1, min(16, int(getattr(option, "maximum_horizon", 16))))
    steps = int(getattr(state, "steps", 0))
    if steps >= horizon:
        return None
    try:
        pending_state = replace(
            state,
            steps=steps + 1,
            state_visits=int(getattr(state, "state_visits", 0)) + 1,
            awaiting_observation=True,
            pending_action_schema=schema,
            terminated=False,
        )
    except TypeError:
        return state, schema
    return pending_state, schema


def _option_schema_for_action(
    option: Any,
    state: Any,
    action: ActionCandidate,
) -> str | None:
    method = getattr(option, "allowed_action_schemas", None)
    if not callable(method):
        return None
    try:
        allowed = tuple(str(item).lower() for item in method(state))
    except (KeyError, TypeError, ValueError):
        return None
    for schema in allowed:
        if schema == "*" or schema.upper() == action.action_name:
            return action.action_name.lower() if schema == "*" else schema
    return None


def _option_allows(option: Any, state: Any, action: ActionCandidate) -> bool:
    return _option_schema_for_action(option, state, action) is not None


def _action_grounding_terms(action: ActionCandidate) -> frozenset[str]:
    terms: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized:
                terms.add(normalized)
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, Sequence) and not isinstance(
            value,
            (bytes, bytearray),
        ):
            for nested in value:
                visit(nested)

    for key, value in action.action_data.items():
        if str(key).strip().lower() != "relation":
            visit(value)
    return frozenset(terms)


def _option_relation_satisfied(
    option: Any,
    state: Any,
    action: ActionCandidate,
    abstract_state: AbstractState | None,
) -> bool:
    """Require structural evidence for relation-labelled option transitions."""

    schema = _option_schema_for_action(option, state, action)
    if schema is None:
        return False
    state_id = str(getattr(state, "state_id", getattr(state, "id", state)))
    required = {
        str(getattr(transition, "relation", "identity")).strip().lower()
        for transition in tuple(getattr(option, "transitions", ()))
        if str(
            getattr(
                transition,
                "source_state_id",
                getattr(transition, "source", ""),
            )
        )
        == state_id
        and str(getattr(transition, "action_schema", "")).lower() == schema
        and str(getattr(transition, "relation", "identity")).strip().lower()
        != "identity"
    }
    if not required:
        return True
    grounded = str(action.action_data.get("relation", "")).strip().lower()
    if grounded not in required or not isinstance(abstract_state, AbstractState):
        return False

    # Provenance is conjunctive: the grounded action must name the transition
    # relation, and the native-frame state must independently prove it.
    if int(dict(abstract_state.topology).get(grounded, 0)) > 0:
        return True
    facts = tuple(
        fact for fact in abstract_state.true_facts if fact.predicate == grounded
    )
    if not facts:
        return False
    grounded_terms = _action_grounding_terms(action)
    return any(
        not fact.terms
        or not grounded_terms
        or bool(grounded_terms & {term.lower() for term in fact.terms})
        for fact in facts
    )


def _fallback_supported_by_option(
    action: ActionCandidate,
    *,
    classes: Sequence[GaugeClass],
    frame_states: Mapping[str, AbstractState],
) -> bool:
    """Require one current posterior option to authorize a fallback action."""

    for gauge_class in classes:
        for particle in gauge_class.particles:
            option = particle.hypothesis.option
            native_state = frame_states.get(_frame_id(particle.hypothesis.frame))
            if not _option_can_initiate(option, native_state):
                continue
            if not _option_allows(option, particle.option_state, action):
                continue
            if _option_relation_satisfied(
                option,
                particle.option_state,
                action,
                native_state,
            ):
                return True
    return False


def _next_option_action(
    option: Any,
    state: Any,
    actions_by_name: Mapping[str, Sequence[ActionCandidate]],
    *,
    previous: ActionCandidate,
) -> ActionCandidate | None:
    method = getattr(option, "allowed_action_schemas", None)
    if not callable(method):
        return None
    try:
        schemas = tuple(str(item).upper() for item in method(state))
    except (KeyError, TypeError, ValueError):
        return None
    for schema in schemas:
        if schema == "*":
            return previous
        candidates = tuple(actions_by_name.get(schema, ()))
        if not candidates:
            continue
        if previous.action_name == schema:
            for candidate in candidates:
                if candidate.key == previous.key:
                    return candidate
        return candidates[0]
    return None


def _prediction_events(packet: PredictionPacket) -> tuple[str, ...]:
    """Convert a deterministic counterfactual packet into option evidence."""

    events: set[str] = set()
    changed = False
    for values in (
        packet.object_deltas,
        packet.relation_deltas,
        packet.topology_deltas,
    ):
        for key, probability in values.items():
            if float(probability) < 0.5:
                continue
            changed = True
            normalized = str(key).strip().lower()
            if normalized:
                events.add(normalized)
                events.add(normalized.rsplit(":", 1)[-1])
    if packet.progress_mean is not None and float(packet.progress_mean) > 0.0:
        events.update(("progress", "state_change"))
        changed = True
    if packet.goal_probability is not None and float(packet.goal_probability) >= 0.5:
        events.update(("goal", "level_complete", "win"))
    if (
        packet.terminal_probability is not None
        and float(packet.terminal_probability) >= 0.5
    ):
        events.update(("terminal", "game_over"))
    if packet.state_after is not None:
        events.update(fact.predicate for fact in packet.state_after.true_facts)
    if changed:
        # ``state_changed`` is the frozen T10.2 learned predicate used by
        # mixed option guards.  Keep the earlier singular spelling as a
        # compatibility alias for baseline programs, but never make a
        # counterfactual rollout disagree with the physical event vocabulary.
        events.update(("state_changed", "state_change"))
    else:
        events.add("no_effect")
    return tuple(sorted(events))


def _stateful_signature(value: Any, option: Any | None = None) -> str:
    method = getattr(option, "stateful_signature", None)
    if callable(method):
        signature = str(method(value)).strip()
        if not signature:
            raise ValueError("option stateful_signature cannot be empty")
        return signature
    return _sha256(_canonicalize(value))[:20]


def _logsumexp(values: Iterable[float]) -> float:
    items = tuple(float(value) for value in values)
    if not items:
        return float("-inf")
    maximum = max(items)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(item - maximum) for item in items))


def _entropy_with_residual(
    weighted_signatures: Sequence[tuple[float, Any]],
    residual_mass: float,
) -> float:
    classes: dict[str, float] = {}
    for probability, signature in weighted_signatures:
        key = _sha256(signature)
        classes[key] = classes.get(key, 0.0) + float(probability)
    if residual_mass > 0.0:
        aggregate = max(classes, key=classes.get, default="residual")
        classes[aggregate] = classes.get(aggregate, 0.0) + residual_mass
    return -sum(
        mass * math.log(max(mass, 1e-300)) for mass in classes.values() if mass > 0.0
    )


__all__ = [
    "FORMAT_VERSION",
    "MAP_BAYES_FACTOR_THRESHOLD",
    "MAP_MASS_THRESHOLD",
    "MAP_STABILITY_TRANSITIONS",
    "MAXIMUM_DECISION_CLASSES",
    "MAXIMUM_GAUGE_CLASSES",
    "NEW_AST_NODE_LOG_PRIOR",
    "GaugeActionAssessment",
    "GaugeClass",
    "GaugeDecision",
    "GaugeDecisionEngine",
    "GaugeParticle",
    "GaugeProgramPosterior",
    "GaugeUpdate",
    "JointGaugeHypothesis",
    "OptionSequenceRanking",
    "OptionSequenceSignatureMass",
    "rank_option_sequence_signatures",
]
