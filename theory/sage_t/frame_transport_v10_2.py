"""Partial, auditable transport between SAGE.T10.2 observer frames.

Observer frames are allowed to use different local names for the same
structural roles, facts, and actions.  A transport records only the witnessed
correspondence between those names.  Missing correspondences remain explicit:
commutativity checks never turn an incomplete projection into positive
evidence.

The implementation depends only on the frozen SAGE.T contracts.  Projection
objects are read by protocol (``state``, ``action``, ``frame_id``,
``complete`` and ``covered_channels``), so light-weight test doubles and the
T10.2 :class:`FrameProjection`/``ProjectedTransition`` wrappers are both
supported.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

from .contracts import (
    ALLOWED_PREDICATES,
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    PredictionPacket,
)

try:  # Runtime import documents the intended integration without hard coupling.
    from .observer_frames_v10_2 import OBSERVER_FRAME_SPECS, FrameProjection
    from .observer_frames_v10_2 import canonical_sha256 as _observer_sha256
    from .observer_frames_v10_2 import state_model_payload as _observer_state_payload
except ImportError:  # pragma: no cover - useful while modules land independently.
    OBSERVER_FRAME_SPECS = ()
    FrameProjection = Any  # type: ignore[misc,assignment]
    _observer_sha256 = None
    _observer_state_payload = None


TRANSPORT_FORMAT_VERSION = "sage-t10.2-frame-transport-v2"
STATE_CHANNELS = frozenset(
    {"entities", "facts", "counters", "registers", "topology", "regime"}
)
PREDICTION_CHANNELS = frozenset(
    {"objects", "relations", "topology", "progress", "terminal", "goal"}
)
_KINDS = ("role", "fact", "action")
_CERTIFIED_TRANSPORT_CONSTRUCTION_TOKEN = object()
_CERTIFIED_ORBIT_CONSTRUCTION_TOKEN = object()
_REGISTERED_FRAME_IDS = frozenset(frame.frame_id for frame in OBSERVER_FRAME_SPECS)
_ALLOWED_PERSISTED_ROLES = frozenset(
    {
        "action_root",
        "actor",
        "container",
        "free_region",
        "goal",
        "movable",
        "neighbor",
        "object",
        "obstacle",
        "player",
        "selected",
        "sink",
        "source",
        "space",
        "structural_class",
        "target",
    }
)
_ALLOWED_PERSISTED_FACTS = frozenset(ALLOWED_PREDICATES - {"same_color"})
_ALLOWED_PERSISTED_ACTIONS = frozenset(f"ACTION{index}" for index in range(1, 65))
_PERSISTED_ORBIT_FIELDS = frozenset(
    {
        "orbit_payload",
        "orbit_hash",
        "source_frame",
        "target_frame",
        "attestation",
    }
)
_PERSISTED_ORBIT_PAYLOAD_FIELDS = frozenset(
    {
        "format_version",
        "kind",
        "frames",
        "symbol_edges",
        "certified_domain",
    }
)
_PERSISTED_ATTESTATION_FIELDS = frozenset(
    {
        "certificate_hash",
        "receipt",
        "source_before_summary_hash",
        "source_after_summary_hash",
        "target_before_summary_hash",
        "target_after_summary_hash",
        "source_observation_hash",
        "target_observation_hash",
        "live_graph_exact_attested",
        "round_trip_exact",
        "summary_commutative_exact",
    }
)
_PERSISTED_ATTESTATION_BINDING_FIELDS = _PERSISTED_ATTESTATION_FIELDS - {"receipt"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        rendered = [_json_safe(item) for item in value]
        return sorted(
            rendered,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    if _observer_sha256 is not None:
        return _observer_sha256(value)
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_exact_fields(
    value: Mapping[str, Any],
    required: frozenset[str],
    *,
    label: str,
) -> None:
    observed = set(value)
    if observed != set(required):
        raise ValueError(
            f"{label} schema drifted; "
            f"missing={sorted(required - observed)}, "
            f"unknown={sorted(observed - required, key=str)}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if value != digest:
        raise ValueError(f"{label} must use canonical lowercase encoding")
    return digest


def _persisted_symbol_allowed(kind: str, symbol: str) -> bool:
    if kind == "role":
        return symbol in _ALLOWED_PERSISTED_ROLES
    if kind == "fact":
        return symbol in _ALLOWED_PERSISTED_FACTS
    if kind == "action":
        return symbol in _ALLOWED_PERSISTED_ACTIONS
    return False


def persisted_attestation_receipt(
    *,
    orbit_hash: str,
    source_frame: str,
    target_frame: str,
    attestation: Mapping[str, Any],
) -> str:
    """Return the reproducible receipt binding one persisted orbit envelope."""

    if not isinstance(attestation, dict):
        raise TypeError("persisted attestation receipt requires a JSON object")
    _require_exact_fields(
        attestation,
        _PERSISTED_ATTESTATION_BINDING_FIELDS,
        label="persisted attestation receipt",
    )
    source = _normalize_frame_id(source_frame)
    target = _normalize_frame_id(target_frame)
    if source_frame != source or target_frame != target:
        raise ValueError("persisted attestation frames must be canonical")
    digest = _require_sha256(orbit_hash, label="persisted orbit hash")
    normalized: dict[str, Any] = {}
    for key, value in attestation.items():
        if key.endswith("_hash"):
            normalized[key] = _require_sha256(
                value,
                label=f"persisted attestation {key}",
            )
        else:
            if not isinstance(value, bool):
                raise TypeError(f"persisted attestation {key} must be boolean")
            normalized[key] = value
    return _sha256(
        {
            "kind": "persisted_structural_orbit_attestation_v2",
            "orbit_hash": digest,
            "source_frame": source,
            "target_frame": target,
            "attestation": normalized,
        }
    )


def _frame_identifier(value: Any) -> str:
    direct = getattr(value, "frame_id", "")
    if direct:
        return str(direct).strip().lower()
    for owner_name in ("frame", "spec"):
        owner = getattr(value, owner_name, None)
        frame_id = getattr(owner, "frame_id", "")
        if frame_id:
            return str(frame_id).strip().lower()
    return ""


def _normalize_frame_id(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("transport frame ids cannot be empty")
    return normalized


def _normalize_symbol(kind: str, value: Any) -> str:
    text = str(value).strip()
    if not text or len(text) > 256 or any(character in text for character in "\r\n"):
        raise ValueError(f"invalid {kind} transport symbol")
    if kind == "action":
        return text.upper()
    return text.lower()


def _normalize_pairs(
    kind: str,
    values: Mapping[str, str] | Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    raw_pairs = values.items() if isinstance(values, Mapping) else values
    normalized: dict[str, str] = {}
    for raw_source, raw_target in raw_pairs:
        source = _normalize_symbol(kind, raw_source)
        target = _normalize_symbol(kind, raw_target)
        previous = normalized.get(source)
        if previous is not None and previous != target:
            raise ValueError(f"conflicting {kind} transport for {source}")
        normalized[source] = target
    return tuple(sorted(normalized.items()))


def _token(kind: str, symbol: str) -> str:
    return f"{kind}:{_normalize_symbol(kind, symbol)}"


def _split_token(token: str) -> tuple[str, str]:
    kind, separator, symbol = str(token).partition(":")
    if not separator or kind not in _KINDS:
        raise ValueError(
            "transport domain entries must be role:, fact:, or action: tokens"
        )
    return kind, _normalize_symbol(kind, symbol)


def _mapping_for(transport: TransportMap, kind: str) -> dict[str, str]:
    return dict(getattr(transport, f"{kind}_map"))


@dataclass(frozen=True)
class TransportMap:
    """Immutable, possibly partial symbolic map between two observer frames.

    ``domain`` contains typed source symbols such as ``role:target`` and
    ``action:ACTION1``.  It may be larger than the explicitly mapped keys;
    those declared-but-unmapped entries are the map's visible partiality.
    Symbols omitted from the domain may still be shared identities, but a
    certificate checks that assumption against the target projection.
    """

    source_frame_id: str
    target_frame_id: str
    role_map: tuple[tuple[str, str], ...] = ()
    fact_map: tuple[tuple[str, str], ...] = ()
    action_map: tuple[tuple[str, str], ...] = ()
    domain: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_frame_id",
            _normalize_frame_id(self.source_frame_id),
        )
        object.__setattr__(
            self,
            "target_frame_id",
            _normalize_frame_id(self.target_frame_id),
        )
        for kind in _KINDS:
            object.__setattr__(
                self,
                f"{kind}_map",
                _normalize_pairs(kind, getattr(self, f"{kind}_map")),
            )
        normalized_domain = {_token(*_split_token(item)) for item in self.domain}
        normalized_domain.update(self.mapped_domain)
        object.__setattr__(self, "domain", frozenset(normalized_domain))

    @property
    def source_frame(self) -> str:
        return self.source_frame_id

    @property
    def target_frame(self) -> str:
        return self.target_frame_id

    @property
    def mapped_domain(self) -> frozenset[str]:
        return frozenset(
            _token(kind, source)
            for kind in _KINDS
            for source, _ in getattr(self, f"{kind}_map")
        )

    @property
    def codomain(self) -> frozenset[str]:
        return frozenset(
            _token(kind, target)
            for kind in _KINDS
            for _, target in getattr(self, f"{kind}_map")
        )

    @property
    def coverage(self) -> float:
        if not self.domain:
            return 1.0
        return len(self.mapped_domain & self.domain) / len(self.domain)

    @property
    def missing_domain(self) -> frozenset[str]:
        return self.domain - self.mapped_domain

    @property
    def ambiguities(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        output: list[tuple[str, tuple[str, ...]]] = []
        for kind in _KINDS:
            sources_by_target: dict[str, list[str]] = defaultdict(list)
            for source, target in getattr(self, f"{kind}_map"):
                sources_by_target[target].append(source)
            for target, sources in sorted(sources_by_target.items()):
                if len(sources) > 1:
                    output.append(
                        (
                            _token(kind, target),
                            tuple(_token(kind, item) for item in sorted(sources)),
                        )
                    )
        return tuple(output)

    @property
    def ambiguity(self) -> int:
        return sum(len(sources) - 1 for _, sources in self.ambiguities)

    @property
    def ambiguous(self) -> bool:
        return bool(self.ambiguities)

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": TRANSPORT_FORMAT_VERSION,
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "role_map": [list(item) for item in self.role_map],
            "fact_map": [list(item) for item in self.fact_map],
            "action_map": [list(item) for item in self.action_map],
            "domain": sorted(self.domain),
        }

    @property
    def canonical_hash(self) -> str:
        return _sha256(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash

    def map_symbol(self, kind: str, value: str) -> str | None:
        if kind not in _KINDS:
            raise ValueError(f"unknown transport symbol kind: {kind}")
        normalized = _normalize_symbol(kind, value)
        mapping = _mapping_for(self, kind)
        if normalized in mapping:
            return mapping[normalized]
        if _token(kind, normalized) in self.domain:
            return None
        return normalized

    def map_role(self, role: str) -> str | None:
        return self.map_symbol("role", role)

    def map_fact(self, fact: str) -> str | None:
        return self.map_symbol("fact", fact)

    def map_action(self, action: str) -> str | None:
        return self.map_symbol("action", action)

    def inverted(self) -> TransportMap | None:
        """Return the unique inverse on the mapped subdomain, if it exists."""

        if self.ambiguous:
            return None
        return TransportMap(
            source_frame_id=self.target_frame_id,
            target_frame_id=self.source_frame_id,
            role_map=tuple((target, source) for source, target in self.role_map),
            fact_map=tuple((target, source) for source, target in self.fact_map),
            action_map=tuple((target, source) for source, target in self.action_map),
            domain=self.codomain,
        )

    @property
    def inverse(self) -> TransportMap | None:
        return self.inverted()

    def validate_round_trip(
        self,
        inverse: TransportMap | None = None,
        *,
        require_complete: bool = True,
    ) -> bool:
        candidate = inverse or self.inverted()
        if candidate is None:
            return False
        if (
            candidate.source_frame_id != self.target_frame_id
            or candidate.target_frame_id != self.source_frame_id
        ):
            return False
        if require_complete and (self.coverage < 1.0 or candidate.coverage < 1.0):
            return False
        forward_domain = set(self.domain) | set(candidate.codomain)
        reverse_domain = set(candidate.domain) | set(self.codomain)
        for token in forward_domain:
            kind, source = _split_token(token)
            target = self.map_symbol(kind, source)
            if target is None or candidate.map_symbol(kind, target) != source:
                return False
        for token in reverse_domain:
            kind, target = _split_token(token)
            source = candidate.map_symbol(kind, target)
            if source is None or self.map_symbol(kind, source) != target:
                return False
        return True

    def transport_state(self, state: AbstractState) -> AbstractState:
        return transport_state(state, self)

    def transport_action(self, action: ActionCandidate) -> ActionCandidate | None:
        return transport_action(action, self)

    def transport_prediction(self, packet: PredictionPacket) -> PredictionPacket:
        return transport_prediction(packet, self)

    def certificate(
        self,
        source_projection: Any,
        target_projection: Any,
        *,
        inverse: TransportMap | None = None,
    ) -> TransportCertificate:
        return TransportCertificate.from_projections(
            self,
            source_projection,
            target_projection,
            inverse=inverse,
        )


def _projection_state(value: Any, *, stage: str = "before") -> AbstractState | None:
    if isinstance(value, AbstractState):
        return value
    state = getattr(value, "state", None)
    if isinstance(state, AbstractState):
        return state
    branch = getattr(value, stage, None)
    if branch is not None:
        state = _projection_state(branch, stage=stage)
        if state is not None:
            return state
    attribute = "state_after" if stage == "after" else "state_before"
    state = getattr(value, attribute, None)
    return state if isinstance(state, AbstractState) else None


def _projection_action(value: Any) -> ActionCandidate | None:
    if isinstance(value, ActionCandidate):
        return value
    action = getattr(value, "action", None)
    if isinstance(action, ActionCandidate):
        return action
    before = getattr(value, "before", None)
    if before is not None:
        return _projection_action(before)
    return None


def _projection_complete(value: Any) -> bool:
    if isinstance(value, AbstractState):
        return True
    complete = getattr(value, "complete", True)
    missing = tuple(getattr(value, "missing", ()))
    return bool(complete and not missing)


def _projection_channels(value: Any) -> frozenset[str]:
    if isinstance(value, AbstractState):
        return STATE_CHANNELS
    channels = getattr(value, "covered_channels", STATE_CHANNELS)
    return frozenset(str(channel).strip().lower() for channel in channels)


def _projection_hash(value: Any, *, stage: str = "before") -> str:
    branch = getattr(value, stage, None)
    if branch is not None:
        return _projection_hash(branch, stage=stage)
    for attribute in ("canonical_hash", "canonical_checksum"):
        checksum = getattr(value, attribute, "")
        if checksum:
            return str(checksum)
    state = _projection_state(value, stage=stage)
    if state is not None:
        return canonical_signature(state)
    return ""


def _projection_domain(value: Any, *, stage: str = "before") -> frozenset[str]:
    state = _projection_state(value, stage=stage)
    domain: set[str] = set()
    if state is not None:
        for entity in state.entities:
            domain.update(_token("role", role) for role in entity.roles)
        for fact in set(state.true_facts) | set(state.false_facts):
            domain.add(_token("fact", fact.predicate))
    action = _projection_action(value)
    if action is not None:
        domain.add(_token("action", action.action_name))
    return frozenset(domain)


def _map_domain_token(transport: TransportMap, token: str) -> str | None:
    kind, symbol = _split_token(token)
    target = transport.map_symbol(kind, symbol)
    return None if target is None else _token(kind, target)


@dataclass(frozen=True)
class TransportCertificate:
    """Evidence that a partial map covers two concrete projections."""

    transport: TransportMap
    source_domain: frozenset[str]
    target_domain: frozenset[str]
    covered_domain: frozenset[str]
    missing_domain: frozenset[str]
    unmatched_target: frozenset[str]
    coverage: float
    ambiguities: tuple[tuple[str, tuple[str, ...]], ...]
    inverse_map: TransportMap | None
    round_trip_exact: bool
    projections_complete: bool
    frame_ids_match: bool
    source_projection_hash: str = ""
    target_projection_hash: str = ""
    source_gauge_signature: str = ""
    target_gauge_signature: str = ""
    source_projection_frame_id: str = ""
    target_projection_frame_id: str = ""
    projection_stage: str = "before"
    _construction_token: InitVar[object] = None
    _issued_by_factory: bool = field(
        init=False,
        repr=False,
        compare=False,
        default=False,
    )

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CERTIFIED_TRANSPORT_CONSTRUCTION_TOKEN:
            raise ValueError(
                "a transport certificate must be issued by from_projections"
            )
        object.__setattr__(self, "_issued_by_factory", True)
        for attribute in (
            "source_domain",
            "target_domain",
            "covered_domain",
            "missing_domain",
            "unmatched_target",
        ):
            object.__setattr__(self, attribute, frozenset(getattr(self, attribute)))
        coverage = float(self.coverage)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("transport certificate coverage must be in [0, 1]")
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(
            self,
            "source_projection_frame_id",
            str(self.source_projection_frame_id).strip().lower(),
        )
        object.__setattr__(
            self,
            "target_projection_frame_id",
            str(self.target_projection_frame_id).strip().lower(),
        )
        stage = str(self.projection_stage).strip().lower()
        if stage not in {"before", "after"}:
            raise ValueError("transport certificate stage must be before or after")
        object.__setattr__(self, "projection_stage", stage)

    @classmethod
    def from_projections(
        cls,
        transport: TransportMap,
        source_projection: Any,
        target_projection: Any,
        *,
        inverse: TransportMap | None = None,
        stage: str = "before",
    ) -> TransportCertificate:
        source_frame = _frame_identifier(source_projection)
        target_frame = _frame_identifier(target_projection)
        frame_ids_match = bool(
            (not source_frame or source_frame == transport.source_frame_id)
            and (not target_frame or target_frame == transport.target_frame_id)
        )
        source_domain = _projection_domain(source_projection, stage=stage)
        target_domain = _projection_domain(target_projection, stage=stage)
        covered: set[str] = set()
        reached: set[str] = set()
        for source in source_domain:
            target = _map_domain_token(transport, source)
            if target is not None and target in target_domain:
                covered.add(source)
                reached.add(target)
        missing = set(source_domain) - covered
        unmatched = set(target_domain) - reached
        coverage = len(covered) / len(source_domain) if source_domain else 1.0
        inverse_map = inverse if inverse is not None else transport.inverted()
        round_trip = bool(
            inverse_map is not None
            and inverse_map.source_frame_id == transport.target_frame_id
            and inverse_map.target_frame_id == transport.source_frame_id
            and transport.validate_round_trip(inverse_map)
            and inverse_map.validate_round_trip(transport)
        )
        if round_trip:
            for source in source_domain:
                target = _map_domain_token(transport, source)
                if (
                    target is None
                    or target not in target_domain
                    or _map_domain_token(inverse_map, target) != source
                ):
                    round_trip = False
                    break
        if round_trip:
            for target in target_domain:
                source = _map_domain_token(inverse_map, target)
                if (
                    source is None
                    or source not in source_domain
                    or _map_domain_token(transport, source) != target
                ):
                    round_trip = False
                    break
        return cls(
            transport=transport,
            source_domain=source_domain,
            target_domain=target_domain,
            covered_domain=frozenset(covered),
            missing_domain=frozenset(missing),
            unmatched_target=frozenset(unmatched),
            coverage=coverage,
            ambiguities=transport.ambiguities,
            inverse_map=inverse_map,
            round_trip_exact=bool(round_trip),
            projections_complete=(
                _projection_complete(source_projection)
                and _projection_complete(target_projection)
            ),
            frame_ids_match=frame_ids_match,
            source_projection_hash=_projection_hash(
                source_projection,
                stage=stage,
            ),
            target_projection_hash=_projection_hash(
                target_projection,
                stage=stage,
            ),
            source_gauge_signature=canonical_signature(
                source_projection,
                transport,
                stage=stage,
            ),
            target_gauge_signature=canonical_signature(
                target_projection,
                stage=stage,
            ),
            source_projection_frame_id=source_frame,
            target_projection_frame_id=target_frame,
            projection_stage=stage,
            _construction_token=_CERTIFIED_TRANSPORT_CONSTRUCTION_TOKEN,
        )

    @property
    def source_frame_id(self) -> str:
        return self.transport.source_frame_id

    @property
    def target_frame_id(self) -> str:
        return self.transport.target_frame_id

    @property
    def source_frame(self) -> str:
        return self.source_frame_id

    @property
    def target_frame(self) -> str:
        return self.target_frame_id

    @property
    def ambiguous(self) -> bool:
        return bool(self.ambiguities)

    @property
    def ambiguity(self) -> int:
        return sum(len(sources) - 1 for _, sources in self.ambiguities)

    @property
    def exact(self) -> bool:
        return bool(
            self.projections_complete
            and self.frame_ids_match
            and not self.ambiguous
            and not self.missing_domain
            and not self.unmatched_target
            and self.round_trip_exact
        )

    @property
    def certifies_gauge_equivalence(self) -> bool:
        """Whether this witness proves equal canonical behavior in both frames."""

        return bool(
            self.exact
            and self.has_frame_provenance
            and self.source_gauge_signature
            and self.source_gauge_signature == self.target_gauge_signature
        )

    @property
    def has_frame_provenance(self) -> bool:
        """Whether both concrete projections identify the declared endpoints."""

        return bool(
            self.source_projection_frame_id == self.source_frame_id
            and self.target_projection_frame_id == self.target_frame_id
        )

    @property
    def gauge_equivalence_key(self) -> str | None:
        """Direction-invariant key available only for a certified gauge map."""

        if not self.certifies_gauge_equivalence:
            return None
        edges = []
        for kind in _KINDS:
            for source, target in getattr(self.transport, f"{kind}_map"):
                endpoints = sorted(
                    (
                        (self.source_frame_id, kind, source),
                        (self.target_frame_id, kind, target),
                    )
                )
                edges.append(endpoints)
        return _sha256(
            {
                "format_version": TRANSPORT_FORMAT_VERSION,
                "kind": "certified_gauge_equivalence",
                "frames": sorted((self.source_frame_id, self.target_frame_id)),
                "symbol_edges": sorted(edges, key=_canonical_json),
            }
        )

    def transport_state(self, state: AbstractState) -> AbstractState:
        return transport_state(state, self.transport)

    def transport_action(self, action: ActionCandidate) -> ActionCandidate | None:
        return transport_action(action, self.transport)

    def transport_prediction(self, packet: PredictionPacket) -> PredictionPacket:
        return transport_prediction(packet, self.transport)

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": TRANSPORT_FORMAT_VERSION,
            "transport_hash": self.transport.canonical_hash,
            "source_domain": sorted(self.source_domain),
            "target_domain": sorted(self.target_domain),
            "covered_domain": sorted(self.covered_domain),
            "missing_domain": sorted(self.missing_domain),
            "unmatched_target": sorted(self.unmatched_target),
            "coverage": round(self.coverage, 12),
            "ambiguities": self.ambiguities,
            "round_trip_exact": self.round_trip_exact,
            "projections_complete": self.projections_complete,
            "frame_ids_match": self.frame_ids_match,
            "source_projection_hash": self.source_projection_hash,
            "target_projection_hash": self.target_projection_hash,
            "source_gauge_signature": self.source_gauge_signature,
            "target_gauge_signature": self.target_gauge_signature,
            "source_projection_frame_id": self.source_projection_frame_id,
            "target_projection_frame_id": self.target_projection_frame_id,
            "projection_stage": self.projection_stage,
            "has_frame_provenance": self.has_frame_provenance,
            "certifies_gauge_equivalence": self.certifies_gauge_equivalence,
            "gauge_equivalence_key": self.gauge_equivalence_key,
        }

    @property
    def canonical_hash(self) -> str:
        return _sha256(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash


@dataclass(frozen=True)
class TransportOrbitWitness:
    """Stable structural witness for one gauge-equivalent frame orbit.

    A :class:`TransportCertificate` is deliberately event-local: it records
    the projections that were compared and therefore must never become part of
    a posterior hypothesis hash.  This witness retains only the certified
    bijection between frame symbols.  Forward and reverse witnesses have the
    same canonical hash, while their ``transport`` member keeps the direction
    needed by the executor.
    """

    transport: TransportMap
    inverse_map: TransportMap
    certified_source_domain: frozenset[str]
    certified_target_domain: frozenset[str]
    certification_receipt: str = field(repr=False, compare=False)
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CERTIFIED_ORBIT_CONSTRUCTION_TOKEN:
            raise ValueError(
                "a gauge orbit must be issued from a transport certificate"
            )
        source_domain = frozenset(self.certified_source_domain)
        target_domain = frozenset(self.certified_target_domain)
        object.__setattr__(self, "certified_source_domain", source_domain)
        object.__setattr__(self, "certified_target_domain", target_domain)
        receipt = str(self.certification_receipt).strip().lower()
        if len(receipt) != 64 or any(
            character not in "0123456789abcdef" for character in receipt
        ):
            raise ValueError("a gauge orbit requires certified projection provenance")
        object.__setattr__(self, "certification_receipt", receipt)
        if not self.transport.domain or not source_domain or not target_domain:
            raise ValueError("a gauge orbit requires a non-empty certified domain")
        if self.transport.coverage < 1.0 or self.transport.ambiguous:
            raise ValueError("a gauge orbit requires a complete unambiguous transport")
        if (
            self.inverse_map.source_frame_id != self.transport.target_frame_id
            or self.inverse_map.target_frame_id != self.transport.source_frame_id
            or not self.transport.validate_round_trip(self.inverse_map)
            or not self.inverse_map.validate_round_trip(self.transport)
        ):
            raise ValueError("a gauge orbit requires an exact inverse round trip")
        for source in source_domain:
            target = _map_domain_token(self.transport, source)
            if (
                target is None
                or target not in target_domain
                or _map_domain_token(self.inverse_map, target) != source
            ):
                raise ValueError(
                    "certified source domain is incompatible with transport"
                )
        for target in target_domain:
            source = _map_domain_token(self.inverse_map, target)
            if (
                source is None
                or source not in source_domain
                or _map_domain_token(self.transport, source) != target
            ):
                raise ValueError("certified target domain is incompatible with inverse")

    @classmethod
    def from_certificate(
        cls,
        certificate: TransportCertificate,
    ) -> TransportOrbitWitness:
        """Forget event-local evidence after it certified structural equality."""

        if not getattr(certificate, "_issued_by_factory", False):
            raise ValueError("transport certificate was not issued by its factory")
        if not certificate.has_frame_provenance:
            raise ValueError(
                "event certificate lacks exact source/target frame provenance"
            )
        if not certificate.certifies_gauge_equivalence:
            raise ValueError("event certificate does not prove gauge equivalence")
        if (
            not certificate.source_domain
            or not certificate.target_domain
            or not certificate.source_projection_hash
            or not certificate.target_projection_hash
        ):
            raise ValueError("event certificate has no non-empty projection provenance")
        inverse = certificate.inverse_map or certificate.transport.inverted()
        if inverse is None:
            raise ValueError("certified gauge transport has no inverse")
        receipt = _sha256(
            {
                "kind": "event_certificate_receipt",
                "certificate_hash": certificate.canonical_hash,
                "source_projection_hash": certificate.source_projection_hash,
                "target_projection_hash": certificate.target_projection_hash,
            }
        )
        return cls(
            certificate.transport,
            inverse,
            certificate.source_domain,
            certificate.target_domain,
            receipt,
            _CERTIFIED_ORBIT_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def from_live_evidence(
        cls,
        before_certificate: TransportCertificate,
        after_certificate: TransportCertificate,
        *,
        source_domain: Iterable[str],
        target_domain: Iterable[str],
        dynamics_commutes: bool,
    ) -> TransportOrbitWitness:
        """Issue a full-vocabulary orbit from two staged live certificates."""

        certificates = (before_certificate, after_certificate)
        if any(
            not getattr(certificate, "_issued_by_factory", False)
            for certificate in certificates
        ):
            raise ValueError("live orbit requires factory-issued certificates")
        if (
            before_certificate.projection_stage != "before"
            or after_certificate.projection_stage != "after"
        ):
            raise ValueError("live orbit requires before and after certificates")
        if any(
            not certificate.certifies_gauge_equivalence for certificate in certificates
        ):
            raise ValueError("live orbit certificates are not exact gauge evidence")
        if (
            before_certificate.transport.canonical_hash
            != after_certificate.transport.canonical_hash
        ):
            raise ValueError("live orbit certificate transports disagree")
        if dynamics_commutes is not True:
            raise ValueError("live orbit requires an exact dynamics diagram")
        transport = before_certificate.transport
        inverse = transport.inverted()
        if inverse is None:
            raise ValueError("live orbit transport has no inverse")
        source = frozenset(_token(*_split_token(item)) for item in source_domain)
        target = frozenset(_token(*_split_token(item)) for item in target_domain)
        if source != transport.mapped_domain or target != transport.codomain:
            raise ValueError("live orbit domains must cover the complete bijection")
        receipt = _sha256(
            {
                "kind": "live_staged_certificate_receipt_v2",
                "before_certificate_hash": before_certificate.canonical_hash,
                "after_certificate_hash": after_certificate.canonical_hash,
                "source_domain": sorted(source),
                "target_domain": sorted(target),
            }
        )
        return cls(
            transport,
            inverse,
            source,
            target,
            receipt,
            _CERTIFIED_ORBIT_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def from_persisted_attestation(
        cls,
        envelope: Mapping[str, Any],
    ) -> TransportOrbitWitness:
        """Rebuild an orbit only from its closed, receipt-bound v2 envelope."""

        if not isinstance(envelope, dict):
            raise TypeError("persisted orbit envelope must be a JSON object")
        _require_exact_fields(
            envelope,
            _PERSISTED_ORBIT_FIELDS,
            label="persisted orbit envelope",
        )
        source_frame = _normalize_frame_id(envelope["source_frame"])
        target_frame = _normalize_frame_id(envelope["target_frame"])
        if (
            envelope["source_frame"] != source_frame
            or envelope["target_frame"] != target_frame
        ):
            raise ValueError("persisted orbit frame ids must be canonical")
        if source_frame == target_frame:
            raise ValueError("persisted orbit requires two distinct frames")
        if (
            source_frame not in _REGISTERED_FRAME_IDS
            or target_frame not in _REGISTERED_FRAME_IDS
        ):
            raise ValueError("persisted orbit references an unregistered frame")

        payload = envelope["orbit_payload"]
        if not isinstance(payload, dict):
            raise TypeError("persisted orbit payload must be a JSON object")
        _require_exact_fields(
            payload,
            _PERSISTED_ORBIT_PAYLOAD_FIELDS,
            label="persisted orbit payload",
        )
        if payload["format_version"] != TRANSPORT_FORMAT_VERSION:
            raise ValueError("unsupported persisted orbit format version")
        if payload["kind"] != "stable_structural_gauge_orbit":
            raise ValueError("unsupported persisted orbit kind")
        frames = payload["frames"]
        expected_frames = sorted((source_frame, target_frame))
        if not isinstance(frames, list) or frames != expected_frames:
            raise ValueError("persisted orbit frames do not match its envelope")

        raw_edges = payload["symbol_edges"]
        if not isinstance(raw_edges, list) or not raw_edges:
            raise ValueError("persisted orbit requires non-empty symbol edges")
        if raw_edges != sorted(raw_edges, key=_canonical_json):
            raise ValueError("persisted orbit symbol edges are not canonical")
        if len({_canonical_json(item) for item in raw_edges}) != len(raw_edges):
            raise ValueError("persisted orbit contains duplicate symbol edges")
        pairs: dict[str, list[tuple[str, str]]] = {kind: [] for kind in _KINDS}
        source_symbols: set[str] = set()
        target_symbols: set[str] = set()
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, list) or len(raw_edge) != 2:
                raise TypeError("persisted orbit edge must contain two endpoints")
            if raw_edge != sorted(raw_edge, key=_canonical_json):
                raise ValueError("persisted orbit endpoints are not canonical")
            decoded: dict[str, tuple[str, str]] = {}
            for raw_endpoint in raw_edge:
                if not isinstance(raw_endpoint, list) or len(raw_endpoint) != 3:
                    raise TypeError(
                        "persisted orbit endpoint must be [frame, kind, symbol]"
                    )
                frame_id, kind, raw_symbol = raw_endpoint
                if frame_id not in {source_frame, target_frame}:
                    raise ValueError("persisted orbit edge escaped its two frames")
                if not isinstance(kind, str) or kind not in _KINDS:
                    raise ValueError("persisted orbit edge has an unknown symbol kind")
                if not isinstance(raw_symbol, str):
                    raise TypeError("persisted orbit symbol must be a string")
                symbol = _normalize_symbol(kind, raw_symbol)
                if raw_symbol != symbol or not _persisted_symbol_allowed(kind, symbol):
                    raise ValueError(
                        "persisted orbit symbol is outside the closed allowlist"
                    )
                if frame_id in decoded:
                    raise ValueError("persisted orbit edge repeats one frame")
                decoded[frame_id] = (kind, symbol)
            if set(decoded) != {source_frame, target_frame}:
                raise ValueError("persisted orbit edge must join both frames")
            source_kind, source_symbol = decoded[source_frame]
            target_kind, target_symbol = decoded[target_frame]
            if source_kind != target_kind:
                raise ValueError("persisted orbit edge cannot cross symbol kinds")
            source_token = _token(source_kind, source_symbol)
            target_token = _token(target_kind, target_symbol)
            if source_token in source_symbols or target_token in target_symbols:
                raise ValueError("persisted orbit symbol edges are not bijective")
            source_symbols.add(source_token)
            target_symbols.add(target_token)
            pairs[source_kind].append((source_symbol, target_symbol))

        raw_domain = payload["certified_domain"]
        if not isinstance(raw_domain, list) or not raw_domain:
            raise ValueError("persisted orbit requires a certified domain")
        if raw_domain != sorted(raw_domain, key=_canonical_json):
            raise ValueError("persisted orbit certified domain is not canonical")
        if len({_canonical_json(item) for item in raw_domain}) != len(raw_domain):
            raise ValueError("persisted orbit contains duplicate domain entries")
        domains = {source_frame: set(), target_frame: set()}
        for raw_entry in raw_domain:
            if not isinstance(raw_entry, list) or len(raw_entry) != 2:
                raise TypeError("persisted orbit domain entry must be [frame, token]")
            frame_id, raw_token = raw_entry
            if frame_id not in domains or not isinstance(raw_token, str):
                raise ValueError("persisted orbit domain entry is invalid")
            kind, symbol = _split_token(raw_token)
            token = _token(kind, symbol)
            if raw_token != token or not _persisted_symbol_allowed(kind, symbol):
                raise ValueError(
                    "persisted orbit domain symbol is outside the closed allowlist"
                )
            domains[frame_id].add(token)
        source_domain = frozenset(domains[source_frame])
        target_domain = frozenset(domains[target_frame])
        if not source_domain or not target_domain:
            raise ValueError("persisted orbit requires two non-empty domains")
        if source_domain != source_symbols or target_domain != target_symbols:
            raise ValueError(
                "persisted orbit domains must fully cover every symbol edge"
            )

        transport = TransportMap(
            source_frame,
            target_frame,
            role_map=tuple(pairs["role"]),
            fact_map=tuple(pairs["fact"]),
            action_map=tuple(pairs["action"]),
            domain=source_domain,
        )
        inverse = transport.inverted()
        if (
            inverse is None
            or transport.ambiguous
            or transport.coverage < 1.0
            or transport.mapped_domain != source_domain
            or transport.codomain != target_domain
            or not transport.validate_round_trip(inverse)
            or not inverse.validate_round_trip(transport)
        ):
            raise ValueError("persisted orbit does not define an exact round trip")

        orbit_hash = _require_sha256(
            envelope["orbit_hash"],
            label="persisted orbit hash",
        )
        if orbit_hash != _sha256(payload):
            raise ValueError("persisted orbit hash mismatch")
        attestation = envelope["attestation"]
        if not isinstance(attestation, dict):
            raise TypeError("persisted orbit attestation must be a JSON object")
        _require_exact_fields(
            attestation,
            _PERSISTED_ATTESTATION_FIELDS,
            label="persisted orbit attestation",
        )
        receipt = _require_sha256(
            attestation["receipt"],
            label="persisted orbit receipt",
        )
        binding = {key: value for key, value in attestation.items() if key != "receipt"}
        expected_receipt = persisted_attestation_receipt(
            orbit_hash=orbit_hash,
            source_frame=source_frame,
            target_frame=target_frame,
            attestation=binding,
        )
        if receipt != expected_receipt:
            raise ValueError("persisted orbit receipt mismatch")
        if attestation["live_graph_exact_attested"] is not True:
            raise ValueError("persisted orbit lacks exact live-graph attestation")
        if attestation["round_trip_exact"] is not True:
            raise ValueError("persisted orbit lacks an exact round trip")
        if not isinstance(attestation["summary_commutative_exact"], bool):
            raise TypeError("persisted summary commutativity flag must be boolean")

        witness = cls(
            transport,
            inverse,
            source_domain,
            target_domain,
            receipt,
            _CERTIFIED_ORBIT_CONSTRUCTION_TOKEN,
        )
        if witness.canonical_hash != orbit_hash:
            raise ValueError("persisted orbit reconstruction changed its hash")
        if _json_safe(witness.canonical_payload) != payload:
            raise ValueError("persisted orbit payload is not canonical")
        return witness

    @property
    def source_frame_id(self) -> str:
        return self.transport.source_frame_id

    @property
    def target_frame_id(self) -> str:
        return self.transport.target_frame_id

    @property
    def source_frame(self) -> str:
        return self.source_frame_id

    @property
    def target_frame(self) -> str:
        return self.target_frame_id

    @property
    def certifies_gauge_equivalence(self) -> bool:
        return True

    @property
    def canonical_payload(self) -> dict[str, Any]:
        edges: list[tuple[tuple[str, str, str], tuple[str, str, str]]] = []
        for kind in _KINDS:
            for source, target in getattr(self.transport, f"{kind}_map"):
                endpoints = tuple(
                    sorted(
                        (
                            (self.source_frame_id, kind, source),
                            (self.target_frame_id, kind, target),
                        )
                    )
                )
                edges.append(endpoints)  # type: ignore[arg-type]
        return {
            "format_version": TRANSPORT_FORMAT_VERSION,
            "kind": "stable_structural_gauge_orbit",
            "frames": sorted((self.source_frame_id, self.target_frame_id)),
            "symbol_edges": sorted(edges, key=_canonical_json),
            "certified_domain": sorted(
                [
                    *(
                        (self.source_frame_id, token)
                        for token in self.certified_source_domain
                    ),
                    *(
                        (self.target_frame_id, token)
                        for token in self.certified_target_domain
                    ),
                ]
            ),
        }

    @property
    def canonical_hash(self) -> str:
        return _sha256(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash

    @property
    def gauge_equivalence_key(self) -> str:
        return self.canonical_hash

    @property
    def node_count(self) -> int:
        return max(
            1, sum(len(getattr(self.transport, f"{kind}_map")) for kind in _KINDS)
        )

    def inverted(self) -> TransportOrbitWitness:
        return TransportOrbitWitness(
            self.inverse_map,
            self.transport,
            self.certified_target_domain,
            self.certified_source_domain,
            self.certification_receipt,
            _CERTIFIED_ORBIT_CONSTRUCTION_TOKEN,
        )

    @property
    def inverse(self) -> TransportOrbitWitness:
        return self.inverted()

    def transport_state(self, state: AbstractState) -> AbstractState:
        return transport_state(state, self.transport)

    def transport_action(self, action: ActionCandidate) -> ActionCandidate | None:
        return transport_action(action, self.transport)

    def transport_prediction(self, packet: PredictionPacket) -> PredictionPacket:
        return transport_prediction(packet, self.transport)


def certify_transport(
    transport: TransportMap,
    source_projection: Any,
    target_projection: Any,
    *,
    inverse: TransportMap | None = None,
    stage: str = "before",
) -> TransportCertificate:
    return TransportCertificate.from_projections(
        transport,
        source_projection,
        target_projection,
        inverse=inverse,
        stage=stage,
    )


def _mapped_role(
    role: str,
    transport: TransportMap,
    *,
    drop_unmapped: bool,
) -> str | None:
    mapped = transport.map_role(role)
    if mapped is None and not drop_unmapped:
        return str(role).lower()
    return mapped


def _transport_fact(
    fact: GroundFact,
    transport: TransportMap,
    *,
    drop_unmapped: bool,
    grounded_ids: frozenset[str],
) -> GroundFact | None:
    # A fact-map entry is a predicate transport, never an authority to rename
    # branch-local grounding.  In particular, do not honor legacy full
    # ``GroundFact.key`` entries when the fact contains an entity reference.
    grounded_reference = any(term in grounded_ids for term in fact.terms) or (
        bool(fact.value) and fact.value in grounded_ids
    )
    exact = dict(transport.fact_map).get(fact.key.lower())
    if exact is not None and not grounded_reference:
        mapped_exact = GroundFact.from_key(exact)
        if not any(term in grounded_ids for term in mapped_exact.terms) and (
            not mapped_exact.value or mapped_exact.value not in grounded_ids
        ):
            return mapped_exact
    predicate = transport.map_fact(fact.predicate)
    if predicate is None:
        if drop_unmapped:
            return None
        predicate = fact.predicate
    terms: list[str] = []
    for term in fact.terms:
        if term in grounded_ids:
            terms.append(term)
            continue
        mapped = transport.map_role(term)
        if mapped is None:
            if drop_unmapped:
                return None
            mapped = term
        terms.append(mapped)
    value = (
        fact.value
        if fact.value in grounded_ids
        else transport.map_role(fact.value)
        if fact.value
        else fact.value
    )
    if value is None:
        if drop_unmapped:
            return None
        value = fact.value
    try:
        return GroundFact(predicate=predicate, terms=tuple(terms), value=value)
    except ValueError:
        return None


def transport_state(
    state: AbstractState,
    transport: TransportMap,
    *,
    drop_unmapped: bool = False,
) -> AbstractState:
    """Transport one state while preserving branch-local entity grounding."""

    grounded_ids = frozenset(entity.entity_id for entity in state.entities)
    entities = []
    for entity in state.entities:
        roles = tuple(
            mapped
            for role in entity.roles
            if (
                mapped := _mapped_role(
                    role,
                    transport,
                    drop_unmapped=drop_unmapped,
                )
            )
            is not None
        )
        entities.append(
            AbstractEntity(
                entity_id=entity.entity_id,
                roles=roles,
                attributes=entity.attributes,
                center=entity.center,
            )
        )

    def facts(values: Iterable[GroundFact]) -> frozenset[GroundFact]:
        return frozenset(
            mapped
            for fact in values
            if (
                mapped := _transport_fact(
                    fact,
                    transport,
                    drop_unmapped=drop_unmapped,
                    grounded_ids=grounded_ids,
                )
            )
            is not None
        )

    registers = tuple(
        (
            key,
            value
            if value in grounded_ids
            else transport.map_role(value) or transport.map_action(value) or value,
        )
        for key, value in state.registers
    )
    return AbstractState(
        entities=tuple(entities),
        true_facts=facts(state.true_facts),
        false_facts=facts(state.false_facts),
        counters=state.counters,
        registers=registers,
        topology=state.topology,
        regime_index=state.regime_index,
    )


def transport_action(
    action: ActionCandidate,
    transport: TransportMap,
) -> ActionCandidate | None:
    action_name = transport.map_action(action.action_name)
    if action_name is None:
        return None
    action_data: dict[str, Any] = {}
    for key, value in action.action_data.items():
        if isinstance(value, str):
            value = transport.map_role(value) or transport.map_action(value) or value
        action_data[str(key)] = value
    return ActionCandidate(action_name, action_data)


def _transport_event_key(key: str, transport: TransportMap) -> str | None:
    normalized = str(key).lower()
    exact = dict(transport.fact_map).get(normalized)
    if exact is not None:
        return exact
    prefix, separator, predicate = normalized.rpartition(":")
    if separator:
        mapped = transport.map_fact(predicate)
        if mapped is None:
            return None
        return f"{prefix}:{mapped}"
    return transport.map_fact(normalized)


def _prediction_symbol_domain(
    packet: PredictionPacket,
    channels: Iterable[str],
) -> frozenset[str]:
    requested = frozenset(str(channel).strip().lower() for channel in channels)
    values: list[Mapping[str, float]] = []
    if "objects" in requested:
        values.append(packet.object_deltas)
    if "relations" in requested:
        values.append(packet.relation_deltas)
    if "topology" in requested:
        values.append(packet.topology_deltas)
    tokens: set[str] = set()
    for mapping in values:
        for raw_key in mapping:
            normalized = str(raw_key).strip().lower()
            _prefix, separator, predicate = normalized.rpartition(":")
            tokens.add(_token("fact", predicate if separator else normalized))
    return frozenset(tokens)


def _domains_cover_dynamics(
    transport: TransportMap,
    inverse: TransportMap | None,
    *,
    source_tokens: frozenset[str],
    target_tokens: frozenset[str],
) -> bool:
    if inverse is None or not source_tokens or not target_tokens:
        return False
    if not source_tokens <= transport.domain or not target_tokens <= inverse.domain:
        return False
    for source in source_tokens:
        target = _map_domain_token(transport, source)
        if (
            target is None
            or target not in target_tokens
            or _map_domain_token(inverse, target) != source
        ):
            return False
    for target in target_tokens:
        source = _map_domain_token(inverse, target)
        if (
            source is None
            or source not in source_tokens
            or _map_domain_token(transport, source) != target
        ):
            return False
    return True


def _transport_delta(
    values: Mapping[str, float],
    transport: TransportMap,
) -> tuple[dict[str, float], bool]:
    output: dict[str, float] = {}
    complete = True
    for key, probability in values.items():
        mapped = _transport_event_key(key, transport)
        if mapped is None:
            complete = False
            continue
        output[mapped] = max(float(probability), output.get(mapped, 0.0))
    return output, complete


def transport_prediction(
    packet: PredictionPacket,
    transport: TransportMap,
) -> PredictionPacket:
    """Transport the symbolic portions of a prediction packet."""

    objects, objects_complete = _transport_delta(packet.object_deltas, transport)
    relations, relations_complete = _transport_delta(
        packet.relation_deltas,
        transport,
    )
    topology, topology_complete = _transport_delta(
        packet.topology_deltas,
        transport,
    )
    known = set(packet.known_channels)
    for channel, complete in (
        ("objects", objects_complete),
        ("relations", relations_complete),
        ("topology", topology_complete),
    ):
        if not complete:
            known.discard(channel)
    return PredictionPacket(
        object_deltas=objects,
        relation_deltas=relations,
        topology_deltas=topology,
        progress_mean=packet.progress_mean,
        progress_distribution=packet.progress_distribution,
        terminal_probability=packet.terminal_probability,
        goal_probability=packet.goal_probability,
        known_channels=frozenset(known),
        residual=packet.residual,
        state_after=(
            None
            if packet.state_after is None
            else transport_state(packet.state_after, transport)
        ),
    )


def _mapping_penalty(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return sum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
    ) / len(keys)


def _distribution_penalty(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return min(
        1.0,
        0.5
        * sum(
            abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
        ),
    )


def _optional_scalar_penalty(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 0.0 if left is right else 1.0
    return min(1.0, abs(float(left) - float(right)))


def _prediction_channel_penalty(
    transported: PredictionPacket,
    direct: PredictionPacket,
    channel: str,
) -> float:
    if channel == "objects":
        return _mapping_penalty(transported.object_deltas, direct.object_deltas)
    if channel == "relations":
        return _mapping_penalty(transported.relation_deltas, direct.relation_deltas)
    if channel == "topology":
        return _mapping_penalty(transported.topology_deltas, direct.topology_deltas)
    if channel == "progress":
        if transported.progress_distribution or direct.progress_distribution:
            return _distribution_penalty(
                transported.progress_distribution,
                direct.progress_distribution,
            )
        return _optional_scalar_penalty(
            transported.progress_mean,
            direct.progress_mean,
        )
    if channel == "terminal":
        return _optional_scalar_penalty(
            transported.terminal_probability,
            direct.terminal_probability,
        )
    if channel == "goal":
        return _optional_scalar_penalty(
            transported.goal_probability,
            direct.goal_probability,
        )
    raise ValueError(f"unknown prediction channel: {channel}")


def prediction_commutativity_penalty(
    transported: PredictionPacket,
    direct: PredictionPacket,
    comparable_channels: Iterable[str],
) -> float:
    """Mean bounded disagreement over explicitly comparable channels."""

    channels = tuple(
        sorted(
            {
                str(channel).strip().lower()
                for channel in comparable_channels
                if str(channel).strip().lower() in PREDICTION_CHANNELS
            }
        )
    )
    if not channels:
        return 1.0
    return sum(
        _prediction_channel_penalty(transported, direct, channel)
        for channel in channels
    ) / len(channels)


def _state_payload(state: AbstractState) -> dict[str, Any]:
    if _observer_state_payload is not None:
        return _observer_state_payload(state)
    return {
        "entities": sorted(
            (
                {
                    "roles": sorted(entity.roles),
                    "attributes": list(entity.attributes),
                }
                for entity in state.entities
            ),
            key=_canonical_json,
        ),
        "true_facts": sorted(fact.key for fact in state.true_facts),
        "false_facts": sorted(fact.key for fact in state.false_facts),
        "counters": list(state.counters),
        "registers": list(state.registers),
        "topology": list(state.topology),
        "regime_index": state.regime_index,
    }


def _state_channel_payload(state: AbstractState, channel: str) -> Any:
    payload = _state_payload(state)
    if channel == "facts":
        return {
            "true": payload["true_facts"],
            "false": payload["false_facts"],
        }
    if channel == "regime":
        return payload["regime_index"]
    return payload[channel]


@dataclass(frozen=True)
class CommutativityResult:
    """One fail-closed state or dynamics commutativity decision."""

    diagram: str
    comparable: bool
    compared_channels: frozenset[str]
    matched_channels: frozenset[str]
    mismatched_channels: frozenset[str]
    incomplete_channels: frozenset[str]
    penalty: float
    reason: str = ""
    transport_hash: str = ""

    def __post_init__(self) -> None:
        diagram = str(self.diagram).strip().lower()
        if diagram not in {"state", "dynamics"}:
            raise ValueError("commutativity diagram must be state or dynamics")
        object.__setattr__(self, "diagram", diagram)
        for attribute in (
            "compared_channels",
            "matched_channels",
            "mismatched_channels",
            "incomplete_channels",
        ):
            object.__setattr__(self, attribute, frozenset(getattr(self, attribute)))
        penalty = float(self.penalty)
        if not math.isfinite(penalty):
            raise ValueError("commutativity penalty must be finite")
        object.__setattr__(self, "penalty", max(0.0, min(1.0, penalty)))

    @property
    def commutes(self) -> bool:
        return bool(
            self.comparable and not self.mismatched_channels and self.penalty <= 1e-12
        )

    @property
    def score(self) -> float:
        if not self.comparable:
            return 0.0
        return max(0.0, 1.0 - self.penalty)

    @property
    def comparable_channels(self) -> frozenset[str]:
        return self.compared_channels

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": TRANSPORT_FORMAT_VERSION,
            "diagram": self.diagram,
            "comparable": self.comparable,
            "compared_channels": sorted(self.compared_channels),
            "matched_channels": sorted(self.matched_channels),
            "mismatched_channels": sorted(self.mismatched_channels),
            "incomplete_channels": sorted(self.incomplete_channels),
            "penalty": round(self.penalty, 12),
            "reason": self.reason,
            "transport_hash": self.transport_hash,
        }

    @property
    def canonical_hash(self) -> str:
        return _sha256(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash


@dataclass(frozen=True)
class CommutativityAudit:
    """Evaluate the two T10.2 commuting diagrams on shared channels."""

    transport: TransportMap
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("commutativity tolerance must be finite and non-negative")
        object.__setattr__(self, "tolerance", tolerance)

    def state(
        self,
        source_projection: Any,
        target_projection: Any,
        *,
        stage: str = "before",
        comparable_channels: Iterable[str] | None = None,
    ) -> CommutativityResult:
        source = _projection_state(source_projection, stage=stage)
        target = _projection_state(target_projection, stage=stage)
        available = (
            _projection_channels(source_projection)
            & _projection_channels(target_projection)
            & STATE_CHANNELS
        )
        requested = (
            available
            if comparable_channels is None
            else frozenset(str(item).strip().lower() for item in comparable_channels)
            & STATE_CHANNELS
        )
        shared = requested & available
        incomplete = set() if comparable_channels is None else set(requested - shared)
        if source is None or target is None:
            incomplete.update(requested & STATE_CHANNELS)
        certificate = None
        if source is not None and target is not None:
            certificate = certify_transport(
                self.transport,
                source_projection,
                target_projection,
                stage=stage,
            )
            missing_kinds = {
                _split_token(token)[0] for token in certificate.missing_domain
            }
            if "role" in missing_kinds:
                incomplete.add("entities")
            if "fact" in missing_kinds:
                incomplete.add("facts")
            if not certificate.frame_ids_match or certificate.ambiguous:
                incomplete.update(shared)
            if not certificate.projections_complete:
                incomplete.update(shared)
        if source is None or target is None or not shared:
            return CommutativityResult(
                diagram="state",
                comparable=False,
                compared_channels=frozenset(),
                matched_channels=frozenset(),
                mismatched_channels=frozenset(),
                incomplete_channels=frozenset(incomplete or requested),
                penalty=1.0,
                reason="missing state or shared projection channel",
                transport_hash=self.transport.canonical_hash,
            )
        transported = transport_state(source, self.transport)
        matched: set[str] = set()
        mismatched: set[str] = set()
        penalties: list[float] = []
        for channel in sorted(shared - incomplete):
            equal = _state_channel_payload(
                transported, channel
            ) == _state_channel_payload(
                target,
                channel,
            )
            (matched if equal else mismatched).add(channel)
            penalties.append(0.0 if equal else 1.0)
        comparable = bool(shared and not incomplete)
        penalty = sum(penalties) / len(penalties) if penalties else 1.0
        return CommutativityResult(
            diagram="state",
            comparable=comparable,
            compared_channels=frozenset(shared - incomplete),
            matched_channels=frozenset(matched),
            mismatched_channels=frozenset(mismatched),
            incomplete_channels=frozenset(incomplete),
            penalty=penalty,
            reason=("" if comparable else "incomplete frame transport"),
            transport_hash=self.transport.canonical_hash,
        )

    def audit_state(self, *args: Any, **kwargs: Any) -> CommutativityResult:
        return self.state(*args, **kwargs)

    def dynamics(
        self,
        source_prediction: PredictionPacket,
        target_prediction: PredictionPacket,
        *,
        source_projection: Any | None = None,
        target_projection: Any | None = None,
        source_action: ActionCandidate | None = None,
        target_action: ActionCandidate | None = None,
        comparable_channels: Iterable[str] | None = None,
    ) -> CommutativityResult:
        transported = transport_prediction(source_prediction, self.transport)
        requested = (
            set(source_prediction.known_channels & target_prediction.known_channels)
            if comparable_channels is None
            else {
                str(channel).strip().lower()
                for channel in comparable_channels
                if str(channel).strip().lower() in PREDICTION_CHANNELS
            }
        )
        shared = (
            requested
            & set(transported.known_channels)
            & set(target_prediction.known_channels)
        )
        incomplete = set(requested) - shared
        transport_valid = bool(
            self.transport.domain
            and self.transport.coverage >= 1.0
            and not self.transport.ambiguous
        )
        inverse = self.transport.inverted()
        transport_valid = bool(
            transport_valid
            and inverse is not None
            and self.transport.validate_round_trip(inverse)
            and inverse.validate_round_trip(self.transport)
        )
        if (source_projection is None) != (target_projection is None):
            transport_valid = False
        elif source_projection is not None and target_projection is not None:
            certificate = certify_transport(
                self.transport,
                source_projection,
                target_projection,
            )
            transport_valid = bool(transport_valid and certificate.exact)
        if not transport_valid:
            incomplete.update(requested or shared or {"transport"})

        selected_source_action = source_action or (
            _projection_action(source_projection)
            if source_projection is not None
            else None
        )
        selected_target_action = target_action or (
            _projection_action(target_projection)
            if target_projection is not None
            else None
        )
        source_symbols = set(_prediction_symbol_domain(source_prediction, requested))
        target_symbols = set(_prediction_symbol_domain(target_prediction, requested))
        if selected_source_action is not None:
            source_symbols.add(_token("action", selected_source_action.action_name))
        if selected_target_action is not None:
            target_symbols.add(_token("action", selected_target_action.action_name))
        if not _domains_cover_dynamics(
            self.transport,
            inverse,
            source_tokens=frozenset(source_symbols),
            target_tokens=frozenset(target_symbols),
        ):
            transport_valid = False
            incomplete.add("transport_domain")
        penalties: dict[str, float] = {
            channel: _prediction_channel_penalty(
                transported,
                target_prediction,
                channel,
            )
            for channel in shared
        }
        if selected_source_action is not None or selected_target_action is not None:
            if selected_source_action is None or selected_target_action is None:
                incomplete.add("action")
            else:
                mapped_action = transport_action(selected_source_action, self.transport)
                if mapped_action is None:
                    incomplete.add("action")
                else:
                    penalties["action"] = float(
                        mapped_action.action_name != selected_target_action.action_name
                    )
                    shared.add("action")

        if (
            source_prediction.state_after is not None
            and target_prediction.state_after is not None
        ):
            state_result = self.state(
                source_prediction.state_after,
                target_prediction.state_after,
            )
            if state_result.comparable:
                penalties["state_after"] = state_result.penalty
                shared.add("state_after")
            else:
                incomplete.add("state_after")

        matched = {
            channel
            for channel, penalty in penalties.items()
            if penalty <= self.tolerance
        }
        mismatched = set(penalties) - matched
        comparable = bool(transport_valid and shared and not incomplete)
        penalty = (
            sum(penalties.values()) / len(penalties)
            if penalties and comparable
            else 1.0
        )
        return CommutativityResult(
            diagram="dynamics",
            comparable=comparable,
            compared_channels=frozenset(penalties),
            matched_channels=frozenset(matched),
            mismatched_channels=frozenset(mismatched),
            incomplete_channels=frozenset(incomplete),
            penalty=penalty,
            reason=(
                ""
                if comparable
                else "no complete shared dynamics channels or certified transport"
            ),
            transport_hash=self.transport.canonical_hash,
        )

    def audit_dynamics(self, *args: Any, **kwargs: Any) -> CommutativityResult:
        return self.dynamics(*args, **kwargs)


def audit_state_commutativity(
    source_projection: Any,
    target_projection: Any,
    transport: TransportMap,
    **kwargs: Any,
) -> CommutativityResult:
    return CommutativityAudit(transport).state(
        source_projection,
        target_projection,
        **kwargs,
    )


def audit_dynamics_commutativity(
    source_prediction: PredictionPacket,
    target_prediction: PredictionPacket,
    transport: TransportMap,
    **kwargs: Any,
) -> CommutativityResult:
    return CommutativityAudit(transport).dynamics(
        source_prediction,
        target_prediction,
        **kwargs,
    )


def _prediction_payload(packet: PredictionPacket) -> dict[str, Any]:
    return {
        "object_deltas": dict(packet.object_deltas),
        "relation_deltas": dict(packet.relation_deltas),
        "topology_deltas": dict(packet.topology_deltas),
        "progress_mean": packet.progress_mean,
        "progress_distribution": dict(packet.progress_distribution),
        "terminal_probability": packet.terminal_probability,
        "goal_probability": packet.goal_probability,
        "known_channels": sorted(packet.known_channels),
        "residual": list(packet.residual),
        "state_after": (
            None if packet.state_after is None else _state_payload(packet.state_after)
        ),
    }


def canonical_signature(
    value: Any,
    transport: TransportMap | None = None,
    *,
    stage: str = "before",
) -> str:
    """Return a deterministic, frame-independent signature when mapped."""

    if isinstance(value, AbstractState):
        state = value if transport is None else transport_state(value, transport)
        payload: Any = {"kind": "state", "value": _state_payload(state)}
    elif isinstance(value, PredictionPacket):
        packet = value if transport is None else transport_prediction(value, transport)
        payload = {"kind": "prediction", "value": _prediction_payload(packet)}
    elif isinstance(value, ActionCandidate):
        action = value if transport is None else transport_action(value, transport)
        payload = {
            "kind": "action",
            "value": None
            if action is None
            else {
                "action_name": action.action_name,
                "action_data": dict(action.action_data),
            },
        }
    elif isinstance(value, TransportMap):
        payload = {"kind": "transport", "value": value.canonical_payload}
    else:
        state = _projection_state(value, stage=stage)
        if state is None:
            payload = {"kind": "generic", "value": _json_safe(value)}
        else:
            mapped_state = (
                state if transport is None else transport_state(state, transport)
            )
            action = _projection_action(value)
            if action is not None and transport is not None:
                action = transport_action(action, transport)
            payload = {
                "kind": "projection",
                "state": _state_payload(mapped_state),
                "action_name": None if action is None else action.action_name,
                "complete": _projection_complete(value),
                "covered_channels": sorted(_projection_channels(value)),
            }
    return _sha256(payload)


def gauge_equivalent(
    source: Any,
    target: Any,
    transport: TransportMap,
    *,
    comparable_channels: Iterable[str] | None = None,
) -> bool:
    """Return whether two complete projections differ only by ``transport``."""

    if isinstance(source, PredictionPacket) and isinstance(target, PredictionPacket):
        return (
            CommutativityAudit(transport)
            .dynamics(
                source,
                target,
                comparable_channels=comparable_channels,
            )
            .commutes
        )
    result = CommutativityAudit(transport).state(
        source,
        target,
        comparable_channels=comparable_channels,
    )
    if not result.commutes:
        return False
    source_action = _projection_action(source)
    target_action = _projection_action(target)
    if source_action is None and target_action is None:
        return True
    if source_action is None or target_action is None:
        return False
    mapped_action = transport_action(source_action, transport)
    return bool(
        mapped_action is not None
        and mapped_action.action_name == target_action.action_name
    )


def gauge_canonical_signature(
    value: Any,
    transport: TransportMap | None = None,
) -> str:
    return canonical_signature(value, transport)


def find_transport(
    transports: Iterable[TransportMap | TransportCertificate | TransportOrbitWitness]
    | Mapping[Any, TransportMap | TransportCertificate | TransportOrbitWitness],
    source_frame: str,
    target_frame: str,
) -> TransportMap | None:
    """Find the best direct transport, or an unambiguous reverse inverse."""

    source = _normalize_frame_id(source_frame)
    target = _normalize_frame_id(target_frame)
    values = transports.values() if isinstance(transports, Mapping) else transports
    materialized = tuple(
        item.transport
        if isinstance(item, (TransportCertificate, TransportOrbitWitness))
        else item
        for item in values
    )
    direct = [
        item
        for item in materialized
        if item.source_frame_id == source and item.target_frame_id == target
    ]
    if direct:
        return min(
            direct,
            key=lambda item: (-item.coverage, item.ambiguous, item.canonical_hash),
        )
    reverse = []
    for item in materialized:
        if item.source_frame_id == target and item.target_frame_id == source:
            inverse = item.inverted()
            if inverse is not None:
                reverse.append(inverse)
    if not reverse:
        return None
    return min(
        reverse,
        key=lambda item: (-item.coverage, item.ambiguous, item.canonical_hash),
    )


__all__ = [
    "PREDICTION_CHANNELS",
    "STATE_CHANNELS",
    "TRANSPORT_FORMAT_VERSION",
    "CommutativityAudit",
    "CommutativityResult",
    "TransportCertificate",
    "TransportMap",
    "TransportOrbitWitness",
    "audit_dynamics_commutativity",
    "audit_state_commutativity",
    "canonical_signature",
    "certify_transport",
    "find_transport",
    "gauge_canonical_signature",
    "gauge_equivalent",
    "persisted_attestation_receipt",
    "prediction_commutativity_penalty",
    "transport_action",
    "transport_prediction",
    "transport_state",
]
