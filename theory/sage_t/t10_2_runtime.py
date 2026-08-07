"""Fail-closed live adapters for the preregistered SAGE.T10.2 protocol.

Raw ARC frames and grounded action arguments exist only inside one call to the
runtime.  Persisted events contain structural projections, outcome summaries,
and audit certificates; they never contain grids, coordinates, colours, or
persistent entity identifiers.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .compact_quotient_v10_2 import (
    assert_compact_quotient,
    quotient_sha256,
    state_quotient_payload,
)
from .contracts import (
    AbstractState,
    ActionCandidate,
    ObservedTransition,
    PredictionPacket,
)
from .factorized_posterior_v10_2 import (
    FactorizedCandidateBank,
    FactorizedControlRefusal,
    FactorizedGaugeProgramPosterior,
    capacity_matched_factorized_bank,
)
from .frame_adapters_v10_2 import project_transition_with_frozen_frames
from .frame_transport_v10_2 import (
    CommutativityAudit,
    TransportMap,
    TransportOrbitWitness,
    certify_transport,
    persisted_attestation_receipt,
    transport_prediction,
)
from .gauge_inference_v10_2 import (
    GaugeDecisionEngine,
    GaugeProgramPosterior,
    JointGaugeHypothesis,
    rank_option_sequence_signatures,
)
from .mixed_automata_v10_2 import generate_mixed_grammar, repeat
from .observer_frames_v10_2 import (
    OBSERVER_FRAME_SPECS,
    FrameProjection,
    PhysicalEventBundle,
    audit_identity_leaks,
    canonical_sha256,
    merge_projection_and_physical_outcome,
    observer_frame_spec,
    state_model_payload,
)
from .t10_2_protocol import (
    AR25_GAME,
    BASELINE_COMMIT,
    BASELINE_FROZEN_SHA256,
    CHALLENGER_RECIPE_FILENAME,
    CONFIRMATION_SEEDS,
    CROSS_FIT_AUDIT_FILENAME,
    DISCOVERY_SEEDS,
    EVENT_FORMAT_VERSION,
    REGISTERED_SOURCE_CONTROLS,
    SOURCE_ACTIONS_PER_RESET,
    SOURCE_GAMES,
    SOURCE_RESETS_PER_GAME_SEED,
    VALIDATION_ACTIONS_PER_RESET,
    VALIDATION_EXEMPT_STOP_REASONS,
    VALIDATION_GAMES,
    VALIDATION_RESETS_PER_GAME_SEED,
    VALIDATION_SEEDS,
    VALIDATION_UNREGISTERED_STOP_REASONS,
    DataGateError,
    FirewallError,
    GateRefusalError,
    ManifestDriftError,
    RuntimeUnavailableError,
    artifact_descriptor,
    enforce_artifact_limit,
    enforce_environment_firewall,
    file_sha256,
    paired_bootstrap_lower,
    read_checked_json,
    read_cross_fit_audit,
    read_event_ledger,
    seal_event,
    signed_payload,
    validate_source_events,
    write_compact_json,
    write_event_ledger,
)

RUNTIME_FORMAT_VERSION = "sage-t10.2-runtime-v2"
MAXIMUM_MODEL_VIEW_BYTES = 32 * 1_024
MAXIMUM_COMPACT_EVENT_BYTES = 48 * 1_024
_SUMMARY_REPLAY_MISSING = ("endpoint_incidence_intentionally_omitted",)
_SUMMARY_REPLAY_STATE_CHANNELS = ("counters", "regime", "topology")
REPLAY_SPLIT = "frozen_source_replay_v4_3"
CHALLENGER_RECIPE_FORMAT_VERSION = "sage-t10.2-frozen-challenger-recipe-v1"
T10_1_FROZEN_MANIFEST_CHECKSUM = (
    "4d1c4dc8b62973187ea5e1c52e698652fdaeb424ae481b56baded0c0b2b9c1a3"
)
_TERMINAL_STATES = frozenset({"GAME_OVER", "WIN", "WON", "TERMINAL"})
_FRAME_IDS = frozenset(frame.frame_id for frame in OBSERVER_FRAME_SPECS)
_CHALLENGER_CODE_PATHS = (
    "theory/sage_t/contracts.py",
    "theory/sage_t/executor.py",
    "theory/sage_t/posterior.py",
    "theory/sage_t/compiler.py",
    "theory/sage_t/progress_witness_v10.py",
    "theory/sage_t/observer_frames_v10_2.py",
    "theory/sage_t/frame_adapters_v10_2.py",
    "theory/sage_t/frame_transport_v10_2.py",
    "theory/sage_t/mixed_automata_v10_2.py",
    "theory/sage_t/gauge_inference_v10_2.py",
    "theory/sage_t/compact_quotient_v10_2.py",
    "theory/sage_t/t10_2_protocol.py",
    "theory/sage_t/t10_2_runtime.py",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _clone(value: Any) -> Any:
    """Return a JSON-only copy, rejecting accidental runtime objects."""

    return json.loads(_canonical_json(value))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_size_bytes(value: Any) -> int:
    return len(_canonical_json(value).encode("utf-8"))


def _assert_compact_event_budget(event: Mapping[str, Any]) -> None:
    model_view = event.get("model_view")
    if not isinstance(model_view, Mapping):
        raise DataGateError("compact event requires a canonical model_view")
    model_bytes = _canonical_size_bytes(model_view)
    if model_bytes > MAXIMUM_MODEL_VIEW_BYTES:
        raise DataGateError(
            f"compact model_view exceeds {MAXIMUM_MODEL_VIEW_BYTES} bytes"
        )
    event_bytes = _canonical_size_bytes(event)
    if event_bytes > MAXIMUM_COMPACT_EVENT_BYTES:
        raise DataGateError(
            f"compact event exceeds {MAXIMUM_COMPACT_EVENT_BYTES} bytes"
        )
    sealed = dict(event) if "event_checksum" in event else seal_event(event)
    sealed_bytes = _canonical_size_bytes(sealed)
    if sealed_bytes > MAXIMUM_COMPACT_EVENT_BYTES:
        raise DataGateError(
            f"sealed compact event exceeds {MAXIMUM_COMPACT_EVENT_BYTES} bytes"
        )


def _summary_state(payload: Mapping[str, Any]) -> AbstractState:
    """Build only the aggregate, endpoint-free state carried by a v2 summary."""

    assert_compact_quotient(payload)
    return AbstractState(
        counters=tuple(
            (str(row["name"]), float(row["amount"])) for row in payload["counter_rows"]
        ),
        topology=tuple(
            (str(row["name"]), int(row["amount"])) for row in payload["topology_rows"]
        ),
        regime_index=int(payload["regime_index"]),
    )


def _finite_delta_mapping(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DataGateError(f"compact {label} must be a mapping")
    output: dict[str, float] = {}
    for raw_key, raw_amount in value.items():
        key = str(raw_key).strip().lower()
        if not key or isinstance(raw_amount, bool):
            raise DataGateError(f"compact {label} contains an invalid row")
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError) as exc:
            raise DataGateError(f"compact {label} must be numeric") from exc
        if not math.isfinite(amount):
            raise DataGateError(f"compact {label} must be finite")
        output[key] = amount
    return dict(sorted(output.items()))


def _structural_packet_from_payload(payload: Mapping[str, Any]) -> PredictionPacket:
    if not isinstance(payload, Mapping):
        raise DataGateError("compact structural observation must be a mapping")
    required = {
        "object_deltas",
        "relation_deltas",
        "topology_deltas",
        "known_channels",
        "residual",
    }
    if set(payload) != required:
        raise DataGateError("compact structural observation schema drifted")
    known_raw = payload["known_channels"]
    residual_raw = payload["residual"]
    if not isinstance(known_raw, list) or any(
        not isinstance(item, str) for item in known_raw
    ):
        raise DataGateError("compact structural known channels are invalid")
    known = frozenset(known_raw) & {"objects", "relations", "topology"}
    if list(known_raw) != sorted(set(known_raw)) or set(known_raw) != set(known):
        raise DataGateError("compact structural channels are not canonical")
    if not isinstance(residual_raw, list):
        raise DataGateError("compact structural residual must be a list")
    residual: list[float] = []
    for raw in residual_raw:
        if isinstance(raw, bool):
            raise DataGateError("compact structural residual must be numeric")
        try:
            amount = float(raw)
        except (TypeError, ValueError) as exc:
            raise DataGateError("compact structural residual must be numeric") from exc
        if not math.isfinite(amount):
            raise DataGateError("compact structural residual must be finite")
        residual.append(amount)
    return PredictionPacket(
        object_deltas=_finite_delta_mapping(
            payload["object_deltas"],
            label="object deltas",
        ),
        relation_deltas=_finite_delta_mapping(
            payload["relation_deltas"],
            label="relation deltas",
        ),
        topology_deltas=_finite_delta_mapping(
            payload["topology_deltas"],
            label="topology deltas",
        ),
        known_channels=known,
        residual=tuple(residual),
    )


@dataclass(frozen=True)
class SummaryReplayTransition:
    """Endpoint-free replay view accepted by ``PhysicalEventBundle``.

    The wrapped frame states carry only aggregate counters, topology and
    regime.  Structural deltas remain the persisted observation packet; no
    entity, fact incidence, or register value is invented during replay.
    """

    event_id: str
    before: FrameProjection
    after: FrameProjection
    structural_observation: PredictionPacket
    before_summary_hash: str
    after_summary_hash: str
    observation_hash: str

    def __post_init__(self) -> None:
        if not str(self.event_id).strip():
            raise ValueError("summary replay transition requires an event id")
        if self.before.stage != "before" or self.after.stage != "after":
            raise ValueError("summary replay transition stages are invalid")
        if self.before.frame_id != self.after.frame_id:
            raise ValueError("summary replay transition frames do not match")
        if self.before.action.key != self.after.action.key:
            raise ValueError("summary replay transition actions do not match")
        if self.before.complete or self.after.complete:
            raise ValueError("summary replay transition must remain incomplete")
        if (
            self.before.missing != _SUMMARY_REPLAY_MISSING
            or self.after.missing != _SUMMARY_REPLAY_MISSING
        ):
            raise ValueError("summary replay transition must declare omitted incidence")
        for label, digest in (
            ("before summary", self.before_summary_hash),
            ("after summary", self.after_summary_hash),
            ("observation", self.observation_hash),
        ):
            if len(str(digest)) != 64 or any(
                character not in "0123456789abcdef" for character in str(digest)
            ):
                raise ValueError(f"summary replay {label} hash is invalid")

    @property
    def frame_id(self) -> str:
        return self.before.frame_id

    @property
    def action(self) -> ActionCandidate:
        return self.before.action

    @property
    def complete(self) -> bool:
        return False

    @property
    def missing(self) -> tuple[str, ...]:
        return _SUMMARY_REPLAY_MISSING

    @property
    def covered_channels(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(_SUMMARY_REPLAY_STATE_CHANNELS)
                | (
                    set(self.structural_observation.known_channels)
                    & {"objects", "relations", "topology"}
                )
            )
        )

    @property
    def provenance(self) -> tuple[str, ...]:
        return ("compact_summary_replay",)

    @property
    def observation(self) -> PredictionPacket:
        return self.structural_observation

    @property
    def projection_outcome(self) -> PredictionPacket:
        return self.structural_observation

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": RUNTIME_FORMAT_VERSION,
            "kind": "endpoint_free_summary_replay",
            "frame_id": self.frame_id,
            "action_name": self.action.action_name,
            "before_summary_hash": self.before_summary_hash,
            "after_summary_hash": self.after_summary_hash,
            "observation_hash": self.observation_hash,
            "complete": False,
            "missing": list(self.missing),
            "covered_channels": list(self.covered_channels),
        }

    @property
    def canonical_hash(self) -> str:
        return _stable_hash(self.canonical_payload)

    @property
    def canonical_checksum(self) -> str:
        return self.canonical_hash

    def as_observed_transition(
        self,
        common_outcome: PredictionPacket,
        *,
        events: Sequence[str] = (),
        reset: bool = False,
    ) -> ObservedTransition:
        observation = merge_projection_and_physical_outcome(
            self.structural_observation,
            common_outcome,
            state_after=self.after.state,
        )
        return ObservedTransition(
            state_before=self.before.state,
            action=self.action,
            state_after=self.after.state,
            observation=observation,
            events=tuple(events),
            reset=bool(reset),
        )


def _invoke(
    function: Callable[..., Any],
    *,
    context: Mapping[str, Any],
    positional: Sequence[Any] = (),
) -> Any:
    """Invoke a dependency-injected hook without masking errors inside it."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*positional)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    accepted = {
        name: value
        for name, value in context.items()
        if accepts_kwargs or name in parameters
    }
    required_unknown = [
        parameter
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.name not in accepted
    ]
    if required_unknown:
        return function(*positional)
    return function(**accepted)


def _value(owner: Any, name: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
        return owner.get(name, default)
    return getattr(owner, name, default)


def _action_name(action: Any) -> str:
    if isinstance(action, str):
        return action.strip().upper()
    if isinstance(action, Mapping):
        raw = action.get("action_name", action.get("name", ""))
    else:
        raw = getattr(action, "action_name", getattr(action, "name", ""))
    return str(raw).strip().upper()


def _action_data(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        raw = action.get("action_data", action.get("action_args", {}))
    else:
        raw = getattr(action, "action_data", getattr(action, "action_args", {}))
    return dict(raw or {}) if isinstance(raw, Mapping) else {}


def _grounding_key(action: Any) -> str:
    return f"{_action_name(action)}:{_canonical_json(_action_data(action))}"


def _action_schema(action: Any) -> tuple[str, int]:
    """A coordinate-free capacity unit used by both confirmation arms."""

    return (_action_name(action), len(_action_data(action)))


def _snapshot_state(snapshot: Any) -> str:
    return str(_value(snapshot, "game_state", _value(snapshot, "state", ""))).upper()


def _snapshot_levels(snapshot: Any) -> int:
    try:
        return int(_value(snapshot, "levels_completed", 0))
    except (TypeError, ValueError):
        return 0


def _is_terminal(snapshot: Any) -> bool:
    return _snapshot_state(snapshot) in _TERMINAL_STATES


def _is_game_over(snapshot: Any) -> bool:
    return _snapshot_state(snapshot) in {"GAME_OVER", "LOST", "FAIL"}


class _LocalArcRuntime:
    """Lazy bridge to the repository's established local ARC runtime."""

    def __init__(self) -> None:
        self._helpers: tuple[Any, ...] | None = None

    def _load(self) -> tuple[Any, ...]:
        if self._helpers is None:
            # This import is deliberately delayed until an authorized lane runs.
            from .progress_witness_v10 import _live_helpers

            self._helpers = tuple(_live_helpers())
        return self._helpers

    def open(self, game_id: str, seed: int) -> Any:
        make_real_env, env_dir, *_ = self._load()
        environment = make_real_env(game_id, Path(env_dir()))
        for method_name in ("set_seed", "seed"):
            method = getattr(environment, method_name, None)
            if callable(method):
                _invoke(method, context={"seed": seed}, positional=(seed,))
                break
        return environment

    def reset(self, environment: Any) -> Any:
        return self._load()[3](environment)

    def legal_actions(self, environment: Any) -> tuple[Any, ...]:
        return tuple(self._load()[2](environment))

    def step(self, environment: Any, action: Any) -> Any:
        return self._load()[4](environment, action)

    def snapshot(
        self, frame: Any, fallback_available_actions: Sequence[Any] = ()
    ) -> Any:
        return self._load()[5](
            frame,
            fallback_available_actions=fallback_available_actions,
        )

    @staticmethod
    def close(environment: Any) -> None:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def _default_runtime_loader() -> _LocalArcRuntime:
    return _LocalArcRuntime()


def _runtime_method(runtime: Any, names: Sequence[str]) -> Callable[..., Any]:
    for name in names:
        method = getattr(runtime, name, None)
        if callable(method):
            return method
    raise RuntimeUnavailableError(f"runtime lacks required method: {'/'.join(names)}")


def _open_runtime(runtime: Any, game_id: str, seed: int) -> Any:
    method = _runtime_method(runtime, ("open", "make_env", "create", "make"))
    return _invoke(
        method,
        context={"game_id": game_id, "game": game_id, "seed": seed},
        positional=(game_id, seed),
    )


def _reset_runtime(runtime: Any, environment: Any) -> Any:
    method = _runtime_method(runtime, ("reset", "reset_env"))
    return _invoke(
        method,
        context={"environment": environment, "env": environment},
        positional=(environment,),
    )


def _legal_runtime(runtime: Any, environment: Any) -> tuple[Any, ...]:
    method = _runtime_method(runtime, ("legal_actions", "valid_actions", "actions"))
    result = _invoke(
        method,
        context={"environment": environment, "env": environment},
        positional=(environment,),
    )
    return tuple(result or ())


def _step_runtime(runtime: Any, environment: Any, action: Any) -> Any:
    method = _runtime_method(runtime, ("step", "take_action", "act"))
    return _invoke(
        method,
        context={"environment": environment, "env": environment, "action": action},
        positional=(environment, action),
    )


def _snapshot_runtime(
    runtime: Any,
    frame: Any,
    *,
    fallback_available_actions: Sequence[Any] = (),
) -> Any:
    method = _runtime_method(runtime, ("snapshot", "snapshot_frame"))
    return _invoke(
        method,
        context={
            "frame": frame,
            "fallback_available_actions": fallback_available_actions,
        },
        positional=(frame,),
    )


def _close_runtime(runtime: Any, environment: Any) -> None:
    method = next(
        (
            getattr(runtime, name)
            for name in ("close", "close_env")
            if callable(getattr(runtime, name, None))
        ),
        None,
    )
    if method is not None:
        _invoke(
            method,
            context={"environment": environment, "env": environment},
            positional=(environment,),
        )
        return
    close = getattr(environment, "close", None)
    if callable(close):
        close()


def _default_bundle(
    *,
    before: Any,
    after: Any,
    action: Any,
    legal_actions: Sequence[Any],
    event_id: str,
    step_index: int,
    game_id: str,
) -> PhysicalEventBundle:
    """Compile raw snapshots transiently, then project all frozen frames."""

    from theory.live_transition_loop import build_transition_record

    from .compiler import compile_transition_record

    before_grid = _value(before, "grid")
    after_grid = _value(after, "grid")
    if before_grid is None or after_grid is None:
        raise RuntimeUnavailableError("runtime snapshots must expose a grid")
    record = build_transition_record(
        action=_action_name(action),
        action_args=_action_data(action),
        grid_before=before_grid,
        grid_after=after_grid,
        available_actions=tuple(_action_name(item) for item in legal_actions),
        game_state_before=_snapshot_state(before) or "NOT_FINISHED",
        game_state_after=_snapshot_state(after) or "NOT_FINISHED",
        levels_completed_before=_snapshot_levels(before),
        levels_completed_after=_snapshot_levels(after),
        timestamp=step_index,
    )
    evidence = compile_transition_record(record, source_game_id=game_id)
    return project_transition_with_frozen_frames(evidence, event_id=event_id)


def _make_bundle(
    builder: Callable[..., Any] | None,
    *,
    before: Any,
    after: Any,
    action: Any,
    legal_actions: Sequence[Any],
    event_id: str,
    step_index: int,
    game_id: str,
) -> PhysicalEventBundle:
    context = {
        "before": before,
        "after": after,
        "action": action,
        "action_name": _action_name(action),
        "action_data": _action_data(action),
        "legal_actions": tuple(legal_actions),
        "event_id": event_id,
        "step_index": step_index,
        "game_id": game_id,
    }
    raw = (
        _default_bundle(**context)
        if builder is None
        else _invoke(
            builder, context=context, positional=(before, after, action, event_id)
        )
    )
    if isinstance(raw, ObservedTransition):
        raw = project_transition_with_frozen_frames(raw, event_id=event_id)
    if not isinstance(raw, PhysicalEventBundle):
        raise DataGateError(
            "bundle builder must return ObservedTransition or PhysicalEventBundle"
        )
    if raw.event_id != event_id:
        raise DataGateError("bundle builder changed the registered physical event id")
    if frozenset(raw.frame_ids) != _FRAME_IDS or len(raw.projections) != 4:
        raise DataGateError(
            "every event requires exactly the four frozen frame projections"
        )
    event_names = {str(item).strip().casefold() for item in raw.events}
    changed = any(
        projection.before.canonical_hash != projection.after.canonical_hash
        for projection in raw.projections
    )
    if changed or "state_change" in event_names:
        event_names.add("state_changed")
    else:
        event_names.add("no_effect")
    return replace(raw, events=tuple(sorted(event_names)))


def _structural_observation_payload(packet: PredictionPacket) -> dict[str, Any]:
    return {
        "object_deltas": dict(sorted(packet.object_deltas.items())),
        "relation_deltas": dict(sorted(packet.relation_deltas.items())),
        "topology_deltas": dict(sorted(packet.topology_deltas.items())),
        "known_channels": sorted(
            set(packet.known_channels) & {"objects", "relations", "topology"}
        ),
        "residual": list(packet.residual),
    }


def _entity_descriptor(entity: Any) -> tuple[Any, ...]:
    return (tuple(sorted(entity.roles)), tuple(sorted(entity.attributes)))


def _correspondence(bundle: PhysicalEventBundle) -> dict[str, Any]:
    confident = 0
    ambiguous = 0
    total = 0
    for projection in bundle.projections:
        before = Counter(
            _entity_descriptor(item) for item in projection.before.state.entities
        )
        after = Counter(
            _entity_descriptor(item) for item in projection.after.state.entities
        )
        total += max(
            len(projection.before.state.entities), len(projection.after.state.entities)
        )
        for descriptor in set(before) | set(after):
            overlap = min(before[descriptor], after[descriptor])
            if before[descriptor] == after[descriptor] == 1:
                confident += overlap
            elif overlap:
                ambiguous += overlap
    denominator = max(1, total)
    # An empty structural view carries no correspondence evidence and is
    # therefore represented as one fully ambiguous sentinel trial.  Both QA
    # fractions remain auditable numerator/denominator ratios.
    fully_ambiguous = ambiguous if total else denominator
    return {
        "method": "unique_structural_descriptor",
        "confident_matches": confident,
        "ambiguous_matches": ambiguous,
        "fully_ambiguous_matches": fully_ambiguous,
        "entities_considered": total,
        "fraction_denominator": denominator,
        "confident_fraction": confident / denominator,
        "fully_ambiguous_fraction": fully_ambiguous / denominator,
    }


def _projection_symbols(projection: Any) -> tuple[set[str], set[str], set[str]]:
    states = (projection.before.state, projection.after.state)
    roles = {
        role for state in states for entity in state.entities for role in entity.roles
    }
    facts = {
        fact.predicate
        for state in states
        for fact in set(state.true_facts) | set(state.false_facts)
    }
    for mapping in (
        projection.observation.object_deltas,
        projection.observation.relation_deltas,
        projection.observation.topology_deltas,
    ):
        for key in mapping:
            normalized = str(key).strip().lower()
            _prefix, separator, predicate = normalized.rpartition(":")
            facts.add(predicate if separator else normalized)
    return roles, facts, {projection.action.action_name}


def _summary_symbols(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    action_name: str,
) -> tuple[set[str], set[str], set[str]]:
    assert_compact_quotient(before)
    assert_compact_quotient(after)
    roles = {
        str(role)
        for payload in (before, after)
        for row in payload["role_rows"]
        for role in row["roles"]
    }
    facts = {
        str(row["predicate"])
        for payload in (before, after)
        for row in payload["fact_rows"]
    }
    action = str(action_name).strip().upper()
    if not action:
        raise DataGateError("compact summary vocabulary requires an action")
    return roles, facts, {action}


def _frame_summary_symbols(
    frame_payload: Mapping[str, Any],
    action_name: str,
) -> tuple[set[str], set[str], set[str]]:
    before = frame_payload.get("before")
    after = frame_payload.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise DataGateError("compact frame lacks before/after summaries")
    return _summary_symbols(before, after, action_name)


def _transported_summary(
    payload: Mapping[str, Any],
    transport: TransportMap,
) -> dict[str, Any] | None:
    assert_compact_quotient(payload)
    role_rows: list[dict[str, Any]] = []
    for row in payload["role_rows"]:
        mapped_roles: list[str] = []
        for role in row["roles"]:
            token = f"role:{role}"
            mapped = transport.map_role(str(role))
            if token not in transport.domain or mapped is None:
                return None
            mapped_roles.append(mapped)
        role_rows.append({"roles": sorted(mapped_roles), "count": int(row["count"])})
    role_rows.sort(key=_canonical_json)
    fact_rows: list[dict[str, Any]] = []
    for row in payload["fact_rows"]:
        predicate = str(row["predicate"])
        mapped = transport.map_fact(predicate)
        if f"fact:{predicate}" not in transport.domain or mapped is None:
            return None
        fact_rows.append(
            {
                "truth": bool(row["truth"]),
                "predicate": mapped,
                "arity": int(row["arity"]),
                "has_literal": bool(row["has_literal"]),
                "count": int(row["count"]),
            }
        )
    fact_rows.sort(key=_canonical_json)
    return {
        "entity_count": int(payload["entity_count"]),
        "fact_count": int(payload["fact_count"]),
        "role_rows": role_rows,
        "fact_rows": fact_rows,
        "counter_rows": payload["counter_rows"],
        "register_rows": payload["register_rows"],
        "topology_rows": payload["topology_rows"],
        "regime_index": int(payload["regime_index"]),
    }


def _summary_comparison_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    assert_compact_quotient(payload)
    return {
        key: payload[key]
        for key in (
            "entity_count",
            "fact_count",
            "role_rows",
            "fact_rows",
            "counter_rows",
            "register_rows",
            "topology_rows",
            "regime_index",
        )
    }


def _summary_transport_commutes(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    transport: TransportMap,
) -> bool:
    transported = _transported_summary(source, transport)
    return transported is not None and transported == _summary_comparison_payload(
        target
    )


def _observation_symbol_tokens(payload: Mapping[str, Any]) -> frozenset[str]:
    tokens: set[str] = set()
    for field in ("object_deltas", "relation_deltas", "topology_deltas"):
        mapping = payload.get(field)
        if not isinstance(mapping, Mapping):
            return frozenset()
        for raw_key in mapping:
            normalized = str(raw_key).strip().lower()
            _prefix, separator, predicate = normalized.rpartition(":")
            tokens.add(f"fact:{predicate if separator else normalized}")
    return frozenset(tokens)


def _observation_transport_commutes(
    source_payload: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    transport: TransportMap,
) -> bool:
    source = _structural_packet_from_payload(source_payload)
    target = _structural_packet_from_payload(target_payload)
    inverse = transport.inverted()
    if inverse is None:
        return False
    source_tokens = _observation_symbol_tokens(source_payload)
    target_tokens = _observation_symbol_tokens(target_payload)
    if not source_tokens <= transport.domain or not target_tokens <= inverse.domain:
        return False
    for token in source_tokens:
        kind, _separator, symbol = token.partition(":")
        mapped = transport.map_symbol(kind, symbol)
        if mapped is None or f"{kind}:{mapped}" not in target_tokens:
            return False
    for token in target_tokens:
        kind, _separator, symbol = token.partition(":")
        mapped = inverse.map_symbol(kind, symbol)
        if mapped is None or f"{kind}:{mapped}" not in source_tokens:
            return False
    common = set(source.known_channels) & set(target.known_channels)
    if not common and (source_tokens or target_tokens):
        return False

    def restricted(packet: PredictionPacket) -> PredictionPacket:
        return PredictionPacket(
            object_deltas=(packet.object_deltas if "objects" in common else {}),
            relation_deltas=(packet.relation_deltas if "relations" in common else {}),
            topology_deltas=(packet.topology_deltas if "topology" in common else {}),
            known_channels=frozenset(common),
            residual=packet.residual,
        )

    transported = transport_prediction(restricted(source), transport)
    return _structural_observation_payload(
        transported
    ) == _structural_observation_payload(restricted(target))


def _summary_pair_commutes(
    source_frame: Mapping[str, Any],
    target_frame: Mapping[str, Any],
    transport: TransportMap,
    *,
    action_name: str,
) -> bool:
    source_before = source_frame.get("before")
    source_after = source_frame.get("after")
    target_before = target_frame.get("before")
    target_after = target_frame.get("after")
    source_observation = source_frame.get("observation")
    target_observation = target_frame.get("observation")
    values = (
        source_before,
        source_after,
        target_before,
        target_after,
        source_observation,
        target_observation,
    )
    if any(not isinstance(value, Mapping) for value in values):
        return False
    mapped_action = transport.map_action(action_name)
    if f"action:{action_name}" not in transport.domain or mapped_action != action_name:
        return False
    return bool(
        _summary_transport_commutes(source_before, target_before, transport)
        and _summary_transport_commutes(source_after, target_after, transport)
        and _observation_transport_commutes(
            source_observation,
            target_observation,
            transport,
        )
    )


def _structural_transport_map(source: Any, target: Any) -> TransportMap:
    source_roles, source_facts, source_actions = _projection_symbols(source)
    target_roles, target_facts, target_actions = _projection_symbols(target)
    return TransportMap(
        source_frame_id=source.frame_id,
        target_frame_id=target.frame_id,
        role_map=tuple((item, item) for item in sorted(source_roles & target_roles)),
        fact_map=tuple((item, item) for item in sorted(source_facts & target_facts)),
        action_map=tuple(
            (item, item) for item in sorted(source_actions & target_actions)
        ),
        domain=frozenset(
            [*(f"role:{item}" for item in source_roles)]
            + [*(f"fact:{item}" for item in source_facts)]
            + [*(f"action:{item}" for item in source_actions)]
        ),
    )


def _transport_evidence(
    bundle: PhysicalEventBundle,
    frames: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = bundle.projection("root_only")
    certificates: list[dict[str, Any]] = []
    persisted_orbits: list[dict[str, Any]] = []
    exact_flags: list[bool] = []
    commuting_flags: list[bool] = []
    summary_flags: list[bool] = []
    for target in bundle.projections:
        transport = _structural_transport_map(root, target)
        audit = CommutativityAudit(transport)
        before_certificate = certify_transport(
            transport,
            root,
            target,
            stage="before",
        )
        after_certificate = certify_transport(
            transport,
            root,
            target,
            stage="after",
        )
        identity_self = target.frame_id == root.frame_id
        if identity_self:
            before_result = audit.state(
                root.before.state,
                root.before.state,
            )
            after_result = audit.state(
                root.after.state,
                root.after.state,
            )
            dynamics_result = audit.dynamics(
                root.observation,
                root.observation,
                source_action=root.action,
                target_action=root.action,
            )
        else:
            before_result = audit.state(root, target, stage="before")
            after_result = audit.state(root, target, stage="after")
            dynamics_result = audit.dynamics(
                root.observation,
                target.observation,
                source_projection=root,
                target_projection=target,
            )
        commutes = bool(
            before_result.commutes
            and after_result.commutes
            and dynamics_result.commutes
        )
        live_exact = bool(
            commutes
            and (
                identity_self
                or (
                    before_certificate.certifies_gauge_equivalence
                    and after_certificate.certifies_gauge_equivalence
                )
            )
        )
        summary_commutes = _summary_pair_commutes(
            frames[root.frame_id],
            frames[target.frame_id],
            transport,
            action_name=bundle.action.action_name,
        )
        replay_exact = bool(live_exact and summary_commutes)
        exact_flags.append(replay_exact)
        commuting_flags.append(commutes)
        summary_flags.append(summary_commutes)
        certificate_hash = _stable_hash(
            {
                "kind": "staged_live_transport_certificate_v2",
                "before": before_certificate.canonical_hash,
                "after": after_certificate.canonical_hash,
                "dynamics": dynamics_result.canonical_hash,
            }
        )
        certificates.append(
            {
                "source_frame": root.frame_id,
                "target_frame": target.frame_id,
                "transport_hash": transport.canonical_hash,
                "certificate_hash": certificate_hash,
                "coverage": min(
                    before_certificate.coverage,
                    after_certificate.coverage,
                ),
                "exact": replay_exact,
                "comparable": replay_exact,
                "mapping_kind": "exact" if replay_exact else "partial",
                "round_trip_exact": bool(
                    before_certificate.round_trip_exact
                    and after_certificate.round_trip_exact
                ),
                "certifies_gauge_equivalence": replay_exact,
                "projection_complete": root.complete and target.complete,
                "live_graph_exact_attested": live_exact,
                "summary_commutative_exact": summary_commutes,
                "commutativity": {
                    "before": before_result.canonical_hash,
                    "after": after_result.canonical_hash,
                    "dynamics": dynamics_result.canonical_hash,
                    "exact": commutes,
                },
            }
        )
        if live_exact and target.frame_id != root.frame_id:
            witness = TransportOrbitWitness.from_live_evidence(
                before_certificate,
                after_certificate,
                source_domain=transport.mapped_domain,
                target_domain=transport.codomain,
                dynamics_commutes=dynamics_result.commutes,
            )
            attestation: dict[str, Any] = {
                "certificate_hash": certificate_hash,
                "source_before_summary_hash": frames[root.frame_id]["before_hash"],
                "source_after_summary_hash": frames[root.frame_id]["after_hash"],
                "target_before_summary_hash": frames[target.frame_id]["before_hash"],
                "target_after_summary_hash": frames[target.frame_id]["after_hash"],
                "source_observation_hash": frames[root.frame_id]["observation_hash"],
                "target_observation_hash": frames[target.frame_id]["observation_hash"],
                "live_graph_exact_attested": True,
                "round_trip_exact": True,
                "summary_commutative_exact": summary_commutes,
            }
            attestation["receipt"] = persisted_attestation_receipt(
                orbit_hash=witness.canonical_hash,
                source_frame=root.frame_id,
                target_frame=target.frame_id,
                attestation=attestation,
            )
            persisted_orbits.append(
                {
                    "orbit_payload": witness.canonical_payload,
                    "orbit_hash": witness.canonical_hash,
                    "source_frame": root.frame_id,
                    "target_frame": target.frame_id,
                    "attestation": attestation,
                }
            )
    permutation_invariant = True
    for projection in bundle.projections:
        for frame_projection in (projection.before, projection.after):
            reversed_state = replace(
                frame_projection.state,
                entities=tuple(reversed(frame_projection.state.entities)),
            )
            permutation_invariant = permutation_invariant and (
                state_model_payload(frame_projection.state)
                == state_model_payload(reversed_state)
            )
    exact = bool(exact_flags and all(exact_flags) and all(commuting_flags))
    exact_count = sum(
        certificate["exact"] and certificate["commutativity"]["exact"]
        for certificate in certificates
    )
    summary = {
        "mapping_kind": "exact" if exact else "partial",
        "comparable": exact,
        "round_trip_exact": exact and all(exact_flags),
        "entity_permutation_invariant": permutation_invariant,
        "commutative_exact": exact and all(commuting_flags),
        "live_graph_exact_attested": bool(commuting_flags and all(commuting_flags)),
        "summary_commutative_exact": bool(summary_flags and all(summary_flags)),
        "certificate_count": len(certificates),
        "exact_certificate_count": exact_count,
        "partial_certificate_count": len(certificates) - exact_count,
        "identity_root_certificate_exact": any(
            item["source_frame"] == "root_only"
            and item["target_frame"] == "root_only"
            and item["exact"]
            and item["round_trip_exact"]
            and item["commutativity"]["exact"]
            for item in certificates
        ),
    }
    return summary, certificates, persisted_orbits


def _event_labels(bundle: PhysicalEventBundle) -> dict[str, bool]:
    names = {str(item).casefold() for item in bundle.events}
    projection_observations = [item.observation for item in bundle.projections]
    changed = any(
        projection.before.canonical_hash != projection.after.canonical_hash
        for projection in bundle.projections
    )
    return {
        "state_changed": changed,
        "no_effect": "no_effect" in names or not changed,
        "relation_changed": any(
            observation.relation_deltas for observation in projection_observations
        ),
        "topology_changed": any(
            observation.topology_deltas for observation in projection_observations
        ),
        "progress": "progress" in names
        or float(bundle.common_outcome.progress_mean or 0.0) > 0.0,
        "level_complete": "level_complete" in names,
        "game_over": "game_over" in names,
        "goal": float(bundle.common_outcome.goal_probability or 0.0) >= 0.5,
    }


def _compact_event(
    bundle: PhysicalEventBundle,
    *,
    controller: str,
    reset_index: int,
    step_index: int,
    progressing_sequence_rank: int | None,
    donor_game_count: int,
    capacity_slots: int,
) -> dict[str, Any]:
    frames: dict[str, dict[str, Any]] = {}
    for projection in bundle.projections:
        before_quotient = state_quotient_payload(projection.before.state)
        after_quotient = state_quotient_payload(projection.after.state)
        observation = _structural_observation_payload(projection.observation)
        frames[projection.frame_id] = {
            "before": before_quotient,
            "after": after_quotient,
            "before_hash": quotient_sha256(before_quotient),
            "after_hash": quotient_sha256(after_quotient),
            "observation": observation,
            "observation_hash": _stable_hash(observation),
            "complete": projection.complete,
            "missing": list(projection.missing),
            "covered_channels": list(projection.covered_channels),
            "provenance": list(projection.provenance),
        }
    model_view = {"frames": frames}
    if _canonical_size_bytes(model_view) > MAXIMUM_MODEL_VIEW_BYTES:
        raise DataGateError(
            f"compact model_view exceeds {MAXIMUM_MODEL_VIEW_BYTES} bytes"
        )
    transport, certificates, persisted_orbits = _transport_evidence(
        bundle,
        frames,
    )
    projection_rows: dict[str, dict[str, Any]] = {}
    for projection in bundle.projections:
        frame = frames[projection.frame_id]
        safe_projection = {
            "format_version": RUNTIME_FORMAT_VERSION,
            "frame_id": projection.frame_id,
            "before_hash": frame["before_hash"],
            "after_hash": frame["after_hash"],
            "observation": frame["observation"],
            "observation_hash": frame["observation_hash"],
            "complete": projection.complete,
            "missing": list(projection.missing),
            "covered_channels": list(projection.covered_channels),
            "provenance": list(projection.provenance),
        }
        projection_rows[projection.frame_id] = {
            **safe_projection,
            "canonical_hash": _stable_hash(safe_projection),
        }
    labels = _event_labels(bundle)
    nonterminal = not labels["game_over"] and not labels["level_complete"]
    event = {
        "format_version": EVENT_FORMAT_VERSION,
        "event_id": bundle.event_id,
        "reset_index": reset_index,
        "step_index": step_index,
        "action": {
            "executed": True,
            "schema": "local_primitive",
            "name": bundle.action.action_name,
            "data": {"parameter_arity": len(bundle.action.action_data)},
        },
        "model_view": model_view,
        "outcome": {
            "progression": float(bundle.common_outcome.progress_mean or 0.0),
            "terminal": bool(
                bundle.common_outcome.terminal_probability is not None
                and bundle.common_outcome.terminal_probability >= 0.5
            ),
            "goal": bool(
                bundle.common_outcome.goal_probability is not None
                and bundle.common_outcome.goal_probability >= 0.5
            ),
        },
        "projections": projection_rows,
        "labels": labels,
        "learned_predicates": ["no_effect", "state_changed"],
        "correspondence": _correspondence(bundle),
        "transport": transport,
        "transport_certificates": certificates,
        "transport_orbits": persisted_orbits,
        "prefix": {
            "nonterminal": nonterminal,
            "evaluable": bool(bundle.common_outcome.known_channels),
            "coherent_frames": sum(
                projection.complete for projection in bundle.projections
            ),
        },
        "selection": {
            "controller": controller,
            "reset_index": reset_index,
            "step_index": step_index,
            "action_name": bundle.action.action_name,
            "parameter_arity": len(bundle.action.action_data),
            "progressing_sequence_rank": progressing_sequence_rank,
            "donor_game_count": donor_game_count,
            "capacity_slots": capacity_slots,
            "legal_grounding": True,
        },
        "provenance": {
            "collector": RUNTIME_FORMAT_VERSION,
            "projector_bank": [frame.frame_id for frame in OBSERVER_FRAME_SPECS],
            "summary_hashes": [
                frames[frame_id][stage]
                for frame_id in sorted(frames)
                for stage in ("before_hash", "after_hash")
            ],
            "observation_hashes": [
                frames[frame_id]["observation_hash"] for frame_id in sorted(frames)
            ],
            "transport_orbit_hashes": [item["orbit_hash"] for item in persisted_orbits],
            "transport_certificate_hashes": [
                item["certificate_hash"] for item in certificates
            ],
            "physical_outcome_known_channels": sorted(
                set(bundle.common_outcome.known_channels)
                & {"progress", "terminal", "goal"}
            ),
            "raw_runtime_state_retained": False,
        },
    }
    leaks = audit_identity_leaks(
        event["model_view"],
        forbidden_game_ids=(*SOURCE_GAMES, *VALIDATION_GAMES, AR25_GAME),
    )
    if leaks:
        raise FirewallError(f"compact model view leaked identity: {leaks[0]}")
    compact = _clone(event)
    _assert_compact_event_budget(compact)
    return compact


def _physical_packet_from_event(event: Mapping[str, Any]) -> PredictionPacket:
    raw = event.get("outcome", {})
    outcome = raw if isinstance(raw, Mapping) else {}
    progress_mean = outcome.get(
        "progression", outcome.get("progress_mean", outcome.get("progress"))
    )
    try:
        progress = None if progress_mean is None else float(progress_mean)
    except (TypeError, ValueError):
        progress = None
    distribution = outcome.get("progress_distribution", {})
    if not isinstance(distribution, Mapping):
        distribution = {}
    terminal = outcome.get("terminal_probability", outcome.get("terminal"))
    goal = outcome.get("goal_probability", outcome.get("goal"))
    known = set(outcome.get("known_channels", ())) & {
        "progress",
        "terminal",
        "goal",
    }
    if progress is not None:
        known.add("progress")
    if terminal is not None:
        known.add("terminal")
    if goal is not None:
        known.add("goal")
    return PredictionPacket(
        progress_mean=progress,
        progress_distribution={
            str(key): float(value) for key, value in distribution.items()
        },
        terminal_probability=None if terminal is None else float(terminal),
        goal_probability=None if goal is None else float(goal),
        known_channels=frozenset(known),
        residual=tuple(float(item) for item in outcome.get("residual", ())),
    )


def _bundle_from_compact_event(event: Mapping[str, Any]) -> PhysicalEventBundle | None:
    """Recover posterior evidence without recovering any raw ARC observation."""

    model_view = event.get("model_view", {})
    frames = model_view.get("frames", {}) if isinstance(model_view, Mapping) else {}
    if not isinstance(frames, Mapping) or set(frames) != _FRAME_IDS:
        return None
    selection = event.get("selection", {})
    if not isinstance(selection, Mapping):
        return None
    action_name = str(selection.get("action_name", "")).strip().upper()
    if not action_name:
        return None
    action = ActionCandidate(action_name)
    event_id = str(event.get("event_id", ""))
    projections: list[SummaryReplayTransition] = []
    for frame_id in sorted(frames):
        raw = frames[frame_id]
        if not isinstance(raw, Mapping):
            return None
        before_payload = raw.get("before")
        after_payload = raw.get("after")
        if not isinstance(before_payload, Mapping) or not isinstance(
            after_payload, Mapping
        ):
            return None
        try:
            before_state = _summary_state(before_payload)
            after_state = _summary_state(after_payload)
        except (TypeError, ValueError):
            return None
        before_hash = str(raw.get("before_hash", ""))
        after_hash = str(raw.get("after_hash", ""))
        if before_hash != quotient_sha256(
            before_payload
        ) or after_hash != quotient_sha256(after_payload):
            return None
        observation_payload = raw.get("observation")
        if not isinstance(observation_payload, Mapping):
            return None
        try:
            observation = _structural_packet_from_payload(observation_payload)
        except (TypeError, ValueError, DataGateError):
            return None
        observation_hash = str(raw.get("observation_hash", ""))
        if observation_hash != _stable_hash(observation_payload):
            return None
        frame = observer_frame_spec(str(frame_id))
        before = FrameProjection(
            frame=frame,
            state=before_state,
            action=action,
            stage="before",
            complete=False,
            missing=_SUMMARY_REPLAY_MISSING,
            covered_channels=_SUMMARY_REPLAY_STATE_CHANNELS,
            provenance=("compact_summary_replay",),
            audit_tags=("endpoint_free_summary",),
        )
        after = FrameProjection(
            frame=frame,
            state=after_state,
            action=action,
            stage="after",
            complete=False,
            missing=_SUMMARY_REPLAY_MISSING,
            covered_channels=_SUMMARY_REPLAY_STATE_CHANNELS,
            provenance=("compact_summary_replay",),
            audit_tags=("endpoint_free_summary",),
        )
        projections.append(
            SummaryReplayTransition(
                event_id=event_id,
                before=before,
                after=after,
                structural_observation=observation,
                before_summary_hash=before_hash,
                after_summary_hash=after_hash,
                observation_hash=observation_hash,
            )
        )
    labels = event.get("labels", {})
    events = tuple(
        sorted(
            str(name)
            for name, value in (labels.items() if isinstance(labels, Mapping) else ())
            if value is True
        )
    )
    return PhysicalEventBundle(
        event_id=event_id,
        action=action,
        common_outcome=_physical_packet_from_event(event),
        projections=tuple(projections),  # type: ignore[arg-type]
        events=events,
    )


def _sequence_groups(
    events: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[str, ...], bool]]:
    groups: dict[tuple[str, int, int], list[tuple[int, str, bool]]] = defaultdict(list)
    for event in events:
        selection = event.get("selection", {})
        if not isinstance(selection, Mapping):
            continue
        try:
            key = (
                str(event.get("game_id", "")),
                int(event.get("seed", -1)),
                int(selection.get("reset_index", -1)),
            )
            step = int(selection.get("step_index", -1))
        except (TypeError, ValueError):
            continue
        name = str(selection.get("action_name", "")).strip().upper()
        if key[2] < 0 or step < 0 or not name:
            continue
        labels = event.get("labels", {})
        progressed = bool(
            isinstance(labels, Mapping)
            and (labels.get("progress") or labels.get("level_complete"))
        )
        groups[key].append((step, name, progressed))
    output: list[tuple[tuple[str, ...], bool]] = []
    for rows in groups.values():
        ordered = sorted(rows)
        output.append(
            (tuple(name for _, name, _ in ordered), any(row[2] for row in ordered))
        )
    return output


def _observed_orbit_witnesses(
    events: Sequence[Mapping[str, Any]],
) -> tuple[TransportOrbitWitness, ...]:
    """Validate receipt-bound summary orbits without inventing a live graph."""

    candidates: dict[str, list[TransportOrbitWitness]] = defaultdict(list)
    for event in events:
        raw_orbits = event.get("transport_orbits", ())
        if not isinstance(raw_orbits, (list, tuple)):
            raise DataGateError("compact transport_orbits must be a sequence")
        if not raw_orbits:
            continue
        model_view = event.get("model_view")
        frames = model_view.get("frames") if isinstance(model_view, Mapping) else None
        selection = event.get("selection")
        action_name = (
            str(selection.get("action_name", "")).strip().upper()
            if isinstance(selection, Mapping)
            else ""
        )
        if (
            not isinstance(frames, Mapping)
            or set(frames) != _FRAME_IDS
            or not action_name
        ):
            raise DataGateError("compact orbit event lacks bound frame summaries")
        for envelope in raw_orbits:
            try:
                witness = TransportOrbitWitness.from_persisted_attestation(envelope)
            except (TypeError, ValueError) as exc:
                raise DataGateError("invalid persisted transport attestation") from exc
            source_frame = witness.source_frame_id
            target_frame = witness.target_frame_id
            source = frames.get(source_frame)
            target = frames.get(target_frame)
            if not isinstance(source, Mapping) or not isinstance(target, Mapping):
                raise DataGateError("persisted transport frames are unavailable")
            attestation = envelope["attestation"]
            expected_hashes = {
                "source_before_summary_hash": source.get("before_hash"),
                "source_after_summary_hash": source.get("after_hash"),
                "target_before_summary_hash": target.get("before_hash"),
                "target_after_summary_hash": target.get("after_hash"),
                "source_observation_hash": source.get("observation_hash"),
                "target_observation_hash": target.get("observation_hash"),
            }
            if any(attestation[key] != value for key, value in expected_hashes.items()):
                raise DataGateError("persisted transport summary hash binding drifted")
            for frame_payload in (source, target):
                before = frame_payload.get("before")
                after = frame_payload.get("after")
                observation = frame_payload.get("observation")
                if (
                    not isinstance(before, Mapping)
                    or not isinstance(after, Mapping)
                    or not isinstance(observation, Mapping)
                    or frame_payload.get("before_hash") != quotient_sha256(before)
                    or frame_payload.get("after_hash") != quotient_sha256(after)
                    or frame_payload.get("observation_hash")
                    != _stable_hash(observation)
                ):
                    raise DataGateError("persisted transport summary evidence drifted")
            source_roles, source_facts, source_actions = _frame_summary_symbols(
                source,
                action_name,
            )
            target_roles, target_facts, target_actions = _frame_summary_symbols(
                target,
                action_name,
            )
            source_domain = frozenset(
                [*(f"role:{item}" for item in source_roles)]
                + [*(f"fact:{item}" for item in source_facts)]
                + [*(f"action:{item}" for item in source_actions)]
                + list(_observation_symbol_tokens(source["observation"]))
            )
            target_domain = frozenset(
                [*(f"role:{item}" for item in target_roles)]
                + [*(f"fact:{item}" for item in target_facts)]
                + [*(f"action:{item}" for item in target_actions)]
                + list(_observation_symbol_tokens(target["observation"]))
            )
            if (
                witness.certified_source_domain != source_domain
                or witness.certified_target_domain != target_domain
                or witness.transport.mapped_domain != source_domain
                or witness.transport.codomain != target_domain
            ):
                raise DataGateError("persisted transport vocabulary coverage drifted")
            revalidated = _summary_pair_commutes(
                source,
                target,
                witness.transport,
                action_name=action_name,
            )
            if revalidated is not attestation["summary_commutative_exact"]:
                raise DataGateError("persisted summary commutativity drifted")
            if not revalidated:
                continue
            candidates[witness.canonical_hash].append(witness)

    selected: list[TransportOrbitWitness] = []
    for orbit_hash in sorted(candidates):
        # Every admitted candidate is exact, total and unambiguous.  Receipt is
        # the deterministic final tie-breaker for duplicate event attestations.
        selected.append(
            min(
                candidates[orbit_hash],
                key=lambda item: (
                    -len(item.certified_source_domain),
                    item.transport.ambiguity,
                    item.certification_receipt,
                ),
            )
        )
    return tuple(selected)


def _synthesize_gauge_candidates(
    events: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 256,
) -> tuple[JointGaugeHypothesis, ...]:
    """Compile the bounded mixed grammar from executed source sequences only.

    The global source vocabulary is capped at four primitive schemas.  Each
    mixed option uses one or two of them, as required by the immutable option
    automaton contract, two learned predicates, and a horizon of at most 16.
    Stable structural transports are induced from the compact projections;
    declared-but-unmapped symbols remain in ``domain`` as explicit partiality.
    """

    from .progress_witness_v10 import compile_progress_program

    limit = max(1, min(256, int(maximum)))
    sequences = sorted(_sequence_groups(events), key=lambda item: (item[0], item[1]))
    global_actions = tuple(
        sorted({action for actions, _ in sequences for action in actions})[:4]
    )
    orbit_witnesses = _observed_orbit_witnesses(events)
    symbols: dict[str, dict[str, set[str]]] = {
        frame.frame_id: {"role": set(), "fact": set(), "action": set()}
        for frame in OBSERVER_FRAME_SPECS
    }
    for event in events:
        model_view = event.get("model_view")
        frames = model_view.get("frames") if isinstance(model_view, Mapping) else None
        selection = event.get("selection")
        action_name = (
            str(selection.get("action_name", "")).strip().upper()
            if isinstance(selection, Mapping)
            else ""
        )
        if not isinstance(frames, Mapping) or not action_name:
            continue
        for raw_frame_id, frame_payload in frames.items():
            frame_id = str(raw_frame_id)
            if frame_id not in symbols or not isinstance(frame_payload, Mapping):
                continue
            roles, facts, actions = _frame_summary_symbols(
                frame_payload,
                action_name,
            )
            symbols[frame_id]["role"].update(roles)
            symbols[frame_id]["fact"].update(facts)
            symbols[frame_id]["action"].update(actions)

    transports_by_frame: dict[str, tuple[TransportMap, ...]] = {}
    for source in OBSERVER_FRAME_SPECS:
        source_symbols = symbols[source.frame_id]
        transports: list[TransportMap] = []
        for target in OBSERVER_FRAME_SPECS:
            if target.frame_id == source.frame_id:
                continue
            target_symbols = symbols[target.frame_id]
            common = {
                kind: source_symbols[kind] & target_symbols[kind]
                for kind in ("role", "fact", "action")
            }
            transports.append(
                TransportMap(
                    source_frame_id=source.frame_id,
                    target_frame_id=target.frame_id,
                    role_map=tuple((item, item) for item in sorted(common["role"])),
                    fact_map=tuple((item, item) for item in sorted(common["fact"])),
                    action_map=tuple((item, item) for item in sorted(common["action"])),
                    domain=frozenset(
                        f"{kind}:{item}"
                        for kind in ("role", "fact", "action")
                        for item in source_symbols[kind]
                    ),
                )
            )
        transports_by_frame[source.frame_id] = tuple(transports)

    candidates: dict[str, JointGaugeHypothesis] = {}
    for actions, observed_progress in sequences:
        if not actions:
            continue
        unique_actions = tuple(
            action for action in dict.fromkeys(actions) if action in global_actions
        )[:4]
        if not unique_actions:
            continue
        option_basis = list(unique_actions[:2])
        option_basis.extend(
            action for action in global_actions if action not in option_basis
        )
        option_basis = option_basis[:2]
        program_actions = tuple(dict.fromkeys((*unique_actions, *option_basis)))[:4]
        option_actions = tuple(action.lower() for action in option_basis)
        horizon = max(2 if len(option_basis) > 1 else 1, min(16, len(actions)))
        options = generate_mixed_grammar(
            option_actions,
            predicates=("state_changed", "no_effect"),
            prime_counts=tuple(count for count in (1, 2) if count < horizon),
            termination_predicate="level_complete",
            noop_termination_predicates=("state_changed", "no_effect"),
            maximum_horizon=horizon,
        )
        for positive in (observed_progress, not observed_progress):
            action_sets = tuple(dict.fromkeys((unique_actions, program_actions)))
            for action_set in action_sets:
                program = compile_progress_program(
                    sequence_length=horizon,
                    action_names=action_set,
                    positive=positive,
                )
                bound = {name.lower() for name in action_set}
                for option in options:
                    if not set(option.action_schemas) <= bound:
                        continue
                    for witness in orbit_witnesses:
                        forward_frame = observer_frame_spec(witness.source_frame_id)
                        inverse = witness.inverted()
                        reverse_frame = observer_frame_spec(inverse.source_frame_id)
                        for orbit_frame, orbit_transport in (
                            (forward_frame, witness),
                            (reverse_frame, inverse),
                        ):
                            hypothesis = JointGaugeHypothesis(
                                world_program=program,
                                frame=orbit_frame,
                                transports=(orbit_transport,),
                                option=option,
                            )
                            candidates.setdefault(hypothesis.canonical_hash, hypothesis)
                            if len(candidates) >= limit:
                                return tuple(candidates.values())
                    for frame in OBSERVER_FRAME_SPECS:
                        hypothesis = JointGaugeHypothesis(
                            world_program=program,
                            frame=frame,
                            transports=transports_by_frame[frame.frame_id],
                            option=option,
                        )
                        candidates.setdefault(hypothesis.canonical_hash, hypothesis)
                        if len(candidates) >= limit:
                            return tuple(candidates.values())
    return tuple(candidates.values())


def _fit_compact_posterior(
    events: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[JointGaugeHypothesis] = (),
    posterior_factory: Callable[..., Any] = GaugeProgramPosterior,
    maximum_candidates: int = 256,
) -> tuple[Any | None, tuple[JointGaugeHypothesis, ...], int, list[str]]:
    bank: Sequence[JointGaugeHypothesis]
    if isinstance(candidates, FactorizedCandidateBank):
        bank = candidates
    else:
        bank = tuple(candidates)
    if not bank:
        bank = _synthesize_gauge_candidates(events, maximum=maximum_candidates)
    if not bank:
        return None, (), 0, ["no_executed_sequence_candidates"]
    posterior = posterior_factory()
    posterior.seed(bank)
    observations = 0
    errors: list[str] = []
    branch: tuple[str, int, int] | None = None
    for event in events:
        try:
            bundle = _bundle_from_compact_event(event)
            if bundle is None:
                continue
            selection = event.get("selection", {})
            current_branch = (
                str(event.get("game_id", "")),
                int(event.get("seed", -1)),
                int(
                    selection.get("reset_index", -1)
                    if isinstance(selection, Mapping)
                    else -1
                ),
            )
            if branch is not None and current_branch != branch:
                start_branch = getattr(posterior, "start_branch", None)
                if callable(start_branch):
                    start_branch()
            branch = current_branch
            posterior.observe(bundle)
            observations += 1
        except Exception as exc:  # noqa: BLE001 - recorded as negative evidence.
            errors.append(type(exc).__name__)
    # Donor evidence fixes posterior masses; a held-out reset starts every
    # option from its frozen initial cursor without using held-out outcomes.
    start_branch = getattr(posterior, "start_branch", None)
    if observations and callable(start_branch):
        start_branch()
    return posterior, bank, observations, errors


def _challenger_code_binding(manifest: Mapping[str, Any]) -> dict[str, str]:
    declared = manifest.get("code_sha256")
    if not isinstance(declared, Mapping):
        raise ManifestDriftError("manifest lacks challenger code bindings")
    missing = [path for path in _CHALLENGER_CODE_PATHS if path not in declared]
    if missing:
        raise ManifestDriftError(
            f"manifest lacks challenger code binding: {missing[0]}"
        )
    root = Path(__file__).resolve().parents[2]
    binding = {path: str(declared[path]) for path in _CHALLENGER_CODE_PATHS}
    current = {path: file_sha256(root / path) for path in _CHALLENGER_CODE_PATHS}
    if current != binding:
        raise ManifestDriftError("challenger source code drifted after freeze")
    return binding


def _posterior_mass_fingerprint(posterior: Any) -> str:
    classes = sorted(
        (
            {
                "key": str(item.key),
                "probability": round(float(item.probability), 15),
                "variants": len(item.particles),
            }
            for item in getattr(posterior, "classes", ())
        ),
        key=lambda item: item["key"],
    )
    return canonical_sha256(
        {
            "classes": classes,
            "residual_mass": round(float(getattr(posterior, "residual_mass", 0.0)), 15),
        }
    )


def _projectable_event_binding(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_ids = [
        str(event.get("event_id", ""))
        for event in events
        if _bundle_from_compact_event(event) is not None
    ]
    return {
        "count": len(event_ids),
        "ordered_event_id_digest": canonical_sha256(event_ids),
    }


def _write_challenger_recipe(
    *,
    manifest: Mapping[str, Any],
    output_dir: Path,
    fresh_path: Path,
    replay_path: Path,
    fresh: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    candidates: Sequence[JointGaugeHypothesis],
    posterior: Any,
    posterior_observations: int,
) -> dict[str, Any]:
    """Persist a compact immutable recipe, never a validation-trained model."""

    code_binding = _challenger_code_binding(manifest)
    candidate_hashes = [candidate.canonical_hash for candidate in candidates]
    if not candidate_hashes or len(candidate_hashes) > 256:
        raise DataGateError("challenger recipe requires 1..256 source candidates")
    recipe = signed_payload(
        {
            "format_version": CHALLENGER_RECIPE_FORMAT_VERSION,
            "kind": "immutable_source_posterior_recipe",
            "manifest_checksum": manifest["manifest_checksum"],
            "code_sha256": code_binding,
            "ledgers": {
                "source_events": {
                    "path": "source_events.jsonl",
                    "artifact": artifact_descriptor(fresh_path),
                    "projectable": _projectable_event_binding(fresh),
                },
                "replay_events": {
                    "path": "replay_events.jsonl",
                    "artifact": artifact_descriptor(replay_path),
                    "projectable": _projectable_event_binding(replay),
                },
            },
            "candidate_bank": {
                "grammar_input": "source_events_only",
                "maximum_candidates": 256,
                "canonical_hashes": candidate_hashes,
                "grammar_retuned": False,
                "prior_retuned": False,
            },
            "posterior_fit": {
                "evidence_order": ["source_events", "replay_events"],
                "observation_count": int(posterior_observations),
                "class_count": len(getattr(posterior, "classes", ())),
                "mass_fingerprint": _posterior_mass_fingerprint(posterior),
            },
            "validation_authority": {
                "candidate_mutation": False,
                "grammar_mutation": False,
                "prior_mutation": False,
                "bayesian_observe_update_replan": True,
            },
        },
        checksum_key="recipe_checksum",
    )
    recipe_path = output_dir / CHALLENGER_RECIPE_FILENAME
    write_compact_json(recipe_path, recipe)
    enforce_artifact_limit(recipe_path, kind="checkpoint")
    return {
        "bound": True,
        "path": CHALLENGER_RECIPE_FILENAME,
        "artifact": artifact_descriptor(recipe_path),
        "recipe_checksum": recipe["recipe_checksum"],
    }


def _verify_recipe_ledger(
    recipe: Mapping[str, Any],
    *,
    output_dir: Path,
    key: str,
) -> tuple[Path, list[dict[str, Any]]]:
    ledgers = recipe.get("ledgers")
    entry = ledgers.get(key) if isinstance(ledgers, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ManifestDriftError(f"challenger recipe lacks ledger: {key}")
    expected_name = (
        "source_events.jsonl" if key == "source_events" else "replay_events.jsonl"
    )
    if entry.get("path") != expected_name:
        raise ManifestDriftError(f"challenger recipe ledger path drifted: {key}")
    path = output_dir / expected_name
    if entry.get("artifact") != artifact_descriptor(path):
        raise ManifestDriftError(f"challenger recipe ledger drifted: {key}")
    events = read_event_ledger(path)
    if entry.get("projectable") != _projectable_event_binding(events):
        raise ManifestDriftError(f"challenger projectable evidence drifted: {key}")
    return path, events


def _rebuild_challenger_posterior(
    *,
    recipe: Mapping[str, Any],
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[GaugeProgramPosterior, tuple[JointGaugeHypothesis, ...]]:
    if recipe.get("format_version") != CHALLENGER_RECIPE_FORMAT_VERSION:
        raise ManifestDriftError("unsupported challenger recipe format")
    if recipe.get("kind") != "immutable_source_posterior_recipe":
        raise ManifestDriftError("unsupported challenger recipe kind")
    if recipe.get("manifest_checksum") != manifest.get("manifest_checksum"):
        raise ManifestDriftError("challenger recipe/manifest binding drifted")
    if recipe.get("code_sha256") != _challenger_code_binding(manifest):
        raise ManifestDriftError("challenger recipe code binding drifted")
    _fresh_path, fresh = _verify_recipe_ledger(
        recipe,
        output_dir=output_dir,
        key="source_events",
    )
    _replay_path, replay = _verify_recipe_ledger(
        recipe,
        output_dir=output_dir,
        key="replay_events",
    )
    validate_source_events(fresh, manifest=manifest, replay=False)
    validate_source_events(replay, manifest=manifest, replay=True)
    validate_source_events([*fresh, *replay], manifest=manifest, replay=None)
    bank_spec = recipe.get("candidate_bank")
    fit_spec = recipe.get("posterior_fit")
    if not isinstance(bank_spec, Mapping) or not isinstance(fit_spec, Mapping):
        raise ManifestDriftError("challenger recipe lacks bank or posterior fit")
    if (
        bank_spec.get("grammar_input") != "source_events_only"
        or bank_spec.get("maximum_candidates") != 256
        or bank_spec.get("grammar_retuned") is not False
        or bank_spec.get("prior_retuned") is not False
    ):
        raise ManifestDriftError("challenger grammar/prior recipe drifted")
    bank = _synthesize_gauge_candidates(fresh, maximum=256)
    hashes = [candidate.canonical_hash for candidate in bank]
    if hashes != bank_spec.get("canonical_hashes"):
        raise ManifestDriftError("challenger candidate bank did not reconstruct")
    posterior, rebuilt_bank, observations, errors = _fit_compact_posterior(
        [*fresh, *replay],
        candidates=bank,
        maximum_candidates=256,
    )
    if posterior is None or errors:
        raise GateRefusalError("challenger source posterior did not reconstruct")
    expected_fit = {
        "evidence_order": ["source_events", "replay_events"],
        "observation_count": observations,
        "class_count": len(posterior.classes),
        "mass_fingerprint": _posterior_mass_fingerprint(posterior),
    }
    if dict(fit_spec) != expected_fit:
        raise ManifestDriftError("challenger posterior reconstruction drifted")
    return posterior, rebuilt_bank


def _posterior_action_scores(posterior: Any) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = defaultdict(float)
    for particle in getattr(posterior, "particles", ()):
        hypothesis = particle.hypothesis
        bindings = hypothesis.world_program.action_bindings
        option = hypothesis.option
        state = getattr(particle, "option_state", None)
        if state is None:
            new_execution = getattr(option, "new_execution", None)
            if callable(new_execution):
                state = new_execution()
            else:
                initial = getattr(option, "initial_state", None)
                state = initial() if callable(initial) else initial
        allowed_method = getattr(option, "allowed_action_schemas", None)
        try:
            allowed_lower = (
                {str(item).strip().lower() for item in allowed_method(state)}
                if callable(allowed_method)
                else set()
            )
        except (KeyError, TypeError, ValueError):
            allowed_lower = set()
        for binding in bindings:
            action_name = str(binding.action_name).upper()
            action_schema = action_name.lower()
            option_weight = (
                1.0
                if not allowed_lower
                or "*" in allowed_lower
                or action_schema in allowed_lower
                else 0.25
            )
            scores[(action_name, 0)] += float(particle.probability) * option_weight
            # Parameter arity is branch-local and not learned; apply the same
            # posterior score to parameterized legal groundings at decision time.
    return dict(scores)


def _capacity_matched_independent_bank(
    candidates: Sequence[JointGaugeHypothesis],
) -> FactorizedCandidateBank:
    """Build the audited outcome-independent ``q(D)q(G)q(F)q(Tau)q(A)`` bank."""

    try:
        return capacity_matched_factorized_bank(
            candidates,
            particle_budget=256,
        )
    except FactorizedControlRefusal as exc:
        raise DataGateError(f"factorized independent posterior refused: {exc}") from exc


def _effective_posterior_capacity(posterior: Any | None) -> tuple[int, int]:
    if posterior is None:
        return 0, 0
    classes = tuple(getattr(posterior, "classes", ()))
    particles = getattr(posterior, "particles", None)
    if particles is None:
        particle_count = sum(
            len(tuple(getattr(candidate_class, "particles", ())))
            for candidate_class in classes
        )
    else:
        particle_count = len(tuple(particles))
    return (
        particle_count,
        len(classes),
    )


class T10_2SourceFactory:
    """Stateful, lazy source factory implementing the frozen two-stage plan."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any] | None = None,
        runtime_loader: Callable[[], Any] | None = None,
        bundle_builder: Callable[..., Any] | None = None,
    ) -> None:
        self._runtime_loader = runtime_loader or _default_runtime_loader
        self._bundle_builder = bundle_builder
        self.manifest_checksum = str((manifest or {}).get("manifest_checksum", ""))
        self._runtime: Any | None = None
        self._discovery_events: dict[str, list[dict[str, Any]]] = {
            game: [] for game in SOURCE_GAMES
        }
        self._discovery_counts: dict[str, Counter[tuple[str, int]]] = {
            game: Counter() for game in SOURCE_GAMES
        }
        self._cross_fit_audit: list[dict[str, Any]] = []

    @property
    def runtime_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def discovery_events(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            game: tuple(_clone(row) for row in rows)
            for game, rows in self._discovery_events.items()
        }

    @property
    def cross_fit_audit(self) -> tuple[dict[str, Any], ...]:
        return tuple(_clone(row) for row in self._cross_fit_audit)

    def _load_runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_loader()
        if self._runtime is None:
            raise RuntimeUnavailableError(
                "runtime loader returned no local ARC runtime"
            )
        return self._runtime

    def __call__(
        self,
        game_id: str,
        seed: int,
        phase: str = "collect",
        split: str | None = None,
        held_out_game: str | None = None,
        training_games: Sequence[str] = (),
    ) -> _SourceLaneEnvironment:
        # The firewall runs before the lazy runtime loader can be touched.
        enforce_environment_firewall(phase=phase, game_id=game_id)
        if split == "discovery":
            if int(seed) not in DISCOVERY_SEEDS or held_out_game is not None:
                raise DataGateError("invalid discovery lane")
        elif split == "leave_one_game_out_confirmation":
            if int(seed) not in CONFIRMATION_SEEDS or held_out_game != game_id:
                raise DataGateError("invalid confirmation lane")
            expected = tuple(game for game in SOURCE_GAMES if game != game_id)
            if tuple(training_games) != expected:
                raise FirewallError(
                    "confirmation donors must be exactly the other two games"
                )
            if any(not self._discovery_events[game] for game in expected):
                raise GateRefusalError(
                    "confirmation requires completed discovery donors"
                )
        else:
            raise DataGateError(f"unregistered source split: {split}")
        return _SourceLaneEnvironment(
            factory=self,
            game_id=game_id,
            seed=int(seed),
            split=str(split),
            held_out_game=held_out_game,
            training_games=tuple(training_games),
        )

    def _remember_discovery(self, game_id: str, event: Mapping[str, Any]) -> None:
        # _compact_event has already discarded all raw frames and groundings.
        compact = _clone(event)
        self._discovery_events[game_id].append(compact)
        selection = compact.get("selection", {})
        schema = (
            str(selection.get("action_name", "")),
            int(selection.get("parameter_arity", 0)),
        )
        self._discovery_counts[game_id][schema] += 1

    def _donor_prior(
        self, held_out_game: str, training_games: Sequence[str]
    ) -> Counter[tuple[str, int]]:
        donors = tuple(training_games)
        expected = tuple(game for game in SOURCE_GAMES if game != held_out_game)
        if donors != expected or held_out_game in donors:
            raise FirewallError(
                "held-out discovery evidence entered its own cross-fit fold"
            )
        prior: Counter[tuple[str, int]] = Counter()
        for game in donors:
            for event in self._discovery_events[game]:
                selection = event.get("selection", {})
                prior[
                    (
                        str(selection.get("action_name", "")),
                        int(selection.get("parameter_arity", 0)),
                    )
                ] += 1
        return prior

    def _donor_posterior_scores(
        self, training_games: Sequence[str]
    ) -> tuple[
        Any | None,
        tuple[JointGaugeHypothesis, ...],
        tuple[Mapping[str, Any], ...],
        dict[tuple[str, int], float],
        dict[str, Any],
    ]:
        donor_events = tuple(
            event for game in training_games for event in self._discovery_events[game]
        )
        posterior, candidates, observations, errors = _fit_compact_posterior(
            donor_events
        )
        scores = {} if posterior is None else _posterior_action_scores(posterior)
        metadata = {
            "posterior_used": posterior is not None,
            "posterior_candidates": len(candidates),
            "posterior_classes": len(
                {candidate.gauge_equivalence_key for candidate in candidates}
            ),
            "posterior_observations": observations,
            "posterior_errors": len(errors),
            "option_conditioned": bool(scores),
        }
        return posterior, candidates, donor_events, scores, metadata


class _SourceLaneEnvironment:
    def __init__(
        self,
        *,
        factory: T10_2SourceFactory,
        game_id: str,
        seed: int,
        split: str,
        held_out_game: str | None,
        training_games: tuple[str, ...],
    ) -> None:
        self.factory = factory
        self.game_id = game_id
        self.seed = seed
        self.split = split
        self.held_out_game = held_out_game
        self.training_games = training_games
        self._environment: Any | None = None

    def _controller_order(self, resets: int) -> tuple[str, ...]:
        if self.split == "discovery":
            return ("balanced_discovery",) * resets
        if resets != SOURCE_RESETS_PER_GAME_SEED:
            raise DataGateError("confirmation requires exactly four resets")
        forward = (
            "learned",
            "capacity_matched_independent",
            "capacity_matched_independent",
            "learned",
        )
        return forward if self.seed % 2 == 0 else tuple(reversed(forward))

    def collect_events(
        self,
        *,
        game_id: str,
        seed: int,
        split: str,
        held_out_game: str | None,
        training_games: Sequence[str],
        resets: int,
        action_budget: int,
        stop_on_progress: bool,
        stop_on_game_over: bool,
    ) -> list[dict[str, Any]]:
        if (game_id, int(seed), split, held_out_game, tuple(training_games)) != (
            self.game_id,
            self.seed,
            self.split,
            self.held_out_game,
            self.training_games,
        ):
            raise FirewallError(
                "source environment context drifted after authorization"
            )
        if not 1 <= int(resets) <= SOURCE_RESETS_PER_GAME_SEED:
            raise DataGateError("source resets exceed the registered bound")
        if not 1 <= int(action_budget) <= SOURCE_ACTIONS_PER_RESET:
            raise DataGateError("source action budget exceeds 64")
        if stop_on_progress is not True or stop_on_game_over is not True:
            raise DataGateError("source collection must stop on progress and GAME_OVER")

        runtime = self.factory._load_runtime()
        self._environment = _open_runtime(runtime, self.game_id, self.seed)
        controllers = self._controller_order(int(resets))
        donor_prior: Counter[tuple[str, int]] = Counter()
        donor_posterior: Any | None = None
        donor_candidates: tuple[JointGaugeHypothesis, ...] = ()
        donor_events: tuple[Mapping[str, Any], ...] = ()
        posterior_scores: dict[tuple[str, int], float] = {}
        posterior_metadata: dict[str, Any] = {}
        if self.split != "discovery":
            donor_prior = self.factory._donor_prior(
                self.game_id,
                self.training_games,
            )
            (
                donor_posterior,
                donor_candidates,
                donor_events,
                posterior_scores,
                posterior_metadata,
            ) = self.factory._donor_posterior_scores(self.training_games)
        capacity_slots = int(
            posterior_metadata.get("posterior_candidates", len(donor_prior))
        )
        independent_candidates: (
            FactorizedCandidateBank | tuple[JointGaugeHypothesis, ...]
        ) = ()
        independent_refusal = ""
        if self.split != "discovery":
            try:
                independent_candidates = _capacity_matched_independent_bank(
                    donor_candidates
                )
            except DataGateError as exc:
                # The control remains an attempted negative control.  Preserve
                # the active collection schedule and surface the refusal in
                # every compact control record instead of aborting other lanes.
                independent_refusal = str(exc)
        events: list[dict[str, Any]] = []
        independent_counts: Counter[tuple[str, int]] = Counter()
        learned_counts: Counter[tuple[str, int]] = Counter()
        grounding_counts: Counter[str] = Counter()
        learned_reset_count = 0
        online_held_out_observations = 0
        independent_observations = 0
        reset_audit: list[dict[str, Any]] = []

        for reset_index, controller in enumerate(controllers):
            active_posterior: Any | None = None
            reset_error_count = 0
            controller_frame_states: dict[str, AbstractState] = {}
            sequence_prefix: list[
                tuple[
                    ActionCandidate,
                    tuple[str, ...],
                    Mapping[str, AbstractState],
                ]
            ] = []
            if controller == "learned":
                if learned_reset_count == 0:
                    active_posterior = donor_posterior
                    reset_error_count += int(
                        posterior_metadata.get("posterior_errors", 0)
                    )
                else:
                    (
                        active_posterior,
                        _bank,
                        _observations,
                        fit_errors,
                    ) = _fit_compact_posterior(
                        donor_events,
                        candidates=donor_candidates,
                        maximum_candidates=256,
                    )
                    reset_error_count += len(fit_errors)
                learned_reset_count += 1
            elif controller == "capacity_matched_independent":
                if isinstance(independent_candidates, FactorizedCandidateBank):
                    (
                        active_posterior,
                        _bank,
                        _donor_observations,
                        fit_errors,
                    ) = _fit_compact_posterior(
                        donor_events,
                        candidates=independent_candidates,
                        posterior_factory=FactorizedGaugeProgramPosterior,
                        maximum_candidates=256,
                    )
                    reset_error_count += len(fit_errors)
            initial_particles, initial_classes = _effective_posterior_capacity(
                active_posterior
            )
            reset_action_count = 0
            reset_online_observations = 0
            stop_reason: str | None = None
            frame = _reset_runtime(runtime, self._environment)
            before = _snapshot_runtime(runtime, frame)
            for step_index in range(int(action_budget)):
                if _is_terminal(before):
                    stop_reason = "game_over" if _is_game_over(before) else "terminal"
                    break
                legal = _legal_runtime(runtime, self._environment)
                legal = tuple(action for action in legal if _action_name(action))
                if not legal:
                    stop_reason = "no_legal_actions"
                    break
                if active_posterior is not None:
                    posterior_scores = _posterior_action_scores(active_posterior)
                ranked = self._rank_actions(
                    legal,
                    controller=controller,
                    donor_prior=donor_prior,
                    posterior_scores=posterior_scores,
                    independent_counts=independent_counts,
                    learned_counts=learned_counts,
                    capacity_slots=capacity_slots,
                    grounding_counts=grounding_counts,
                    reset_index=reset_index,
                    step_index=step_index,
                )
                decision_metadata = {
                    "engine_used": False,
                    "reason": "ranked_nonlearned_controller",
                    "normalized_entropy": None,
                }
                selected = ranked[0]
                if active_posterior is not None:
                    selected, decision_metadata = self._posterior_decision(
                        legal,
                        posterior=active_posterior,
                        frame_states=controller_frame_states,
                        fallback=selected,
                    )
                    if selected is None:
                        stop_reason = "policy_abstained"
                        break
                schema = _action_schema(selected)
                grounding_counts[_grounding_key(selected)] += 1
                if controller == "capacity_matched_independent":
                    independent_counts[schema] += 1
                elif controller == "learned":
                    learned_counts[schema] += 1
                event_id = _stable_hash(
                    {
                        "runtime": RUNTIME_FORMAT_VERSION,
                        "lane": [self.game_id, self.seed, self.split],
                        "reset": reset_index,
                        "step": step_index,
                        "controller": controller,
                    }
                )
                after_frame = _step_runtime(runtime, self._environment, selected)
                after = _snapshot_runtime(
                    runtime,
                    after_frame,
                    fallback_available_actions=legal,
                )
                bundle = _make_bundle(
                    self.factory._bundle_builder,
                    before=before,
                    after=after,
                    action=selected,
                    legal_actions=legal,
                    event_id=event_id,
                    step_index=step_index,
                    game_id=self.game_id,
                )
                posterior_update_error = ""
                if active_posterior is not None:
                    try:
                        active_posterior.observe(bundle)
                        reset_online_observations += 1
                        if controller == "learned":
                            online_held_out_observations += 1
                        else:
                            independent_observations += 1
                    except Exception as exc:  # noqa: BLE001 - retained in ledger.
                        posterior_update_error = type(exc).__name__
                        reset_error_count += 1
                    controller_frame_states = {
                        projection.frame_id: projection.after.state
                        for projection in bundle.projections
                    }
                sequence_prefix.append(
                    (
                        bundle.action,
                        tuple(bundle.events),
                        {
                            projection.frame_id: projection.before.state
                            for projection in bundle.projections
                        },
                    )
                )
                sequence_ranking = None
                if (
                    controller == "learned"
                    and active_posterior is not None
                    and not posterior_update_error
                ):
                    sequence_ranking = rank_option_sequence_signatures(
                        active_posterior,
                        sequence_prefix,
                    )
                progressing_sequence_rank = (
                    sequence_ranking.best_compatible_rank
                    if sequence_ranking is not None
                    and _event_labels(bundle)["progress"]
                    else None
                )
                event = _compact_event(
                    bundle,
                    controller=controller,
                    reset_index=reset_index,
                    step_index=step_index,
                    progressing_sequence_rank=progressing_sequence_rank,
                    donor_game_count=len(self.training_games),
                    capacity_slots=capacity_slots,
                )
                event["selection"].update(
                    {
                        "cross_fit_model": (
                            "gauge_decision_engine_online_option"
                            if controller == "learned"
                            and posterior_metadata.get("posterior_used")
                            else (
                                "capacity_matched_factorized_independent_posterior"
                                if controller == "capacity_matched_independent"
                                else "none"
                            )
                        ),
                        "posterior_candidate_count": int(
                            posterior_metadata.get("posterior_candidates", 0)
                        ),
                        "posterior_class_capacity": int(
                            independent_candidates.metrics.target_classes
                            if controller == "capacity_matched_independent"
                            and isinstance(
                                independent_candidates,
                                FactorizedCandidateBank,
                            )
                            else posterior_metadata.get("posterior_classes", 0)
                        ),
                        "posterior_observation_count": int(
                            posterior_metadata.get("posterior_observations", 0)
                        ),
                        "online_posterior_observation_index": (
                            online_held_out_observations
                            if controller == "learned"
                            else independent_observations
                        ),
                        "posterior_update_error": posterior_update_error,
                        "compatible_option_sequence_mass": (
                            sequence_ranking.compatible_posterior_mass
                            if sequence_ranking is not None
                            else None
                        ),
                        "compatible_option_sequence_signatures": (
                            sequence_ranking.compatible_signature_count
                            if sequence_ranking is not None
                            else 0
                        ),
                        "ranked_option_sequence_signatures": (
                            sequence_ranking.signature_count
                            if sequence_ranking is not None
                            else 0
                        ),
                        "explicit_option_sequence_mass": (
                            sequence_ranking.explicit_posterior_mass
                            if sequence_ranking is not None
                            else None
                        ),
                        "residual_option_sequence_mass": (
                            sequence_ranking.residual_posterior_mass
                            if sequence_ranking is not None
                            else None
                        ),
                        "decision_engine_used": bool(decision_metadata["engine_used"]),
                        "decision_reason": str(decision_metadata["reason"]),
                        "decision_entropy": decision_metadata["normalized_entropy"],
                        "option_conditioned": bool(
                            controller == "learned"
                            and posterior_metadata.get("option_conditioned")
                        ),
                        "posterior_family": (
                            "strict_five_factor_variational_control"
                            if isinstance(
                                active_posterior,
                                FactorizedGaugeProgramPosterior,
                            )
                            else (
                                "joint_gauge_posterior"
                                if active_posterior is not None
                                else "none"
                            )
                        ),
                        "factorized_bank_capacity_matched": bool(
                            isinstance(
                                independent_candidates,
                                FactorizedCandidateBank,
                            )
                            and independent_candidates.metrics.capacity_matched
                        ),
                        "factorized_target_particles": (
                            independent_candidates.metrics.target_particles
                            if isinstance(
                                independent_candidates,
                                FactorizedCandidateBank,
                            )
                            else 0
                        ),
                        "factorized_target_classes": (
                            independent_candidates.metrics.target_classes
                            if isinstance(
                                independent_candidates,
                                FactorizedCandidateBank,
                            )
                            else 0
                        ),
                        "factorized_mdl_prior_preserved": isinstance(
                            active_posterior,
                            FactorizedGaugeProgramPosterior,
                        ),
                        "factorized_control_refusal": independent_refusal,
                    }
                )
                _assert_compact_event_budget(event)
                events.append(event)
                reset_action_count += 1
                if self.split == "discovery":
                    self.factory._remember_discovery(self.game_id, event)
                labels = event["labels"]
                before = after
                if (
                    labels["progress"]
                    or labels["level_complete"]
                    or labels["game_over"]
                    or _is_terminal(after)
                ):
                    stop_reason = (
                        "game_over"
                        if labels["game_over"] or _is_game_over(after)
                        else "progression"
                        if labels["progress"] or labels["level_complete"]
                        else "terminal"
                    )
                    break
            if stop_reason is None:
                stop_reason = "budget_exhausted"
            final_particles, final_classes = _effective_posterior_capacity(
                active_posterior
            )
            if self.split != "discovery":
                reset_audit.append(
                    {
                        "reset_index": reset_index,
                        "controller": controller,
                        "action_count": reset_action_count,
                        "online_observations": reset_online_observations,
                        "error_count": reset_error_count,
                        "initial_particle_count": initial_particles,
                        "initial_class_count": initial_classes,
                        "final_particle_count": final_particles,
                        "final_class_count": final_classes,
                        "stop_reason": stop_reason,
                    }
                )
        if self.split != "discovery":
            donor_ids = [str(event.get("event_id", "")) for event in donor_events]
            self.factory._cross_fit_audit.append(
                {
                    "held_out_game": self.game_id,
                    "seed": self.seed,
                    "training_games": list(self.training_games),
                    "donor_event_count": len(donor_ids),
                    "donor_event_ids_sha256": canonical_sha256(donor_ids),
                    "held_out_prefit_events_used": 0,
                    "resets": reset_audit,
                }
            )
        return events

    def _posterior_decision(
        self,
        legal: Sequence[Any],
        *,
        posterior: Any,
        frame_states: Mapping[str, AbstractState],
        fallback: Any,
    ) -> tuple[Any | None, dict[str, Any]]:
        """Select one legal grounding through the bounded gauge decision engine."""

        candidates = tuple(
            ActionCandidate(_action_name(action), _action_data(action))
            for action in legal
        )
        by_key = {
            candidate.key: action
            for candidate, action in zip(candidates, legal, strict=True)
        }
        fallback_candidate = ActionCandidate(
            _action_name(fallback), _action_data(fallback)
        )
        runtime = self.factory._load_runtime()
        danger_method = next(
            (
                getattr(runtime, name)
                for name in ("is_dangerous_action", "danger_veto")
                if callable(getattr(runtime, name, None))
            ),
            None,
        )

        def danger_veto(candidate: ActionCandidate) -> bool:
            if danger_method is None:
                return False
            raw = by_key.get(candidate.key)
            return bool(
                _invoke(
                    danger_method,
                    context={
                        "environment": self._environment,
                        "env": self._environment,
                        "action": raw,
                    },
                    positional=(self._environment, raw),
                )
            )

        engine = GaugeDecisionEngine(
            maximum_classes=64,
            maximum_option_horizon=16,
        )
        decision = engine.decide(
            posterior,
            frame_states,
            candidates,
            danger_veto=danger_veto,
            fallback_action=fallback_candidate,
        )
        chosen = decision.action
        if chosen is None:
            return None, {
                "engine_used": True,
                "reason": decision.reason,
                "normalized_entropy": float(decision.normalized_entropy),
            }
        selected = by_key.get(chosen.key)
        if selected is None:
            raise DataGateError(
                "gauge decision escaped the current legal grounding set"
            )
        return selected, {
            "engine_used": True,
            "reason": decision.reason,
            "normalized_entropy": float(decision.normalized_entropy),
        }

    def _rank_actions(
        self,
        legal: Sequence[Any],
        *,
        controller: str,
        donor_prior: Counter[tuple[str, int]],
        posterior_scores: Mapping[tuple[str, int], float],
        independent_counts: Counter[tuple[str, int]],
        learned_counts: Counter[tuple[str, int]],
        capacity_slots: int,
        grounding_counts: Counter[str],
        reset_index: int,
        step_index: int,
    ) -> list[Any]:
        def rank(action: Any) -> tuple[Any, ...]:
            schema = _action_schema(action)
            tie = _stable_hash(
                {
                    "seed": self.seed,
                    "reset": reset_index,
                    "step": step_index,
                    "grounding": _grounding_key(action),
                }
            )
            if controller == "learned":
                posterior_score = posterior_scores.get(
                    schema, posterior_scores.get((schema[0], 0), 0.0)
                )
                return (
                    -posterior_score,
                    -donor_prior[schema],
                    learned_counts[schema],
                    tie,
                )
            if controller == "capacity_matched_independent":
                posterior_score = posterior_scores.get(
                    schema, posterior_scores.get((schema[0], 0), 0.0)
                )
                return (-posterior_score, independent_counts[schema], tie)
            return (
                self.factory._discovery_counts[self.game_id][schema],
                grounding_counts[_grounding_key(action)],
                tie,
            )

        return sorted(legal, key=rank)

    def close(self) -> None:
        if self._environment is None:
            return
        runtime = self.factory._load_runtime()
        _close_runtime(runtime, self._environment)
        self._environment = None


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataGateError(f"missing replay {label}")
    return value


def _project_v4_3_trace(
    trace: Mapping[str, Any],
    *,
    event_id: str,
    game_id: str,
) -> PhysicalEventBundle:
    """Compile raw V4.3 frames transiently and retain only frozen projections."""

    from theory.live_transition_loop import build_transition_record

    from .compiler import compile_transition_record

    record = build_transition_record(
        action=str(trace["selected_action_name"]).upper(),
        action_args=dict(trace["selected_action_data"]),
        grid_before=trace["frame_before"],
        grid_after=trace["frame_after"],
        available_actions=tuple(trace["available_action_names"]),
        game_state_before=str(trace["game_state_before"]),
        game_state_after=str(trace["game_state_after"]),
        levels_completed_before=int(trace["levels_completed_before"]),
        levels_completed_after=int(trace["levels_completed_after"]),
        timestamp=int(trace["step_index"]),
    )
    evidence = compile_transition_record(record, source_game_id=game_id)
    bundle = project_transition_with_frozen_frames(evidence, event_id=event_id)
    event_names = {str(item).strip().casefold() for item in bundle.events}
    changed = any(
        projection.before.canonical_hash != projection.after.canonical_hash
        for projection in bundle.projections
    )
    event_names.add("state_changed" if changed else "no_effect")
    return replace(bundle, events=tuple(sorted(event_names)))


def _compact_replay_event(
    *,
    manifest: Mapping[str, Any],
    game_id: str,
    shard_sha256: str,
    row: Mapping[str, Any],
    line_number: int,
    arm_name: str,
    conversion_code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    required_top = (
        "format_version",
        "game_id",
        "source_split",
        "policy_seed",
        "reset_index",
        "root_index",
        "path",
        "depth",
        "context",
        "pair_digest",
        "expected_pre_state_sha256",
        "replay_pre_state_sha256",
    )
    if any(key not in row for key in required_top):
        raise DataGateError(f"incomplete V4.3 replay provenance at line {line_number}")
    if row["format_version"] != "sage12-bound-trajectory-v4.3":
        raise DataGateError(f"unsupported replay format at line {line_number}")
    if row["source_split"] not in {"source_train", "source_validation"}:
        raise DataGateError(f"invalid V4.3 source split at line {line_number}")
    if int(row["depth"]) not in {0, 1, 2} or len(str(row["path"])) != int(row["depth"]):
        raise DataGateError(f"invalid V4.3 tree provenance at line {line_number}")
    if not isinstance(row["context"], Sequence) or len(row["context"]) != 8:
        raise DataGateError(f"incomplete V4.3 context at line {line_number}")
    arm = _required_mapping(row.get(arm_name), label=f"{arm_name} arm")
    action = _required_mapping(arm.get("action"), label="action")
    trace = _required_mapping(arm.get("trace"), label="trace")
    effects = _required_mapping(trace.get("effects"), label="effects")
    required_trace = (
        "trace_digest",
        "format_version",
        "game_id",
        "source_split",
        "selected_action_name",
        "selected_action_data",
        "available_action_names",
        "frame_before",
        "frame_after",
        "frame_before_sha256",
        "frame_after_sha256",
        "game_state_before",
        "game_state_after",
        "levels_completed_before",
        "levels_completed_after",
        "policy_seed",
        "reset_index",
        "step_index",
    )
    required_arm = ("arm", "post_state_sha256", "replay_pre_state_sha256")
    if any(key not in trace for key in required_trace) or any(
        key not in arm for key in required_arm
    ):
        raise DataGateError(f"incomplete V4.3 arm provenance at line {line_number}")
    if str(row["game_id"]) != game_id.split("-", 1)[0]:
        raise FirewallError("replay row escaped its registered source shard")
    if str(trace["game_id"]) != str(row["game_id"]):
        raise DataGateError(f"V4.3 trace game drift at line {line_number}")
    if (
        int(trace["policy_seed"]) != int(row["policy_seed"])
        or int(trace["reset_index"]) != int(row["reset_index"])
        or str(trace["source_split"]) != str(row["source_split"])
    ):
        raise DataGateError(f"V4.3 trace provenance drift at line {line_number}")
    selected_name = str(trace["selected_action_name"]).upper()
    action_name = str(action.get("name", "")).upper()
    selected_data = trace["selected_action_data"]
    action_args = action.get("action_args", {})
    if (
        selected_name != action_name
        or not isinstance(selected_data, Mapping)
        or not isinstance(action_args, Mapping)
        or dict(selected_data) != dict(action_args)
        or selected_name
        not in {str(item).upper() for item in trace["available_action_names"]}
    ):
        raise DataGateError(f"V4.3 executed action drift at line {line_number}")
    from theory.sage12.action_target_data import grid_sha256

    if (
        grid_sha256(trace["frame_before"]) != trace["frame_before_sha256"]
        or grid_sha256(trace["frame_after"]) != trace["frame_after_sha256"]
    ):
        raise DataGateError(f"V4.3 frame hash drift at line {line_number}")
    trace_payload = {
        "game_id": trace["game_id"],
        "source_split": trace["source_split"],
        "policy_seed": int(trace["policy_seed"]),
        "reset_index": int(trace["reset_index"]),
        "step_index": int(trace["step_index"]),
        "frame_before_sha256": trace["frame_before_sha256"],
        "action_name": selected_name,
        "action_data": dict(selected_data),
    }
    if _stable_hash(trace_payload) != trace["trace_digest"]:
        raise DataGateError(f"V4.3 trace digest drift at line {line_number}")
    shared_pre = {
        str(row["expected_pre_state_sha256"]),
        str(row["replay_pre_state_sha256"]),
        str(arm["replay_pre_state_sha256"]),
    }
    if len(shared_pre) != 1:
        raise DataGateError(f"V4.3 replay pre-state drift at line {line_number}")
    pair_payload = {
        "game_id": row["game_id"],
        "source_split": row["source_split"],
        "policy_seed": int(row["policy_seed"]),
        "reset_index": int(row["reset_index"]),
        "root_index": int(row["root_index"]),
        "path": str(row["path"]),
        "pre": row["expected_pre_state_sha256"],
        "left": _required_mapping(row.get("left"), label="left arm")
        .get("trace", {})
        .get("trace_digest"),
        "right": _required_mapping(row.get("right"), label="right arm")
        .get("trace", {})
        .get("trace_digest"),
    }
    if _stable_hash(pair_payload) != row["pair_digest"]:
        raise DataGateError(f"V4.3 pair digest drift at line {line_number}")
    labels_raw = _required_mapping(effects.get("labels", {}), label="effect labels")
    labels = {
        name: bool(labels_raw.get(name, False))
        for name in (
            "actor_displaced",
            "target_created",
            "target_moved",
            "target_removed",
        )
    }
    labels.update(
        {
            "no_effect": bool(effects.get("noop", False)),
            "state_changed": not bool(effects.get("noop", False)),
            "progress": int(trace["levels_completed_after"])
            > int(trace["levels_completed_before"]),
            "level_complete": bool(effects.get("level_complete", False)),
            "game_over": bool(effects.get("game_over", False)),
        }
    )
    event_id = _stable_hash(
        {
            "kind": "frozen_source_replay",
            "shard": shard_sha256,
            "line": line_number,
            "arm": arm_name,
            "trace": trace["trace_digest"],
        }
    )
    bundle = _project_v4_3_trace(trace, event_id=event_id, game_id=game_id)
    event = _compact_event(
        bundle,
        controller="frozen_v4_3_replay",
        reset_index=int(trace["reset_index"]),
        step_index=int(trace["step_index"]),
        progressing_sequence_rank=None,
        donor_game_count=0,
        capacity_slots=0,
    )
    event.update(
        {
            "game_id": game_id,
            "seed": int(row["policy_seed"]),
            "split": REPLAY_SPLIT,
        }
    )
    event["labels"].update(labels)
    event["selection"].update(
        {
            "controller": "frozen_v4_3_replay",
            "legal_grounding": True,
        }
    )
    event["provenance"].update(
        {
            "kind": "frozen_source_replay",
            "game_id": game_id,
            "seed": int(row["policy_seed"]),
            "split": REPLAY_SPLIT,
            "manifest_checksum": manifest["manifest_checksum"],
            "source_format": row["format_version"],
            "source_shard_sha256": shard_sha256,
            "source_row_sha256": canonical_sha256(row),
            "source_line": line_number,
            "pair_digest": str(row["pair_digest"]),
            "arm": str(arm["arm"]),
            "trace_digest": str(trace["trace_digest"]),
            "expected_pre_state_sha256": str(row["expected_pre_state_sha256"]),
            "replay_pre_state_sha256": str(arm["replay_pre_state_sha256"]),
            "post_state_sha256": str(arm["post_state_sha256"]),
            "frame_before_sha256": str(trace["frame_before_sha256"]),
            "frame_after_sha256": str(trace["frame_after_sha256"]),
            "conversion_code_sha256": dict(conversion_code_sha256),
            "converter_sha256": str(conversion_code_sha256["converter"]),
            "projector_sha256": str(conversion_code_sha256["projector"]),
            "observer_frames_sha256": str(conversion_code_sha256["observer_frames"]),
            "compiler_contract_sha256": str(
                conversion_code_sha256["compiler_contract"]
            ),
            "raw_frames_retained": False,
            "graphs_retained": False,
        }
    )
    if "compiler" in conversion_code_sha256:
        event["provenance"]["compiler_sha256"] = str(conversion_code_sha256["compiler"])
    leaks = audit_identity_leaks(
        event["model_view"],
        forbidden_game_ids=(*SOURCE_GAMES, *VALIDATION_GAMES, AR25_GAME),
    )
    if leaks:
        raise FirewallError(f"replay model view leaked identity: {leaks[0]}")
    _assert_compact_event_budget(event)
    sealed = seal_event(event)
    _assert_compact_event_budget(sealed)
    return sealed


def build_v4_3_replay_ledger(
    *,
    manifest: Mapping[str, Any],
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
    shard_paths: Mapping[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Build a compact ledger from exactly the three hash-bound V4.3 shards."""

    registry = manifest.get("frozen_source_shards")
    if not isinstance(registry, Mapping) or set(registry) != set(SOURCE_GAMES):
        raise ManifestDriftError("manifest must bind exactly the three source shards")
    overrides = dict(shard_paths or {})
    if overrides and set(overrides) != set(SOURCE_GAMES):
        raise DataGateError("shard overrides must cover exactly the source games")
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    code_registry = manifest.get("code_sha256", {})
    if not isinstance(code_registry, Mapping):
        raise ManifestDriftError("manifest lacks conversion code bindings")
    required_code = {
        "converter": "theory/sage_t/t10_2_runtime.py",
        "projector": "theory/sage_t/frame_adapters_v10_2.py",
        "observer_frames": "theory/sage_t/observer_frames_v10_2.py",
        "compiler_contract": "theory/sage_t/contracts.py",
    }
    conversion_code_sha256: dict[str, str] = {}
    for role, relative in required_code.items():
        expected = str(code_registry.get(relative, ""))
        if not expected or file_sha256(root / relative) != expected:
            raise ManifestDriftError(f"conversion code binding drifted: {relative}")
        conversion_code_sha256[role] = expected
    compiler_path = "theory/sage_t/compiler.py"
    if compiler_path in code_registry:
        expected_compiler = str(code_registry[compiler_path])
        if file_sha256(root / compiler_path) != expected_compiler:
            raise ManifestDriftError("conversion code binding drifted: compiler")
        conversion_code_sha256["compiler"] = expected_compiler
    rows: list[dict[str, Any]] = []
    for game_id in SOURCE_GAMES:
        descriptor = _required_mapping(registry[game_id], label="shard descriptor")
        registered_path = descriptor.get("path")
        expected_sha = str(descriptor.get("sha256", ""))
        if not registered_path or not expected_sha:
            raise ManifestDriftError(f"incomplete frozen shard binding: {game_id}")
        path = Path(overrides.get(game_id, registered_path))
        if not path.is_absolute():
            path = root / path
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise ManifestDriftError(f"frozen source shard drifted: {game_id}")
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line:
                continue
            try:
                source_row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DataGateError(
                    f"invalid V4.3 JSONL row {line_number}: {game_id}"
                ) from exc
            if not isinstance(source_row, Mapping):
                raise DataGateError(f"non-object V4.3 row {line_number}: {game_id}")
            for arm_name in ("left", "right"):
                rows.append(
                    _compact_replay_event(
                        manifest=manifest,
                        game_id=game_id,
                        shard_sha256=actual_sha,
                        row=source_row,
                        line_number=line_number,
                        arm_name=arm_name,
                        conversion_code_sha256=conversion_code_sha256,
                    )
                )
    validate_source_events(rows, manifest=manifest, replay=True)
    if output_path is not None:
        write_event_ledger(output_path, rows)
    return rows


def build_replay_ledger(**kwargs: Any) -> list[dict[str, Any]]:
    """Public short alias for :func:`build_v4_3_replay_ledger`."""

    return build_v4_3_replay_ledger(**kwargs)


def _event_progress(event: Mapping[str, Any]) -> int:
    outcome = event.get("outcome", {})
    if isinstance(outcome, Mapping):
        for key in ("progression", "progress", "progress_mean"):
            try:
                value = float(outcome.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and value > 0.0:
                return max(1, math.ceil(value))
    labels = event.get("labels", {})
    return int(isinstance(labels, Mapping) and bool(labels.get("progress")))


def _confirmation_metrics(
    events: Sequence[Mapping[str, Any]],
    cross_fit_audit: Mapping[str, Any],
) -> dict[str, Any]:
    units: dict[tuple[str, int], dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"levels": 0, "actions": 0})
    )
    positive_ranks: dict[str, int] = {}
    for event in events:
        if event.get("split") != "leave_one_game_out_confirmation":
            continue
        selection = event.get("selection", {})
        if not isinstance(selection, Mapping):
            continue
        controller = str(selection.get("controller", ""))
        if controller not in {"learned", "capacity_matched_independent"}:
            continue
        unit = (str(event.get("game_id", "")), int(event.get("seed", -1)))
        units[unit][controller]["actions"] += 1
        progress = _event_progress(event)
        units[unit][controller]["levels"] += progress
        if controller == "learned" and progress:
            try:
                rank = int(selection.get("progressing_sequence_rank"))
            except (TypeError, ValueError):
                continue
            if rank > 0:
                game = unit[0]
                positive_ranks[game] = min(rank, positive_ranks.get(game, rank))
    paired_deltas: list[float] = []
    per_game: dict[str, int] = defaultdict(int)
    learned_levels = 0
    independent_levels = 0
    complete_pairs = 0
    for (game, _seed), arms in sorted(units.items()):
        if set(arms) != {"learned", "capacity_matched_independent"}:
            continue
        learned = arms["learned"]
        independent = arms["capacity_matched_independent"]
        if not learned["actions"] or not independent["actions"]:
            continue
        complete_pairs += 1
        learned_levels += learned["levels"]
        independent_levels += independent["levels"]
        per_game[game] += learned["levels"] - independent["levels"]
        paired_deltas.append(
            1000.0 * learned["levels"] / learned["actions"]
            - 1000.0 * independent["levels"] / independent["actions"]
        )
    return {
        "expected_pairs": len(SOURCE_GAMES) * len(CONFIRMATION_SEEDS),
        "registered_pairs": int(cross_fit_audit.get("registered_unit_count", 0)),
        "schedule_checks": dict(cross_fit_audit.get("checks", {})),
        "schedule_passed": cross_fit_audit.get("passed") is True,
        "complete_pairs": complete_pairs,
        "positive_fold_ranks": positive_ranks,
        "paired_rate_ci_lower": paired_bootstrap_lower(paired_deltas),
        "learned_levels": learned_levels,
        "independent_levels": independent_levels,
        "nonnegative_games": sum(value >= 0 for value in per_game.values()),
    }


def _executed_grammar(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sequences: dict[
        tuple[str, int, int],
        list[tuple[int, str, Mapping[str, Any]]],
    ] = defaultdict(list)
    for event in events:
        selection = event.get("selection", {})
        if not isinstance(selection, Mapping):
            continue
        try:
            reset = int(selection.get("reset_index", -1))
            step = int(selection.get("step_index", -1))
            seed = int(event.get("seed", -1))
        except (TypeError, ValueError):
            continue
        if reset < 0 or step < 0:
            continue
        key = (str(event.get("game_id", "")), seed, reset)
        sequences[key].append((step, str(selection.get("action_name", "")), event))

    records: list[dict[str, Any]] = []
    for (game, seed, reset), rows in sorted(sequences.items()):
        ordered = sorted(rows)
        executed_rows = [
            (step, name, event)
            for step, name, event in ordered
            if isinstance(event.get("action"), Mapping)
            and event["action"].get("executed") is True
        ]
        if not executed_rows:
            continue
        executed_steps = [step for step, _, _ in executed_rows]
        all_actions_executed = len(executed_rows) == len(ordered)
        steps_contiguous = executed_steps == list(range(len(executed_steps)))
        sequence_levels = sum(_event_progress(event) for _, _, event in executed_rows)
        progress_at_end = bool(
            sequence_levels > 0 and _event_progress(executed_rows[-1][2]) > 0
        )
        game_overs = 0
        illegal = 0
        errors = 0
        for _, _, event in ordered:
            labels = event.get("labels", {})
            selection = event.get("selection", {})
            if isinstance(labels, Mapping) and (
                labels.get("game_over") or labels.get("GAME_OVER")
            ):
                game_overs += 1
            if (
                isinstance(selection, Mapping)
                and selection.get("legal_grounding") is False
            ):
                illegal += 1
            posterior_error = (
                selection.get("posterior_update_error")
                if isinstance(selection, Mapping)
                else ""
            )
            if event.get("error") or posterior_error:
                errors += 1
        sequence_hash = _stable_hash(
            {
                "events": [event.get("event_id", "") for _, _, event in executed_rows],
                "actions": [name for _, name, _ in executed_rows],
            }
        )
        records.append(
            {
                "game": game,
                "seed": seed,
                "reset": reset,
                "actions": len(executed_rows),
                "levels": sequence_levels,
                "errors": errors,
                "illegal_actions": illegal,
                "game_overs": game_overs,
                "all_actions_executed": all_actions_executed,
                "steps_contiguous": steps_contiguous,
                "progress_at_end": progress_at_end,
                "oracle_eligible": bool(
                    all_actions_executed
                    and steps_contiguous
                    and progress_at_end
                    and errors == 0
                    and illegal == 0
                    and game_overs == 0
                ),
                "sequence_hash": sequence_hash,
            }
        )

    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["oracle_eligible"] is True:
            by_game[str(record["game"])].append(record)

    def oracle_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -int(item["levels"]),
            int(item["errors"]),
            int(item["illegal_actions"]),
            int(item["game_overs"]),
            int(item["actions"]),
            str(item["sequence_hash"]),
        )

    oracle_records = [
        min(game_records, key=oracle_key)
        for _game, game_records in sorted(by_game.items())
        if game_records
    ]
    progress_games = {
        str(record["game"]) for record in oracle_records if int(record["levels"]) > 0
    }
    eligible_records = [
        record for record in records if record["oracle_eligible"] is True
    ]
    best_record = min(eligible_records, key=oracle_key) if eligible_records else None
    return {
        "progress_games": len(progress_games),
        "positive_folds": sorted(progress_games),
        "actions": sum(int(record["actions"]) for record in oracle_records),
        "levels": sum(int(record["levels"]) for record in oracle_records),
        "errors": sum(int(record["errors"]) for record in oracle_records),
        "illegal_actions": sum(
            int(record["illegal_actions"]) for record in oracle_records
        ),
        "game_overs": sum(int(record["game_overs"]) for record in oracle_records),
        "oracle_sequences": len(oracle_records),
        "oracle_sequence_hashes": sorted(
            str(record["sequence_hash"]) for record in oracle_records
        ),
        "executed_sequences": len(records),
        "eligible_sequences": len(eligible_records),
        "executed_sequence_hashes": sorted(
            str(record["sequence_hash"]) for record in records
        ),
        "best_executed_sequence": (
            {
                key: best_record[key]
                for key in (
                    "actions",
                    "levels",
                    "errors",
                    "illegal_actions",
                    "game_overs",
                    "all_actions_executed",
                    "steps_contiguous",
                    "progress_at_end",
                    "oracle_eligible",
                    "sequence_hash",
                )
            }
            if best_record is not None
            else {
                "actions": 0,
                "levels": 0,
                "errors": 0,
                "illegal_actions": 0,
                "game_overs": 0,
                "all_actions_executed": False,
                "steps_contiguous": False,
                "progress_at_end": False,
                "oracle_eligible": False,
                "sequence_hash": "",
            }
        ),
        "best_executed_sequence_levels": (
            int(best_record["levels"]) if best_record is not None else 0
        ),
        "all_executed_actions": sum(int(record["actions"]) for record in records),
        "all_executed_errors": sum(int(record["errors"]) for record in records),
        "all_executed_illegal_actions": sum(
            int(record["illegal_actions"]) for record in records
        ),
        "all_executed_game_overs": sum(int(record["game_overs"]) for record in records),
    }


def _grammar_oracle_passes(grammar: Mapping[str, Any]) -> bool:
    """Apply the preregistered grammar gate to executed-sequence evidence."""

    return bool(
        int(grammar.get("actions", 0)) > 0
        and int(grammar.get("progress_games", 0)) >= 2
        and int(grammar.get("levels", 0)) >= 2
        and int(grammar.get("errors", 1)) == 0
        and int(grammar.get("illegal_actions", 1)) == 0
        and int(grammar.get("game_overs", 1)) == 0
    )


def _run_control(
    runner: Callable[..., Any] | None,
    name: str,
    *,
    events: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    posterior: Any,
) -> dict[str, Any]:
    if runner is None:
        return {
            "completed": True,
            "passed": False,
            "reason": "no injected override; internal result retained",
        }
    try:
        raw = _invoke(
            runner,
            context={
                "name": name,
                "control": name,
                "events": tuple(events),
                "manifest": manifest,
                "posterior": posterior,
            },
            positional=(name, tuple(events)),
        )
    except Exception as exc:  # noqa: BLE001 - failures are evidence, not success.
        return {
            "completed": True,
            "passed": False,
            "reason": f"control error: {type(exc).__name__}",
        }
    if isinstance(raw, bool):
        return {"completed": True, "passed": raw}
    if not isinstance(raw, Mapping):
        return {
            "completed": True,
            "passed": False,
            "reason": "control returned no metrics",
        }
    result = _clone(raw)
    result["completed"] = True
    result["passed"] = result.get("passed") is True
    return result


def _identity_probe_accuracy(events: Sequence[Mapping[str, Any]]) -> float:
    """Seed-held-out game probe over identity-free event signatures."""

    samples: list[tuple[int, str, str]] = []
    for event in events:
        try:
            seed = int(event.get("seed", -1))
        except (TypeError, ValueError):
            continue
        game = str(event.get("game_id", ""))
        model = event.get("model_view")
        if seed < 0 or game not in SOURCE_GAMES or not isinstance(model, Mapping):
            continue
        selection = event.get("selection", {})
        action_name = (
            str(selection.get("action_name", ""))
            if isinstance(selection, Mapping)
            else ""
        )
        samples.append(
            (seed, game, _stable_hash({"model": model, "action": action_name}))
        )
    seeds = sorted({seed for seed, _, _ in samples})
    if len(seeds) < 2:
        return 1.0
    correct = 0
    total = 0
    for held_out_seed in seeds:
        train = [item for item in samples if item[0] != held_out_seed]
        test = [item for item in samples if item[0] == held_out_seed]
        if not train or not test:
            continue
        majority = Counter(game for _, game, _ in train).most_common(1)[0][0]
        by_signature: dict[str, Counter[str]] = defaultdict(Counter)
        for _, game, signature in train:
            by_signature[signature][game] += 1
        for _, game, signature in test:
            prediction = (
                by_signature[signature].most_common(1)[0][0]
                if by_signature[signature]
                else majority
            )
            correct += int(prediction == game)
            total += 1
    return correct / total if total else 1.0


def _swap_mismatch_rate(
    events: Sequence[Mapping[str, Any]], key: str
) -> tuple[int, float]:
    values = [event.get(key) for event in events if event.get(key) is not None]
    if len(values) < 2:
        return len(values), 0.0
    shifted = values[1:] + values[:1]
    mismatches = sum(
        _stable_hash(left) != _stable_hash(right)
        for left, right in zip(values, shifted, strict=True)
    )
    return len(values), mismatches / len(values)


def _immediate_noop_dedup(
    events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    """Apply the frozen repeat-controller's exact immediate no-op dedup rule."""

    retained: list[Mapping[str, Any]] = []
    previous: tuple[Any, ...] | None = None
    removed = 0
    for event in events:
        selection = event.get("selection", {})
        labels = event.get("labels", {})
        branch = (
            str(event.get("game_id", "")),
            int(event.get("seed", -1)),
            int(
                selection.get("reset_index", -1)
                if isinstance(selection, Mapping)
                else -1
            ),
        )
        signature = (
            branch,
            str(selection.get("action_name", ""))
            if isinstance(selection, Mapping)
            else "",
            _stable_hash(event.get("model_view", {})),
        )
        no_effect = isinstance(labels, Mapping) and labels.get("no_effect") is True
        if no_effect and previous == signature:
            removed += 1
            continue
        retained.append(event)
        previous = signature
    return tuple(retained), removed


def _t10_1_repeat_candidates(
    events: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 64,
) -> tuple[JointGaugeHypothesis, ...]:
    """Re-express the frozen T10.1 repeat-only behavior on compact evidence."""

    from .progress_witness_v10 import compile_progress_program

    candidates: dict[str, JointGaugeHypothesis] = {}
    frame = observer_frame_spec("root_only")
    for actions, observed_progress in sorted(
        _sequence_groups(events), key=lambda item: (item[0], item[1])
    ):
        if not actions:
            continue
        names = tuple(sorted(set(actions)))[:4]
        horizon = max(1, min(16, len(actions)))
        for positive in (observed_progress, not observed_progress):
            program = compile_progress_program(
                sequence_length=horizon,
                action_names=names,
                positive=positive,
            )
            for name in names:
                hypothesis = JointGaugeHypothesis(
                    world_program=program,
                    frame=frame,
                    transports=(),
                    option=repeat(
                        name.lower(),
                        termination_predicate="level_complete",
                        maximum_horizon=horizon,
                    ),
                )
                candidates.setdefault(hypothesis.canonical_hash, hypothesis)
                if len(candidates) >= max(1, min(64, int(maximum))):
                    return tuple(candidates.values())
    return tuple(candidates.values())


def _diagnostic_rows(
    events: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 64,
) -> tuple[tuple[Mapping[str, Any], PhysicalEventBundle], ...]:
    rows: list[tuple[Mapping[str, Any], PhysicalEventBundle]] = []
    for event in events:
        bundle = _bundle_from_compact_event(event)
        if bundle is None:
            continue
        rows.append((event, bundle))
        if len(rows) >= max(1, min(64, int(maximum))):
            break
    return tuple(rows)


def _positive_hypothesis(hypothesis: JointGaugeHypothesis) -> bool:
    goal = getattr(hypothesis.world_program, "goal_rule", None)
    return str(getattr(goal, "family", "")) != "no_progress"


def _outcome_grounded_diagnostic(
    candidate_pairs: Sequence[tuple[JointGaugeHypothesis, JointGaugeHypothesis]],
    rows: Sequence[tuple[Mapping[str, Any], PhysicalEventBundle]],
) -> dict[str, Any]:
    """Run one capacity/prior-matched posterior diagnostic on paired bundles.

    The score is the mean log posterior probability assigned to the observed
    progress/no-progress outcome after each real source transition.  It is an
    offline causal diagnostic, not a substitute for the preregistered active
    learned-vs-independent advantage.
    """

    pairs = tuple(candidate_pairs)[:64]
    if not pairs or not rows:
        return {
            "completed": False,
            "observations": 0,
            "candidate_count": len(pairs),
            "errors": ["missing_candidates_or_structural_bundles"],
        }
    if any(
        not isinstance(reference, JointGaugeHypothesis)
        or not isinstance(candidate, JointGaugeHypothesis)
        for reference, candidate in pairs
    ):
        return {
            "completed": False,
            "observations": 0,
            "candidate_count": len(pairs),
            "errors": ["non_gauge_candidate"],
        }
    references = tuple(reference for reference, _ in pairs)
    alternatives = tuple(candidate for _, candidate in pairs)
    if len({reference.canonical_hash for reference in references}) != len(references):
        return {
            "completed": False,
            "observations": 0,
            "candidate_count": len(pairs),
            "errors": ["reference_capacity_collapse"],
        }
    if len({candidate.canonical_hash for candidate in alternatives}) != len(
        alternatives
    ):
        return {
            "completed": False,
            "observations": 0,
            "candidate_count": len(pairs),
            "errors": ["ablation_capacity_collapse"],
        }
    prior_vector = tuple(float(reference.log_prior) for reference in references)
    if not all(math.isfinite(value) for value in prior_vector):
        return {
            "completed": False,
            "observations": 0,
            "candidate_count": len(pairs),
            "errors": ["nonfinite_reference_prior"],
        }
    try:
        evidence_fingerprint = _stable_hash(
            [
                {
                    "event_id": bundle.event_id,
                    "bundle_checksum": bundle.canonical_checksum,
                }
                for _, bundle in rows
            ]
        )
        prior_vector_sha256 = _stable_hash(list(prior_vector))
    except (TypeError, ValueError):
        return {
            "completed": False,
            "observations": 0,
            "candidate_count": len(pairs),
            "errors": ["noncanonical_diagnostic_evidence"],
        }
    prior_by_hash = {
        candidate.canonical_hash: float(reference.log_prior)
        for reference, candidate in pairs
    }
    posterior = GaugeProgramPosterior(maximum_classes=64)
    posterior.seed(alternatives)
    if len(posterior.particles) != len(alternatives):
        return {
            "completed": False,
            "observations": 0,
            "candidate_count": len(alternatives),
            "retained_candidate_count": len(posterior.particles),
            "errors": ["posterior_capacity_mismatch"],
        }
    # This diagnostic intentionally freezes the reference priors while
    # changing exactly one component.  GaugeProgramPosterior has no public
    # fixed-prior seeding hook, so the isolated control resets its particles
    # before any observation and then invokes the normalizer.
    posterior._particles = [
        replace(
            particle,
            log_prior=prior_by_hash[particle.hypothesis.canonical_hash],
            log_weight=prior_by_hash[particle.hypothesis.canonical_hash],
        )
        for particle in posterior.particles
    ]
    posterior._normalize()

    log_scores: list[float] = []
    truth_probabilities: list[float] = []
    progression_ranks: list[int] = []
    errors: list[str] = []
    observations = 0
    branch: tuple[str, int, int] | None = None
    sequence_prefix: list[
        tuple[
            ActionCandidate,
            tuple[str, ...],
            Mapping[str, AbstractState],
        ]
    ] = []
    option_sequence_observations = 0
    option_compatible_observations = 0
    option_compatible_progress_observations = 0
    option_compatible_masses: list[float] = []
    positive_prefix_count = 0
    compatible_positive_prefix_masses: list[float] = []
    for event, bundle in rows:
        selection = event.get("selection", {})
        current_branch = (
            str(event.get("game_id", "")),
            int(event.get("seed", -1)),
            int(
                selection.get("reset_index", -1)
                if isinstance(selection, Mapping)
                else -1
            ),
        )
        if branch is not None and current_branch != branch:
            posterior.start_branch()
            sequence_prefix.clear()
        branch = current_branch
        try:
            posterior.observe(bundle)
        except Exception as exc:  # noqa: BLE001 - negative control evidence.
            errors.append(type(exc).__name__)
            continue
        observations += 1
        sequence_prefix.append(
            (
                bundle.action,
                tuple(bundle.events),
                {
                    projection.frame_id: projection.before.state
                    for projection in bundle.projections
                },
            )
        )
        sequence_ranking = rank_option_sequence_signatures(
            posterior,
            sequence_prefix,
        )
        option_sequence_observations += 1
        if sequence_ranking.compatible_signature_count > 0:
            option_compatible_observations += 1
            option_compatible_masses.append(sequence_ranking.compatible_posterior_mass)
        particles = tuple(
            sorted(
                posterior.particles,
                key=lambda item: (
                    item.probability,
                    item.hypothesis.canonical_hash,
                ),
                reverse=True,
            )
        )
        positive_mass = sum(
            particle.probability
            for particle in particles
            if _positive_hypothesis(particle.hypothesis)
        )
        progressed = _event_progress(event) > 0
        if progressed:
            positive_prefix_count += 1
            compatible_positive_prefix_masses.append(
                sequence_ranking.compatible_posterior_mass
                if sequence_ranking.compatible_signature_count > 0
                else 0.0
            )
            if sequence_ranking.compatible_signature_count > 0:
                option_compatible_progress_observations += 1
        truth_probability = positive_mass if progressed else 1.0 - positive_mass
        truth_probability = max(1e-12, min(1.0, truth_probability))
        truth_probabilities.append(truth_probability)
        log_scores.append(math.log(truth_probability))
        if progressed:
            ranks = [
                index
                for index, particle in enumerate(particles, start=1)
                if _positive_hypothesis(particle.hypothesis)
            ]
            if ranks:
                progression_ranks.append(min(ranks))
    return {
        "completed": bool(rows) and observations == len(rows) and not errors,
        "score_name": "mean_observed_outcome_log_posterior_probability",
        "mean_outcome_log_probability": (
            sum(log_scores) / len(log_scores) if log_scores else -1.0e9
        ),
        "mean_truth_probability": (
            sum(truth_probabilities) / len(truth_probabilities)
            if truth_probabilities
            else 0.0
        ),
        "progression_ranks": progression_ranks,
        "observations": observations,
        "executed_actions": observations,
        "option_sequence_observations": option_sequence_observations,
        "option_compatible_observations": option_compatible_observations,
        "option_compatible_progress_observations": (
            option_compatible_progress_observations
        ),
        "positive_prefix_count": positive_prefix_count,
        "compatible_positive_prefix_count": (option_compatible_progress_observations),
        "mean_compatible_positive_prefix_mass": (
            sum(compatible_positive_prefix_masses)
            / len(compatible_positive_prefix_masses)
            if compatible_positive_prefix_masses
            else 0.0
        ),
        "mean_option_compatible_mass": (
            sum(option_compatible_masses) / len(option_compatible_masses)
            if option_compatible_masses
            else 0.0
        ),
        "candidate_count": len(alternatives),
        "positive_candidate_count": sum(
            _positive_hypothesis(candidate) for candidate in alternatives
        ),
        "negative_candidate_count": sum(
            not _positive_hypothesis(candidate) for candidate in alternatives
        ),
        "evidence_fingerprint": evidence_fingerprint,
        "prior_vector_sha256": prior_vector_sha256,
        "fixed_reference_priors": True,
        "capacity_matched": len(posterior.particles) == len(alternatives),
        "errors": errors,
    }


def _ablation_component(
    candidate: JointGaugeHypothesis,
    name: str,
) -> Any:
    if name == "deterministically_permuted_transport":
        return sorted(
            (transport.canonical_payload for transport in candidate.transports),
            key=_canonical_json,
        )
    if name == "frame_swap":
        return candidate.frame.canonical_payload
    if name == "binding_swap":
        return [
            (binding.action_name, binding.operator, binding.target_role)
            for binding in candidate.world_program.action_bindings
        ]
    if name == "dynamics_swap":
        return [repr(rule) for rule in candidate.world_program.transition_rules]
    if name == "goal_swap":
        return {
            "progress_rule": repr(candidate.world_program.progress_rule),
            "terminal_rules": [
                repr(rule) for rule in candidate.world_program.terminal_rules
            ],
            "goal_rule": repr(candidate.world_program.goal_rule),
        }
    if name == "option_swap":
        return candidate.option.canonical_payload
    raise KeyError(name)


def _replace_ablation_component(
    candidate: JointGaugeHypothesis,
    donor: JointGaugeHypothesis,
    name: str,
) -> JointGaugeHypothesis:
    if name == "deterministically_permuted_transport":
        return replace(candidate, transports=donor.transports)
    if name == "frame_swap":
        return replace(candidate, frame=donor.frame)
    if name == "binding_swap":
        program = replace(
            candidate.world_program,
            action_bindings=donor.world_program.action_bindings,
        )
        return replace(candidate, world_program=program)
    if name == "dynamics_swap":
        program = replace(
            candidate.world_program,
            transition_rules=donor.world_program.transition_rules,
        )
        return replace(candidate, world_program=program)
    if name == "goal_swap":
        program = replace(
            candidate.world_program,
            progress_rule=donor.world_program.progress_rule,
            terminal_rules=donor.world_program.terminal_rules,
            goal_rule=donor.world_program.goal_rule,
        )
        return replace(candidate, world_program=program)
    if name == "option_swap":
        return replace(candidate, option=donor.option)
    raise KeyError(name)


def _ablation_pairs(
    candidates: Sequence[Any],
    name: str,
) -> tuple[
    tuple[tuple[JointGaugeHypothesis, JointGaugeHypothesis], ...],
    str,
]:
    bank = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, JointGaugeHypothesis)
            ),
            key=lambda candidate: candidate.canonical_hash,
        )
    )[:64]
    if len(bank) != min(64, len(candidates)):
        return (), "non_gauge_candidate"
    if len({candidate.canonical_hash for candidate in bank}) != len(bank):
        return (), "reference_capacity_collapse"
    if name == "no_transport":
        output: list[tuple[JointGaugeHypothesis, JointGaugeHypothesis]] = []
        alternative_hashes: set[str] = set()
        for candidate in bank:
            if not candidate.transports:
                continue
            try:
                altered = replace(candidate, transports=())
            except (TypeError, ValueError):
                continue
            if (
                altered.canonical_hash == candidate.canonical_hash
                or altered.canonical_hash in alternative_hashes
            ):
                continue
            alternative_hashes.add(altered.canonical_hash)
            output.append((candidate, altered))
        if not output:
            return (), "no_unique_transport_removal_pairs"
        return tuple(output), ""

    swap_names = {
        "deterministically_permuted_transport",
        "frame_swap",
        "binding_swap",
        "dynamics_swap",
        "goal_swap",
        "option_swap",
    }
    if name not in swap_names:
        return (), "unknown_ablation"
    if len(bank) < 2:
        return (), "fixed_point_free_permutation_requires_two_candidates"

    source_tokens = tuple(
        _stable_hash(_ablation_component(candidate, name)) for candidate in bank
    )
    for offset in range(1, len(bank)):
        donors = bank[offset:] + bank[:offset]
        donor_tokens = tuple(
            _stable_hash(_ablation_component(donor, name)) for donor in donors
        )
        if any(
            source == donor
            for source, donor in zip(source_tokens, donor_tokens, strict=True)
        ):
            continue
        if Counter(source_tokens) != Counter(donor_tokens):
            continue
        alternatives: list[JointGaugeHypothesis] = []
        try:
            alternatives = [
                _replace_ablation_component(candidate, donor, name)
                for candidate, donor in zip(bank, donors, strict=True)
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if any(
            candidate.canonical_hash == altered.canonical_hash
            for candidate, altered in zip(bank, alternatives, strict=True)
        ):
            continue
        if len({candidate.canonical_hash for candidate in alternatives}) != len(
            alternatives
        ):
            continue
        return tuple(zip(bank, alternatives, strict=True)), ""
    return (), "no_valid_fixed_point_free_component_permutation"


def _paired_ablation_result(
    name: str,
    *,
    candidates: Sequence[Any],
    rows: Sequence[tuple[Mapping[str, Any], PhysicalEventBundle]],
) -> dict[str, Any]:
    pairs, refusal = _ablation_pairs(candidates, name)
    reference = _outcome_grounded_diagnostic(
        tuple((candidate, candidate) for candidate, _ in pairs),
        rows,
    )
    ablated = _outcome_grounded_diagnostic(pairs, rows)
    reference_score = float(reference.get("mean_outcome_log_probability", -1.0e9))
    ablated_score = float(ablated.get("mean_outcome_log_probability", -1.0e9))
    same_evidence = bool(
        reference.get("evidence_fingerprint")
        and reference.get("evidence_fingerprint") == ablated.get("evidence_fingerprint")
    )
    same_observations = bool(
        int(reference.get("observations", 0)) > 0
        and reference.get("observations") == ablated.get("observations")
    )
    same_capacity = bool(
        pairs
        and reference.get("candidate_count") == len(pairs)
        and ablated.get("candidate_count") == len(pairs)
        and reference.get("capacity_matched") is True
        and ablated.get("capacity_matched") is True
    )
    same_priors = bool(
        reference.get("fixed_reference_priors") is True
        and ablated.get("fixed_reference_priors") is True
        and reference.get("prior_vector_sha256") == ablated.get("prior_vector_sha256")
    )
    discriminant_classes = bool(
        int(reference.get("positive_candidate_count", 0)) > 0
        and int(reference.get("negative_candidate_count", 0)) > 0
        and reference.get("positive_candidate_count")
        == ablated.get("positive_candidate_count")
        and reference.get("negative_candidate_count")
        == ablated.get("negative_candidate_count")
    )
    component_multiset_preserved = bool(name != "no_transport" and pairs)
    ablation_contract_preserved = bool(
        pairs and (name == "no_transport" or component_multiset_preserved)
    )
    evaluable = bool(
        pairs
        and not refusal
        and reference.get("completed") is True
        and ablated.get("completed") is True
        and same_evidence
        and same_observations
        and same_capacity
        and same_priors
        and discriminant_classes
        and ablation_contract_preserved
        and math.isfinite(reference_score)
        and math.isfinite(ablated_score)
    )
    degradation = reference_score - ablated_score if evaluable else 0.0
    return {
        "attempted": True,
        "completed": True,
        "passed": evaluable and degradation > 0.0,
        "evaluable": evaluable,
        "refusal": refusal,
        "changed_candidates": len(pairs),
        "candidate_count": len(pairs),
        "paired_observations": int(ablated.get("observations", 0)),
        "same_evidence": same_evidence,
        "same_observations": same_observations,
        "same_capacity": same_capacity,
        "same_priors": same_priors,
        "discriminant_classes": discriminant_classes,
        "component_multiset_preserved": component_multiset_preserved,
        "ablation_contract_preserved": ablation_contract_preserved,
        "degradation": degradation,
        "reference": dict(reference),
        "ablated": ablated,
        "active_advantage_required_separately": True,
        "reason": (
            "paired outcome-grounded degradation"
            if evaluable
            else refusal
            or "component ablation could not be executed at matched capacity"
        ),
    }


def _single_frame_control(
    frame_id: str,
    events: Sequence[Mapping[str, Any]],
    candidates: Sequence[Any],
) -> dict[str, Any]:
    selected = tuple(
        candidate
        for candidate in candidates
        if str(getattr(getattr(candidate, "frame", None), "frame_id", "")) == frame_id
    )
    if not selected:
        return {
            "completed": True,
            "passed": False,
            "candidate_count": 0,
            "observation_count": 0,
            "reason": "no candidate for frame",
        }
    posterior = GaugeProgramPosterior()
    posterior.seed(selected)
    observations = 0
    errors = 0
    branch: tuple[str, int, int] | None = None
    for event in events:
        try:
            bundle = _bundle_from_compact_event(event)
            if bundle is None:
                continue
            selection = event.get("selection", {})
            current_branch = (
                str(event.get("game_id", "")),
                int(event.get("seed", -1)),
                int(
                    selection.get("reset_index", -1)
                    if isinstance(selection, Mapping)
                    else -1
                ),
            )
            if branch is not None and current_branch != branch:
                posterior.start_branch()
            branch = current_branch
            restricted = PhysicalEventBundle(
                event_id=bundle.event_id,
                action=bundle.action,
                common_outcome=bundle.common_outcome,
                projections=(bundle.projection(frame_id),),
                events=bundle.events,
                reset=bundle.reset,
            )
            posterior.observe(restricted)
            observations += 1
        except Exception:  # noqa: BLE001 - error count is the control result.
            errors += 1
    return {
        "completed": True,
        "passed": observations > 0 and errors == 0 and bool(posterior.classes),
        "candidate_count": len(selected),
        "observation_count": observations,
        "class_count": len(posterior.classes),
        "errors": errors,
    }


def _paired_advantage(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("attempted") is True
        and result.get("evaluable") is True
        and result.get("passed") is True
        and float(result.get("degradation", 0.0)) > 0.0
        and result.get("same_evidence") is True
        and result.get("same_observations") is True
        and result.get("same_capacity") is True
        and result.get("same_priors") is True
        and result.get("discriminant_classes") is True
        and result.get("ablation_contract_preserved") is True
        and int(result.get("paired_observations", 0)) > 0
    )


def _structural_delta_event_count(
    rows: Sequence[tuple[Mapping[str, Any], PhysicalEventBundle]],
) -> int:
    def changed(packet: PredictionPacket) -> bool:
        return any(
            abs(float(value)) > 0.0
            for deltas in (
                packet.object_deltas,
                packet.relation_deltas,
                packet.topology_deltas,
            )
            for value in deltas.values()
        )

    return sum(
        any(changed(projection.observation) for projection in bundle.projections)
        for _, bundle in rows
    )


def _internal_control_results(
    *,
    manifest: Mapping[str, Any],
    fresh: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    candidates: Sequence[Any],
    posterior: Any,
    posterior_observations: int,
    posterior_errors: Sequence[str],
    confirmation: Mapping[str, Any],
    grammar: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {
        name: {"completed": False, "passed": False}
        for name in REGISTERED_SOURCE_CONTROLS
    }
    deduplicated_fresh, noop_duplicates = _immediate_noop_dedup(fresh)
    t10_1_candidates = _t10_1_repeat_candidates(deduplicated_fresh)
    t10_1_rows = _diagnostic_rows(deduplicated_fresh)
    t10_1_diagnostic = _outcome_grounded_diagnostic(
        tuple((candidate, candidate) for candidate in t10_1_candidates),
        t10_1_rows,
    )
    baseline_binding_verified = bool(
        manifest.get("baseline_commit") == BASELINE_COMMIT
        and manifest.get("baseline_frozen_code_sha256") == BASELINE_FROZEN_SHA256
    )
    results["t10_1_behavior_frozen_baseline"].update(
        {
            "completed": True,
            "passed": baseline_binding_verified
            and t10_1_diagnostic.get("completed") is True,
            "execution_mode": "offline_behavior_frozen_replay",
            "active_execution": False,
            "execution_attempted": True,
            "behavior_contract": (
                "frozen_repeat_only_with_immediate_noop_deduplication"
            ),
            "baseline_commit": BASELINE_COMMIT,
            "baseline_frozen_code_sha256": dict(BASELINE_FROZEN_SHA256),
            "code_binding_verified": baseline_binding_verified,
            "input_actions": len(fresh),
            "deduplicated_noops": noop_duplicates,
            "executed_actions": int(t10_1_diagnostic.get("observations", 0)),
            "observation_count": int(t10_1_diagnostic.get("observations", 0)),
            "candidate_count": len(t10_1_candidates),
            "progression_ranks": t10_1_diagnostic.get("progression_ranks", []),
            "diagnostic": t10_1_diagnostic,
            "reason": (
                "offline behavior-frozen replay over compact executed source sequences"
                if baseline_binding_verified
                and t10_1_diagnostic.get("completed") is True
                else (
                    "offline behavior-frozen replay failed: frozen code binding "
                    "or compact structural observations unavailable"
                )
            ),
        }
    )
    confirmation_events = [
        event
        for event in fresh
        if event.get("split") == "leave_one_game_out_confirmation"
    ]
    learned_capacity = {
        int(selection.get("posterior_candidate_count", 0) or 0)
        for event in confirmation_events
        if isinstance((selection := event.get("selection", {})), Mapping)
        and selection.get("controller") == "learned"
    }
    independent_capacity = {
        int(selection.get("posterior_candidate_count", 0) or 0)
        for event in confirmation_events
        if isinstance((selection := event.get("selection", {})), Mapping)
        and selection.get("controller") == "capacity_matched_independent"
    }
    learned_class_capacity = {
        int(selection.get("posterior_class_capacity", 0) or 0)
        for event in confirmation_events
        if isinstance((selection := event.get("selection", {})), Mapping)
        and selection.get("controller") == "learned"
    }
    independent_class_capacity = {
        int(selection.get("posterior_class_capacity", 0) or 0)
        for event in confirmation_events
        if isinstance((selection := event.get("selection", {})), Mapping)
        and selection.get("controller") == "capacity_matched_independent"
    }
    independent_selections = tuple(
        selection
        for event in confirmation_events
        if isinstance((selection := event.get("selection", {})), Mapping)
        and selection.get("controller") == "capacity_matched_independent"
    )
    learned_observations = sum(
        isinstance(event.get("selection"), Mapping)
        and event["selection"].get("controller") == "learned"
        and int(event["selection"].get("online_posterior_observation_index", 0) or 0)
        > 0
        for event in confirmation_events
    )
    independent_observations = sum(
        isinstance(event.get("selection"), Mapping)
        and event["selection"].get("controller") == "capacity_matched_independent"
        and int(event["selection"].get("online_posterior_observation_index", 0) or 0)
        > 0
        for event in confirmation_events
    )
    learned_donor_observations = {
        int(selection.get("posterior_observation_count", 0) or 0)
        for event in confirmation_events
        if isinstance((selection := event.get("selection", {})), Mapping)
        and selection.get("controller") == "learned"
    }
    independent_donor_observations = {
        int(selection.get("posterior_observation_count", 0) or 0)
        for event in confirmation_events
        if isinstance((selection := event.get("selection", {})), Mapping)
        and selection.get("controller") == "capacity_matched_independent"
    }
    schedule_checks = confirmation.get("schedule_checks", {})
    capacity_matched = bool(
        isinstance(schedule_checks, Mapping)
        and schedule_checks.get("effective_capacity_matched_by_fold") is True
    )
    factorized_control_verified = bool(
        independent_selections
        and all(
            selection.get("posterior_family")
            == "strict_five_factor_variational_control"
            and selection.get("factorized_bank_capacity_matched") is True
            and selection.get("factorized_mdl_prior_preserved") is True
            and not str(selection.get("factorized_control_refusal", ""))
            and int(selection.get("factorized_target_particles", 0) or 0)
            == int(selection.get("posterior_candidate_count", 0) or 0)
            and int(selection.get("factorized_target_classes", 0) or 0)
            == int(selection.get("posterior_class_capacity", 0) or 0)
            for selection in independent_selections
        )
    )
    results["capacity_matched_independent_posterior"].update(
        {
            "completed": True,
            "passed": int(confirmation["complete_pairs"])
            == int(confirmation["expected_pairs"])
            and confirmation.get("schedule_passed") is True
            and capacity_matched
            and learned_observations > 0
            and independent_observations > 0
            and learned_donor_observations == independent_donor_observations
            and bool(learned_donor_observations)
            and factorized_control_verified,
            "paired_units": int(confirmation["complete_pairs"]),
            "expected_paired_units": int(confirmation["expected_pairs"]),
            "cross_fit_schedule_checks": dict(schedule_checks),
            "learned_levels": int(confirmation["learned_levels"]),
            "independent_levels": int(confirmation["independent_levels"]),
            "paired_rate_ci_lower": float(confirmation["paired_rate_ci_lower"]),
            "learned_candidate_capacity": sorted(learned_capacity),
            "independent_candidate_capacity": sorted(independent_capacity),
            "learned_class_capacity": sorted(learned_class_capacity),
            "independent_class_capacity": sorted(independent_class_capacity),
            "capacity_matched": capacity_matched,
            "posterior_family": "strict_five_factor_variational_control",
            "factorized_control_verified": factorized_control_verified,
            "learned_online_observations": learned_observations,
            "independent_online_observations": independent_observations,
            "learned_donor_observations": sorted(learned_donor_observations),
            "independent_donor_observations": sorted(independent_donor_observations),
            "donor_observations_matched": bool(learned_donor_observations)
            and learned_donor_observations == independent_donor_observations,
        }
    )
    for frame_id in sorted(_FRAME_IDS):
        results[f"single_frame_{frame_id}"] = _single_frame_control(
            frame_id, fresh, candidates
        )

    exact_certificates = 0
    partial_certificates = 0
    identity_exact = 0
    nontrivial_exact_commutative_certificates = 0
    nontrivial_exact_frame_pairs: set[tuple[str, str]] = set()
    for event in fresh:
        transport = event.get("transport", {})
        if isinstance(transport, Mapping):
            exact_certificates += int(transport.get("exact_certificate_count", 0) or 0)
            partial_certificates += int(
                transport.get("partial_certificate_count", 0) or 0
            )
            identity_exact += int(
                transport.get("identity_root_certificate_exact") is True
            )
        for certificate in event.get("transport_certificates", ()):
            if not isinstance(certificate, Mapping):
                continue
            source_frame = str(certificate.get("source_frame", ""))
            target_frame = str(certificate.get("target_frame", ""))
            commutativity = certificate.get("commutativity")
            nontrivial_exact = bool(
                source_frame
                and target_frame
                and source_frame != target_frame
                and certificate.get("exact") is True
                and certificate.get("round_trip_exact") is True
                and certificate.get("certifies_gauge_equivalence") is True
                and isinstance(commutativity, Mapping)
                and commutativity.get("exact") is True
            )
            if nontrivial_exact:
                nontrivial_exact_commutative_certificates += 1
                nontrivial_exact_frame_pairs.add((source_frame, target_frame))
    results["identity_only_transport"].update(
        {
            "completed": True,
            "passed": identity_exact > 0,
            "exact_identity_events": identity_exact,
            "exact_certificate_count": exact_certificates,
        }
    )
    diagnostic_bank = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, JointGaugeHypothesis)
    )[:64]
    diagnostic_rows = _diagnostic_rows(fresh)
    reference_diagnostic = _outcome_grounded_diagnostic(
        tuple((candidate, candidate) for candidate in diagnostic_bank),
        diagnostic_rows,
    )
    for name in (
        "no_transport",
        "deterministically_permuted_transport",
        "frame_swap",
        "binding_swap",
        "dynamics_swap",
        "goal_swap",
        "option_swap",
    ):
        results[name] = _paired_ablation_result(
            name,
            candidates=diagnostic_bank,
            rows=diagnostic_rows,
        )
    results["deterministically_permuted_transport"].update(
        {
            "detected_nonexact_certificates": partial_certificates,
            "certificate_evidence_is_not_active_advantage": True,
        }
    )
    for name, key in (
        ("frame_swap", "model_view"),
        ("binding_swap", "model_view"),
        ("dynamics_swap", "model_view"),
        ("goal_swap", "outcome"),
        ("option_swap", "selection"),
    ):
        count, mismatch = _swap_mismatch_rate(fresh, key)
        results[name].update(
            {
                "component_pairs": count,
                "structural_mismatch_rate": mismatch,
            }
        )
    results["early_map_collapse"].update(
        {
            "completed": True,
            "passed": posterior is not None
            and not bool(getattr(posterior, "collapsed", True)),
            "collapsed": bool(getattr(posterior, "collapsed", False)),
            "observation_count": posterior_observations,
        }
    )
    signatures = [
        _stable_hash(
            {
                "model": event.get("model_view"),
                "outcome": event.get("outcome"),
                "action": event.get("selection", {}).get("action_name")
                if isinstance(event.get("selection"), Mapping)
                else "",
            }
        )
        for event in fresh
    ]
    duplicate_count = len(signatures) - len(set(signatures))
    results["immediate_noop_deduplication"].update(
        {
            "completed": True,
            "passed": True,
            "events": len(signatures),
            "deduplicable_events": max(duplicate_count, noop_duplicates),
            "deduplication_applied_to_t10_1_control": True,
            "retained_t10_1_events": len(deduplicated_fresh),
        }
    )
    executed_source_actions = sum(
        isinstance(event.get("action"), Mapping)
        and event["action"].get("executed") is True
        for event in fresh
    )
    reference_observations = int(reference_diagnostic.get("observations", 0))
    reference_errors = list(reference_diagnostic.get("errors", ()))
    offline_evidence = {
        "execution_mode": "offline_observed_execution_evidence",
        "active_execution": False,
        "executed_actions": executed_source_actions,
    }
    best_sequence = grammar.get("best_executed_sequence", {})
    if not isinstance(best_sequence, Mapping):
        best_sequence = {}
    best_sequence_hash = str(best_sequence.get("sequence_hash", ""))
    executed_sequence_hashes = {
        str(value) for value in grammar.get("executed_sequence_hashes", ())
    }
    results["best_executed_sequence_oracle"].update(
        {
            "completed": True,
            "passed": int(best_sequence.get("actions", 0)) > 0
            and int(best_sequence.get("levels", 0)) > 0
            and int(best_sequence.get("errors", 1)) == 0
            and int(best_sequence.get("illegal_actions", 1)) == 0
            and int(best_sequence.get("game_overs", 1)) == 0
            and best_sequence.get("all_actions_executed") is True
            and best_sequence.get("steps_contiguous") is True
            and best_sequence.get("progress_at_end") is True
            and best_sequence.get("oracle_eligible") is True
            and bool(best_sequence_hash)
            and best_sequence_hash in executed_sequence_hashes,
            "execution_mode": "offline_best_executed_sequence",
            "active_execution": False,
            "actions": int(best_sequence.get("actions", 0)),
            "levels": int(best_sequence.get("levels", 0)),
            "errors": int(best_sequence.get("errors", 0)),
            "illegal_actions": int(best_sequence.get("illegal_actions", 0)),
            "game_overs": int(best_sequence.get("game_overs", 0)),
            "all_actions_executed": (best_sequence.get("all_actions_executed") is True),
            "steps_contiguous": best_sequence.get("steps_contiguous") is True,
            "progress_at_end": best_sequence.get("progress_at_end") is True,
            "hash_is_executed": best_sequence_hash in executed_sequence_hashes,
            "oracle_sequence_hash": best_sequence_hash,
            "evidence": "single_best_sequence_selected_from_executed_resets_only",
        }
    )
    grammar_passed = _grammar_oracle_passes(grammar)
    results["grammar_oracle"].update(
        {
            "completed": True,
            "passed": grammar_passed,
            "execution_mode": "offline_best_executed_sequence_per_game",
            "active_execution": False,
            "actions": int(grammar["actions"]),
            "levels": int(grammar["levels"]),
            "progress_games": int(grammar["progress_games"]),
            "errors": int(grammar["errors"]),
            "illegal_actions": int(grammar["illegal_actions"]),
            "game_overs": int(grammar["game_overs"]),
            "oracle_sequences": int(grammar["oracle_sequences"]),
            "oracle_sequence_hashes": grammar["oracle_sequence_hashes"],
            "executed_sequence_hashes": grammar["executed_sequence_hashes"],
            "evidence": "best_sequences_selected_from_executed_resets_only",
        }
    )
    certified_orbit_witness_candidates = sum(
        isinstance(transport, TransportOrbitWitness)
        and transport.certifies_gauge_equivalence
        and transport.source_frame_id != transport.target_frame_id
        for candidate in candidates
        for transport in getattr(candidate, "transports", ())
    )
    posterior_classes = tuple(getattr(posterior, "classes", ())) if posterior else ()
    merged_gauge_classes = 0
    maximum_gauge_variants = 0
    for gauge_class in posterior_classes:
        particles = tuple(getattr(gauge_class, "particles", ()))
        variant_hashes = {
            str(getattr(getattr(particle, "hypothesis", None), "canonical_hash", ""))
            for particle in particles
        } - {""}
        frame_ids = {
            str(
                getattr(
                    getattr(getattr(particle, "hypothesis", None), "frame", None),
                    "frame_id",
                    "",
                )
            )
            for particle in particles
        } - {""}
        variant_count = len(variant_hashes)
        maximum_gauge_variants = max(maximum_gauge_variants, variant_count)
        if variant_count > 1 and len(frame_ids) > 1:
            merged_gauge_classes += 1
    no_transport_control = results["no_transport"]
    permuted_transport_control = results["deterministically_permuted_transport"]
    transport_causal = bool(
        _paired_advantage(no_transport_control)
        and _paired_advantage(permuted_transport_control)
    )
    transport_diagnostic_errors = sorted(
        {
            str(error)
            for control in (no_transport_control, permuted_transport_control)
            for arm_name in ("reference", "ablated")
            for error in (
                control.get(arm_name, {}).get("errors", ())
                if isinstance(control.get(arm_name), Mapping)
                else ()
            )
        }
    )
    results["transport_oracle"].update(
        {
            **offline_evidence,
            "completed": True,
            "passed": executed_source_actions > 0
            and nontrivial_exact_commutative_certificates > 0
            and certified_orbit_witness_candidates > 0
            and merged_gauge_classes > 0
            and transport_causal,
            "observation_count": min(
                int(no_transport_control.get("paired_observations", 0)),
                int(permuted_transport_control.get("paired_observations", 0)),
            ),
            "exact_certificate_count": exact_certificates,
            "partial_certificate_count": partial_certificates,
            "nontrivial_exact_commutative_certificate_count": (
                nontrivial_exact_commutative_certificates
            ),
            "nontrivial_exact_frame_pair_count": len(nontrivial_exact_frame_pairs),
            "certified_orbit_witness_candidate_count": (
                certified_orbit_witness_candidates
            ),
            "posterior_merged_gauge_class_count": merged_gauge_classes,
            "maximum_gauge_variants_in_class": maximum_gauge_variants,
            "no_transport_causal_advantage": _paired_advantage(no_transport_control),
            "permuted_transport_causal_advantage": _paired_advantage(
                permuted_transport_control
            ),
            "causal_degradations": {
                "no_transport": float(no_transport_control.get("degradation", 0.0)),
                "deterministically_permuted_transport": float(
                    permuted_transport_control.get("degradation", 0.0)
                ),
            },
            "diagnostic_errors": transport_diagnostic_errors,
            "identity_only_is_insufficient": True,
            "evidence": (
                "nontrivial_exact_commutative_transport_witnesses_and_"
                "posterior_gauge_class_merging_from_executed_source_actions"
            ),
        }
    )
    dynamics_control = results["dynamics_swap"]
    dynamics_changed_events = _structural_delta_event_count(diagnostic_rows)
    dynamics_diagnostic_errors = sorted(
        {
            str(error)
            for arm_name in ("reference", "ablated")
            for error in (
                dynamics_control.get(arm_name, {}).get("errors", ())
                if isinstance(dynamics_control.get(arm_name), Mapping)
                else ()
            )
        }
    )
    results["dynamics_oracle"].update(
        {
            **offline_evidence,
            "completed": True,
            "passed": executed_source_actions > 0
            and dynamics_changed_events > 0
            and _paired_advantage(dynamics_control),
            "observation_count": int(dynamics_control.get("paired_observations", 0)),
            "structural_delta_events": dynamics_changed_events,
            "dynamics_swap_causal_advantage": _paired_advantage(dynamics_control),
            "dynamics_swap_degradation": float(
                dynamics_control.get("degradation", 0.0)
            ),
            "diagnostic_errors": dynamics_diagnostic_errors,
            "evidence": "paired_dynamics_intervention_on_executed_actions",
        }
    )
    goal_control = results["goal_swap"]
    progress_events = sum(_event_progress(event) > 0 for event, _ in diagnostic_rows)
    nonprogress_events = len(diagnostic_rows) - progress_events
    goal_diagnostic_errors = sorted(
        {
            str(error)
            for arm_name in ("reference", "ablated")
            for error in (
                goal_control.get(arm_name, {}).get("errors", ())
                if isinstance(goal_control.get(arm_name), Mapping)
                else ()
            )
        }
    )
    results["goal_oracle"].update(
        {
            **offline_evidence,
            "completed": True,
            "passed": executed_source_actions > 0
            and progress_events > 0
            and nonprogress_events > 0
            and _paired_advantage(goal_control),
            "observation_count": int(goal_control.get("paired_observations", 0)),
            "executed_progress_events": progress_events,
            "executed_nonprogress_events": nonprogress_events,
            "goal_swap_causal_advantage": _paired_advantage(goal_control),
            "goal_swap_degradation": float(goal_control.get("degradation", 0.0)),
            "diagnostic_errors": goal_diagnostic_errors,
            "evidence": "paired_goal_intervention_on_executed_outcomes",
        }
    )
    option_count = sum(
        getattr(candidate, "option", None) is not None for candidate in candidates
    )
    option_control = results["option_swap"]
    option_reference = (
        option_control.get("reference", {})
        if isinstance(option_control.get("reference"), Mapping)
        else {}
    )
    option_ablated = (
        option_control.get("ablated", {})
        if isinstance(option_control.get("ablated"), Mapping)
        else {}
    )
    option_reference_mass = float(
        option_reference.get("mean_compatible_positive_prefix_mass", 0.0)
    )
    option_ablated_mass = float(
        option_ablated.get("mean_compatible_positive_prefix_mass", 0.0)
    )
    option_diagnostic_errors = sorted(
        {
            str(error)
            for diagnostic in (option_reference, option_ablated)
            for error in diagnostic.get("errors", ())
        }
    )
    results["option_oracle"].update(
        {
            **offline_evidence,
            "completed": True,
            "passed": option_count > 0
            and _paired_advantage(option_control)
            and int(option_reference.get("positive_prefix_count", 0)) > 0
            and int(option_reference.get("compatible_positive_prefix_count", 0)) > 0
            and option_reference_mass > option_ablated_mass
            and not option_diagnostic_errors,
            "option_count": option_count,
            "observation_count": int(option_control.get("paired_observations", 0)),
            "positive_prefix_count": int(
                option_reference.get("positive_prefix_count", 0)
            ),
            "compatible_progress_observations": int(
                option_reference.get("compatible_positive_prefix_count", 0)
            ),
            "reference_positive_prefix_mass": option_reference_mass,
            "ablated_positive_prefix_mass": option_ablated_mass,
            "option_swap_causal_advantage": _paired_advantage(option_control),
            "option_swap_degradation": float(option_control.get("degradation", 0.0)),
            "diagnostic_errors": option_diagnostic_errors,
            "evidence": "paired_option_intervention_on_positive_prefixes",
        }
    )
    complete_count = sum(
        getattr(candidate, "world_program", None) is not None
        and getattr(candidate, "frame", None) is not None
        and getattr(candidate, "option", None) is not None
        for candidate in candidates
    )
    complete_control_names = (
        "frame_swap",
        "no_transport",
        "deterministically_permuted_transport",
        "binding_swap",
        "dynamics_swap",
        "goal_swap",
        "option_swap",
    )
    factor_advantages = {
        name: _paired_advantage(results[name]) for name in complete_control_names
    }
    factor_degradations = {
        name: float(results[name].get("degradation", 0.0))
        for name in complete_control_names
    }
    complete_diagnostic_errors = sorted(
        {
            *map(str, reference_errors),
            *(
                str(error)
                for name in complete_control_names
                for arm_name in ("reference", "ablated")
                for error in (
                    results[name].get(arm_name, {}).get("errors", ())
                    if isinstance(results[name].get(arm_name), Mapping)
                    else ()
                )
            ),
        }
    )
    minimum_factor_degradation = (
        min(factor_degradations.values()) if factor_degradations else 0.0
    )
    results["complete_program_oracle"].update(
        {
            **offline_evidence,
            "completed": True,
            "passed": complete_count > 0
            and reference_diagnostic.get("completed") is True
            and reference_observations > 0
            and int(reference_diagnostic.get("positive_candidate_count", 0)) > 0
            and int(reference_diagnostic.get("negative_candidate_count", 0)) > 0
            and all(factor_advantages.values())
            and minimum_factor_degradation > 0.0
            and int(option_reference.get("compatible_positive_prefix_count", 0)) > 0
            and option_reference_mass > option_ablated_mass
            and not reference_errors
            and not posterior_errors,
            "complete_candidates": complete_count,
            "observation_count": reference_observations,
            "posterior_observations": posterior_observations,
            "posterior_errors": list(posterior_errors),
            "diagnostic_errors": complete_diagnostic_errors,
            "factor_advantages": factor_advantages,
            "factor_degradations": factor_degradations,
            "minimum_factor_degradation": minimum_factor_degradation,
            "evidence": "complete_joint_program_under_paired_factor_interventions",
        }
    )
    return results


def _normalize_control_result(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach one strict execution/scientific outcome to every control."""

    normalized = dict(result)
    attempted = normalized.get("attempted", normalized.get("completed")) is True
    scientific_pass = (
        normalized.get("scientific_pass", normalized.get("passed")) is True
    )
    if "execution_ok" in normalized:
        execution_ok = normalized.get("execution_ok") is True
    elif name.startswith("single_frame_"):
        execution_ok = bool(
            attempted
            and int(normalized.get("candidate_count", 0) or 0) > 0
            and int(normalized.get("observation_count", 0) or 0) > 0
            and int(normalized.get("errors", 0) or 0) == 0
        )
    elif name in {
        "no_transport",
        "deterministically_permuted_transport",
        "frame_swap",
        "binding_swap",
        "dynamics_swap",
        "goal_swap",
        "option_swap",
    }:
        execution_ok = attempted and normalized.get("evaluable") is True
    elif name == "capacity_matched_independent_posterior":
        schedule = normalized.get("cross_fit_schedule_checks", {})
        execution_ok = bool(
            attempted
            and isinstance(schedule, Mapping)
            and schedule
            and all(value is True for value in schedule.values())
            and normalized.get("factorized_control_verified") is True
            and normalized.get("donor_observations_matched") is True
        )
    elif name == "t10_1_behavior_frozen_baseline":
        diagnostic = normalized.get("diagnostic", {})
        execution_ok = bool(
            attempted
            and normalized.get("code_binding_verified") is True
            and isinstance(diagnostic, Mapping)
            and diagnostic.get("completed") is True
            and not diagnostic.get("errors")
        )
    elif name == "early_map_collapse":
        execution_ok = bool(
            attempted and int(normalized.get("observation_count", 0) or 0) > 0
        )
    elif name == "immediate_noop_deduplication":
        execution_ok = bool(attempted and int(normalized.get("events", 0) or 0) > 0)
    elif name in {"best_executed_sequence_oracle", "grammar_oracle"}:
        execution_ok = bool(
            attempted
            and int(normalized.get("actions", 0) or 0) > 0
            and int(normalized.get("errors", 0) or 0) == 0
        )
    elif name in {
        "transport_oracle",
        "dynamics_oracle",
        "goal_oracle",
        "option_oracle",
        "complete_program_oracle",
    }:
        execution_ok = bool(
            attempted
            and int(normalized.get("executed_actions", 0) or 0) > 0
            and not normalized.get("diagnostic_errors")
            and not normalized.get("posterior_errors")
        )
    else:
        # Identity-only transport is evaluated exhaustively over the collected
        # source events; absence of an exact identity certificate is a
        # scientific failure, not an execution exception.
        execution_ok = attempted
    normalized.update(
        {
            "attempted": attempted,
            "execution_ok": execution_ok,
            "scientific_pass": scientific_pass,
            # Compatibility fields retain their old meaning for consumers,
            # while ``completed`` can no longer conceal a refused execution.
            "completed": attempted and execution_ok,
            "passed": scientific_pass,
        }
    )
    return normalized


def run_source_trainer(
    *,
    manifest: Mapping[str, Any],
    compile_report: Mapping[str, Any],
    replay_report: Mapping[str, Any],
    output_dir: str | Path,
    candidates: Sequence[Any] = (),
    control_runner: Callable[..., Any] | None = None,
    posterior_factory: Callable[..., Any] = GaugeProgramPosterior,
    bundle_builder: Callable[..., Any] | None = None,
    event_path: str | Path | None = None,
    replay_event_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit/evaluate source evidence as a ``source_train_phase`` callback.

    Missing controls, posterior observations, ranks, or oracle outcomes remain
    explicit failures.  The function never turns absent evidence into success.
    """

    destination = Path(output_dir)
    fresh_path = Path(event_path or destination / "source_events.jsonl")
    replay_path = Path(replay_event_path or destination / "replay_events.jsonl")
    fresh = read_event_ledger(fresh_path)
    replay = read_event_ledger(replay_path)
    validate_source_events(fresh, manifest=manifest, replay=False)
    validate_source_events(replay, manifest=manifest, replay=True)
    validate_source_events([*fresh, *replay], manifest=manifest, replay=None)
    cross_fit_path = destination / CROSS_FIT_AUDIT_FILENAME
    cross_fit_audit = read_cross_fit_audit(
        cross_fit_path,
        manifest=manifest,
        source_event_path=fresh_path,
        source_events=fresh,
    )
    compile_integrity_passed = bool(
        compile_report.get("status") == "PASS_T10_2_QA"
        or (
            compile_report.get("status") == "T10_2_FRESH_INTEGRITY_COMPLETE"
            and compile_report.get("integrity_passed") is True
        )
    )
    if not compile_integrity_passed:
        raise GateRefusalError("source trainer requires passing fresh-source QA")
    if replay_report.get("status") != "T10_2_SOURCE_REPLAY_COMPLETE":
        raise GateRefusalError("source trainer requires completed frozen replay")

    candidate_tuple = tuple(candidates) or _synthesize_gauge_candidates(
        fresh, maximum=256
    )
    posterior: Any = None
    posterior_observations = 0
    posterior_errors: list[str] = []
    if candidate_tuple:
        posterior = posterior_factory()
        seed_method = getattr(posterior, "seed", None)
        observe = getattr(posterior, "observe", None)
        if not callable(seed_method) or not callable(observe):
            raise DataGateError(
                "posterior requires seed(hypotheses) and observe(bundle)"
            )
        seed_method(candidate_tuple)
        branch: tuple[str, int, int] | None = None
        for event in [*fresh, *replay]:
            try:
                bundle = (
                    _invoke(
                        bundle_builder,
                        context={"event": event},
                        positional=(event,),
                    )
                    if bundle_builder is not None
                    else _bundle_from_compact_event(event)
                )
                if bundle is None:
                    continue
                selection = event.get("selection", {})
                current_branch = (
                    str(event.get("game_id", "")),
                    int(event.get("seed", -1)),
                    int(
                        selection.get("reset_index", -1)
                        if isinstance(selection, Mapping)
                        else -1
                    ),
                )
                start_branch = getattr(posterior, "start_branch", None)
                if (
                    branch is not None
                    and current_branch != branch
                    and callable(start_branch)
                ):
                    start_branch()
                branch = current_branch
                observe(bundle)
                posterior_observations += 1
            except Exception as exc:  # noqa: BLE001 - retained as negative evidence.
                posterior_errors.append(type(exc).__name__)

    confirmation = _confirmation_metrics(fresh, cross_fit_audit)
    grammar = _executed_grammar(fresh)
    all_events = (*fresh, *replay)
    control_results = _internal_control_results(
        manifest=manifest,
        fresh=fresh,
        replay=replay,
        candidates=candidate_tuple,
        posterior=posterior,
        posterior_observations=posterior_observations,
        posterior_errors=posterior_errors,
        confirmation=confirmation,
        grammar=grammar,
    )
    if control_runner is not None:
        for name in REGISTERED_SOURCE_CONTROLS:
            override = _run_control(
                control_runner,
                name,
                events=all_events,
                manifest=manifest,
                posterior=posterior,
            )
            control_results[name] = {
                **control_results[name],
                **override,
                "injected_override": True,
            }
    control_results = {
        name: _normalize_control_result(name, control_results[name])
        for name in REGISTERED_SOURCE_CONTROLS
    }
    attempted = {
        name: result["attempted"] is True for name, result in control_results.items()
    }
    execution_ok = {
        name: result["execution_ok"] is True for name, result in control_results.items()
    }
    completed = {
        name: result.get("completed") is True
        for name, result in control_results.items()
    }

    def degradation(name: str) -> float:
        result = control_results[name]
        try:
            value = float(result.get("degradation", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    def passed(name: str) -> bool:
        return bool(
            control_results[name].get("completed") is True
            and control_results[name].get("passed") is True
        )

    grammar_levels = int(grammar["levels"])
    learned_levels = int(confirmation["learned_levels"])
    probe_accuracy = _identity_probe_accuracy(fresh)
    posterior_classes = (
        len(getattr(posterior, "classes", ())) if posterior is not None else 0
    )
    option_schemas = sorted(
        {
            str(getattr(getattr(candidate, "option", None), "schema", ""))
            for candidate in candidate_tuple
            if getattr(candidate, "option", None) is not None
        }
        - {""}
    )
    candidates_with_transports = sum(
        bool(getattr(candidate, "transports", ())) for candidate in candidate_tuple
    )
    partial_transport_count = sum(
        float(getattr(transport, "coverage", 1.0)) < 1.0
        for candidate in candidate_tuple
        for transport in getattr(candidate, "transports", ())
    )
    common_posterior_passed = bool(
        posterior is not None
        and posterior_observations > 0
        and posterior_classes > 0
        and not posterior_errors
    )
    recipe_binding: dict[str, Any] = {
        "bound": False,
        "path": CHALLENGER_RECIPE_FILENAME,
        "reason": "non_default_or_incomplete_source_fit",
    }
    default_recipe_fit = bool(
        not candidates
        and posterior_factory is GaugeProgramPosterior
        and bundle_builder is None
        and posterior is not None
        and candidate_tuple
        and posterior_observations > 0
        and not posterior_errors
        and fresh_path.resolve() == (destination / "source_events.jsonl").resolve()
        and replay_path.resolve() == (destination / "replay_events.jsonl").resolve()
    )
    if default_recipe_fit:
        recipe_binding = _write_challenger_recipe(
            manifest=manifest,
            output_dir=destination,
            fresh_path=fresh_path,
            replay_path=replay_path,
            fresh=fresh,
            replay=replay,
            candidates=candidate_tuple,
            posterior=posterior,
            posterior_observations=posterior_observations,
        )
    metrics = {
        "attempted_controls": attempted,
        "execution_ok_controls": execution_ok,
        "completed_controls": completed,
        "grammar_oracle": grammar,
        "learned": {
            "positive_fold_ranks": confirmation["positive_fold_ranks"],
            "oracle_level_recovery": (
                min(1.0, learned_levels / grammar_levels) if grammar_levels else 0.0
            ),
            "nonnegative_games": confirmation["nonnegative_games"],
            "paired_rate_ci_lower": confirmation["paired_rate_ci_lower"],
            "game_seed_probe_accuracy_increment": max(
                0.0, probe_accuracy - (1.0 / len(SOURCE_GAMES))
            ),
            "illegal_actions": grammar["all_executed_illegal_actions"],
            "errors": grammar["all_executed_errors"] + len(posterior_errors),
            "common_posterior_passed": common_posterior_passed,
            "option_synthesis_passed": passed("option_oracle"),
        },
        "controls": {
            "no_transport_degradation": degradation("no_transport"),
            "binding_swap_degradation": degradation("binding_swap"),
            "capacity_matched_independent_posterior_passed": passed(
                "capacity_matched_independent_posterior"
            ),
            "transport_oracle_passed": passed("transport_oracle"),
            "dynamics_oracle_passed": passed("dynamics_oracle"),
            "goal_oracle_passed": passed("goal_oracle"),
            "best_executed_sequence_oracle_passed": passed(
                "best_executed_sequence_oracle"
            ),
            "option_oracle_passed": passed("option_oracle"),
            "complete_program_oracle_passed": passed("complete_program_oracle"),
        },
        "control_results": control_results,
        "source_evidence": {
            "format_version": "sage-t10.2-source-control-evidence-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "fresh_events": artifact_descriptor(fresh_path),
            "replay_events": artifact_descriptor(replay_path),
            "cross_fit_audit": artifact_descriptor(cross_fit_path),
            "cross_fit_audit_checksum": cross_fit_audit["audit_checksum"],
            "fresh_event_ids_sha256": canonical_sha256(
                [str(row["event_id"]).strip() for row in fresh]
            ),
            "replay_event_ids_sha256": canonical_sha256(
                [str(row["event_id"]).strip() for row in replay]
            ),
            "combined_event_ids_sha256": canonical_sha256(
                [
                    *(
                        {
                            "ledger": "fresh",
                            "event_id": str(row["event_id"]).strip(),
                        }
                        for row in fresh
                    ),
                    *(
                        {
                            "ledger": "replay",
                            "event_id": str(row["event_id"]).strip(),
                        }
                        for row in replay
                    ),
                ]
            ),
            "control_results_sha256": canonical_sha256(control_results),
        },
        "posterior": {
            "implementation": (None if posterior is None else type(posterior).__name__),
            "candidate_count": len(candidate_tuple),
            "observation_count": posterior_observations,
            "class_count": posterior_classes,
            "errors": posterior_errors,
            "used": posterior is not None,
            "maximum_candidate_budget": 256,
            "decision_class_budget": 64,
            "option_schemas": option_schemas,
            "candidates_with_transports": candidates_with_transports,
            "partial_transport_count": partial_transport_count,
        },
        "paired_active": confirmation,
        "cross_fit_audit": {
            "artifact": artifact_descriptor(cross_fit_path),
            "audit_checksum": cross_fit_audit["audit_checksum"],
            "registered_unit_count": cross_fit_audit["registered_unit_count"],
            "checks": dict(cross_fit_audit["checks"]),
            "passed": cross_fit_audit["passed"],
        },
        "safety_gate_passed": grammar["all_executed_illegal_actions"] == 0
        and grammar["all_executed_errors"] == 0
        and grammar["all_executed_game_overs"] == 0
        and not posterior_errors,
        "resource_gate_passed": True,
        "challenger_recipe": recipe_binding,
        "firewall": {
            "source_only": True,
            "validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
        },
    }
    return _clone(metrics)


def _verified_source_report(
    source_report: Mapping[str, Any] | str | Path,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = (
        read_checked_json(source_report)
        if isinstance(source_report, (str, Path))
        else dict(source_report)
    )
    if manifest is not None and report.get("manifest_checksum") != manifest.get(
        "manifest_checksum"
    ):
        raise ManifestDriftError("source report/manifest binding drifted")
    checks = report.get("checks", {})
    firewall = report.get("firewall", {})
    if (
        report.get("status") != "PASS_T10_2_SOURCE_GATE"
        or report.get("verdict") != "PASS_T10_2_SOURCE_GATE"
        or report.get("passed") is not True
        or not isinstance(checks, Mapping)
        or not all(value is True for value in checks.values())
        or not isinstance(firewall, Mapping)
        or firewall.get("source_validation_opened") is not True
        or any(
            firewall.get(name) is True
            for name in ("ar25_opened", "holdout_opened", "production_authority")
        )
    ):
        raise GateRefusalError("validation requires a verified source PASS")
    return _clone(report)


def _load_t10_1_behavior_binding(repo_root: str | Path) -> dict[str, Any]:
    """Verify T10.1 without reading its source-validation result."""

    root = Path(repo_root).resolve()
    manifest_path = (
        root / "theory" / "sage_t" / "sage_t10_1_source_validation_manifest.json"
    )
    manifest = read_checked_json(
        manifest_path,
        checksum_key="manifest_checksum",
        require_canonical=False,
    )
    if manifest.get("manifest_checksum") != T10_1_FROZEN_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.1 frozen manifest checksum drifted")
    if (
        manifest.get("format_version") != "sage-t10.1-source-validation-progress-v1"
        or manifest.get("status") != "FROZEN_BEFORE_T10_1_SOURCE_VALIDATION"
        or manifest.get("source_validation_games") != list(VALIDATION_GAMES)
    ):
        raise ManifestDriftError("T10.1 frozen behavior manifest drifted")
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping) or dict(policy) != {
        "adaptation_after_results": False,
        "control_family": "repeat_apply_until_progress",
        "game_source_code_allowed": False,
        "hidden_counterfactual_arms_allowed": False,
        "stored_winning_paths_allowed": False,
        "stop_condition": "first positive level delta",
    }:
        raise ManifestDriftError("T10.1 policy contract drifted")
    firewall = manifest.get("firewall")
    if not isinstance(firewall, Mapping) or (
        firewall.get("source_validation_opened") is not True
        or any(
            firewall.get(name) is True
            for name in ("ar25_opened", "holdout_opened", "production_authority")
        )
    ):
        raise ManifestDriftError("T10.1 firewall drifted")
    code = manifest.get("code_sha256")
    if not isinstance(code, Mapping):
        raise ManifestDriftError("T10.1 code binding is absent")
    current = {name: file_sha256(root / "theory" / "sage_t" / name) for name in code}
    if current != dict(code):
        raise ManifestDriftError("T10.1 behavior code drifted")
    parent_path = (
        root / "training" / "sage_t" / "progress_witness_v10_0b" / "report.json"
    )
    parent = read_checked_json(
        parent_path,
        checksum_key="report_checksum",
        require_canonical=False,
    )
    if parent.get("status") != "PASS_T10_0_AUTHORIZE_T10_1" or parent.get(
        "report_checksum"
    ) != manifest.get("parent_t10_0b_report_checksum"):
        raise ManifestDriftError("T10.1 parent source gate drifted")
    return _clone(manifest)


class T10_1BehaviorFrozenPolicyFactory:
    """Build the literal T10.1 search projection allowed by 14 resets.

    The projection uses one registered reset for each one-step effect probe,
    then one reset for each frozen ``candidate_macros`` result.  It refuses
    before its first action whenever the complete candidate set cannot fit;
    there is no repeat-first or historical-result fallback.
    """

    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(
            repo_root or Path(__file__).resolve().parents[2]
        ).resolve()
        self.binding = _load_t10_1_behavior_binding(self.repo_root)
        from .progress_witness_v10 import SearchConfig

        self.config = SearchConfig(**dict(self.binding["search_config"]))

    def __call__(
        self,
        *,
        controller: str,
        game_id: str,
        seed: int,
        posterior_reset: bool,
        learning_enabled: bool,
        **_context: Any,
    ) -> _T10_1PairedProjectionPolicy:
        if controller != "t10_1":
            raise DataGateError("T10.1 factory received another controller")
        if game_id not in VALIDATION_GAMES or int(seed) not in VALIDATION_SEEDS:
            raise FirewallError("T10.1 factory received an unregistered pair")
        if posterior_reset is not True or learning_enabled is not False:
            raise DataGateError("T10.1 validation must be frozen and reset")
        return _T10_1PairedProjectionPolicy(
            game_id=game_id,
            seed=int(seed),
            config=self.config,
            manifest_checksum=str(self.binding["manifest_checksum"]),
        )


class _T10_1PairedProjectionPolicy:
    def __init__(
        self,
        *,
        game_id: str,
        seed: int,
        config: Any,
        manifest_checksum: str,
    ) -> None:
        self.game_id = game_id
        self.seed = seed
        self.config = config
        self.manifest_checksum = manifest_checksum
        self._reset_index = -1
        self._inventory: tuple[Any, ...] = ()
        self._scan_count = 0
        self._initial_state: AbstractState | None = None
        self._scan_rows: list[Any] = []
        self._macros: tuple[Any, ...] | None = None
        self._mode = "uninitialized"
        self._last_scan_action: Any | None = None
        self._last_noop_signature: tuple[Any, ...] | None = None
        self._witness_found = False
        self.retained_observations = 0
        self.deduplicated_immediate_noops = 0

    @property
    def behavior_projection(self) -> str:
        return "exact_one_step_scan_then_frozen_candidate_macros_across_14_resets"

    def reset(
        self,
        *,
        reset_index: int,
        posterior_reset: bool,
        learning_enabled: bool,
    ) -> None:
        if bool(posterior_reset) is not (int(reset_index) == 0):
            raise DataGateError("T10.1 paired reset schedule drifted")
        if learning_enabled is not False:
            raise DataGateError("T10.1 cannot learn during validation")
        if int(reset_index) != self._reset_index + 1:
            raise DataGateError("T10.1 resets must be sequential")
        self._reset_index = int(reset_index)
        self._mode = "pending"
        self._last_scan_action = None

    def _initialize(self, snapshot: Any, legal_actions: Sequence[Any]) -> None:
        from theory.live_transition_loop import build_observation

        from .compiler import compile_observation
        from .progress_witness_v10 import GroundedAction, chain_successor_macro

        grid = _value(snapshot, "grid")
        if grid is None:
            raise RuntimeUnavailableError(
                "exact T10.1 projection requires the raw grid transiently"
            )
        self._inventory = tuple(
            GroundedAction.from_view(action) for action in legal_actions
        )
        self._scan_count = min(
            len(self._inventory), int(self.config.maximum_one_step_actions)
        )
        observation = build_observation(
            grid,
            available_actions=tuple(item.action_name for item in self._inventory),
            game_state=_snapshot_state(snapshot),
            levels_completed=_snapshot_levels(snapshot),
        )
        self._initial_state = compile_observation(observation)
        path_count = int(
            chain_successor_macro(
                self._initial_state,
                self._inventory,
                config=self.config,
            )
            is not None
        )
        maximum_effect_macros = min(
            self._scan_count,
            int(self.config.maximum_effect_representatives),
        )
        maximum_macros = min(
            int(self.config.maximum_candidate_macros),
            path_count + maximum_effect_macros,
        )
        if self._scan_count + maximum_macros > VALIDATION_RESETS_PER_GAME_SEED:
            raise RuntimeUnavailableError(
                "exact T10.1 scan+macro projection exceeds the registered 14 resets"
            )

    @staticmethod
    def _concrete_action(wanted: Any, legal_actions: Sequence[Any]) -> Any:
        from .progress_witness_v10 import GroundedAction

        matches = [
            action
            for action in legal_actions
            if GroundedAction.from_view(action).key == wanted.key
        ]
        if len(matches) == 1:
            return matches[0]
        return {"action_name": wanted.action_name, "action_data": wanted.data}

    def select_action(
        self,
        *,
        legal_actions: Sequence[Any],
        snapshot: Any,
        reset_index: int,
        step_index: int,
        learning_enabled: bool,
        **_context: Any,
    ) -> Any | None:
        if learning_enabled is not False:
            raise DataGateError("T10.1 cannot learn during validation")
        if int(reset_index) != self._reset_index:
            raise DataGateError("T10.1 decision/reset context drifted")
        if not self._inventory:
            self._initialize(snapshot, legal_actions)
        if self._witness_found:
            self._mode = "complete"
            return None
        if self._reset_index < self._scan_count:
            self._mode = "scan"
            if int(step_index) > 0:
                return None
            wanted = self._inventory[self._reset_index]
            self._last_scan_action = wanted
            return self._concrete_action(wanted, legal_actions)
        if self._macros is None:
            from .progress_witness_v10 import candidate_macros

            if len(self._scan_rows) != self._scan_count or self._initial_state is None:
                raise DataGateError("T10.1 macro generation lacks its complete scan")
            self._macros = candidate_macros(
                self._initial_state,
                self._inventory,
                tuple(self._scan_rows),
                config=self.config,
            )
            if self._scan_count + len(self._macros) > VALIDATION_RESETS_PER_GAME_SEED:
                raise RuntimeUnavailableError(
                    "exact T10.1 candidate macros exceed the registered resets"
                )
        macro_index = self._reset_index - self._scan_count
        if macro_index >= len(self._macros):
            self._mode = "complete"
            return None
        self._mode = "macro"
        macro = self._macros[macro_index]
        if int(step_index) >= len(macro.actions):
            return None
        return self._concrete_action(macro.actions[int(step_index)], legal_actions)

    def registered_stop_reason(self, **_context: Any) -> str | None:
        """Declare only the preregistered exhaustion of this frozen option."""

        if self._mode in {"scan", "macro", "complete"}:
            return "option_exhausted"
        return None

    @staticmethod
    def _effect_row(before: Any, after: Any, action: Any) -> Any:
        import numpy as np

        from .progress_witness_v10 import (
            EffectScanRow,
            GroundedAction,
            _sha256_payload,
            _terminal,
            _visual_digest,
        )

        before_grid = np.asarray(_value(before, "grid"))
        after_grid = np.asarray(_value(after, "grid"))
        changed = (
            int(np.sum(before_grid != after_grid))
            if before_grid.shape == after_grid.shape
            else int(max(before_grid.size, after_grid.size))
        )
        delta = max(0, _snapshot_levels(after) - _snapshot_levels(before))
        effect_key = _sha256_payload(
            {
                "visual": _visual_digest(after_grid),
                "changed": changed,
                "level_delta": delta,
                "terminal": _terminal(_snapshot_state(after)),
            }
        )[:20]
        return EffectScanRow(
            action=GroundedAction.from_view(action),
            effect_key=effect_key,
            changed_cells=changed,
            level_delta=delta,
            game_state=_snapshot_state(after),
            latency_ms=0.0,
        )

    def observe(
        self,
        *,
        before: Any,
        after: Any,
        action: Any,
        learning_enabled: bool,
        **_context: Any,
    ) -> None:
        if learning_enabled is not False:
            raise DataGateError("T10.1 cannot learn during validation")
        row = self._effect_row(before, after, action)
        if self._mode == "scan":
            if (
                self._last_scan_action is None
                or row.action.key != self._last_scan_action.key
            ):
                raise DataGateError("T10.1 scan grounding drifted")
            self._scan_rows.append(row)
        from .progress_witness_v10 import _visual_digest

        signature = (
            self._reset_index,
            row.action.action_name,
            _visual_digest(_value(before, "grid")),
        )
        no_effect = row.changed_cells == 0 and row.level_delta == 0
        if no_effect and signature == self._last_noop_signature:
            self.deduplicated_immediate_noops += 1
        else:
            self.retained_observations += 1
            self._last_noop_signature = signature
        if self._mode == "macro" and row.level_delta > 0:
            self._witness_found = True


class T10_2GaugePolicyFactory:
    """Reconstruct a fresh source-bound posterior for every validation pair."""

    def __init__(
        self,
        *,
        source_report: Mapping[str, Any] | str | Path,
        manifest: Mapping[str, Any],
        output_dir: str | Path,
        bundle_builder: Callable[..., Any] | None = None,
    ) -> None:
        self.manifest = _clone(manifest)
        self.output_dir = Path(output_dir)
        self.source_report = _verified_source_report(source_report, manifest)
        self.bundle_builder = bundle_builder
        metrics = self.source_report.get("metrics")
        binding = (
            metrics.get("challenger_recipe") if isinstance(metrics, Mapping) else None
        )
        if not isinstance(binding, Mapping) or binding.get("bound") is not True:
            raise GateRefusalError("source PASS lacks a bound challenger recipe")
        if binding.get("path") != CHALLENGER_RECIPE_FILENAME:
            raise ManifestDriftError("source report challenger recipe path drifted")
        recipe_path = self.output_dir / CHALLENGER_RECIPE_FILENAME
        if binding.get("artifact") != artifact_descriptor(recipe_path):
            raise ManifestDriftError("source report challenger recipe drifted")
        self.recipe = read_checked_json(
            recipe_path,
            checksum_key="recipe_checksum",
        )
        if binding.get("recipe_checksum") != self.recipe.get("recipe_checksum"):
            raise ManifestDriftError("source report challenger checksum drifted")
        inputs = self.source_report.get("inputs")
        ledgers = self.recipe.get("ledgers")
        if not isinstance(inputs, Mapping) or not isinstance(ledgers, Mapping):
            raise ManifestDriftError("source report/recipe ledger binding is absent")
        for report_key, recipe_key in (
            ("fresh_events", "source_events"),
            ("replay_events", "replay_events"),
        ):
            entry = ledgers.get(recipe_key)
            if not isinstance(entry, Mapping) or inputs.get(report_key) != entry.get(
                "artifact"
            ):
                raise ManifestDriftError("source report/recipe ledger binding drifted")
        # Full deterministic reconstruction is deliberately completed before
        # any validation environment can be opened.
        _rebuild_challenger_posterior(
            recipe=self.recipe,
            output_dir=self.output_dir,
            manifest=self.manifest,
        )

    def __call__(
        self,
        *,
        controller: str,
        game_id: str,
        seed: int,
        posterior_reset: bool,
        learning_enabled: bool,
        runtime: Any,
        **_context: Any,
    ) -> _FrozenGaugeValidationPolicy:
        if controller != "t10_2":
            raise DataGateError("T10.2 factory received another controller")
        if game_id not in VALIDATION_GAMES or int(seed) not in VALIDATION_SEEDS:
            raise FirewallError("T10.2 factory received an unregistered pair")
        if posterior_reset is not True or learning_enabled is not False:
            raise DataGateError("T10.2 validation must be frozen and reset")
        posterior, bank = _rebuild_challenger_posterior(
            recipe=self.recipe,
            output_dir=self.output_dir,
            manifest=self.manifest,
        )
        return _FrozenGaugeValidationPolicy(
            game_id=game_id,
            seed=int(seed),
            runtime=runtime,
            posterior=posterior,
            candidate_bank=bank,
            bundle_builder=self.bundle_builder,
        )


class _FrozenGaugeValidationPolicy:
    def __init__(
        self,
        *,
        game_id: str,
        seed: int,
        runtime: Any,
        posterior: GaugeProgramPosterior,
        candidate_bank: Sequence[JointGaugeHypothesis],
        bundle_builder: Callable[..., Any] | None,
    ) -> None:
        self.game_id = game_id
        self.seed = seed
        self.runtime = runtime
        self.posterior = posterior
        self.candidate_bank = tuple(candidate_bank)
        self.bundle_builder = bundle_builder
        self.engine = GaugeDecisionEngine(
            maximum_classes=64,
            maximum_option_horizon=16,
        )
        self.frame_states: dict[str, AbstractState] = {}
        self._reset_index = -1
        self._step_index = -1
        self._last_legal: tuple[Any, ...] = ()
        self.online_observations = 0

    @property
    def behavior_projection(self) -> str:
        return "frozen_source_gauge_posterior_observe_update_replan"

    def reset(
        self,
        *,
        reset_index: int,
        posterior_reset: bool,
        learning_enabled: bool,
    ) -> None:
        if bool(posterior_reset) is not (int(reset_index) == 0):
            raise DataGateError("T10.2 paired reset schedule drifted")
        if learning_enabled is not False:
            raise DataGateError("T10.2 grammar/priors are frozen in validation")
        if int(reset_index) != self._reset_index + 1:
            raise DataGateError("T10.2 resets must be sequential")
        self._reset_index = int(reset_index)
        self._step_index = -1
        self._last_legal = ()
        self.frame_states = {}
        self.posterior.start_branch()

    def select_action(
        self,
        *,
        legal_actions: Sequence[Any],
        environment: Any,
        reset_index: int,
        step_index: int,
        learning_enabled: bool,
        **_context: Any,
    ) -> Any | None:
        if learning_enabled is not False:
            raise DataGateError("T10.2 grammar/priors are frozen in validation")
        if int(reset_index) != self._reset_index:
            raise DataGateError("T10.2 decision/reset context drifted")
        legal = tuple(legal_actions)
        if not legal:
            return None
        self._step_index = int(step_index)
        self._last_legal = legal
        candidates = tuple(
            ActionCandidate(_action_name(action), _action_data(action))
            for action in legal
        )
        by_key = {
            candidate.key: action
            for candidate, action in zip(candidates, legal, strict=True)
        }
        fallback_raw = min(legal, key=_grounding_key)
        fallback = ActionCandidate(
            _action_name(fallback_raw),
            _action_data(fallback_raw),
        )
        danger_method = next(
            (
                getattr(self.runtime, name)
                for name in ("is_dangerous_action", "danger_veto")
                if callable(getattr(self.runtime, name, None))
            ),
            None,
        )

        def danger_veto(candidate: ActionCandidate) -> bool:
            if danger_method is None:
                return False
            raw = by_key.get(candidate.key)
            return bool(
                _invoke(
                    danger_method,
                    context={
                        "environment": environment,
                        "env": environment,
                        "action": raw,
                    },
                    positional=(environment, raw),
                )
            )

        decision = self.engine.decide(
            self.posterior,
            self.frame_states,
            candidates,
            danger_veto=danger_veto,
            fallback_action=fallback,
        )
        if decision.action is None:
            return None
        selected = by_key.get(decision.action.key)
        if selected is None:
            raise DataGateError("T10.2 decision escaped legal grounding")
        return selected

    def observe(
        self,
        *,
        before: Any,
        after: Any,
        action: Any,
        learning_enabled: bool,
        reset_index: int | None = None,
        step_index: int | None = None,
        legal_actions: Sequence[Any] | None = None,
        **_context: Any,
    ) -> None:
        if learning_enabled is not False:
            raise DataGateError("T10.2 grammar/priors are frozen in validation")
        reset = self._reset_index if reset_index is None else int(reset_index)
        step = self._step_index if step_index is None else int(step_index)
        if reset != self._reset_index or step != self._step_index:
            raise DataGateError("T10.2 observation/decision context drifted")
        legal = tuple(legal_actions or self._last_legal)
        event_id = _stable_hash(
            {
                "runtime": RUNTIME_FORMAT_VERSION,
                "controller": "t10_2_frozen_validation",
                "pair": [self.game_id, self.seed],
                "reset": reset,
                "step": step,
            }
        )
        bundle = _make_bundle(
            self.bundle_builder,
            before=before,
            after=after,
            action=action,
            legal_actions=legal,
            event_id=event_id,
            step_index=step,
            game_id=self.game_id,
        )
        self.posterior.observe(bundle)
        self.online_observations += 1
        self.frame_states = {
            projection.frame_id: projection.after.state
            for projection in bundle.projections
        }


class T10_2ValidationFactory:
    """Paired, behavior-frozen validation adapter.

    There is intentionally no approximate T10.1 fallback.  The caller must
    inject the exact behavior-frozen T10.1 policy factory and the frozen T10.2
    candidate factory; otherwise ``run_validation`` refuses before opening an
    environment.
    """

    def __init__(
        self,
        *,
        source_report: Mapping[str, Any] | str | Path,
        manifest: Mapping[str, Any] | None = None,
        runtime_loader: Callable[[], Any] | None = None,
        t10_1_policy_factory: Callable[..., Any] | None = None,
        t10_2_policy_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.source_report = _verified_source_report(source_report, manifest)
        self._runtime_loader = runtime_loader or _default_runtime_loader
        self._runtime: Any | None = None
        self._policy_factories = {
            "t10_1": t10_1_policy_factory,
            "t10_2": t10_2_policy_factory,
        }
        self.limitations = tuple(
            message
            for controller, message in (
                (
                    "t10_1",
                    "exact behavior-frozen T10.1 policy factory is not configured",
                ),
                ("t10_2", "frozen T10.2 policy factory is not configured"),
            )
            if self._policy_factories[controller] is None
        )

    @property
    def runtime_loaded(self) -> bool:
        return self._runtime is not None

    def _load_runtime(self) -> Any:
        if self.limitations:
            raise RuntimeUnavailableError("; ".join(self.limitations))
        if self._runtime is None:
            self._runtime = self._runtime_loader()
        if self._runtime is None:
            raise RuntimeUnavailableError(
                "runtime loader returned no validation runtime"
            )
        return self._runtime

    def __call__(
        self,
        game_id: str,
        seed: int,
        phase: str = "validate",
        split: str | None = None,
        **_context: Any,
    ) -> _ValidationPairEnvironment:
        del split
        enforce_environment_firewall(
            phase=phase,
            game_id=game_id,
            source_gate_passed=True,
        )
        return _ValidationPairEnvironment(self, game_id=game_id, seed=int(seed))


class _ValidationPairEnvironment:
    def __init__(
        self, factory: T10_2ValidationFactory, *, game_id: str, seed: int
    ) -> None:
        self.factory = factory
        self.game_id = game_id
        self.seed = seed
        self._opened: list[Any] = []

    def run_validation(
        self,
        *,
        game_id: str,
        seed: int,
        controller_order: Sequence[str],
        resets: int,
        action_budget: int,
        posterior_reset: bool,
        learning_enabled: bool,
        **_context: Any,
    ) -> dict[str, Any]:
        if (game_id, int(seed)) != (self.game_id, self.seed):
            raise FirewallError("validation pair context drifted")
        order = tuple(controller_order)
        if order not in {("t10_1", "t10_2"), ("t10_2", "t10_1")}:
            raise DataGateError(
                "validation order must contain each frozen controller once"
            )
        if int(resets) != VALIDATION_RESETS_PER_GAME_SEED:
            raise DataGateError("validation requires exactly 14 resets per controller")
        if int(action_budget) != VALIDATION_ACTIONS_PER_RESET:
            raise DataGateError("validation requires exactly 96 actions per reset")
        if posterior_reset is not True or learning_enabled is not False:
            raise DataGateError(
                "validation requires posterior reset and disabled learning"
            )
        runtime = self.factory._load_runtime()
        started = time.perf_counter()
        summaries: dict[str, dict[str, Any]] = {}
        for controller in order:
            policy_factory = self.factory._policy_factories[controller]
            if policy_factory is None:  # guarded by _load_runtime; defensive.
                raise RuntimeUnavailableError(f"missing frozen {controller} policy")
            policy = _invoke(
                policy_factory,
                context={
                    "controller": controller,
                    "game_id": self.game_id,
                    "seed": self.seed,
                    "posterior_reset": True,
                    "learning_enabled": False,
                    "runtime": runtime,
                },
                positional=(self.game_id, self.seed),
            )
            environment = _open_runtime(runtime, self.game_id, self.seed)
            self._opened.append(environment)
            try:
                summaries[controller] = self._run_arm(
                    runtime,
                    environment,
                    policy,
                    resets=int(resets),
                    action_budget=int(action_budget),
                )
            finally:
                _close_runtime(runtime, environment)
                self._opened.remove(environment)
                close_policy = getattr(policy, "close", None)
                if callable(close_policy):
                    close_policy()
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "baseline": summaries["t10_1"],
            "t10_2": summaries["t10_2"],
            "controller_order": list(order),
            "counterbalanced": True,
            "posterior_reset": True,
            "learning_between_controllers": False,
            "wall_seconds": time.perf_counter() - started,
        }

    def _run_arm(
        self,
        runtime: Any,
        environment: Any,
        policy: Any,
        *,
        resets: int,
        action_budget: int,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "levels": 0,
            "legal_actions": 0,
            "game_overs": 0,
            "illegal_actions": 0,
            "errors": 0,
            "planned_actions": 0,
            "completed_actions": 0,
            "unregistered_stops": 0,
            "reset_summaries": [],
            "decision_latency_ms": [],
            "observation_latency_ms": [],
        }
        for reset_index in range(resets):
            reset_policy = getattr(policy, "reset", None)
            if callable(reset_policy):
                _invoke(
                    reset_policy,
                    context={
                        "reset_index": reset_index,
                        "posterior_reset": reset_index == 0,
                        "learning_enabled": False,
                    },
                )
            frame = _reset_runtime(runtime, environment)
            before = _snapshot_runtime(runtime, frame)
            reset_completed_actions = 0
            stop_reason: str | None = None
            for step_index in range(action_budget):
                if _is_terminal(before):
                    stop_reason = "game_over" if _is_game_over(before) else "terminal"
                    break
                legal = tuple(
                    action
                    for action in _legal_runtime(runtime, environment)
                    if _action_name(action)
                )
                if not legal:
                    stop_reason = "no_legal_actions"
                    break
                decision_started = time.perf_counter()
                try:
                    choice = self._policy_action(
                        policy,
                        legal=legal,
                        snapshot=before,
                        environment=environment,
                        reset_index=reset_index,
                        step_index=step_index,
                    )
                except RuntimeUnavailableError:
                    raise
                except Exception:  # noqa: BLE001 - compact count, no raw error text.
                    summary["errors"] += 1
                    stop_reason = "decision_error"
                    break
                if choice is None:
                    registered_reason = self._policy_registered_stop_reason(
                        policy,
                        snapshot=before,
                        environment=environment,
                        reset_index=reset_index,
                        step_index=step_index,
                    )
                    stop_reason = registered_reason or "policy_abstained"
                    break
                summary["decision_latency_ms"].append(
                    (time.perf_counter() - decision_started) * 1000.0
                )
                selected = self._ground_choice(choice, legal)
                if selected is None:
                    summary["illegal_actions"] += 1
                    stop_reason = "illegal_action"
                    break
                try:
                    after_frame = _step_runtime(runtime, environment, selected)
                    after = _snapshot_runtime(
                        runtime,
                        after_frame,
                        fallback_available_actions=legal,
                    )
                except Exception:  # noqa: BLE001
                    summary["errors"] += 1
                    stop_reason = "step_error"
                    break
                summary["legal_actions"] += 1
                summary["completed_actions"] += 1
                reset_completed_actions += 1
                delta = max(0, _snapshot_levels(after) - _snapshot_levels(before))
                summary["levels"] += delta
                observation_started = time.perf_counter()
                observe = getattr(policy, "observe", None)
                if callable(observe):
                    try:
                        _invoke(
                            observe,
                            context={
                                "before": before,
                                "after": after,
                                "action": selected,
                                "legal_actions": legal,
                                "reset_index": reset_index,
                                "step_index": step_index,
                                "learning_enabled": False,
                            },
                            positional=(before, selected, after),
                        )
                    except RuntimeUnavailableError:
                        raise
                    except Exception:  # noqa: BLE001
                        summary["errors"] += 1
                        stop_reason = "observation_error"
                        break
                summary["observation_latency_ms"].append(
                    (time.perf_counter() - observation_started) * 1000.0
                )
                before = after
                if _is_terminal(after):
                    stop_reason = "game_over" if _is_game_over(after) else "terminal"
                elif delta > 0:
                    stop_reason = "progression"
                if stop_reason is not None:
                    break
            if stop_reason is None:
                stop_reason = "budget_exhausted"
            if stop_reason in VALIDATION_EXEMPT_STOP_REASONS:
                reset_planned_actions = reset_completed_actions
            else:
                # A full reset budget remains in the denominator for every
                # non-preregistered early stop.  This prevents abstention,
                # errors, or missing legal actions from shrinking the gate.
                reset_planned_actions = action_budget
            if stop_reason in VALIDATION_UNREGISTERED_STOP_REASONS:
                summary["unregistered_stops"] += 1
            summary["planned_actions"] += reset_planned_actions
            summary["reset_summaries"].append(
                {
                    "reset_index": reset_index,
                    "planned_actions": reset_planned_actions,
                    "completed_actions": reset_completed_actions,
                    "stop_reason": stop_reason,
                }
            )
            if stop_reason == "game_over":
                summary["game_overs"] += 1
        projection = getattr(policy, "behavior_projection", None)
        if projection is not None:
            summary["behavior_projection"] = str(projection)
        for name in (
            "retained_observations",
            "deduplicated_immediate_noops",
            "online_observations",
        ):
            value = getattr(policy, name, None)
            if value is not None:
                summary[name] = int(value)
        return summary

    @staticmethod
    def _policy_registered_stop_reason(
        policy: Any,
        *,
        snapshot: Any,
        environment: Any,
        reset_index: int,
        step_index: int,
    ) -> str | None:
        declaration = getattr(policy, "registered_stop_reason", None)
        if not callable(declaration):
            return None
        reason = _invoke(
            declaration,
            context={
                "snapshot": snapshot,
                "environment": environment,
                "reset_index": reset_index,
                "step_index": step_index,
                "learning_enabled": False,
            },
        )
        if reason is None:
            return None
        if reason != "option_exhausted":
            raise DataGateError(
                "frozen policy declared an unregistered validation stop reason"
            )
        return "option_exhausted"

    def _policy_action(
        self,
        policy: Any,
        *,
        legal: Sequence[Any],
        snapshot: Any,
        environment: Any,
        reset_index: int,
        step_index: int,
    ) -> Any:
        decision = next(
            (
                getattr(policy, name)
                for name in ("select_action", "decide", "act")
                if callable(getattr(policy, name, None))
            ),
            policy if callable(policy) else None,
        )
        if decision is None:
            raise RuntimeUnavailableError("frozen policy exposes no decision method")
        return _invoke(
            decision,
            context={
                "legal_actions": tuple(legal),
                "snapshot": snapshot,
                "environment": environment,
                "reset_index": reset_index,
                "step_index": step_index,
                "learning_enabled": False,
            },
            positional=(snapshot, tuple(legal)),
        )

    @staticmethod
    def _ground_choice(choice: Any, legal: Sequence[Any]) -> Any | None:
        if any(choice is action for action in legal):
            return choice
        wanted_key = _grounding_key(choice)
        exact = [action for action in legal if _grounding_key(action) == wanted_key]
        if len(exact) == 1:
            return exact[0]
        wanted_name = _action_name(choice)
        by_name = [action for action in legal if _action_name(action) == wanted_name]
        return by_name[0] if len(by_name) == 1 else None

    def close(self) -> None:
        if not self._opened:
            return
        runtime = self.factory._load_runtime()
        for environment in tuple(self._opened):
            _close_runtime(runtime, environment)
            self._opened.remove(environment)


__all__ = [
    "CHALLENGER_RECIPE_FORMAT_VERSION",
    "MAXIMUM_COMPACT_EVENT_BYTES",
    "MAXIMUM_MODEL_VIEW_BYTES",
    "REPLAY_SPLIT",
    "RUNTIME_FORMAT_VERSION",
    "T10_1_FROZEN_MANIFEST_CHECKSUM",
    "SummaryReplayTransition",
    "T10_1BehaviorFrozenPolicyFactory",
    "T10_2GaugePolicyFactory",
    "T10_2SourceFactory",
    "T10_2ValidationFactory",
    "build_replay_ledger",
    "build_v4_3_replay_ledger",
    "run_source_trainer",
]
