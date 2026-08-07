"""Formally non-injective summaries for persisted SAGE.T10.2 evidence.

Version 2 is an endpoint-free multiset summary, not a graph encoding.  It
counts entity role signatures and fact signatures while deliberately omitting
entity attributes, fact arguments, literal values, and register values.  Two
non-isomorphic graphs can therefore have exactly the same payload and hash.
That collision is a required privacy property: persisted evidence cannot be
used to reconstruct an active :class:`~theory.sage_t.contracts.AbstractState`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import ALLOWED_PREDICATES, AbstractState
from .observer_frames_v10_2 import audit_identity_leaks

FORMAT_VERSION = "sage-t10.2-structural-quotient-v2"
SUMMARY_KIND = "non_injective_multiset_counts"

MAXIMUM_PAYLOAD_BYTES = 8 * 1_024
MAXIMUM_ROLE_ROWS = 64
MAXIMUM_FACT_ROWS = 128
MAXIMUM_COUNTER_ROWS = 16
MAXIMUM_REGISTER_ROWS = 4
MAXIMUM_TOPOLOGY_ROWS = 32
MAXIMUM_TOTAL_ROWS = 192
MAXIMUM_TOTAL_ENTITIES = 4_096
MAXIMUM_TOTAL_FACTS = 16_384
MAXIMUM_FACT_ARITY = 4

ALLOWED_ROLES = frozenset(
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
# The frozen lower-level contract still exposes colour equivalence for older
# experiments. T10.2's structural frame bank deliberately does not: keeping
# it out of this vocabulary prevents transport induction from reintroducing a
# colour-dependent transferable program through the compact ledger.
ALLOWED_FACT_PREDICATES = frozenset(ALLOWED_PREDICATES - {"same_color"})
ALLOWED_COUNTER_NAMES = frozenset(
    {
        "actor_count",
        "adjacent_degree",
        "contact_degree",
        "entity_count",
        "frame_rank",
        "neighbor_classes",
        "progress",
        "relation_count",
        "stage_rank",
        "step_count",
        "structural_classes",
        "target_count",
    }
)
ALLOWED_REGISTER_NAMES = frozenset({"action_root", "actor", "selected", "target"})
ALLOWED_TOPOLOGY_NAMES = frozenset(
    {
        "action_root_component_size",
        "actor_component_size",
        "actor_root_distance",
        "articulation_points",
        "boundary_connected_free_regions",
        "bridges",
        "component_count",
        "connected_components",
        "contact_edges",
        "cycle_rank",
        "euler_characteristic",
        "free_region_count",
        "free_regions",
        "holes",
        "largest_free_region_bucket",
        "node_count",
        "object_components",
        "object_count",
        "reachable_free_regions",
        "root_bridge_incidence",
        "root_is_articulation",
        "structural_edges",
    }
)

FORBIDDEN_FULL_GRAPH_KEYS = frozenset(
    {
        "class",
        "classes",
        "edge",
        "edges",
        "endpoint",
        "endpoints",
        "entities",
        "entity_id",
        "entity_ids",
        "false_facts",
        "full_graph",
        "graph",
        "graphs",
        "id",
        "identifier",
        "identifiers",
        "ids",
        "node",
        "nodes",
        "object_id",
        "object_ids",
        "source",
        "sources",
        "target",
        "targets",
        "term",
        "terms",
        "token",
        "tokens",
        "true_facts",
        "value",
    }
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "format_version",
        "summary_kind",
        "entity_count",
        "fact_count",
        "role_rows",
        "fact_rows",
        "counter_rows",
        "register_rows",
        "topology_rows",
        "regime_index",
    }
)
_ROLE_ROW_FIELDS = frozenset({"roles", "count"})
_FACT_ROW_FIELDS = frozenset({"truth", "predicate", "arity", "has_literal", "count"})
_NUMERIC_ROW_FIELDS = frozenset({"name", "amount"})
_REGISTER_ROW_FIELDS = frozenset({"name"})


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("quotient payload must be finite JSON data") from exc


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    label: str,
) -> None:
    observed = set(value)
    if observed != set(required):
        missing = sorted(required - observed)
        unknown = sorted(observed - required, key=str)
        raise ValueError(
            f"{label} schema drifted; missing={missing}, unknown={unknown}"
        )


def _require_json_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _require_rows(value: Any, *, label: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds its row budget of {maximum}")
    return value


def _bounded_integer(
    value: Any,
    *,
    label: str,
    maximum: int,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer, not a boolean")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _positive_count(value: Any, *, label: str, maximum: int) -> int:
    return _bounded_integer(value, label=label, minimum=1, maximum=maximum)


def _require_sorted_unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    signatures: Sequence[Any],
    label: str,
) -> None:
    signature_text = [_canonical_json(item) for item in signatures]
    if len(set(signature_text)) != len(signature_text):
        raise ValueError(f"{label} contains duplicate rows")
    rendered = [_canonical_json(item) for item in rows]
    if rendered != sorted(rendered):
        raise ValueError(f"{label} rows are not in canonical order")


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("quotient keys must be strings")
            normalized = key.strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_FULL_GRAPH_KEYS or normalized.endswith(
                ("_id", "_ids", "_identifier", "_identifiers")
            ):
                raise ValueError(
                    "graph endpoints and identifiers are forbidden in the quotient"
                )
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def _require_unique_source_names(
    rows: Sequence[tuple[str, Any]],
    *,
    label: str,
) -> None:
    names = tuple(name for name, _value in rows)
    if len(set(names)) != len(names):
        raise ValueError(f"source state contains duplicate {label} names")


def state_quotient_payload(state: AbstractState) -> dict[str, Any]:
    """Return an endpoint-free, formally non-injective state summary."""

    if not isinstance(state, AbstractState):
        raise TypeError("state quotient requires AbstractState")
    entity_ids = tuple(entity.entity_id for entity in state.entities)
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("structural quotient requires unique local entity ids")
    if len(entity_ids) > MAXIMUM_TOTAL_ENTITIES:
        raise ValueError("source state exceeds the entity budget")
    fact_count = len(state.true_facts) + len(state.false_facts)
    if fact_count > MAXIMUM_TOTAL_FACTS:
        raise ValueError("source state exceeds the fact budget")
    _require_unique_source_names(state.counters, label="counter")
    _require_unique_source_names(state.registers, label="register")
    _require_unique_source_names(state.topology, label="topology")

    role_counts = Counter(tuple(entity.roles) for entity in state.entities)
    role_rows = [
        {"roles": list(roles), "count": int(count)}
        for roles, count in role_counts.items()
    ]
    role_rows.sort(key=_canonical_json)

    fact_counts: Counter[tuple[bool, str, int, bool]] = Counter()
    for truth, facts in ((True, state.true_facts), (False, state.false_facts)):
        for fact in facts:
            if len(fact.terms) > MAXIMUM_FACT_ARITY:
                raise ValueError("source fact exceeds the maximum arity")
            fact_counts[(truth, fact.predicate, len(fact.terms), bool(fact.value))] += 1
    fact_rows = [
        {
            "truth": truth,
            "predicate": predicate,
            "arity": arity,
            "has_literal": has_literal,
            "count": int(count),
        }
        for (truth, predicate, arity, has_literal), count in fact_counts.items()
    ]
    fact_rows.sort(key=_canonical_json)

    counter_rows = [{"name": name, "amount": value} for name, value in state.counters]
    counter_rows.sort(key=_canonical_json)
    register_rows = [{"name": name} for name, _value in state.registers]
    register_rows.sort(key=_canonical_json)
    topology_rows = [{"name": name, "amount": value} for name, value in state.topology]
    topology_rows.sort(key=_canonical_json)

    payload = {
        "format_version": FORMAT_VERSION,
        "summary_kind": SUMMARY_KIND,
        "entity_count": len(entity_ids),
        "fact_count": fact_count,
        "role_rows": role_rows,
        "fact_rows": fact_rows,
        "counter_rows": counter_rows,
        "register_rows": register_rows,
        "topology_rows": topology_rows,
        "regime_index": int(state.regime_index),
    }
    assert_compact_quotient(payload)
    return payload


def assert_compact_quotient(payload: Mapping[str, Any]) -> None:
    """Validate the closed, bounded v2 summary and evidence firewall."""

    root = _require_json_mapping(payload, label="structural quotient")
    _reject_forbidden_keys(root)
    _require_exact_fields(root, required=_TOP_LEVEL_FIELDS, label="quotient")
    if root["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported structural quotient format version")
    if root["summary_kind"] != SUMMARY_KIND:
        raise ValueError("unsupported structural quotient summary kind")

    entity_count = _bounded_integer(
        root["entity_count"],
        label="quotient entity count",
        maximum=MAXIMUM_TOTAL_ENTITIES,
    )
    fact_count = _bounded_integer(
        root["fact_count"],
        label="quotient fact count",
        maximum=MAXIMUM_TOTAL_FACTS,
    )
    _bounded_integer(
        root["regime_index"],
        label="quotient regime index",
        maximum=MAXIMUM_TOTAL_ENTITIES,
    )

    role_rows = _require_rows(
        root["role_rows"], label="quotient role rows", maximum=MAXIMUM_ROLE_ROWS
    )
    fact_rows = _require_rows(
        root["fact_rows"], label="quotient fact rows", maximum=MAXIMUM_FACT_ROWS
    )
    counter_rows = _require_rows(
        root["counter_rows"],
        label="quotient counter rows",
        maximum=MAXIMUM_COUNTER_ROWS,
    )
    register_rows = _require_rows(
        root["register_rows"],
        label="quotient register rows",
        maximum=MAXIMUM_REGISTER_ROWS,
    )
    topology_rows = _require_rows(
        root["topology_rows"],
        label="quotient topology rows",
        maximum=MAXIMUM_TOPOLOGY_ROWS,
    )
    total_rows = sum(
        len(rows)
        for rows in (
            role_rows,
            fact_rows,
            counter_rows,
            register_rows,
            topology_rows,
        )
    )
    if total_rows > MAXIMUM_TOTAL_ROWS:
        raise ValueError(
            f"quotient exceeds the total row budget of {MAXIMUM_TOTAL_ROWS}"
        )

    normalized_roles: list[Mapping[str, Any]] = []
    role_signatures: list[Any] = []
    role_total = 0
    for raw in role_rows:
        row = _require_json_mapping(raw, label="quotient role row")
        _require_exact_fields(row, required=_ROLE_ROW_FIELDS, label="role row")
        roles = _require_rows(
            row["roles"], label="quotient role signature", maximum=len(ALLOWED_ROLES)
        )
        if any(not isinstance(role, str) for role in roles):
            raise TypeError("quotient roles must be strings")
        if roles != sorted(roles) or len(set(roles)) != len(roles):
            raise ValueError("quotient roles must be sorted and unique")
        if any(role not in ALLOWED_ROLES for role in roles):
            raise ValueError("quotient role is outside the closed allowlist")
        count = _positive_count(
            row["count"],
            label="quotient role count",
            maximum=MAXIMUM_TOTAL_ENTITIES,
        )
        role_total += count
        normalized_roles.append(row)
        role_signatures.append(roles)
    _require_sorted_unique_rows(
        normalized_roles,
        signatures=role_signatures,
        label="quotient role rows",
    )
    if role_total != entity_count:
        raise ValueError("quotient role counts do not match entity_count")

    normalized_facts: list[Mapping[str, Any]] = []
    fact_signatures: list[Any] = []
    fact_total = 0
    for raw in fact_rows:
        row = _require_json_mapping(raw, label="quotient fact row")
        _require_exact_fields(row, required=_FACT_ROW_FIELDS, label="fact row")
        if not isinstance(row["truth"], bool):
            raise TypeError("quotient fact truth must be boolean")
        predicate = row["predicate"]
        if not isinstance(predicate, str):
            raise TypeError("quotient fact predicate must be a string")
        if predicate not in ALLOWED_FACT_PREDICATES:
            raise ValueError("quotient fact predicate is outside the closed allowlist")
        arity = _bounded_integer(
            row["arity"],
            label="quotient fact arity",
            maximum=MAXIMUM_FACT_ARITY,
        )
        if not isinstance(row["has_literal"], bool):
            raise TypeError("quotient literal-presence flag must be boolean")
        count = _positive_count(
            row["count"],
            label="quotient fact row count",
            maximum=MAXIMUM_TOTAL_FACTS,
        )
        fact_total += count
        normalized_facts.append(row)
        fact_signatures.append([row["truth"], predicate, arity, row["has_literal"]])
    _require_sorted_unique_rows(
        normalized_facts,
        signatures=fact_signatures,
        label="quotient fact rows",
    )
    if fact_total != fact_count:
        raise ValueError("quotient fact row counts do not match fact_count")

    def validate_numeric_rows(
        rows: Sequence[Any],
        *,
        label: str,
        allowed_names: frozenset[str],
        integer: bool,
    ) -> None:
        normalized: list[Mapping[str, Any]] = []
        names: list[str] = []
        for raw in rows:
            row = _require_json_mapping(raw, label=f"quotient {label} row")
            _require_exact_fields(
                row,
                required=_NUMERIC_ROW_FIELDS,
                label=f"{label} row",
            )
            name = row["name"]
            if not isinstance(name, str):
                raise TypeError(f"quotient {label} name must be a string")
            if name not in allowed_names:
                raise ValueError(
                    f"quotient {label} name is outside the closed allowlist"
                )
            value = row["amount"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"quotient {label} value must be numeric, not boolean")
            if not math.isfinite(float(value)):
                raise ValueError(f"quotient {label} value must be finite")
            if integer and not isinstance(value, int):
                raise TypeError(f"quotient {label} value must be an integer")
            normalized.append(row)
            names.append(name)
        _require_sorted_unique_rows(
            normalized,
            signatures=names,
            label=f"quotient {label} rows",
        )

    validate_numeric_rows(
        counter_rows,
        label="counter",
        allowed_names=ALLOWED_COUNTER_NAMES,
        integer=False,
    )
    validate_numeric_rows(
        topology_rows,
        label="topology",
        allowed_names=ALLOWED_TOPOLOGY_NAMES,
        integer=True,
    )

    normalized_registers: list[Mapping[str, Any]] = []
    register_names: list[str] = []
    for raw in register_rows:
        row = _require_json_mapping(raw, label="quotient register row")
        _require_exact_fields(
            row,
            required=_REGISTER_ROW_FIELDS,
            label="register row",
        )
        name = row["name"]
        if not isinstance(name, str):
            raise TypeError("quotient register name must be a string")
        if name not in ALLOWED_REGISTER_NAMES:
            raise ValueError("quotient register name is outside the closed allowlist")
        normalized_registers.append(row)
        register_names.append(name)
    _require_sorted_unique_rows(
        normalized_registers,
        signatures=register_names,
        label="quotient register rows",
    )

    rendered = _canonical_json(root)
    if len(rendered.encode("utf-8")) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError(
            f"quotient exceeds the {MAXIMUM_PAYLOAD_BYTES}-byte payload budget"
        )
    leaks = audit_identity_leaks(root)
    if leaks:
        raise ValueError(f"identity leak in structural quotient: {leaks[0]}")


def state_quotient_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one validated v2 state summary in canonical JSON order."""

    assert_compact_quotient(payload)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def quotient_sha256(payload: Mapping[str, Any]) -> str:
    """Backward-compatible name for :func:`state_quotient_sha256`."""

    return state_quotient_sha256(payload)


def state_from_quotient_payload(payload: Mapping[str, Any]) -> AbstractState:
    """Refuse active rehydration from the formally non-injective v2 summary."""

    assert_compact_quotient(payload)
    raise RuntimeError(
        "structural quotient v2 is non-injective and cannot be rehydrated"
    )


__all__ = [
    "ALLOWED_COUNTER_NAMES",
    "ALLOWED_FACT_PREDICATES",
    "ALLOWED_REGISTER_NAMES",
    "ALLOWED_ROLES",
    "ALLOWED_TOPOLOGY_NAMES",
    "FORBIDDEN_FULL_GRAPH_KEYS",
    "FORMAT_VERSION",
    "MAXIMUM_COUNTER_ROWS",
    "MAXIMUM_FACT_ARITY",
    "MAXIMUM_FACT_ROWS",
    "MAXIMUM_PAYLOAD_BYTES",
    "MAXIMUM_REGISTER_ROWS",
    "MAXIMUM_ROLE_ROWS",
    "MAXIMUM_TOPOLOGY_ROWS",
    "MAXIMUM_TOTAL_ENTITIES",
    "MAXIMUM_TOTAL_FACTS",
    "MAXIMUM_TOTAL_ROWS",
    "SUMMARY_KIND",
    "assert_compact_quotient",
    "quotient_sha256",
    "state_from_quotient_payload",
    "state_quotient_payload",
    "state_quotient_sha256",
]
