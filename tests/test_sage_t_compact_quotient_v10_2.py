from __future__ import annotations

import itertools
import json
import math
from copy import deepcopy
from dataclasses import replace

import pytest

from theory.sage_t.compact_quotient_v10_2 import (
    ALLOWED_FACT_PREDICATES,
    ALLOWED_ROLES,
    FORMAT_VERSION,
    MAXIMUM_COUNTER_ROWS,
    MAXIMUM_FACT_ARITY,
    MAXIMUM_FACT_ROWS,
    MAXIMUM_REGISTER_ROWS,
    MAXIMUM_ROLE_ROWS,
    MAXIMUM_TOPOLOGY_ROWS,
    MAXIMUM_TOTAL_ENTITIES,
    MAXIMUM_TOTAL_FACTS,
    MAXIMUM_TOTAL_ROWS,
    SUMMARY_KIND,
    assert_compact_quotient,
    quotient_sha256,
    state_from_quotient_payload,
    state_quotient_payload,
    state_quotient_sha256,
)
from theory.sage_t.contracts import AbstractEntity, AbstractState, GroundFact


def _tree_state(
    arm_lengths: tuple[int, int, int],
    *,
    renamed: bool = False,
    reverse_entities: bool = False,
    register_value: str | None = None,
) -> AbstractState:
    prefix = "branch" if renamed else "local"
    center = f"{prefix}_center"
    entities = [AbstractEntity(center, ("actor", "object"))]
    facts: set[GroundFact] = set()
    for arm_index, length in enumerate(arm_lengths):
        previous = center
        for offset in range(1, length + 1):
            entity_id = f"{prefix}_arm_{arm_index}_{offset}"
            entities.append(AbstractEntity(entity_id, ("object",)))
            facts.add(GroundFact("contact", (previous, entity_id)))
            previous = entity_id
    if reverse_entities:
        entities.reverse()
    return AbstractState(
        entities=tuple(entities),
        true_facts=frozenset(facts),
        counters=(("actor_count", 1.0), ("entity_count", 7.0)),
        registers=(("selected", register_value or center),),
        topology=(
            ("connected_components", 1),
            ("node_count", 7),
            ("structural_edges", 6),
        ),
        regime_index=3,
    )


def _branch_lengths(state: AbstractState) -> tuple[int, ...]:
    adjacency = {entity.entity_id: set() for entity in state.entities}
    for fact in state.true_facts:
        if fact.predicate != "contact":
            continue
        left, right = fact.terms
        adjacency[left].add(right)
        adjacency[right].add(left)
    center = next(node for node, neighbors in adjacency.items() if len(neighbors) == 3)
    lengths = []
    for start in adjacency[center]:
        previous = center
        current = start
        length = 1
        while len(adjacency[current]) == 2:
            following = next(item for item in adjacency[current] if item != previous)
            previous, current = current, following
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths))


def _payload() -> dict[str, object]:
    return state_quotient_payload(_tree_state((4, 1, 1)))


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def test_two_nonisomorphic_seven_node_trees_have_exactly_one_v2_summary() -> None:
    long_arm = _tree_state((4, 1, 1))
    balanced_arms = _tree_state((3, 2, 1), renamed=True)

    assert _branch_lengths(long_arm) == (1, 1, 4)
    assert _branch_lengths(balanced_arms) == (1, 2, 3)
    left = state_quotient_payload(long_arm)
    right = state_quotient_payload(balanced_arms)

    assert left == right
    assert state_quotient_sha256(left) == state_quotient_sha256(right)
    assert quotient_sha256(left) == state_quotient_sha256(left)
    assert left["format_version"] == FORMAT_VERSION
    assert left["summary_kind"] == SUMMARY_KIND
    assert left["entity_count"] == 7
    assert left["fact_count"] == 6

    serialized = _canonical(left)
    for forbidden in (
        "endpoint",
        "token",
        "terms",
        "source",
        "target",
        "class",
        "identifier",
        "local_center",
        "branch_center",
    ):
        assert f'"{forbidden}"' not in serialized


def test_input_permutation_renaming_attributes_and_register_values_are_erased() -> None:
    original = _tree_state((4, 1, 1), register_value="local_private_binding")
    changed_entities = tuple(
        replace(
            entity,
            attributes=(("linked", original.entities[-1].entity_id),),
        )
        for entity in original.entities
    )
    attributed = replace(original, entities=changed_entities)
    renamed = _tree_state(
        (4, 1, 1),
        renamed=True,
        reverse_entities=True,
        register_value="bp35-0a0ad940",
    )

    expected = state_quotient_payload(original)
    assert state_quotient_payload(attributed) == expected
    assert state_quotient_payload(renamed) == expected
    assert expected["register_rows"] == [{"name": "selected"}]
    rendered = _canonical(expected)
    assert "local_private_binding" not in rendered
    assert "bp35-0a0ad940" not in rendered


def test_v2_refuses_active_rehydration() -> None:
    with pytest.raises(RuntimeError, match="non-injective.*cannot be rehydrated"):
        state_from_quotient_payload(_payload())


def test_schema_version_and_summary_kind_are_closed() -> None:
    payload = _payload()
    wrong_version = {**payload, "format_version": "sage-t10.2-structural-quotient-v1"}
    with pytest.raises(ValueError, match="format version"):
        assert_compact_quotient(wrong_version)

    wrong_kind = {**payload, "summary_kind": "graph_encoding"}
    with pytest.raises(ValueError, match="summary kind"):
        assert_compact_quotient(wrong_kind)

    missing = dict(payload)
    missing.pop("regime_index")
    with pytest.raises(ValueError, match="missing=.*regime_index"):
        assert_compact_quotient(missing)

    with pytest.raises(ValueError, match="unknown=.*debug_summary"):
        assert_compact_quotient({**payload, "debug_summary": {}})


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "endpoint",
        "token",
        "terms",
        "source",
        "target",
        "class",
        "identifier",
        "nodes",
        "edges",
        "entity_id",
    ),
)
def test_forbidden_graph_and_identity_keys_are_rejected_recursively(
    forbidden_key: str,
) -> None:
    payload = deepcopy(_payload())
    payload["counter_rows"][0][forbidden_key] = "private"
    with pytest.raises(ValueError, match="endpoints and identifiers"):
        assert_compact_quotient(payload)


def test_register_rows_cannot_carry_values_or_bindings() -> None:
    payload = deepcopy(_payload())
    payload["register_rows"][0]["value"] = "local_center"
    with pytest.raises(ValueError, match="forbidden|register row schema drifted"):
        assert_compact_quotient(payload)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("role_rows", "roles", ["rogue_role"], "role.*allowlist"),
        ("fact_rows", "predicate", "rogue_fact", "predicate.*allowlist"),
        ("counter_rows", "name", "rogue_counter", "counter name.*allowlist"),
        ("register_rows", "name", "rogue_register", "register name.*allowlist"),
        ("topology_rows", "name", "rogue_topology", "topology name.*allowlist"),
    ),
)
def test_every_symbolic_row_uses_a_closed_allowlist(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = deepcopy(_payload())
    payload[section][0][field] = value
    with pytest.raises(ValueError, match=message):
        assert_compact_quotient(payload)


def test_rows_and_role_signatures_must_be_canonical_and_unique() -> None:
    unsorted = deepcopy(_payload())
    unsorted["role_rows"].reverse()
    with pytest.raises(ValueError, match="canonical order"):
        assert_compact_quotient(unsorted)

    unsorted_roles = deepcopy(_payload())
    unsorted_roles["role_rows"][0]["roles"].reverse()
    with pytest.raises(ValueError, match="roles must be sorted"):
        assert_compact_quotient(unsorted_roles)

    duplicate = deepcopy(_payload())
    duplicate["fact_rows"].append(deepcopy(duplicate["fact_rows"][0]))
    with pytest.raises(ValueError, match="duplicate rows"):
        assert_compact_quotient(duplicate)

    duplicate_counter = replace(
        _tree_state((4, 1, 1)),
        counters=(("actor_count", 1.0), ("actor_count", 2.0)),
    )
    with pytest.raises(ValueError, match="duplicate counter names"):
        state_quotient_payload(duplicate_counter)


@pytest.mark.parametrize(
    ("section", "maximum"),
    (
        ("role_rows", MAXIMUM_ROLE_ROWS),
        ("fact_rows", MAXIMUM_FACT_ROWS),
        ("counter_rows", MAXIMUM_COUNTER_ROWS),
        ("register_rows", MAXIMUM_REGISTER_ROWS),
        ("topology_rows", MAXIMUM_TOPOLOGY_ROWS),
    ),
)
def test_each_row_family_has_its_exact_budget(section: str, maximum: int) -> None:
    payload = deepcopy(_payload())
    payload[section] = [deepcopy(payload[section][0])] * (maximum + 1)
    with pytest.raises(ValueError, match=f"row budget of {maximum}"):
        assert_compact_quotient(payload)


def test_total_rows_entities_facts_and_arity_have_hard_budgets() -> None:
    rows = deepcopy(_payload())
    rows["role_rows"] = [deepcopy(rows["role_rows"][0])] * MAXIMUM_ROLE_ROWS
    rows["fact_rows"] = [deepcopy(rows["fact_rows"][0])] * MAXIMUM_FACT_ROWS
    rows["counter_rows"] = [deepcopy(rows["counter_rows"][0])]
    rows["register_rows"] = []
    rows["topology_rows"] = []
    with pytest.raises(ValueError, match=f"total row budget of {MAXIMUM_TOTAL_ROWS}"):
        assert_compact_quotient(rows)

    entities = deepcopy(_payload())
    entities["entity_count"] = MAXIMUM_TOTAL_ENTITIES + 1
    with pytest.raises(ValueError, match="entity count"):
        assert_compact_quotient(entities)

    facts = deepcopy(_payload())
    facts["fact_count"] = MAXIMUM_TOTAL_FACTS + 1
    with pytest.raises(ValueError, match="fact count"):
        assert_compact_quotient(facts)

    arity = deepcopy(_payload())
    arity["fact_rows"][0]["arity"] = MAXIMUM_FACT_ARITY + 1
    with pytest.raises(ValueError, match="fact arity"):
        assert_compact_quotient(arity)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("entity_count",), True, "entity count.*boolean"),
        (("fact_count",), True, "fact count.*boolean"),
        (("regime_index",), True, "regime index.*boolean"),
        (("role_rows", 0, "count"), True, "role count.*boolean"),
        (("fact_rows", 0, "arity"), True, "fact arity.*boolean"),
        (("fact_rows", 0, "count"), True, "fact row count.*boolean"),
        (("counter_rows", 0, "amount"), True, "counter value.*boolean"),
        (("topology_rows", 0, "amount"), True, "topology value.*boolean"),
    ),
)
def test_boolean_values_never_pass_as_integers(
    path: tuple[object, ...],
    value: object,
    message: str,
) -> None:
    payload = deepcopy(_payload())
    cursor: object = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(TypeError, match=message):
        assert_compact_quotient(payload)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_counter_values_are_rejected(value: float) -> None:
    payload = deepcopy(_payload())
    payload["counter_rows"][0]["amount"] = value
    with pytest.raises(ValueError, match="counter value must be finite"):
        assert_compact_quotient(payload)


def test_valid_but_oversized_schema_is_stopped_at_eight_kibibytes() -> None:
    roles = sorted(ALLOWED_ROLES)
    role_rows = []
    for mask in range(MAXIMUM_ROLE_ROWS):
        selected = [role for index, role in enumerate(roles[:6]) if mask & (1 << index)]
        role_rows.append({"roles": selected, "count": 1})
    role_rows.sort(key=_canonical)

    fact_rows = []
    signatures = itertools.product(
        (False, True),
        sorted(ALLOWED_FACT_PREDICATES),
        range(MAXIMUM_FACT_ARITY + 1),
        (False, True),
    )
    for truth, predicate, arity, has_literal in itertools.islice(
        signatures,
        MAXIMUM_FACT_ROWS,
    ):
        fact_rows.append(
            {
                "truth": truth,
                "predicate": predicate,
                "arity": arity,
                "has_literal": has_literal,
                "count": 1,
            }
        )
    fact_rows.sort(key=_canonical)

    payload = {
        "format_version": FORMAT_VERSION,
        "summary_kind": SUMMARY_KIND,
        "entity_count": MAXIMUM_ROLE_ROWS,
        "fact_count": MAXIMUM_FACT_ROWS,
        "role_rows": role_rows,
        "fact_rows": fact_rows,
        "counter_rows": [],
        "register_rows": [],
        "topology_rows": [],
        "regime_index": 0,
    }
    assert len(_canonical(payload).encode("utf-8")) > 8 * 1_024
    with pytest.raises(ValueError, match="8192-byte payload budget"):
        assert_compact_quotient(payload)


def test_source_fact_arity_and_duplicate_entity_ids_fail_before_summarizing() -> None:
    oversized_fact = GroundFact(
        "contact",
        ("a", "b", "c", "d", "e"),
    )
    with pytest.raises(ValueError, match="maximum arity"):
        state_quotient_payload(AbstractState(true_facts=frozenset({oversized_fact})))

    duplicate_ids = AbstractState(
        entities=(
            AbstractEntity("duplicate", ("actor",)),
            AbstractEntity("duplicate", ("object",)),
        )
    )
    with pytest.raises(ValueError, match="unique local entity ids"):
        state_quotient_payload(duplicate_ids)


def test_colour_relation_is_outside_the_t10_2_summary_vocabulary() -> None:
    state = AbstractState(
        entities=(AbstractEntity("local", ("object",)),),
        true_facts=frozenset({GroundFact("same_color", ("local", "local"))}),
    )
    with pytest.raises(ValueError, match="closed allowlist"):
        state_quotient_payload(state)
