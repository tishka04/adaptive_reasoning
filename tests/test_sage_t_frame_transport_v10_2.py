from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from theory.sage_t import gauge_inference_v10_2 as gauge_inference
from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    PredictionPacket,
)
from theory.sage_t.frame_transport_v10_2 import (
    CommutativityAudit,
    TransportCertificate,
    TransportMap,
    TransportOrbitWitness,
    canonical_signature,
    find_transport,
    gauge_equivalent,
    persisted_attestation_receipt,
    prediction_commutativity_penalty,
    transport_action,
    transport_prediction,
    transport_state,
)
from theory.sage_t.observer_frames_v10_2 import (
    ACTION_ALIGNED_RELATIONAL_FRAME,
    ROOT_ONLY_FRAME,
    FrameProjection,
    ProjectedTransition,
)


def _state(
    *,
    prefix: str,
    target_role: str,
    relation: str,
    changed: bool = False,
) -> AbstractState:
    actor_id = f"{prefix}_actor"
    target_id = f"{prefix}_target"
    facts = {
        GroundFact("exists", (actor_id,)),
        GroundFact("exists", (target_id,)),
        GroundFact(relation, (actor_id, target_id)),
    }
    if changed:
        facts.add(GroundFact("changed", (target_id,)))
    return AbstractState(
        entities=(
            AbstractEntity(actor_id, ("actor", "object")),
            AbstractEntity(target_id, ("object", target_role)),
        ),
        true_facts=frozenset(facts),
        counters=(("step_count", float(changed)),),
        topology=(("component_count", 2),),
    )


def _projection_pair(
    *,
    changed: bool = False,
    complete: bool = True,
) -> tuple[FrameProjection, FrameProjection]:
    source = FrameProjection(
        frame=ROOT_ONLY_FRAME,
        state=_state(
            prefix="source",
            target_role="source_goal",
            relation="near",
            changed=changed,
        ),
        action=ActionCandidate("ACTION1"),
        complete=complete,
        missing=() if complete else ("relational_binding",),
        covered_channels=(
            "entities",
            "facts",
            "counters",
            "topology",
            "regime",
        )
        if complete
        else ("entities",),
    )
    target = FrameProjection(
        frame=ACTION_ALIGNED_RELATIONAL_FRAME,
        state=_state(
            prefix="target",
            target_role="destination",
            relation="adjacent",
            changed=changed,
        ),
        action=ActionCandidate("ACTION2"),
        complete=complete,
        missing=() if complete else ("relational_binding",),
        covered_channels=(
            "entities",
            "facts",
            "counters",
            "topology",
            "regime",
        )
        if complete
        else ("entities",),
    )
    return source, target


def _exact_transport(*, pair_order: bool = False) -> TransportMap:
    role_pairs = (
        ("actor", "actor"),
        ("object", "object"),
        ("source_goal", "destination"),
    )
    fact_pairs = (
        ("changed", "changed"),
        ("component_count_changed", "component_count_changed"),
        ("near", "adjacent"),
        ("exists", "exists"),
    )
    if pair_order:
        role_pairs = tuple(reversed(role_pairs))
        fact_pairs = tuple(reversed(fact_pairs))
    return TransportMap(
        source_frame_id=ROOT_ONLY_FRAME.frame_id,
        target_frame_id=ACTION_ALIGNED_RELATIONAL_FRAME.frame_id,
        role_map=role_pairs,
        fact_map=fact_pairs,
        action_map=(("ACTION1", "ACTION2"),),
        domain=frozenset(
            {
                "role:actor",
                "role:object",
                "role:source_goal",
                "fact:exists",
                "fact:changed",
                "fact:component_count_changed",
                "fact:near",
                "action:ACTION1",
            }
        ),
    )


def _prediction(
    state: AbstractState,
    *,
    relation: str,
) -> PredictionPacket:
    return PredictionPacket(
        object_deltas={"changed": 0.9},
        relation_deltas={f"relation_added:{relation}": 0.8},
        topology_deltas={"component_count_changed": 0.7},
        progress_mean=1.0,
        progress_distribution={"increase": 1.0},
        terminal_probability=0.0,
        goal_probability=0.75,
        known_channels=frozenset(
            {"objects", "relations", "topology", "progress", "terminal", "goal"}
        ),
        state_after=state,
    )


def _persisted_orbit_envelope() -> dict[str, object]:
    action = ActionCandidate("ACTION1")
    source_state = AbstractState(
        entities=(AbstractEntity("source_actor", ("actor", "object")),),
        true_facts=frozenset({GroundFact("exists", ("source_actor",))}),
    )
    target_state = AbstractState(
        entities=(AbstractEntity("target_actor", ("actor", "object")),),
        true_facts=frozenset({GroundFact("exists", ("target_actor",))}),
    )
    source = FrameProjection(
        frame=ROOT_ONLY_FRAME,
        state=source_state,
        action=action,
    )
    target = FrameProjection(
        frame=ACTION_ALIGNED_RELATIONAL_FRAME,
        state=target_state,
        action=action,
    )
    transport = TransportMap(
        source.frame_id,
        target.frame_id,
        role_map=(("actor", "actor"), ("object", "object")),
        fact_map=(("exists", "exists"),),
        action_map=(("ACTION1", "ACTION1"),),
        domain=frozenset(
            {"role:actor", "role:object", "fact:exists", "action:ACTION1"}
        ),
    )
    witness = TransportOrbitWitness.from_certificate(
        transport.certificate(source, target)
    )
    digest = lambda label: hashlib.sha256(label.encode("utf-8")).hexdigest()
    attestation: dict[str, object] = {
        "certificate_hash": digest("certificate"),
        "source_before_summary_hash": digest("source-before"),
        "source_after_summary_hash": digest("source-after"),
        "target_before_summary_hash": digest("target-before"),
        "target_after_summary_hash": digest("target-after"),
        "source_observation_hash": digest("source-observation"),
        "target_observation_hash": digest("target-observation"),
        "live_graph_exact_attested": True,
        "round_trip_exact": True,
        "summary_commutative_exact": True,
    }
    attestation["receipt"] = persisted_attestation_receipt(
        orbit_hash=witness.canonical_hash,
        source_frame=witness.source_frame_id,
        target_frame=witness.target_frame_id,
        attestation=attestation,
    )
    return json.loads(
        json.dumps(
            {
                "orbit_payload": witness.canonical_payload,
                "orbit_hash": witness.canonical_hash,
                "source_frame": witness.source_frame_id,
                "target_frame": witness.target_frame_id,
                "attestation": attestation,
            }
        )
    )


def test_exact_and_partial_certificates_are_explicit_and_immutable() -> None:
    source, target = _projection_pair()
    exact_map = _exact_transport()
    exact = TransportCertificate.from_projections(exact_map, source, target)

    assert exact.exact
    assert exact.coverage == 1.0
    assert exact.missing_domain == frozenset()
    assert exact.ambiguity == 0
    assert exact.round_trip_exact
    assert exact.certifies_gauge_equivalence
    assert exact.gauge_equivalence_key is not None
    with pytest.raises(FrozenInstanceError):
        exact.coverage = 0.0  # type: ignore[misc]

    partial_map = TransportMap(
        source.frame_id,
        target.frame_id,
        role_map=(("source_goal", "destination"),),
        action_map=(("ACTION1", "ACTION2"),),
        domain=frozenset({"role:source_goal", "fact:near", "action:ACTION1"}),
    )
    partial = partial_map.certificate(source, target)

    assert not partial.exact
    assert not partial.certifies_gauge_equivalence
    assert partial.gauge_equivalence_key is None
    assert 0.0 < partial.coverage < 1.0
    assert "fact:near" in partial.missing_domain


def test_inverse_round_trip_and_lookup_are_exact() -> None:
    source, target = _projection_pair(changed=True)
    transport = _exact_transport()
    inverse = transport.inverse

    assert inverse is not None
    assert transport.validate_round_trip(inverse)
    assert (
        transport_state(
            transport_state(source.state, transport),
            inverse,
        )
        == source.state
    )
    assert transport_action(source.action, transport) == target.action
    assert find_transport((transport,), target.frame_id, source.frame_id) == inverse

    forward_certificate = transport.certificate(source, target)
    reverse_certificate = inverse.certificate(target, source)
    assert (
        forward_certificate.gauge_equivalence_key
        == reverse_certificate.gauge_equivalence_key
    )


def test_structural_orbit_witness_is_stable_and_forgets_event_evidence() -> None:
    source, target = _projection_pair(changed=True)
    transport = _exact_transport()
    forward_certificate = transport.certificate(source, target)
    inverse = transport.inverse
    assert inverse is not None
    reverse_certificate = inverse.certificate(target, source)

    forward = TransportOrbitWitness.from_certificate(forward_certificate)
    reverse = TransportOrbitWitness.from_certificate(reverse_certificate)

    assert forward.canonical_hash == reverse.canonical_hash
    assert forward.gauge_equivalence_key == reverse.gauge_equivalence_key
    assert "source_projection_hash" not in forward.canonical_payload
    assert forward_certificate.canonical_hash != reverse_certificate.canonical_hash
    assert find_transport((forward,), target.frame_id, source.frame_id) == inverse


def test_joint_transport_lookup_uses_an_exact_reverse_map() -> None:
    source, target = _projection_pair(changed=True)
    transport = _exact_transport()

    found = gauge_inference._find_transport(
        (transport,),
        target.frame_id,
        source.frame_id,
    )

    assert found == transport.inverse


def test_event_certificate_cannot_enter_joint_posterior_transport_hooks() -> None:
    source, target = _projection_pair(changed=True)
    certificate = _exact_transport().certificate(source, target)
    source_packet = _prediction(source.state, relation="near")

    found = gauge_inference._find_transport(
        (certificate,),
        source.frame_id,
        target.frame_id,
    )
    transported = gauge_inference._transport_prediction(source_packet, found)

    assert found is None
    assert transported is None


def test_map_and_gauge_hashes_are_permutation_invariant() -> None:
    source, target = _projection_pair()
    first = _exact_transport()
    reordered = _exact_transport(pair_order=True)

    assert first.canonical_hash == reordered.canonical_hash
    assert canonical_signature(source, first) == canonical_signature(target)
    assert gauge_equivalent(source, target, first)


def test_state_projection_diagram_commutes_on_shared_channels() -> None:
    source, target = _projection_pair()
    result = CommutativityAudit(_exact_transport()).audit_state(source, target)

    assert result.comparable
    assert result.commutes
    assert result.penalty == 0.0
    assert result.mismatched_channels == frozenset()
    assert result.compared_channels == frozenset(source.covered_channels)


def test_dynamics_diagram_commutes_after_prediction_transport() -> None:
    source, target = _projection_pair(changed=True)
    transport = _exact_transport()
    source_packet = _prediction(source.state, relation="near")
    target_packet = _prediction(target.state, relation="adjacent")
    transported = transport_prediction(source_packet, transport)

    assert (
        prediction_commutativity_penalty(
            transported,
            target_packet,
            source_packet.known_channels,
        )
        == 0.0
    )
    result = CommutativityAudit(transport).audit_dynamics(
        source_packet,
        target_packet,
        source_projection=source,
        target_projection=target,
    )
    assert result.comparable
    assert result.commutes
    assert result.penalty == 0.0
    assert "state_after" in result.matched_channels
    assert "action" in result.matched_channels


def test_incomplete_projection_and_disjoint_predictions_are_not_comparable() -> None:
    source, target = _projection_pair(complete=False)
    state_result = CommutativityAudit(_exact_transport()).state(source, target)
    assert not state_result.comparable
    assert not state_result.commutes
    assert state_result.incomplete_channels

    source_packet = PredictionPacket(
        progress_mean=1.0,
        known_channels=frozenset({"progress"}),
    )
    target_packet = PredictionPacket(
        terminal_probability=0.0,
        known_channels=frozenset({"terminal"}),
    )
    dynamics = CommutativityAudit(_exact_transport()).dynamics(
        source_packet,
        target_packet,
    )
    assert not dynamics.comparable
    assert not dynamics.commutes


def test_random_fact_map_and_binding_swap_degrade_commutativity() -> None:
    source, target = _projection_pair(changed=True)
    source_packet = _prediction(source.state, relation="near")
    target_packet = _prediction(target.state, relation="adjacent")
    exact = CommutativityAudit(_exact_transport()).dynamics(
        source_packet,
        target_packet,
        source_action=source.action,
        target_action=target.action,
    )
    corrupted = TransportMap(
        source.frame_id,
        target.frame_id,
        role_map=(("source_goal", "destination"),),
        fact_map=(("near", "west_of"),),
        action_map=(("ACTION1", "ACTION3"),),
    )
    degraded = CommutativityAudit(corrupted).dynamics(
        source_packet,
        target_packet,
        source_action=source.action,
        target_action=target.action,
    )

    assert exact.score == 1.0
    assert not degraded.comparable
    assert degraded.score < exact.score
    assert degraded.incomplete_channels


def test_many_to_one_transport_is_ambiguous_and_has_no_inverse() -> None:
    ambiguous = TransportMap(
        "root_only",
        "action_aligned_relational",
        role_map=(("left_goal", "goal"), ("right_goal", "goal")),
    )

    assert ambiguous.ambiguous
    assert ambiguous.ambiguity == 1
    assert ambiguous.inverse is None
    assert not ambiguous.validate_round_trip()


def test_orbit_witness_requires_certificate_provenance_and_nonempty_domain() -> None:
    source, _ = _projection_pair()
    target = FrameProjection(
        frame=ACTION_ALIGNED_RELATIONAL_FRAME,
        state=source.state,
        action=source.action,
        complete=source.complete,
        missing=source.missing,
        covered_channels=source.covered_channels,
    )
    empty = TransportMap(source.frame_id, target.frame_id)
    inverse = empty.inverse
    assert inverse is not None

    with pytest.raises(TypeError):
        TransportOrbitWitness(empty, inverse)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="issued from a transport certificate"):
        TransportOrbitWitness(
            empty,
            inverse,
            frozenset({"action:ACTION1"}),
            frozenset({"action:ACTION1"}),
            "0" * 64,
        )

    certificate = empty.certificate(source, target)
    assert certificate.certifies_gauge_equivalence
    with pytest.raises(ValueError, match="non-empty certified domain"):
        TransportOrbitWitness.from_certificate(certificate)


def test_orbit_witness_rejects_states_without_declared_frame_provenance() -> None:
    source, target = _projection_pair()
    certificate = TransportCertificate.from_projections(
        _exact_transport(),
        source.state,
        target.state,
    )

    assert certificate.exact
    assert not certificate.has_frame_provenance
    assert not certificate.certifies_gauge_equivalence
    with pytest.raises(ValueError, match="frame provenance"):
        TransportOrbitWitness.from_certificate(certificate)


def test_transport_certificate_is_factory_issued_and_stage_specific() -> None:
    source, target = _projection_pair()
    transport = _exact_transport()
    certificate = transport.certificate(source, target)

    with pytest.raises(ValueError, match="issued by from_projections"):
        replace(certificate, coverage=0.5)
    with pytest.raises(ValueError, match="issued by from_projections"):
        TransportCertificate(
            transport=transport,
            source_domain=certificate.source_domain,
            target_domain=certificate.target_domain,
            covered_domain=certificate.covered_domain,
            missing_domain=certificate.missing_domain,
            unmatched_target=certificate.unmatched_target,
            coverage=certificate.coverage,
            ambiguities=certificate.ambiguities,
            inverse_map=certificate.inverse_map,
            round_trip_exact=certificate.round_trip_exact,
            projections_complete=certificate.projections_complete,
            frame_ids_match=certificate.frame_ids_match,
        )

    source_before = replace(source, stage="before")
    target_before = replace(target, stage="before")
    source_after = replace(
        source,
        stage="after",
        state=replace(source.state, counters=(("step_count", 1.0),)),
    )
    target_after = replace(
        target,
        stage="after",
        state=replace(target.state, counters=(("step_count", 2.0),)),
    )
    source_transition = ProjectedTransition(
        "source-stage-fixture",
        source_before,
        source_after,
    )
    target_transition = ProjectedTransition(
        "target-stage-fixture",
        target_before,
        target_after,
    )
    before = TransportCertificate.from_projections(
        transport,
        source_transition,
        target_transition,
        stage="before",
    )
    after = TransportCertificate.from_projections(
        transport,
        source_transition,
        target_transition,
        stage="after",
    )

    assert before.projection_stage == "before"
    assert before.certifies_gauge_equivalence
    assert after.projection_stage == "after"
    assert after.exact
    assert not after.certifies_gauge_equivalence
    assert after.source_gauge_signature != after.target_gauge_signature


def test_dynamics_requires_domain_relevant_to_compared_symbols() -> None:
    packet = PredictionPacket(
        relation_deltas={"relation_added:near": 1.0},
        known_channels=frozenset({"relations"}),
    )
    unrelated = TransportMap(
        "root_only",
        "action_aligned_relational",
        fact_map=(("exists", "exists"),),
        domain=frozenset({"fact:exists"}),
    )

    result = CommutativityAudit(unrelated).dynamics(packet, packet)

    assert not result.comparable
    assert not result.commutes
    assert "transport_domain" in result.incomplete_channels


@pytest.mark.parametrize(
    "tamper",
    ("edge", "domain", "orbit_hash", "receipt", "summary_hash"),
)
def test_persisted_orbit_attestation_rejects_tampering(tamper: str) -> None:
    envelope = _persisted_orbit_envelope()
    assert TransportOrbitWitness.from_persisted_attestation(envelope)
    corrupted = copy.deepcopy(envelope)
    if tamper == "edge":
        corrupted["orbit_payload"]["symbol_edges"][0][0][2] = "same_color"
    elif tamper == "domain":
        corrupted["orbit_payload"]["certified_domain"].pop()
    elif tamper == "orbit_hash":
        corrupted["orbit_hash"] = "0" * 64
    elif tamper == "receipt":
        corrupted["attestation"]["receipt"] = "0" * 64
    else:
        corrupted["attestation"]["source_before_summary_hash"] = "f" * 64

    with pytest.raises((TypeError, ValueError)):
        TransportOrbitWitness.from_persisted_attestation(corrupted)


def test_certificate_rejects_inverse_with_incompatible_frame_endpoints() -> None:
    source, target = _projection_pair()
    transport = _exact_transport()
    wrong_inverse = TransportMap(
        "unrelated_source",
        "unrelated_target",
        role_map=tuple((target, source) for source, target in transport.role_map),
        fact_map=tuple((target, source) for source, target in transport.fact_map),
        action_map=tuple((target, source) for source, target in transport.action_map),
        domain=transport.codomain,
    )

    certificate = TransportCertificate.from_projections(
        transport,
        source,
        target,
        inverse=wrong_inverse,
    )

    assert not transport.validate_round_trip(wrong_inverse)
    assert not certificate.round_trip_exact
    assert not certificate.exact
    assert not certificate.certifies_gauge_equivalence


@pytest.mark.parametrize(
    "failure",
    ("wrong_endpoints", "ambiguous", "partial", "incompatible_inverse"),
)
def test_dynamics_audit_fails_closed_for_invalid_transport(failure: str) -> None:
    source, target = _projection_pair()
    packet = PredictionPacket(
        progress_mean=1.0,
        known_channels=frozenset({"progress"}),
    )
    if failure == "wrong_endpoints":
        transport = TransportMap(
            target.frame_id,
            source.frame_id,
            action_map=(("ACTION1", "ACTION2"),),
        )
    elif failure == "ambiguous":
        transport = TransportMap(
            source.frame_id,
            target.frame_id,
            role_map=(("left", "goal"), ("right", "goal")),
            action_map=(("ACTION1", "ACTION2"),),
        )
    elif failure == "partial":
        transport = TransportMap(
            source.frame_id,
            target.frame_id,
            action_map=(("ACTION1", "ACTION2"),),
            domain=frozenset({"action:ACTION1", "fact:near"}),
        )
    else:

        class IncompatibleInverseTransport(TransportMap):
            def inverted(self) -> TransportMap | None:
                return TransportMap(
                    "unrelated_source",
                    "unrelated_target",
                    action_map=(("ACTION2", "ACTION1"),),
                )

        transport = IncompatibleInverseTransport(
            source.frame_id,
            target.frame_id,
            action_map=(("ACTION1", "ACTION2"),),
        )

    result = CommutativityAudit(transport).dynamics(
        packet,
        packet,
        source_projection=source,
        target_projection=target,
    )

    assert not result.comparable
    assert not result.commutes
    assert result.penalty == 1.0
    assert result.incomplete_channels


def test_transport_state_preserves_grounded_ids_despite_symbol_collisions() -> None:
    entity_id = "source_goal"
    fact = GroundFact("exists", (entity_id,))
    valued_fact = GroundFact("same_attribute", (entity_id,), value=entity_id)
    state = AbstractState(
        entities=(AbstractEntity(entity_id, ("source_goal",)),),
        true_facts=frozenset({fact, valued_fact}),
        registers=(("target", entity_id),),
    )
    transport = TransportMap(
        "root_only",
        "action_aligned_relational",
        role_map=(("source_goal", "destination"),),
        fact_map=(
            (fact.key, GroundFact("exists", ("destination",)).key),
            ("exists", "exists"),
        ),
    )

    transported = transport_state(state, transport)

    assert transported.entities[0].entity_id == entity_id
    assert transported.entities[0].roles == ("destination",)
    assert transported.registers == (("target", entity_id),)
    assert all(
        term == entity_id
        for item in transported.true_facts
        for term in (*item.terms, item.value)
        if term
    )
    inverse = transport.inverse
    assert inverse is not None
    assert transport_state(transported, inverse) == state
