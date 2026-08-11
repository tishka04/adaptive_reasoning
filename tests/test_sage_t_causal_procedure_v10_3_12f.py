from __future__ import annotations

import json
import math

import pytest

from theory.sage_t.causal_procedure_v10_3_12f import (
    ARMS,
    MODEL_FAMILIES,
    CausalLabelQAError,
    CausalOutcome,
    CausalProcedureController,
    CausalProcedurePrior,
    CausalProcedureSpec,
    InterventionSignature,
    ProcedurePhase,
    SourceProcedureProjection,
    abstract_context_signature,
    assert_causal_transfer_safe,
    compile_source_prior,
    evaluate_source_prior,
    permuted_prior,
    preflight_prior,
    qa_source_projections,
    uniform_prior,
)
from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
)


def _state(
    index: int = 0,
    *,
    relation_target: str = "",
    entity_prefix: str = "e",
    palette: str = "red",
    mirrored: bool = False,
) -> AbstractState:
    centers = ((0.0, 0.0), (0.0, 1.0), (0.0, 2.0))
    if mirrored:
        centers = tuple((2.0 - row, 2.0 - column) for row, column in centers)
    entities = tuple(
        AbstractEntity(
            f"{entity_prefix}{position + 1}",
            ("node",),
            (("appearance", palette),),
            center,
        )
        for position, center in enumerate(centers)
    )
    facts = set()
    if relation_target:
        facts.add(
            GroundFact(
                "reachable",
                (entities[0].entity_id, f"{entity_prefix}{relation_target}"),
            )
        )
    return AbstractState(
        entities=entities,
        true_facts=frozenset(facts),
        counters=(("levels_completed", 0.0),),
        topology=(("components", index + 1),),
        regime_index=index,
    )


def _transition(
    before: AbstractState,
    action: ActionCandidate,
    after: AbstractState,
    *,
    events: tuple[str, ...] = (),
    objects: dict[str, float] | None = None,
    terminal: bool = False,
) -> ObservedTransition:
    return ObservedTransition(
        state_before=before,
        action=action,
        state_after=after,
        observation=PredictionPacket(
            object_deltas=objects or {},
            terminal_probability=float(terminal),
            known_channels=frozenset({"objects", "terminal"}),
            state_after=after,
        ),
        events=events,
    )


def _source_rows() -> tuple[SourceProcedureProjection, ...]:
    rows = []
    for group in range(6):
        rows.append(
            SourceProcedureProjection(
                source_slot="source_a",
                group_index=group,
                inferred_family="stable_repeat",
                outcome_mode="persistent_motion" if group % 2 else "relation_change",
                correspondence_confidence=0.91,
                level_delta=int(group == 5),
            )
        )
        rows.append(
            SourceProcedureProjection(
                source_slot="source_b",
                group_index=group,
                inferred_family="relational_successor",
                outcome_mode="relation_change" if group % 2 else "action_space_change",
                correspondence_confidence=0.88,
                level_delta=int(group == 5),
            )
        )
    return tuple(rows)


def test_public_contract_is_frozen_and_transfer_safe() -> None:
    assert ARMS == (
        "source_closed_loop",
        "uniform_closed_loop",
        "permuted_source_closed_loop",
        "source_open_loop",
    )
    assert MODEL_FAMILIES == (
        "stable_repeat",
        "relational_successor",
        "state_conditioned_switch",
        "null_or_unsafe",
    )
    spec = CausalProcedureSpec()
    assert spec.maximum_hypotheses == 8
    assert spec.maximum_candidates == 16
    assert spec.posterior_threshold == 0.80
    assert spec.posterior_margin == 0.20
    assert spec.verification_contexts == 2
    assert spec.stagnation_limit == 4
    assert spec.maximum_revisions == 2
    assert spec.option_horizon == 16
    assert_causal_transfer_safe(spec.safe_payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"game_id": "hidden"},
        {"action_name": "ACTION6"},
        {"coordinates": [1, 2]},
        {"color": 4},
        {"entity_id": "e7"},
        {"frame_hash": "f" * 64},
        {"trajectory": ["step"]},
        {"abstract_note": "lp85"},
    ),
)
def test_transfer_firewall_rejects_grounded_payloads(payload: dict) -> None:
    with pytest.raises(ValueError):
        assert_causal_transfer_safe(payload)


def test_intervention_and_context_are_d4_palette_id_and_action_invariant() -> None:
    left = ActionCandidate("ACTION6", {"x": 0, "y": 1})
    right = ActionCandidate("ACTION19", {"x": 98, "y": -4})
    assert InterventionSignature.from_candidate(left) == InterventionSignature.from_candidate(right)

    first = _state(entity_prefix="e", palette="red", mirrored=False)
    transformed = _state(entity_prefix="z", palette="blue", mirrored=True)
    assert abstract_context_signature(first) == abstract_context_signature(transformed)


def test_outcome_rejects_relation_conflicts_and_excludes_birth_death_from_persistence() -> None:
    before = _state()
    after = _state(1)
    action = ActionCandidate("ACTION6", {"x": 0, "y": 0})
    conflict = _transition(
        before,
        action,
        after,
        events=("relation_added:contact", "relation_removed:contact"),
    )
    with pytest.raises(ValueError, match="conflicting relation delta"):
        CausalOutcome.from_observed_transition(conflict)

    birth_death = _transition(
        before,
        action,
        after,
        events=("created", "removed"),
        objects={"created": 1.0, "removed": 1.0},
    )
    outcome = CausalOutcome.from_observed_transition(birth_death)
    assert outcome.persistent_moves == 0
    assert outcome.persistent_transformations == 0
    assert outcome.relations_added == ()
    assert outcome.relations_removed == ()
    assert outcome.level_delta == 0


def test_source_prior_uses_equal_source_mass_and_grouped_holdout() -> None:
    rows = _source_rows()
    qa = qa_source_projections(rows)
    assert qa["passed"] is True
    assert qa["source_contribution_fractions"] == {"source_a": 0.5, "source_b": 0.5}

    compilation = compile_source_prior(rows)
    weights = compilation.prior.weights
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    assert max(weights.values()) <= 0.70
    assert max(weights.values()) > min(weights.values())
    assert weights["stable_repeat"] == pytest.approx(weights["relational_successor"])
    assert compilation.snapshot()["source_contribution_fractions"] == {
        "source_a": 0.5,
        "source_b": 0.5,
    }

    evaluation = evaluate_source_prior(compilation.prior, rows)
    assert evaluation["grouped_leave_one_root_out"] is True
    assert evaluation["folds"] == 12
    assert evaluation["passed"] is True
    for result in evaluation["per_source"].values():
        assert result["log_loss_improvement_fraction"] >= 0.05
        assert result["identification_cost_reduction_fraction"] >= 0.20


def test_source_qa_fails_closed_on_universal_labels() -> None:
    rows = tuple(
        SourceProcedureProjection(
            source_slot=slot,
            group_index=group,
            inferred_family=(
                "stable_repeat" if slot == "source_a" else "relational_successor"
            ),
            outcome_mode="universal_mode",
        )
        for slot in ("source_a", "source_b")
        for group in range(3)
    )
    qa = qa_source_projections(rows)
    assert qa["passed"] is False
    assert qa["verdict"] == "CAUSAL_LABEL_QA_MISS"
    with pytest.raises(CausalLabelQAError):
        compile_source_prior(rows)


def test_uniform_and_permuted_priors_preserve_only_the_required_controls() -> None:
    source = compile_source_prior(_source_rows()).prior
    uniform = uniform_prior()
    wrong = permuted_prior(source)
    assert set(uniform.weights.values()) == {0.25}
    assert sorted(wrong.weights.values()) == sorted(source.weights.values())
    assert wrong.weights != source.weights
    assert preflight_prior(source)["passed"] is True
    restored = CausalProcedurePrior(source.snapshot())
    assert restored.prior_checksum == source.prior_checksum


def test_closed_loop_reaches_control_then_revises_on_predictive_mismatch() -> None:
    prior = CausalProcedurePrior(
        weights={
            "stable_repeat": 0.85,
            "relational_successor": 0.05,
            "state_conditioned_switch": 0.05,
            "null_or_unsafe": 0.05,
        }
    )
    controller = CausalProcedureController("source_closed_loop", prior=prior)
    candidate = ActionCandidate("ACTION6", {"x": 0, "y": 0})
    productive = CausalOutcome(persistent_moves=1)

    first = controller.propose(_state(0), [candidate])
    first_update = controller.observe(
        state_before=_state(0),
        state_after=_state(1),
        selected=first.candidate,
        outcome=productive,
    )
    assert first_update.phase_after == ProcedurePhase.VERIFY.value

    second = controller.propose(_state(1), [candidate])
    second_update = controller.observe(
        state_before=_state(1),
        state_after=_state(2),
        selected=second.candidate,
        outcome=productive,
    )
    assert second_update.phase_after == ProcedurePhase.CONTROL.value

    third = controller.propose(_state(2), [candidate])
    mismatch = controller.observe(
        state_before=_state(2),
        state_after=_state(2),
        selected=third.candidate,
        outcome=CausalOutcome(noop=True),
    )
    assert mismatch.mismatch is True
    assert mismatch.revised is True
    assert mismatch.phase_after == ProcedurePhase.REVISE.value
    assert controller.summary()["revisions"] == 1
    assert controller.summary()["legacy_fallback_actions"] == 0
    assert "ACTION6" not in json.dumps(controller.summary())

    controller.reset()
    summary = controller.summary()
    assert summary["actions"] == 0
    assert summary["belief"]["observations"] == 0
    assert summary["belief"]["distinct_interventions"] == 0


def _successor_choice(relation_target: str) -> int:
    prior = CausalProcedurePrior(
        weights={
            "stable_repeat": 0.05,
            "relational_successor": 0.85,
            "state_conditioned_switch": 0.05,
            "null_or_unsafe": 0.05,
        }
    )
    controller = CausalProcedureController("source_closed_loop", prior=prior)
    candidates = tuple(
        ActionCandidate("ACTION6", {"x": index, "y": 0}) for index in range(3)
    )
    start = _state(0)
    first = controller.propose(start, candidates)
    assert first.candidate is not None
    assert first.candidate.action_data["x"] == 0
    related = _state(1, relation_target=relation_target)
    controller.observe(
        state_before=start,
        state_after=related,
        selected=first.candidate,
        outcome=CausalOutcome(relations_added=("reachable",)),
    )
    successor = controller.propose(related, candidates)
    assert successor.candidate is not None
    assert successor.intervention is not None
    assert successor.intervention.target_role == "node"
    return int(successor.candidate.action_data["x"])


def test_relational_successor_proposal_uses_current_state_relation_and_role() -> None:
    assert _successor_choice("2") == 1
    assert _successor_choice("3") == 2


def test_candidate_order_does_not_change_the_selected_intervention() -> None:
    prior = uniform_prior()
    candidates = tuple(
        ActionCandidate("ACTION6", {"x": index, "y": 0}) for index in range(24)
    )
    forward = CausalProcedureController("uniform_closed_loop", scope=2, prior=prior)
    reverse = CausalProcedureController("uniform_closed_loop", scope=2, prior=prior)
    left = forward.propose(_state(), candidates)
    right = reverse.propose(_state(), tuple(reversed(candidates)))
    assert left.candidate is not None and right.candidate is not None
    assert left.candidate.key == right.candidate.key
    assert left.candidates_inspected == right.candidates_inspected == 16


def test_level_delta_is_the_only_success_credit_and_stops_control() -> None:
    controller = CausalProcedureController("uniform_closed_loop")
    candidate = ActionCandidate("ACTION6")
    decision = controller.propose(_state(), [candidate])
    update = controller.observe(
        state_before=_state(),
        state_after=_state(1),
        selected=decision.candidate,
        outcome=CausalOutcome(persistent_moves=1, level_delta=1),
    )
    assert update.abstained is True
    assert update.reason == "level_progress_observed"
    assert controller.summary()["level_deltas"] == 1

