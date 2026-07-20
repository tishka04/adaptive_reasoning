"""Active cross-branch mediated acquisition tests for SAGE.8w."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.live_transition_loop import build_observation
from theory.online_causal_option import OnlineCausalOptionStore
from theory.online_causal_subgoal_graph import CausalSubgoalEdge
from theory.online_mediated_replication import (
    MediatedReplicationStatus,
    OnlineMediatedReplicationStore,
)
from theory.online_semantic_intervention import semantic_intervention_anchor


def _observation():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2, 2] = 8
    grid[7, 7] = 4
    return build_observation(
        grid,
        available_actions=["ACTION6"],
        infer_players=False,
    )


def _anchor(x: int = 2, y: int = 2):
    observation = _observation()
    return semantic_intervention_anchor(
        "ACTION6",
        {"x": x, "y": y},
        observation,
    )


def _outcome(
    status: str = "needs_mediator_contrast",
    *,
    gain: float = 1.0,
    candidates=("mediator:a", "mediator:b"),
):
    return {
        "observed": True,
        "status": status,
        "gain": gain,
        "mode_signature": "mode:one",
        "action_transfer_signature": _anchor().transfer_signature,
        "candidate_mediator_signatures": list(candidates),
        "supported_mediator_signature": (
            candidates[0] if status == "supported" and candidates else ""
        ),
    }


def _request(memory: OnlineMediatedReplicationStore, *, branch: int = 1):
    request_id = memory.observe_hypothesis(
        option_id="option",
        edge_key="edge",
        objective_id="objective",
        downstream_subgoal_id="subgoal",
        anchor=_anchor(),
        branch_index=branch,
        context_signature="source-context",
        mediated_outcome=_outcome(),
    )
    assert request_id
    return request_id


def test_progressive_ambiguity_creates_one_persistent_replication_request():
    memory = OnlineMediatedReplicationStore()

    first = _request(memory)
    second = _request(memory)

    assert first == second
    request = memory.requests()[0]
    assert request.status == MediatedReplicationStatus.PENDING
    assert request.source_branch == 1
    assert request.candidate_mediator_signatures == (
        "mediator:a",
        "mediator:b",
    )
    assert memory.summary()["requests_created"] == 1


def test_request_cannot_activate_in_source_branch_but_activates_after_reopening():
    memory = OnlineMediatedReplicationStore()
    request_id = _request(memory)

    blocked = memory.note_opening(
        option_id="option",
        edge_key="edge",
        branch_index=1,
        opening_context="same-branch",
    )
    memory.start_branch(2)
    assert memory.preferred_preparation_edge_key() == "edge"
    memory.note_preparation_action("edge")
    activated = memory.note_opening(
        option_id="option",
        edge_key="edge",
        branch_index=2,
        opening_context="independent-opening",
    )

    assert blocked == ""
    assert activated == request_id
    assert memory.active_request is not None
    assert memory.active_request.active_branch == 2
    assert memory.summary()["same_branch_blocks"] == 1
    assert memory.summary()["preparation_actions"] == 1


def test_active_request_selects_only_the_same_semantic_intervention():
    memory = OnlineMediatedReplicationStore()
    request_id = _request(memory)
    memory.start_branch(2)
    memory.note_opening(
        option_id="option",
        edge_key="edge",
        branch_index=2,
        opening_context="independent-opening",
    )

    exact = memory.predict(option_id="option", anchor=_anchor())
    other = memory.predict(option_id="option", anchor=_anchor(7, 7))

    assert exact is not None and exact.compatible is True
    assert exact.cross_branch is True
    assert exact.selection_rank == 9
    assert other is not None and other.compatible is False
    memory.note_selection(exact)
    assert memory.requests()[0].selected_actions == 1
    assert memory.summary()["selections"] == 1
    assert exact.request_id == request_id


def test_online_replication_confirms_without_any_terminal_label():
    memory = OnlineMediatedReplicationStore()
    request_id = _request(memory)
    memory.start_branch(2)
    memory.note_opening(
        option_id="option",
        edge_key="edge",
        branch_index=2,
        opening_context="independent-opening",
    )
    prediction = memory.predict(option_id="option", anchor=_anchor())
    assert prediction is not None
    memory.note_selection(prediction)

    memory.observe_hypothesis(
        option_id="option",
        edge_key="edge",
        objective_id="objective",
        downstream_subgoal_id="subgoal",
        anchor=_anchor(),
        branch_index=2,
        context_signature="replication-transition",
        mediated_outcome=_outcome(
            "supported",
            candidates=("mediator:a",),
        ),
        selected_request_id=request_id,
    )

    request = memory.requests()[0]
    assert request.status == MediatedReplicationStatus.CONFIRMED
    assert request.supported_mediator_signature == "mediator:a"
    assert memory.summary()["confirmations"] == 1


def test_contradiction_refutes_and_repeated_ambiguity_expires_boundedly():
    refuted = OnlineMediatedReplicationStore()
    request_id = _request(refuted)
    refuted.start_branch(2)
    refuted.note_opening(
        option_id="option", edge_key="edge", branch_index=2,
        opening_context="opening-2",
    )
    prediction = refuted.predict(option_id="option", anchor=_anchor())
    assert prediction is not None
    refuted.note_selection(prediction)
    refuted.observe_hypothesis(
        option_id="option", edge_key="edge", objective_id="objective",
        downstream_subgoal_id="subgoal", anchor=_anchor(), branch_index=2,
        context_signature="contradiction",
        mediated_outcome=_outcome("contradicted", gain=-1.0, candidates=()),
        selected_request_id=request_id,
    )
    assert refuted.requests()[0].status == MediatedReplicationStatus.REFUTED

    bounded = OnlineMediatedReplicationStore(max_attempts_per_request=1)
    bounded_id = _request(bounded)
    bounded.start_branch(2)
    bounded.note_opening(
        option_id="option", edge_key="edge", branch_index=2,
        opening_context="opening-2",
    )
    bounded_prediction = bounded.predict(option_id="option", anchor=_anchor())
    assert bounded_prediction is not None
    bounded.note_selection(bounded_prediction)
    bounded.observe_hypothesis(
        option_id="option", edge_key="edge", objective_id="objective",
        downstream_subgoal_id="subgoal", anchor=_anchor(), branch_index=2,
        context_signature="still-ambiguous", mediated_outcome=_outcome(),
        selected_request_id=bounded_id,
    )
    assert bounded.requests()[0].status == MediatedReplicationStatus.EXPIRED


def test_causal_option_prioritizes_reserved_replication_action():
    observation = _observation()
    options = OnlineCausalOptionStore()
    edge = CausalSubgoalEdge(
        edge_key="edge",
        source_objective_id="source",
        target_objective_id="objective",
        minimum_independent_support=2,
    )
    edge.support_events = 2
    edge.support_contexts.update({"a", "b"})
    edge.support_branches.update({0, 1})
    options.sync_confirmed_edges([edge])
    option = options.options()[0]
    options.mediated_replications.observe_hypothesis(
        option_id=option.option_id,
        edge_key=edge.edge_key,
        objective_id="objective",
        downstream_subgoal_id="",
        anchor=_anchor(),
        branch_index=0,
        context_signature="source",
        mediated_outcome=_outcome(),
    )
    options.start_branch()
    options.note_openings([edge.edge_key], context_signature="reopened")
    predictions = options.mediated_replication_action_predictions(
        observation,
        safe_actions=["ACTION6"],
        click_actions=(
            SimpleNamespace(action_args={"x": 2, "y": 2}),
            SimpleNamespace(action_args={"x": 7, "y": 7}),
        ),
    )

    selection = options.select_downstream(
        observation,
        safe_actions=["ACTION6"],
        click_actions=(
            SimpleNamespace(action_args={"x": 2, "y": 2}),
            SimpleNamespace(action_args={"x": 7, "y": 7}),
        ),
        mediated_replication_predictions=predictions,
    )

    assert selection is not None
    assert selection.action_data == {"x": 2, "y": 2}
    assert selection.mediated_replication_request_id
    assert selection.mediated_cross_branch_replication is True
    assert selection.mediated_exact_semantic_replication is True


def test_replication_ablation_records_nothing():
    memory = OnlineMediatedReplicationStore(enabled=False)

    request_id = memory.observe_hypothesis(
        option_id="option", edge_key="edge", objective_id="objective",
        downstream_subgoal_id="subgoal", anchor=_anchor(), branch_index=1,
        context_signature="source", mediated_outcome=_outcome(),
    )

    assert request_id == ""
    assert memory.summary()["requests_created"] == 0
    assert memory.requests() == []
