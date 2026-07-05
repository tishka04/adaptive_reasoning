from theory.m2.mock_world_model import (
    MockWorldModel,
    generate_world_model_raw_proposals,
)
from theory.m2.world_model_adapter import frontier_state_from_request

from m2_test_helpers import frontier_request


def test_world_model_predictions_are_priority_only():
    frontier = frontier_request()
    state = frontier_state_from_request(frontier)
    predictions = MockWorldModel().score_candidate_actions(
        state,
        ["ACTION3", "ACTION4"],
    )

    assert predictions[0].epistemic_value > 0
    assert predictions[0].recommended_metric in {
        "contact_graph_before_after",
        "local_patch_before_after",
    }


def test_world_model_raw_proposals_do_not_add_support():
    proposals = generate_world_model_raw_proposals(frontier_request())

    assert proposals
    assert all(proposal.source == "world_model" for proposal in proposals)
    assert all(proposal.raw_support is None for proposal in proposals)
