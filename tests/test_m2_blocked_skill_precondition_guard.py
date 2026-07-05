from theory.m2.mock_world_model import generate_world_model_raw_proposals
from theory.m2.normalizer import normalize_raw_proposal
from theory.m2.schema import FrontierConditionedHypothesis
from theory.m2.testability_compiler import compile_m3_request

from m2_test_helpers import frontier_request


def test_blocked_skill_candidate_is_not_ready_for_m3_when_precondition_active():
    frontier = frontier_request(
        reason="confirmed_skill_blocked_by_failed_precondition",
        context="after_ACTION3_live_after_ACTION6",
        blocked_skill="ACTION6",
        fallback_action="ACTION4",
    )
    raw = [
        proposal
        for proposal in generate_world_model_raw_proposals(frontier)
        if proposal.candidate_action == "ACTION6"
    ][0]

    hypothesis = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(hypothesis, FrontierConditionedHypothesis)
    assert hypothesis.candidate_action == "ACTION6"
    assert hypothesis.testability.testable is False
    assert hypothesis.testability.blocking_reason == (
        "blocked_skill_precondition_still_active"
    )
    assert hypothesis.support == 0

    request = compile_m3_request(hypothesis)
    assert request.status == "BLOCKED_NOT_TESTABLE"
    assert request.support == 0
