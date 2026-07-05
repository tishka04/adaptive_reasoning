from theory.m2.normalizer import normalize_raw_proposal
from theory.m2.schema import FrontierConditionedHypothesis, RejectedProposal

from m2_test_helpers import frontier_request, raw_proposal


def test_normalizer_forces_candidate_only_fields():
    frontier = frontier_request()
    raw = raw_proposal(
        frontier=frontier,
        raw_status="CONFIRMED",
        raw_support=1,
        raw_truth_status="CONFIRMED",
    )

    result = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(result, FrontierConditionedHypothesis)
    assert result.status == "UNRESOLVED"
    assert result.support == 0
    assert result.truth_status == "NOT_EVALUATED_BY_M2"
    assert result.revision_performed is False
    assert {
        "status_forced_unresolved",
        "support_forced_zero",
        "truth_status_forced_not_evaluated_by_m2",
    }.issubset(set(result.source_generation.normalization_warnings))


def test_normalizer_rejects_unknown_action():
    frontier = frontier_request()
    raw = raw_proposal(frontier=frontier, candidate_action="ACTION9")

    result = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(result, RejectedProposal)
    assert result.reason == "unknown_candidate_action"


def test_normalizer_fills_missing_controls_with_m3_dynamic_policy():
    frontier = frontier_request()
    raw = raw_proposal(frontier=frontier, suggested_control_actions=())

    result = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(result, FrontierConditionedHypothesis)
    assert result.testability.control_policy == "m3_dynamic_available_controls"
    assert result.testability.suggested_control_actions
