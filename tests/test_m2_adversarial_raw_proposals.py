from theory.m2.normalizer import normalize_raw_proposal, raw_proposals_from_json_text
from theory.m2.schema import FrontierConditionedHypothesis, RejectedProposal

from m2_test_helpers import frontier_request, raw_proposal


def test_adversarial_generator_confirmed_text_is_sanitized():
    frontier = frontier_request()
    raw = raw_proposal(
        frontier=frontier,
        predicted_effect="ACTION4 is confirmed by prior trace",
        raw_status="CONFIRMED",
    )

    result = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(result, FrontierConditionedHypothesis)
    assert result.status == "UNRESOLVED"
    assert result.support == 0


def test_adversarial_unknown_metric_rejected():
    frontier = frontier_request()
    result = normalize_raw_proposal(
        raw_proposal(frontier=frontier, predicted_metric="unknown_soft_metric"),
        frontier_request=frontier,
    )

    assert isinstance(result, RejectedProposal)
    assert result.reason == "unknown_metric"


def test_malformed_json_produces_no_final_hypothesis():
    assert raw_proposals_from_json_text("{bad json") == ()


def test_revision_true_raw_proposal_is_rejected():
    frontier = frontier_request()
    result = normalize_raw_proposal(
        raw_proposal(frontier=frontier, raw_revision_performed=True),
        frontier_request=frontier,
    )

    assert isinstance(result, RejectedProposal)
    assert result.reason == "raw_revision_performed_true"
