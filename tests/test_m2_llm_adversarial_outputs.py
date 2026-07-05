from theory.m2.mock_llm_generator import MockLLMGenerator
from theory.m2.normalizer import normalize_raw_proposal
from theory.m2.schema import FrontierConditionedHypothesis, RejectedProposal

from m2_test_helpers import frontier_request


def test_mock_llm_confirmed_output_is_normalized():
    frontier = frontier_request()
    raw = MockLLMGenerator(mode="confirmed").generate(frontier)[0]
    result = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(result, FrontierConditionedHypothesis)
    assert result.status == "UNRESOLVED"
    assert result.support == 0


def test_mock_llm_unknown_action_is_rejected():
    frontier = frontier_request()
    raw = MockLLMGenerator(mode="action9").generate(frontier)[0]
    result = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(result, RejectedProposal)
    assert result.reason == "unknown_candidate_action"


def test_mock_llm_unknown_metric_is_rejected():
    frontier = frontier_request()
    raw = MockLLMGenerator(mode="unknown_metric").generate(frontier)[0]
    result = normalize_raw_proposal(raw, frontier_request=frontier)

    assert isinstance(result, RejectedProposal)
    assert result.reason == "unknown_metric"
