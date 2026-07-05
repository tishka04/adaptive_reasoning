from theory.m2.llm_adapter import build_llm_prompt_payload
from theory.m2.mock_llm_generator import MockLLMGenerator

from m2_test_helpers import frontier_request


def test_llm_adapter_prompt_contains_required_frontier_fields():
    frontier = frontier_request(blocked_skill="ACTION6")
    payload = build_llm_prompt_payload(frontier)

    assert payload["frontier_context_id"] == frontier["frontier_context_id"]
    assert payload["blocked_skill"] == "ACTION6"
    assert "allowed_actions" in payload
    assert "available_metrics" in payload
    assert "do_not_confirm" in payload["constraints"]


def test_mock_llm_generator_truncates_to_five():
    proposals = MockLLMGenerator(mode="many").generate(frontier_request())

    assert len(proposals) == 5
    assert all(proposal.source == "llm" for proposal in proposals)
