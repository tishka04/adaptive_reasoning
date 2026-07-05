from dataclasses import replace

from theory.m2.hypothesis_merger import merge_hypotheses

from m2_test_helpers import frontier_request, valid_hypothesis


def test_merger_deduplicates_and_records_multiple_sources():
    frontier = frontier_request()
    left = valid_hypothesis(frontier=frontier)
    right = replace(
        left,
        source_generation=replace(
            left.source_generation,
            sources=("llm",),
            raw_proposal_ids=("raw::llm::001",),
            rationales=("LLM duplicate.",),
        ),
    )

    merged = merge_hypotheses(
        [left, right],
        frontiers_by_request_id={frontier["request_id"]: frontier},
    )

    assert len(merged) == 1
    assert set(merged[0].source_generation.sources) == {"heuristic", "llm"}
    assert merged[0].source_generation.priority_score_counted_as_support is False
    assert merged[0].support == 0
