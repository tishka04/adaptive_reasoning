from theory.m2.multi_source_collision_fixture import (
    run_multi_source_collision_fixture,
)


def test_true_multi_source_collision_records_sources_without_support():
    payload = run_multi_source_collision_fixture()
    summary = payload["summary"]
    hypothesis = payload["merged_hypotheses"][0]

    assert summary["raw_proposals_seen"] == 2
    assert summary["deduplicated_hypotheses"] == 1
    assert summary["merged_sources_recorded"] is True
    assert set(hypothesis["source_generation"]["sources"]) == {"heuristic", "llm"}
    assert hypothesis["source_generation"]["priority_score_counted_as_support"] is False
    assert hypothesis["support"] == 0
