from theory.m2.frontier_conditioned_hypotheses import (
    run_frontier_conditioned_hypotheses_from_payload,
)

from m2_test_helpers import frontier_request


def test_multi_source_mode_keeps_support_zero_and_records_sources():
    outputs = run_frontier_conditioned_hypotheses_from_payload(
        {"frontier_requests": [frontier_request()]},
        input_frontier_path="fixture",
        generator_mode="heuristic_plus_mock_llm_plus_mock_world_model",
        llm_enabled=True,
        world_model_enabled=True,
    )

    summary = outputs["hypothesis_payload"]["summary"]
    audit_summary = outputs["generation_audit"]["summary"]
    hypotheses = [
        hypothesis
        for batch in outputs["hypothesis_payload"]["hypothesis_batches"]
        for hypothesis in batch["candidate_hypotheses"]
    ]

    assert summary["hypotheses_generated"] >= 2
    assert audit_summary["support_unchanged_at_zero"] is True
    assert audit_summary["priority_score_counted_as_support"] is False
    assert all(hypothesis["support"] == 0 for hypothesis in hypotheses)
