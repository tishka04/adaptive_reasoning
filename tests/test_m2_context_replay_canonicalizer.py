from theory.m2.context_replay import canonical_context_replay
from theory.m2.frontier_conditioned_hypotheses import (
    run_frontier_conditioned_hypotheses_from_payload,
)

from m2_test_helpers import frontier_request


def test_canonical_context_replay_preserves_live_after_action6_prefix():
    frontier = frontier_request(
        reason="confirmed_skill_blocked_by_failed_precondition",
        context="after_ACTION3_live_after_ACTION6",
        blocked_skill="ACTION6",
        fallback_action="ACTION4",
    )

    replay = canonical_context_replay(frontier, proposed_replay=("ACTION3",))

    assert replay == ("ACTION6", "ACTION3")


def test_m2_outputs_use_canonical_replay_for_live_after_frontier():
    frontier = frontier_request(
        reason="confirmed_skill_blocked_by_failed_precondition",
        context="after_ACTION3_live_after_ACTION6",
        blocked_skill="ACTION6",
        fallback_action="ACTION4",
    )
    outputs = run_frontier_conditioned_hypotheses_from_payload(
        {"frontier_requests": [frontier]},
        input_frontier_path="fixture",
    )
    hypotheses = outputs["hypothesis_payload"]["hypothesis_batches"][0][
        "candidate_hypotheses"
    ]
    requests = outputs["m3_payload"]["experiment_requests"]

    assert hypotheses
    assert all(
        hypothesis["testability"]["required_context_replay"] == ["ACTION6", "ACTION3"]
        for hypothesis in hypotheses
    )
    assert all(
        request["context_replay"] == ["ACTION6", "ACTION3"]
        for request in requests
    )
