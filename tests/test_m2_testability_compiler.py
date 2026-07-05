from theory.m2.frontier_conditioned_hypotheses import (
    run_frontier_conditioned_hypotheses_from_payload,
)
from theory.m2.testability_compiler import compile_m3_request

from m2_test_helpers import frontier_request, valid_hypothesis


def test_testability_compiler_builds_ready_request():
    request = compile_m3_request(valid_hypothesis())

    assert request.status == "READY_FOR_M3"
    assert request.control_policy == "m3_dynamic_available_controls"
    assert request.support == 0
    assert request.falsification_criterion.metric == "local_patch_before_after"


def test_bp35_fixture_produces_at_least_four_m3_requests():
    payload = {
        "frontier_requests": [
            frontier_request(reason="context_not_covered_by_scope", context="after_ACTION6"),
            frontier_request(
                reason="confirmed_skill_blocked_by_failed_precondition",
                context="after_ACTION3_live_after_ACTION6",
                fallback_action="ACTION4",
                blocked_skill="ACTION6",
            ),
        ]
    }

    outputs = run_frontier_conditioned_hypotheses_from_payload(
        payload,
        input_frontier_path="fixture",
    )

    assert outputs["hypothesis_payload"]["summary"]["testable_hypotheses"] >= 4
    assert outputs["m3_payload"]["summary"]["ready_for_m3"] >= 4
