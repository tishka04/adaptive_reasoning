from theory.m2.schema import (
    M2_HYPOTHESIS_STATUS,
    M2_TRUTH_STATUS,
    M3CandidateExperimentRequest,
)
from theory.m2.testability_compiler import compile_m3_request

from m2_test_helpers import valid_hypothesis


def test_schema_defaults_keep_hypothesis_unresolved_candidate_only():
    hypothesis = valid_hypothesis()

    assert hypothesis.status == M2_HYPOTHESIS_STATUS
    assert hypothesis.support == 0
    assert hypothesis.controlled_test_required is True
    assert hypothesis.truth_status == M2_TRUTH_STATUS
    assert hypothesis.revision_performed is False
    assert hypothesis.wrong_confirmations == 0
    assert hypothesis.trace_support_counted_as_proof is False
    assert hypothesis.prior_counted_as_proof is False
    assert hypothesis.source_generation.priority_score_counted_as_support is False


def test_schema_compiles_m3_request_without_truth_claim():
    request = compile_m3_request(valid_hypothesis())

    assert isinstance(request, M3CandidateExperimentRequest)
    assert request.status == "READY_FOR_M3"
    assert request.truth_status == M2_TRUTH_STATUS
    assert request.support == 0
    assert request.control_policy == "m3_dynamic_available_controls"
