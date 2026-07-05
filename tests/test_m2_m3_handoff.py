import json

from theory.m2.frontier_conditioned_hypotheses import (
    run_frontier_conditioned_hypotheses_from_payload,
)
from theory.m2.m3_handoff import run_m3_handoff_validation

from m2_test_helpers import frontier_request


def test_m3_handoff_can_rank_m2_requests_without_mutating_m2(tmp_path):
    outputs = run_frontier_conditioned_hypotheses_from_payload(
        {"frontier_requests": [frontier_request()]},
        input_frontier_path="fixture",
    )
    path = tmp_path / "m3_candidate_experiment_requests.json"
    path.write_text(json.dumps(outputs["m3_payload"]), encoding="utf-8")

    payload = run_m3_handoff_validation(m3_requests_path=path)

    assert payload["summary"]["m3_requests_loadable"] is True
    assert payload["summary"]["m3_selector_can_rank_m2_requests"] is True
    assert payload["summary"]["m3_can_execute_at_least_one_request"] is True
    assert payload["summary"]["m2_truth_status_unchanged"] == "NOT_EVALUATED_BY_M2"
    assert payload["summary"]["m2_artifacts_mutated_by_m3"] is False
