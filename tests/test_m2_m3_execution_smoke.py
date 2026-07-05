import json

import theory.m2.m3_execution_smoke as smoke
from theory.m2.frontier_conditioned_hypotheses import (
    run_frontier_conditioned_hypotheses_from_payload,
)

from m2_test_helpers import frontier_request


def test_m3_execution_smoke_selects_exact_contextual_request(monkeypatch, tmp_path):
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
    path = tmp_path / "m3_candidate_experiment_requests.json"
    path.write_text(json.dumps(outputs["m3_payload"]), encoding="utf-8")

    def fake_execute_contextual_m2_request(
        request,
        *,
        environments_dir=None,
        target_context_signature="",
    ):
        assert request.context_replay == ("ACTION6", "ACTION3")
        assert request.target_action == "ACTION4"
        return {
            "controlled_experiments_run": 1,
            "support_events": 1,
            "contradiction_events": 0,
            "support": 0,
            "revision_performed": False,
            "wrong_confirmations": 0,
        }

    monkeypatch.setattr(
        smoke,
        "execute_contextual_m2_request",
        fake_execute_contextual_m2_request,
    )

    payload = smoke.run_m3_execution_smoke(m3_requests_path=path)
    summary = payload["summary"]

    assert summary["m3_context_replay_exact"] is True
    assert summary["m3_executed_target_vs_control"] is True
    assert summary["support_events"] == 1
    assert summary["m2_artifacts_mutated_by_m3"] is False
    assert summary["a32_remains_only_verdict_location"] is True
