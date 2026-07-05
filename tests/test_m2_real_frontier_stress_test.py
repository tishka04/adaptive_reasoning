import json

from theory.m2.real_frontier_stress_test import run_real_frontier_stress_test

from m2_test_helpers import frontier_request


def test_real_frontier_stress_reports_ratios_without_fixed_minimum(tmp_path):
    path = tmp_path / "frontier_handoff_requests.json"
    path.write_text(
        json.dumps({"frontier_requests": [frontier_request(), frontier_request(context="after_ACTION3")]}),
        encoding="utf-8",
    )

    payload = run_real_frontier_stress_test(frontier_paths=[path])
    summary = payload["summary"]

    assert summary["real_frontier_files_consumed"] == 1
    assert summary["frontier_requests_consumed"] == 2
    assert summary["hypotheses_generated"] >= 1
    assert "testable_ratio" in summary
    assert summary["wrong_confirmations"] == 0
