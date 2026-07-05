import json

from theory.m2.frontier_intake import run_frontier_intake

from m2_test_helpers import frontier_request


def test_frontier_intake_reads_a40_and_filters_ready_requests(tmp_path):
    path = tmp_path / "frontier_handoff_requests.json"
    closed = frontier_request()
    closed["request_id"] = "frontier::closed"
    closed["ready_for_m1_or_m3"] = False
    path.write_text(
        json.dumps({"frontier_requests": [frontier_request(), closed]}),
        encoding="utf-8",
    )

    payload = run_frontier_intake(frontier_path=path)

    assert payload["summary"]["frontier_requests_consumed"] == 1
    assert payload["summary"]["open_frontiers"] == 1
    assert payload["summary"]["closed_or_invalid_frontiers"] == 1
    assert payload["config"]["inputs_read"] == ["A40"]


def test_frontier_intake_empty_a40_is_valid(tmp_path):
    path = tmp_path / "frontier_handoff_requests.json"
    path.write_text(json.dumps({"frontier_requests": []}), encoding="utf-8")

    payload = run_frontier_intake(frontier_path=path)

    assert payload["summary"]["frontier_requests_consumed"] == 0
    assert payload["summary"]["ready_for_generation"] is False
    assert payload["wrong_confirmations"] == 0


def test_frontier_intake_marks_incomplete_frontier_invalid(tmp_path):
    path = tmp_path / "frontier_handoff_requests.json"
    path.write_text(
        json.dumps({"frontier_requests": [{"ready_for_m1_or_m3": True}]}),
        encoding="utf-8",
    )

    payload = run_frontier_intake(frontier_path=path)

    assert payload["summary"]["closed_or_invalid_frontiers"] == 1
    assert payload["frontier_requests"] == []
