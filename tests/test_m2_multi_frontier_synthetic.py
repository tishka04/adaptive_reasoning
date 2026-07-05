from theory.m2.synthetic_frontier_stress import run_synthetic_multi_frontier_stress


def test_synthetic_multi_frontier_stress_covers_distinct_families():
    payload = run_synthetic_multi_frontier_stress()
    summary = payload["summary"]

    assert summary["synthetic_frontiers_consumed"] >= 4
    assert summary["distinct_hypothesis_families"] >= 4
    assert summary["invalid_frontiers_rejected_cleanly"] is True
    assert summary["empty_input_valid"] is True
    assert summary["wrong_confirmations"] == 0
