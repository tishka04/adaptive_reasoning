import json
from pathlib import Path

import pytest

from theory.p2.objective_conversion_frontier_records import (
    OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
    OBJECTIVE_CONVERSION_FRONTIER,
    TERMINAL_SAFE_BUT_PASSIVE,
    build_objective_conversion_frontier_record,
    run_objective_conversion_frontier_records,
    validate_objective_conversion_frontier_record,
)
from theory.p2.policy_frontier_records import TRUTH_STATUS


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _p3g1_payload(
    *,
    status: str = "POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY",
    support: int = 0,
    verdict: bool = False,
    completion: bool = False,
    terminal_reduced: bool = True,
) -> dict:
    return {
        "summary": {
            "policy_utility_status": status,
            "best_objective_aware_condition": "objective_aware_abstract_policy_lambda_0",
            "best_objective_aware_lambda_terminal_risk": 0.0,
            "p3g0_mean_progress_proxy": 140.0,
            "best_objective_mean_progress_proxy": 132.5,
            "p3g0_terminal_rate": 0.75,
            "best_objective_terminal_rate": 0.0,
            "p3g0_terminal_adjusted_progress": 20.0,
            "best_objective_terminal_adjusted_progress": 132.5,
            "progress_proxy_preserved_vs_p3g0": False,
            "terminal_rate_reduced_vs_p3g0": terminal_reduced,
            "objective_completion_signal": completion,
            "policy_result_counted_as_scientific_verdict": verdict,
            "candidate_model_counted_as_confirmed_mechanic": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "policy_result_counted_as_scientific_verdict": verdict,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_objective_conversion_frontier_records_builds_review_ready_record(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "p3g1.json", _p3g1_payload())

    payload = run_objective_conversion_frontier_records(
        p3_objective_aware_consolidation_path=source,
    )

    assert payload["summary"]["frontier_records"] == 1
    assert payload["summary"]["frontier_type"] == OBJECTIVE_CONVERSION_FRONTIER
    assert payload["summary"]["frontier_reason"] == TERMINAL_SAFE_BUT_PASSIVE
    assert payload["summary"]["ready_for_objective_conversion_review"] is True
    assert payload["summary"]["ready_for_m2_or_m3"] is False
    assert payload["summary"]["support"] == 0
    record = payload["objective_conversion_frontier_records"][0]
    assert record["blocked_capability"] == OBJECTIVE_CONVERSION_AFTER_SAFE_STOP
    assert record["ready_for_objective_conversion_review"] is True
    assert record["ready_for_m2_or_m3"] is False
    assert record["a33_ready"] is False
    assert record["support"] == 0
    assert record["truth_status"] == TRUTH_STATUS
    assert "post_safe_stop_objective_conversion" in record["desired_hypothesis_families"]


def test_objective_conversion_frontier_records_noops_when_completion_exists(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "p3g1.json", _p3g1_payload(completion=True))

    payload = run_objective_conversion_frontier_records(
        p3_objective_aware_consolidation_path=source,
    )

    assert payload["objective_conversion_frontier_records"] == []
    assert payload["summary"]["frontier_records"] == 0
    assert payload["summary"]["ready_for_objective_conversion_review"] is False
    assert payload["summary"]["support"] == 0


@pytest.mark.parametrize(
    ("support", "verdict"),
    [(1, False), (0, True)],
)
def test_objective_conversion_frontier_records_rejects_bad_source(
    tmp_path: Path,
    support: int,
    verdict: bool,
) -> None:
    source = _write_json(
        tmp_path / "p3g1.json",
        _p3g1_payload(support=support, verdict=verdict),
    )

    with pytest.raises(ValueError):
        run_objective_conversion_frontier_records(
            p3_objective_aware_consolidation_path=source,
        )


def test_validate_objective_conversion_record_rejects_handoff_or_support() -> None:
    record = build_objective_conversion_frontier_record(
        _p3g1_payload()["summary"],
        source_path="p3g1.json",
    )
    record["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        validate_objective_conversion_frontier_record(record)

    record["support"] = 0
    record["ready_for_m2_or_m3"] = True
    with pytest.raises(ValueError, match="not direct M2/M3 handoffs"):
        validate_objective_conversion_frontier_record(record)
