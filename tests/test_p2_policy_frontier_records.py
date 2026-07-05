import json

import pytest

from theory.p2 import policy_frontier_records as p2


def _p1_trigger_payload(*, real_frontier=False, synthetic_frontier=True):
    real_record = (
        {
            "frontier_id": "p1::bp35::conditional_movement_refresh::exhausted_refresh",
            "source_label": "p1_10_conditional_movement_refresh_matrix",
            "game_id": "bp35-0a0ad940",
            "frontier_context_id": "after_soft_stale_and_conditional_movement_refresh_exhaustion",
            "frontier_reason": "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH",
            "exhausted_policy": "conditional_movement_refresh",
            "refresh_candidates": ["ACTION3", "ACTION4", "ACTION1", "ACTION2"],
            "ready_for_p2_frontier_extraction": True,
            "policy_result_counted_as_scientific_verdict": False,
            "frontier_trigger_counted_as_confirmation": False,
            "synthetic_fixture": False,
            "synthetic_fixture_not_for_scientific_handoff": False,
            "status": "UNRESOLVED",
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
            "revision_performed": False,
            "wrong_confirmations": 0,
        }
        if real_frontier
        else None
    )
    synthetic_record = (
        {
            "frontier_id": "p1::bp35::conditional_movement_refresh::exhausted_refresh::synthetic_fixture",
            "source_label": "synthetic_saturation_fixture",
            "game_id": "bp35-0a0ad940",
            "frontier_context_id": "after_soft_stale_and_conditional_movement_refresh_exhaustion",
            "frontier_reason": "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH",
            "exhausted_policy": "conditional_movement_refresh",
            "refresh_candidates": ["ACTION3", "ACTION4", "ACTION1", "ACTION2"],
            "ready_for_p2_frontier_extraction": True,
            "policy_result_counted_as_scientific_verdict": False,
            "frontier_trigger_counted_as_confirmation": False,
            "synthetic_fixture": True,
            "synthetic_fixture_not_for_scientific_handoff": True,
            "status": "UNRESOLVED",
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
            "revision_performed": False,
            "wrong_confirmations": 0,
        }
        if synthetic_frontier
        else None
    )
    return {
        "config": {
            "schema_version": "p1.bp35_policy_frontier_trigger.v1",
            "inputs_read": ["P1.10"],
        },
        "real_rollout_evaluation": {
            "frontier_triggered": bool(real_frontier),
            "frontier_reason": (
                "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH"
                if real_frontier
                else "NO_MOVEMENT_REFRESH_SATURATION_OBSERVED"
            ),
            "frontier_record": real_record,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "synthetic_saturation_fixture": {
            "frontier_triggered": bool(synthetic_frontier),
            "frontier_reason": (
                "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH"
                if synthetic_frontier
                else "NO_SYNTHETIC_FRONTIER"
            ),
            "frontier_record": synthetic_record,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "summary": {
            "real_frontier_triggered": bool(real_frontier),
            "synthetic_fixture_frontier_triggered": bool(synthetic_frontier),
            "policy_result_counted_as_scientific_verdict": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_policy_frontier_records_keeps_real_empty_and_synthetic_schema_only(tmp_path):
    path = tmp_path / "p1_trigger.json"
    path.write_text(json.dumps(_p1_trigger_payload()), encoding="utf-8")

    payload = p2.run_policy_frontier_records(p1_frontier_trigger_path=path)

    assert payload["real_frontier_records"] == []
    assert len(payload["synthetic_frontier_records"]) == 1
    synthetic = payload["synthetic_frontier_records"][0]
    assert synthetic["ready_for_m2_or_m3"] is False
    assert synthetic["synthetic_fixture_not_for_scientific_handoff"] is True
    assert payload["summary"]["real_handoffs_produced"] == 0
    assert payload["summary"]["synthetic_records_for_schema_validation_only"] == 1
    assert payload["summary"]["synthetic_records_counted_as_handoff"] is False
    assert payload["summary"]["support"] == 0


def test_policy_frontier_records_allows_real_frontier_handoff_when_present(tmp_path):
    path = tmp_path / "p1_trigger.json"
    path.write_text(
        json.dumps(_p1_trigger_payload(real_frontier=True, synthetic_frontier=False)),
        encoding="utf-8",
    )

    payload = p2.run_policy_frontier_records(p1_frontier_trigger_path=path)

    assert len(payload["real_frontier_records"]) == 1
    assert payload["real_frontier_records"][0]["ready_for_m2_or_m3"] is True
    assert payload["summary"]["real_handoffs_produced"] == 1
    assert payload["summary"]["synthetic_records_for_schema_validation_only"] == 0
    assert payload["summary"]["support"] == 0


def test_policy_frontier_record_rejects_synthetic_handoff():
    record = p2.PolicyFrontierRecord(
        frontier_id="synthetic",
        source="unit",
        game_id="bp35-0a0ad940",
        frontier_context_id="ctx",
        frontier_reason="reason",
        exhausted_policy="conditional_movement_refresh",
        refresh_candidates=("ACTION3",),
        ready_for_m2_or_m3=True,
        synthetic_fixture=True,
        synthetic_fixture_not_for_scientific_handoff=True,
    )

    with pytest.raises(ValueError, match="synthetic fixture cannot be ready"):
        p2.validate_policy_frontier_record(record)


def test_policy_frontier_records_rejects_support_in_p1_summary(tmp_path):
    payload = _p1_trigger_payload()
    payload["summary"]["support"] = 1
    path = tmp_path / "p1_trigger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="support must remain 0"):
        p2.run_policy_frontier_records(p1_frontier_trigger_path=path)


def test_policy_frontier_record_rejects_verdict_flags():
    record = p2.PolicyFrontierRecord(
        frontier_id="real",
        source="unit",
        game_id="bp35-0a0ad940",
        frontier_context_id="ctx",
        frontier_reason="reason",
        exhausted_policy="conditional_movement_refresh",
        refresh_candidates=("ACTION3",),
        ready_for_m2_or_m3=True,
        synthetic_fixture=False,
        synthetic_fixture_not_for_scientific_handoff=False,
        policy_result_counted_as_scientific_verdict=True,
    )

    with pytest.raises(ValueError, match="scientific verdict"):
        p2.validate_policy_frontier_record(record)
