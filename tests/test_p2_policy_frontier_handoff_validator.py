import json

import pytest

from theory.p2 import policy_frontier_handoff_validator as validator
from theory.p2.policy_frontier_records import TRUTH_STATUS


def _record(*, synthetic=False, ready=True, support=0, verdict=False):
    return {
        "frontier_id": (
            "p1::bp35::conditional_movement_refresh::exhausted_refresh"
            + ("::synthetic_fixture" if synthetic else "")
        ),
        "source": "P1.11.synthetic_fixture" if synthetic else "P1.11",
        "game_id": "bp35-0a0ad940",
        "frontier_context_id": "after_soft_stale_and_conditional_movement_refresh_exhaustion",
        "frontier_reason": "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH",
        "exhausted_policy": "conditional_movement_refresh",
        "refresh_candidates": ["ACTION3", "ACTION4", "ACTION1", "ACTION2"],
        "ready_for_m2_or_m3": ready,
        "synthetic_fixture": synthetic,
        "synthetic_fixture_not_for_scientific_handoff": synthetic,
        "policy_result_counted_as_scientific_verdict": verdict,
        "frontier_trigger_counted_as_confirmation": False,
        "status": "UNRESOLVED",
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _records_payload(*, real_records=None, synthetic_records=None):
    real_records = list(real_records or [])
    synthetic_records = list(synthetic_records or [])
    return {
        "config": {
            "schema_version": "p2.policy_frontier_record.v1",
            "source": "P1.11",
        },
        "real_frontier_records": real_records,
        "synthetic_frontier_records": synthetic_records,
        "summary": {
            "real_frontier_records": len(real_records),
            "synthetic_frontier_records": len(synthetic_records),
            "real_handoffs_produced": len(
                [row for row in real_records if row.get("ready_for_m2_or_m3")]
            ),
            "synthetic_records_counted_as_handoff": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "frontier_records_counted_as_confirmation": False,
    }


def test_handoff_validator_rejects_current_synthetic_schema_record(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(_records_payload(synthetic_records=[_record(synthetic=True, ready=False)])),
        encoding="utf-8",
    )

    payload = validator.run_policy_frontier_handoff_validation(
        policy_frontier_records_path=path,
    )

    assert payload["summary"]["records_seen"] == 1
    assert payload["summary"]["real_records_seen"] == 0
    assert payload["summary"]["synthetic_records_seen"] == 1
    assert payload["summary"]["handoffs_accepted"] == 0
    assert payload["summary"]["handoffs_rejected"] == 1
    assert payload["summary"]["rejection_reasons"] == [
        "NOT_READY_FOR_M2_OR_M3",
        "SYNTHETIC_FIXTURE_NOT_FOR_SCIENTIFIC_HANDOFF",
    ]
    assert payload["summary"]["a40_write_performed"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["support"] == 0


def test_handoff_validator_accepts_real_ready_record(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(_records_payload(real_records=[_record(synthetic=False, ready=True)])),
        encoding="utf-8",
    )

    payload = validator.run_policy_frontier_handoff_validation(
        policy_frontier_records_path=path,
    )

    assert payload["summary"]["records_seen"] == 1
    assert payload["summary"]["real_records_seen"] == 1
    assert payload["summary"]["synthetic_records_seen"] == 0
    assert payload["summary"]["handoffs_accepted"] == 1
    assert payload["summary"]["handoffs_rejected"] == 0
    assert payload["accepted_handoffs"][0]["handoff_accepted"] is True
    assert payload["accepted_handoffs"][0]["support"] == 0


def test_handoff_validator_rejects_real_record_not_ready():
    result = validator.evaluate_record_for_handoff(_record(synthetic=False, ready=False))

    assert result["handoff_accepted"] is False
    assert result["rejection_reasons"] == ["NOT_READY_FOR_M2_OR_M3"]
    assert result["a40_write_performed"] is False
    assert result["support"] == 0


def test_handoff_validator_rejects_support_or_verdict_flags():
    support_result = validator.evaluate_record_for_handoff(
        _record(synthetic=False, ready=True, support=1)
    )
    verdict_result = validator.evaluate_record_for_handoff(
        _record(synthetic=False, ready=True, verdict=True)
    )

    assert "SUPPORT_MUST_REMAIN_ZERO" in support_result["rejection_reasons"]
    assert "POLICY_RESULT_COUNTED_AS_VERDICT" in verdict_result["rejection_reasons"]
    assert support_result["handoff_accepted"] is False
    assert verdict_result["handoff_accepted"] is False


def test_handoff_validator_rejects_source_summary_that_counts_synthetic_handoff(tmp_path):
    payload = _records_payload(synthetic_records=[_record(synthetic=True, ready=False)])
    payload["summary"]["synthetic_records_counted_as_handoff"] = True
    path = tmp_path / "records.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="synthetic records cannot"):
        validator.run_policy_frontier_handoff_validation(
            policy_frontier_records_path=path,
        )
