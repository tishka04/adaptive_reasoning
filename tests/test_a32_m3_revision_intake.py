import json

from theory.a32_m3_revision_intake import (
    ACCEPTED_CANDIDATE,
    REJECTED_CANDIDATE,
    build_a32_revision_intake_candidates,
    candidate_record_dict,
    rejection_reason,
    run_a32_m3_revision_intake,
)
from theory.epistemic_metrics import HypothesisStatus


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


def _queue_item(**overrides):
    item = {
        "queue_item_id": "m3_6::0001::" + KEY,
        "game_id": "bp35-0a0ad940",
        "hypothesis_key": KEY,
        "description": "ACTION6 position_effect_candidate via local_patch_before_after",
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "contradictions": 0,
        "controlled_test_required": True,
        "evidence_summary": {
            "controlled_experiments_run": 3,
            "support_events": 3,
            "independent_support_events": 2,
            "reused_control_support_events": 1,
            "contradiction_events": 0,
        },
        "control_evidence": [
            {
                "target_action": "ACTION6",
                "control_action": "ACTION3",
                "support_events": 1,
                "independent_support_events": 1,
                "reused_control_support_events": 0,
                "contradiction_events": 0,
                "status": "UNRESOLVED",
            }
        ],
    }
    item.update(overrides)
    return item


def test_a32_accepts_candidate_only_queue_item_as_unresolved_record():
    accepted, rejected = build_a32_revision_intake_candidates(
        {"queue_items": [_queue_item()]}
    )
    candidate = accepted[0]
    record = candidate.to_record()
    record_payload = candidate_record_dict(record)

    assert rejected == ()
    assert candidate.intake_status == ACCEPTED_CANDIDATE
    assert candidate.requested_next_step == "A15_A31_REVISION_DECISION_REQUIRED"
    assert candidate.status == "UNRESOLVED"
    assert candidate.revision_status == "CANDIDATE_ONLY"
    assert candidate.support == 0
    assert candidate.contradictions == 0
    assert candidate.controlled_test_required is True
    assert record.status == HypothesisStatus.UNRESOLVED
    assert record.support == 0
    assert record.contradictions == 0
    assert record_payload["status"] == "unresolved"


def test_a32_rejects_items_that_would_smuggle_verdict_or_weak_evidence():
    confirmed = _queue_item(status="CONFIRMED")
    supported = _queue_item(support=1)
    weak = _queue_item(
        evidence_summary={
            "controlled_experiments_run": 3,
            "support_events": 3,
            "independent_support_events": 1,
            "reused_control_support_events": 2,
            "contradiction_events": 0,
        }
    )

    assert rejection_reason(confirmed) == "status_not_unresolved"
    assert rejection_reason(supported) == "support_must_remain_zero"
    assert rejection_reason(weak) == "insufficient_independent_support_events"

    accepted, rejected = build_a32_revision_intake_candidates(
        {"queue_items": [confirmed, supported, weak]}
    )

    assert accepted == ()
    assert {row["intake_status"] for row in rejected} == {REJECTED_CANDIDATE}


def test_a32_run_scores_without_confirmation_or_revision(tmp_path):
    queue_path = tmp_path / "a15_revision_queue.json"
    queue_path.write_text(
        json.dumps({"queue_items": [_queue_item()]}),
        encoding="utf-8",
    )

    payload = run_a32_m3_revision_intake(queue_path=queue_path)

    assert payload["summary"]["accepted_candidates"] == 1
    assert payload["summary"]["candidate_records"] == 1
    assert payload["summary"]["revision_status"] == "CANDIDATE_ONLY"
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["revision_performed"] is False
    assert payload["summary"]["hypotheses_confirmed"] == 0
    assert payload["summary"]["hypotheses_refuted"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["candidate_records"][0]["status"] == "unresolved"
    assert payload["candidate_records"][0]["support"] == 0
