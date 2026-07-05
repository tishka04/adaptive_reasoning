import json

from theory.a32.revision_decisions import (
    FOLLOWUP_REQUIRED,
    REVISION_ACCEPTED_AS_CONFIRMED,
    REVISION_REJECTED_AS_INSUFFICIENT,
    build_a32_revision_decisions,
    decision_label,
    decision_reasons,
    run_a32_revision_decision_consumer,
)


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
            _control("ACTION3", independent=1, reused=0),
            _control("ACTION4", independent=1, reused=0),
            _control(
                "ACTION3",
                independent=0,
                reused=1,
                reuse_reason="no_unused_distinct_control_available",
            ),
        ],
    }
    item.update(overrides)
    return item


def _control(control, *, independent, reused, reuse_reason=""):
    return {
        "target_action": "ACTION6",
        "control_action": control,
        "predicted_metric": "local_patch_before_after",
        "baseline_signal": 0.0,
        "perturbation_signal": 1.0,
        "effect_size": 1.0,
        "direction": "support",
        "support_events": 1,
        "independent_support_events": independent,
        "reused_control_support_events": reused,
        "contradiction_events": 0,
        "control_reuse_reason": reuse_reason,
        "status": "UNRESOLVED",
    }


def test_a32_accepts_confirmed_revision_only_after_queue_guardrails():
    decisions = build_a32_revision_decisions({"queue_items": [_queue_item()]})
    decision = decisions[0].to_dict()

    assert decision["decision"] == REVISION_ACCEPTED_AS_CONFIRMED
    assert decision["reasons"] == ["a32_revision_criteria_satisfied"]
    assert decision["input_record"]["status"] == "unresolved"
    assert decision["input_record"]["support"] == 0
    assert decision["decision_record"]["status"] == "confirmed"
    assert decision["decision_record"]["support"] == 2
    assert decision["decision_record"]["experiments_spent"] == 3
    assert decision["revision_performed"] is True
    assert decision["wrong_confirmations"] == 0


def test_a32_requires_two_distinct_independent_controls():
    item = _queue_item(
        control_evidence=[
            _control("ACTION3", independent=1, reused=0),
            _control("ACTION3", independent=1, reused=0),
            _control("ACTION3", independent=0, reused=1),
        ],
    )
    reasons = decision_reasons(
        item,
        evidence=item["evidence_summary"],
        controls=item["control_evidence"],
    )

    assert "insufficient_distinct_independent_controls" in reasons
    assert decision_label(reasons) == FOLLOWUP_REQUIRED


def test_a32_rejects_structurally_invalid_queue_items_as_insufficient():
    item = _queue_item(status="CONFIRMED", support=1)
    decisions = build_a32_revision_decisions({"queue_items": [item]})
    decision = decisions[0].to_dict()

    assert decision["decision"] == REVISION_REJECTED_AS_INSUFFICIENT
    assert "queue_status_not_unresolved" in decision["reasons"]
    assert "queue_support_not_zero_before_revision" in decision["reasons"]
    assert decision["decision_record"]["status"] == "unresolved"
    assert decision["decision_record"]["support"] == 0
    assert decision["revision_performed"] is False


def test_a32_followup_required_for_contradiction_or_weak_evidence():
    contradicted = _queue_item(
        evidence_summary={
            "controlled_experiments_run": 3,
            "support_events": 3,
            "independent_support_events": 2,
            "reused_control_support_events": 1,
            "contradiction_events": 1,
        }
    )
    weak = _queue_item(
        evidence_summary={
            "controlled_experiments_run": 2,
            "support_events": 2,
            "independent_support_events": 1,
            "reused_control_support_events": 1,
            "contradiction_events": 0,
        }
    )
    decisions = build_a32_revision_decisions({"queue_items": [contradicted, weak]})

    assert [decision.decision for decision in decisions] == [
        FOLLOWUP_REQUIRED,
        FOLLOWUP_REQUIRED,
    ]


def test_a32_run_writes_decision_artifact_without_mutating_m3(tmp_path):
    queue_path = tmp_path / "a15_revision_queue.json"
    queue_path.write_text(
        json.dumps({"queue_items": [_queue_item()]}),
        encoding="utf-8",
    )

    payload = run_a32_revision_decision_consumer(queue_path=queue_path)

    assert payload["summary"]["queue_items_consumed"] == 1
    assert payload["summary"]["revision_accepted_as_confirmed"] == 1
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["m3_artifacts_mutated"] is False
    assert payload["decision_records"][0]["status"] == "confirmed"
    assert payload["input_records"][0]["status"] == "unresolved"
