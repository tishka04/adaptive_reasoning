import json

from theory.epistemic_metrics import HypothesisStatus
from theory.m3.a15_revision_queue import (
    build_a15_revision_queue_items,
    candidate_record_dict,
    run_a15_revision_queue_generation,
    summarize_a15_revision_queue,
)


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


def _planning_payload(*, ready=True, contradiction_events=0):
    support_events = 3 if ready else 2
    independent = 2 if ready else 1
    return {
        "propose_ready_for_A15_revision": ready,
        "planning_state": {
            "ledger_entries": [
                {
                    "key": KEY,
                    "game_id": "bp35-0a0ad940",
                    "description": (
                        "ACTION6 position_effect_candidate via "
                        "local_patch_before_after"
                    ),
                    "status": "unresolved",
                    "support": 0,
                    "contradictions": 0,
                    "controlled_test_required": True,
                }
            ]
        },
        "updated_ledger_entries": [
            {
                "key": KEY,
                "game_id": "bp35-0a0ad940",
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "contradictions": 0,
                "controlled_experiments_run": 3,
                "support_events": support_events,
                "independent_support_events": independent,
                "reused_control_support_events": support_events - independent,
                "contradiction_events": contradiction_events,
                "controlled_test_required": True,
                "wrong_confirmations": 0,
            }
        ],
        "controlled_experiments": [
            _experiment("ACTION3", independent=1, reused=0),
            _experiment("ACTION4", independent=1, reused=0),
            _experiment(
                "ACTION3",
                independent=0,
                reused=1,
                reuse_reason="no_unused_distinct_control_available",
            ),
        ],
    }


def _experiment(control, *, independent, reused, reuse_reason=""):
    row = {
        "hypothesis_key": KEY,
        "game_id": "bp35-0a0ad940",
        "target_action": "ACTION6",
        "control_action": control,
        "predicted_metric": "local_patch_before_after",
        "delta": {
            "baseline_signal": 0.0,
            "perturbation_signal": 1.0,
            "effect_size": 1.0,
            "direction": "support",
        },
        "support_events": 1,
        "independent_support_events": independent,
        "reused_control_support_events": reused,
        "contradiction_events": 0,
        "status": "UNRESOLVED",
    }
    if reuse_reason:
        row["control_reuse_reason"] = reuse_reason
    return row


def test_a15_revision_queue_filters_ready_candidate_and_preserves_guardrails():
    items = build_a15_revision_queue_items(
        _planning_payload(),
        source_planning_artifact="diagnostics/m3/scientific_planning_bp35.json",
    )
    item = items[0]
    record = item.to_record()
    record_payload = candidate_record_dict(item)

    assert len(items) == 1
    assert item.status == "UNRESOLVED"
    assert item.revision_status == "CANDIDATE_ONLY"
    assert item.support == 0
    assert item.contradictions == 0
    assert item.controlled_test_required is True
    assert item.evidence_summary["support_events"] == 3
    assert item.evidence_summary["independent_support_events"] == 2
    assert item.evidence_summary["reused_control_support_events"] == 1
    assert item.control_evidence[2]["control_reuse_reason"] == (
        "no_unused_distinct_control_available"
    )
    assert record.status == HypothesisStatus.UNRESOLVED
    assert record.support == 0
    assert record.contradictions == 0
    assert record_payload["status"] == "unresolved"
    assert record_payload["support"] == 0


def test_a15_revision_queue_rejects_not_ready_or_contradicted_candidates():
    not_ready = build_a15_revision_queue_items(_planning_payload(ready=False))
    contradicted = build_a15_revision_queue_items(
        _planning_payload(ready=True, contradiction_events=1)
    )

    assert not_ready == ()
    assert contradicted == ()


def test_queue_generation_scores_candidate_without_confirmation(tmp_path):
    planning_path = tmp_path / "scientific_planning.json"
    planning_path.write_text(json.dumps(_planning_payload()), encoding="utf-8")

    payload = run_a15_revision_queue_generation(planning_result_path=planning_path)
    summary = summarize_a15_revision_queue(
        build_a15_revision_queue_items(_planning_payload()),
        {},
    )

    assert payload["summary"]["queue_items"] == 1
    assert payload["summary"]["revision_status"] == "CANDIDATE_ONLY"
    assert payload["summary"]["support"] == 0
    assert payload["summary"]["wrong_confirmations"] == 0
    assert payload["scores_by_game"]["bp35-0a0ad940"]["hypotheses_confirmed"] == 0
    assert payload["scores_by_game"]["bp35-0a0ad940"]["wrong_confirmations"] == 0
    assert payload["candidate_records"][0]["status"] == "unresolved"
    assert summary["wrong_confirmations"] == 0
