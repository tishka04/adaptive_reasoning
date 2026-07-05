import copy
import json

from theory.a32.patch_similarity_revision_intake import (
    ACCEPTED_FOR_SCIENTIFIC_REVISION,
    REJECTED_FROM_INTAKE,
    build_a32_patch_similarity_revision_intake_candidates,
    rejection_reason,
    run_a32_patch_similarity_revision_intake,
)


def _queue_item(**overrides):
    item = {
        "queue_item_id": "m3_21::bp35::ACTION4_ACTION6::local_patch_transformability",
        "source_generativity_consolidation_id": (
            "m3_20::bp35::ACTION4_ACTION6::patch_similarity_generativity"
        ),
        "game_id": "bp35-0a0ad940",
        "hypothesis_key": (
            "patch_similarity_rule::bp35-0a0ad940::ACTION4_ACTION6::"
            "local_patch_transformability"
        ),
        "description": (
            "ACTION4 after ACTION6/ACTION3 may open patch-similar ACTION6 "
            "affordances selected by local_patch_transformability."
        ),
        "candidate_rule_family": "local_patch_transformability",
        "candidate_mechanic": "repositioning_opens_patch_similar_action6_affordances",
        "candidate_generativity": (
            "SUPPORTED_BY_SEQUENTIAL_PATCH_SIMILAR_EXPANSION_CANDIDATE_ONLY"
        ),
        "context_replay": ["ACTION6", "ACTION3", "ACTION4"],
        "context_replay_args": [{"x": 18, "y": 0}, {}, {}],
        "target_action": "ACTION6",
        "successful_args_total": [
            {"x": 12, "y": 0},
            {"x": 24, "y": 0},
            {"x": 30, "y": 12},
            {"x": 36, "y": 12},
            {"x": 42, "y": 12},
            {"x": 48, "y": 12},
        ],
        "failed_args": [{"x": 30, "y": 0}],
        "success_metrics": [
            "local_patch_before_after",
            "object_positions_before_after",
        ],
        "diagnostic_metrics": [
            "changed_pixels",
            "contact_graph_before_after",
        ],
        "changed_pixels_role": "effect_radar_not_success_metric",
        "ready_for_a32_revision": True,
        "ready_for_a32_revision_is_not_verdict": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "diagnostic_contradictions_counted_as_refutation": False,
        "evidence_summary": {
            "controlled_experiments_run": 8,
            "successful_args_total_count": 6,
            "failed_args_count": 1,
            "new_expansion_successes_count": 2,
            "source_success_metric_support_events": 4,
            "source_success_metric_contradiction_events": 0,
            "source_diagnostic_contradiction_events": 2,
            "diagnostic_contradictions_counted_as_refutation": False,
            "success_metrics": [
                "local_patch_before_after",
                "object_positions_before_after",
            ],
            "diagnostic_metrics": [
                "changed_pixels",
                "contact_graph_before_after",
            ],
            "changed_pixels_role": "effect_radar_not_success_metric",
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
            "wrong_confirmations": 0,
        },
    }
    item.update(overrides)
    return item


def _payload(item=None, **summary_overrides):
    summary = {
        "queue_items": 1,
        "a33_write_performed": False,
        "support": 0,
        "wrong_confirmations": 0,
    }
    summary.update(summary_overrides)
    return {
        "summary": summary,
        "queue_items": [item or _queue_item()],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def test_a32_patch_similarity_intake_accepts_unresolved_queue_item():
    accepted, rejected = build_a32_patch_similarity_revision_intake_candidates(
        _payload()
    )

    assert rejected == ()
    candidate = accepted[0]
    record = candidate.to_record()
    assert candidate.intake_status == ACCEPTED_FOR_SCIENTIFIC_REVISION
    assert candidate.requested_next_step == "A15_A31_PATCH_SIMILARITY_REVIEW_REQUIRED"
    assert candidate.status == "UNRESOLVED"
    assert candidate.revision_status == "CANDIDATE_ONLY"
    assert candidate.support == 0
    assert candidate.truth_status == "NOT_EVALUATED_BY_A32_INTAKE"
    assert candidate.revision_performed is False
    assert candidate.wrong_confirmations == 0
    assert candidate.changed_pixels_role == "effect_radar_not_success_metric"
    assert candidate.diagnostic_contradictions_counted_as_refutation is False
    assert record.status.value == "unresolved"
    assert record.support == 0
    assert record.contradictions == 0
    assert record.experiments_spent == 8


def test_a32_patch_similarity_intake_run_outputs_unresolved_record(tmp_path):
    path = tmp_path / "patch_similarity_a32_revision_queue.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    payload = run_a32_patch_similarity_revision_intake(queue_path=path)

    summary = payload["summary"]
    assert summary["queue_items_consumed"] == 1
    assert summary["accepted_for_scientific_revision"] == 1
    assert summary["rejected_from_intake"] == 0
    assert summary["candidate_records"] == 1
    assert summary["successful_args_total_count"] == 6
    assert summary["failed_args_count"] == 1
    assert summary["source_success_metric_contradiction_events"] == 0
    assert summary["source_diagnostic_contradiction_events"] == 2
    assert summary["diagnostic_contradictions_counted_as_refutation"] is False
    assert summary["changed_pixels_kept_diagnostic"] is True
    assert summary["hypotheses_confirmed"] == 0
    assert summary["hypotheses_refuted"] == 0
    assert summary["revision_performed"] is False
    assert summary["support"] == 0
    assert summary["wrong_confirmations"] == 0
    assert payload["candidate_records"][0]["status"] == "unresolved"
    assert payload["candidate_records"][0]["support"] == 0
    assert payload["accepted_candidates"][0]["intake_status"] == (
        ACCEPTED_FOR_SCIENTIFIC_REVISION
    )


def test_a32_patch_similarity_intake_rejects_verdict_smuggling():
    supported = _queue_item(support=1)
    contradicted = _queue_item(
        evidence_summary={
            **_queue_item()["evidence_summary"],
            "source_success_metric_contradiction_events": 1,
        }
    )
    a33_attempt = _payload(a33_write_performed=True)

    assert rejection_reason(supported, queue_payload=_payload()) == (
        "support_must_remain_zero"
    )
    assert rejection_reason(contradicted, queue_payload=_payload()) == (
        "success_metric_contradiction_events_present"
    )
    assert rejection_reason(_queue_item(), queue_payload=a33_attempt) == (
        "a33_write_attempted"
    )

    accepted, rejected = build_a32_patch_similarity_revision_intake_candidates(
        {"queue_items": [supported, contradicted], "summary": {}}
    )
    assert accepted == ()
    assert {row["intake_status"] for row in rejected} == {REJECTED_FROM_INTAKE}


def test_a32_patch_similarity_intake_diagnostic_contradiction_does_not_reject():
    item = _queue_item(
        evidence_summary={
            **_queue_item()["evidence_summary"],
            "source_diagnostic_contradiction_events": 99,
        }
    )
    accepted, rejected = build_a32_patch_similarity_revision_intake_candidates(
        _payload(item)
    )

    assert rejected == ()
    assert len(accepted) == 1
    assert accepted[0].evidence_summary["source_diagnostic_contradiction_events"] == 99
    assert accepted[0].diagnostic_contradictions_counted_as_refutation is False


def test_a32_patch_similarity_intake_rejects_diagnostic_refutation_interpretation():
    item = _queue_item(diagnostic_contradictions_counted_as_refutation=True)
    assert rejection_reason(item, queue_payload=_payload()) == (
        "diagnostic_contradiction_interpreted_as_refutation"
    )


def test_a32_patch_similarity_intake_does_not_mutate_input(tmp_path):
    source = _payload()
    before = copy.deepcopy(source)
    path = tmp_path / "patch_similarity_a32_revision_queue.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    run_a32_patch_similarity_revision_intake(queue_path=path)

    assert json.loads(path.read_text(encoding="utf-8")) == before
