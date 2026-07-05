from theory.epistemic_metrics import HypothesisStatus
from theory.m1.scientific_integration_pretest import (
    ledger_entry_from_revision_candidate,
    score_entries_by_game,
    summarize_scientific_integration,
)


def _revision():
    return {
        "revision_candidate_id": "revision::best",
        "a15_a31_ready": True,
        "prediction": {
            "game_id": "bp35",
            "action": "ACTION6",
            "mechanic_family": "position_effect_candidate",
            "predicted_metric": "local_patch_before_after",
        },
        "a15_a31_revision_proposal": {
            "key": "mechanic_prediction::bp35::ACTION6::position_effect_candidate::local_patch_before_after",
            "description": "ACTION6 position_effect_candidate via local_patch_before_after",
            "proposed_status": "UNRESOLVED",
            "support": 0,
            "contradictions": 0,
            "experiments_spent": 0,
            "controlled_test_required": True,
            "observation_counted_as_confirmation": False,
        },
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def test_revision_candidate_enters_ledger_as_unresolved_record():
    entry = ledger_entry_from_revision_candidate(_revision())
    record = entry.to_record()

    assert entry.entered_scientific_ledger is True
    assert entry.controlled_test_required is True
    assert entry.observation_counted_as_confirmation is False
    assert record.status == HypothesisStatus.UNRESOLVED
    assert record.support == 0
    assert record.contradictions == 0
    assert record.experiments_spent == 0


def test_scientific_score_has_no_wrong_confirmation_for_unresolved_entry():
    entry = ledger_entry_from_revision_candidate(_revision())
    scores = score_entries_by_game([entry])

    assert scores["bp35"].wrong_confirmations == 0
    assert scores["bp35"].hypotheses_confirmed == 0
    assert scores["bp35"].experiment_actions == 0


def test_summarize_scientific_integration_reports_guardrails():
    entry = ledger_entry_from_revision_candidate(_revision())
    summary = summarize_scientific_integration(
        [entry],
        score_entries_by_game([entry]),
    )

    assert summary["entered_scientific_ledger"] == 1
    assert summary["unresolved_records"] == 1
    assert summary["confirmed_records"] == 0
    assert summary["support_total"] == 0
    assert summary["experiments_spent_total"] == 0
    assert summary["controlled_tests_required"] == 1
    assert summary["observation_counted_as_confirmation"] is False
    assert summary["revision_performed"] is False
    assert summary["wrong_confirmations"] == 0
