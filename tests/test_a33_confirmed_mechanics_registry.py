import json

from theory.a32.revision_decisions import (
    FOLLOWUP_REQUIRED,
    REVISION_ACCEPTED_AS_CONFIRMED,
)
from theory.a33.confirmed_mechanics_registry import (
    build_confirmed_mechanics_registry,
    mechanic_prediction_spec,
    run_confirmed_mechanics_registry_generation,
)


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


def _decision(*, decision=REVISION_ACCEPTED_AS_CONFIRMED):
    return {
        "queue_item_id": "m3_6::0001::" + KEY,
        "game_id": "bp35-0a0ad940",
        "key": KEY,
        "description": "ACTION6 position_effect_candidate via local_patch_before_after",
        "decision": decision,
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
            _control("ACTION3", independent=0, reused=1),
        ],
        "decision_record": {
            "key": KEY,
            "description": "ACTION6 position_effect_candidate via local_patch_before_after",
            "status": "confirmed" if decision == REVISION_ACCEPTED_AS_CONFIRMED else "unresolved",
            "support": 2 if decision == REVISION_ACCEPTED_AS_CONFIRMED else 0,
            "contradictions": 0,
            "experiments_spent": 3,
        },
    }


def _control(control, *, independent, reused):
    return {
        "target_action": "ACTION6",
        "control_action": control,
        "predicted_metric": "local_patch_before_after",
        "support_events": 1,
        "independent_support_events": independent,
        "reused_control_support_events": reused,
        "contradiction_events": 0,
    }


def test_mechanic_prediction_spec_parses_registry_key():
    assert mechanic_prediction_spec(KEY) == {
        "game_id": "bp35-0a0ad940",
        "action": "ACTION6",
        "mechanic_family": "position_effect_candidate",
        "predicted_metric": "local_patch_before_after",
    }


def test_registry_keeps_only_confirmed_a32_decisions_and_independent_support():
    entries = build_confirmed_mechanics_registry(
        {
            "revision_decisions": [
                _decision(),
                _decision(decision=FOLLOWUP_REQUIRED),
            ]
        },
        source_artifact="diagnostics/a32/a15_revision_decisions.json",
    )
    entry = entries[0]

    assert len(entries) == 1
    assert entry.key == KEY
    assert entry.game_id == "bp35-0a0ad940"
    assert entry.action == "ACTION6"
    assert entry.mechanic_family == "position_effect_candidate"
    assert entry.predicted_metric == "local_patch_before_after"
    assert entry.confirmed_support_independent == 2
    assert entry.experiments_spent == 3
    assert entry.control_actions_used == ("ACTION3", "ACTION4")
    assert entry.reused_control_support_excluded == 1
    assert entry.known_scope == "local_context"
    assert entry.trace_support_counted_as_proof is False
    assert entry.prior_counted_as_proof is False
    assert entry.unresolved_candidates_excluded is True
    assert entry.support_reused_as_independent is False


def test_registry_generation_writes_compact_confirmed_memory(tmp_path):
    decisions_path = tmp_path / "a15_revision_decisions.json"
    decisions_path.write_text(
        json.dumps({"revision_decisions": [_decision()]}),
        encoding="utf-8",
    )

    payload = run_confirmed_mechanics_registry_generation(
        decisions_path=decisions_path,
    )

    assert payload["summary"]["confirmed_mechanics"] == 1
    assert payload["summary"]["confirmed_support_independent_total"] == 2
    assert payload["summary"]["reused_control_support_excluded_total"] == 1
    assert payload["summary"]["trace_support_counted_as_proof"] is False
    assert payload["summary"]["prior_counted_as_proof"] is False
    assert payload["confirmed_mechanics"][0]["known_scope"] == "local_context"
