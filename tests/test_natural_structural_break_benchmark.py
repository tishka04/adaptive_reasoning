from __future__ import annotations

from theory.natural_structural_break_benchmark import (
    summarize_natural_structural_break_protocol,
)


def _row(**overrides):
    row = {
        "levels_completed_delta": 1,
        "max_level_reached": 1,
        "wins": 0,
        "controller_errors": 0,
        "structural_breaks_detected": 0,
        "structural_old_theory_suspensions": 0,
        "structural_revision_hypotheses_generated": 0,
        "structural_revision_actions": 0,
        "structural_revision_confirmations": 0,
        "structural_revision_refutations": 0,
        "structural_arbitration_decisions": 0,
        "structural_family_transfers": 0,
        "structural_family_transfer_actions": 0,
        "structural_theory_switches": 0,
        "structural_theory_reactivations": 0,
    }
    row.update(overrides)
    return row


def _payload(*, enabled: bool, row):
    return {
        "held_out_games": ["natural-game"],
        "seeds": [0],
        "action_budget_per_reset": 80,
        "resets_per_game_seed_arm": 4,
        "paired_protocol": {
            "protocol_gate_passed": True,
            "terminal_relational_stencil_relation_permuted_in_unified": False,
            "online_structural_break_detection_enabled_in_unified": enabled,
        },
        "pairs": [
            {
                "game_id": "natural-game",
                "unified": row,
            },
        ],
    }


def test_natural_break_protocol_accepts_only_post_evaluation_causal_gain():
    active = _payload(
        enabled=True,
        row=_row(
            levels_completed_delta=6,
            max_level_reached=5,
            structural_breaks_detected=1,
            structural_old_theory_suspensions=1,
            structural_revision_hypotheses_generated=3,
            structural_revision_actions=7,
            structural_revision_confirmations=1,
            structural_arbitration_decisions=4,
            structural_family_transfers=1,
            structural_family_transfer_actions=2,
            structural_theory_switches=2,
            structural_theory_reactivations=1,
        ),
    )
    ablated = _payload(
        enabled=False,
        row=_row(
            levels_completed_delta=3,
            max_level_reached=3,
        ),
    )

    report = summarize_natural_structural_break_protocol(
        active,
        ablated,
    )

    assert report["protocol"]["protocol_gate_passed"] is True
    assert report["protocol"]["relation_permutation_used"] is False
    assert report["natural_break_candidates"] == 1
    assert report["causal_natural_revisions"] == 1
    assert report["games"][0]["natural_revision_gate_passed"] is True
    assert (
        report["games"][0]["strong_natural_revision_gate_passed"]
        is True
    )


def test_break_without_terminal_revision_is_reported_but_not_claimed():
    active = _payload(
        enabled=True,
        row=_row(
            structural_breaks_detected=1,
            structural_revision_hypotheses_generated=3,
        ),
    )
    ablated = _payload(enabled=False, row=_row())

    report = summarize_natural_structural_break_protocol(
        active,
        ablated,
    )

    assert report["natural_break_candidates"] == 1
    assert report["causal_natural_revisions"] == 0
    assert report["any_natural_revision_gate_passed"] is False
