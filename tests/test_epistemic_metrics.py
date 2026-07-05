"""Tests for the epistemic success metric (step C of the consolidation plan)."""

from __future__ import annotations

from theory.epistemic_metrics import (
    GroundTruthFact,
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    mechanic_key,
    normalize_operator_kind,
    score_beliefs,
)
from theory.ar25_oracle import build_ar25_oracle
from theory.correspondence_hypothesis import correspondence_key


def _synthetic_oracle() -> MechanicsOracle:
    oracle = MechanicsOracle(game_id="synthetic")
    # True mechanics (one demonstrated by the human)
    oracle.add(GroundTruthFact(mechanic_key("ACTION1", "global_transform"), True,
                               demonstrated_by_human=True))
    oracle.add(GroundTruthFact(mechanic_key("ACTION6", "noop"), True))
    # False mechanics
    oracle.add(GroundTruthFact(mechanic_key("ACTION2", "move"), False))
    oracle.add(GroundTruthFact(mechanic_key("ACTION3", "move"), False))
    return oracle


def test_normalize_and_key():
    assert normalize_operator_kind("transform_like") == "global_transform"
    assert normalize_operator_kind("no_op") == "noop"
    assert mechanic_key("action5", "move_like") == "action_effect::ACTION5::move"


def test_score_counts_every_outcome():
    oracle = _synthetic_oracle()
    ledger = [
        # correct confirmation of a human-demonstrated true mechanic
        HypothesisRecord(mechanic_key("ACTION1", "global_transform"),
                         status=HypothesisStatus.CONFIRMED, experiments_spent=2),
        # correct refutation of a false mechanic
        HypothesisRecord(mechanic_key("ACTION2", "move"),
                         status=HypothesisStatus.REFUTED, experiments_spent=1),
        # wrong confirmation: confirmed a false mechanic
        HypothesisRecord(mechanic_key("ACTION3", "move"),
                         status=HypothesisStatus.CONFIRMED, experiments_spent=1),
        # missed refutation: a true noop entertained but left unresolved...
        # (ACTION6 noop is TRUE, so this is NOT a missed refutation)
        HypothesisRecord(mechanic_key("ACTION6", "noop"),
                         status=HypothesisStatus.UNRESOLVED, experiments_spent=1),
        # unverifiable: no oracle fact for this key
        HypothesisRecord(mechanic_key("ACTION7", "click"),
                         status=HypothesisStatus.CONFIRMED, experiments_spent=1),
    ]

    score = score_beliefs(ledger, oracle)

    assert score.hypotheses_confirmed == 2          # ACTION1, ACTION3
    assert score.correct_confirmations == 1         # ACTION1
    assert score.wrong_confirmations == 1           # ACTION3
    assert score.hypotheses_refuted == 1            # ACTION2
    assert score.correct_refutations == 1
    assert score.wrong_refutations == 0
    assert score.missed_refutations == 0            # ACTION6 fact is true
    assert score.unverifiable == 1                  # ACTION7 click
    # 2 correct updates / 6 experiment actions
    assert score.experiment_actions == 6
    assert abs(score.experiment_efficiency - 2 / 6) < 1e-9
    # one human-demonstrated true mechanic, and it was confirmed
    assert abs(score.human_alignment - 1.0) < 1e-9
    assert abs(score.confirmation_precision - 0.5) < 1e-9


def test_missed_refutation_detected():
    oracle = MechanicsOracle(game_id="m")
    oracle.add(GroundTruthFact(mechanic_key("ACTION2", "move"), False))
    ledger = [
        HypothesisRecord(mechanic_key("ACTION2", "move"),
                         status=HypothesisStatus.UNRESOLVED, experiments_spent=3),
    ]
    score = score_beliefs(ledger, oracle)
    assert score.missed_refutations == 1
    assert score.experiment_efficiency == 0.0  # no correct updates


def test_experiment_actions_override():
    oracle = _synthetic_oracle()
    ledger = [
        HypothesisRecord(mechanic_key("ACTION1", "global_transform"),
                         status=HypothesisStatus.CONFIRMED, experiments_spent=99),
    ]
    score = score_beliefs(ledger, oracle, experiment_actions=4)
    assert score.experiment_actions == 4
    assert abs(score.experiment_efficiency - 1 / 4) < 1e-9


def test_ar25_oracle_loads_from_artifacts():
    oracle = build_ar25_oracle()
    # Artefacts exist in the repo, so we expect a populated oracle.
    assert oracle.game_id == "ar25-e3c63847"
    assert len(oracle.facts) > 0
    # Empirical: ar25 has no avatar movement -> moving claims are false.
    assert oracle.verdict(mechanic_key("ACTION1", "move")) is False
    # ACTION6 is a no-op in ar25.
    assert oracle.verdict(mechanic_key("ACTION6", "noop")) is True
    # Human-demonstrated: correspondence goal family + ACTION5 control_switch role.
    assert oracle.verdict("goal_family::correspondence") is True
    assert oracle.verdict("action_role::ACTION5::control_switch") is True
    assert oracle.verdict(
        correspondence_key("ACTION2", "validates", (10, 11))
    ) is True
    # Human anti-pattern encodes a known-false win rule.
    assert oracle.verdict("win_rule::all_shapes_must_match") is False
    # At least one human-demonstrated true mechanic exists for alignment scoring.
    assert len(oracle.human_true_keys()) >= 1
