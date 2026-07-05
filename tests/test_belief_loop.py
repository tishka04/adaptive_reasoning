"""Step A prototype: hypothesis -> discriminating experiment -> revision.

Judged by the epistemic metric (step C), not by any game score.
"""

from __future__ import annotations

from theory.epistemic_metrics import HypothesisStatus, mechanic_key
from theory.experiment_designer import DiscriminatingExperimentDesigner
from theory.mechanic_hypothesis import GameTheory, ObservedEffect
from theory.ar25_replay import load_action_outcomes, run_ar25_belief_loop
from theory.role_hypotheses import (
    ActionRoleHypothesis,
    GoalFamilyHypothesis,
    action_role_key,
    goal_family_key,
)


# ── unit: designer separates competing hypotheses ───────────────

def test_designer_prefers_unprobed_then_resolves():
    theory = GameTheory("t")
    theory.seed_actions(["ACTION1", "ACTION2"])
    designer = DiscriminatingExperimentDesigner()

    first = designer.design(theory, ["ACTION1", "ACTION2"])
    assert first is not None
    assert first.action in ("ACTION1", "ACTION2")
    assert len(first.competing_keys) == 2  # two live hypotheses are separated

    # Resolve ACTION1 as a clear global transform; it should stop being chosen.
    transform = ObservedEffect(num_changed=200)
    for _ in range(8):
        theory.observe("ACTION1", transform, was_experiment=True)
    assert theory.dominant("ACTION1") is not None
    assert theory.dominant("ACTION1").kind == "global_transform"

    choice = designer.design(theory, ["ACTION1"])
    # ACTION1 fully resolved -> nothing left to discriminate.
    assert choice is None


# ── integration: closed-loop ar25 prototype scored epistemically ──

def test_game_theory_emits_action_role_and_goal_family_records():
    theory = GameTheory("semantic")
    theory.seed_actions(["ACTION5"])
    theory.add_semantic_hypotheses(
        action_roles=[
            ActionRoleHypothesis(
                action="ACTION5",
                role="control_switch",
                evidence_for=["human_prior"],
                prior_confidence=0.7,
            )
        ],
        goal_families=[
            GoalFamilyHypothesis(
                family="correspondence",
                evidence_for=["human_prior"],
                prior_confidence=0.7,
            )
        ],
    )

    theory.observe("ACTION5", ObservedEffect(num_changed=25), was_experiment=True)
    records = {record.key: record for record in theory.to_ledger()}

    role_key = action_role_key("ACTION5", "control_switch")
    family_key = goal_family_key("correspondence")
    assert records[role_key].status == HypothesisStatus.CONFIRMED
    assert records[family_key].status == HypothesisStatus.CONFIRMED


def test_ar25_belief_loop_is_epistemically_sound():
    outcomes = load_action_outcomes()
    assert outcomes, "expected recorded ar25 action outcomes"

    theory, score, stats = run_ar25_belief_loop(budget=300)

    # The loop terminates by RESOLVING hypotheses, not by exhausting budget.
    assert stats["experiments"] < 300

    # Knowledge quality, judged against ground truth:
    assert score.wrong_confirmations == 0            # never accepted a false mechanic
    assert score.confirmation_precision == 1.0       # every confirmation is correct
    assert score.correct_confirmations >= 6          # the transform/noop mechanics
    assert score.missed_refutations == 0             # no false mechanic left open
    assert score.experiment_efficiency >= 0.5

    # The agent recovers the two robust mechanics directly.
    assert theory.dominant("ACTION6") is not None
    assert theory.dominant("ACTION6").kind == "noop"
    assert theory.dominant("ACTION1").kind == "global_transform"

    # ACTION5's low-amplitude selector/control effect is treated as click-like
    # at the effect layer and control_switch at the role layer.
    assert score.wrong_refutations == 0
    assert score.human_alignment > 0.0
    assert score.confirmation_precision == 1.0
