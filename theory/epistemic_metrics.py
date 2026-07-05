"""Epistemic evaluation of a belief-revision loop.

The success criterion for the 'theory of game' architecture is EPISTEMIC,
not ludic: how many mechanic hypotheses the agent confirmed / refuted
*correctly*, how efficiently (per experiment action spent), and how well it
recovers the mechanics a human actually relied on.

This module is intentionally game-agnostic. An environment-specific
``MechanicsOracle`` supplies the ground truth (see ``theory.ar25_oracle``).

Contract for step A (belief loop): produce a list of ``HypothesisRecord``
whose ``key`` is built with ``mechanic_key`` (or any string that the oracle
also uses), and call ``score_beliefs(ledger, oracle)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional


# =====================================================================
# Canonical key vocabulary (shared by oracle + belief loop)
# =====================================================================

# Maps the empirical action-ontology vocabulary ("transform_like", "no_op",
# "move_like", ...) onto the v3 OperatorKind vocabulary so that hypotheses
# produced by the belief loop (which speak OperatorKind) match oracle facts.
_OPERATOR_KIND_ALIASES = {
    "transform_like": "global_transform",
    "transform": "global_transform",
    "global_transform": "global_transform",
    "no_op": "noop",
    "noop": "noop",
    "move_like": "move",
    "move": "move",
    "click_like": "click",
    "click": "click",
    "lethal_like": "lethal",
    "lethal": "lethal",
    "interact_like": "interact",
    "interact": "interact",
}


def normalize_operator_kind(raw: str) -> str:
    """Normalise an operator-type label to the v3 OperatorKind vocabulary."""
    key = str(raw or "").strip().lower()
    if key in _OPERATOR_KIND_ALIASES:
        return _OPERATOR_KIND_ALIASES[key]
    if key.endswith("_like"):
        return key[: -len("_like")]
    return key or "unknown"


def mechanic_key(action: str, kind: str) -> str:
    """Canonical key for an action-effect mechanic hypothesis/fact."""
    return f"action_effect::{str(action).upper()}::{normalize_operator_kind(kind)}"


# =====================================================================
# Data contract
# =====================================================================

class HypothesisStatus(str, Enum):
    UNRESOLVED = "unresolved"   # entertained but not yet decided
    CONFIRMED = "confirmed"     # the loop accepted it as true
    REFUTED = "refuted"         # the loop rejected it as false


@dataclass
class HypothesisRecord:
    """One mechanic hypothesis the belief loop entertained, with its verdict."""

    key: str
    description: str = ""
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED
    support: int = 0
    contradictions: int = 0
    experiments_spent: int = 0  # actions deliberately spent testing this


@dataclass
class GroundTruthFact:
    """A known-true or known-false mechanic for one game."""

    key: str
    truth_value: bool
    description: str = ""
    demonstrated_by_human: bool = False
    source: str = ""


@dataclass
class MechanicsOracle:
    """Ground-truth mechanics for a single game."""

    game_id: str
    facts: Dict[str, GroundTruthFact] = field(default_factory=dict)

    def add(self, fact: GroundTruthFact) -> None:
        self.facts[fact.key] = fact

    def verdict(self, key: str) -> Optional[bool]:
        """True / False if the mechanic is known, else None (unverifiable)."""
        fact = self.facts.get(key)
        return None if fact is None else fact.truth_value

    def human_true_keys(self) -> List[str]:
        """Keys of true mechanics the human's successful trace relied on."""
        return [
            key
            for key, fact in self.facts.items()
            if fact.truth_value and fact.demonstrated_by_human
        ]


# =====================================================================
# Score
# =====================================================================

@dataclass
class EpistemicScore:
    """Knowledge-quality report card for one belief-revision episode."""

    game_id: str = ""

    # The six metrics requested for step C
    hypotheses_confirmed: int = 0
    hypotheses_refuted: int = 0
    wrong_confirmations: int = 0       # confirmed but actually false
    missed_refutations: int = 0        # false, entertained, never refuted
    experiment_efficiency: float = 0.0  # correct updates / experiment actions
    human_alignment: float = 0.0       # recall of human-demonstrated mechanics

    # Supporting counts (needed for efficiency, useful for diagnosis)
    correct_confirmations: int = 0
    correct_refutations: int = 0
    wrong_refutations: int = 0         # refuted but actually true
    unverifiable: int = 0             # no matching oracle fact
    experiment_actions: int = 0

    @property
    def confirmation_precision(self) -> float:
        if self.hypotheses_confirmed == 0:
            return 0.0
        return self.correct_confirmations / self.hypotheses_confirmed

    def to_dict(self) -> Dict[str, object]:
        return {
            "game_id": self.game_id,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "wrong_confirmations": self.wrong_confirmations,
            "missed_refutations": self.missed_refutations,
            "experiment_efficiency": round(self.experiment_efficiency, 4),
            "human_alignment": round(self.human_alignment, 4),
            "correct_confirmations": self.correct_confirmations,
            "correct_refutations": self.correct_refutations,
            "wrong_refutations": self.wrong_refutations,
            "unverifiable": self.unverifiable,
            "experiment_actions": self.experiment_actions,
            "confirmation_precision": round(self.confirmation_precision, 4),
        }


def score_beliefs(
    ledger: Iterable[HypothesisRecord],
    oracle: MechanicsOracle,
    *,
    experiment_actions: Optional[int] = None,
) -> EpistemicScore:
    """Score a belief ledger against ground truth.

    ``experiment_actions`` overrides the action budget used for efficiency;
    when None it is summed from ``HypothesisRecord.experiments_spent``.
    """
    score = EpistemicScore(game_id=oracle.game_id)
    confirmed_true_keys: set[str] = set()
    spent = 0

    for rec in ledger:
        spent += int(rec.experiments_spent)
        verdict = oracle.verdict(rec.key)

        if verdict is None:
            score.unverifiable += 1
            continue

        if rec.status == HypothesisStatus.CONFIRMED:
            score.hypotheses_confirmed += 1
            if verdict:
                score.correct_confirmations += 1
                confirmed_true_keys.add(rec.key)
            else:
                score.wrong_confirmations += 1
        elif rec.status == HypothesisStatus.REFUTED:
            score.hypotheses_refuted += 1
            if verdict:
                score.wrong_refutations += 1
            else:
                score.correct_refutations += 1
        else:  # UNRESOLVED
            if verdict is False:
                # A false mechanic was entertained but never rejected.
                score.missed_refutations += 1

    score.experiment_actions = (
        int(experiment_actions) if experiment_actions is not None else spent
    )
    correct_updates = score.correct_confirmations + score.correct_refutations
    score.experiment_efficiency = correct_updates / max(1, score.experiment_actions)

    human_keys = set(oracle.human_true_keys())
    if human_keys:
        score.human_alignment = len(human_keys & confirmed_true_keys) / len(human_keys)

    return score
