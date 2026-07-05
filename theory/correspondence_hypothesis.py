"""Explicit correspondence hypotheses over live transitions.

These hypotheses sit above action effects: they ask whether an action helps
establish or validate a relation between two object families, such as the
yellow/purple shape correspondence in ar25.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from .epistemic_metrics import HypothesisRecord, HypothesisStatus

MIN_SUPPORT = 2
MIN_NET_SIGNAL = 0.5
MIN_CONFIDENCE = 0.60
IMPROVEMENT_EPSILON = 0.25
REGRESSION_EPSILON = 2.0

_PREDICATES_BY_RELATION = {
    "validates": (
        "SameShape",
        "AlignedWith",
        "PairedWith",
        "CorrespondenceCount",
        "CorrespondenceImprovesAfter",
    ),
    "improves": (
        "SameShape",
        "AlignedWith",
        "PairedWith",
        "CorrespondenceImprovesAfter",
    ),
    "establishes": (
        "SameShape",
        "AlignedWith",
        "PairedWith",
        "CorrespondenceCount",
    ),
    "modifies": (
        "AlignedWith",
        "PairedWith",
        "CorrespondenceImprovesAfter",
    ),
}


def normalize_pair_colors(pair_colors: Iterable[Any]) -> Tuple[int, int]:
    colors = tuple(int(color) for color in pair_colors)
    if len(colors) != 2:
        raise ValueError(f"expected two pair colors, got {colors!r}")
    return colors  # type: ignore[return-value]


def correspondence_key(
    action: str,
    relation: str,
    pair_colors: Tuple[int, int],
) -> str:
    """Canonical key for correspondence hypotheses."""
    first, second = normalize_pair_colors(pair_colors)
    return f"correspondence::{str(action).upper()}::{relation}::colors{first}_{second}"


def predicate_names_for_relation(relation: str) -> Tuple[str, ...]:
    """Human-facing predicates represented by a correspondence relation."""
    key = str(relation or "improves").strip().lower()
    return _PREDICATES_BY_RELATION.get(key, ("PairedWith",))


@dataclass
class CorrespondenceRule:
    """A confirmed source-target relation induced from live transitions."""

    action: str
    relation: str
    pair_colors: Tuple[int, int]
    predicates: Tuple[str, ...]
    support: int
    confidence: float
    source_hypothesis_key: str = ""

    def __post_init__(self) -> None:
        self.action = str(self.action).upper()
        self.relation = str(self.relation or "improves").strip().lower()
        self.pair_colors = normalize_pair_colors(self.pair_colors)
        self.predicates = tuple(str(name) for name in self.predicates)

    @property
    def key(self) -> str:
        return self.source_hypothesis_key or correspondence_key(
            self.action,
            self.relation,
            self.pair_colors,
        )


@dataclass
class CorrespondenceObservation:
    """One action's relation-level signal before/after a transition."""

    action: str
    pair_colors: Tuple[int, int]
    match_delta: float = 0.0
    global_delta: float = 0.0
    matched_pairs_delta: int = 0
    unmatched_total_delta: int = 0
    level_complete: bool = False
    game_over: bool = False

    @property
    def improves(self) -> bool:
        return (
            self.match_delta > IMPROVEMENT_EPSILON
            or self.global_delta > IMPROVEMENT_EPSILON
            or self.matched_pairs_delta > 0
            or self.unmatched_total_delta < 0
        )

    @property
    def regresses_strongly(self) -> bool:
        return (
            self.match_delta < -REGRESSION_EPSILON
            and self.global_delta < -REGRESSION_EPSILON
            and self.matched_pairs_delta <= 0
            and self.unmatched_total_delta >= 0
            and not self.level_complete
        )

    @classmethod
    def from_transition(
        cls,
        record: Any,
        *,
        pair_colors: Tuple[int, int],
    ) -> "CorrespondenceObservation":
        before_grid = getattr(getattr(record, "obs_before", None), "raw_grid")
        after_grid = getattr(getattr(record, "obs_after", None), "raw_grid")
        before_match, after_match, before_global, after_global = _score_grids(
            before_grid,
            after_grid,
            pair_colors=pair_colors,
        )
        before_unmatched = before_match.unmatched_first + before_match.unmatched_second
        after_unmatched = after_match.unmatched_first + after_match.unmatched_second
        diff = getattr(record, "diff", None)
        return cls(
            action=str(getattr(getattr(record, "action", None), "name", "")).upper(),
            pair_colors=normalize_pair_colors(pair_colors),
            match_delta=float(after_match.score - before_match.score),
            global_delta=float(after_global.score - before_global.score),
            matched_pairs_delta=int(after_match.matched_pairs - before_match.matched_pairs),
            unmatched_total_delta=int(after_unmatched - before_unmatched),
            level_complete=bool(getattr(diff, "level_complete", False)),
            game_over=bool(getattr(diff, "game_over", False)),
        )


@dataclass
class CorrespondenceHypothesis:
    """A falsifiable claim about an action and a source-target relation."""

    action: str
    relation: str
    pair_colors: Tuple[int, int] = (10, 11)
    statement: str = ""
    support: int = 0
    contradictions: int = 0
    experiments_spent: int = 0
    cumulative_match_delta: float = 0.0
    cumulative_global_delta: float = 0.0
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    def __post_init__(self) -> None:
        self.action = str(self.action).upper()
        self.relation = str(self.relation or "improves").strip().lower()
        self.pair_colors = normalize_pair_colors(self.pair_colors)
        if not self.statement:
            first, second = self.pair_colors
            self.statement = (
                f"{self.action} {self.relation} correspondence "
                f"between colors {first} and {second}"
            )

    @property
    def key(self) -> str:
        return correspondence_key(self.action, self.relation, self.pair_colors)

    @property
    def confidence(self) -> float:
        total = self.support + self.contradictions
        if total == 0:
            return 0.0
        return self.support / total

    @property
    def net_signal(self) -> float:
        return self.cumulative_match_delta + self.cumulative_global_delta

    def observe(
        self,
        observation: CorrespondenceObservation,
        *,
        was_experiment: bool = False,
    ) -> None:
        if observation.action != self.action:
            return
        if normalize_pair_colors(observation.pair_colors) != self.pair_colors:
            return
        if was_experiment:
            self.experiments_spent += 1
        self.cumulative_match_delta += observation.match_delta
        self.cumulative_global_delta += observation.global_delta

        if self.relation == "validates":
            if observation.level_complete:
                self.support += 1
            elif observation.game_over:
                self.contradictions += 1
        elif self.relation in {"improves", "establishes", "modifies"}:
            if observation.improves or observation.level_complete:
                self.support += 1
            elif observation.regresses_strongly:
                self.contradictions += 1
        self._recompute_status()

    def _recompute_status(self) -> None:
        if self.support >= MIN_SUPPORT and self.confidence >= MIN_CONFIDENCE:
            if self.relation == "validates" or self.net_signal >= MIN_NET_SIGNAL:
                self.status = HypothesisStatus.CONFIRMED
                return
        if self.contradictions >= MIN_SUPPORT and self.contradictions > self.support:
            self.status = HypothesisStatus.REFUTED
            return
        self.status = HypothesisStatus.UNRESOLVED

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=self.statement,
            status=self.status,
            support=self.support,
            contradictions=self.contradictions,
            experiments_spent=self.experiments_spent,
        )

    def to_rule(self) -> CorrespondenceRule | None:
        if self.status != HypothesisStatus.CONFIRMED:
            return None
        return CorrespondenceRule(
            action=self.action,
            relation=self.relation,
            pair_colors=self.pair_colors,
            predicates=predicate_names_for_relation(self.relation),
            support=self.support,
            confidence=self.confidence,
            source_hypothesis_key=self.key,
        )


def load_task_program_correspondence_hypotheses(
    path: Path,
    *,
    pair_colors: Tuple[int, int] = (10, 11),
) -> List[CorrespondenceHypothesis]:
    """Seed relation hypotheses from correspondence-oriented subgoals."""
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        program = json.load(handle)

    hypotheses: dict[tuple[str, str], CorrespondenceHypothesis] = {}
    for subgoal in program.get("subgoal_tests", []) or []:
        text = " ".join(
            str(subgoal.get(name, ""))
            for name in ("id", "description", "verification", "expected_signal")
        ).lower()
        if "correspond" not in text and "match" not in text:
            continue
        expected = str(subgoal.get("expected_signal", "")).lower()
        for action in subgoal.get("prefer_actions", []) or []:
            action_name = str(action).upper()
            relation = "validates" if expected == "level_advance" else "improves"
            hypotheses.setdefault(
                (action_name, relation),
                CorrespondenceHypothesis(
                    action=action_name,
                    relation=relation,
                    pair_colors=pair_colors,
                ),
            )

    return list(hypotheses.values())


def _score_grids(before_grid: Any, after_grid: Any, *, pair_colors: Tuple[int, int]):
    from task_program_guided_level7 import global_correspondence_score, match_score

    before_match = match_score(before_grid, pair_colors=pair_colors)
    after_match = match_score(after_grid, pair_colors=pair_colors)
    before_global = global_correspondence_score(before_grid, pair_colors=pair_colors)
    after_global = global_correspondence_score(after_grid, pair_colors=pair_colors)
    return before_match, after_match, before_global, after_global
