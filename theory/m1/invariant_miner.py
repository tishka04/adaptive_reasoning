"""M1.2 multi-game invariant mining over raw M1.1 observations.

The miner consumes only level-1 fields emitted by ``observation_dataset``:
change flags, numeric counts, and direct before/after measurements. It does not
read A12 predicate names and does not produce confirmed hypotheses.

Run:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.invariant_miner
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from .observation_dataset import DEFAULT_OUTPUT_PATH as DEFAULT_DATASET_PATH
from .observation_dataset import RawTransitionObservation

DEFAULT_OUTPUT_DIR = Path("diagnostics") / "m1"
DEFAULT_ACCEPTED_PATH = DEFAULT_OUTPUT_DIR / "accepted_invariants.json"
DEFAULT_REJECTED_PATH = DEFAULT_OUTPUT_DIR / "rejected_invariants.json"
MIN_GAMES = 2
MIN_INTRA_GAME_SUPPORT = 0.6
MIN_NOVELTY = 0.3
MAX_CONTEXTS_PER_INVARIANT = 24
REJECTED_REASONS = {"single_game", "low_novelty", "low_support"}
EXCLUDED_ACTIONS = ("RESET",)

EXISTING_PREDICATE_SIGNATURES: Tuple[Tuple[str, str, str], ...] = (
    ("source_target_color_transform", "color_pair_change", "appears"),
    ("paired_with", "color_contact", "preserved"),
    ("same_shape", "shape_profile", "preserved"),
    ("aligned_with", "position", "preserved"),
    ("adjacent_to", "adjacency", "preserved"),
)


@dataclass(frozen=True)
class LatentInvariant:
    """A raw-pattern invariant candidate emitted by M1.2."""

    name: str
    attribute: str
    outcome: str
    support: int
    contexts: Tuple[Dict[str, Any], ...] = ()
    games_supporting: frozenset[str] = field(default_factory=frozenset)
    cross_game_score: float = 0.0
    novelty_score: float = 0.0
    rejected_reason: str | None = None
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attribute": self.attribute,
            "outcome": self.outcome,
            "support": int(self.support),
            "contexts": list(self.contexts),
            "games_supporting": sorted(self.games_supporting),
            "cross_game_score": round(float(self.cross_game_score), 6),
            "novelty_score": round(float(self.novelty_score), 6),
            "rejected_reason": self.rejected_reason,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "LatentInvariant":
        return cls(
            name=str(row["name"]),
            attribute=str(row["attribute"]),
            outcome=str(row["outcome"]),
            support=int(row.get("support", 0)),
            contexts=tuple(dict(context) for context in row.get("contexts", [])),
            games_supporting=frozenset(str(game) for game in row.get("games_supporting", [])),
            cross_game_score=float(row.get("cross_game_score", 0.0)),
            novelty_score=float(row.get("novelty_score", 0.0)),
            rejected_reason=row.get("rejected_reason"),
            trace_support_counted_as_proof=bool(
                row.get("trace_support_counted_as_proof", False)
            ),
            prior_counted_as_proof=bool(row.get("prior_counted_as_proof", False)),
        )


@dataclass(frozen=True)
class RawOutcomeRate:
    """Outcome rates for one game/action/raw-attribute cell."""

    game_id: str
    action: str
    attribute: str
    observations: int
    outcome_counts: Dict[str, int]

    def rate(self, outcome: str) -> float:
        return _safe_ratio(self.outcome_counts.get(outcome, 0), self.observations)


@dataclass(frozen=True)
class InvariantMiningResult:
    """Accepted and rejected M1.2 invariants."""

    accepted_invariants: Tuple[LatentInvariant, ...]
    rejected_invariants: Tuple[LatentInvariant, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_count": len(self.accepted_invariants),
            "rejected_count": len(self.rejected_invariants),
            "accepted_invariants": [
                invariant.to_dict() for invariant in self.accepted_invariants
            ],
            "rejected_invariants": [
                invariant.to_dict() for invariant in self.rejected_invariants
            ],
        }


def mine_invariants(
    dataset: Iterable[RawTransitionObservation | Mapping[str, Any]],
    *,
    min_games: int = MIN_GAMES,
    min_intra_game_support: float = MIN_INTRA_GAME_SUPPORT,
    min_novelty: float = MIN_NOVELTY,
    excluded_actions: Sequence[str] = EXCLUDED_ACTIONS,
) -> InvariantMiningResult:
    """Mine cross-game raw invariants and keep rejected candidates for M2."""
    rates = summarize_raw_outcome_rates(dataset, excluded_actions=excluded_actions)
    candidates = _candidate_invariants(
        rates,
        min_games=min_games,
        min_intra_game_support=min_intra_game_support,
    )
    return _filter_by_transfer_and_novelty(
        candidates,
        min_games=min_games,
        min_novelty=min_novelty,
    )


def summarize_raw_outcome_rates(
    dataset: Iterable[RawTransitionObservation | Mapping[str, Any]],
    *,
    excluded_actions: Sequence[str] = EXCLUDED_ACTIONS,
) -> Tuple[RawOutcomeRate, ...]:
    """Compute direct rates per ``(game_id, action, raw_attribute)``."""
    excluded = {str(action).upper() for action in excluded_actions}
    counts: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[Tuple[str, str, str]] = Counter()

    for observation in dataset:
        row = _row_dict(observation)
        game_id = str(row.get("game_id", ""))
        action = str(row.get("action", "")).upper()
        if not game_id or not action or action in excluded:
            continue
        for attribute, outcome in raw_attribute_outcomes(row):
            key = (game_id, action, attribute)
            counts[key][outcome] += 1
            totals[key] += 1

    rates: List[RawOutcomeRate] = []
    for (game_id, action, attribute), outcome_counts in sorted(counts.items()):
        rates.append(
            RawOutcomeRate(
                game_id=game_id,
                action=action,
                attribute=attribute,
                observations=int(totals[(game_id, action, attribute)]),
                outcome_counts=dict(sorted(outcome_counts.items())),
            )
        )
    return tuple(rates)


def raw_attribute_outcomes(row: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Map one M1.1 row into raw attribute/outcome observations."""
    outcomes: List[Tuple[str, str]] = []
    _append_changed_flag(outcomes, row, "shape_changed", "shape_profile")
    _append_changed_flag(outcomes, row, "color_changed", "color_distribution")
    _append_changed_flag(outcomes, row, "position_changed", "position")
    _append_changed_flag(outcomes, row, "adjacency_changed", "adjacency")
    _append_changed_flag(outcomes, row, "object_count_changed", "object_count")
    _append_changed_flag(outcomes, row, "player_moved", "player_position")

    if "num_cells_changed" in row:
        outcomes.append(
            (
                "changed_cells",
                "modified" if _as_number(row.get("num_cells_changed")) > 0 else "preserved",
            )
        )
    if "level_progressed" in row:
        outcomes.append(("level_progress", "appears" if row.get("level_progressed") else "absent"))
    if "game_over" in row:
        outcomes.append(("terminal_state", "appears" if row.get("game_over") else "absent"))
    if "grid_shape_before" in row and "grid_shape_after" in row:
        outcomes.append(
            (
                "grid_extent",
                "modified"
                if list(row.get("grid_shape_before") or []) != list(row.get("grid_shape_after") or [])
                else "preserved",
            )
        )
    if "counts_by_color_before" in row and "counts_by_color_after" in row:
        outcomes.append(
            (
                "color_counts",
                "modified"
                if dict(row.get("counts_by_color_before") or {})
                != dict(row.get("counts_by_color_after") or {})
                else "preserved",
            )
        )
    if "object_counts_by_color_before" in row and "object_counts_by_color_after" in row:
        outcomes.append(
            (
                "object_color_counts",
                "modified"
                if dict(row.get("object_counts_by_color_before") or {})
                != dict(row.get("object_counts_by_color_after") or {})
                else "preserved",
            )
        )
    if "object_sizes_before" in row and "object_sizes_after" in row:
        outcomes.append(
            (
                "shape_inventory",
                "modified"
                if list(row.get("object_sizes_before") or [])
                != list(row.get("object_sizes_after") or [])
                else "preserved",
            )
        )
    if "created_object_count" in row:
        outcomes.append(
            (
                "object_creation",
                "appears" if _as_number(row.get("created_object_count")) > 0 else "absent",
            )
        )
    if "removed_object_count" in row:
        outcomes.append(
            (
                "object_removal",
                "appears" if _as_number(row.get("removed_object_count")) > 0 else "absent",
            )
        )
    if "color_pairs_changed" in row:
        outcomes.append(
            (
                "color_pair_change",
                "appears" if list(row.get("color_pairs_changed") or []) else "absent",
            )
        )
    if "adjacency_pairs_before" in row and "adjacency_pairs_after" in row:
        outcomes.append(
            (
                "color_contact",
                "modified"
                if list(row.get("adjacency_pairs_before") or [])
                != list(row.get("adjacency_pairs_after") or [])
                else "preserved",
            )
        )
    if "object_motion_vectors" in row:
        outcomes.append(
            (
                "object_motion",
                "appears" if list(row.get("object_motion_vectors") or []) else "absent",
            )
        )
    return tuple(outcomes)


def cross_game_score(per_game_support: Mapping[str, float]) -> float:
    if not per_game_support:
        return 0.0
    min_support = min(float(value) for value in per_game_support.values())
    return round(min_support * math.log1p(len(per_game_support)), 6)


def novelty_score(
    attribute: str,
    outcome: str,
    accepted: Iterable[LatentInvariant] = (),
) -> float:
    tokens = _signature_tokens(attribute, outcome)
    comparison_signatures = [
        _existing_predicate_tokens(existing_name, existing_attribute, existing_outcome)
        for existing_name, existing_attribute, existing_outcome in EXISTING_PREDICATE_SIGNATURES
    ]
    comparison_signatures.extend(
        _signature_tokens(invariant.attribute, invariant.outcome)
        for invariant in accepted
    )
    max_similarity = max(
        (_jaccard(tokens, other) for other in comparison_signatures),
        default=0.0,
    )
    return round(max(0.0, 1.0 - max_similarity), 6)


def write_invariant_outputs(
    result: InvariantMiningResult,
    *,
    accepted_path: str | Path = DEFAULT_ACCEPTED_PATH,
    rejected_path: str | Path = DEFAULT_REJECTED_PATH,
) -> None:
    accepted = Path(accepted_path)
    rejected = Path(rejected_path)
    accepted.parent.mkdir(parents=True, exist_ok=True)
    rejected.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_text(
        json.dumps(
            [invariant.to_dict() for invariant in result.accepted_invariants],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rejected.write_text(
        json.dumps(
            [invariant.to_dict() for invariant in result.rejected_invariants],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_invariants_json(path: str | Path) -> Tuple[LatentInvariant, ...]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(LatentInvariant.from_dict(row) for row in rows)


def iter_observation_rows(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _candidate_invariants(
    rates: Sequence[RawOutcomeRate],
    *,
    min_games: int,
    min_intra_game_support: float,
) -> Tuple[LatentInvariant, ...]:
    candidates: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    all_outcomes: Dict[str, set[str]] = defaultdict(set)
    for rate in rates:
        all_outcomes[rate.attribute].update(rate.outcome_counts)

    for rate in rates:
        for outcome in sorted(all_outcomes[rate.attribute]):
            outcome_count = int(rate.outcome_counts.get(outcome, 0))
            rate_value = rate.rate(outcome)
            candidates[(rate.attribute, outcome)].append(
                {
                    "game_id": rate.game_id,
                    "action": rate.action,
                    "attribute": rate.attribute,
                    "outcome": outcome,
                    "observations": int(rate.observations),
                    "support": outcome_count,
                    "rate": round(rate_value, 6),
                    "supports_threshold": bool(
                        outcome_count > 0 and rate_value >= min_intra_game_support
                    ),
                }
            )

    invariants: List[LatentInvariant] = []
    for (attribute, outcome), contexts in candidates.items():
        supporting = [context for context in contexts if context["supports_threshold"]]
        per_game_support: Dict[str, float] = {}
        for context in supporting:
            game_id = str(context["game_id"])
            per_game_support[game_id] = max(
                per_game_support.get(game_id, 0.0),
                float(context["rate"]),
            )
        reason = None
        if not supporting:
            reason = "low_support"
        elif len(per_game_support) < min_games:
            reason = "single_game"
        support = sum(int(context["support"]) for context in supporting)
        invariants.append(
            LatentInvariant(
                name=_invariant_name(attribute, outcome),
                attribute=attribute,
                outcome=outcome,
                support=support,
                contexts=tuple(_trim_contexts(contexts)),
                games_supporting=frozenset(per_game_support),
                cross_game_score=cross_game_score(per_game_support),
                novelty_score=0.0,
                rejected_reason=reason,
            )
        )
    return tuple(
        sorted(
            invariants,
            key=lambda invariant: (
                invariant.rejected_reason is not None,
                -invariant.cross_game_score,
                -invariant.support,
                invariant.attribute,
                invariant.outcome,
            ),
        )
    )


def _filter_by_transfer_and_novelty(
    candidates: Sequence[LatentInvariant],
    *,
    min_games: int,
    min_novelty: float,
) -> InvariantMiningResult:
    accepted: List[LatentInvariant] = []
    rejected: List[LatentInvariant] = []
    for candidate in candidates:
        score = novelty_score(candidate.attribute, candidate.outcome, accepted)
        reason = candidate.rejected_reason
        if reason is None and len(candidate.games_supporting) < min_games:
            reason = "single_game"
        if reason is None and score < min_novelty:
            reason = "low_novelty"
        invariant = LatentInvariant(
            name=candidate.name,
            attribute=candidate.attribute,
            outcome=candidate.outcome,
            support=candidate.support,
            contexts=candidate.contexts,
            games_supporting=candidate.games_supporting,
            cross_game_score=candidate.cross_game_score,
            novelty_score=score,
            rejected_reason=reason,
        )
        if reason is None:
            accepted.append(invariant)
        else:
            rejected.append(invariant)
    accepted.sort(
        key=lambda invariant: (
            -invariant.cross_game_score,
            -invariant.support,
            invariant.attribute,
            invariant.outcome,
        )
    )
    rejected.sort(
        key=lambda invariant: (
            invariant.rejected_reason or "",
            -invariant.cross_game_score,
            -invariant.support,
            invariant.attribute,
            invariant.outcome,
        )
    )
    return InvariantMiningResult(tuple(accepted), tuple(rejected))


def _trim_contexts(contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        contexts,
        key=lambda context: (
            not context["supports_threshold"],
            -float(context["rate"]),
            -int(context["support"]),
            str(context["game_id"]),
            str(context["action"]),
        ),
    )[:MAX_CONTEXTS_PER_INVARIANT]


def _append_changed_flag(
    outcomes: List[Tuple[str, str]],
    row: Mapping[str, Any],
    field_name: str,
    attribute: str,
) -> None:
    if field_name in row:
        outcomes.append((attribute, "modified" if bool(row.get(field_name)) else "preserved"))


def _signature_tokens(attribute: str, outcome: str) -> frozenset[str]:
    raw_tokens = re.split(r"[^a-z0-9]+", f"{attribute}_{outcome}".lower())
    tokens = {_normalise_signature_token(token) for token in raw_tokens if token}
    tokens.discard("")
    return frozenset(tokens)


def _existing_predicate_tokens(name: str, attribute: str, outcome: str) -> frozenset[str]:
    raw_tokens = re.split(
        r"[^a-z0-9]+",
        f"{name}_{attribute}_{outcome}".lower(),
    )
    tokens = {_normalise_signature_token(token) for token in raw_tokens if token}
    tokens.discard("")
    return frozenset(tokens)


def _normalise_signature_token(token: str) -> str:
    mapping = {
        "profiles": "shape",
        "profile": "shape",
        "inventory": "shape",
        "inventories": "shape",
        "sizes": "shape",
        "size": "shape",
        "distribution": "count",
        "distributions": "count",
        "counts": "count",
        "created": "creation",
        "create": "creation",
        "removed": "removal",
        "remove": "removal",
        "contacts": "adjacency",
        "contact": "adjacency",
        "adjacent": "adjacency",
        "positions": "position",
        "motion": "position",
        "moved": "position",
        "cells": "cell",
        "changes": "change",
        "changed": "change",
        "transform": "change",
        "transformation": "change",
        "terminal": "terminal",
        "state": "terminal",
    }
    return mapping.get(token, token)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _invariant_name(attribute: str, outcome: str) -> str:
    return f"{attribute}_{outcome}"


def _row_dict(observation: RawTransitionObservation | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(observation, Mapping):
        return dict(observation)
    if hasattr(observation, "to_dict"):
        return observation.to_dict()
    raise TypeError(f"Unsupported observation type: {type(observation)!r}")


def _as_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine M1.2 multi-game invariants.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-games", type=int, default=MIN_GAMES)
    parser.add_argument(
        "--min-intra-game-support",
        type=float,
        default=MIN_INTRA_GAME_SUPPORT,
    )
    parser.add_argument("--min-novelty", type=float, default=MIN_NOVELTY)
    parser.add_argument("--include-reset", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    excluded_actions: Sequence[str] = () if args.include_reset else EXCLUDED_ACTIONS
    result = mine_invariants(
        iter_observation_rows(args.dataset),
        min_games=args.min_games,
        min_intra_game_support=args.min_intra_game_support,
        min_novelty=args.min_novelty,
        excluded_actions=excluded_actions,
    )
    accepted_path = args.out_dir / "accepted_invariants.json"
    rejected_path = args.out_dir / "rejected_invariants.json"
    write_invariant_outputs(
        result,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
    )
    summary = {
        "dataset": str(args.dataset),
        "accepted_count": len(result.accepted_invariants),
        "rejected_count": len(result.rejected_invariants),
        "accepted_path": str(accepted_path),
        "rejected_path": str(rejected_path),
        "min_games": args.min_games,
        "min_intra_game_support": args.min_intra_game_support,
        "min_novelty": args.min_novelty,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
