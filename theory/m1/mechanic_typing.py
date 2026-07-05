"""M1.3d-c game mechanic typing diagnostics.

This module asks whether a blocked game is really well represented as a
source-color experiment. It uses raw M1.1 observations plus grounding metrics;
it does not generate, validate, or confirm hypotheses.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .grounding_autopsy import DEFAULT_GROUNDING_AUTOPSY_JSON_PATH
from .observation_dataset import DEFAULT_OUTPUT_PATH as DEFAULT_OBSERVATION_PATH

DEFAULT_MECHANIC_TYPING_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "mechanic_typing.json"
)


@dataclass(frozen=True)
class GameMechanicProfile:
    """Mechanic typing profile for one game."""

    game_id: str
    observations_total: int
    non_reset_observations: int
    raw_rates: Dict[str, float]
    mechanic_scores: Dict[str, float]
    grounding_fit: Dict[str, float]
    mechanic_tags: Tuple[str, ...]
    representation_warning: str | None = None
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "observations_total": int(self.observations_total),
            "non_reset_observations": int(self.non_reset_observations),
            "raw_rates": dict(self.raw_rates),
            "mechanic_scores": dict(self.mechanic_scores),
            "grounding_fit": dict(self.grounding_fit),
            "mechanic_tags": list(self.mechanic_tags),
            "representation_warning": self.representation_warning,
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def load_observation_rows(path: str | Path = DEFAULT_OBSERVATION_PATH) -> List[Dict[str, Any]]:
    """Load M1.1 raw observation rows as dictionaries."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_grounding_fit(
    path: str | Path = DEFAULT_GROUNDING_AUTOPSY_JSON_PATH,
) -> Dict[str, Dict[str, float]]:
    """Load source-color grounding fit from M1.3c+ autopsy output."""
    json_path = Path(path)
    if not json_path.exists():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    fit: Dict[str, Dict[str, float]] = {}
    for game in payload.get("games", []) or []:
        game_id = str(game.get("game_id", ""))
        funnel = game.get("grounding_funnel", {}) or {}
        new_funnel = game.get("new_pair_grounding_funnel", {}) or {}
        discovered = int(funnel.get("pairs_discovered", 0) or 0)
        new_discovered = int(new_funnel.get("pairs_discovered", 0) or 0)
        fit[game_id] = {
            "source_color_predictiveness": _safe_ratio(
                int(funnel.get("pairs_actionable_source", 0) or 0),
                discovered,
            ),
            "source_not_selectable_rate": _safe_ratio(
                int(funnel.get("pairs_blocked_by_unselectable_source", 0) or 0),
                discovered,
            ),
            "new_pair_source_color_predictiveness": _safe_ratio(
                int(new_funnel.get("pairs_actionable_source", 0) or 0),
                new_discovered,
            ),
            "new_pair_source_not_selectable_rate": _safe_ratio(
                int(new_funnel.get("pairs_blocked_by_unselectable_source", 0) or 0),
                new_discovered,
            ),
            "pairs_entering_agenda_rate": _safe_ratio(
                int(funnel.get("pairs_entering_agenda", 0) or 0),
                discovered,
            ),
            "new_pairs_entering_agenda_rate": _safe_ratio(
                int(new_funnel.get("pairs_entering_agenda", 0) or 0),
                new_discovered,
            ),
        }
    return fit


def profile_game_mechanics(
    rows: Sequence[Mapping[str, Any]],
    *,
    grounding_fit: Mapping[str, Mapping[str, float]] | None = None,
) -> Tuple[GameMechanicProfile, ...]:
    """Profile all games represented in raw M1.1 observation rows."""
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("game_id", ""))].append(row)

    profiles: List[GameMechanicProfile] = []
    fit_by_game = grounding_fit or {}
    for game_id in sorted(grouped):
        game_rows = grouped[game_id]
        profiles.append(
            profile_one_game(
                game_id,
                game_rows,
                grounding_fit=fit_by_game.get(game_id, {}),
            )
        )
    return tuple(profiles)


def profile_one_game(
    game_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    grounding_fit: Mapping[str, float] | None = None,
) -> GameMechanicProfile:
    """Profile one game using raw transition observations."""
    non_reset = [row for row in rows if str(row.get("action", "")) != "RESET"]
    denominator = max(1, len(non_reset))
    changed = [row for row in non_reset if int(row.get("num_cells_changed", 0) or 0) > 0]
    with_args = [row for row in non_reset if bool(row.get("action_args"))]
    without_args = [row for row in non_reset if not bool(row.get("action_args"))]

    raw_rates = {
        "color_change_rate": _bool_rate(non_reset, "color_changed"),
        "position_change_rate": _bool_rate(non_reset, "position_changed"),
        "object_motion_rate": _non_empty_rate(non_reset, "object_motion_vectors"),
        "object_creation_rate": _positive_rate(non_reset, "created_object_count"),
        "object_removal_rate": _positive_rate(non_reset, "removed_object_count"),
        "object_count_change_rate": _bool_rate(non_reset, "object_count_changed"),
        "contact_change_rate": _bool_rate(non_reset, "adjacency_changed"),
        "shape_change_rate": _bool_rate(non_reset, "shape_changed"),
        "player_motion_rate": _bool_rate(non_reset, "player_moved"),
        "level_progress_rate": _bool_rate(non_reset, "level_progressed"),
        "action_argument_rate": round(len(with_args) / denominator, 4),
        "action_argument_predictiveness": _argument_predictiveness(
            with_args,
            without_args,
        ),
        "changed_transition_rate": round(len(changed) / denominator, 4),
        "mean_changed_cell_ratio": _mean_number(non_reset, "changed_cell_ratio"),
        "mean_color_pairs_changed": _mean_list_len(non_reset, "color_pairs_changed"),
        "mean_motion_vectors": _mean_list_len(non_reset, "object_motion_vectors"),
    }
    fit = dict(grounding_fit or {})
    fit.setdefault("source_color_predictiveness", 0.0)
    fit.setdefault("source_not_selectable_rate", 0.0)
    fit.setdefault("new_pair_source_color_predictiveness", 0.0)
    fit.setdefault("new_pair_source_not_selectable_rate", 0.0)
    fit.setdefault("pairs_entering_agenda_rate", 0.0)
    fit.setdefault("new_pairs_entering_agenda_rate", 0.0)

    scores = _mechanic_scores(raw_rates, fit)
    tags = _mechanic_tags(scores)
    warning = _representation_warning(raw_rates, scores, fit)
    return GameMechanicProfile(
        game_id=game_id,
        observations_total=len(rows),
        non_reset_observations=len(non_reset),
        raw_rates=raw_rates,
        mechanic_scores=scores,
        grounding_fit={key: round(float(value), 4) for key, value in sorted(fit.items())},
        mechanic_tags=tags,
        representation_warning=warning,
    )


def run_mechanic_typing(
    *,
    observation_path: str | Path = DEFAULT_OBSERVATION_PATH,
    grounding_autopsy_path: str | Path = DEFAULT_GROUNDING_AUTOPSY_JSON_PATH,
) -> Dict[str, Any]:
    """Run M1.3d-c mechanic typing and return a JSON-ready payload."""
    rows = load_observation_rows(observation_path)
    grounding_fit = load_grounding_fit(grounding_autopsy_path)
    profiles = profile_game_mechanics(rows, grounding_fit=grounding_fit)
    return {
        "config": {
            "observation_path": str(observation_path),
            "grounding_autopsy_path": str(grounding_autopsy_path),
            "reset_transitions_excluded_from_rates": True,
        },
        "profiles": [profile.to_dict() for profile in profiles],
        "summary": summarize_profiles(profiles),
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def summarize_profiles(profiles: Sequence[GameMechanicProfile]) -> Dict[str, Any]:
    """Compact comparative summary for mechanic typing."""
    return {
        "game_count": len(profiles),
        "warnings": {
            profile.game_id: profile.representation_warning
            for profile in profiles
            if profile.representation_warning
        },
        "mechanic_tags_by_game": {
            profile.game_id: list(profile.mechanic_tags) for profile in profiles
        },
        "source_color_predictiveness_by_game": {
            profile.game_id: profile.grounding_fit.get(
                "source_color_predictiveness",
                0.0,
            )
            for profile in profiles
        },
    }


def write_mechanic_typing(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_MECHANIC_TYPING_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mechanic_scores(
    raw_rates: Mapping[str, float],
    grounding_fit: Mapping[str, float],
) -> Dict[str, float]:
    source_fit = float(grounding_fit.get("source_color_predictiveness", 0.0))
    scores = {
        "color_change": (
            0.65 * raw_rates["color_change_rate"]
            + 0.35 * min(1.0, raw_rates["mean_color_pairs_changed"] / 3.0)
        ),
        "color_source_grounding": source_fit,
        "position_motion": _mean(
            raw_rates["position_change_rate"],
            raw_rates["object_motion_rate"],
            raw_rates["player_motion_rate"],
        ),
        "object_lifecycle": _mean(
            raw_rates["object_creation_rate"],
            raw_rates["object_removal_rate"],
            raw_rates["object_count_change_rate"],
        ),
        "topology_contact": raw_rates["contact_change_rate"],
        "shape_object": _mean(
            raw_rates["shape_change_rate"],
            raw_rates["object_count_change_rate"],
        ),
        "action_argument": raw_rates["action_argument_rate"],
    }
    return {key: round(float(value), 4) for key, value in sorted(scores.items())}


def _mechanic_tags(scores: Mapping[str, float]) -> Tuple[str, ...]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    tags = [name for name, value in ordered if value >= 0.5]
    if not tags and ordered:
        tags = [ordered[0][0]]
    return tuple(tags[:4])


def _representation_warning(
    raw_rates: Mapping[str, float],
    scores: Mapping[str, float],
    grounding_fit: Mapping[str, float],
) -> str | None:
    source_score = float(grounding_fit.get("source_color_predictiveness", 0.0))
    blocked_rate = float(grounding_fit.get("source_not_selectable_rate", 0.0))
    non_color_strength = max(
        float(scores.get("position_motion", 0.0)),
        float(scores.get("topology_contact", 0.0)),
        float(scores.get("object_lifecycle", 0.0)),
        float(scores.get("shape_object", 0.0)),
    )
    if blocked_rate >= 0.9 and source_score <= 0.1 and non_color_strength >= 0.5:
        return "color_source_schema_misaligned_with_observed_mechanics"
    if raw_rates.get("color_change_rate", 0.0) >= 0.7 and source_score <= 0.1:
        return "color_changes_exist_but_source_color_grounding_is_weak"
    return None


def _bool_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 4)


def _positive_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(
        sum(int(row.get(key, 0) or 0) > 0 for row in rows) / len(rows),
        4,
    )


def _non_empty_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 4)


def _mean_number(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(
        sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows),
        4,
    )


def _mean_list_len(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(len(row.get(key, []) or []) for row in rows) / len(rows), 4)


def _argument_predictiveness(
    with_args: Sequence[Mapping[str, Any]],
    without_args: Sequence[Mapping[str, Any]],
) -> float:
    if not with_args or not without_args:
        return 0.0
    with_change = sum(int(row.get("num_cells_changed", 0) or 0) > 0 for row in with_args)
    without_change = sum(
        int(row.get("num_cells_changed", 0) or 0) > 0 for row in without_args
    )
    return round(
        max(0.0, with_change / len(with_args) - without_change / len(without_args)),
        4,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _mean(*values: float) -> float:
    return round(sum(float(value) for value in values) / max(1, len(values)), 4)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M1.3d-c mechanic typing.")
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATION_PATH)
    parser.add_argument(
        "--grounding-autopsy",
        type=Path,
        default=DEFAULT_GROUNDING_AUTOPSY_JSON_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_MECHANIC_TYPING_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_mechanic_typing(
        observation_path=args.observations,
        grounding_autopsy_path=args.grounding_autopsy,
    )
    write_mechanic_typing(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
