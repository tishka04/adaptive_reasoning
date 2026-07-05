"""M1.3j mechanic observation to revision candidate.

M1.3j translates a structured mechanic observation into an unresolved revision
candidate for a future A15-A31 controlled test. It does not revise beliefs and
does not confirm anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .recommended_experiment_choice import (
    DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH,
)

DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "mechanic_revision_candidates.json"
)


@dataclass(frozen=True)
class MechanicPredictionCandidate:
    """A testable mechanic prediction extracted from a mechanic observation."""

    candidate_id: str
    game_id: str
    mechanic_family: str
    action: str
    predicted_metric: str
    expected_outcome: str
    observed_outcome: str
    observed_delta_summary: Dict[str, Any] = field(default_factory=dict)
    key: str = ""
    status: str = "UNRESOLVED"
    controlled_test_required: bool = True
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "mechanic_family": self.mechanic_family,
            "action": self.action,
            "predicted_metric": self.predicted_metric,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
            "observed_delta_summary": dict(self.observed_delta_summary),
            "key": self.key or mechanic_prediction_key(
                self.game_id,
                self.action,
                self.mechanic_family,
                self.predicted_metric,
            ),
            "status": self.status,
            "controlled_test_required": self.controlled_test_required,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
        }


@dataclass(frozen=True)
class MechanicRevisionCandidate:
    """A proposed A15-A31 revision candidate that remains unresolved."""

    revision_candidate_id: str
    prediction: MechanicPredictionCandidate
    proposed_status: str = "UNRESOLVED"
    proposed_support: int = 0
    proposed_contradictions: int = 0
    proposed_experiments_spent: int = 0
    revision_reason: str = "mechanic_observation_available_controlled_test_required"
    a15_a31_ready: bool = True
    revision_performed: bool = False
    wrong_confirmations: int = 0
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_candidate_id": self.revision_candidate_id,
            "prediction": self.prediction.to_dict(),
            "a15_a31_revision_proposal": {
                "key": self.prediction.to_dict()["key"],
                "description": (
                    f"{self.prediction.action} {self.prediction.mechanic_family} "
                    f"via {self.prediction.predicted_metric}"
                ),
                "proposed_status": self.proposed_status,
                "support": int(self.proposed_support),
                "contradictions": int(self.proposed_contradictions),
                "experiments_spent": int(self.proposed_experiments_spent),
                "controlled_test_required": True,
                "observation_counted_as_confirmation": False,
            },
            "revision_reason": self.revision_reason,
            "a15_a31_ready": self.a15_a31_ready,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
        }


def run_mechanic_revision_candidate_generation(
    *,
    choice_path: str | Path = DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Generate unresolved revision candidates from M1.3i output."""
    payload = json.loads(Path(choice_path).read_text(encoding="utf-8"))
    observation = dict(payload.get("mechanic_observation", {}) or {})
    if not observation:
        raise ValueError("recommended choice payload has no mechanic_observation")
    prediction = prediction_candidate_from_observation(observation)
    revision = revision_candidate_from_prediction(prediction)
    return {
        "config": {
            "choice_path": str(choice_path),
        },
        "summary": summarize_revision_candidates([revision]),
        "predictions": [prediction.to_dict()],
        "revision_candidates": [revision.to_dict()],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def prediction_candidate_from_observation(
    observation: Mapping[str, Any],
) -> MechanicPredictionCandidate:
    observed_delta = dict(observation.get("observed_delta", {}) or {})
    game_id = str(observation.get("game_id", ""))
    action = str(observation.get("action", ""))
    mechanic_family = str(observation.get("mechanic_family", ""))
    predicted_metric = str(observation.get("predicted_metric", ""))
    return MechanicPredictionCandidate(
        candidate_id=str(observation.get("candidate_id", "")),
        game_id=game_id,
        mechanic_family=mechanic_family,
        action=action,
        predicted_metric=predicted_metric,
        expected_outcome=str(observation.get("expected_outcome", "")),
        observed_outcome=observed_outcome_from_delta(
            predicted_metric,
            observed_delta,
        ),
        observed_delta_summary=summarize_observed_delta(
            predicted_metric,
            observed_delta,
        ),
        key=mechanic_prediction_key(
            game_id,
            action,
            mechanic_family,
            predicted_metric,
        ),
    )


def revision_candidate_from_prediction(
    prediction: MechanicPredictionCandidate,
) -> MechanicRevisionCandidate:
    return MechanicRevisionCandidate(
        revision_candidate_id=f"revision::{prediction.candidate_id}",
        prediction=prediction,
    )


def observed_outcome_from_delta(
    predicted_metric: str,
    delta: Mapping[str, Any],
) -> str:
    metric = str(predicted_metric)
    if metric == "local_patch_before_after":
        changed = int(delta.get("local_changed_pixels", 0) or 0)
        return "local_patch_changed" if changed > 0 else "local_patch_unchanged"
    if metric == "object_counts_before_after":
        count_delta = int(delta.get("object_count_delta", 0) or 0)
        if count_delta > 0:
            return "object_count_increased"
        if count_delta < 0:
            return "object_count_decreased"
        return "object_count_preserved"
    if metric == "contact_graph_before_after":
        added = len(delta.get("contact_pairs_added", []) or [])
        removed = len(delta.get("contact_pairs_removed", []) or [])
        if added or removed:
            return "contact_graph_changed"
        return "contact_graph_preserved"
    if metric == "object_positions_before_after":
        moved = int(delta.get("moved_component_count", 0) or 0)
        return "objects_moved" if moved > 0 else "object_positions_preserved"
    if metric == "object_shape_zone_before_after":
        return (
            "shape_zone_changed"
            if bool(delta.get("shape_zone_changed"))
            else "shape_zone_preserved"
        )
    return "metric_observed" if bool(delta.get("changed")) else "metric_unchanged"


def summarize_observed_delta(
    predicted_metric: str,
    delta: Mapping[str, Any],
) -> Dict[str, Any]:
    result = {
        "metric": str(delta.get("metric", predicted_metric)),
        "changed": bool(delta.get("changed")),
        "changed_pixels": int(delta.get("changed_pixels", 0) or 0),
        "changed_cell_ratio": float(delta.get("changed_cell_ratio", 0.0) or 0.0),
    }
    metric = str(predicted_metric)
    if metric == "local_patch_before_after":
        result["local_changed_pixels"] = int(
            delta.get("local_changed_pixels", 0) or 0
        )
        result["patch_bbox"] = list(delta.get("patch_bbox", []) or [])
    elif metric == "object_counts_before_after":
        result["object_count_delta"] = int(delta.get("object_count_delta", 0) or 0)
        result["object_count_delta_by_color"] = dict(
            delta.get("object_count_delta_by_color", {}) or {}
        )
    elif metric == "contact_graph_before_after":
        result["contact_pairs_added"] = list(delta.get("contact_pairs_added", []) or [])
        result["contact_pairs_removed"] = list(
            delta.get("contact_pairs_removed", []) or []
        )
    elif metric == "object_positions_before_after":
        result["moved_component_count"] = int(
            delta.get("moved_component_count", 0) or 0
        )
    elif metric == "object_shape_zone_before_after":
        result["shape_zone_changed"] = bool(delta.get("shape_zone_changed"))
    return result


def summarize_revision_candidates(
    revisions: Sequence[MechanicRevisionCandidate],
) -> Dict[str, Any]:
    return {
        "mechanic_predictions": len(revisions),
        "revision_candidates": len(revisions),
        "a15_a31_revision_proposals": len(
            [revision for revision in revisions if revision.a15_a31_ready]
        ),
        "controlled_tests_required": len(
            [revision for revision in revisions if revision.prediction.controlled_test_required]
        ),
        "revision_performed": any(revision.revision_performed for revision in revisions),
        "observation_counted_as_confirmation": any(
            revision.observation_counted_as_confirmation for revision in revisions
        ),
        "wrong_confirmations": sum(revision.wrong_confirmations for revision in revisions),
    }


def mechanic_prediction_key(
    game_id: str,
    action: str,
    mechanic_family: str,
    predicted_metric: str,
) -> str:
    return "::".join(
        [
            "mechanic_prediction",
            str(game_id),
            str(action),
            str(mechanic_family),
            str(predicted_metric),
        ]
    )


def write_mechanic_revision_candidates(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3j mechanic observation to revision candidate.",
    )
    parser.add_argument(
        "--choice",
        type=Path,
        default=DEFAULT_RECOMMENDED_EXPERIMENT_CHOICE_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MECHANIC_REVISION_CANDIDATES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_mechanic_revision_candidate_generation(choice_path=args.choice)
    write_mechanic_revision_candidates(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
                "observation_counted_as_confirmation": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
