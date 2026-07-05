"""Mock world-model predictions for M2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .normalizer import allowed_actions_for_frontier, default_controls_for_action
from .schema import RawHypothesisProposal
from .world_model_adapter import (
    FrontierState,
    WorldModelActionPrediction,
    WorldModelHypothesisGenerator,
    frontier_state_from_request,
)


class MockWorldModel(WorldModelHypothesisGenerator):
    def score_candidate_actions(
        self,
        frontier_state: FrontierState,
        allowed_actions: list[str],
    ) -> list[WorldModelActionPrediction]:
        predictions = []
        for index, action in enumerate(allowed_actions):
            predictions.append(
                WorldModelActionPrediction(
                    candidate_action=action,
                    predicted_change_probability=0.72 - 0.05 * index,
                    predicted_local_signal_probability=0.41,
                    predicted_topology_change_probability=0.33,
                    predicted_object_count_change_probability=0.12,
                    predicted_contact_graph_change_probability=0.52,
                    predicted_observables={
                        "changed_pixels": {"mean": 12.4, "uncertainty": 0.31},
                        "local_patch_change": {"probability": 0.41},
                        "contact_graph_change": {"probability": 0.52},
                    },
                    uncertainty=0.28 + 0.03 * index,
                    epistemic_value=0.74 - 0.04 * index,
                    recommended_metric="contact_graph_before_after"
                    if index == 0
                    else "local_patch_before_after",
                )
            )
        return predictions


def build_mock_world_model_predictions_payload(
    frontier_requests: Sequence[Mapping[str, Any]],
    *,
    model: WorldModelHypothesisGenerator | None = None,
) -> Dict[str, Any]:
    scorer = model or MockWorldModel()
    batches = []
    total_predictions = 0
    for frontier in frontier_requests:
        actions = list(allowed_actions_for_frontier(frontier))
        state = frontier_state_from_request(frontier)
        predictions = scorer.score_candidate_actions(state, actions)
        total_predictions += len(predictions)
        batches.append(
            {
                "frontier_context_id": str(frontier.get("frontier_context_id", "")),
                "source_request_id": str(frontier.get("request_id", "")),
                "frontier_state": state.to_dict(),
                "predictions": [prediction.to_dict() for prediction in predictions],
            }
        )
    return {
        "config": {"schema_version": "m2.mock_world_model_predictions.v1"},
        "prediction_batches": batches,
        "summary": {
            "frontier_requests_consumed": len(frontier_requests),
            "world_model_predictions_loaded": total_predictions > 0,
            "world_model_predictions": total_predictions,
            "world_model_scores_counted_as_support": False,
            "wrong_confirmations": 0,
        },
    }


def write_mock_world_model_predictions(
    payload: Mapping[str, Any],
    output_path: str | Path = Path("diagnostics") / "m2" / "mock_world_model_predictions.json",
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_world_model_raw_proposals(
    frontier_request: Mapping[str, Any],
    *,
    model: WorldModelHypothesisGenerator | None = None,
) -> tuple[RawHypothesisProposal, ...]:
    actions = list(allowed_actions_for_frontier(frontier_request))
    state = frontier_state_from_request(frontier_request)
    predictions = (model or MockWorldModel()).score_candidate_actions(state, actions)
    proposals = []
    for index, prediction in enumerate(predictions[:3], start=1):
        proposals.append(
            world_model_prediction_to_raw_proposal(
                prediction,
                frontier_request,
                index=index,
            )
        )
    return tuple(proposals)


def world_model_prediction_to_raw_proposal(
    prediction: WorldModelActionPrediction,
    frontier_request: Mapping[str, Any],
    *,
    index: int,
) -> RawHypothesisProposal:
    action = prediction.candidate_action
    actions = allowed_actions_for_frontier(frontier_request)
    context = str(frontier_request.get("frontier_context_id", "unknown"))
    return RawHypothesisProposal(
        proposal_id=f"raw::{context}::world_model::{index:03d}",
        source="world_model",
        source_request_id=str(frontier_request.get("request_id", "")),
        game_id=str(frontier_request.get("game_id", "")),
        frontier_context_id=context,
        frontier_reason=str(frontier_request.get("reason", "")),
        frontier_step=_optional_int(frontier_request.get("source_step")),
        hypothesis_family="world_model_epistemic_probe",
        candidate_action=action,
        predicted_metric=prediction.recommended_metric,
        predicted_effect=(
            f"{action} has high predicted observable change under the mock world model"
        ),
        rationale=(
            "World-model score is used as priority only, not evidence or support."
        ),
        suggested_control_actions=default_controls_for_action(action, actions),
        required_context_replay=tuple(
            str(item) for item in frontier_request.get("context_signature", []) or []
        ),
        expected_signal_type="target_action_signal_exceeds_dynamic_control_signal",
        priority_hint=float(prediction.epistemic_value),
        world_model_uncertainty=float(prediction.uncertainty),
        raw_payload={"world_model_prediction": prediction.to_dict()},
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
