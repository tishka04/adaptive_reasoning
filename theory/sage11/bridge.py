"""Typed bridge from the graph world model to candidate-only hypotheses."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .atoms import (
    HashAtomVocabulary,
    TypedAtom,
    frame_diff_atoms,
    observation_atoms,
)
from .authority import (
    NeuralActionCandidate,
    NeuralActionPrediction,
)
from .dataset import state_digest
from .model import Sage11GraphWorldModel
from .streaming_features import (
    CHANGED_BUCKETS,
    StreamingFeatureSchema,
    StreamingFeatureTracker,
    StreamingStepContext,
)


class WorldModelActionPredictor:
    """Score legal actions without granting symbolic support or authority."""

    def __init__(
        self,
        model: Sage11GraphWorldModel,
        *,
        feature_schema: StreamingFeatureSchema,
        vocabulary: HashAtomVocabulary | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.feature_schema = feature_schema
        if model.config.streaming_features != feature_schema.feature_count:
            raise ValueError(
                "model and live streaming feature widths do not match"
            )
        expected_schema = model.config.streaming_feature_schema_checksum
        if expected_schema and expected_schema != feature_schema.checksum:
            raise ValueError(
                "model and live streaming feature schema checksums differ"
            )
        self.feature_tracker = StreamingFeatureTracker(feature_schema)
        self.vocabulary = vocabulary or HashAtomVocabulary(
            model.config.vocab_size
        )
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.model.to(self.device)
        self.model.eval()
        self._reset_index = 0
        self._step_index = 0
        self._pending_context: StreamingStepContext | None = None

    @property
    def _sequence_key(self) -> tuple[str, int]:
        return ("live", self._reset_index)

    def __call__(
        self,
        observation: Any,
        candidates: Sequence[NeuralActionCandidate],
    ) -> Sequence[NeuralActionPrediction]:
        if not candidates:
            return ()
        atoms = observation_atoms(observation)
        atom_keys = tuple(atom.key for atom in atoms)
        digest = state_digest(
            observation.raw_grid,
            game_state=observation.game_state,
            levels_completed=observation.levels_completed,
        )
        context = self.feature_tracker.begin_step(
            sequence_key=self._sequence_key,
            step_index=self._step_index,
            atoms_before=atom_keys,
            state_digest_before=digest,
        )
        self._pending_context = context
        encoded = self.vocabulary.encode(atoms)
        if not encoded:
            encoded = (0,)
        atom_ids = torch.tensor(
            [encoded for _ in candidates],
            dtype=torch.long,
            device=self.device,
        )
        atom_mask = atom_ids != 0
        streaming_features = torch.from_numpy(np.stack(
            [
                self.feature_tracker.encode_action(
                    context,
                    action_name=candidate.action_name,
                    action_data=candidate.action_data,
                )
                for candidate in candidates
            ]
        )).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(
                atom_ids,
                atom_mask,
                streaming_features,
            )
            changed_probabilities = torch.softmax(
                output["changed_cells_logits"],
                dim=-1,
            )
            changed_confidence, changed_classes = (
                changed_probabilities.max(dim=-1)
            )
            player_moved = torch.sigmoid(output["player_moved_logit"])
            progress = torch.sigmoid(output["progress_logit"])
            risk = torch.sigmoid(output["risk_logit"])
            noop = torch.sigmoid(output["noop_logit"])
            information = (
                output["next_latent_variance"].mean(dim=-1)
                + output["changed_cells_logits_variance"].mean(dim=-1)
                + output["player_moved_logit_variance"]
            )
        predictions = []
        for index, candidate in enumerate(candidates):
            hypotheses = [
                TypedAtom(
                    "effect",
                    "changed_cells",
                    (
                        CHANGED_BUCKETS[
                            int(changed_classes[index].item())
                        ],
                    ),
                ),
                TypedAtom(
                    "effect",
                    "player_moved",
                    (
                        str(
                            float(player_moved[index].item()) >= 0.5
                        ),
                    ),
                ),
            ]
            if float(progress[index].item()) >= 0.5:
                hypotheses.append(TypedAtom(
                    "progress",
                    "candidate_progress",
                ))
            if float(risk[index].item()) >= 0.5:
                hypotheses.append(TypedAtom(
                    "risk",
                    "candidate_unsafe",
                ))
            predictions.append(NeuralActionPrediction(
                action_name=candidate.action_name,
                action_data=dict(candidate.action_data),
                predicted_progress=float(progress[index].item()),
                predicted_effect=float(
                    (
                        1.0
                        - changed_probabilities[index, 0].item()
                    )
                    * changed_confidence[index].item()
                ),
                predicted_information_gain=float(
                    information[index].item()
                ),
                predicted_risk=float(risk[index].item()),
                predicted_noop=float(noop[index].item()),
                epistemic_variance=float(
                    output["progress_logit_variance"][index].item()
                ),
                hypotheses=tuple(hypotheses),
            ))
        return tuple(predictions)

    def observe_transition(self, record: Any) -> None:
        """Advance live streaming state from one observed transition."""
        context = self._pending_context
        self._pending_context = None
        if context is None:
            return
        action = record.action
        action_data = {
            key: value
            for key, value in {
                "x": getattr(action, "x", None),
                "y": getattr(action, "y", None),
            }.items()
            if value is not None
        }
        atoms_after = tuple(
            atom.key
            for atom in observation_atoms(record.obs_after)
        )
        digest_after = state_digest(
            record.obs_after.raw_grid,
            game_state=record.obs_after.game_state,
            levels_completed=record.obs_after.levels_completed,
        )
        effects = tuple(
            atom.key
            for atom in frame_diff_atoms(record.diff)
        )
        self.feature_tracker.observe_transition(
            context,
            action_name=str(action.name),
            action_data=action_data,
            atoms_after=atoms_after,
            state_digest_after=digest_after,
            effect_atoms=effects,
        )
        self._step_index += 1

    def on_reset(self) -> None:
        """Start a new live feature sequence without retaining target state."""
        self.feature_tracker.reset(self._sequence_key)
        self._pending_context = None
        self._reset_index += 1
        self._step_index = 0


__all__ = ["WorldModelActionPredictor"]
