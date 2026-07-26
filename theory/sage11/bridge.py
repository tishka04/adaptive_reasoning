"""Typed bridge from the graph world model to candidate-only hypotheses."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .atoms import (
    HashAtomVocabulary,
    TypedAtom,
    action_features,
    observation_atoms,
)
from .authority import (
    NeuralActionCandidate,
    NeuralActionPrediction,
)
from .model import Sage11GraphWorldModel


class WorldModelActionPredictor:
    """Score legal actions without granting symbolic support or authority."""

    def __init__(
        self,
        model: Sage11GraphWorldModel,
        *,
        vocabulary: HashAtomVocabulary | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
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

    def __call__(
        self,
        observation: Any,
        candidates: Sequence[NeuralActionCandidate],
    ) -> Sequence[NeuralActionPrediction]:
        if not candidates:
            return ()
        atoms = observation_atoms(observation)
        encoded = self.vocabulary.encode(atoms)
        if not encoded:
            encoded = (0,)
        atom_ids = torch.tensor(
            [encoded for _ in candidates],
            dtype=torch.long,
            device=self.device,
        )
        atom_mask = atom_ids != 0
        actions = torch.tensor(
            [
                action_features(
                    candidate.action_name,
                    candidate.action_data,
                )
                for candidate in candidates
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            output = self.model(atom_ids, atom_mask, actions)
            effect_probabilities = torch.softmax(
                output["effect_logits"],
                dim=-1,
            )
            effect_confidence, effect_classes = (
                effect_probabilities.max(dim=-1)
            )
            progress = torch.sigmoid(output["progress_logit"])
            risk = torch.sigmoid(output["risk_logit"])
            noop = torch.sigmoid(output["noop_logit"])
            changed = torch.sigmoid(output["changed_logit"])
            information = (
                output["next_latent_variance"].mean(dim=-1)
                + output["effect_logits_variance"].mean(dim=-1)
            )
        predictions = []
        for index, candidate in enumerate(candidates):
            hypotheses = [
                TypedAtom(
                    "effect",
                    "class",
                    (str(int(effect_classes[index].item())),),
                )
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
                    changed[index].item()
                    * effect_confidence[index].item()
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


__all__ = ["WorldModelActionPredictor"]
