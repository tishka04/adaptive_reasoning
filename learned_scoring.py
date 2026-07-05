"""Shared learned scoring used by the passive diagnostics and the active search.

The learned score for ``(state, action)`` is built without stepping the env:

    1. Model A predicts the action's effect (a few deltas).
    2. Those deltas are applied to the current feature vector to *synthesize* a
       predicted next state.
    3. Model B scores the predicted next state (progress up, danger down).

A matching ``handmade_score`` (the generic heuristic teacher) is provided so
the two can be compared on identical branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from joblib import load

from abstraction_dataset_io import (
    ACTION_EFFECT_TARGETS,
    ACTION_INDEX,
    ACTIONS,
    HISTORY_FEATURE_SCHEMA,
    history_feature_vector,
    largest_component_feature_vector,
)
from extract_state_abstractions import FEATURE_SCHEMA, LARGEST_COMPONENT_FEATURE_SCHEMA

# delta_<name> target -> base feature name it perturbs.
_TARGET_TO_FEATURE = {t: t[len("delta_") :] for t in ACTION_EFFECT_TARGETS}
_FEATURE_POS = {name: i for i, name in enumerate(FEATURE_SCHEMA)}


@dataclass
class ActionScore:
    action: str
    learned_score: float
    general_score: float
    handmade_score: float
    predicted_progress: float
    predicted_danger: float
    predicted_delta: Dict[str, float]
    predicted_macro_scores: Dict[str, float]
    macro_bonus: float
    predicted_break_probability: float
    break_bonus: float
    break_bonus_applied: bool


class LearnedScorer:
    def __init__(
        self,
        action_effect_path: str,
        value_path: str,
        *,
        macro_scores_path: Optional[str] = None,
        break_classifier_path: Optional[str] = None,
        progress_weight: float = 1.0,
        correspondence_weight: float = 0.1,
        danger_weight: float = 1.0,
        macro_score_weight: float = 0.05,
        macro_align_weight: float = 1.0,
        macro_correspond_weight: float = 1.0,
        macro_break_weight: float = 0.5,
        macro_explore_penalty: float = 0.5,
        macro_avoid_penalty: float = 1.0,
        break_bonus_weight: float = 2.0,
        break_noop_threshold: float = 0.7,
        break_danger_threshold: float = 0.25,
        action6_break_threshold: float = 0.85,
    ) -> None:
        self.effect = load(action_effect_path)
        self.value = load(value_path)
        self.macro_scores = load(macro_scores_path) if macro_scores_path else None
        self.break_classifier = load(break_classifier_path) if break_classifier_path else None
        self.effect_model = self.effect["model"]
        self.effect_targets: List[str] = self.effect["target_names"]
        self.effect_input_names: List[str] = self.effect.get(
            "input_feature_names",
            list(FEATURE_SCHEMA) + [f"is_{action}" for action in ACTIONS],
        )
        self.macro_model = self.macro_scores["model"] if self.macro_scores else None
        self.macro_targets: List[str] = (
            list(self.macro_scores.get("target_names", [])) if self.macro_scores else []
        )
        self.macro_input_names: List[str] = (
            list(self.macro_scores.get("input_feature_names", [])) if self.macro_scores else []
        )
        self.break_model = self.break_classifier["model"] if self.break_classifier else None
        self.break_input_names: List[str] = (
            list(self.break_classifier.get("input_feature_names", [])) if self.break_classifier else []
        )
        self.progress_reg = self.value["progress_regressor"]
        self.level_clf = self.value["level_up_classifier"]
        self.danger_clf = self.value["danger_classifier"]
        self.progress_weight = progress_weight
        self.correspondence_weight = correspondence_weight
        self.danger_weight = danger_weight
        self.macro_score_weight = macro_score_weight
        self.macro_align_weight = macro_align_weight
        self.macro_correspond_weight = macro_correspond_weight
        self.macro_break_weight = macro_break_weight
        self.macro_explore_penalty = macro_explore_penalty
        self.macro_avoid_penalty = macro_avoid_penalty
        self.break_bonus_weight = break_bonus_weight
        self.break_noop_threshold = break_noop_threshold
        self.break_danger_threshold = break_danger_threshold
        self.action6_break_threshold = action6_break_threshold

    # -- prediction helpers --------------------------------------------------
    def _state_vector(self, feats: Dict[str, float]) -> np.ndarray:
        return np.array([float(feats.get(n, 0.0)) for n in FEATURE_SCHEMA], dtype=np.float32)

    def _action_onehot(self, action: str) -> np.ndarray:
        vec = np.zeros(len(ACTIONS), dtype=np.float32)
        idx = ACTION_INDEX.get(action)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _input_values(
        self,
        feats: Dict[str, float],
        action: str,
        history_features: Mapping[str, Any] | None,
        largest_component_features: Mapping[str, float] | None,
    ) -> Dict[str, float]:
        values: Dict[str, float] = {name: float(feats.get(name, 0.0)) for name in FEATURE_SCHEMA}
        values.update(
            {
                name: float(value)
                for name, value in zip(
                    LARGEST_COMPONENT_FEATURE_SCHEMA,
                    largest_component_feature_vector(largest_component_features),
                )
            }
        )
        values.update(
            {
                name: float(value)
                for name, value in zip(
                    HISTORY_FEATURE_SCHEMA,
                    history_feature_vector(history_features),
                )
            }
        )
        values.update(
            {
                f"is_{candidate}": float(value)
                for candidate, value in zip(ACTIONS, self._action_onehot(action))
            }
        )
        return values

    def _input_for_names(
        self,
        names: List[str],
        feats: Dict[str, float],
        action: str,
        history_features: Mapping[str, Any] | None,
        largest_component_features: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        values = self._input_values(feats, action, history_features, largest_component_features)
        return np.array([float(values.get(name, 0.0)) for name in names], dtype=np.float32)

    def _effect_input(
        self,
        feats: Dict[str, float],
        action: str,
        history_features: Mapping[str, Any] | None,
    ) -> np.ndarray:
        return self._input_for_names(
            self.effect_input_names,
            feats,
            action,
            history_features,
        )

    def predict_delta(
        self,
        feats: Dict[str, float],
        action: str,
        history_features: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        x = self._effect_input(feats, action, history_features)[None, :]
        pred = self.effect_model.predict(x)[0]
        return {name: float(pred[i]) for i, name in enumerate(self.effect_targets)}

    def synthesize_next_vector(self, feats: Dict[str, float], delta: Dict[str, float]) -> np.ndarray:
        vec = self._state_vector(feats).copy()
        for target, value in delta.items():
            base = _TARGET_TO_FEATURE.get(target)
            if base is not None and base in _FEATURE_POS:
                vec[_FEATURE_POS[base]] += float(value)
        return vec

    def _danger_proba(self, x: np.ndarray) -> float:
        clf = self.danger_clf
        if isinstance(clf, dict):
            return float(clf.get("constant", 0.0))
        return float(clf.predict_proba(x)[0, 1])

    def _macro_scores(
        self,
        feats: Dict[str, float],
        action: str,
        history_features: Mapping[str, Any] | None,
    ) -> Dict[str, float]:
        if self.macro_model is None or not self.macro_input_names:
            return {}
        x = self._input_for_names(self.macro_input_names, feats, action, history_features)[None, :]
        pred = self.macro_model.predict(x)[0]
        return {
            name: max(0.0, float(pred[i]))
            for i, name in enumerate(self.macro_targets)
        }

    def _macro_bonus(self, scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        raw = (
            self.macro_align_weight * float(scores.get("align", 0.0))
            + self.macro_correspond_weight * float(scores.get("correspond", 0.0))
            + self.macro_break_weight * float(scores.get("break", 0.0))
            - self.macro_explore_penalty * float(scores.get("explore", 0.0))
            - self.macro_avoid_penalty * float(scores.get("avoid", 0.0))
        )
        return float(self.macro_score_weight * raw)

    def _break_probability(
        self,
        feats: Dict[str, float],
        action: str,
        history_features: Mapping[str, Any] | None,
        largest_component_features: Mapping[str, float] | None,
    ) -> float:
        if self.break_model is None or not self.break_input_names:
            return 0.0
        x = self._input_for_names(
            self.break_input_names,
            feats,
            action,
            history_features,
            largest_component_features,
        )[None, :]
        classes = list(getattr(self.break_model, "classes_", []))
        if 1 not in classes:
            return 0.0
        class_index = classes.index(1)
        return float(self.break_model.predict_proba(x)[0, class_index])

    def _break_bonus(
        self,
        *,
        action: str,
        break_probability: float,
        predicted_danger: float,
        predicted_macro_scores: Dict[str, float],
    ) -> tuple[float, bool]:
        if break_probability <= 0.0:
            return 0.0, False
        no_op_prob = float(predicted_macro_scores.get("explore", 0.0))
        if predicted_danger > self.break_danger_threshold:
            return 0.0, False
        if no_op_prob > self.break_noop_threshold:
            return 0.0, False
        if action == "ACTION6" and break_probability < self.action6_break_threshold:
            return 0.0, False
        return float(self.break_bonus_weight * break_probability), True

    # -- scoring -------------------------------------------------------------
    def score_action(
        self,
        feats: Dict[str, float],
        action: str,
        history_features: Mapping[str, Any] | None = None,
        largest_component_features: Mapping[str, float] | None = None,
    ) -> ActionScore:
        delta = self.predict_delta(feats, action, history_features)
        next_vec = self.synthesize_next_vector(feats, delta)[None, :]
        progress = float(self.progress_reg.predict(next_vec)[0])
        danger = self._danger_proba(next_vec)
        corr = float(delta.get("delta_top_pair_0_global_correspondence", 0.0))
        general = (
            self.progress_weight * progress
            + self.correspondence_weight * corr
            - self.danger_weight * danger
        )
        macro_scores = self._macro_scores(feats, action, history_features)
        macro_bonus = self._macro_bonus(macro_scores)
        break_probability = self._break_probability(
            feats,
            action,
            history_features,
            largest_component_features,
        )
        break_bonus, break_applied = self._break_bonus(
            action=action,
            break_probability=break_probability,
            predicted_danger=danger,
            predicted_macro_scores=macro_scores,
        )
        learned = general + macro_bonus + break_bonus
        handmade = handmade_score(feats, delta)
        return ActionScore(
            action=action,
            learned_score=float(learned),
            general_score=float(general),
            handmade_score=float(handmade),
            predicted_progress=progress,
            predicted_danger=danger,
            predicted_delta=delta,
            predicted_macro_scores=macro_scores,
            macro_bonus=float(macro_bonus),
            predicted_break_probability=float(break_probability),
            break_bonus=float(break_bonus),
            break_bonus_applied=break_applied,
        )

    def score_actions(
        self,
        feats: Dict[str, float],
        actions: List[str],
        history_features: Mapping[str, Any] | None = None,
        largest_component_features_by_action: Mapping[str, Mapping[str, float]] | None = None,
    ) -> List[ActionScore]:
        return [
            self.score_action(
                feats,
                action,
                history_features,
                (
                    largest_component_features_by_action.get(action)
                    if largest_component_features_by_action
                    else None
                ),
            )
            for action in actions
        ]


def handmade_score(feats: Dict[str, float], delta: Dict[str, float]) -> float:
    """Generic heuristic teacher score over a predicted/observed delta.

    Mirrors the teacher used in build_abstraction_dataset._heuristic_policy:
    reward rising correspondence and structural change, lightly reward breaking
    the largest component apart. No ar25-specific colors.
    """

    d_corr = float(delta.get("delta_top_pair_0_global_correspondence", 0.0))
    d_break = -float(delta.get("delta_largest_component_size", 0.0))
    d_count = float(delta.get("delta_component_count", 0.0))
    return 2.0 * d_corr + 0.05 * max(0.0, d_break) + 0.1 * abs(d_count)


def blended_score(handmade: float, learned: float, weight: float) -> float:
    """(1-w)*handmade + w*learned, for progressive replacement."""

    weight = max(0.0, min(1.0, float(weight)))
    return (1.0 - weight) * handmade + weight * learned
