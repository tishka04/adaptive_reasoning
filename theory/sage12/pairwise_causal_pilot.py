"""SAGE12 V4.4 antisymmetric paired-causal binding pilot.

V4.4 reuses only the published V4.3 source counterfactual pairs.  It predicts
which arm of an identical-prestate intervention pair produces an effect,
rather than predicting an absolute arm outcome.  Pair features are explicit
left-minus-right differences and every learned model has no intercept, so
swapping complete arms exactly inverts its prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from theory.non_ar25_active_micro_run import _env_dir
from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION

from .bound_mechanic_pilot import (
    BindingPairRecord,
    BindingSignature,
    BoundEvent,
    _categorical_identity_probe,
    _collect_game,
    _collection_summary,
    _write_json,
    _write_jsonl,
    load_pairs,
    pair_windows,
)

FORMAT_VERSION = "sage12-paired-causal-v4.4"
MANIFEST_FORMAT_VERSION = "sage12-paired-causal-pilot-v4.4"
PREFLIGHT_FORMAT_VERSION = "sage12-paired-causal-preflight-v4.4"
RESULT_FORMAT_VERSION = "sage12-paired-causal-result-v4.4"
MODEL_FORMAT_VERSION = "sage12-paired-linear-model-v4.4"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "pairwise_causal_pilot_v4_4"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"
V43_OUTPUT_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"

AUTHORITATIVE_EFFECTS = ("target_created", "target_removed")
DIAGNOSTIC_EFFECTS = ("target_moved",)
PROJECTIONS = ("minimal", "relational", "typed")
MODEL_MODES = (
    "structured",
    "history_no_binding",
    "action_only",
    "binding_only",
    "template",
)
BASELINE_MODES = (
    "history_no_binding",
    "action_only",
    "binding_only",
    "template",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def load_frozen_manifest(
    path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported SAGE12 V4.4 manifest")
    expected = str(payload.get("manifest_checksum", ""))
    clean = dict(payload)
    clean.pop("manifest_checksum", None)
    if expected != _checksum(clean):
        raise ValueError("SAGE12 V4.4 manifest checksum mismatch")
    if tuple(payload["source_train_games"]) != SOURCE_TRAIN:
        raise ValueError("V4.4 source split drift")
    if tuple(payload["source_validation_games"]) != SOURCE_VALIDATION:
        raise ValueError("V4.4 validation split drift")
    return payload


@dataclass(frozen=True)
class PairEffectExample:
    """One source-only pair with audit provenance outside the model view."""

    pair_id: str
    game_id: str
    source_split: str
    context: tuple[BoundEvent, ...]
    left_action_name: str
    left_action_family: str
    left_binding: BindingSignature
    right_action_name: str
    right_action_family: str
    right_binding: BindingSignature
    outcomes: Mapping[str, tuple[bool, bool]]
    applicable: Mapping[str, bool]

    @classmethod
    def from_pair(cls, pair: BindingPairRecord) -> PairEffectExample:
        left, right = pair_windows(pair)
        return cls(
            pair_id=pair.pair_digest,
            game_id=pair.game_id,
            source_split=pair.source_split,
            context=pair.context,
            left_action_name=left.query_action_name,
            left_action_family=left.query_action_family,
            left_binding=left.query_binding,
            right_action_name=right.query_action_name,
            right_action_family=right.query_action_family,
            right_binding=right.query_binding,
            outcomes={
                effect: (
                    bool(left.labels[effect]),
                    bool(right.labels[effect]),
                )
                for effect in (*AUTHORITATIVE_EFFECTS, *DIAGNOSTIC_EFFECTS)
            },
            applicable={
                effect: bool(left.applicable[effect] and right.applicable[effect])
                for effect in (*AUTHORITATIVE_EFFECTS, *DIAGNOSTIC_EFFECTS)
            },
        )

    def is_discordant(self, effect: str) -> bool:
        left, right = self.outcomes[effect]
        return bool(self.applicable[effect] and left != right)

    def direction(self, effect: str) -> int:
        if not self.is_discordant(effect):
            raise ValueError("pair direction requires a discordant outcome")
        return int(self.outcomes[effect][0])

    def model_view(self, projection: str, mode: str) -> dict[str, float]:
        left = _arm_features(
            action_name=self.left_action_name,
            action_family=self.left_action_family,
            binding=self.left_binding,
            context=self.context,
            projection=projection,
            mode=mode,
        )
        right = _arm_features(
            action_name=self.right_action_name,
            action_family=self.right_action_family,
            binding=self.right_binding,
            context=self.context,
            projection=projection,
            mode=mode,
        )
        return _difference(left, right)

    def binding_swapped_view(self, projection: str, mode: str) -> dict[str, float]:
        left = _arm_features(
            action_name=self.left_action_name,
            action_family=self.left_action_family,
            binding=self.right_binding,
            context=self.context,
            projection=projection,
            mode=mode,
        )
        right = _arm_features(
            action_name=self.right_action_name,
            action_family=self.right_action_family,
            binding=self.left_binding,
            context=self.context,
            projection=projection,
            mode=mode,
        )
        return _difference(left, right)


def build_examples(
    pairs: Sequence[BindingPairRecord],
) -> list[PairEffectExample]:
    return [PairEffectExample.from_pair(pair) for pair in pairs]


def _binding_features(binding: BindingSignature, projection: str) -> dict[str, float]:
    return {
        f"binding:{key}={value}": 1.0
        for key, value in binding.model_view(projection).items()
    }


def _context_match_features(
    *,
    action_name: str,
    action_family: str,
    binding: BindingSignature,
    context: Sequence[BoundEvent],
    projection: str,
    include_binding: bool,
) -> dict[str, float]:
    features: dict[str, float] = {}
    matchers: list[tuple[str, Any]] = [
        ("exact_action", lambda event: event.action_name == action_name),
        ("action_family", lambda event: event.action_family == action_family),
    ]
    if include_binding:
        binding_key = binding.key(projection)
        minimal_key = binding.key("minimal")
        matchers.extend(
            [
                (
                    "projected_binding",
                    lambda event: event.binding.key(projection) == binding_key,
                ),
                (
                    "minimal_binding",
                    lambda event: event.binding.key("minimal") == minimal_key,
                ),
                (
                    "action_x_binding",
                    lambda event: (
                        event.action_name == action_name
                        and event.binding.key(projection) == binding_key
                    ),
                ),
            ]
        )
    for matcher_name, matcher in matchers:
        selected = [event for event in context if matcher(event)]
        features[f"history:{matcher_name}:support"] = len(selected) / max(
            1, len(context)
        )
        for effect in AUTHORITATIVE_EFFECTS:
            eligible = [event for event in selected if event.applicable[effect]]
            if eligible:
                rate = sum(int(event.effects[effect]) for event in eligible) / len(
                    eligible
                )
                signed = 2.0 * rate - 1.0
            else:
                signed = 0.0
            features[f"history:{matcher_name}:{effect}:signed_rate"] = signed
            features[f"history:{matcher_name}:{effect}:coverage"] = len(eligible) / max(
                1, len(context)
            )
    return features


def _arm_features(
    *,
    action_name: str,
    action_family: str,
    binding: BindingSignature,
    context: Sequence[BoundEvent],
    projection: str,
    mode: str,
) -> dict[str, float]:
    if mode == "template":
        return {}
    features: dict[str, float] = {}
    if mode in {"structured", "history_no_binding", "action_only"}:
        features[f"action:name={action_name}"] = 1.0
        features[f"action:family={action_family}"] = 1.0
    if mode in {"structured", "binding_only"}:
        features.update(_binding_features(binding, projection))
    if mode in {"structured", "history_no_binding"}:
        features.update(
            _context_match_features(
                action_name=action_name,
                action_family=action_family,
                binding=binding,
                context=context,
                projection=projection,
                include_binding=mode == "structured",
            )
        )
    if mode == "structured":
        for key, value in binding.model_view(projection).items():
            features[f"interaction:action={action_name}:binding:{key}={value}"] = 1.0
    return features


def _difference(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    result = {}
    for key in set(left) | set(right):
        value = float(left.get(key, 0.0)) - float(right.get(key, 0.0))
        if abs(value) > 1e-12:
            result[key] = value
    return result


def validate_model_view(example: PairEffectExample, projection: str, mode: str) -> None:
    rendered = _canonical(example.model_view(projection, mode)).lower()
    forbidden = (
        example.game_id.lower(),
        "game_id",
        "pair_id",
        "frame",
        "sha256",
        "object_id",
        "policy_seed",
        "reset_index",
        "root_index",
        '"row"',
        '"col"',
        '"x"',
        '"y"',
        "outcome",
        "label",
    )
    for token in forbidden:
        if token and token in rendered:
            raise ValueError(f"forbidden V4.4 model token: {token}")


@dataclass(frozen=True)
class AntisymmetricLinearModel:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.coefficients):
            raise ValueError("V4.4 feature/coefficient length mismatch")
        if self.temperature <= 0:
            raise ValueError("V4.4 temperature must be positive")

    def raw_logit(self, row: Mapping[str, float]) -> float:
        return float(
            sum(
                coefficient * float(row.get(name, 0.0))
                for name, coefficient in zip(self.feature_names, self.coefficients)
            )
        )

    def predict(self, row: Mapping[str, float]) -> float:
        value = np.clip(self.temperature * self.raw_logit(row), -50.0, 50.0)
        return float(1.0 / (1.0 + math.exp(-float(value))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "temperature": self.temperature,
            "fit_intercept": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AntisymmetricLinearModel:
        return cls(
            feature_names=tuple(
                str(value) for value in payload.get("feature_names", ())
            ),
            coefficients=tuple(
                float(value) for value in payload.get("coefficients", ())
            ),
            temperature=float(payload.get("temperature", 1.0)),
        )


def _fit_model(
    rows: Sequence[Mapping[str, float]], labels: Sequence[int]
) -> AntisymmetricLinearModel:
    if not rows:
        return AntisymmetricLinearModel((), ())
    augmented_rows = list(rows) + [
        {key: -float(value) for key, value in row.items()} for row in rows
    ]
    augmented_labels = list(labels) + [1 - int(value) for value in labels]
    vectorizer = DictVectorizer(sparse=True, sort=True)
    # The feature vocabulary is deliberately tiny. A dense matrix also avoids
    # platform-specific SciPy int64 sparse indices rejected by liblinear.
    matrix = vectorizer.fit_transform(augmented_rows).toarray()
    if matrix.shape[1] == 0 or len(set(augmented_labels)) < 2:
        return AntisymmetricLinearModel((), ())
    estimator = LogisticRegression(
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        class_weight="balanced",
        max_iter=1000,
        random_state=144,
    )
    estimator.fit(matrix, np.asarray(augmented_labels, dtype=int))
    return AntisymmetricLinearModel(
        feature_names=tuple(vectorizer.get_feature_names_out().tolist()),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
    )


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    if not len(logits) or np.allclose(logits, 0.0):
        return 1.0
    candidates = np.linspace(0.10, 5.0, 197)
    scores = []
    for temperature in candidates:
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(temperature * logits, -50.0, 50.0))
        )
        scores.append((float(np.mean((probabilities - labels) ** 2)), temperature))
    return float(min(scores)[1])


def _template_probability(example: PairEffectExample, effect: str) -> float:
    def score(binding: BindingSignature) -> float:
        if effect == "target_created":
            return 1.0 if binding.kind == "free_slot" else -1.0
        return 1.0 if binding.kind == "occupied_object" else -1.0

    delta = score(example.left_binding) - score(example.right_binding)
    return float(1.0 / (1.0 + math.exp(-delta)))


def _predict_rows(
    model: AntisymmetricLinearModel,
    examples: Sequence[PairEffectExample],
    *,
    projection: str,
    mode: str,
    binding_swap: bool,
) -> np.ndarray:
    if mode == "template":
        raise ValueError("template predictions require an effect")
    return np.asarray(
        [
            model.predict(
                example.binding_swapped_view(projection, mode)
                if binding_swap
                else example.model_view(projection, mode)
            )
            for example in examples
        ],
        dtype=np.float64,
    )


def _logo_predictions(
    examples: Sequence[PairEffectExample],
    projection: str,
    *,
    binding_swap: bool = False,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, AntisymmetricLinearModel]],
]:
    predictions = {
        mode: {
            effect: np.full(len(examples), np.nan, dtype=np.float64)
            for effect in AUTHORITATIVE_EFFECTS
        }
        for mode in MODEL_MODES
    }
    fold_models: dict[str, dict[str, AntisymmetricLinearModel]] = {}
    games = sorted({example.game_id for example in examples})
    for held_out in games:
        fold_models[held_out] = {}
        for effect in AUTHORITATIVE_EFFECTS:
            train = [
                example
                for example in examples
                if example.game_id != held_out and example.is_discordant(effect)
            ]
            test_indices = [
                index
                for index, example in enumerate(examples)
                if example.game_id == held_out and example.is_discordant(effect)
            ]
            test = [examples[index] for index in test_indices]
            if not test:
                continue
            for mode in MODEL_MODES:
                key = f"{effect}:{mode}"
                if mode == "template":
                    probabilities = np.asarray(
                        [_template_probability(example, effect) for example in test],
                        dtype=np.float64,
                    )
                    fold_models[held_out][key] = AntisymmetricLinearModel((), ())
                else:
                    rows = [example.model_view(projection, mode) for example in train]
                    labels = [example.direction(effect) for example in train]
                    model = _fit_model(rows, labels)
                    probabilities = _predict_rows(
                        model,
                        test,
                        projection=projection,
                        mode=mode,
                        binding_swap=binding_swap,
                    )
                    fold_models[held_out][key] = model
                predictions[mode][effect][test_indices] = probabilities
    return predictions, fold_models


def _calibrate_logo(
    examples: Sequence[PairEffectExample],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
    calibrated = {
        mode: {effect: values.copy() for effect, values in effects.items()}
        for mode, effects in predictions.items()
    }
    temperatures: dict[str, dict[str, float]] = {mode: {} for mode in MODEL_MODES}
    for mode in MODEL_MODES:
        for effect in AUTHORITATIVE_EFFECTS:
            eligible = np.asarray(
                [example.is_discordant(effect) for example in examples],
                dtype=bool,
            )
            labels = np.asarray(
                [
                    example.direction(effect)
                    for example in examples
                    if example.is_discordant(effect)
                ],
                dtype=np.float64,
            )
            raw = predictions[mode][effect][eligible]
            clipped = np.clip(raw, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped))
            temperature = _fit_temperature(logits, labels)
            temperatures[mode][effect] = temperature
            calibrated[mode][effect][eligible] = 1.0 / (
                1.0 + np.exp(-np.clip(temperature * logits, -50.0, 50.0))
            )
    return calibrated, temperatures


def _ece(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if not len(labels):
        return 0.0
    value = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(selected):
            value += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected]))
                - float(np.mean(labels[selected]))
            )
    return value


def pair_metrics(
    examples: Sequence[PairEffectExample],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    per_effect = {}
    all_labels = []
    all_probabilities = []
    for effect in AUTHORITATIVE_EFFECTS:
        eligible = np.asarray(
            [example.is_discordant(effect) for example in examples],
            dtype=bool,
        )
        labels = np.asarray(
            [
                example.direction(effect)
                for example in examples
                if example.is_discordant(effect)
            ],
            dtype=np.int8,
        )
        probabilities = predictions[effect][eligible]
        accuracy = float(np.mean((probabilities >= 0.5) == labels))
        per_effect[effect] = {
            "discordant_pairs": len(labels),
            "left_positive": int(np.sum(labels)),
            "right_positive": int(len(labels) - np.sum(labels)),
            "accuracy": accuracy,
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "ece": _ece(labels, probabilities),
        }
        all_labels.extend(labels.tolist())
        all_probabilities.extend(probabilities.tolist())
    labels_array = np.asarray(all_labels, dtype=np.int8)
    probability_array = np.asarray(all_probabilities, dtype=np.float64)
    return {
        "macro_accuracy": float(
            np.mean([item["accuracy"] for item in per_effect.values()])
        ),
        "macro_brier": float(np.mean([item["brier"] for item in per_effect.values()])),
        "macro_ece": float(np.mean([item["ece"] for item in per_effect.values()])),
        "micro_accuracy": float(np.mean((probability_array >= 0.5) == labels_array)),
        "per_effect": per_effect,
    }


def _brier_skill(model: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    denominator = float(baseline["macro_brier"])
    return (
        float((denominator - float(model["macro_brier"])) / denominator)
        if denominator > 0
        else 0.0
    )


def _capacity(examples: Sequence[PairEffectExample]) -> dict[str, Any]:
    effects = {}
    for effect in (*AUTHORITATIVE_EFFECTS, *DIAGNOSTIC_EFFECTS):
        per_game = {
            game: sum(
                example.game_id == game and example.is_discordant(effect)
                for example in examples
            )
            for game in sorted({example.game_id for example in examples})
        }
        effects[effect] = {
            "discordant_pairs": sum(per_game.values()),
            "games_with_at_least_10": sum(value >= 10 for value in per_game.values()),
            "per_game": per_game,
            "authority": effect in AUTHORITATIVE_EFFECTS,
        }
    return {"pairs": len(examples), "effects": effects}


def _identity_probe(
    examples: Sequence[PairEffectExample], projection: str
) -> dict[str, Any]:
    labels = [example.game_id for example in examples]
    action = _categorical_identity_probe(
        [example.model_view(projection, "action_only") for example in examples],
        labels,
    )
    structured = _categorical_identity_probe(
        [example.model_view(projection, "structured") for example in examples],
        labels,
    )
    return {
        "action_difference": action,
        "structured_difference": structured,
        "gain": float(structured["accuracy"] - action["accuracy"]),
    }


def _binding_swap_predictions(
    examples: Sequence[PairEffectExample],
    projection: str,
    fold_models: Mapping[str, Mapping[str, AntisymmetricLinearModel]],
    temperatures: Mapping[str, Mapping[str, float]],
) -> dict[str, np.ndarray]:
    result = {
        effect: np.full(len(examples), np.nan, dtype=np.float64)
        for effect in AUTHORITATIVE_EFFECTS
    }
    for index, example in enumerate(examples):
        for effect in AUTHORITATIVE_EFFECTS:
            if not example.is_discordant(effect):
                continue
            model = fold_models[example.game_id][f"{effect}:structured"]
            raw = model.predict(example.binding_swapped_view(projection, "structured"))
            clipped = float(np.clip(raw, 1e-6, 1 - 1e-6))
            logit = math.log(clipped / (1.0 - clipped))
            temperature = float(temperatures["structured"][effect])
            result[effect][index] = 1.0 / (
                1.0 + math.exp(-float(np.clip(temperature * logit, -50, 50)))
            )
    return result


def _arm_swap_error(
    examples: Sequence[PairEffectExample],
    projection: str,
    fold_models: Mapping[str, Mapping[str, AntisymmetricLinearModel]],
) -> float:
    errors = []
    for example in examples:
        for effect in AUTHORITATIVE_EFFECTS:
            if not example.is_discordant(effect):
                continue
            model = fold_models[example.game_id][f"{effect}:structured"]
            row = example.model_view(projection, "structured")
            swapped = {key: -value for key, value in row.items()}
            errors.append(abs(model.predict(swapped) - (1 - model.predict(row))))
    return max(errors, default=0.0)


def _per_game_transfer(
    examples: Sequence[PairEffectExample],
    structured: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result = {}
    for game in sorted({example.game_id for example in examples}):
        selected = [
            index
            for index, example in enumerate(examples)
            if example.game_id == game
            and any(example.is_discordant(effect) for effect in AUTHORITATIVE_EFFECTS)
        ]
        if not selected:
            result[game] = {"scoreable_pairs": 0, "status": "NOT_SCOREABLE"}
            continue
        correct_model = correct_baseline = total = 0
        for index in selected:
            example = examples[index]
            for effect in AUTHORITATIVE_EFFECTS:
                if not example.is_discordant(effect):
                    continue
                label = example.direction(effect)
                correct_model += int((structured[effect][index] >= 0.5) == label)
                correct_baseline += int((baseline[effect][index] >= 0.5) == label)
                total += 1
        result[game] = {
            "scoreable_pairs": total,
            "structured_accuracy": correct_model / total,
            "baseline_accuracy": correct_baseline / total,
            "accuracy_gain": (correct_model - correct_baseline) / total,
            "status": "SCORED",
        }
    return result


def _bootstrap_accuracy_gain(
    examples: Sequence[PairEffectExample],
    structured: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    contributions = []
    for index, example in enumerate(examples):
        model_correct = baseline_correct = total = 0
        for effect in AUTHORITATIVE_EFFECTS:
            if not example.is_discordant(effect):
                continue
            label = example.direction(effect)
            model_correct += int((structured[effect][index] >= 0.5) == label)
            baseline_correct += int((baseline[effect][index] >= 0.5) == label)
            total += 1
        if total:
            contributions.append((model_correct - baseline_correct) / total)
    rng = np.random.default_rng(seed)
    values = [
        float(np.mean(rng.choice(contributions, size=len(contributions), replace=True)))
        for _ in range(samples)
    ]
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _fit_final_models(
    examples: Sequence[PairEffectExample],
    projection: str,
    temperatures: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for mode in MODEL_MODES:
        models[mode] = {}
        for effect in AUTHORITATIVE_EFFECTS:
            if mode == "template":
                models[mode][effect] = {"template": True}
                continue
            eligible = [
                example for example in examples if example.is_discordant(effect)
            ]
            model = _fit_model(
                [example.model_view(projection, mode) for example in eligible],
                [example.direction(effect) for example in eligible],
            )
            calibrated = AntisymmetricLinearModel(
                model.feature_names,
                model.coefficients,
                temperature=float(temperatures[mode][effect]),
            )
            models[mode][effect] = calibrated.to_dict()
    payload: dict[str, Any] = {
        "format_version": MODEL_FORMAT_VERSION,
        "projection": projection,
        "effects": list(AUTHORITATIVE_EFFECTS),
        "models": models,
        "source_pairs": len(examples),
        "fit_intercept": False,
        "arm_swap_is_exact_inversion": True,
    }
    payload["model_checksum"] = _checksum(payload)
    return payload


def _projection_preflight(
    examples: Sequence[PairEffectExample],
    projection: str,
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, fold_models = _logo_predictions(examples, projection)
    calibrated, temperatures = _calibrate_logo(examples, raw)
    metrics = {mode: pair_metrics(examples, calibrated[mode]) for mode in MODEL_MODES}
    stronger = min(BASELINE_MODES, key=lambda mode: metrics[mode]["macro_brier"])
    skill = _brier_skill(metrics["structured"], metrics[stronger])
    accuracy_gain = (
        metrics["structured"]["macro_accuracy"] - metrics[stronger]["macro_accuracy"]
    )
    identity = _identity_probe(examples, projection)
    swapped = _binding_swap_predictions(examples, projection, fold_models, temperatures)
    swapped_metrics = pair_metrics(examples, swapped)
    swap_drop = (
        metrics["structured"]["macro_accuracy"] - swapped_metrics["macro_accuracy"]
    )
    arm_swap_error = _arm_swap_error(examples, projection, fold_models)
    per_game = _per_game_transfer(
        examples, calibrated["structured"], calibrated[stronger]
    )
    bootstrap = _bootstrap_accuracy_gain(
        examples,
        calibrated["structured"],
        calibrated[stronger],
        samples=int(frozen["evaluation"]["bootstrap_samples"]),
        seed=int(frozen["evaluation"]["random_seed"]),
    )
    gates_cfg = frozen["gates"]
    gates = {
        "minimum_macro_brier_skill": skill
        >= float(gates_cfg["minimum_macro_brier_skill"]),
        "minimum_macro_accuracy_gain": accuracy_gain
        >= float(gates_cfg["minimum_macro_accuracy_gain"]),
        "minimum_binding_swap_accuracy_drop": swap_drop
        >= float(gates_cfg["minimum_binding_swap_accuracy_drop"]),
        "maximum_identity_gain": identity["gain"]
        <= float(gates_cfg["maximum_identity_gain_over_action"]),
        "maximum_macro_ece": metrics["structured"]["macro_ece"]
        <= float(gates_cfg["maximum_macro_ece"]),
        "exact_arm_swap_inversion": arm_swap_error
        <= float(gates_cfg["maximum_arm_swap_error"]),
        "every_scoreable_game_nonnegative": all(
            item.get("accuracy_gain", 0.0) >= 0.0
            for item in per_game.values()
            if item["status"] == "SCORED"
        ),
        "bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
    }
    result = {
        "projection": projection,
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "metrics": metrics,
        "stronger_baseline": stronger,
        "macro_brier_skill": skill,
        "macro_accuracy_gain": accuracy_gain,
        "identity": identity,
        "binding_swap": {
            "metrics": swapped_metrics,
            "accuracy_drop": swap_drop,
        },
        "arm_swap_maximum_error": arm_swap_error,
        "per_game": per_game,
        "bootstrap_accuracy_gain": bootstrap,
        "gates": gates,
    }
    final_models = _fit_final_models(examples, projection, temperatures)
    return result, final_models


def run_source_preflight(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    pairs = load_pairs(
        V43_OUTPUT_DIR / "source_train_shards",
        tuple(frozen["source_train_games"]),
    )
    examples = build_examples(pairs)
    for projection in PROJECTIONS:
        for mode in MODEL_MODES:
            if mode != "template":
                for example in examples:
                    validate_model_view(example, projection, mode)
    capacity = _capacity(examples)
    gates_cfg = frozen["gates"]
    capacity_gates = {
        "minimum_discordant_pairs": all(
            capacity["effects"][effect]["discordant_pairs"]
            >= int(gates_cfg["minimum_source_discordant_pairs_per_effect"])
            for effect in AUTHORITATIVE_EFFECTS
        ),
        "minimum_games_with_capacity": all(
            capacity["effects"][effect]["games_with_at_least_10"]
            >= int(gates_cfg["minimum_source_games_with_10_discordant"])
            for effect in AUTHORITATIVE_EFFECTS
        ),
        "movement_remains_diagnostic": (
            capacity["effects"]["target_moved"]["authority"] is False
        ),
        "strict_json_validity": True,
        "support_zero": True,
        "source_v43_checksum_matches": (
            frozen["source_corpus"]["collection_report_checksum"]
            == json.loads(
                (V43_OUTPUT_DIR / "source_train_collection_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["report_checksum"]
        ),
    }
    projection_results = {}
    models = {}
    for projection in PROJECTIONS:
        result, model = _projection_preflight(examples, projection, frozen)
        projection_results[projection] = result
        models[projection] = model
    eligible = [
        projection
        for projection in PROJECTIONS
        if projection_results[projection]["status"] == "PASS"
    ]
    selected = None
    if eligible and all(capacity_gates.values()):
        best = max(
            projection_results[projection]["macro_brier_skill"]
            for projection in eligible
        )
        selected = next(
            projection
            for projection in PROJECTIONS
            if projection in eligible
            and best - projection_results[projection]["macro_brier_skill"]
            <= float(frozen["projection"]["simplicity_tie_margin"])
        )
    status = "PASS" if selected else "FAIL_CLOSED"
    payload: dict[str, Any] = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "status": status,
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "source_corpus_report_checksum": frozen["source_corpus"][
            "collection_report_checksum"
        ],
        "capacity": capacity,
        "capacity_gates": capacity_gates,
        "projection_results": projection_results,
        "selected_projection": selected,
        "validation_opened": False,
        "source_only": True,
        "movement_authority": False,
        "validation_collection_authorized": bool(selected),
        "world_model_fit_authorized": False,
        "ebm_fit_authorized": False,
        "controller_authorized": False,
    }
    payload["preflight_checksum"] = _checksum(payload)
    _write_json(destination / "source_preflight.json", payload)
    freeze = {
        "format_version": "sage12-paired-projection-freeze-v4.4",
        "status": status,
        "selected_projection": selected,
        "preflight_checksum": payload["preflight_checksum"],
        "frozen_before_validation_collection": True,
    }
    freeze["projection_freeze_checksum"] = _checksum(freeze)
    _write_json(destination / "projection_freeze.json", freeze)
    if selected:
        _write_json(destination / "source_model.json", models[selected])
    return payload


def run_validation_collection(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    preflight = json.loads(
        (destination / "source_preflight.json").read_text(encoding="utf-8")
    )
    if preflight.get("status") != "PASS" or not preflight.get(
        "validation_collection_authorized", False
    ):
        payload: dict[str, Any] = {
            "format_version": "sage12-paired-validation-collection-v4.4",
            "status": "SKIPPED_SOURCE_PREFLIGHT",
            "preflight_checksum": preflight.get("preflight_checksum"),
            "validation_opened": False,
        }
        payload["report_checksum"] = _checksum(payload)
        _write_json(destination / "validation_collection_manifest.json", payload)
        return payload
    config = frozen["validation_collection"]
    root = Path(environments_dir) if environments_dir else _env_dir()
    shard_dir = destination / "validation_shards"
    reports = {}
    for game in frozen["source_validation_games"]:
        pairs, report = _collect_game(
            game=game,
            source_split="source_validation",
            root_quota=int(config["roots_per_game"]),
            seeds=tuple(int(value) for value in config["seeds"]),
            action_budget=int(config["action_budget_per_reset"]),
            maximum_resets=int(config["maximum_resets_per_game"]),
            tree_depth=int(config["tree_depth"]),
            environment_root=root,
        )
        _write_jsonl(
            shard_dir / f"{game}.jsonl",
            (pair.to_dict() for pair in pairs),
        )
        reports[game] = report
    payload = _collection_summary(
        split="source_validation",
        games=tuple(frozen["source_validation_games"]),
        shard_dir=shard_dir,
        reports=reports,
        frozen=frozen,
    )
    _write_json(destination / "validation_collection_manifest.json", payload)
    return payload


def _load_model_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.get("model_checksum", ""))
    clean = dict(payload)
    clean.pop("model_checksum", None)
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("unsupported V4.4 model bundle")
    if expected != _checksum(clean):
        raise ValueError("V4.4 model checksum mismatch")
    return payload


def _bundle_predictions(
    examples: Sequence[PairEffectExample],
    bundle: Mapping[str, Any],
    *,
    binding_swap: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    projection = str(bundle["projection"])
    result = {
        mode: {
            effect: np.full(len(examples), np.nan, dtype=np.float64)
            for effect in AUTHORITATIVE_EFFECTS
        }
        for mode in MODEL_MODES
    }
    for mode in MODEL_MODES:
        for effect in AUTHORITATIVE_EFFECTS:
            for index, example in enumerate(examples):
                if not example.is_discordant(effect):
                    continue
                if mode == "template":
                    value = _template_probability(example, effect)
                else:
                    model = AntisymmetricLinearModel.from_dict(
                        bundle["models"][mode][effect]
                    )
                    view = (
                        example.binding_swapped_view(projection, mode)
                        if binding_swap
                        else example.model_view(projection, mode)
                    )
                    value = model.predict(view)
                result[mode][effect][index] = value
    return result


def run_validation_evaluation(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    preflight = json.loads(
        (destination / "source_preflight.json").read_text(encoding="utf-8")
    )
    if preflight.get("status") != "PASS":
        payload: dict[str, Any] = {
            "format_version": RESULT_FORMAT_VERSION,
            "status": "SKIPPED_SOURCE_PREFLIGHT",
            "preflight_checksum": preflight.get("preflight_checksum"),
            "validation_opened": False,
            "world_model_protocol_authorized": False,
            "ebm_fit_authorized": False,
            "controller_authorized": False,
        }
        payload["result_checksum"] = _checksum(payload)
        _write_json(destination / "pilot_result.json", payload)
        return payload
    bundle = _load_model_bundle(destination / "source_model.json")
    pairs = load_pairs(
        destination / "validation_shards",
        tuple(frozen["source_validation_games"]),
    )
    examples = build_examples(pairs)
    predictions = _bundle_predictions(examples, bundle)
    metrics = {mode: pair_metrics(examples, predictions[mode]) for mode in MODEL_MODES}
    projection = str(bundle["projection"])
    stronger = str(preflight["projection_results"][projection]["stronger_baseline"])
    skill = _brier_skill(metrics["structured"], metrics[stronger])
    accuracy_gain = (
        metrics["structured"]["macro_accuracy"] - metrics[stronger]["macro_accuracy"]
    )
    swapped = _bundle_predictions(examples, bundle, binding_swap=True)
    swapped_metrics = pair_metrics(examples, swapped["structured"])
    swap_drop = (
        metrics["structured"]["macro_accuracy"] - swapped_metrics["macro_accuracy"]
    )
    per_game = _per_game_transfer(
        examples, predictions["structured"], predictions[stronger]
    )
    capacity = _capacity(examples)
    gates_cfg = frozen["validation_gates"]
    gates = {
        "minimum_validation_discordant_pairs": all(
            capacity["effects"][effect]["discordant_pairs"]
            >= int(gates_cfg["minimum_discordant_pairs_per_effect"])
            for effect in AUTHORITATIVE_EFFECTS
        ),
        "minimum_macro_brier_skill": skill
        >= float(gates_cfg["minimum_macro_brier_skill"]),
        "minimum_macro_accuracy_gain": accuracy_gain
        >= float(gates_cfg["minimum_macro_accuracy_gain"]),
        "minimum_binding_swap_accuracy_drop": swap_drop
        >= float(gates_cfg["minimum_binding_swap_accuracy_drop"]),
        "maximum_macro_ece": metrics["structured"]["macro_ece"]
        <= float(gates_cfg["maximum_macro_ece"]),
        "every_scoreable_game_nonnegative": all(
            item.get("accuracy_gain", 0.0) >= 0.0
            for item in per_game.values()
            if item["status"] == "SCORED"
        ),
        "source_preflight_passed": preflight["status"] == "PASS",
    }
    passed = all(gates.values())
    rows = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "pair_id": example.pair_id,
                "game_id": example.game_id,
                "outcomes": example.outcomes,
                "applicable": example.applicable,
                "structured_probabilities": {
                    effect: (
                        float(predictions["structured"][effect][index])
                        if example.is_discordant(effect)
                        else None
                    )
                    for effect in AUTHORITATIVE_EFFECTS
                },
            }
        )
    _write_jsonl(destination / "validation_predictions.jsonl", rows)
    payload = {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "preflight_checksum": preflight["preflight_checksum"],
        "model_checksum": bundle["model_checksum"],
        "projection": projection,
        "capacity": capacity,
        "metrics": metrics,
        "stronger_baseline": stronger,
        "macro_brier_skill": skill,
        "macro_accuracy_gain": accuracy_gain,
        "binding_swap": {
            "metrics": swapped_metrics,
            "accuracy_drop": swap_drop,
        },
        "per_game": per_game,
        "gates": gates,
        "world_model_protocol_authorized": passed,
        "world_model_fit_authorized": False,
        "ebm_fit_authorized": False,
        "controller_authorized": False,
    }
    payload["result_checksum"] = _checksum(payload)
    _write_json(destination / "pilot_result.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "collect-validation", "evaluate"),
    )
    parser.add_argument("--frozen-manifest", default=str(DEFAULT_FROZEN_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environments-dir")
    args = parser.parse_args(argv)
    common = {
        "frozen_manifest_path": args.frozen_manifest,
        "output_dir": args.output_dir,
    }
    if args.command == "preflight":
        result = run_source_preflight(**common)
    elif args.command == "collect-validation":
        result = run_validation_collection(
            environments_dir=args.environments_dir, **common
        )
    else:
        result = run_validation_evaluation(**common)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITATIVE_EFFECTS",
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DIAGNOSTIC_EFFECTS",
    "AntisymmetricLinearModel",
    "PairEffectExample",
    "build_examples",
    "load_frozen_manifest",
    "main",
    "pair_metrics",
    "run_source_preflight",
    "run_validation_collection",
    "run_validation_evaluation",
    "validate_model_view",
]
