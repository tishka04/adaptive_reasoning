"""Constrained, game-invariant SAGE12 proposal pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .compiler import HypothesisCompiler
from .controller import SemanticActionCandidate
from .hypotheses import hypotheses_from_json
from .proposal_pilot_data import (
    ProposalPilotTrace,
    graph_from_mapping,
    read_trace_shard,
)


FORMAT_VERSION = "sage12-constrained-effect-pilot-v2"
PREFLIGHT_FORMAT_VERSION = "sage12-invariant-motif-preflight-v2"
RESULT_FORMAT_VERSION = "sage12-constrained-effect-result-v2"
PREDICTION_FORMAT_VERSION = "sage12-constrained-prediction-v2"
DEFAULT_INPUT_DIR = Path("training") / "sage12" / "proposal_pilot_v1"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "constrained_pilot_v2"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"
EFFECT_LABELS = (
    "changed",
    "player_moved",
    "level_complete",
    "game_over",
)
PRIMARY_EFFECT_LABELS = ("changed", "player_moved")
_EXPANDED_MOTIF_KEYS = (
    "actor_present",
    "actor_touching_other",
    "actor_near_other",
    "actor_aligned_other",
    "actor_vertical_pair",
    "actor_horizontal_pair",
)
MOTIF_KEYS = ("actor_interaction",)
_RELATION_FAMILIES = {
    "contact": "actor_touching_other",
    "adjacent": "actor_touching_other",
    "near": "actor_near_other",
    "aligned": "actor_aligned_other",
    "north_of": "actor_vertical_pair",
    "south_of": "actor_vertical_pair",
    "east_of": "actor_horizontal_pair",
    "west_of": "actor_horizontal_pair",
}


def invariant_motif(
    trace: ProposalPilotTrace,
    *,
    variant: str = "original",
) -> dict[str, int]:
    """Return one binary local interaction bit without scene signatures."""
    expanded = _expanded_motif(trace, variant=variant)
    return {
        "actor_interaction": int(
            expanded["actor_touching_other"]
            or expanded["actor_near_other"]
        )
    }


def _expanded_motif(
    trace: ProposalPilotTrace,
    *,
    variant: str = "original",
) -> dict[str, int]:
    """Return source-train-only design features for leakage ablations."""
    if variant not in {"original", "relation_shuffle"}:
        raise ValueError(f"unsupported motif variant: {variant}")
    graph = (
        trace.scene_graph
        if variant == "original"
        else trace.relation_shuffle_graph
    )
    entities = {
        str(item["entity_id"]): tuple(str(role) for role in item["roles"])
        for item in graph.get("entities", ())
    }
    actors = {
        entity_id
        for entity_id, roles in entities.items()
        if "player" in roles
    }
    motif = {key: 0 for key in _EXPANDED_MOTIF_KEYS}
    motif["actor_present"] = int(bool(actors))
    if not actors:
        return motif
    for relation in graph.get("relations", ()):
        subject = str(relation["subject_id"])
        target = str(relation["object_id"])
        connects_actor = (
            subject in actors and target not in actors
        ) or (
            target in actors and subject not in actors
        )
        if not connects_actor:
            continue
        family = _RELATION_FAMILIES.get(str(relation["kind"]))
        if family is not None:
            motif[family] = 1
    return motif


def invariant_prompt(
    trace: ProposalPilotTrace,
    *,
    variant: str = "original",
) -> str:
    """Serialize only the frozen selected action and binary local motif."""
    motif = invariant_motif(trace, variant=variant)
    state = " ".join(
        f"{key}={'yes' if motif[key] else 'no'}"
        for key in MOTIF_KEYS
    )
    return (
        "Rank typed one-step causal effects for an already selected legal "
        "action. Use only the normalized local motif; do not infer a game, "
        "layout, object inventory, or hidden rule. "
        f"action={trace.selected_action_name} {state}. "
        "The effect vocabulary is changed, player_moved, level_complete, "
        "game_over."
    )


def render_hypotheses_json(
    trace: ProposalPilotTrace,
    predicted: Mapping[str, bool],
) -> str:
    """Render classifier decisions as strict support-zero typed JSON."""
    hypotheses = []
    for index, label in enumerate(EFFECT_LABELS):
        if not bool(predicted.get(label, False)):
            continue
        predicate: dict[str, Any]
        if label == "player_moved":
            predicate = {
                "name": "moved",
                "subject": {"role": "player", "selector": "any"},
            }
        else:
            predicate = {"name": label}
        hypotheses.append(
            {
                "hypothesis_id": f"constrained_{index}",
                "action_name": trace.selected_action_name,
                "action_data": dict(trace.selected_action_data),
                "preconditions": [],
                "effects": [
                    {
                        "predicate": predicate,
                        "operation": "assert",
                    }
                ],
                "confidence": 1.0,
                "rationale": "",
                "source": "qwen_linear_head",
                "support": 0,
            }
        )
    return json.dumps(
        {"hypotheses": hypotheses},
        sort_keys=True,
        separators=(",", ":"),
    )


class FrozenQwenEmbedder:
    """Extract last-token embeddings from the frozen local Qwen model."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        batch_size: int,
        maximum_input_tokens: int,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.maximum_input_tokens = max(32, int(maximum_input_tokens))
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def encode(
        self,
        prompts: Sequence[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import torch

        tokenizer, model = self._load()
        vectors = []
        input_lengths = []
        batch_seconds = []
        for offset in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[offset : offset + self.batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Encode the supplied abstract causal "
                                "classification problem."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch_prompts
            ]
            inputs = tokenizer(
                rendered,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            maximum = int(lengths.max().item())
            if maximum > self.maximum_input_tokens:
                raise RuntimeError(
                    "constrained pilot prompt exceeds frozen token cap: "
                    f"{maximum} > {self.maximum_input_tokens}"
                )
            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }
            if str(model.device).startswith("cuda"):
                torch.cuda.synchronize(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
            if str(model.device).startswith("cuda"):
                torch.cuda.synchronize(model.device)
            batch_seconds.append(time.perf_counter() - started)
            hidden = output.last_hidden_state
            indices = lengths.to(hidden.device) - 1
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[rows, indices]
            vectors.append(pooled.float().cpu().numpy())
            input_lengths.extend(int(value) for value in lengths.tolist())
        matrix = np.concatenate(vectors, axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        return matrix.astype(np.float32), {
            "rows": len(prompts),
            "batches": len(batch_seconds),
            "batch_size": self.batch_size,
            "inference_seconds": sum(batch_seconds),
            "median_batch_seconds": statistics.median(batch_seconds),
            "maximum_input_tokens": max(input_lengths),
            "mean_input_tokens": statistics.fmean(input_lengths),
            "device": str(model.device),
        }

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map={"": self.device},
            local_files_only=True,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model


def run_source_train_preflight(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Audit motif leakage and label capacity without opening validation."""
    input_path = Path(input_dir)
    destination = Path(output_dir)
    collection = _read_json(input_path / "collection_manifest.json")
    train = _load_traces(
        collection,
        input_dir=input_path,
        split="source_train",
    )
    motifs = [invariant_motif(trace) for trace in train]
    expanded_motifs = [_expanded_motif(trace) for trace in train]
    motif_probe = _identity_probe(
        motifs,
        [trace.game_id for trace in train],
    )
    action_rows = [
        {f"action:{trace.selected_action_name}": 1}
        for trace in train
    ]
    action_probe = _identity_probe(
        action_rows,
        [trace.game_id for trace in train],
    )
    combined_rows = [
        {
            **motif,
            f"action:{trace.selected_action_name}": 1,
        }
        for trace, motif in zip(train, motifs)
    ]
    combined_probe = _identity_probe(
        combined_rows,
        [trace.game_id for trace in train],
    )
    candidate_views = {
        "actor_present_only": lambda motif: {
            "actor_present": motif["actor_present"]
        },
        "touch_only": lambda motif: {
            "actor_touching_other": motif["actor_touching_other"]
        },
        "near_only": lambda motif: {
            "actor_near_other": motif["actor_near_other"]
        },
        "interaction_any": lambda motif: {
            "actor_interaction": int(
                motif["actor_touching_other"]
                or motif["actor_near_other"]
            )
        },
        "alignment_only": lambda motif: {
            "actor_aligned_other": motif["actor_aligned_other"]
        },
        "axis_any": lambda motif: {
            "actor_axis_pair": int(
                motif["actor_vertical_pair"]
                or motif["actor_horizontal_pair"]
            )
        },
        "interaction_and_alignment": lambda motif: {
            "actor_interaction": int(
                motif["actor_touching_other"]
                or motif["actor_near_other"]
            ),
            "actor_aligned_other": motif["actor_aligned_other"],
        },
    }
    candidate_diagnostics = {}
    labels = [trace.game_id for trace in train]
    for name, project in candidate_views.items():
        state_rows = [project(motif) for motif in expanded_motifs]
        state_probe = _identity_probe(state_rows, labels)
        conditioned_probe = _identity_probe(
            [
                {**state, **action}
                for state, action in zip(state_rows, action_rows)
            ],
            labels,
        )
        candidate_diagnostics[name] = {
            "state_only": state_probe,
            "selected_action_plus_state": conditioned_probe,
            "gain_over_selected_action": (
                conditioned_probe["accuracy"]
                - action_probe["accuracy"]
            ),
        }
    label_counts = {
        label: sum(_target(trace)[index] for trace in train)
        for index, label in enumerate(EFFECT_LABELS)
    }
    payload = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "status": "SOURCE_TRAIN_ONLY",
        "source_train_rows": len(train),
        "source_train_games": sorted({trace.game_id for trace in train}),
        "label_positive_counts": label_counts,
        "unique_motifs": len(
            {
                tuple(motif[key] for key in MOTIF_KEYS)
                for motif in motifs
            }
        ),
        "identity_probe": {
            "motif_only": motif_probe,
            "selected_action_only": action_probe,
            "selected_action_plus_motif": combined_probe,
            "motif_gain_over_action": (
                combined_probe["accuracy"] - action_probe["accuracy"]
            ),
            "source_train_design_candidates": candidate_diagnostics,
        },
        "source_validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
    }
    payload["result_checksum"] = _payload_checksum(payload)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination / "source_train_preflight.json", payload)
    return payload


def run_evaluation(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Fit on source train and evaluate the frozen constrained Stage A V2."""
    frozen = _load_frozen_manifest(Path(frozen_manifest_path))
    destination = Path(output_dir)
    input_path = Path(frozen["source_corpus"]["path"])
    collection = _read_json(input_path / "collection_manifest.json")
    if (
        _payload_checksum(collection)
        != frozen["source_corpus"]["collection_manifest_checksum"]
    ):
        raise ValueError("source collection manifest checksum mismatch")
    traces = _load_traces(collection, input_dir=input_path)
    train = [
        trace for trace in traces if trace.source_split == "source_train"
    ]
    validation = [
        trace
        for trace in traces
        if trace.source_split == "source_validation"
    ]
    if len(train) != int(frozen["source_corpus"]["source_train_rows"]):
        raise ValueError("source-train row count mismatch")
    if len(validation) != int(
        frozen["source_corpus"]["source_validation_rows"]
    ):
        raise ValueError("source-validation row count mismatch")

    embedder = FrozenQwenEmbedder(
        model_path=str(frozen["model"]["path"]),
        device=str(frozen["model"]["device"]),
        batch_size=int(frozen["model"]["batch_size"]),
        maximum_input_tokens=int(
            frozen["model"]["maximum_input_tokens"]
        ),
    )
    train_prompts = [invariant_prompt(trace) for trace in train]
    validation_prompts = [
        invariant_prompt(trace) for trace in validation
    ]
    shuffled_prompts = [
        invariant_prompt(trace, variant="relation_shuffle")
        for trace in validation
    ]
    train_embeddings, train_timing = embedder.encode(train_prompts)
    validation_embeddings, validation_timing = embedder.encode(
        validation_prompts
    )
    shuffled_embeddings, shuffled_timing = embedder.encode(
        shuffled_prompts
    )
    train_targets = np.asarray(
        [_target(trace) for trace in train],
        dtype=np.int64,
    )
    validation_targets = np.asarray(
        [_target(trace) for trace in validation],
        dtype=np.int64,
    )

    qwen_models = _fit_multilabel(
        train_embeddings,
        train_targets,
        random_state=int(frozen["training"]["random_state"]),
    )
    qwen_probabilities = _predict_multilabel(
        qwen_models,
        validation_embeddings,
    )
    shuffle_probabilities = _predict_multilabel(
        qwen_models,
        shuffled_embeddings,
    )
    threshold = float(frozen["training"]["decision_threshold"])
    qwen_predictions = qwen_probabilities >= threshold
    shuffle_predictions = shuffle_probabilities >= threshold

    action_train, action_validation = _vectorize_features(
        [
            {f"action:{trace.selected_action_name}": 1}
            for trace in train
        ],
        [
            {f"action:{trace.selected_action_name}": 1}
            for trace in validation
        ],
    )
    action_models = _fit_multilabel(
        action_train,
        train_targets,
        random_state=int(frozen["training"]["random_state"]),
    )
    action_predictions = (
        _predict_multilabel(action_models, action_validation) >= threshold
    )
    motif_train, motif_validation = _vectorize_features(
        [
            {
                **invariant_motif(trace),
                f"action:{trace.selected_action_name}": 1,
            }
            for trace in train
        ],
        [
            {
                **invariant_motif(trace),
                f"action:{trace.selected_action_name}": 1,
            }
            for trace in validation
        ],
    )
    motif_models = _fit_multilabel(
        motif_train,
        train_targets,
        random_state=int(frozen["training"]["random_state"]),
    )
    motif_predictions = (
        _predict_multilabel(motif_models, motif_validation) >= threshold
    )
    template_predictions = np.zeros_like(validation_targets, dtype=bool)
    template_predictions[:, EFFECT_LABELS.index("level_complete")] = True

    rendered_predictions = []
    compiled_predictions = np.zeros_like(qwen_predictions, dtype=bool)
    compiled_shuffle = np.zeros_like(shuffle_predictions, dtype=bool)
    strict_json = 0
    parsed_hypotheses = 0
    support_zero = 0
    grounded = 0
    for row_index, trace in enumerate(validation):
        original_map = {
            label: bool(qwen_predictions[row_index, index])
            for index, label in enumerate(EFFECT_LABELS)
        }
        shuffled_map = {
            label: bool(shuffle_predictions[row_index, index])
            for index, label in enumerate(EFFECT_LABELS)
        }
        raw = render_hypotheses_json(trace, original_map)
        shuffled_raw = render_hypotheses_json(trace, shuffled_map)
        parsed = hypotheses_from_json(raw, maximum=len(EFFECT_LABELS))
        shuffled_parsed = hypotheses_from_json(
            shuffled_raw,
            maximum=len(EFFECT_LABELS),
        )
        strict_json += 1
        parsed_hypotheses += len(parsed)
        support_zero += sum(item.support == 0 for item in parsed)
        legal = (
            SemanticActionCandidate(
                trace.selected_action_name,
                trace.selected_action_data,
            ),
        )
        original_compilation = HypothesisCompiler().compile(
            parsed,
            graph=graph_from_mapping(trace.scene_graph),
            legal_candidates=legal,
        )
        shuffled_compilation = HypothesisCompiler().compile(
            shuffled_parsed,
            graph=graph_from_mapping(trace.relation_shuffle_graph),
            legal_candidates=legal,
        )
        grounded += len(
            {
                option.hypothesis_id
                for option in original_compilation.options
            }
        )
        compiled_predictions[row_index] = _compiled_label_vector(
            original_compilation.options
        )
        compiled_shuffle[row_index] = _compiled_label_vector(
            shuffled_compilation.options
        )
        rendered_predictions.append(
            {
                "format_version": PREDICTION_FORMAT_VERSION,
                "trace_digest": trace.digest,
                "game_id": trace.game_id,
                "source_split": trace.source_split,
                "target": {
                    label: bool(validation_targets[row_index, index])
                    for index, label in enumerate(EFFECT_LABELS)
                },
                "probabilities": {
                    label: float(qwen_probabilities[row_index, index])
                    for index, label in enumerate(EFFECT_LABELS)
                },
                "shuffle_probabilities": {
                    label: float(shuffle_probabilities[row_index, index])
                    for index, label in enumerate(EFFECT_LABELS)
                },
                "typed_json": raw,
                "parsed_hypotheses": len(parsed),
                "grounded_hypotheses": len(
                    {
                        option.hypothesis_id
                        for option in original_compilation.options
                    }
                ),
                "compiler_rejections": list(
                    original_compilation.rejected
                ),
            }
        )

    metrics = _evaluation_metrics(
        frozen,
        validation=validation,
        targets=validation_targets,
        qwen=compiled_predictions,
        shuffle=compiled_shuffle,
        action=action_predictions,
        motif=motif_predictions,
        template=template_predictions,
    )
    identity = _identity_metrics(train)
    metrics.update(
        {
            "strict_json_validity": strict_json / len(validation),
            "support_zero_rate": (
                support_zero / parsed_hypotheses
                if parsed_hypotheses
                else 1.0
            ),
            "grounded_hypothesis_rate": (
                grounded / parsed_hypotheses
                if parsed_hypotheses
                else 0.0
            ),
            "parsed_hypotheses": parsed_hypotheses,
            "game_identity_probe": identity,
        }
    )
    gates = _gates(frozen, metrics)
    passed = all(gates.values())
    prediction_path = destination / "predictions.jsonl"
    _write_jsonl_atomic(prediction_path, rendered_predictions)
    payload = {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "all_gates_passed": passed,
        "authorized_next_stage": (
            "semantic_world_model_pilot" if passed else "none"
        ),
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "source_collection_manifest_checksum": _payload_checksum(
            collection
        ),
        "source_train_rows": len(train),
        "source_validation_rows": len(validation),
        "metrics": metrics,
        "gates": gates,
        "timing": {
            "train_original": train_timing,
            "validation_original": validation_timing,
            "validation_relation_shuffle": shuffled_timing,
        },
        "runtime": _runtime_metadata(),
        "predictions_sha256": _file_sha256(prediction_path),
        "world_model_fit_started": False,
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
    }
    payload["result_checksum"] = _payload_checksum(payload)
    _write_json_atomic(destination / "pilot_result.json", payload)
    return payload


def _evaluation_metrics(
    frozen: Mapping[str, Any],
    *,
    validation: Sequence[ProposalPilotTrace],
    targets: np.ndarray,
    qwen: np.ndarray,
    shuffle: np.ndarray,
    action: np.ndarray,
    motif: np.ndarray,
    template: np.ndarray,
) -> dict[str, Any]:
    primary = [EFFECT_LABELS.index(label) for label in PRIMARY_EFFECT_LABELS]
    method_predictions = {
        "qwen_linear_head": qwen,
        "relation_shuffle": shuffle,
        "action_only": action,
        "motif_logistic": motif,
        "deterministic_template": template,
    }
    method_metrics = {
        name: _multilabel_metrics(targets, predictions, primary)
        for name, predictions in method_predictions.items()
    }
    baseline_names = (
        "action_only",
        "motif_logistic",
        "deterministic_template",
    )
    stronger_name = max(
        baseline_names,
        key=lambda name: method_metrics[name]["primary_macro_f1"],
    )
    stronger = method_metrics[stronger_name]["primary_macro_f1"]
    qwen_score = method_metrics["qwen_linear_head"]["primary_macro_f1"]
    shuffle_score = method_metrics["relation_shuffle"]["primary_macro_f1"]
    per_game = {}
    for game in frozen["source_validation_games"]:
        indices = [
            index
            for index, trace in enumerate(validation)
            if trace.game_id == game
        ]
        target_rows = targets[indices]
        qwen_rows = _multilabel_metrics(
            target_rows,
            qwen[indices],
            primary,
        )
        baselines = {
            name: _multilabel_metrics(
                target_rows,
                method_predictions[name][indices],
                primary,
            )
            for name in baseline_names
        }
        game_baseline_name = max(
            baseline_names,
            key=lambda name: baselines[name]["primary_macro_f1"],
        )
        per_game[game] = {
            "rows": len(indices),
            "qwen_primary_macro_f1": qwen_rows["primary_macro_f1"],
            "stronger_baseline": game_baseline_name,
            "stronger_baseline_primary_macro_f1": baselines[
                game_baseline_name
            ]["primary_macro_f1"],
            "gain": (
                qwen_rows["primary_macro_f1"]
                - baselines[game_baseline_name]["primary_macro_f1"]
            ),
        }
    return {
        "effect_labels": list(EFFECT_LABELS),
        "primary_effect_labels": list(PRIMARY_EFFECT_LABELS),
        "methods": method_metrics,
        "stronger_baseline": stronger_name,
        "stronger_baseline_primary_macro_f1": stronger,
        "qwen_gain_over_stronger_baseline": qwen_score - stronger,
        "relation_shuffle_degradation": qwen_score - shuffle_score,
        "per_game": per_game,
    }


def _multilabel_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    primary_indices: Sequence[int],
) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    per_label = {}
    for index, label in enumerate(EFFECT_LABELS):
        per_label[label] = {
            "positives": int(targets[:, index].sum()),
            "predicted_positives": int(predictions[:, index].sum()),
            "precision": float(
                precision_score(
                    targets[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    targets[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    targets[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
        }
    return {
        "primary_macro_f1": float(
            f1_score(
                targets[:, primary_indices],
                predictions[:, primary_indices],
                average="macro",
                zero_division=0,
            )
        ),
        "primary_macro_recall": float(
            recall_score(
                targets[:, primary_indices],
                predictions[:, primary_indices],
                average="macro",
                zero_division=0,
            )
        ),
        "all_label_macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "per_label": per_label,
    }


def _gates(
    frozen: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, bool]:
    thresholds = frozen["gates"]
    identity = metrics["game_identity_probe"]
    return {
        "strict_json_validity": metrics["strict_json_validity"]
        >= float(thresholds["minimum_strict_json_validity"]),
        "support_zero": metrics["support_zero_rate"]
        >= float(thresholds["minimum_support_zero_rate"]),
        "grounded_hypotheses": metrics["grounded_hypothesis_rate"]
        >= float(thresholds["minimum_grounded_hypothesis_rate"]),
        "primary_macro_f1_gain": metrics[
            "qwen_gain_over_stronger_baseline"
        ]
        >= float(thresholds["minimum_primary_macro_f1_gain"]),
        "relation_shuffle": metrics["relation_shuffle_degradation"]
        >= float(thresholds["minimum_relation_shuffle_degradation"]),
        "per_game_nonnegative": all(
            item["gain"] >= 0.0 for item in metrics["per_game"].values()
        ),
        "motif_identity": identity["motif_only"]["accuracy"]
        <= identity["motif_only"]["majority_accuracy"]
        + float(thresholds["maximum_motif_identity_gain_over_majority"]),
        "action_conditioned_identity": identity[
            "motif_gain_over_action"
        ]
        <= float(
            thresholds["maximum_identity_gain_over_selected_action"]
        ),
    }


def _identity_metrics(
    traces: Sequence[ProposalPilotTrace],
) -> dict[str, Any]:
    labels = [trace.game_id for trace in traces]
    motifs = [invariant_motif(trace) for trace in traces]
    actions = [
        {f"action:{trace.selected_action_name}": 1}
        for trace in traces
    ]
    combined = [
        {**motif, **action}
        for motif, action in zip(motifs, actions)
    ]
    motif_probe = _identity_probe(motifs, labels)
    action_probe = _identity_probe(actions, labels)
    combined_probe = _identity_probe(combined, labels)
    return {
        "motif_only": motif_probe,
        "selected_action_only": action_probe,
        "selected_action_plus_motif": combined_probe,
        "motif_gain_over_action": (
            combined_probe["accuracy"] - action_probe["accuracy"]
        ),
    }


def _identity_probe(
    feature_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    matrix = DictVectorizer(sparse=True).fit_transform(feature_rows)
    label_array = np.asarray(labels)
    scores = cross_val_score(
        LogisticRegression(max_iter=1_000, random_state=12),
        matrix,
        label_array,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=12),
        scoring="accuracy",
    )
    majority = max(Counter(labels).values()) / len(labels)
    return {
        "accuracy": float(np.mean(scores)),
        "fold_accuracies": [float(value) for value in scores],
        "majority_accuracy": float(majority),
        "gain_over_majority": float(np.mean(scores) - majority),
    }


def _fit_multilabel(
    matrix: Any,
    targets: np.ndarray,
    *,
    random_state: int,
) -> list[Any]:
    from sklearn.linear_model import LogisticRegression

    models = []
    for index in range(targets.shape[1]):
        labels = targets[:, index]
        if len(set(int(value) for value in labels)) < 2:
            models.append(float(labels[0]))
            continue
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state,
            solver="liblinear",
        )
        model.fit(matrix, labels)
        models.append(model)
    return models


def _predict_multilabel(
    models: Sequence[Any],
    matrix: Any,
) -> np.ndarray:
    columns = []
    for model in models:
        if isinstance(model, float):
            columns.append(np.full(matrix.shape[0], model, dtype=float))
        else:
            columns.append(model.predict_proba(matrix)[:, 1])
    return np.column_stack(columns)


def _vectorize_features(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any]:
    from sklearn.feature_extraction import DictVectorizer

    vectorizer = DictVectorizer(sparse=True)
    train = vectorizer.fit_transform(train_rows)
    validation = vectorizer.transform(validation_rows)
    return train, validation


def _compiled_label_vector(options: Sequence[Any]) -> np.ndarray:
    predicted = {label: False for label in EFFECT_LABELS}
    for option in options:
        for effect in option.asserted_effects:
            name = str(effect).split("|", 1)[0]
            if name == "moved":
                predicted["player_moved"] = True
            elif name in predicted:
                predicted[name] = True
    return np.asarray(
        [predicted[label] for label in EFFECT_LABELS],
        dtype=bool,
    )


def _target(trace: ProposalPilotTrace) -> tuple[int, ...]:
    return (
        int(trace.changed),
        int(trace.player_moved),
        int(trace.level_complete),
        int(trace.game_over),
    )


def _load_traces(
    collection: Mapping[str, Any],
    *,
    input_dir: Path,
    split: str | None = None,
) -> list[ProposalPilotTrace]:
    traces = []
    for shard in collection["shards"]:
        if split is not None and str(shard["source_split"]) != split:
            continue
        relative = Path(str(shard["path"]))
        path = input_dir / "shards" / relative.name
        if _file_sha256(path) != str(shard["sha256"]):
            raise ValueError(f"source shard checksum mismatch: {path}")
        traces.extend(read_trace_shard(path))
    return traces


def _load_frozen_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported constrained-pilot manifest")
    if payload.get("status") != "FROZEN_BEFORE_VALIDATION":
        raise ValueError("constrained pilot was not frozen")
    actual = _payload_checksum(payload)
    if actual != payload.get("manifest_checksum"):
        raise ValueError(
            f"constrained manifest checksum mismatch: {actual}"
        )
    return payload


def _runtime_metadata() -> dict[str, Any]:
    import sklearn
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else ""
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_checksum", None)
    canonical.pop("manifest_checksum", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the constrained SAGE12 effect pilot V2."
    )
    parser.add_argument(
        "command",
        choices=("preflight", "evaluate"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=DEFAULT_FROZEN_MANIFEST_PATH,
    )
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_source_train_preflight(
            input_dir=args.input_dir,
            output_dir=args.out_dir,
        )
    else:
        result = run_evaluation(
            frozen_manifest_path=args.frozen_manifest,
            output_dir=args.out_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FROZEN_MANIFEST_PATH",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "EFFECT_LABELS",
    "FORMAT_VERSION",
    "FrozenQwenEmbedder",
    "invariant_motif",
    "invariant_prompt",
    "render_hypotheses_json",
    "run_evaluation",
    "run_source_train_preflight",
]
