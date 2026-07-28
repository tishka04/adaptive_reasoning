"""SAGE12 V4.9 object-relative student for the semantic teacher.

The student is a compact DeepSets graph predictor.  It receives hashed
categorical root and relation tokens, pools an unordered set of neighboring
objects, predicts the frozen teacher vocabulary, and is trained with:

* masked multi-label imitation;
* same-prestate productive-effect ranking;
* an adversarial game-identity confusion objective.

Every reported prediction is outer leave-one-game-out (LOGO).  The module can
also export the seven base effects as V4.7 ``SlotAnnotation`` records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN

from .compiler import SLOT_EFFECTS
from .integration_pilot import load_complete_roots
from .integration_pilot_v4_7 import load_slot_examples
from .semantic_teacher_v4_9 import (
    BASE_EFFECTS,
    DEFAULT_OUTPUT_DIR,
    SEMANTIC_EFFECTS,
    PairLink,
    SemanticTeacherRecord,
    _checksum,
    _file_sha256,
    _json_safe,
    _read_json,
    _write_json,
    _write_jsonl,
    load_manifest,
    load_pair_links,
    load_teacher_records,
)

RESULT_VERSION = "sage12-object-relative-student-result-v4.9"
PREDICTION_VERSION = "sage12-object-relative-logo-prediction-v4.9"
SLOT_EXPORT_VERSION = "sage12-object-relative-slot-annotations-v4.9"
DEFAULT_V43_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"


@dataclass(frozen=True)
class TensorizedGraphs:
    root_ids: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_mask: np.ndarray
    labels: np.ndarray
    applicable: np.ndarray


def _token_id(token: str, buckets: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (int(buckets) - 1)


def _tokens(prefix: str, values: Mapping[str, Any]) -> list[str]:
    return [
        f"{prefix}.{key}={str(value).lower()}" for key, value in sorted(values.items())
    ]


def _action_root(root: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: root[key]
        for key in ("action_name", "action_family", "requested_direction")
    }


def tensorize_records(
    records: Sequence[SemanticTeacherRecord],
    *,
    hash_buckets: int,
    maximum_neighbors: int,
    mode: str = "full",
    relation_shuffle: bool = False,
    reverse_neighbors: bool = False,
) -> TensorizedGraphs:
    if mode not in {"full", "root_only", "action_only"}:
        raise ValueError(f"unknown graph mode: {mode}")
    graphs = [
        record.graph.relation_shuffled() if relation_shuffle else record.graph
        for record in records
    ]
    root_token_rows = []
    neighbor_token_rows = []
    for graph in graphs:
        root = _action_root(graph.root) if mode == "action_only" else dict(graph.root)
        root_token_rows.append(_tokens("root", root))
        neighbors = [] if mode != "full" else list(graph.neighbors)
        if reverse_neighbors:
            neighbors = list(reversed(neighbors))
        neighbor_token_rows.append(
            [_tokens("neighbor", item) for item in neighbors[:maximum_neighbors]]
        )
    root_width = max(max((len(row) for row in root_token_rows), default=1), 1)
    node_width = max(
        max(
            (len(tokens) for rows in neighbor_token_rows for tokens in rows),
            default=1,
        ),
        1,
    )
    roots = np.zeros((len(records), root_width), dtype=np.int64)
    nodes = np.zeros((len(records), maximum_neighbors, node_width), dtype=np.int64)
    mask = np.zeros((len(records), maximum_neighbors), dtype=np.float32)
    for index, tokens in enumerate(root_token_rows):
        roots[index, : len(tokens)] = [
            _token_id(token, hash_buckets) for token in tokens
        ]
    for index, rows in enumerate(neighbor_token_rows):
        for node_index, tokens in enumerate(rows):
            nodes[index, node_index, : len(tokens)] = [
                _token_id(token, hash_buckets) for token in tokens
            ]
            mask[index, node_index] = 1.0
    labels = np.asarray(
        [
            [float(record.labels[effect]) for effect in SEMANTIC_EFFECTS]
            for record in records
        ],
        dtype=np.float32,
    )
    applicable = np.asarray(
        [
            [float(record.applicable[effect]) for effect in SEMANTIC_EFFECTS]
            for record in records
        ],
        dtype=np.float32,
    )
    return TensorizedGraphs(
        root_ids=roots,
        neighbor_ids=nodes,
        neighbor_mask=mask,
        labels=labels,
        applicable=applicable,
    )


def _torch_model(
    *,
    hash_buckets: int,
    embedding_width: int,
    hidden_width: int,
    effect_count: int,
    identity_classes: int,
) -> Any:
    import torch

    class ObjectRelativeDeepSets(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(
                hash_buckets,
                embedding_width,
                padding_idx=0,
            )
            self.node_encoder = torch.nn.Sequential(
                torch.nn.Linear(embedding_width, embedding_width),
                torch.nn.GELU(),
                torch.nn.LayerNorm(embedding_width),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(embedding_width * 3, hidden_width),
                torch.nn.GELU(),
                torch.nn.Dropout(0.10),
                torch.nn.LayerNorm(hidden_width),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.GELU(),
            )
            self.effect_head = torch.nn.Linear(hidden_width, effect_count)
            self.identity_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_width, max(32, hidden_width // 2)),
                torch.nn.GELU(),
                torch.nn.Linear(max(32, hidden_width // 2), identity_classes),
            )

        @staticmethod
        def _mean_tokens(ids: Any, embeddings: Any) -> Any:
            token_mask = (ids != 0).to(embeddings.dtype)
            total = (embeddings * token_mask.unsqueeze(-1)).sum(dim=-2)
            denominator = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return total / denominator

        def encode(
            self,
            root_ids: Any,
            neighbor_ids: Any,
            neighbor_mask: Any,
        ) -> Any:
            root = self._mean_tokens(root_ids, self.embedding(root_ids))
            nodes = self._mean_tokens(
                neighbor_ids,
                self.embedding(neighbor_ids),
            )
            nodes = self.node_encoder(nodes)
            mask = neighbor_mask.unsqueeze(-1)
            mean = (nodes * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            masked = nodes.masked_fill(mask == 0, -1e4)
            maximum = masked.max(dim=1).values
            empty = neighbor_mask.sum(dim=1, keepdim=True) == 0
            maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
            return self.trunk(torch.cat((root, mean, maximum), dim=-1))

        def forward(
            self,
            root_ids: Any,
            neighbor_ids: Any,
            neighbor_mask: Any,
        ) -> tuple[Any, Any]:
            latent = self.encode(root_ids, neighbor_ids, neighbor_mask)
            return self.effect_head(latent), latent

    return ObjectRelativeDeepSets()


def _select_device(requested: str) -> str:
    import torch

    normalized = str(requested).lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _masked_effect_loss(logits: Any, labels: Any, masks: Any) -> Any:
    import torch

    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    per_effect = []
    for index in range(raw.shape[1]):
        applicable = masks[:, index]
        per_effect.append(
            (raw[:, index] * applicable).sum() / applicable.sum().clamp_min(1.0)
        )
    return torch.stack(per_effect).mean()


def _batch_arrays(tensors: TensorizedGraphs, indices: np.ndarray) -> tuple[Any, ...]:
    import torch

    return (
        torch.as_tensor(tensors.root_ids[indices], dtype=torch.long),
        torch.as_tensor(tensors.neighbor_ids[indices], dtype=torch.long),
        torch.as_tensor(tensors.neighbor_mask[indices], dtype=torch.float32),
        torch.as_tensor(tensors.labels[indices], dtype=torch.float32),
        torch.as_tensor(tensors.applicable[indices], dtype=torch.float32),
    )


def _fit_one_model(
    records: Sequence[SemanticTeacherRecord],
    tensors: TensorizedGraphs,
    *,
    train_indices: np.ndarray,
    pair_links: Sequence[PairLink],
    parameters: Mapping[str, Any],
    device: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_games = sorted({records[index].game_id for index in train_indices})
    game_index = {game: index for index, game in enumerate(train_games)}
    model = _torch_model(
        hash_buckets=int(parameters["hash_buckets"]),
        embedding_width=int(parameters["embedding_width"]),
        hidden_width=int(parameters["hidden_width"]),
        effect_count=len(SEMANTIC_EFFECTS),
        identity_classes=len(train_games),
    ).to(device)
    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("identity_head.")
    ]
    optimizer = torch.optim.AdamW(
        main_parameters,
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    adversary_optimizer = torch.optim.AdamW(
        model.identity_head.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    batch_size = int(parameters["batch_size"])
    identity_weight = float(parameters["identity_adversary_weight"])
    ranking_weight = float(parameters["pairwise_ranking_weight"])
    productive_index = SEMANTIC_EFFECTS.index("productive")
    by_digest = {record.trace_digest: index for index, record in enumerate(records)}
    training_set = {int(item) for item in train_indices}
    pair_indices = []
    for link in pair_links:
        left = by_digest.get(link.left_trace_digest)
        right = by_digest.get(link.right_trace_digest)
        if (
            left is None
            or right is None
            or left not in training_set
            or right not in training_set
        ):
            continue
        delta = records[left].productive_score - records[right].productive_score
        if abs(delta) > 1e-9:
            pair_indices.append((left, right, 1.0 if delta > 0 else -1.0))

    started = time.perf_counter()
    final_losses: dict[str, float] = {}
    for epoch in range(int(parameters["epochs"])):
        model.train()
        shuffled = np.asarray(train_indices, dtype=np.int64).copy()
        np.random.default_rng(seed + epoch).shuffle(shuffled)
        supervised_losses = []
        confusion_losses = []
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            root, nodes, mask, labels, applicable = _batch_arrays(tensors, indices)
            root = root.to(device)
            nodes = nodes.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            applicable = applicable.to(device)
            identities = torch.as_tensor(
                [game_index[records[index].game_id] for index in indices],
                dtype=torch.long,
                device=device,
            )

            adversary_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, detached_latent = model(root, nodes, mask)
            identity_logits = model.identity_head(detached_latent.detach())
            identity_loss = torch.nn.functional.cross_entropy(
                identity_logits, identities
            )
            identity_loss.backward()
            adversary_optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            logits, latent = model(root, nodes, mask)
            semantic_loss = _masked_effect_loss(logits, labels, applicable)
            identity_logits = model.identity_head(latent)
            identity_probabilities = torch.softmax(identity_logits, dim=-1)
            uniform = torch.full_like(
                identity_probabilities,
                1.0 / identity_probabilities.shape[-1],
            )
            confusion = torch.nn.functional.kl_div(
                torch.log(identity_probabilities.clamp_min(1e-8)),
                uniform,
                reduction="batchmean",
            )
            loss = semantic_loss + identity_weight * confusion
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_parameters, 2.0)
            optimizer.step()
            supervised_losses.append(float(semantic_loss.detach().cpu()))
            confusion_losses.append(float(confusion.detach().cpu()))

        ranking_losses = []
        if pair_indices:
            ordered_pairs = list(pair_indices)
            random.Random(seed + 10_000 + epoch).shuffle(ordered_pairs)
            for start in range(0, len(ordered_pairs), batch_size):
                chunk = ordered_pairs[start : start + batch_size]
                left_indices = np.asarray([item[0] for item in chunk], dtype=np.int64)
                right_indices = np.asarray([item[1] for item in chunk], dtype=np.int64)
                signs = torch.as_tensor(
                    [item[2] for item in chunk],
                    dtype=torch.float32,
                    device=device,
                )
                left = _batch_arrays(tensors, left_indices)
                right = _batch_arrays(tensors, right_indices)
                optimizer.zero_grad(set_to_none=True)
                left_logits, _ = model(
                    left[0].to(device),
                    left[1].to(device),
                    left[2].to(device),
                )
                right_logits, _ = model(
                    right[0].to(device),
                    right[1].to(device),
                    right[2].to(device),
                )
                difference = (
                    left_logits[:, productive_index] - right_logits[:, productive_index]
                )
                ranking = torch.nn.functional.softplus(-signs * difference).mean()
                (ranking_weight * ranking).backward()
                torch.nn.utils.clip_grad_norm_(main_parameters, 2.0)
                optimizer.step()
                ranking_losses.append(float(ranking.detach().cpu()))

        final_losses = {
            "semantic": float(np.mean(supervised_losses)),
            "identity_confusion": float(np.mean(confusion_losses)),
            "pairwise_ranking": (
                float(np.mean(ranking_losses)) if ranking_losses else 0.0
            ),
        }
    runtime = time.perf_counter() - started
    model.eval()
    return model, {
        "runtime_seconds": runtime,
        "train_rows": len(train_indices),
        "train_games": train_games,
        "ranking_pairs": len(pair_indices),
        "final_losses": final_losses,
    }


def _predict_model(
    model: Any,
    tensors: TensorizedGraphs,
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            root, nodes, mask, _labels, _applicable = _batch_arrays(
                tensors, batch_indices
            )
            logits, _ = model(
                root.to(device),
                nodes.to(device),
                mask.to(device),
            )
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return (
        np.concatenate(predictions, axis=0)
        if predictions
        else np.zeros((0, len(SEMANTIC_EFFECTS)), dtype=np.float32)
    )


def _action_only_probabilities(
    records: Sequence[SemanticTeacherRecord],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(test_indices), len(SEMANTIC_EFFECTS)), dtype=np.float64)
    for effect_index, effect in enumerate(SEMANTIC_EFFECTS):
        global_rows = [
            records[index]
            for index in train_indices
            if records[index].applicable[effect]
        ]
        global_probability = (
            (sum(row.labels[effect] for row in global_rows) + 1.0)
            / (len(global_rows) + 2.0)
            if global_rows
            else 0.5
        )
        by_action: dict[str, list[SemanticTeacherRecord]] = defaultdict(list)
        for index in train_indices:
            record = records[index]
            if record.applicable[effect]:
                by_action[str(record.graph.root["action_name"])].append(record)
        for row_index, index in enumerate(test_indices):
            action = str(records[index].graph.root["action_name"])
            rows = by_action.get(action, [])
            result[row_index, effect_index] = (
                (sum(row.labels[effect] for row in rows) + 1.0) / (len(rows) + 2.0)
                if rows
                else global_probability
            )
    return result


def _brier_metrics(
    records: Sequence[SemanticTeacherRecord],
    predictions: np.ndarray,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    briers = []
    for effect_index, effect in enumerate(SEMANTIC_EFFECTS):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.applicable[effect]
            ],
            dtype=np.int64,
        )
        labels = np.asarray(
            [records[index].labels[effect] for index in indices],
            dtype=np.float64,
        )
        probabilities = predictions[indices, effect_index]
        brier = float(np.mean(np.square(probabilities - labels)))
        positives = int(labels.sum())
        rows[effect] = {
            "brier": brier,
            "applicable": len(indices),
            "positive": positives,
            "recall_at_0_5": (
                float(np.sum((probabilities >= 0.5) & (labels == 1)) / positives)
                if positives
                else None
            ),
        }
        briers.append(brier)
    return {
        "macro_brier": float(np.mean(briers)),
        "base_macro_brier": float(
            np.mean([rows[effect]["brier"] for effect in BASE_EFFECTS])
        ),
        "functional_macro_brier": float(
            np.mean(
                [
                    rows[effect]["brier"]
                    for effect in SEMANTIC_EFFECTS
                    if effect not in BASE_EFFECTS
                ]
            )
        ),
        "per_effect": rows,
    }


def _pair_ranking_metrics(
    records: Sequence[SemanticTeacherRecord],
    pair_links: Sequence[PairLink],
    predictions: np.ndarray,
) -> dict[str, Any]:
    by_digest = {record.trace_digest: index for index, record in enumerate(records)}
    productive_index = SEMANTIC_EFFECTS.index("productive")
    correct = 0
    ties = 0
    rows = 0
    per_game: dict[str, list[bool]] = defaultdict(list)
    for link in pair_links:
        left = by_digest.get(link.left_trace_digest)
        right = by_digest.get(link.right_trace_digest)
        if left is None or right is None:
            continue
        truth = records[left].productive_score - records[right].productive_score
        if abs(truth) <= 1e-9:
            continue
        predicted = (
            predictions[left, productive_index] - predictions[right, productive_index]
        )
        rows += 1
        if abs(predicted) <= 1e-12:
            ties += 1
            outcome = False
        else:
            outcome = bool(math.copysign(1.0, predicted) == math.copysign(1.0, truth))
        correct += int(outcome)
        per_game[link.game_id].append(outcome)
    return {
        "discordant_pairs": rows,
        "accuracy": correct / rows if rows else 0.0,
        "ties": ties,
        "per_game": {
            game: {
                "pairs": len(values),
                "accuracy": float(np.mean(values)) if values else None,
            }
            for game, values in sorted(per_game.items())
        },
    }


def _identity_probe(
    records: Sequence[SemanticTeacherRecord],
    predictions: np.ndarray,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    labels = np.asarray([record.game_id for record in records])
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4_909)
    model = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        random_state=4_909,
    )
    accuracy = float(np.mean(cross_val_score(model, predictions, labels, cv=folds)))
    return {
        "majority_accuracy": float(majority),
        "accuracy": accuracy,
        "gain_over_majority": accuracy - majority,
    }


def _completion_recall_at_8(
    records: Sequence[SemanticTeacherRecord],
    predictions: np.ndarray,
) -> dict[str, Any]:
    effect_index = SEMANTIC_EFFECTS.index("level_complete")
    total = 0
    captured = 0
    per_game = {}
    for game in SOURCE_TRAIN:
        indices = np.asarray(
            [index for index, record in enumerate(records) if record.game_id == game],
            dtype=np.int64,
        )
        positives = [
            index for index in indices if records[index].labels["level_complete"]
        ]
        if not positives:
            continue
        ranked = indices[
            np.argsort(-predictions[indices, effect_index], kind="stable")
        ][:8]
        found = len({int(item) for item in ranked} & set(positives))
        total += len(positives)
        captured += found
        per_game[game] = {
            "positives": len(positives),
            "captured": found,
        }
    return {
        "positives": total,
        "captured": captured,
        "recall_at_8": captured / total if total else 0.0,
        "per_game": per_game,
    }


def _per_game_brier(
    records: Sequence[SemanticTeacherRecord],
    predictions: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for game in SOURCE_TRAIN:
        indices = [
            index for index, record in enumerate(records) if record.game_id == game
        ]
        values = []
        for effect_index, effect in enumerate(SEMANTIC_EFFECTS):
            applicable = [
                index for index in indices if records[index].applicable[effect]
            ]
            labels = np.asarray(
                [records[index].labels[effect] for index in applicable],
                dtype=np.float64,
            )
            values.append(
                float(
                    np.mean(np.square(predictions[applicable, effect_index] - labels))
                )
            )
        result[game] = {
            "rows": len(indices),
            "macro_brier": float(np.mean(values)),
        }
    return result


def _prediction_rows(
    records: Sequence[SemanticTeacherRecord],
    original: np.ndarray,
    root_only: np.ndarray,
    action_only: np.ndarray,
    shuffled: np.ndarray,
) -> Iterable[dict[str, Any]]:
    for index, record in enumerate(records):
        yield {
            "format_version": PREDICTION_VERSION,
            "example_id": record.example_id,
            "trace_digest": record.trace_digest,
            "game_id": record.game_id,
            "probabilities": {
                "object_relative": {
                    effect: float(original[index, effect_index])
                    for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                },
                "root_only": {
                    effect: float(root_only[index, effect_index])
                    for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                },
                "action_only": {
                    effect: float(action_only[index, effect_index])
                    for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                },
                "relation_shuffle": {
                    effect: float(shuffled[index, effect_index])
                    for effect_index, effect in enumerate(SEMANTIC_EFFECTS)
                },
            },
        }


def evaluate_student(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    qa = _read_json(destination / "teacher_qa.json")
    if not qa.get("teacher_ready"):
        raise RuntimeError("semantic teacher QA is not ready")
    records = load_teacher_records(destination)
    pair_links = load_pair_links(destination)
    parameters = dict(manifest["training"])
    maximum_neighbors = int(manifest["student_view"]["maximum_neighbors"])
    selected_device = _select_device(device)

    full_tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="full",
    )
    shuffled_tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="full",
        relation_shuffle=True,
    )
    reversed_tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="full",
        reverse_neighbors=True,
    )
    root_tensors = tensorize_records(
        records,
        hash_buckets=int(parameters["hash_buckets"]),
        maximum_neighbors=maximum_neighbors,
        mode="root_only",
    )

    shape = (len(records), len(SEMANTIC_EFFECTS))
    original = np.zeros(shape, dtype=np.float64)
    root_only = np.zeros(shape, dtype=np.float64)
    action_only = np.zeros(shape, dtype=np.float64)
    relation_shuffle = np.zeros(shape, dtype=np.float64)
    neighbor_reversed = np.zeros(shape, dtype=np.float64)
    folds = []
    for fold_index, held_out_game in enumerate(SOURCE_TRAIN):
        train_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id != held_out_game
            ],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.game_id == held_out_game
            ],
            dtype=np.int64,
        )
        full_model, full_runtime = _fit_one_model(
            records,
            full_tensors,
            train_indices=train_indices,
            pair_links=pair_links,
            parameters=parameters,
            device=selected_device,
            seed=int(parameters["seed"]) + fold_index * 100,
        )
        root_model, root_runtime = _fit_one_model(
            records,
            root_tensors,
            train_indices=train_indices,
            pair_links=pair_links,
            parameters=parameters,
            device=selected_device,
            seed=int(parameters["seed"]) + fold_index * 100 + 1,
        )
        batch_size = int(parameters["batch_size"])
        original[test_indices] = _predict_model(
            full_model,
            full_tensors,
            test_indices,
            device=selected_device,
            batch_size=batch_size,
        )
        relation_shuffle[test_indices] = _predict_model(
            full_model,
            shuffled_tensors,
            test_indices,
            device=selected_device,
            batch_size=batch_size,
        )
        neighbor_reversed[test_indices] = _predict_model(
            full_model,
            reversed_tensors,
            test_indices,
            device=selected_device,
            batch_size=batch_size,
        )
        root_only[test_indices] = _predict_model(
            root_model,
            root_tensors,
            test_indices,
            device=selected_device,
            batch_size=batch_size,
        )
        action_only[test_indices] = _action_only_probabilities(
            records,
            train_indices,
            test_indices,
        )
        folds.append(
            {
                "held_out_game": held_out_game,
                "training_rows": len(train_indices),
                "validation_rows": len(test_indices),
                "object_relative": full_runtime,
                "root_only": root_runtime,
            }
        )

    prediction_path = destination / "logo_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        _prediction_rows(
            records,
            original,
            root_only,
            action_only,
            relation_shuffle,
        ),
    )
    metrics = {
        "object_relative": _brier_metrics(records, original),
        "root_only": _brier_metrics(records, root_only),
        "action_only": _brier_metrics(records, action_only),
        "relation_shuffle": _brier_metrics(records, relation_shuffle),
    }
    pair_metrics = {
        "object_relative": _pair_ranking_metrics(records, pair_links, original),
        "root_only": _pair_ranking_metrics(records, pair_links, root_only),
        "action_only": _pair_ranking_metrics(records, pair_links, action_only),
        "relation_shuffle": _pair_ranking_metrics(
            records, pair_links, relation_shuffle
        ),
    }
    identity = _identity_probe(records, original)
    completion = _completion_recall_at_8(records, original)
    maximum_order_delta = float(np.max(np.abs(original - neighbor_reversed)))
    thresholds = manifest["evaluation"]["decision_thresholds"]
    checks = {
        "teacher_ready": bool(qa["teacher_ready"]),
        "full_macro_brier_gain_over_action_only_strictly_positive": (
            metrics["action_only"]["macro_brier"]
            - metrics["object_relative"]["macro_brier"]
            > 0.0
        ),
        "full_macro_brier_gain_over_root_only_strictly_positive": (
            metrics["root_only"]["macro_brier"]
            - metrics["object_relative"]["macro_brier"]
            > 0.0
        ),
        "productive_pair_accuracy_gain_over_action_only_strictly_positive": (
            pair_metrics["object_relative"]["accuracy"]
            - pair_metrics["action_only"]["accuracy"]
            > 0.0
        ),
        "relation_shuffle_brier_degradation_strictly_positive": (
            metrics["relation_shuffle"]["macro_brier"]
            - metrics["object_relative"]["macro_brier"]
            > 0.0
        ),
        "neighbor_permutation_invariance": (
            maximum_order_delta
            <= float(thresholds["neighbor_permutation_max_probability_delta"])
        ),
        "semantic_output_identity_accuracy_maximum": (
            identity["accuracy"]
            <= float(thresholds["semantic_output_identity_accuracy_maximum"])
        ),
        "completion_recall_at_8_minimum": (
            completion["recall_at_8"]
            >= float(thresholds["completion_recall_at_8_minimum"])
        ),
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "teacher_qa_checksum": qa["qa_checksum"],
        "verdict": (
            "OBJECT_RELATIVE_TEACHER_IMITATION_SUPPORTED"
            if passed
            else "OBJECT_RELATIVE_TEACHER_IMITATION_NOT_YET_SUPPORTED"
        ),
        "checks": checks,
        "confirmatory": False,
        "authority_promoted": False,
        "device": selected_device,
        "records": len(records),
        "games": list(SOURCE_TRAIN),
        "metrics": metrics,
        "macro_brier_gain": {
            "over_action_only": (
                metrics["action_only"]["macro_brier"]
                - metrics["object_relative"]["macro_brier"]
            ),
            "over_root_only": (
                metrics["root_only"]["macro_brier"]
                - metrics["object_relative"]["macro_brier"]
            ),
            "relation_shuffle_degradation": (
                metrics["relation_shuffle"]["macro_brier"]
                - metrics["object_relative"]["macro_brier"]
            ),
        },
        "productive_pair_ranking": pair_metrics,
        "semantic_output_game_identity_probe": identity,
        "completion_recall_at_8": completion,
        "neighbor_permutation_max_probability_delta": maximum_order_delta,
        "per_game_transfer": _per_game_brier(records, original),
        "folds": folds,
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "artifact_sha256": {
            "teacher_corpus": _file_sha256(destination / "teacher_corpus.jsonl"),
            "logo_predictions": _file_sha256(prediction_path),
        },
        "interpretation": {
            "teacher_uses_post_transition_state": True,
            "student_uses_pre_action_state_only": True,
            "student_predictions_are_logo": True,
            "absolute_coordinates_enter_student": False,
            "colors_or_raw_values_enter_student": False,
            "completion_capacity_is_observed_not_synthetic": True,
            "slot_export_is_compatible_with_v4_7_base_effects": True,
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "student_result.json", result)
    export_v47_annotations(output_dir=destination)
    return result


def _load_prediction_lookup(
    output_dir: Path,
) -> dict[str, dict[str, Mapping[str, float]]]:
    result = {}
    with (output_dir / "logo_predictions.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["trace_digest"])] = dict(row["probabilities"])
    return result


def export_v47_annotations(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    """Export LOGO base effects for the unchanged V4.7 world-model boundary."""

    destination = Path(output_dir)
    lookup = _load_prediction_lookup(destination)
    roots = load_complete_roots(v43_dir)
    examples = load_slot_examples(roots)
    by_position = {(item.root_key, item.path, item.side): item for item in examples}
    rows = []
    missing = []
    for root in roots:
        for path, pair in sorted(root.tree.items()):
            for side, arm in zip("LR", (pair.left, pair.right)):
                example = by_position[(root.root_key, path, side)]
                predictions = lookup.get(arm.trace.trace_digest)
                if predictions is None:
                    missing.append(arm.trace.trace_digest)
                    continue
                for variant, source in (
                    ("object_relative", "object_relative_deepsets_logo_v4_9"),
                    (
                        "relation_shuffle",
                        "object_relative_relation_shuffle_logo_v4_9",
                    ),
                ):
                    probabilities = predictions[variant]
                    rows.append(
                        {
                            "format_version": SLOT_EXPORT_VERSION,
                            "slot_id": example.slot.slot_id,
                            "example_id": example.example_id,
                            "game_id": example.game_id,
                            "variant": variant,
                            "effect_probabilities": {
                                effect: float(probabilities[effect])
                                for effect in SLOT_EFFECTS
                            },
                            "source": source,
                            "support": 0,
                        }
                    )
    if missing:
        raise ValueError(f"{len(missing)} V4.3 traces lack LOGO semantic predictions")
    path = destination / "v4_7_slot_annotations.jsonl"
    _write_jsonl(path, rows)
    summary: dict[str, Any] = {
        "format_version": SLOT_EXPORT_VERSION,
        "slots": len(examples),
        "rows": len(rows),
        "variants": ["object_relative", "relation_shuffle"],
        "missing": 0,
        "sha256": _file_sha256(path),
        "source_validation_opened": False,
        "holdout_opened": False,
    }
    summary["checksum"] = _checksum(summary)
    _write_json(destination / "v4_7_slot_export.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--device", default="cuda:0")
    export = subparsers.add_parser("export-v47")
    export.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        payload = evaluate_student(
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        payload = export_v47_annotations(output_dir=args.output_dir)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TensorizedGraphs",
    "evaluate_student",
    "export_v47_annotations",
    "tensorize_records",
]
