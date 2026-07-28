"""SAGE12 V4.8 paired semantic adaptation and end-to-end pilot.

The experiment stays inside the eleven registered source-train games.  It
derives same-prestate action pairs from the published SAGE11 corpus, combines
them with the replay-verified V4.3 pairs, embeds two orderings of an
identity-free comparison prompt with the local frozen Qwen2.5 0.5B model, and
fits a small external low-rank residual adapter.  Every semantic prediction
used downstream is leave-one-game-out.

The V4.7 world model, trajectory features, EBM, candidate trees, controller
choice, and fold-local baseline selection are reused without changing their
hyperparameters.  This is an exploratory architecture test, not an authority
promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import SOURCE_TRAIN

from .compiler import SLOT_EFFECTS, SlotAnnotation
from .integration_pilot import ExecutedRoot, load_complete_roots
from .integration_pilot_v4_7 import (
    EBM_EPOCHS,
    EBM_HIDDEN_WIDTH,
    EBM_INPUT_WIDTH,
    EBM_LEARNING_RATE,
    FORBIDDEN_MODEL_FIELDS,
    RELATION_FIELDS,
    SEED as V47_SEED,
    SlotExample,
    _action_only_choice,
    _binding_view,
    _decision_row,
    _event_view,
    _file_sha256,
    _fit_nested_world,
    _leaf_value,
    _nodes,
    _paired_bootstrap,
    _select_path,
    _select_primary_baseline,
    _sequence_only_choice,
    _shuffle_relation_payload,
    _summarize,
    _train_ebm,
    _world_metrics,
    load_annotations as load_v47_annotations,
    load_slot_examples,
)

FORMAT_VERSION = "sage12-paired-semantic-adapter-v4.8"
MANIFEST_VERSION = "sage12-paired-semantic-adapter-manifest-v4.8"
SEMANTIC_RESULT_VERSION = "sage12-paired-semantic-result-v4.8"
RESULT_VERSION = "sage12-paired-semantic-integration-result-v4.8"

DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "semantic_adapter_v4_8"
DEFAULT_SAGE11_DIR = Path("training") / "sage11" / "source_dataset_v2"
DEFAULT_V43_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"
DEFAULT_V47_DIR = Path("training") / "sage12" / "integration_pilot_v4_7"
DEFAULT_MODEL_PATH = Path("models") / "qwen2_5_0.5b_instruct"

SEED = 4_808
REPRESENTATIONS = ("minimal", "invariant_context")
TRAIN_EFFECTS = SLOT_EFFECTS + ("progress",)
PAIR_CLASSES = ("neither", "left", "right", "both")
SWAP_CLASS_INDEX = np.asarray([0, 2, 1, 3], dtype=np.int64)
ADAPTER_RANK = 16
ADAPTER_EPOCHS = 24
ADAPTER_BATCH_SIZE = 512
ADAPTER_LEARNING_RATE = 0.002
MAXIMUM_INPUT_TOKENS = 256
QWEN_BATCH_SIZE = 48

MOVE_DIRECTIONS = {
    "ACTION1": "up",
    "ACTION2": "down",
    "ACTION3": "left",
    "ACTION4": "right",
}
SOURCE_BUCKET_CAPS = {
    "level_complete": 10_000,
    "game_over": 100,
    "progress": 150,
    "moved": 100,
    "changed": 100,
    "neutral": 40,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bucket(value: int) -> str:
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 3:
        return "few"
    return "many"


def _action_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row["action_name"]),
        _canonical(row.get("action_data", {})),
    )


def _source_labels(row: Mapping[str, Any]) -> dict[str, bool]:
    effects = set(row.get("effect_atoms", ()))
    labels = row.get("labels", {})
    progress = any(
        bool(labels.get(name, False))
        for name in (
            "frontier_credit",
            "route_confirmation",
            "subeffect_relay",
            "subgoal_graph_advance",
            "level_completed",
            "terminal_event",
            "won",
        )
    )
    return {
        "changed": bool(row.get("changed", not row.get("noop", False))),
        "moved": "effect:player_moved(True)" in effects,
        "target_created": False,
        "target_removed": False,
        "target_moved": False,
        "level_complete": bool(labels.get("level_completed", False))
        or "progress:level_complete(True)" in effects,
        "game_over": bool(labels.get("terminal_event", False))
        or "risk:game_over(True)" in effects,
        "progress": progress,
    }


def _pair_class(left: bool, right: bool) -> int:
    if left and right:
        return 3
    if left:
        return 1
    if right:
        return 2
    return 0


def _relative_click_descriptors(
    left_data: Mapping[str, Any],
    right_data: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    required = ("x", "y")
    if not all(key in left_data and key in right_data for key in required):
        unknown = {"relation_to_other": "unknown", "pair_distance": "unknown"}
        return dict(unknown), dict(unknown)
    dx = int(right_data["x"]) - int(left_data["x"])
    dy = int(right_data["y"]) - int(left_data["y"])
    horizontal = "right" if dx > 0 else "left" if dx < 0 else "aligned"
    vertical = "below" if dy > 0 else "above" if dy < 0 else "aligned"
    opposite_h = {"right": "left", "left": "right", "aligned": "aligned"}
    opposite_v = {"below": "above", "above": "below", "aligned": "aligned"}
    distance = abs(dx) + abs(dy)
    distance_bucket = (
        "same"
        if distance == 0
        else "near"
        if distance <= 8
        else "medium"
        if distance <= 24
        else "far"
    )
    return (
        {
            "relation_to_other": f"{horizontal}_{vertical}",
            "pair_distance": distance_bucket,
        },
        {
            "relation_to_other": (
                f"{opposite_h[horizontal]}_{opposite_v[vertical]}"
            ),
            "pair_distance": distance_bucket,
        },
    )


def _source_descriptors(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    def base(row: Mapping[str, Any]) -> dict[str, Any]:
        name = str(row["action_name"])
        family = "move" if name in MOVE_DIRECTIONS else "click" if name == "ACTION6" else "other"
        return {
            "action_name": name,
            "action_family": family,
            "kind": "move_destination" if family == "move" else "unknown_target",
            "requested_direction": MOVE_DIRECTIONS.get(name, "none"),
        }

    left_view, right_view = base(left), base(right)
    if left_view["action_family"] == right_view["action_family"] == "click":
        left_relation, right_relation = _relative_click_descriptors(
            left.get("action_data", {}),
            right.get("action_data", {}),
        )
        left_view.update(left_relation)
        right_view.update(right_relation)
    return left_view, right_view


def _source_context(row: Mapping[str, Any]) -> dict[str, Any]:
    atoms = [
        str(atom)
        for atom in row.get("atoms_before", ())
        if str(atom).startswith(("progress:", "state:"))
    ]
    roles = [
        str(atom).split("(", 1)[1].rstrip(")")
        for atom in row.get("atoms_before", ())
        if str(atom).startswith("object:role_present(")
    ]
    aspect_counts = Counter(
        role.rsplit(":", 1)[-1] if ":" in role else role for role in roles
    )
    return {
        "state_atoms": sorted(atoms),
        "role_aspect_counts": {
            key: _bucket(value) for key, value in sorted(aspect_counts.items())
        },
    }


def _context_from_event_views(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    effect_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    for event in events[-8:]:
        family_counts[str(event["action_family"])] += 1
        for effect, value in event["observed_effects"].items():
            if value:
                effect_counts[str(effect)] += 1
        binding = event["binding"]
        for name in RELATION_FIELDS:
            relation_counts[f"{name}:{binding[name]}"] += 1
    return {
        "recent_action_family_counts": {
            key: _bucket(value) for key, value in sorted(family_counts.items())
        },
        "recent_effect_counts": {
            key: _bucket(value) for key, value in sorted(effect_counts.items())
        },
        "recent_relation_counts": {
            key: _bucket(value) for key, value in sorted(relation_counts.items())
        },
    }


def _slot_descriptor(
    example: SlotExample,
    signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_name": example.slot.action_name,
        **dict(signature or example.slot.semantic_signature),
    }


def _primary_contrast(
    left: Mapping[str, bool],
    right: Mapping[str, bool],
) -> str:
    for effect in (
        "level_complete",
        "game_over",
        "progress",
        "moved",
        "changed",
    ):
        if bool(left[effect]) != bool(right[effect]):
            return effect
    return "neutral"


def _build_source_pairs(sage11_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = (
        defaultdict(dict)
    )
    raw_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for game in SOURCE_TRAIN:
        path = sage11_dir / "shards" / f"{game}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("source_split") != "source_train":
                    raise ValueError(f"non-training row in source shard {game}")
                raw_counts[game]["rows"] += 1
                labels = _source_labels(row)
                for effect, value in labels.items():
                    if value:
                        raw_counts[game][effect] += 1
                key = (game, str(row["state_digest_before"]))
                action = _action_key(row)
                old = groups[key].get(action)
                priority = tuple(
                    int(labels[name])
                    for name in (
                        "level_complete",
                        "game_over",
                        "progress",
                        "changed",
                        "moved",
                    )
                )
                if old is None or priority > old["_priority"]:
                    groups[key][action] = {**row, "_priority": priority}

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    capacity = Counter()
    for (game, state_digest), action_map in groups.items():
        rows = sorted(action_map.values(), key=_action_key)
        if len(rows) < 2:
            continue
        capacity["states_with_multiple_actions"] += 1
        for left, right in itertools.combinations(rows, 2):
            left_labels = _source_labels(left)
            right_labels = _source_labels(right)
            contrast = _primary_contrast(left_labels, right_labels)
            capacity["all_pairs"] += 1
            capacity[f"{contrast}_pairs"] += 1
            left_view, right_view = _source_descriptors(left, right)
            pair_id = "s11_" + _checksum(
                {
                    "game": game,
                    "state": state_digest,
                    "left": _action_key(left),
                    "right": _action_key(right),
                }
            )[:20]
            candidates[(game, contrast)].append(
                {
                    "format_version": FORMAT_VERSION,
                    "pair_id": pair_id,
                    "game_id": game,
                    "origin": "sage11_same_prestate",
                    "audit": {
                        "state_digest": state_digest,
                        "left_action_key": list(_action_key(left)),
                        "right_action_key": list(_action_key(right)),
                    },
                    "context": _source_context(left),
                    "left": left_view,
                    "right": right_view,
                    "class_targets": {
                        effect: _pair_class(
                            left_labels[effect], right_labels[effect]
                        )
                        for effect in TRAIN_EFFECTS
                    },
                    "label_mask": {
                        effect: effect
                        not in ("target_created", "target_removed", "target_moved")
                        for effect in TRAIN_EFFECTS
                    },
                    "contrast_bucket": contrast,
                }
            )

    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for game in SOURCE_TRAIN:
        for contrast, cap in SOURCE_BUCKET_CAPS.items():
            rows = candidates.get((game, contrast), [])
            ordered = sorted(
                rows,
                key=lambda row: _checksum({"seed": SEED, "id": row["pair_id"]}),
            )
            subset = ordered[:cap]
            selected.extend(subset)
            selected_counts[game][contrast] += len(subset)
    selected.sort(key=lambda row: (row["game_id"], row["pair_id"]))
    return selected, {
        "raw_per_game": {
            game: dict(raw_counts[game]) for game in SOURCE_TRAIN
        },
        "capacity": dict(capacity),
        "selected_per_game": {
            game: dict(selected_counts[game]) for game in SOURCE_TRAIN
        },
        "selected_pairs": len(selected),
    }


def _build_v43_pairs(
    roots: Sequence[ExecutedRoot],
) -> tuple[list[dict[str, Any]], tuple[SlotExample, ...]]:
    examples = load_slot_examples(roots)
    result = []
    for left, right in _nodes(examples):
        labels = {
            effect: _pair_class(
                bool(left.labels[effect]), bool(right.labels[effect])
            )
            for effect in SLOT_EFFECTS
        }
        labels["progress"] = _pair_class(
            left.utility > 0.0, right.utility > 0.0
        )
        result.append(
            {
                "format_version": FORMAT_VERSION,
                "pair_id": "v43_" + _checksum(left.node_id)[:20],
                "game_id": left.game_id,
                "origin": "v43_replay_verified",
                "audit": {"node_id": left.node_id},
                "context": _context_from_event_views(
                    [_event_view(event) for event in left.context[-8:]]
                ),
                "left": _slot_descriptor(left),
                "right": _slot_descriptor(right),
                "class_targets": labels,
                "label_mask": {effect: True for effect in TRAIN_EFFECTS},
                "contrast_bucket": _primary_contrast(
                    {**left.labels, "progress": left.utility > 0.0},
                    {**right.labels, "progress": right.utility > 0.0},
                ),
            }
        )
    return result, examples


def render_pair_prompt(
    pair: Mapping[str, Any],
    *,
    representation: str,
    swapped: bool = False,
) -> str:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation}")
    left_key, right_key = ("right", "left") if swapped else ("left", "right")
    payload = {
        "task": (
            "compare two legal interventions from the same pre-state; "
            "encode which effects are caused by left and right"
        ),
        "common": (
            pair.get("context", {})
            if representation == "invariant_context"
            else {}
        ),
        "left": pair[left_key],
        "right": pair[right_key],
    }
    prompt = _canonical(payload)
    lowered = prompt.lower()
    for forbidden in FORBIDDEN_MODEL_FIELDS:
        if forbidden in lowered:
            raise ValueError(f"forbidden model-input token: {forbidden}")
    game = str(pair.get("game_id", ""))
    if game and game.lower() in lowered:
        raise ValueError("game identity leaked into pair prompt")
    if "state_digest" in lowered or "node_id" in lowered or '"x"' in lowered or '"y"' in lowered:
        raise ValueError("audit identity or raw coordinates leaked into pair prompt")
    return prompt


def _source_fingerprints(sage11_dir: Path, v43_dir: Path) -> dict[str, Any]:
    source = []
    for game in SOURCE_TRAIN:
        path = sage11_dir / "shards" / f"{game}.jsonl"
        source.append(
            {
                "game": game,
                "rows": sum(1 for _ in path.open(encoding="utf-8")),
                "sha256": _file_sha256(path),
            }
        )
    v43 = []
    for path in sorted((v43_dir / "source_train_shards").glob("*.jsonl")):
        v43.append(
            {
                "path": path.as_posix(),
                "sha256": _file_sha256(path),
            }
        )
    return {"sage11_source_train": source, "v43_source_train": v43}


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sage11_dir: str | Path = DEFAULT_SAGE11_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    destination = Path(output_dir)
    sage11 = Path(sage11_dir)
    v43 = Path(v43_dir)
    roots = load_complete_roots(v43)
    source_pairs, source_preflight = _build_source_pairs(sage11)
    tree_pairs, examples = _build_v43_pairs(roots)
    corpus = source_pairs + tree_pairs
    _write_jsonl(destination / "pair_corpus.jsonl", corpus)
    by_origin = Counter(row["origin"] for row in corpus)
    by_effect_contrast = {
        effect: sum(
            int(row["class_targets"][effect] in (1, 2))
            for row in corpus
            if row["label_mask"][effect]
        )
        for effect in TRAIN_EFFECTS
    }
    fingerprints = _source_fingerprints(sage11, v43)
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "created_for": FORMAT_VERSION,
        "source_train_games": list(SOURCE_TRAIN),
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "source_fingerprints": fingerprints,
        "pair_corpus": {
            "rows": len(corpus),
            "by_origin": dict(by_origin),
            "contrasts_by_effect": by_effect_contrast,
            "sha256": _file_sha256(destination / "pair_corpus.jsonl"),
            "sampling_caps_per_game": SOURCE_BUCKET_CAPS,
            "source_preflight": source_preflight,
        },
        "evaluation_population": {
            "complete_roots": len(roots),
            "complete_nodes": len(_nodes(examples)),
            "semantic_slots": len(examples),
        },
        "qwen_encoder": {
            "model_path": Path(model_path).as_posix(),
            "model_sha256": _file_sha256(Path(model_path) / "model.safetensors"),
            "frozen": True,
            "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
            "batch_size": QWEN_BATCH_SIZE,
            "representations": list(REPRESENTATIONS),
            "selection": (
                "lowest SAGE11-only LOGO macro Brier; then lower output "
                "identity accuracy; then minimal"
            ),
        },
        "adapter": {
            "kind": "external_low_rank_residual_adapter",
            "rank": ADAPTER_RANK,
            "epochs": ADAPTER_EPOCHS,
            "batch_size": ADAPTER_BATCH_SIZE,
            "learning_rate": ADAPTER_LEARNING_RATE,
            "pair_classes": list(PAIR_CLASSES),
            "effects": list(TRAIN_EFFECTS),
            "base_qwen_weights_trainable": False,
            "every_prediction_leave_one_game_out": True,
        },
        "downstream_freeze": {
            "world_model": "V4.7 RegularizedSlotWorldModel unchanged",
            "ebm_input_width": EBM_INPUT_WIDTH,
            "ebm_hidden_width": EBM_HIDDEN_WIDTH,
            "ebm_epochs": EBM_EPOCHS,
            "ebm_learning_rate": EBM_LEARNING_RATE,
            "controller": "V4.7 depth-three first-action selection unchanged",
            "baseline_selection": "training games only in each outer fold",
        },
        "decision_thresholds": {
            "adapter_brier_better_than_action_only": True,
            "adapted_over_structured_mean_gain_strictly_positive": True,
            "adapted_over_primary_baseline_mean_gain_strictly_positive": True,
            "nonnegative_games_minimum": 6,
            "original_over_relation_shuffle_mean_gain_strictly_positive": True,
            "completion_events_selected_minimum": 2,
            "semantic_output_identity_accuracy_maximum": 0.60,
            "confirmatory_ci_required": False,
        },
        "confirmatory_gate": False,
        "authority_promotion_allowed": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    path = Path(output_dir) / "frozen_manifest.json"
    manifest = _read_json(path)
    checksum = manifest.pop("manifest_checksum")
    if checksum != _checksum(manifest):
        raise ValueError("V4.8 manifest checksum mismatch")
    manifest["manifest_checksum"] = checksum
    corpus_path = Path(output_dir) / "pair_corpus.jsonl"
    if _file_sha256(corpus_path) != manifest["pair_corpus"]["sha256"]:
        raise ValueError("V4.8 pair corpus checksum mismatch")
    if tuple(manifest["source_train_games"]) != SOURCE_TRAIN:
        raise ValueError("source-train registry mismatch")
    return manifest


class FrozenQwenPairEncoder:
    """Extract one frozen last-token vector per comparison prompt."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str,
        batch_size: int = QWEN_BATCH_SIZE,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.batch_size = int(batch_size)

    def encode(
        self,
        prompts: Sequence[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map={"": self.device},
            local_files_only=True,
        )
        model.eval()
        vectors: list[np.ndarray] = []
        lengths: list[int] = []
        seconds: list[float] = []
        for offset in range(0, len(prompts), self.batch_size):
            subset = prompts[offset : offset + self.batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Represent this causal action comparison "
                                "without guessing a game identity."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in subset
            ]
            inputs = tokenizer(
                rendered,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            batch_lengths = inputs["attention_mask"].sum(dim=1)
            maximum = int(batch_lengths.max().item())
            if maximum > MAXIMUM_INPUT_TOKENS:
                raise RuntimeError(
                    f"pair prompt exceeds token cap: {maximum} > "
                    f"{MAXIMUM_INPUT_TOKENS}"
                )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                hidden = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                index = batch_lengths.to(model.device) - 1
                pooled = hidden[
                    torch.arange(hidden.shape[0], device=model.device), index
                ]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
            vectors.append(pooled.float().cpu().numpy())
            lengths.extend(int(value) for value in batch_lengths.tolist())
        matrix = np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0))
        return matrix, {
            "device": self.device,
            "prompts": len(prompts),
            "batches": len(seconds),
            "inference_seconds": float(sum(seconds)),
            "median_batch_seconds": float(statistics.median(seconds)),
            "mean_input_tokens": float(statistics.fmean(lengths)),
            "maximum_input_tokens": max(lengths),
            "hidden_width": int(matrix.shape[1]),
        }


def _relation_shuffle_pairs(
    examples: Sequence[SlotExample],
) -> list[dict[str, Any]]:
    result = []
    for node in _nodes(examples):
        left, right = node
        if left.path != "":
            continue
        context, signatures = _shuffle_relation_payload(
            left.context, node, salt=left.node_id
        )
        result.append(
            {
                "format_version": FORMAT_VERSION,
                "pair_id": f"shuffle_{left.node_id}",
                "game_id": left.game_id,
                "origin": "v43_relation_shuffle",
                "audit": {"node_id": left.node_id},
                "context": _context_from_event_views(context),
                "left": _slot_descriptor(left, signatures[0]),
                "right": _slot_descriptor(right, signatures[1]),
                "class_targets": {effect: 0 for effect in TRAIN_EFFECTS},
                "label_mask": {effect: False for effect in TRAIN_EFFECTS},
            }
        )
    return result


def extract_embeddings(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    corpus = _read_jsonl(destination / "pair_corpus.jsonl")
    examples = load_slot_examples(load_complete_roots(v43_dir))
    shuffled = _relation_shuffle_pairs(examples)
    prompt_map: dict[str, str] = {}
    index_rows = []
    for pair in corpus + shuffled:
        row = {"pair_id": pair["pair_id"], "prompts": {}}
        for representation in REPRESENTATIONS:
            row["prompts"][representation] = {}
            for orientation, swapped in (("original", False), ("swapped", True)):
                prompt = render_pair_prompt(
                    pair, representation=representation, swapped=swapped
                )
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                prompt_map.setdefault(digest, prompt)
                row["prompts"][representation][orientation] = digest
        index_rows.append(row)
    hashes = sorted(prompt_map)
    encoder = FrozenQwenPairEncoder(
        model_path=manifest["qwen_encoder"]["model_path"],
        device=device,
        batch_size=int(manifest["qwen_encoder"]["batch_size"]),
    )
    vectors, runtime = encoder.encode([prompt_map[key] for key in hashes])
    np.savez_compressed(
        destination / "embedding_cache.npz",
        hashes=np.asarray(hashes),
        vectors=vectors.astype(np.float16),
    )
    _write_jsonl(destination / "embedding_index.jsonl", index_rows)
    summary = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "corpus_pairs": len(corpus),
        "relation_shuffle_pairs": len(shuffled),
        "unique_prompts": len(hashes),
        "runtime": runtime,
        "cache_sha256": _file_sha256(destination / "embedding_cache.npz"),
        "index_sha256": _file_sha256(destination / "embedding_index.jsonl"),
        "base_qwen_weights_updated": False,
    }
    _write_json(destination / "embedding_summary.json", summary)
    return summary


def _load_embedding_lookup(
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    payload = np.load(output_dir / "embedding_cache.npz")
    vectors = payload["vectors"].astype(np.float32)
    lookup = {
        str(key): vectors[index]
        for index, key in enumerate(payload["hashes"].tolist())
    }
    index = {
        row["pair_id"]: row["prompts"]
        for row in _read_jsonl(output_dir / "embedding_index.jsonl")
    }
    return lookup, index


def _record_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    representation: str,
    lookup: Mapping[str, np.ndarray],
    index: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original = np.stack(
        [
            lookup[index[row["pair_id"]][representation]["original"]]
            for row in records
        ]
    )
    swapped = np.stack(
        [
            lookup[index[row["pair_id"]][representation]["swapped"]]
            for row in records
        ]
    )
    targets = np.asarray(
        [
            [int(row["class_targets"][effect]) for effect in TRAIN_EFFECTS]
            for row in records
        ],
        dtype=np.int64,
    )
    mask = np.asarray(
        [
            [bool(row["label_mask"][effect]) for effect in TRAIN_EFFECTS]
            for row in records
        ],
        dtype=bool,
    )
    return original, swapped, targets, mask


def _fit_adapter(
    original: np.ndarray,
    swapped: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    class Adapter(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.down = nn.Linear(width, ADAPTER_RANK, bias=False)
            self.up = nn.Linear(ADAPTER_RANK, width, bias=False)
            self.norm = nn.LayerNorm(width)
            self.heads = nn.ModuleList(
                nn.Linear(width, len(PAIR_CLASSES)) for _ in TRAIN_EFFECTS
            )
            nn.init.zeros_(self.up.weight)

        def forward(self, values: Any) -> Any:
            adapted = values + self.up(torch.nn.functional.gelu(self.down(values)))
            adapted = self.norm(adapted)
            return torch.stack([head(adapted) for head in self.heads], dim=1)

    augmented_x = np.concatenate((original, swapped), axis=0)
    swapped_targets = SWAP_CLASS_INDEX[targets]
    augmented_y = np.concatenate((targets, swapped_targets), axis=0)
    augmented_mask = np.concatenate((mask, mask), axis=0)
    x = torch.as_tensor(augmented_x, dtype=torch.float32)
    y = torch.as_tensor(augmented_y, dtype=torch.long)
    active = torch.as_tensor(augmented_mask, dtype=torch.bool)
    model = Adapter(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ADAPTER_LEARNING_RATE,
        weight_decay=0.01,
    )
    weights = []
    for effect_index in range(len(TRAIN_EFFECTS)):
        counts = np.bincount(
            augmented_y[augmented_mask[:, effect_index], effect_index],
            minlength=len(PAIR_CLASSES),
        ).astype(np.float64)
        present = counts > 0
        values = np.ones(len(PAIR_CLASSES), dtype=np.float64)
        if np.any(present):
            values[present] = np.clip(
                counts[present].sum() / (len(PAIR_CLASSES) * counts[present]),
                0.25,
                8.0,
            )
        weights.append(torch.as_tensor(values, dtype=torch.float32, device=device))
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    for _epoch in range(ADAPTER_EPOCHS):
        order = rng.permutation(len(x))
        epoch_losses = []
        for offset in range(0, len(order), ADAPTER_BATCH_SIZE):
            indices = order[offset : offset + ADAPTER_BATCH_SIZE]
            batch_x = x[indices].to(device)
            batch_y = y[indices].to(device)
            batch_mask = active[indices].to(device)
            logits = model(batch_x)
            total = torch.zeros((), device=device)
            terms = 0
            for effect_index in range(len(TRAIN_EFFECTS)):
                selected = batch_mask[:, effect_index]
                if not bool(selected.any()):
                    continue
                total = total + torch.nn.functional.cross_entropy(
                    logits[selected, effect_index],
                    batch_y[selected, effect_index],
                    weight=weights[effect_index],
                )
                terms += 1
            if terms == 0:
                continue
            loss = total / terms
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    return model.eval(), {
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "epochs": len(losses),
        "trainable_parameters": trainable,
        "state_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
    }


def _predict_adapter(
    model: Any,
    original: np.ndarray,
    swapped: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    import torch

    result = []
    with torch.inference_mode():
        for offset in range(0, len(original), ADAPTER_BATCH_SIZE):
            left = torch.as_tensor(
                original[offset : offset + ADAPTER_BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            right = torch.as_tensor(
                swapped[offset : offset + ADAPTER_BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            p_original = torch.softmax(model(left), dim=2).cpu().numpy()
            p_swapped = torch.softmax(model(right), dim=2).cpu().numpy()
            restored = p_swapped[:, :, SWAP_CLASS_INDEX]
            result.append(0.5 * (p_original + restored))
    return np.concatenate(result, axis=0)


def _arm_probabilities(class_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = class_probabilities[:, :, 1] + class_probabilities[:, :, 3]
    right = class_probabilities[:, :, 2] + class_probabilities[:, :, 3]
    return left, right


def _direct_metrics(
    records: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    left, right = _arm_probabilities(probabilities)
    effects = {}
    for effect_index, effect in enumerate(TRAIN_EFFECTS):
        targets = []
        predicted = []
        class_correct = []
        for row_index, row in enumerate(records):
            if not row["label_mask"][effect]:
                continue
            target_class = int(row["class_targets"][effect])
            targets.extend(
                (
                    int(target_class in (1, 3)),
                    int(target_class in (2, 3)),
                )
            )
            predicted.extend(
                (left[row_index, effect_index], right[row_index, effect_index])
            )
            class_correct.append(
                int(np.argmax(probabilities[row_index, effect_index]) == target_class)
            )
        target = np.asarray(targets, dtype=np.float64)
        predicted_array = np.asarray(predicted, dtype=np.float64)
        positive = target == 1
        effects[effect] = {
            "arms": len(target),
            "positives": int(target.sum()),
            "brier": float(np.mean((predicted_array - target) ** 2)),
            "recall_at_0_5": (
                float(np.mean(predicted_array[positive] >= 0.5))
                if np.any(positive)
                else 1.0
            ),
            "pair_class_accuracy": float(np.mean(class_correct)),
        }
    return {
        "effects": effects,
        "macro_brier": float(np.mean([row["brier"] for row in effects.values()])),
        "macro_pair_class_accuracy": float(
            np.mean([row["pair_class_accuracy"] for row in effects.values()])
        ),
    }


def _identity_accuracy(
    records: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    matrix = probabilities.reshape(len(probabilities), -1)
    labels = np.asarray([row["game_id"] for row in records])
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    counts = Counter(labels)
    folds = min(5, min(counts.values()))
    if folds < 2:
        return {"majority_accuracy": float(majority), "accuracy": 1.0}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    score = float(np.mean(cross_val_score(model, matrix, labels, cv=splitter)))
    return {"majority_accuracy": float(majority), "accuracy": score}


def _action_only_oof(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    result = np.zeros(
        (len(records), len(TRAIN_EFFECTS), len(PAIR_CLASSES)), dtype=np.float64
    )
    games = sorted({row["game_id"] for row in records})
    for held in games:
        train = [row for row in records if row["game_id"] != held]
        indices = [i for i, row in enumerate(records) if row["game_id"] == held]
        counts: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [1, 1]
        )
        global_counts: dict[str, list[int]] = defaultdict(lambda: [1, 1])
        for row in train:
            for side_index, side in enumerate(("left", "right")):
                name = str(row[side]["action_name"])
                for effect_index, effect in enumerate(TRAIN_EFFECTS):
                    if not row["label_mask"][effect]:
                        continue
                    cls = int(row["class_targets"][effect])
                    target = int(
                        cls in ((1, 3) if side_index == 0 else (2, 3))
                    )
                    counts[(name, effect)][target] += 1
                    global_counts[effect][target] += 1
        for index in indices:
            row = records[index]
            arm = []
            for side in ("left", "right"):
                values = []
                for effect in TRAIN_EFFECTS:
                    count = counts.get(
                        (str(row[side]["action_name"]), effect),
                        global_counts[effect],
                    )
                    values.append(count[1] / sum(count))
                arm.append(values)
            for effect_index in range(len(TRAIN_EFFECTS)):
                p_left, p_right = arm[0][effect_index], arm[1][effect_index]
                result[index, effect_index] = (
                    (1 - p_left) * (1 - p_right),
                    p_left * (1 - p_right),
                    (1 - p_left) * p_right,
                    p_left * p_right,
                )
    return result


def run_semantic_adaptation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    corpus = _read_jsonl(destination / "pair_corpus.jsonl")
    lookup, index = _load_embedding_lookup(destination)
    source = [row for row in corpus if row["origin"] == "sage11_same_prestate"]
    representation_rows = {}
    representation_predictions = {}
    fold_runtime: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        probabilities = np.zeros(
            (len(source), len(TRAIN_EFFECTS), len(PAIR_CLASSES)),
            dtype=np.float64,
        )
        original, swapped, targets, mask = _record_matrix(
            source,
            representation=representation,
            lookup=lookup,
            index=index,
        )
        for fold_index, held in enumerate(SOURCE_TRAIN):
            train_indices = np.asarray(
                [i for i, row in enumerate(source) if row["game_id"] != held]
            )
            valid_indices = np.asarray(
                [i for i, row in enumerate(source) if row["game_id"] == held]
            )
            model, fit = _fit_adapter(
                original[train_indices],
                swapped[train_indices],
                targets[train_indices],
                mask[train_indices],
                seed=SEED + 100 * fold_index + 10 * REPRESENTATIONS.index(representation),
                device=device,
            )
            probabilities[valid_indices] = _predict_adapter(
                model,
                original[valid_indices],
                swapped[valid_indices],
                device=device,
            )
            fold_runtime.append(
                {
                    "stage": "representation_selection",
                    "representation": representation,
                    "held_out_game": held,
                    **fit,
                }
            )
        metrics = _direct_metrics(source, probabilities)
        identity = _identity_accuracy(source, probabilities)
        representation_rows[representation] = {
            "direct_metrics": metrics,
            "semantic_output_identity": identity,
        }
        representation_predictions[representation] = probabilities
    selected = min(
        REPRESENTATIONS,
        key=lambda name: (
            representation_rows[name]["direct_metrics"]["macro_brier"],
            representation_rows[name]["semantic_output_identity"]["accuracy"],
            REPRESENTATIONS.index(name),
        ),
    )

    v43_records = [row for row in corpus if row["origin"] == "v43_replay_verified"]
    shuffled_records = [
        {
            "format_version": FORMAT_VERSION,
            "pair_id": row["pair_id"],
            "game_id": row["pair_id"].split("_", 1)[1].split(":", 1)[0],
            "origin": "v43_relation_shuffle",
            "left": {},
            "right": {},
            "context": {},
            "class_targets": {effect: 0 for effect in TRAIN_EFFECTS},
            "label_mask": {effect: False for effect in TRAIN_EFFECTS},
        }
        for row in _read_jsonl(destination / "embedding_index.jsonl")
        if row["pair_id"].startswith("shuffle_")
    ]
    original_all, swapped_all, targets_all, mask_all = _record_matrix(
        corpus,
        representation=selected,
        lookup=lookup,
        index=index,
    )
    corpus_position = {row["pair_id"]: i for i, row in enumerate(corpus)}
    v43_probabilities = np.zeros(
        (len(v43_records), len(TRAIN_EFFECTS), len(PAIR_CLASSES)),
        dtype=np.float64,
    )
    v43_position = {row["pair_id"]: i for i, row in enumerate(v43_records)}
    shuffled_probabilities: dict[str, np.ndarray] = {}
    for fold_index, held in enumerate(SOURCE_TRAIN):
        train_indices = np.asarray(
            [i for i, row in enumerate(corpus) if row["game_id"] != held]
        )
        held_records = [row for row in v43_records if row["game_id"] == held]
        held_indices = np.asarray(
            [corpus_position[row["pair_id"]] for row in held_records]
        )
        model, fit = _fit_adapter(
            original_all[train_indices],
            swapped_all[train_indices],
            targets_all[train_indices],
            mask_all[train_indices],
            seed=SEED + 1000 + fold_index,
            device=device,
        )
        held_probs = _predict_adapter(
            model,
            original_all[held_indices],
            swapped_all[held_indices],
            device=device,
        )
        for row, values in zip(held_records, held_probs):
            v43_probabilities[v43_position[row["pair_id"]]] = values
        held_shuffled = [
            row for row in shuffled_records if row["game_id"] == held
        ]
        if held_shuffled:
            so, ss, _, _ = _record_matrix(
                held_shuffled,
                representation=selected,
                lookup=lookup,
                index=index,
            )
            predicted = _predict_adapter(model, so, ss, device=device)
            shuffled_probabilities.update(
                {
                    row["pair_id"].removeprefix("shuffle_"): values
                    for row, values in zip(held_shuffled, predicted)
                }
            )
        fold_runtime.append(
            {
                "stage": "combined_logo_semantics",
                "representation": selected,
                "held_out_game": held,
                **fit,
            }
        )

    annotations = []
    for row, probabilities in zip(v43_records, v43_probabilities):
        left, right = _arm_probabilities(probabilities[None, ...])
        node_id = row["audit"]["node_id"]
        annotations.append(
            {
                "format_version": FORMAT_VERSION,
                "node_id": node_id,
                "game_id": row["game_id"],
                "variant": "original",
                "left": {
                    effect: float(left[0, TRAIN_EFFECTS.index(effect)])
                    for effect in SLOT_EFFECTS
                },
                "right": {
                    effect: float(right[0, TRAIN_EFFECTS.index(effect)])
                    for effect in SLOT_EFFECTS
                },
            }
        )
    for node_id, probabilities in sorted(shuffled_probabilities.items()):
        left, right = _arm_probabilities(probabilities[None, ...])
        annotations.append(
            {
                "format_version": FORMAT_VERSION,
                "node_id": node_id,
                "game_id": node_id.split(":", 1)[0],
                "variant": "relation_shuffle",
                "left": {
                    effect: float(left[0, TRAIN_EFFECTS.index(effect)])
                    for effect in SLOT_EFFECTS
                },
                "right": {
                    effect: float(right[0, TRAIN_EFFECTS.index(effect)])
                    for effect in SLOT_EFFECTS
                },
            }
        )
    _write_jsonl(destination / "semantic_annotations.jsonl", annotations)

    action_baseline = _action_only_oof(v43_records)
    direct = _direct_metrics(v43_records, v43_probabilities)
    direct_action = _direct_metrics(v43_records, action_baseline)
    semantic_identity = _identity_accuracy(v43_records, v43_probabilities)
    result: dict[str, Any] = {
        "format_version": SEMANTIC_RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "selected_representation": selected,
        "representation_selection": representation_rows,
        "source_action_only": _direct_metrics(
            source, _action_only_oof(source)
        ),
        "v43_logo_direct_metrics": direct,
        "v43_action_only_direct_metrics": direct_action,
        "v43_macro_brier_gain_over_action_only": (
            direct_action["macro_brier"] - direct["macro_brier"]
        ),
        "semantic_output_identity": semantic_identity,
        "fold_training": fold_runtime,
        "annotations_sha256": _file_sha256(
            destination / "semantic_annotations.jsonl"
        ),
        "base_qwen_weights_updated": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "semantic_result.json", result)
    return result


def _load_semantic_annotations(
    output_dir: Path,
    examples: Sequence[SlotExample],
) -> tuple[dict[str, SlotAnnotation], dict[str, SlotAnnotation]]:
    by_node = {
        (row["node_id"], row["variant"]): row
        for row in _read_jsonl(output_dir / "semantic_annotations.jsonl")
    }
    original: dict[str, SlotAnnotation] = {}
    shuffled: dict[str, SlotAnnotation] = {}
    for left, right in _nodes(examples):
        row = by_node[(left.node_id, "original")]
        for item, side in ((left, "left"), (right, "right")):
            original[item.slot.slot_id] = SlotAnnotation(
                slot_id=item.slot.slot_id,
                effect_probabilities=row[side],
                source="qwen_pair_low_rank_logo",
                support=0,
            )
        shuffled_row = by_node.get((left.node_id, "relation_shuffle"))
        if shuffled_row is not None:
            for item, side in ((left, "left"), (right, "right")):
                shuffled[item.slot.slot_id] = SlotAnnotation(
                    slot_id=item.slot.slot_id,
                    effect_probabilities=shuffled_row[side],
                    source="qwen_pair_low_rank_relation_shuffle_logo",
                    support=0,
                )
    return original, shuffled


def _annotation_identity(
    examples: Sequence[SlotExample],
    annotations: Mapping[str, SlotAnnotation],
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    matrix = np.asarray(
        [
            [annotations[item.slot.slot_id].effect_probabilities[e] for e in SLOT_EFFECTS]
            for item in examples
        ]
    )
    labels = np.asarray([item.game_id for item in examples])
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    return {
        "majority_accuracy": float(majority),
        "accuracy": float(np.mean(cross_val_score(model, matrix, labels, cv=folds))),
    }


def _completion_capture(
    roots: Sequence[ExecutedRoot],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    opportunities = {}
    for root in roots:
        completing = []
        for side in "LR":
            trace = root.arm("", side).trace
            if (
                trace.effects.level_complete
                or trace.levels_completed_after > trace.levels_completed_before
                or str(trace.game_state_after).upper() == "WIN"
            ):
                completing.append(side)
        if completing:
            opportunities[root.root_key] = set(completing)
    by_method: dict[str, int] = Counter()
    for row in decisions:
        if row["root_key"] in opportunities:
            by_method[row["method"]] += int(
                row["selected_side"] in opportunities[row["root_key"]]
            )
    return {
        "opportunities": len(opportunities),
        "selected_by_method": dict(by_method),
    }


def evaluate_end_to_end(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    v47_dir: str | Path = DEFAULT_V47_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    semantic = _read_json(destination / "semantic_result.json")
    roots = load_complete_roots(v43_dir)
    examples = load_slot_examples(roots)
    adapted, shuffled = _load_semantic_annotations(destination, examples)
    oracle = {item.slot.slot_id: item.annotation() for item in examples}
    by_position = {
        (item.root_key, item.path, item.side): item for item in examples
    }
    games = sorted({root.game_id for root in roots})
    decisions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    held_predictions = {"structured": {}, "adapted": {}, "oracle": {}}

    for fold_index, held in enumerate(games):
        training_roots = tuple(root for root in roots if root.game_id != held)
        validation_roots = tuple(root for root in roots if root.game_id == held)
        training = tuple(item for item in examples if item.game_id != held)
        validation = tuple(item for item in examples if item.game_id == held)
        structured_model, structured_oof = _fit_nested_world(
            training,
            annotations=None,
            use_annotations=False,
            seed=SEED + 100 * fold_index,
        )
        adapted_model, adapted_oof = _fit_nested_world(
            training,
            annotations=adapted,
            use_annotations=True,
            seed=SEED + 100 * fold_index + 10,
        )
        oracle_model, oracle_oof = _fit_nested_world(
            training,
            annotations=oracle,
            use_annotations=True,
            seed=SEED + 100 * fold_index + 20,
        )
        structured_validation = structured_model.predict(validation)
        adapted_validation = adapted_model.predict(validation, adapted)
        oracle_validation = oracle_model.predict(validation, oracle)
        held_predictions["structured"].update(structured_validation)
        held_predictions["adapted"].update(adapted_validation)
        held_predictions["oracle"].update(oracle_validation)
        ebms = {
            "structured": _train_ebm(
                training_roots,
                structured_oof,
                by_position,
                depth=3,
                seed=SEED + 1000 + fold_index,
            ),
            "adapted": _train_ebm(
                training_roots,
                adapted_oof,
                by_position,
                depth=3,
                seed=SEED + 2000 + fold_index,
            ),
            "oracle": _train_ebm(
                training_roots,
                oracle_oof,
                by_position,
                depth=3,
                seed=SEED + 3000 + fold_index,
            ),
        }
        primary = _select_primary_baseline(training_roots)
        folds.append(
            {
                "format_version": FORMAT_VERSION,
                "held_out_game": held,
                "training_games": sorted({item.game_id for item in training}),
                "primary_baseline": primary,
                "training_slots": len(training),
                "validation_slots": len(validation),
                "ebm_pairs": {
                    name: model.trained_pairs for name, model in ebms.items()
                },
            }
        )
        for root in validation_roots:
            baseline_paths = {
                "deterministic_left": "L",
                "action_only": _action_only_choice(root, training_roots),
                "action_sequence_only": _sequence_only_choice(root, training_roots),
            }
            for name, path in baseline_paths.items():
                decisions.append(_decision_row(root, method=name, selected_path=path))
            decisions.append(
                _decision_row(
                    root,
                    method="primary_baseline",
                    selected_path=baseline_paths[primary],
                    baseline_method=primary,
                )
            )
            structured_path = _select_path(
                root,
                structured_validation,
                by_position,
                ebms["structured"],
                depth=3,
            )
            adapted_path = _select_path(
                root,
                adapted_validation,
                by_position,
                ebms["adapted"],
                depth=3,
            )
            oracle_path = _select_path(
                root,
                oracle_validation,
                by_position,
                ebms["oracle"],
                depth=3,
            )
            decisions.extend(
                (
                    _decision_row(
                        root,
                        method="structured_depth3_ebm",
                        selected_path=structured_path,
                    ),
                    _decision_row(
                        root,
                        method="adapted_semantics_depth3_ebm",
                        selected_path=adapted_path,
                    ),
                    _decision_row(
                        root,
                        method="oracle_annotation_depth3_ebm",
                        selected_path=oracle_path,
                    ),
                )
            )
            root_items = tuple(
                by_position[(root.root_key, "", side)] for side in "LR"
            )
            shuffled_map = dict(adapted)
            for item in root_items:
                shuffled_map[item.slot.slot_id] = shuffled[item.slot.slot_id]
            shuffled_root_predictions = adapted_model.predict(
                root_items, shuffled_map
            )
            shuffle_predictions = dict(adapted_validation)
            shuffle_predictions.update(shuffled_root_predictions)
            shuffle_path = _select_path(
                root,
                shuffle_predictions,
                by_position,
                ebms["adapted"],
                depth=3,
            )
            decisions.append(
                _decision_row(
                    root,
                    method="adapted_relation_shuffle_depth3_ebm",
                    selected_path=shuffle_path,
                )
            )

    _write_jsonl(destination / "decisions.jsonl", decisions)
    _write_jsonl(destination / "folds.jsonl", folds)
    methods = sorted({row["method"] for row in decisions})
    by_method = {
        method: [row for row in decisions if row["method"] == method]
        for method in methods
    }
    metrics = {
        method: _summarize(rows) for method, rows in by_method.items()
    }
    adapted_rows = by_method["adapted_semantics_depth3_ebm"]
    baseline_rows = by_method["primary_baseline"]
    comparisons = {
        "adapted_over_primary_baseline": _paired_bootstrap(
            adapted_rows, baseline_rows, seed=SEED
        ),
        "adapted_over_structured": _paired_bootstrap(
            adapted_rows,
            by_method["structured_depth3_ebm"],
            seed=SEED + 1,
        ),
        "original_over_relation_shuffle": _paired_bootstrap(
            adapted_rows,
            by_method["adapted_relation_shuffle_depth3_ebm"],
            seed=SEED + 2,
        ),
        "oracle_annotations_over_primary_baseline": _paired_bootstrap(
            by_method["oracle_annotation_depth3_ebm"],
            baseline_rows,
            seed=SEED + 3,
        ),
    }
    baseline_per_game = metrics["primary_baseline"]["per_game"]
    adapted_per_game = metrics["adapted_semantics_depth3_ebm"]["per_game"]
    nonnegative = sum(
        adapted_per_game[game]["mean_utility"]
        >= baseline_per_game[game]["mean_utility"]
        for game in games
    )
    comparisons["nonnegative_games"] = nonnegative
    comparisons["games"] = len(games)
    original_by_root = {row["root_key"]: row for row in adapted_rows}
    comparisons["relation_shuffle_action_change_rate"] = float(
        np.mean(
            [
                original_by_root[row["root_key"]]["executed_action_key"]
                != row["executed_action_key"]
                for row in by_method["adapted_relation_shuffle_depth3_ebm"]
            ]
        )
    )
    completion = _completion_capture(roots, decisions)
    adapted_completion = completion["selected_by_method"].get(
        "adapted_semantics_depth3_ebm", 0
    )

    v47_annotation_rows = load_v47_annotations(v47_dir)
    v47_original = {}
    for item in examples:
        v47_original[item.slot.slot_id] = v47_annotation_rows[
            (item.slot.slot_id, "original")
        ]
    identity = {
        "adapted_semantic_outputs": _annotation_identity(examples, adapted),
        "v4_7_zero_shot_outputs": _annotation_identity(examples, v47_original),
    }
    thresholds = manifest["decision_thresholds"]
    checks = {
        "adapter_brier_better_than_action_only": (
            semantic["v43_macro_brier_gain_over_action_only"] > 0.0
        ),
        "adapted_over_structured_mean_gain_strictly_positive": (
            comparisons["adapted_over_structured"]["mean_gain"] > 0.0
        ),
        "adapted_over_primary_baseline_mean_gain_strictly_positive": (
            comparisons["adapted_over_primary_baseline"]["mean_gain"] > 0.0
        ),
        "nonnegative_games_minimum": (
            nonnegative >= int(thresholds["nonnegative_games_minimum"])
        ),
        "original_over_relation_shuffle_mean_gain_strictly_positive": (
            comparisons["original_over_relation_shuffle"]["mean_gain"] > 0.0
        ),
        "completion_events_selected_minimum": (
            adapted_completion
            >= min(
                int(thresholds["completion_events_selected_minimum"]),
                int(completion["opportunities"]),
            )
        ),
        "semantic_output_identity_accuracy_maximum": (
            identity["adapted_semantic_outputs"]["accuracy"]
            <= float(thresholds["semantic_output_identity_accuracy_maximum"])
        ),
    }
    passed = all(checks.values())
    verdict = (
        "EXPLORATORY_ARCHITECTURE_SUPPORT"
        if passed
        else "EXPLORATORY_ARCHITECTURE_NOT_SUPPORTED"
    )
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "semantic_result_checksum": semantic["result_checksum"],
        "verdict": verdict,
        "checks": checks,
        "confirmatory_gate": False,
        "authority_promoted": False,
        "source_validation_opened": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "roots": len(roots),
        "nodes": len(_nodes(examples)),
        "semantic_slots": len(examples),
        "metrics": metrics,
        "comparisons": comparisons,
        "completion_capture": completion,
        "world_model_metrics": {
            name: _world_metrics(examples, predictions)
            for name, predictions in held_predictions.items()
        },
        "game_signature_probe": identity,
        "artifact_sha256": {
            "semantic_annotations": _file_sha256(
                destination / "semantic_annotations.jsonl"
            ),
            "decisions": _file_sha256(destination / "decisions.jsonl"),
            "folds": _file_sha256(destination / "folds.jsonl"),
        },
        "interpretation": {
            "qwen_base_frozen": True,
            "adapter_is_external_not_transformer_lora": True,
            "semantic_predictions_are_logo": True,
            "downstream_v4_7_hyperparameters_changed": False,
            "future_tree_topology_is_non_deployable": True,
            "confidence_interval_required_for_exploratory_verdict": False,
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    embed = subparsers.add_parser("embed")
    embed.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    embed.add_argument("--device", default="cuda:0")
    adapt = subparsers.add_parser("adapt")
    adapt.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    adapt.add_argument("--device", default="cuda:0")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(output_dir=args.output_dir)
    elif args.command == "embed":
        payload = extract_embeddings(
            output_dir=args.output_dir, device=args.device
        )
    elif args.command == "adapt":
        payload = run_semantic_adaptation(
            output_dir=args.output_dir, device=args.device
        )
    elif args.command == "evaluate":
        payload = evaluate_end_to_end(output_dir=args.output_dir)
    else:
        if not (args.output_dir / "frozen_manifest.json").exists():
            freeze_manifest(output_dir=args.output_dir)
        extract_embeddings(output_dir=args.output_dir, device=args.device)
        run_semantic_adaptation(output_dir=args.output_dir, device=args.device)
        payload = evaluate_end_to_end(output_dir=args.output_dir)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrozenQwenPairEncoder",
    "evaluate_end_to_end",
    "extract_embeddings",
    "freeze_manifest",
    "load_manifest",
    "render_pair_prompt",
    "run_semantic_adaptation",
]
