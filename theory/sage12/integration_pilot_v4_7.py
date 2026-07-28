"""SAGE12 V4.7 candidate-complete semantic-slot integration pilot.

The pilot is deliberately source-only and offline.  It uses every legal arm
in the replay-verified V4.3 trees, scores fixed effect bits with the frozen
local Qwen model, fits leave-one-game-out world models and pairwise energies,
and evaluates receding-horizon first-action choices.  Future V4.3 slot
descriptors are an explicit non-deployable topology oracle; future outcomes
are never supplied to learned components except in named oracle diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bound_mechanic_pilot import BindingSignature, BoundEvent
from .compiler import (
    SLOT_EFFECTS,
    SemanticActionSlot,
    SlotAnnotation,
)
from .energy import PairwiseTrajectoryEBM
from .integration_pilot import (
    UTILITY_WEIGHTS,
    ExecutedRoot,
    load_complete_roots,
    step_utility,
)

FORMAT_VERSION = "sage12-candidate-complete-slots-v4.7"
MANIFEST_VERSION = "sage12-candidate-complete-slots-manifest-v4.7"
QWEN_VERSION = "sage12-candidate-complete-slots-qwen-v4.7"
RESULT_VERSION = "sage12-candidate-complete-slots-result-v4.7"

DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "integration_pilot_v4_7"
DEFAULT_V43_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"
DEFAULT_MODEL_PATH = Path("models") / "qwen2_5_0.5b_instruct"

SEED = 4_707
BOOTSTRAP_SAMPLES = 1_000
QWEN_BATCH_SIZE = 32
MAXIMUM_INPUT_TOKENS = 512
EBM_INPUT_WIDTH = 8
EBM_HIDDEN_WIDTH = 32
EBM_EPOCHS = 150
EBM_LEARNING_RATE = 0.003

RELATION_FIELDS = (
    "requested_direction",
    "path_status",
    "actor_relation",
    "actor_relative_direction",
)
FORBIDDEN_MODEL_FIELDS = (
    "game_id",
    "root_key",
    "policy_seed",
    "reset_index",
    "frame_before",
    "frame_after",
    "trace_digest",
    "target_object_id",
    '"row"',
    '"col"',
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _binding_view(binding: BindingSignature) -> dict[str, Any]:
    return {
        "kind": binding.kind,
        "action_family": binding.action_family,
        "requested_direction": binding.requested_direction,
        "occupied": int(binding.occupied),
        "path_status": binding.path_status,
        "actor_relation": binding.actor_relation,
        "actor_relative_direction": binding.actor_relative_direction,
        "target_area_bucket": binding.target_area_bucket,
        "target_aspect_bucket": binding.target_aspect_bucket,
        "target_affordance": binding.target_affordance,
    }


def _event_view(event: BoundEvent) -> dict[str, Any]:
    return {
        "action_name": event.action_name,
        "action_family": event.action_family,
        "binding": _binding_view(event.binding),
        "observed_effects": {
            name: int(bool(value)) for name, value in sorted(event.effects.items())
        },
    }


def _compact_binding(view: Mapping[str, Any]) -> list[Any]:
    return [
        view["kind"],
        view["action_family"],
        view["requested_direction"],
        int(view["occupied"]),
        view["path_status"],
        view["actor_relation"],
        view["actor_relative_direction"],
        view["target_area_bucket"],
        view["target_aspect_bucket"],
        view["target_affordance"],
    ]


def _compact_event(view: Mapping[str, Any]) -> list[Any]:
    effects = view["observed_effects"]
    return [
        view["action_name"],
        view["action_family"],
        _compact_binding(view["binding"]),
        [int(effects[name]) for name in sorted(effects)],
    ]


def _trace_labels(trace: Any) -> dict[str, bool]:
    labels = trace.effects.labels
    return {
        "changed": not bool(trace.effects.noop),
        "moved": bool(labels["actor_displaced"]),
        "target_created": bool(labels["target_created"]),
        "target_removed": bool(labels["target_removed"]),
        "target_moved": bool(labels["target_moved"]),
        "level_complete": bool(
            trace.effects.level_complete
            or trace.levels_completed_after > trace.levels_completed_before
            or str(trace.game_state_after).upper() == "WIN"
        ),
        "game_over": bool(
            trace.effects.game_over
            or str(trace.game_state_after).upper() == "GAME_OVER"
        ),
    }


def _applicability(binding: BindingSignature) -> dict[str, bool]:
    targetable = binding.kind != "targetless"
    occupied = binding.kind == "occupied_object"
    return {
        "changed": True,
        "moved": True,
        "target_created": targetable,
        "target_removed": occupied,
        "target_moved": occupied,
        "level_complete": True,
        "game_over": True,
    }


@dataclass(frozen=True)
class SlotExample:
    example_id: str
    node_id: str
    root_key: str
    game_id: str
    path: str
    side: str
    context: tuple[BoundEvent, ...]
    slot: SemanticActionSlot
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    utility: float

    def annotation(self, *, source: str = "oracle_annotation") -> SlotAnnotation:
        return SlotAnnotation(
            slot_id=self.slot.slot_id,
            effect_probabilities={
                effect: float(bool(self.labels[effect])) for effect in SLOT_EFFECTS
            },
            source=source,
            support=0,
        )


def load_slot_examples(
    roots: Sequence[ExecutedRoot],
) -> tuple[SlotExample, ...]:
    result = []
    for root in roots:
        for path, pair in sorted(root.tree.items()):
            node_id = f"{root.root_key}:{path or 'root'}"
            for side, arm in zip("LR", (pair.left, pair.right)):
                binding = arm.event.binding
                signature = _binding_view(binding)
                slot_id = "slot_" + _checksum(
                    {
                        "node": node_id,
                        "side": side,
                        "action": arm.action.to_dict(),
                    }
                )[:16]
                slot = SemanticActionSlot(
                    slot_id=slot_id,
                    action_name=arm.action.name,
                    action_data=dict(arm.action.action_args),
                    semantic_signature=signature,
                )
                result.append(
                    SlotExample(
                        example_id=f"{node_id}:{side}",
                        node_id=node_id,
                        root_key=root.root_key,
                        game_id=root.game_id,
                        path=path,
                        side=side,
                        context=tuple(pair.context),
                        slot=slot,
                        labels=_trace_labels(arm.trace),
                        applicable=_applicability(binding),
                        utility=step_utility(arm.trace),
                    )
                )
    return tuple(result)


def _nodes(
    examples: Sequence[SlotExample],
) -> tuple[tuple[SlotExample, SlotExample], ...]:
    grouped: dict[str, list[SlotExample]] = defaultdict(list)
    for example in examples:
        grouped[example.node_id].append(example)
    result = []
    for node_id, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda item: item.side)
        if [item.side for item in ordered] != ["L", "R"]:
            raise ValueError(f"node {node_id} does not contain exactly L/R slots")
        result.append((ordered[0], ordered[1]))
    return tuple(result)


def _shuffle_relation_payload(
    context: Sequence[BoundEvent],
    slots: Sequence[SlotExample],
    *,
    salt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context_views = [_event_view(event) for event in context]
    slot_views = [dict(item.slot.semantic_signature) for item in slots]
    bindings = [item["binding"] for item in context_views] + slot_views
    rng = random.Random(f"{SEED}:{salt}")
    for field in RELATION_FIELDS:
        values = [binding[field] for binding in bindings]
        rng.shuffle(values)
        for binding, value in zip(bindings, values):
            binding[field] = value
    return context_views, slot_views


def render_slot_prompt(
    node: tuple[SlotExample, SlotExample],
    *,
    relation_shuffle: bool = False,
) -> str:
    left, _right = node
    if relation_shuffle:
        context, slots = _shuffle_relation_payload(
            left.context,
            node,
            salt=left.node_id,
        )
    else:
        context = [_event_view(event) for event in left.context[-8:]]
        slots = [dict(item.slot.semantic_signature) for item in node]
    payload = {
        "task": "predict effects; output exactly 14 bits",
        "effect_order": list(SLOT_EFFECTS),
        "binding_order": [
            "kind",
            "action_family",
            "requested_direction",
            "occupied",
            "path_status",
            "actor_relation",
            "actor_relative_direction",
            "target_area_bucket",
            "target_aspect_bucket",
            "target_affordance",
        ],
        "recent_8": [_compact_event(item) for item in context[-8:]],
        "slots_0_then_1": [
            [item.slot.action_name, _compact_binding(slots[index])]
            for index, item in enumerate(node)
        ],
        "answer": "",
    }
    prompt = _canonical(payload)
    lowered = prompt.lower()
    for token in FORBIDDEN_MODEL_FIELDS:
        if token in lowered:
            raise ValueError(f"forbidden model-input token: {token}")
    for item in node:
        if item.game_id.lower() in lowered or item.root_key.lower() in lowered:
            raise ValueError("game or root identity leaked into Qwen prompt")
    return prompt


class ConstrainedQwenBitDecoder:
    """Autoregressively restrict every output position to token 0 or token 1."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str,
        batch_size: int = QWEN_BATCH_SIZE,
        maximum_input_tokens: int = MAXIMUM_INPUT_TOKENS,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.maximum_input_tokens = max(64, int(maximum_input_tokens))
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
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

    def score(
        self,
        prompts: Sequence[str],
        *,
        bit_count: int,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        import torch

        tokenizer, model = self._load()
        zero = tokenizer.encode("0", add_special_tokens=False)
        one = tokenizer.encode("1", add_special_tokens=False)
        if len(zero) != 1 or len(one) != 1 or zero[0] == one[0]:
            raise RuntimeError("Qwen tokenizer does not expose atomic 0/1 tokens")
        token_ids = torch.tensor([zero[0], one[0]], device=model.device)
        all_scores: list[list[float]] = []
        input_lengths: list[int] = []
        batch_seconds: list[float] = []
        for offset in range(0, len(prompts), self.batch_size):
            subset = prompts[offset : offset + self.batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a frozen binary semantic-effect "
                                "classifier. Output only the requested bits."
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
                add_special_tokens=False,
                return_tensors="pt",
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            maximum = int(lengths.max().item())
            if maximum > self.maximum_input_tokens:
                raise RuntimeError(
                    f"Qwen prompt exceeds cap: {maximum} > "
                    f"{self.maximum_input_tokens}"
                )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            scores = [[] for _ in subset]
            past = None
            if str(model.device).startswith("cuda"):
                torch.cuda.synchronize(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                for _ in range(bit_count):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    probabilities = _masked_binary_probabilities(
                        output.logits[:, -1, :],
                        token_ids,
                    )
                    one_probabilities = probabilities[:, 1]
                    chosen = token_ids[
                        (one_probabilities >= 0.5).to(torch.long)
                    ].unsqueeze(1)
                    for row, value in zip(scores, one_probabilities.tolist()):
                        row.append(float(value))
                    past = output.past_key_values
                    input_ids = chosen
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ),
                        dim=1,
                    )
            if str(model.device).startswith("cuda"):
                torch.cuda.synchronize(model.device)
            batch_seconds.append(time.perf_counter() - started)
            all_scores.extend(scores)
            input_lengths.extend(int(value) for value in lengths.tolist())
        return all_scores, {
            "rows": len(prompts),
            "batches": len(batch_seconds),
            "batch_size": self.batch_size,
            "bit_count": bit_count,
            "inference_seconds": float(sum(batch_seconds)),
            "median_batch_seconds": float(statistics.median(batch_seconds)),
            "maximum_input_tokens": max(input_lengths, default=0),
            "mean_input_tokens": (
                float(statistics.fmean(input_lengths)) if input_lengths else 0.0
            ),
            "device": str(model.device),
            "zero_token_id": int(zero[0]),
            "one_token_id": int(one[0]),
        }


def _masked_binary_probabilities(logits: Any, token_ids: Any) -> Any:
    """Return a normalized two-token distribution and ignore all other logits."""
    import torch

    if token_ids.numel() != 2:
        raise ValueError("binary decoder requires exactly two token ids")
    binary_logits = logits.index_select(1, token_ids)
    return torch.softmax(binary_logits.float(), dim=1)


def _source_fingerprint(v43_dir: Path) -> dict[str, Any]:
    shards = []
    for path in sorted((v43_dir / "source_train_shards").glob("*.jsonl")):
        shards.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "v43_manifest_sha256": _file_sha256(
            v43_dir / "frozen_manifest.json"
        ),
        "shards": shards,
        "combined_sha256": _checksum(shards),
    }


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    source = Path(v43_dir)
    roots = load_complete_roots(source)
    examples = load_slot_examples(roots)
    nodes = _nodes(examples)
    games = sorted({root.game_id for root in roots})
    root_nodes = [node for node in nodes if node[0].path == ""]
    payload: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "created_for": FORMAT_VERSION,
        "source_only": True,
        "holdout_opened": False,
        "live_environment_opened": False,
        "authority_promotion_allowed": False,
        "source_fingerprint": _source_fingerprint(source),
        "games": games,
        "complete_roots": len(roots),
        "complete_nodes": len(nodes),
        "semantic_slots": len(examples),
        "roots_per_game": {
            game: sum(root.game_id == game for root in roots) for game in games
        },
        "slot_effects": list(SLOT_EFFECTS),
        "qwen": {
            "model_path": DEFAULT_MODEL_PATH.as_posix(),
            "model_sha256": _file_sha256(
                DEFAULT_MODEL_PATH / "model.safetensors"
            ),
            "frozen": True,
            "fine_tuning": False,
            "decoder": "autoregressive token-logit mask to atomic 0/1",
            "temperature": None,
            "sampling": False,
            "bits_per_node": 2 * len(SLOT_EFFECTS),
            "batch_size": QWEN_BATCH_SIZE,
            "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
            "original_node_ids": [node[0].node_id for node in nodes],
            "relation_shuffle_node_ids": [
                node[0].node_id for node in root_nodes
            ],
        },
        "world_model": {
            "outer_split": "leave_one_game_out",
            "inner_split": "leave_one_game_out",
            "effect_heads": "regularized logistic or Beta constant",
            "utility_head": "ridge",
            "variants": [
                "structured",
                "structured_plus_qwen",
                "structured_plus_oracle_annotations",
            ],
            "calibration": "Platt on inner out-of-game predictions only",
        },
        "topology": {
            "primary": "V4.3 future pre-action slot descriptors",
            "future_outcomes_visible": False,
            "non_deployable_oracle": True,
            "deployable_control": "reuse root slots for all three depths",
            "depth": 3,
            "leaves_per_root": 8,
        },
        "ebm": {
            "input_width": EBM_INPUT_WIDTH,
            "hidden_width": EBM_HIDDEN_WIDTH,
            "epochs": EBM_EPOCHS,
            "learning_rate": EBM_LEARNING_RATE,
            "training_pairs": "all unequal within-root leaf pairs",
            "training_features": "nested out-of-game world predictions",
        },
        "utility": dict(UTILITY_WEIGHTS),
        "baselines": [
            "deterministic_left",
            "action_only",
            "action_sequence_only",
            "template_v4_6_reported",
            "qwen_v4_6_reported",
        ],
        "primary_baseline_selection": "training games only per outer fold",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "verdict": {
            "exploratory_support_if": (
                "full Qwen stack mean gain > 0 and nonnegative on >= 6/11 games"
            ),
            "otherwise": (
                "attribute the first positive oracle recovery to Qwen semantics, "
                "world model, energy, or rollout topology; otherwise multiple "
                "bottlenecks"
            ),
            "confidence_intervals": "descriptive; no threshold adjustment",
            "can_promote_authority": False,
        },
    }
    payload["manifest_checksum"] = _checksum(payload)
    _write_json(destination / "frozen_manifest.json", payload)
    return payload


def load_manifest(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    payload = _read_json(Path(output_dir) / "frozen_manifest.json")
    if payload.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported SAGE12 V4.7 manifest")
    expected = str(payload.get("manifest_checksum", ""))
    clean = dict(payload)
    clean.pop("manifest_checksum", None)
    if expected != _checksum(clean):
        raise ValueError("SAGE12 V4.7 manifest checksum mismatch")
    return payload


def generate_qwen(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    roots = load_complete_roots(v43_dir)
    nodes = _nodes(load_slot_examples(roots))
    expected_original = set(manifest["qwen"]["original_node_ids"])
    expected_shuffled = set(manifest["qwen"]["relation_shuffle_node_ids"])
    actual = {node[0].node_id for node in nodes}
    if actual != expected_original:
        raise ValueError("V4.7 node set differs from frozen manifest")
    requests: list[tuple[str, tuple[SlotExample, SlotExample], str, str]] = []
    for node in nodes:
        node_id = node[0].node_id
        prompt = render_slot_prompt(node)
        requests.append((node_id, node, "original", prompt))
        if node_id in expected_shuffled:
            shuffled = render_slot_prompt(node, relation_shuffle=True)
            requests.append((node_id, node, "relation_shuffle", shuffled))
    decoder = ConstrainedQwenBitDecoder(
        model_path=manifest["qwen"]["model_path"],
        device=device,
        batch_size=int(manifest["qwen"]["batch_size"]),
        maximum_input_tokens=int(manifest["qwen"]["maximum_input_tokens"]),
    )
    scores, runtime = decoder.score(
        [request[3] for request in requests],
        bit_count=int(manifest["qwen"]["bits_per_node"]),
    )
    rows = []
    for (node_id, node, variant, prompt), values in zip(requests, scores):
        if len(values) != 2 * len(SLOT_EFFECTS):
            raise RuntimeError("constrained decoder returned wrong bit count")
        annotations = []
        for slot_index, example in enumerate(node):
            start = slot_index * len(SLOT_EFFECTS)
            probabilities = {
                effect: float(values[start + effect_index])
                for effect_index, effect in enumerate(SLOT_EFFECTS)
            }
            annotation = SlotAnnotation(
                slot_id=example.slot.slot_id,
                effect_probabilities=probabilities,
                source="qwen_constrained_bits",
                support=0,
            )
            annotations.append(
                {
                    "slot_id": annotation.slot_id,
                    "support": annotation.support,
                    "effect_probabilities": probabilities,
                    "bits": "".join(
                        "1" if probabilities[effect] >= 0.5 else "0"
                        for effect in SLOT_EFFECTS
                    ),
                }
            )
        rows.append(
            {
                "format_version": QWEN_VERSION,
                "node_id": node_id,
                "variant": variant,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "annotations": annotations,
                "strict_bitstream_valid": True,
                "compiler_slot_coverage": 1.0,
            }
        )
    rows.sort(key=lambda row: (row["node_id"], row["variant"]))
    _write_jsonl(destination / "qwen_outputs.jsonl", rows)
    summary: dict[str, Any] = {
        "format_version": QWEN_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "requests": len(rows),
        "original_nodes": sum(row["variant"] == "original" for row in rows),
        "relation_shuffle_nodes": sum(
            row["variant"] == "relation_shuffle" for row in rows
        ),
        "strict_bitstream_validity": float(
            np.mean([row["strict_bitstream_valid"] for row in rows])
        ),
        "compiler_slot_coverage": float(
            np.mean([row["compiler_slot_coverage"] for row in rows])
        ),
        "runtime": runtime,
        "outputs_sha256": _file_sha256(destination / "qwen_outputs.jsonl"),
    }
    summary["summary_checksum"] = _checksum(summary)
    _write_json(destination / "qwen_summary.json", summary)
    return summary


def load_annotations(
    output_dir: str | Path,
) -> dict[tuple[str, str], SlotAnnotation]:
    result: dict[tuple[str, str], SlotAnnotation] = {}
    with (Path(output_dir) / "qwen_outputs.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            variant = str(row["variant"])
            for item in row["annotations"]:
                annotation = SlotAnnotation(
                    slot_id=str(item["slot_id"]),
                    effect_probabilities={
                        effect: float(item["effect_probabilities"][effect])
                        for effect in SLOT_EFFECTS
                    },
                    source="qwen_constrained_bits",
                    support=int(item["support"]),
                )
                result[(annotation.slot_id, variant)] = annotation
    return result


def _feature_dict(
    example: SlotExample,
    annotation: SlotAnnotation | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query.action_name": example.slot.action_name,
    }
    for key, value in example.slot.semantic_signature.items():
        result[f"query.{key}"] = value
    for position, event in enumerate(example.context[-8:]):
        result[f"context.{position}.action_name"] = event.action_name
        result[f"context.{position}.action_family"] = event.action_family
        for key, value in _binding_view(event.binding).items():
            result[f"context.{position}.{key}"] = value
        for effect, value in event.effects.items():
            result[f"context.{position}.effect.{effect}"] = int(bool(value))
    if annotation is not None:
        for effect in SLOT_EFFECTS:
            result[f"qwen.{effect}"] = float(
                annotation.effect_probabilities[effect]
            )
    rendered = _canonical(result).lower()
    for token in FORBIDDEN_MODEL_FIELDS:
        if token in rendered:
            raise ValueError(f"forbidden world-model feature token: {token}")
    if example.game_id.lower() in rendered or example.root_key.lower() in rendered:
        raise ValueError("game signature leaked into world-model input")
    return result


@dataclass(frozen=True)
class WorldPrediction:
    effects: Mapping[str, float]
    utility: float
    uncertainty: float


@dataclass
class _ProbabilityHead:
    constant: float | None
    model: Any | None
    positives: int
    rows: int


@dataclass
class _Calibrator:
    constant: float | None = None
    model: Any | None = None

    def apply(self, values: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(values), float(self.constant), dtype=np.float64)
        if self.model is None:
            return values.astype(np.float64)
        logits = np.log(
            np.clip(values, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - values, 1e-6, 1.0)
        ).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]

    def to_dict(self) -> dict[str, Any]:
        if self.constant is not None:
            return {"kind": "constant", "probability": self.constant}
        if self.model is None:
            return {"kind": "identity"}
        return {
            "kind": "platt",
            "coefficient": float(self.model.coef_[0, 0]),
            "intercept": float(self.model.intercept_[0]),
        }


class RegularizedSlotWorldModel:
    """Seven effect heads plus a ridge immediate-utility head."""

    def __init__(self, *, use_annotations: bool, seed: int) -> None:
        self.use_annotations = bool(use_annotations)
        self.seed = int(seed)
        self.vectorizer: Any | None = None
        self.effect_heads: dict[str, _ProbabilityHead] = {}
        self.utility_model: Any | None = None
        self.utility_constant = 0.0
        self.utility_residual_std = 1.0
        self.calibrators = {
            effect: _Calibrator() for effect in SLOT_EFFECTS
        }

    def fit(
        self,
        examples: Sequence[SlotExample],
        annotations: Mapping[str, SlotAnnotation] | None = None,
    ) -> RegularizedSlotWorldModel:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge

        annotation_map = annotations or {}
        features = [
            _feature_dict(
                example,
                annotation_map.get(example.slot.slot_id)
                if self.use_annotations
                else None,
            )
            for example in examples
        ]
        self.vectorizer = DictVectorizer(sparse=True)
        matrix = self.vectorizer.fit_transform(features)
        for effect_index, effect in enumerate(SLOT_EFFECTS):
            indices = np.asarray(
                [
                    index
                    for index, example in enumerate(examples)
                    if bool(example.applicable[effect])
                ],
                dtype=np.int64,
            )
            targets = np.asarray(
                [
                    int(bool(examples[index].labels[effect]))
                    for index in indices
                ],
                dtype=np.int64,
            )
            positives = int(targets.sum())
            rows = len(targets)
            if rows == 0 or positives < 2 or rows - positives < 2:
                self.effect_heads[effect] = _ProbabilityHead(
                    constant=(positives + 1.0) / (rows + 2.0),
                    model=None,
                    positives=positives,
                    rows=rows,
                )
                continue
            model = LogisticRegression(
                C=1.0,
                max_iter=500,
                solver="liblinear",
                random_state=self.seed + effect_index,
            )
            model.fit(matrix[indices], targets)
            self.effect_heads[effect] = _ProbabilityHead(
                constant=None,
                model=model,
                positives=positives,
                rows=rows,
            )
        utilities = np.asarray(
            [example.utility for example in examples], dtype=np.float64
        )
        self.utility_constant = float(np.mean(utilities)) if len(utilities) else 0.0
        if len(utilities) >= 2:
            self.utility_model = Ridge(alpha=10.0, solver="lsqr")
            self.utility_model.fit(matrix, utilities)
            residuals = utilities - self.utility_model.predict(matrix)
            self.utility_residual_std = max(
                1e-6, float(np.sqrt(np.mean(residuals**2)))
            )
        return self

    def raw_predict(
        self,
        examples: Sequence[SlotExample],
        annotations: Mapping[str, SlotAnnotation] | None = None,
    ) -> dict[str, WorldPrediction]:
        if self.vectorizer is None:
            raise RuntimeError("world model has not been fitted")
        annotation_map = annotations or {}
        matrix = self.vectorizer.transform(
            [
                _feature_dict(
                    example,
                    annotation_map.get(example.slot.slot_id)
                    if self.use_annotations
                    else None,
                )
                for example in examples
            ]
        )
        raw: dict[str, np.ndarray] = {}
        for effect in SLOT_EFFECTS:
            head = self.effect_heads[effect]
            if head.constant is not None:
                values = np.full(
                    len(examples), float(head.constant), dtype=np.float64
                )
            else:
                values = head.model.predict_proba(matrix)[:, 1]
            raw[effect] = values
        utilities = (
            self.utility_model.predict(matrix)
            if self.utility_model is not None
            else np.full(len(examples), self.utility_constant)
        )
        return {
            example.example_id: WorldPrediction(
                effects={
                    effect: (
                        float(raw[effect][index])
                        if example.applicable[effect]
                        else 0.0
                    )
                    for effect in SLOT_EFFECTS
                },
                utility=float(utilities[index]),
                uncertainty=float(self.utility_residual_std),
            )
            for index, example in enumerate(examples)
        }

    def predict(
        self,
        examples: Sequence[SlotExample],
        annotations: Mapping[str, SlotAnnotation] | None = None,
    ) -> dict[str, WorldPrediction]:
        raw = self.raw_predict(examples, annotations)
        by_effect = {
            effect: self.calibrators[effect].apply(
                np.asarray(
                    [raw[example.example_id].effects[effect] for example in examples]
                )
            )
            for effect in SLOT_EFFECTS
        }
        return {
            example.example_id: WorldPrediction(
                effects={
                    effect: (
                        float(by_effect[effect][index])
                        if example.applicable[effect]
                        else 0.0
                    )
                    for effect in SLOT_EFFECTS
                },
                utility=raw[example.example_id].utility,
                uncertainty=raw[example.example_id].uncertainty,
            )
            for index, example in enumerate(examples)
        }

    def fit_calibrators(
        self,
        examples: Sequence[SlotExample],
        predictions: Mapping[str, WorldPrediction],
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        for index, effect in enumerate(SLOT_EFFECTS):
            applicable = [
                example
                for example in examples
                if bool(example.applicable[effect])
            ]
            targets = np.asarray(
                [int(bool(example.labels[effect])) for example in applicable],
                dtype=np.int64,
            )
            probabilities = np.asarray(
                [predictions[example.example_id].effects[effect] for example in applicable],
                dtype=np.float64,
            )
            positives = int(targets.sum())
            if len(targets) == 0 or positives < 2 or len(targets) - positives < 2:
                self.calibrators[effect] = _Calibrator(
                    constant=(positives + 1.0) / (len(targets) + 2.0)
                )
                continue
            logits = np.log(
                np.clip(probabilities, 1e-6, 1.0 - 1e-6)
                / np.clip(1.0 - probabilities, 1e-6, 1.0)
            ).reshape(-1, 1)
            model = LogisticRegression(
                C=1.0,
                solver="liblinear",
                random_state=self.seed + 100 + index,
            )
            model.fit(logits, targets)
            self.calibrators[effect] = _Calibrator(model=model)

    def to_dict(self) -> dict[str, Any]:
        if self.vectorizer is None:
            raise RuntimeError("world model has not been fitted")
        effects = {}
        for effect, head in self.effect_heads.items():
            if head.constant is not None:
                model = {"kind": "beta_constant", "probability": head.constant}
            else:
                model = {
                    "kind": "logistic",
                    "intercept": float(head.model.intercept_[0]),
                    "coefficients": [
                        float(value) for value in head.model.coef_[0].tolist()
                    ],
                }
            model.update({"rows": head.rows, "positives": head.positives})
            effects[effect] = model
        utility = {
            "kind": "ridge" if self.utility_model is not None else "constant",
            "intercept": (
                float(self.utility_model.intercept_)
                if self.utility_model is not None
                else self.utility_constant
            ),
            "coefficients": (
                [float(value) for value in self.utility_model.coef_.tolist()]
                if self.utility_model is not None
                else []
            ),
            "residual_std": self.utility_residual_std,
        }
        return {
            "use_annotations": self.use_annotations,
            "feature_names": list(self.vectorizer.get_feature_names_out()),
            "effects": effects,
            "utility": utility,
            "calibrators": {
                effect: calibrator.to_dict()
                for effect, calibrator in self.calibrators.items()
            },
        }


def _annotations_for_examples(
    examples: Sequence[SlotExample],
    annotations: Mapping[tuple[str, str], SlotAnnotation],
    *,
    variant: str,
) -> dict[str, SlotAnnotation]:
    result = {}
    for example in examples:
        key = (example.slot.slot_id, variant)
        if key not in annotations:
            raise ValueError(f"missing {variant} annotation for {example.example_id}")
        result[example.slot.slot_id] = annotations[key]
    return result


def _fit_nested_world(
    training: Sequence[SlotExample],
    *,
    annotations: Mapping[str, SlotAnnotation] | None,
    use_annotations: bool,
    seed: int,
) -> tuple[
    RegularizedSlotWorldModel,
    dict[str, WorldPrediction],
]:
    games = sorted({example.game_id for example in training})
    oof: dict[str, WorldPrediction] = {}
    for index, game in enumerate(games):
        inner_train = [example for example in training if example.game_id != game]
        inner_valid = [example for example in training if example.game_id == game]
        inner = RegularizedSlotWorldModel(
            use_annotations=use_annotations,
            seed=seed + index + 1,
        ).fit(inner_train, annotations)
        oof.update(inner.raw_predict(inner_valid, annotations))
    outer = RegularizedSlotWorldModel(
        use_annotations=use_annotations,
        seed=seed,
    ).fit(training, annotations)
    outer.fit_calibrators(training, oof)
    calibrated_oof = {}
    for effect in SLOT_EFFECTS:
        ordered = list(training)
        values = outer.calibrators[effect].apply(
            np.asarray([oof[item.example_id].effects[effect] for item in ordered])
        )
        for index, example in enumerate(ordered):
            prior = calibrated_oof.get(example.example_id)
            effects = dict(prior.effects) if prior is not None else {}
            effects[effect] = (
                float(values[index]) if example.applicable[effect] else 0.0
            )
            calibrated_oof[example.example_id] = WorldPrediction(
                effects=effects,
                utility=oof[example.example_id].utility,
                uncertainty=oof[example.example_id].uncertainty,
            )
    return outer, calibrated_oof


def _true_world_predictions(
    examples: Sequence[SlotExample],
) -> dict[str, WorldPrediction]:
    return {
        example.example_id: WorldPrediction(
            effects={
                effect: float(bool(example.labels[effect]))
                for effect in SLOT_EFFECTS
            },
            utility=example.utility,
            uncertainty=0.0,
        )
        for example in examples
    }


def _entropy(probability: float) -> float:
    probability = min(1.0 - 1e-9, max(1e-9, float(probability)))
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def _trajectory_features(
    root: ExecutedRoot,
    path: str,
    predictions: Mapping[str, WorldPrediction],
    examples_by_position: Mapping[tuple[str, str, str], SlotExample],
    *,
    depth: int,
    root_reuse: bool = False,
) -> tuple[float, ...]:
    steps = []
    prefix = ""
    for marker in path[:depth]:
        lookup_path = "" if root_reuse else prefix
        example = examples_by_position[(root.root_key, lookup_path, marker)]
        steps.append(predictions[example.example_id])
        prefix += marker
    discount = float(UTILITY_WEIGHTS["discount"])
    predicted_return = sum(
        (discount**index) * prediction.utility
        for index, prediction in enumerate(steps)
    )
    success = 1.0 - math.prod(
        1.0 - prediction.effects["level_complete"]
        for prediction in steps
    )
    failure = 1.0 - math.prod(
        1.0 - prediction.effects["game_over"]
        for prediction in steps
    )
    productive = sum(
        prediction.effects["changed"] for prediction in steps
    )
    entropy = float(
        np.mean(
            [
                _entropy(prediction.effects[effect])
                for prediction in steps
                for effect in SLOT_EFFECTS
            ]
        )
    )
    uncertainty = math.sqrt(
        sum(
            ((discount**index) * prediction.uncertainty) ** 2
            for index, prediction in enumerate(steps)
        )
    )
    contradiction = sum(
        prediction.effects["target_created"]
        * prediction.effects["target_removed"]
        + prediction.effects["target_removed"]
        * prediction.effects["target_moved"]
        for prediction in steps
    )
    return (
        float(predicted_return),
        float(success),
        float(failure),
        float(productive),
        float(entropy),
        float(uncertainty),
        float(len(steps)),
        float(contradiction),
    )


def _leaf_value(root: ExecutedRoot, path: str) -> float:
    prefix = ""
    total = 0.0
    discount = float(UTILITY_WEIGHTS["discount"])
    for depth, marker in enumerate(path):
        total += (discount**depth) * step_utility(root.arm(prefix, marker).trace)
        prefix += marker
    return total


def _train_ebm(
    roots: Sequence[ExecutedRoot],
    predictions: Mapping[str, WorldPrediction],
    examples_by_position: Mapping[tuple[str, str, str], SlotExample],
    *,
    depth: int,
    seed: int,
    root_reuse: bool = False,
) -> PairwiseTrajectoryEBM:
    import torch

    torch.set_num_threads(1)
    preferred = []
    rejected = []
    for root in roots:
        paths = (
            ("L", "R")
            if depth == 1
            else tuple("".join(bits) for bits in itertools.product("LR", repeat=3))
        )
        rows = []
        for path in paths:
            features = _trajectory_features(
                root,
                path,
                predictions,
                examples_by_position,
                depth=depth,
                root_reuse=root_reuse,
            )
            value = (
                root.branch_value(path[0])
                if depth == 1
                else _leaf_value(root, path)
            )
            rows.append((path, features, value))
        for left, right in itertools.combinations(rows, 2):
            if math.isclose(left[2], right[2]):
                continue
            better, worse = (left, right) if left[2] > right[2] else (right, left)
            preferred.append(better[1])
            rejected.append(worse[1])
    model = PairwiseTrajectoryEBM(
        input_width=EBM_INPUT_WIDTH,
        hidden_width=EBM_HIDDEN_WIDTH,
        seed=seed,
    ).to("cpu")
    if preferred:
        model.fit_pairs(
            preferred,
            rejected,
            epochs=EBM_EPOCHS,
            learning_rate=EBM_LEARNING_RATE,
        )
    return model


def _select_path(
    root: ExecutedRoot,
    predictions: Mapping[str, WorldPrediction],
    examples_by_position: Mapping[tuple[str, str, str], SlotExample],
    ebm: PairwiseTrajectoryEBM,
    *,
    depth: int,
    root_reuse: bool = False,
) -> str:
    paths = (
        ("L", "R")
        if depth == 1
        else tuple("".join(bits) for bits in itertools.product("LR", repeat=3))
    )
    features = [
        _trajectory_features(
            root,
            path,
            predictions,
            examples_by_position,
            depth=depth,
            root_reuse=root_reuse,
        )
        for path in paths
    ]
    energies = ebm.energies(features)
    return min(zip(paths, energies), key=lambda item: (item[1], item[0]))[0]


def _action_only_choice(
    root: ExecutedRoot,
    training_roots: Sequence[ExecutedRoot],
) -> str:
    by_name: dict[str, list[float]] = defaultdict(list)
    all_values = []
    for item in training_roots:
        for side, candidate in zip("LR", item.candidates):
            value = item.branch_value(side)
            by_name[candidate.action_name].append(value)
            all_values.append(value)
    fallback = float(np.mean(all_values)) if all_values else 0.0
    return max(
        zip("LR", root.candidates),
        key=lambda item: (
            float(np.mean(by_name[item[1].action_name]))
            if by_name[item[1].action_name]
            else fallback,
            item[1].key,
        ),
    )[0]


def _sequence_only_choice(
    root: ExecutedRoot,
    training_roots: Sequence[ExecutedRoot],
) -> str:
    values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    all_values = []
    for item in training_roots:
        for bits in itertools.product("LR", repeat=3):
            path = "".join(bits)
            prefix = ""
            names = []
            for marker in path:
                names.append(item.arm(prefix, marker).action.name)
                prefix += marker
            value = _leaf_value(item, path)
            values[tuple(names)].append(value)
            all_values.append(value)
    fallback = float(np.mean(all_values)) if all_values else 0.0
    choices = []
    for bits in itertools.product("LR", repeat=3):
        path = "".join(bits)
        prefix = ""
        names = []
        for marker in path:
            names.append(root.arm(prefix, marker).action.name)
            prefix += marker
        score = (
            float(np.mean(values[tuple(names)]))
            if values[tuple(names)]
            else fallback
        )
        choices.append((score, path))
    return max(choices, key=lambda item: (item[0], item[1]))[1]


def _decision_row(
    root: ExecutedRoot,
    *,
    method: str,
    selected_path: str,
    baseline_method: str = "",
) -> dict[str, Any]:
    side = selected_path[0]
    left = root.branch_value("L")
    right = root.branch_value("R")
    selected = root.branch_value(side)
    best = max(left, right)
    worst = min(left, right)
    leaf = (
        _leaf_value(root, selected_path)
        if len(selected_path) == 3
        else selected
    )
    all_leaf_values = [
        _leaf_value(root, "".join(bits))
        for bits in itertools.product("LR", repeat=3)
    ]
    oracle_leaf = max(all_leaf_values)
    trace = root.arm("", side).trace
    informative = not math.isclose(left, right)
    return {
        "format_version": FORMAT_VERSION,
        "root_key": root.root_key,
        "game_id": root.game_id,
        "method": method,
        "baseline_method": baseline_method,
        "selected_path": selected_path,
        "selected_side": side,
        "executed_action_key": root.candidates["LR".index(side)].key,
        "coverage": True,
        "selected_value": selected,
        "oracle_value": best,
        "regret": best - selected,
        "informative": informative,
        "oracle_action_correct": (
            not informative or math.isclose(selected, best)
        ),
        "normalized_utility": (
            1.0
            if math.isclose(best, worst)
            else (selected - worst) / (best - worst)
        ),
        "selected_leaf_value": leaf,
        "oracle_leaf_value": oracle_leaf,
        "leaf_regret": oracle_leaf - leaf,
        "oracle_leaf_correct": math.isclose(leaf, oracle_leaf),
        "unsafe_first_action": bool(
            trace.effects.game_over
            or str(trace.game_state_after).upper() == "GAME_OVER"
        ),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    informative = [row for row in rows if row["informative"]]
    per_game = {}
    for game in sorted({str(row["game_id"]) for row in rows}):
        subset = [row for row in rows if row["game_id"] == game]
        info = [row for row in subset if row["informative"]]
        per_game[game] = {
            "rows": len(subset),
            "mean_utility": float(
                np.mean([row["selected_value"] for row in subset])
            ),
            "mean_leaf_utility": float(
                np.mean([row["selected_leaf_value"] for row in subset])
            ),
            "mean_regret": float(np.mean([row["regret"] for row in subset])),
            "oracle_action_accuracy": (
                float(np.mean([row["oracle_action_correct"] for row in info]))
                if info
                else 1.0
            ),
        }
    return {
        "rows": len(rows),
        "informative_rows": len(informative),
        "coverage": float(np.mean([row["coverage"] for row in rows])),
        "mean_utility": float(np.mean([row["selected_value"] for row in rows])),
        "mean_oracle_utility": float(
            np.mean([row["oracle_value"] for row in rows])
        ),
        "mean_regret": float(np.mean([row["regret"] for row in rows])),
        "mean_normalized_utility": float(
            np.mean([row["normalized_utility"] for row in rows])
        ),
        "oracle_action_accuracy": (
            float(np.mean([row["oracle_action_correct"] for row in informative]))
            if informative
            else 1.0
        ),
        "mean_leaf_utility": float(
            np.mean([row["selected_leaf_value"] for row in rows])
        ),
        "mean_leaf_regret": float(
            np.mean([row["leaf_regret"] for row in rows])
        ),
        "oracle_leaf_accuracy": float(
            np.mean([row["oracle_leaf_correct"] for row in rows])
        ),
        "unsafe_first_action_rate": float(
            np.mean([row["unsafe_first_action"] for row in rows])
        ),
        "per_game": per_game,
    }


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    field: str = "selected_value",
    seed: int,
) -> dict[str, float]:
    by_root = {str(row["root_key"]): float(row[field]) for row in baseline}
    deltas = np.asarray(
        [
            float(row[field]) - by_root[str(row["root_key"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
            for _ in range(BOOTSTRAP_SAMPLES)
        ]
    )
    return {
        "mean_gain": float(np.mean(deltas)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def _ece(targets: np.ndarray, probabilities: np.ndarray) -> float:
    result = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if not np.any(mask):
            continue
        result += float(np.mean(mask)) * abs(
            float(np.mean(probabilities[mask]))
            - float(np.mean(targets[mask]))
        )
    return result


def _world_metrics(
    examples: Sequence[SlotExample],
    predictions: Mapping[str, WorldPrediction],
) -> dict[str, Any]:
    effects = {}
    for effect in SLOT_EFFECTS:
        subset = [item for item in examples if item.applicable[effect]]
        targets = np.asarray(
            [int(item.labels[effect]) for item in subset], dtype=np.float64
        )
        probabilities = np.asarray(
            [predictions[item.example_id].effects[effect] for item in subset],
            dtype=np.float64,
        )
        effects[effect] = {
            "rows": len(subset),
            "positives": int(targets.sum()),
            "brier": float(np.mean((probabilities - targets) ** 2)),
            "ece": _ece(targets, probabilities),
            "recall_at_0_5": (
                float(np.mean(probabilities[targets == 1] >= 0.5))
                if np.any(targets == 1)
                else 1.0
            ),
        }
    utility_target = np.asarray(
        [item.utility for item in examples], dtype=np.float64
    )
    utility_prediction = np.asarray(
        [predictions[item.example_id].utility for item in examples],
        dtype=np.float64,
    )
    return {
        "effects": effects,
        "mean_brier": float(
            np.mean([value["brier"] for value in effects.values()])
        ),
        "mean_ece": float(
            np.mean([value["ece"] for value in effects.values()])
        ),
        "utility_rmse": float(
            np.sqrt(np.mean((utility_prediction - utility_target) ** 2))
        ),
    }


def _identity_probe(
    examples: Sequence[SlotExample],
    annotations: Mapping[str, SlotAnnotation],
) -> dict[str, float]:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    labels = np.asarray([item.game_id for item in examples])
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    structured = [_feature_dict(item, None) for item in examples]
    qwen = [
        _feature_dict(item, annotations[item.slot.slot_id]) for item in examples
    ]

    def score(features: Sequence[Mapping[str, Any]]) -> float:
        model = make_pipeline(
            DictVectorizer(sparse=True),
            LogisticRegression(
                C=1.0,
                max_iter=500,
                solver="liblinear",
                random_state=SEED,
            ),
        )
        return float(
            np.mean(cross_val_score(model, features, labels, cv=folds))
        )

    return {
        "majority_accuracy": float(majority),
        "structured_accuracy": score(structured),
        "structured_plus_qwen_accuracy": score(qwen),
    }


def _select_primary_baseline(
    training_roots: Sequence[ExecutedRoot],
) -> str:
    values = {
        "deterministic_left": [],
        "action_only": [],
        "action_sequence_only": [],
    }
    for root in training_roots:
        values["deterministic_left"].append(root.branch_value("L"))
        action_side = _action_only_choice(root, training_roots)
        values["action_only"].append(root.branch_value(action_side))
        path = _sequence_only_choice(root, training_roots)
        values["action_sequence_only"].append(root.branch_value(path[0]))
    return max(
        values,
        key=lambda name: (float(np.mean(values[name])), name),
    )


def _historical_v46() -> dict[str, Any]:
    path = Path("training") / "sage12" / "integration_pilot_v4_6" / "result.json"
    if not path.exists():
        return {"available": False}
    result = _read_json(path)
    names = (
        "template_world_heuristic",
        "template_world_learned_ebm",
        "qwen_repaired_world_learned_ebm",
    )
    return {
        "available": True,
        "result_checksum": result.get("result_checksum", ""),
        "metrics": {
            name: result.get("metrics", {}).get(name, {}) for name in names
        },
    }


def evaluate(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    roots = load_complete_roots(v43_dir)
    examples = load_slot_examples(roots)
    annotations_all = load_annotations(destination)
    originals = _annotations_for_examples(
        examples, annotations_all, variant="original"
    )
    oracle_annotations = {
        example.slot.slot_id: example.annotation() for example in examples
    }
    examples_by_position = {
        (item.root_key, item.path, item.side): item for item in examples
    }
    games = sorted({root.game_id for root in roots})
    decisions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    held_world_predictions: dict[str, dict[str, WorldPrediction]] = {
        "structured": {},
        "qwen": {},
        "oracle_annotation": {},
    }

    for fold_index, held_game in enumerate(games):
        training_roots = tuple(
            root for root in roots if root.game_id != held_game
        )
        validation_roots = tuple(
            root for root in roots if root.game_id == held_game
        )
        training = tuple(
            item for item in examples if item.game_id != held_game
        )
        validation = tuple(
            item for item in examples if item.game_id == held_game
        )
        original_training = {
            item.slot.slot_id: originals[item.slot.slot_id] for item in training
        }
        oracle_training = {
            item.slot.slot_id: oracle_annotations[item.slot.slot_id]
            for item in training
        }
        structured_model, structured_oof = _fit_nested_world(
            training,
            annotations=None,
            use_annotations=False,
            seed=SEED + 100 * fold_index,
        )
        qwen_model, qwen_oof = _fit_nested_world(
            training,
            annotations=original_training,
            use_annotations=True,
            seed=SEED + 100 * fold_index + 10,
        )
        oracle_model, oracle_annotation_oof = _fit_nested_world(
            training,
            annotations=oracle_training,
            use_annotations=True,
            seed=SEED + 100 * fold_index + 20,
        )
        structured_validation = structured_model.predict(validation)
        qwen_validation = qwen_model.predict(validation, originals)
        oracle_annotation_validation = oracle_model.predict(
            validation, oracle_annotations
        )
        held_world_predictions["structured"].update(structured_validation)
        held_world_predictions["qwen"].update(qwen_validation)
        held_world_predictions["oracle_annotation"].update(
            oracle_annotation_validation
        )
        true_training = _true_world_predictions(training)
        true_validation = _true_world_predictions(validation)

        ebms = {
            "structured3": _train_ebm(
                training_roots,
                structured_oof,
                examples_by_position,
                depth=3,
                seed=SEED + 1000 + fold_index,
            ),
            "qwen3": _train_ebm(
                training_roots,
                qwen_oof,
                examples_by_position,
                depth=3,
                seed=SEED + 2000 + fold_index,
            ),
            "qwen1": _train_ebm(
                training_roots,
                qwen_oof,
                examples_by_position,
                depth=1,
                seed=SEED + 3000 + fold_index,
            ),
            "qwen_reuse3": _train_ebm(
                training_roots,
                qwen_oof,
                examples_by_position,
                depth=3,
                seed=SEED + 4000 + fold_index,
                root_reuse=True,
            ),
            "oracle_annotation3": _train_ebm(
                training_roots,
                oracle_annotation_oof,
                examples_by_position,
                depth=3,
                seed=SEED + 5000 + fold_index,
            ),
            "oracle_world3": _train_ebm(
                training_roots,
                true_training,
                examples_by_position,
                depth=3,
                seed=SEED + 6000 + fold_index,
            ),
        }
        primary_baseline = _select_primary_baseline(training_roots)
        model_dir = destination / "models"
        _write_json(
            model_dir / f"{held_game}_structured.json",
            structured_model.to_dict(),
        )
        _write_json(
            model_dir / f"{held_game}_qwen.json",
            qwen_model.to_dict(),
        )
        _write_json(
            model_dir / f"{held_game}_oracle_annotation.json",
            oracle_model.to_dict(),
        )
        fold_rows.append(
            {
                "format_version": FORMAT_VERSION,
                "held_out_game": held_game,
                "training_games": sorted(
                    {item.game_id for item in training}
                ),
                "training_slots": len(training),
                "validation_slots": len(validation),
                "primary_baseline": primary_baseline,
                "ebm_pairs": {
                    name: model.trained_pairs for name, model in ebms.items()
                },
                "models": {
                    name: _file_sha256(model_dir / f"{held_game}_{name}.json")
                    for name in (
                        "structured",
                        "qwen",
                        "oracle_annotation",
                    )
                },
            }
        )

        for root in validation_roots:
            baseline_paths = {
                "deterministic_left": "L",
                "action_only": _action_only_choice(root, training_roots),
                "action_sequence_only": _sequence_only_choice(
                    root, training_roots
                ),
            }
            for name, path in baseline_paths.items():
                decisions.append(
                    _decision_row(root, method=name, selected_path=path)
                )
            decisions.append(
                _decision_row(
                    root,
                    method="primary_baseline",
                    selected_path=baseline_paths[primary_baseline],
                    baseline_method=primary_baseline,
                )
            )
            structured_path = _select_path(
                root,
                structured_validation,
                examples_by_position,
                ebms["structured3"],
                depth=3,
            )
            qwen_path = _select_path(
                root,
                qwen_validation,
                examples_by_position,
                ebms["qwen3"],
                depth=3,
            )
            depth1_path = _select_path(
                root,
                qwen_validation,
                examples_by_position,
                ebms["qwen1"],
                depth=1,
            )
            reuse_path = _select_path(
                root,
                qwen_validation,
                examples_by_position,
                ebms["qwen_reuse3"],
                depth=3,
                root_reuse=True,
            )
            oracle_annotation_path = _select_path(
                root,
                oracle_annotation_validation,
                examples_by_position,
                ebms["oracle_annotation3"],
                depth=3,
            )
            oracle_world_path = _select_path(
                root,
                true_validation,
                examples_by_position,
                ebms["oracle_world3"],
                depth=3,
            )
            oracle_energy_path = max(
                (
                    "".join(bits)
                    for bits in itertools.product("LR", repeat=3)
                ),
                key=lambda path: (_leaf_value(root, path), path),
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
                        method="qwen_depth3_ebm",
                        selected_path=qwen_path,
                    ),
                    _decision_row(
                        root,
                        method="qwen_depth1_ebm",
                        selected_path=depth1_path,
                    ),
                    _decision_row(
                        root,
                        method="qwen_root_reuse_depth3_ebm",
                        selected_path=reuse_path,
                    ),
                    _decision_row(
                        root,
                        method="oracle_annotation_depth3_ebm",
                        selected_path=oracle_annotation_path,
                    ),
                    _decision_row(
                        root,
                        method="oracle_world_learned_ebm",
                        selected_path=oracle_world_path,
                    ),
                    _decision_row(
                        root,
                        method="oracle_energy",
                        selected_path=oracle_energy_path,
                    ),
                )
            )

            root_examples = tuple(
                examples_by_position[(root.root_key, "", side)]
                for side in "LR"
            )
            shuffled_map = dict(originals)
            for item in root_examples:
                shuffled_map[item.slot.slot_id] = annotations_all[
                    (item.slot.slot_id, "relation_shuffle")
                ]
            shuffled_root_predictions = qwen_model.predict(
                root_examples, shuffled_map
            )
            shuffled_predictions = dict(qwen_validation)
            shuffled_predictions.update(shuffled_root_predictions)
            shuffled_path = _select_path(
                root,
                shuffled_predictions,
                examples_by_position,
                ebms["qwen3"],
                depth=3,
            )
            decisions.append(
                _decision_row(
                    root,
                    method="qwen_relation_shuffle_depth3_ebm",
                    selected_path=shuffled_path,
                )
            )

    _write_jsonl(destination / "decisions.jsonl", decisions)
    _write_jsonl(destination / "folds.jsonl", fold_rows)
    methods = sorted({row["method"] for row in decisions})
    summaries = {
        method: _summarize(
            [row for row in decisions if row["method"] == method]
        )
        for method in methods
    }
    by_method = {
        method: [row for row in decisions if row["method"] == method]
        for method in methods
    }
    full = by_method["qwen_depth3_ebm"]
    baseline = by_method["primary_baseline"]
    comparisons = {
        "full_qwen_over_primary_baseline": _paired_bootstrap(
            full, baseline, seed=SEED
        ),
        "qwen_increment_over_structured": _paired_bootstrap(
            full, by_method["structured_depth3_ebm"], seed=SEED + 1
        ),
        "depth3_over_depth1": _paired_bootstrap(
            full, by_method["qwen_depth1_ebm"], seed=SEED + 2
        ),
        "oracle_topology_over_root_reuse": _paired_bootstrap(
            full, by_method["qwen_root_reuse_depth3_ebm"], seed=SEED + 3
        ),
        "original_over_relation_shuffle": _paired_bootstrap(
            full,
            by_method["qwen_relation_shuffle_depth3_ebm"],
            seed=SEED + 4,
        ),
        "oracle_annotations_over_primary_baseline": _paired_bootstrap(
            by_method["oracle_annotation_depth3_ebm"],
            baseline,
            seed=SEED + 5,
        ),
        "oracle_world_over_primary_baseline": _paired_bootstrap(
            by_method["oracle_world_learned_ebm"],
            baseline,
            seed=SEED + 6,
        ),
        "oracle_energy_over_primary_baseline": _paired_bootstrap(
            by_method["oracle_energy"],
            baseline,
            seed=SEED + 7,
        ),
    }
    original_by_root = {row["root_key"]: row for row in full}
    shuffled_rows = by_method["qwen_relation_shuffle_depth3_ebm"]
    comparisons["relation_shuffle_action_change_rate"] = float(
        np.mean(
            [
                original_by_root[row["root_key"]]["executed_action_key"]
                != row["executed_action_key"]
                for row in shuffled_rows
            ]
        )
    )
    baseline_per_game = summaries["primary_baseline"]["per_game"]
    full_per_game = summaries["qwen_depth3_ebm"]["per_game"]
    nonnegative_games = sum(
        full_per_game[game]["mean_utility"]
        >= baseline_per_game[game]["mean_utility"]
        for game in games
    )
    comparisons["full_qwen_nonnegative_games"] = nonnegative_games
    comparisons["full_qwen_games"] = len(games)

    full_gain = comparisons["full_qwen_over_primary_baseline"]["mean_gain"]
    if full_gain > 0.0 and nonnegative_games >= 6:
        verdict = "EXPLORATORY_SUPPORT"
        bottleneck = "none"
    elif (
        comparisons["oracle_annotations_over_primary_baseline"]["mean_gain"]
        > 0.0
    ):
        verdict = "CURRENT_STACK_NEGATIVE_QWEN_SEMANTICS_BOTTLENECK"
        bottleneck = "QWEN_SEMANTICS"
    elif comparisons["oracle_world_over_primary_baseline"]["mean_gain"] > 0.0:
        verdict = "CURRENT_STACK_NEGATIVE_WORLD_MODEL_BOTTLENECK"
        bottleneck = "WORLD_MODEL"
    elif comparisons["oracle_energy_over_primary_baseline"]["mean_gain"] > 0.0:
        verdict = "CURRENT_STACK_NEGATIVE_ENERGY_BOTTLENECK"
        bottleneck = "ENERGY"
    elif (
        comparisons["oracle_topology_over_root_reuse"]["mean_gain"] > 0.0
        and summaries["qwen_root_reuse_depth3_ebm"]["mean_utility"]
        > summaries["primary_baseline"]["mean_utility"]
    ):
        verdict = "CURRENT_STACK_NEGATIVE_ROLLOUT_TOPOLOGY_BOTTLENECK"
        bottleneck = "ROLLOUT_TOPOLOGY"
    else:
        verdict = "CURRENT_STACK_NEGATIVE_MULTIPLE_BOTTLENECKS"
        bottleneck = "MULTIPLE"

    qwen_summary = _read_json(destination / "qwen_summary.json")
    world_metrics = {
        name: _world_metrics(examples, predictions)
        for name, predictions in held_world_predictions.items()
    }
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "verdict": verdict,
        "bottleneck_attribution": bottleneck,
        "confirmatory_gate": False,
        "authority_promoted": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "topology_is_non_deployable_oracle": True,
        "roots": len(roots),
        "nodes": len(_nodes(examples)),
        "semantic_slots": len(examples),
        "qwen": qwen_summary,
        "metrics": summaries,
        "world_model_metrics": world_metrics,
        "comparisons": comparisons,
        "game_signature_probe": _identity_probe(examples, originals),
        "historical_v4_6_baselines": _historical_v46(),
        "artifact_sha256": {
            "qwen_outputs": _file_sha256(destination / "qwen_outputs.jsonl"),
            "qwen_summary": _file_sha256(destination / "qwen_summary.json"),
            "decisions": _file_sha256(destination / "decisions.jsonl"),
            "folds": _file_sha256(destination / "folds.jsonl"),
            "models_combined": _checksum(
                [
                    {
                        "path": path.as_posix(),
                        "sha256": _file_sha256(path),
                    }
                    for path in sorted((destination / "models").glob("*.json"))
                ]
            ),
        },
        "interpretation": {
            "candidate_coverage_is_structural": True,
            "qwen_support_is_always_zero": True,
            "future_outcomes_exposed_to_learned_stack": False,
            "oracle_replacements_are_diagnostic_only": True,
            "thresholds_lowered_after_observation": False,
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        command.add_argument("--v43-dir", type=Path, default=DEFAULT_V43_DIR)
    generate = subparsers.add_parser("generate-qwen")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    generate.add_argument("--v43-dir", type=Path, default=DEFAULT_V43_DIR)
    generate.add_argument("--device", default="cuda:0")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--v43-dir", type=Path, default=DEFAULT_V43_DIR)
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(
            output_dir=args.output_dir, v43_dir=args.v43_dir
        )
    elif args.command == "generate-qwen":
        payload = generate_qwen(
            output_dir=args.output_dir,
            v43_dir=args.v43_dir,
            device=args.device,
        )
    elif args.command == "evaluate":
        payload = evaluate(
            output_dir=args.output_dir, v43_dir=args.v43_dir
        )
    else:
        freeze_manifest(output_dir=args.output_dir, v43_dir=args.v43_dir)
        generate_qwen(
            output_dir=args.output_dir,
            v43_dir=args.v43_dir,
            device=args.device,
        )
        payload = evaluate(
            output_dir=args.output_dir, v43_dir=args.v43_dir
        )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConstrainedQwenBitDecoder",
    "RegularizedSlotWorldModel",
    "SlotExample",
    "WorldPrediction",
    "evaluate",
    "freeze_manifest",
    "generate_qwen",
    "load_annotations",
    "load_manifest",
    "load_slot_examples",
    "render_slot_prompt",
]
