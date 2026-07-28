"""SAGE12 V4.6 global integration pilot with an explicit oracle ladder.

This module is an offline, source-only architecture probe.  It composes the
existing hypothesis compiler, semantic world model, trajectory energy, and
controller over the replay-verified binary action trees collected by V4.3.
It never grants live authority and it never opens a held-out game.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .action_target_data import ActionTargetTrace, build_observation
from .bound_mechanic_pilot import (
    ActionSpec,
    BindingPairRecord,
    BranchArm,
)
from .bound_mechanic_pilot import (
    load_frozen_manifest as load_v43_manifest,
)
from .compiler import CompiledSemanticOption, HypothesisCompiler
from .controller import (
    HierarchicalSubgoal,
    Sage12Config,
    Sage12Mode,
    SemanticActionCandidate,
    SemanticPlanningController,
)
from .energy import HeuristicTrajectoryEnergy, PairwiseTrajectoryEBM
from .hypotheses import (
    ALLOWED_PREDICATES,
    SemanticHypothesis,
    hypotheses_from_json,
)
from .llm import (
    HypothesisGenerationResult,
    LocalHypothesisGenerator,
    TemplateHypothesisGenerator,
    TransformersJSONModel,
    TransformersModelConfig,
)
from .scene_graph import GroundedRelation, SceneGraph, build_scene_graph
from .world_model import SemanticWorldModel

FORMAT_VERSION = "sage12-global-integration-v4.6"
MANIFEST_VERSION = "sage12-global-integration-manifest-v4.6"
QWEN_VERSION = "sage12-global-integration-qwen-v4.6"
RESULT_VERSION = "sage12-global-integration-result-v4.6"

DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "integration_pilot_v4_6"
DEFAULT_V43_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"
DEFAULT_MODEL_PATH = Path("models") / "qwen2_5_0.5b_instruct"

COMPLETE_PATHS = frozenset({"", "L", "R", "LL", "LR", "RL", "RR"})
GENERIC_EFFECTS = (
    "changed",
    "moved",
    "progress",
    "level_complete",
    "game_over",
)
UTILITY_WEIGHTS = {
    "level_complete": 20.0,
    "game_over": -20.0,
    "productive": 2.0,
    "actor_displaced": 1.0,
    "target_effect": 1.0,
    "changed_cells_cap": 1.0,
    "changed_cells_log_divisor": 8.0,
    "discount": 0.9,
}
QWEN_ROOTS_PER_GAME = 4
QWEN_VARIANTS = ("original", "relation_shuffle")
BOOTSTRAP_SAMPLES = 1_000
SEED = 4_606


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


def _action_key(action: ActionSpec | SemanticActionCandidate) -> str:
    if isinstance(action, ActionSpec):
        name = action.name
        data = action.action_args
    else:
        name = action.action_name
        data = action.action_data
    return (
        str(name).strip().upper()
        + ":"
        + json.dumps(
            dict(data),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


@dataclass(frozen=True)
class ExecutedRoot:
    """One complete three-step binary tree from a common initial state."""

    root_key: str
    game_id: str
    tree: Mapping[str, BindingPairRecord]

    @property
    def root_pair(self) -> BindingPairRecord:
        return self.tree[""]

    @property
    def candidates(self) -> tuple[SemanticActionCandidate, ...]:
        pair = self.root_pair
        return tuple(
            SemanticActionCandidate(arm.action.name, arm.action.action_args)
            for arm in (pair.left, pair.right)
        )

    def arm(self, path: str, marker: str) -> BranchArm:
        pair = self.tree[path]
        return pair.left if marker == "L" else pair.right

    def side_for_action_key(self, key: str) -> str | None:
        for side, candidate in zip("LR", self.candidates):
            if candidate.key == key:
                return side
        return None

    def branch_value(self, first: str) -> float:
        values = []
        for suffix in itertools.product("LR", repeat=2):
            path = first + "".join(suffix)
            prefix = ""
            total = 0.0
            for depth, marker in enumerate(path):
                trace = self.arm(prefix, marker).trace
                total += (float(UTILITY_WEIGHTS["discount"]) ** depth) * step_utility(
                    trace
                )
                prefix += marker
            values.append(total)
        return max(values)

    def immediate_value(self, side: str) -> float:
        return step_utility(self.arm("", side).trace)


def step_utility(trace: ActionTargetTrace) -> float:
    """Frozen hierarchical utility for one actually executed transition."""
    effects = trace.effects
    labels = effects.labels
    before = np.asarray(trace.frame_before)
    after = np.asarray(trace.frame_after)
    changed = int(np.count_nonzero(before != after))
    complete = bool(
        effects.level_complete
        or trace.levels_completed_after > trace.levels_completed_before
        or trace.game_state_after.upper() == "WIN"
    )
    game_over = bool(effects.game_over or trace.game_state_after.upper() == "GAME_OVER")
    changed_credit = min(
        float(UTILITY_WEIGHTS["changed_cells_cap"]),
        math.log1p(changed) / float(UTILITY_WEIGHTS["changed_cells_log_divisor"]),
    )
    target_effects = sum(
        int(bool(labels[name]))
        for name in ("target_created", "target_removed", "target_moved")
    )
    return (
        float(UTILITY_WEIGHTS["level_complete"]) * int(complete)
        + float(UTILITY_WEIGHTS["game_over"]) * int(game_over)
        + float(UTILITY_WEIGHTS["productive"]) * int(not effects.noop)
        + float(UTILITY_WEIGHTS["actor_displaced"])
        * int(bool(labels["actor_displaced"]))
        + float(UTILITY_WEIGHTS["target_effect"]) * target_effects
        + changed_credit
    )


def load_complete_roots(
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> tuple[ExecutedRoot, ...]:
    source = Path(v43_dir) / "source_train_shards"
    grouped: dict[str, dict[str, BindingPairRecord]] = defaultdict(dict)
    for path in sorted(source.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                pair = BindingPairRecord.from_dict(json.loads(line))
                grouped[pair.root_key][pair.path] = pair
    roots = []
    for key, tree in sorted(grouped.items()):
        if set(tree) != COMPLETE_PATHS:
            continue
        roots.append(
            ExecutedRoot(
                root_key=key,
                game_id=tree[""].game_id,
                tree=dict(tree),
            )
        )
    return tuple(roots)


def _source_fingerprint(v43_dir: Path) -> dict[str, Any]:
    shard_rows = []
    for path in sorted((v43_dir / "source_train_shards").glob("*.jsonl")):
        shard_rows.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "v43_manifest_sha256": _file_sha256(v43_dir / "frozen_manifest.json"),
        "shards": shard_rows,
        "combined_sha256": _checksum(shard_rows),
    }


def _select_qwen_roots(
    roots: Sequence[ExecutedRoot],
    *,
    per_game: int = QWEN_ROOTS_PER_GAME,
    seed: int = SEED,
) -> tuple[str, ...]:
    by_game: dict[str, list[ExecutedRoot]] = defaultdict(list)
    for root in roots:
        by_game[root.game_id].append(root)
    selected = []
    for game_id in sorted(by_game):
        ranked = sorted(
            by_game[game_id],
            key=lambda root: hashlib.sha256(
                f"{seed}:{root.root_key}".encode()
            ).hexdigest(),
        )
        selected.extend(root.root_key for root in ranked[:per_game])
    return tuple(selected)


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    """Freeze all data, utility, sampling, decoding, and interpretation rules."""
    destination = Path(output_dir)
    source = Path(v43_dir)
    v43_manifest = load_v43_manifest(source / "frozen_manifest.json")
    roots = load_complete_roots(source)
    games = tuple(sorted({root.game_id for root in roots}))
    qwen_roots = _select_qwen_roots(roots)
    payload: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "created_for": FORMAT_VERSION,
        "source_only": True,
        "holdout_opened": False,
        "live_environment_opened": False,
        "authority_promotion_allowed": False,
        "v43_manifest_checksum": v43_manifest["manifest_checksum"],
        "source_fingerprint": _source_fingerprint(source),
        "games": list(games),
        "complete_roots": len(roots),
        "roots_per_game": {
            game: sum(root.game_id == game for root in roots) for game in games
        },
        "qwen_sample": {
            "seed": SEED,
            "roots_per_game": QWEN_ROOTS_PER_GAME,
            "root_keys": list(qwen_roots),
            "variants": list(QWEN_VARIANTS),
        },
        "utility": dict(UTILITY_WEIGHTS),
        "oracle_ladder": [
            "oracle_pipeline",
            "qwen_repaired_oracle_world_oracle_energy",
            "qwen_repaired_learned_world_oracle_energy",
            "qwen_repaired_learned_world_learned_ebm",
        ],
        "ablations": [
            "deterministic_left",
            "action_only",
            "template_oracle",
            "template_world_heuristic",
            "template_world_learned_ebm",
            "qwen_strict",
            "qwen_relation_shuffle",
            "no_hierarchy_depth_1",
            "no_learned_ebm",
        ],
        "cross_game_evaluation": "leave_one_game_out",
        "world_model": {
            "implementation": "SemanticWorldModel",
            "action_key_mode": "name",
            "maximum_depth": 3,
            "beam_width": 16,
        },
        "ebm": {
            "implementation": "PairwiseTrajectoryEBM",
            "hidden_width": 16,
            "epochs": 100,
            "learning_rate": 0.003,
            "device": "cpu",
        },
        "qwen": {
            "model_path": DEFAULT_MODEL_PATH.as_posix(),
            "model_sha256": _file_sha256(DEFAULT_MODEL_PATH / "model.safetensors"),
            "temperature": 0.0,
            "sampling": False,
            "maximum_new_tokens": 256,
            "maximum_hypotheses": 8,
            "maximum_entities": 24,
            "maximum_relations": 96,
            "device_preference": "cuda:0",
            "decoding_changed_from_v1": False,
            "post_decode_adapter": (
                "strict parser plus separately reported deterministic "
                "fence/schema normalization"
            ),
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "interpretation": {
            "confirmatory_gate": False,
            "architecture_refuted_in_scope_if": (
                "oracle_pipeline has no headroom over deterministic/action-only "
                "controls or cannot preserve >=0.95 oracle action accuracy"
            ),
            "exploratory_support_if": (
                "full Qwen learned chain has positive point-estimate utility "
                "gain over the stronger same-root action/template baseline and "
                "non-negative gain on at least 6 games"
            ),
            "otherwise": (
                "localize the earliest oracle-ladder collapse; do not claim "
                "global architectural refutation"
            ),
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
        raise ValueError("unsupported SAGE12 V4.6 manifest")
    expected = str(payload.get("manifest_checksum", ""))
    clean = dict(payload)
    clean.pop("manifest_checksum", None)
    if expected != _checksum(clean):
        raise ValueError("SAGE12 V4.6 manifest checksum mismatch")
    return payload


def _root_observation(root: ExecutedRoot) -> Any:
    trace = root.root_pair.left.trace
    return build_observation(
        trace.frame_before,
        available_actions=trace.available_action_names,
        game_state=trace.game_state_before,
        levels_completed=trace.levels_completed_before,
    )


def _relation_shuffled(graph: SceneGraph, *, salt: str) -> SceneGraph:
    ids = [entity.entity_id for entity in graph.entities]
    if len(ids) < 2:
        return graph
    shuffled = list(ids)
    random.Random(f"{SEED}:{salt}").shuffle(shuffled)
    mapping = dict(zip(ids, shuffled))
    relations = tuple(
        GroundedRelation(
            relation.kind,
            mapping[relation.subject_id],
            mapping[relation.object_id],
        )
        for relation in graph.relations
    )
    state = {
        predicate
        for predicate in graph.state_predicates
        if predicate.startswith(
            ("exists|", "level_complete|", "game_over|")
        )
    }
    state.update(relation.key for relation in relations)
    signature = hashlib.sha256(
        _canonical(
            {
                "base": graph.signature,
                "relations": sorted(item.key for item in relations),
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return SceneGraph(
        entities=graph.entities,
        relations=tuple(sorted(relations, key=lambda item: item.key)),
        state_predicates=frozenset(state),
        signature=signature,
    )


_FENCE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def repair_qwen_hypotheses(
    raw: str,
    *,
    legal_actions: Sequence[str],
    maximum: int = 8,
) -> tuple[SemanticHypothesis, ...]:
    """Deterministically normalize common Qwen format drift.

    This adapter does not invent actions or outcomes.  It accepts only a legal
    action name and an allowed effect predicate already present in the model
    output.  Its results are reported separately from strict JSON validity.
    """
    text = str(raw).strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [
            index for index in (text.find("{"), text.find("[")) if index >= 0
        ]
        if not start_candidates:
            return ()
        start = min(start_candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            return ()
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return ()

    try:
        strict = hypotheses_from_json(text, maximum=maximum)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        strict = ()
    if strict:
        return strict

    legal = {str(action).strip().upper() for action in legal_actions}
    top_items: Sequence[Any]
    if isinstance(payload, Mapping):
        nested = payload.get("hypotheses", ())
        top_items = nested if isinstance(nested, list) else (payload,)
    elif isinstance(payload, list):
        top_items = payload
    else:
        return ()

    normalized = []
    seen: set[tuple[str, str]] = set()
    for outer_index, outer in enumerate(top_items):
        if not isinstance(outer, Mapping):
            continue
        default_action = (
            str(outer.get("action_id", outer.get("action_name", ""))).strip().upper()
        )
        inner_items = outer.get("hypotheses")
        if not isinstance(inner_items, list):
            inner_items = [outer]
        for inner_index, item in enumerate(inner_items):
            if not isinstance(item, Mapping):
                continue
            action = (
                str(item.get("action_id", item.get("action_name", default_action)))
                .strip()
                .upper()
            )
            if action not in legal and default_action in legal:
                action = default_action
            if action not in legal:
                continue
            effect = str(item.get("effect", "")).strip().lower()
            if not effect:
                effects = item.get("effects")
                if isinstance(effects, list) and effects:
                    first = effects[0]
                    if isinstance(first, Mapping):
                        predicate = first.get("predicate", {})
                        if isinstance(predicate, Mapping):
                            effect = str(predicate.get("name", "")).lower()
            if effect not in ALLOWED_PREDICATES:
                continue
            pair = (action, effect)
            if pair in seen:
                continue
            seen.add(pair)
            normalized.append(
                SemanticHypothesis.from_mapping(
                    {
                        "hypothesis_id": (f"repair_{outer_index}_{inner_index}"),
                        "action_name": action,
                        "effects": [
                            {"predicate": {"name": effect}, "operation": "assert"}
                        ],
                        "confidence": 0.5,
                        "source": "qwen_repair",
                        "support": 0,
                    }
                )
            )
            if len(normalized) >= maximum:
                return tuple(normalized)
    return tuple(normalized)


def generate_qwen(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Generate the frozen original and relation-shuffled Qwen streams."""
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    roots = {root.root_key: root for root in load_complete_roots(v43_dir)}
    selected = tuple(manifest["qwen_sample"]["root_keys"])
    missing = sorted(set(selected) - set(roots))
    if missing:
        raise ValueError(f"frozen Qwen roots missing from V4.3: {missing[:3]}")

    model = TransformersJSONModel(
        TransformersModelConfig(
            model_path=str(manifest["qwen"]["model_path"]),
            device=device,
            temperature=float(manifest["qwen"]["temperature"]),
            maximum_input_tokens=8_192,
        )
    )
    generator = LocalHypothesisGenerator(
        model,
        maximum_hypotheses=int(manifest["qwen"]["maximum_hypotheses"]),
        maximum_tokens=int(manifest["qwen"]["maximum_new_tokens"]),
        maximum_entities=int(manifest["qwen"]["maximum_entities"]),
        maximum_relations=int(manifest["qwen"]["maximum_relations"]),
    )
    output_path = destination / "qwen_outputs.jsonl"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                existing[(row["root_key"], row["variant"])] = row

    rows = dict(existing)
    for root_key in selected:
        root = roots[root_key]
        observation = _root_observation(root)
        graph = build_scene_graph(observation)
        actions = tuple(
            sorted({candidate.action_name for candidate in root.candidates})
        )
        for variant in QWEN_VARIANTS:
            key = (root_key, variant)
            if key in rows:
                continue
            prompt_graph = (
                graph
                if variant == "original"
                else _relation_shuffled(graph, salt=root_key)
            )
            started = time.perf_counter()
            result = generator.generate(
                graph=prompt_graph,
                available_actions=actions,
                subgoal="level_complete",
            )
            elapsed = time.perf_counter() - started
            repaired = repair_qwen_hypotheses(
                result.raw_response,
                legal_actions=actions,
                maximum=int(manifest["qwen"]["maximum_hypotheses"]),
            )
            rows[key] = {
                "format_version": QWEN_VERSION,
                "manifest_checksum": manifest["manifest_checksum"],
                "root_key": root_key,
                "game_id": root.game_id,
                "variant": variant,
                "scene_signature": prompt_graph.signature,
                "legal_actions": list(actions),
                "inference_seconds": elapsed,
                "strict_valid": not bool(result.parse_error),
                "strict_count": len(result.hypotheses),
                "strict_hypotheses": [item.to_mapping() for item in result.hypotheses],
                "repair_valid": bool(repaired),
                "repair_count": len(repaired),
                "repaired_hypotheses": [item.to_mapping() for item in repaired],
                "parse_error": result.parse_error,
                "raw_response": result.raw_response,
            }
            _write_jsonl(
                output_path,
                (rows[item] for item in sorted(rows)),
            )

    ordered = [rows[key] for key in sorted(rows)]
    summary = {
        "format_version": QWEN_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "device": device,
        "rows": len(ordered),
        "strict_valid_rate": (
            sum(row["strict_valid"] for row in ordered) / len(ordered)
        ),
        "repair_valid_rate": (
            sum(row["repair_valid"] for row in ordered) / len(ordered)
        ),
        "median_inference_seconds": float(
            np.median([row["inference_seconds"] for row in ordered])
        ),
        "total_inference_seconds": float(
            sum(row["inference_seconds"] for row in ordered)
        ),
        "outputs_sha256": _file_sha256(output_path),
    }
    summary["summary_checksum"] = _checksum(summary)
    _write_json(destination / "qwen_summary.json", summary)
    return summary


def _generation_from_mapping(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> HypothesisGenerationResult:
    return HypothesisGenerationResult(
        hypotheses=tuple(
            SemanticHypothesis.from_mapping(item) for item in payload.get(field, ())
        ),
        parse_error=(
            str(payload.get("parse_error", "")) if field == "strict_hypotheses" else ""
        ),
        raw_response=str(payload.get("raw_response", "")),
    )


def _observed_effects(trace: ActionTargetTrace) -> frozenset[str]:
    names = set()
    if not trace.effects.noop:
        names.add("changed|-|-|")
    if trace.effects.labels["actor_displaced"]:
        names.add("moved|-|-|")
    if (
        trace.effects.level_complete
        or trace.levels_completed_after > trace.levels_completed_before
        or trace.game_state_after.upper() == "WIN"
    ):
        names.update({"progress|-|-|", "level_complete|-|-|"})
    if trace.effects.game_over or trace.game_state_after.upper() == "GAME_OVER":
        names.add("game_over|-|-|")
    return frozenset(names)


def _training_option(
    trace: ActionTargetTrace,
    effect: str,
) -> CompiledSemanticOption:
    return CompiledSemanticOption(
        option_id=f"train_{trace.trace_digest[:10]}_{effect}",
        hypothesis_id=f"observed_{effect}",
        action_name=trace.selected_action_name,
        action_data=dict(trace.selected_action_data),
        bindings={},
        preconditions=(),
        asserted_effects=(f"{effect}|-|-|",),
        retracted_effects=(),
        confidence=0.0,
        source="executed_source_train",
    )


def train_world_model(
    roots: Sequence[ExecutedRoot],
) -> SemanticWorldModel:
    model = SemanticWorldModel(action_key_mode="name")
    for root in roots:
        for pair in root.tree.values():
            for arm in (pair.left, pair.right):
                observed = _observed_effects(arm.trace)
                for effect in GENERIC_EFFECTS:
                    model.observe(
                        _training_option(arm.trace, effect),
                        observed,
                    )
    return model


def _compile(
    generation: HypothesisGenerationResult,
    *,
    graph: SceneGraph,
    candidates: Sequence[SemanticActionCandidate],
) -> tuple[CompiledSemanticOption, ...]:
    result = HypothesisCompiler().compile(
        generation.hypotheses,
        graph=graph,
        legal_candidates=candidates,
    )
    return result.options


def _best_features_by_action(
    *,
    model: SemanticWorldModel,
    graph: SceneGraph,
    options: Sequence[CompiledSemanticOption],
    maximum_depth: int,
    beam_width: int = 16,
) -> dict[str, tuple[float, ...]]:
    trajectories = model.rollout(
        initial_state=graph.state_predicates,
        options=options,
        maximum_depth=maximum_depth,
        beam_width=beam_width,
    )
    energy = HeuristicTrajectoryEnergy()
    best: dict[str, tuple[float, tuple[float, ...]]] = {}
    for trajectory in trajectories:
        breakdown = energy.score(
            trajectory,
            goal_predicate="level_complete",
        )
        key = trajectory.first_option.action_key
        current = best.get(key)
        if current is None or breakdown.total < current[0]:
            best[key] = (breakdown.total, breakdown.features())
    return {key: value[1] for key, value in best.items()}


def train_ebm(
    roots: Sequence[ExecutedRoot],
    *,
    world_model: SemanticWorldModel,
    graph_cache: Mapping[str, SceneGraph],
    seed: int,
) -> PairwiseTrajectoryEBM:
    preferred = []
    rejected = []
    template = TemplateHypothesisGenerator()
    for root in roots:
        graph = graph_cache[root.root_key]
        generation = template.generate(
            graph=graph,
            available_actions=tuple(
                sorted({item.action_name for item in root.candidates})
            ),
            subgoal="level_complete",
        )
        options = _compile(
            generation,
            graph=graph,
            candidates=root.candidates,
        )
        features = _best_features_by_action(
            model=world_model,
            graph=graph,
            options=options,
            maximum_depth=3,
        )
        left_key, right_key = (item.key for item in root.candidates)
        if left_key not in features or right_key not in features:
            continue
        left_value = root.branch_value("L")
        right_value = root.branch_value("R")
        if math.isclose(left_value, right_value):
            continue
        if left_value > right_value:
            preferred.append(features[left_key])
            rejected.append(features[right_key])
        else:
            preferred.append(features[right_key])
            rejected.append(features[left_key])
    model = PairwiseTrajectoryEBM(hidden_width=16, seed=seed).to("cpu")
    if preferred:
        model.fit_pairs(
            preferred,
            rejected,
            epochs=100,
            learning_rate=0.003,
        )
    return model


class _FixedGenerator:
    def __init__(self, result: HypothesisGenerationResult) -> None:
        self.result = result

    def generate(self, **_: Any) -> HypothesisGenerationResult:
        return self.result


def _controller_choice(
    root: ExecutedRoot,
    *,
    observation: Any,
    generation: HypothesisGenerationResult,
    world_model: SemanticWorldModel,
    learned_energy: PairwiseTrajectoryEBM | None,
    maximum_depth: int,
) -> str | None:
    controller = SemanticPlanningController(
        game_id=root.game_id,
        generator=_FixedGenerator(generation),
        world_model=world_model,
        learned_energy=learned_energy,
        config=Sage12Config(
            mode=Sage12Mode.ACTIVE,
            proposal_gate_passed=True,
            world_model_gate_passed=True,
            energy_gate_passed=True,
            active_gate_passed=True,
            maximum_depth=maximum_depth,
            beam_width=16,
            maximum_advisory_risk=1.0,
            maximum_hypotheses=8,
            use_learned_energy=learned_energy is not None,
        ),
    )
    left = root.candidates[0]
    result = controller.arbitrate(
        symbolic_action_name=left.action_name,
        symbolic_action_data=left.action_data,
        symbolic_source="offline_fallback",
        observation=observation,
        candidates=root.candidates,
        protected_competence_available=False,
        danger_veto=lambda _name, _data: False,
        subgoals=(
            HierarchicalSubgoal(
                goal_id="complete_level",
                predicate="level_complete",
                priority=1.0,
                depth=0,
            ),
            HierarchicalSubgoal(
                goal_id="productive_change",
                predicate="changed",
                parent_goal_id="complete_level",
                priority=0.5,
                depth=1,
            ),
        ),
    )
    if not result.selected_option_id:
        return None
    return SemanticActionCandidate(result.action_name, result.action_data).key


def _oracle_choice(
    root: ExecutedRoot,
    *,
    allowed_keys: Iterable[str] | None = None,
    myopic: bool = False,
) -> str | None:
    allowed = set(allowed_keys) if allowed_keys is not None else None
    rows = []
    for side, candidate in zip("LR", root.candidates):
        if allowed is not None and candidate.key not in allowed:
            continue
        value = root.immediate_value(side) if myopic else root.branch_value(side)
        rows.append((value, candidate.key))
    if not rows:
        return None
    return max(rows, key=lambda item: (item[0], item[1]))[1]


def _world_covered_keys(
    *,
    model: SemanticWorldModel,
    graph: SceneGraph,
    options: Sequence[CompiledSemanticOption],
    maximum_depth: int = 3,
) -> frozenset[str]:
    trajectories = model.rollout(
        initial_state=graph.state_predicates,
        options=options,
        maximum_depth=maximum_depth,
        beam_width=16,
    )
    return frozenset(trajectory.first_option.action_key for trajectory in trajectories)


def _action_only_choice(
    root: ExecutedRoot,
    training_roots: Sequence[ExecutedRoot],
) -> str:
    by_name: dict[str, list[float]] = defaultdict(list)
    all_values = []
    for train_root in training_roots:
        for side, candidate in zip("LR", train_root.candidates):
            value = train_root.branch_value(side)
            by_name[candidate.action_name].append(value)
            all_values.append(value)
    fallback = float(np.mean(all_values)) if all_values else 0.0
    return max(
        root.candidates,
        key=lambda candidate: (
            float(np.mean(by_name[candidate.action_name]))
            if by_name[candidate.action_name]
            else fallback,
            candidate.key,
        ),
    ).key


def _decision_row(
    root: ExecutedRoot,
    *,
    method: str,
    selected_key: str | None,
) -> dict[str, Any]:
    advisory_key = selected_key
    fallback = selected_key is None
    if selected_key is None:
        selected_key = root.candidates[0].key
    side = root.side_for_action_key(selected_key)
    if side is None:
        fallback = True
        side = "L"
        selected_key = root.candidates[0].key
    left = root.branch_value("L")
    right = root.branch_value("R")
    best = max(left, right)
    worst = min(left, right)
    selected_value = root.branch_value(side)
    informative = not math.isclose(left, right)
    selected_trace = root.arm("", side).trace
    unsafe = bool(
        selected_trace.effects.game_over
        or selected_trace.game_state_after.upper() == "GAME_OVER"
    )
    return {
        "format_version": FORMAT_VERSION,
        "root_key": root.root_key,
        "game_id": root.game_id,
        "method": method,
        "advisory_action_key": advisory_key,
        "executed_action_key": selected_key,
        "selected_side": side,
        "fallback": fallback,
        "coverage": not fallback,
        "selected_value": selected_value,
        "oracle_value": best,
        "worst_value": worst,
        "regret": best - selected_value,
        "informative": informative,
        "oracle_action_correct": (
            not informative or math.isclose(selected_value, best)
        ),
        "normalized_utility": (
            1.0
            if math.isclose(best, worst)
            else (selected_value - worst) / (best - worst)
        ),
        "immediate_value": root.immediate_value(side),
        "unsafe_first_action": unsafe,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    informative = [row for row in rows if row["informative"]]
    by_game = {}
    for game in sorted({str(row["game_id"]) for row in rows}):
        subset = [row for row in rows if row["game_id"] == game]
        info = [row for row in subset if row["informative"]]
        by_game[game] = {
            "rows": len(subset),
            "coverage": float(np.mean([row["coverage"] for row in subset])),
            "mean_utility": float(np.mean([row["selected_value"] for row in subset])),
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
        "mean_oracle_utility": float(np.mean([row["oracle_value"] for row in rows])),
        "mean_regret": float(np.mean([row["regret"] for row in rows])),
        "mean_normalized_utility": float(
            np.mean([row["normalized_utility"] for row in rows])
        ),
        "oracle_action_accuracy": (
            float(np.mean([row["oracle_action_correct"] for row in informative]))
            if informative
            else 1.0
        ),
        "unsafe_first_action_rate": float(
            np.mean([row["unsafe_first_action"] for row in rows])
        ),
        "per_game": by_game,
    }


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    baseline = {
        str(row["root_key"]): float(row["selected_value"]) for row in baseline_rows
    }
    deltas = np.asarray(
        [
            float(row["selected_value"]) - baseline[str(row["root_key"])]
            for row in rows
            if str(row["root_key"]) in baseline
        ],
        dtype=np.float64,
    )
    if not len(deltas):
        return {"mean_gain": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = float(
            np.mean(rng.choice(deltas, size=len(deltas), replace=True))
        )
    return {
        "mean_gain": float(np.mean(deltas)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def _identity_probe(
    roots: Sequence[ExecutedRoot],
    graph_cache: Mapping[str, SceneGraph],
) -> dict[str, float]:
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return {"available": 0.0}
    features = []
    targets = []
    for root in roots:
        graph = graph_cache[root.root_key]
        row: dict[str, float] = {}
        for candidate in root.candidates:
            row[f"action={candidate.action_name}"] = (
                row.get(f"action={candidate.action_name}", 0.0) + 1.0
            )
        for entity in graph.entities:
            for role in entity.roles:
                key = f"entity={role}:{entity.area_bucket}:{entity.aspect_bucket}"
                row[key] = row.get(key, 0.0) + 1.0
        for relation in graph.relations:
            key = f"relation={relation.kind}"
            row[key] = row.get(key, 0.0) + 1.0
        features.append(row)
        targets.append(root.game_id)
    model = make_pipeline(
        DictVectorizer(),
        LogisticRegression(max_iter=500, random_state=SEED),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accuracy = float(np.mean(cross_val_score(model, features, targets, cv=folds)))
    counts = defaultdict(int)
    for target in targets:
        counts[target] += 1
    majority = max(counts.values()) / len(targets)
    return {
        "available": 1.0,
        "cross_validated_accuracy": accuracy,
        "majority_accuracy": majority,
        "gain": accuracy - majority,
    }


def evaluate(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v43_dir: str | Path = DEFAULT_V43_DIR,
) -> dict[str, Any]:
    """Run the complete LOGO oracle ladder and learned-chain evaluation."""
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    roots = load_complete_roots(v43_dir)
    qwen_keys = set(manifest["qwen_sample"]["root_keys"])
    qwen_rows: dict[tuple[str, str], dict[str, Any]] = {}
    with (destination / "qwen_outputs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qwen_rows[(row["root_key"], row["variant"])] = row
    expected = {(key, variant) for key in qwen_keys for variant in QWEN_VARIANTS}
    if set(qwen_rows) != expected:
        raise ValueError("Qwen stream is incomplete or differs from freeze")

    observations = {root.root_key: _root_observation(root) for root in roots}
    graph_cache = {
        key: build_scene_graph(observation) for key, observation in observations.items()
    }
    decisions = []
    fold_rows = []
    games = sorted({root.game_id for root in roots})
    template = TemplateHypothesisGenerator()

    for fold_index, game in enumerate(games):
        training = tuple(root for root in roots if root.game_id != game)
        validation = tuple(root for root in roots if root.game_id == game)
        world = train_world_model(training)
        ebm = train_ebm(
            training,
            world_model=world,
            graph_cache=graph_cache,
            seed=SEED + fold_index,
        )
        fold_rows.append(
            {
                "held_out_game": game,
                "training_roots": len(training),
                "validation_roots": len(validation),
                "world_model": world.summary(),
                "ebm_training_pairs": ebm.trained_pairs,
                "ebm_device": ebm.device,
            }
        )

        for root in validation:
            graph = graph_cache[root.root_key]
            observation = observations[root.root_key]
            candidates = root.candidates
            action_only = _action_only_choice(root, training)
            decisions.extend(
                (
                    _decision_row(
                        root,
                        method="deterministic_left",
                        selected_key=candidates[0].key,
                    ),
                    _decision_row(
                        root,
                        method="action_only",
                        selected_key=action_only,
                    ),
                    _decision_row(
                        root,
                        method="oracle_direct",
                        selected_key=_oracle_choice(root),
                    ),
                    _decision_row(
                        root,
                        method="oracle_myopic",
                        selected_key=_oracle_choice(root, myopic=True),
                    ),
                )
            )

            oracle_hypotheses = []
            for index, candidate in enumerate(candidates):
                oracle_hypotheses.append(
                    SemanticHypothesis.from_mapping(
                        {
                            "hypothesis_id": f"oracle_{index}",
                            "action_name": candidate.action_name,
                            "action_data": dict(candidate.action_data),
                            "effects": [{"predicate": {"name": "level_complete"}}],
                            "confidence": 1.0,
                            "source": "oracle",
                            "support": 0,
                        }
                    )
                )
            oracle_generation = HypothesisGenerationResult(
                hypotheses=tuple(oracle_hypotheses)
            )
            oracle_options = _compile(
                oracle_generation, graph=graph, candidates=candidates
            )
            decisions.append(
                _decision_row(
                    root,
                    method="oracle_pipeline",
                    selected_key=_oracle_choice(
                        root,
                        allowed_keys={option.action_key for option in oracle_options},
                    ),
                )
            )

            template_generation = template.generate(
                graph=graph,
                available_actions=tuple(
                    sorted({item.action_name for item in candidates})
                ),
                subgoal="level_complete",
            )
            template_options = _compile(
                template_generation, graph=graph, candidates=candidates
            )
            template_covered = {option.action_key for option in template_options}
            template_world_covered = _world_covered_keys(
                model=world,
                graph=graph,
                options=template_options,
            )
            decisions.extend(
                (
                    _decision_row(
                        root,
                        method="template_oracle",
                        selected_key=_oracle_choice(
                            root, allowed_keys=template_covered
                        ),
                    ),
                    _decision_row(
                        root,
                        method="template_learned_world_oracle_energy",
                        selected_key=_oracle_choice(
                            root, allowed_keys=template_world_covered
                        ),
                    ),
                    _decision_row(
                        root,
                        method="template_world_heuristic",
                        selected_key=_controller_choice(
                            root,
                            observation=observation,
                            generation=template_generation,
                            world_model=world,
                            learned_energy=None,
                            maximum_depth=3,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="template_world_learned_ebm",
                        selected_key=_controller_choice(
                            root,
                            observation=observation,
                            generation=template_generation,
                            world_model=world,
                            learned_energy=ebm,
                            maximum_depth=3,
                        ),
                    ),
                    _decision_row(
                        root,
                        method="template_world_no_hierarchy",
                        selected_key=_controller_choice(
                            root,
                            observation=observation,
                            generation=template_generation,
                            world_model=world,
                            learned_energy=ebm,
                            maximum_depth=1,
                        ),
                    ),
                )
            )

            if root.root_key not in qwen_keys:
                continue
            for variant in QWEN_VARIANTS:
                qwen = qwen_rows[(root.root_key, variant)]
                prefix = "qwen" if variant == "original" else "qwen_relation_shuffle"
                strict_generation = _generation_from_mapping(
                    qwen, field="strict_hypotheses"
                )
                repaired_generation = _generation_from_mapping(
                    qwen, field="repaired_hypotheses"
                )
                strict_options = _compile(
                    strict_generation, graph=graph, candidates=candidates
                )
                repaired_options = _compile(
                    repaired_generation, graph=graph, candidates=candidates
                )
                repaired_world_keys = _world_covered_keys(
                    model=world,
                    graph=graph,
                    options=repaired_options,
                )
                decisions.extend(
                    (
                        _decision_row(
                            root,
                            method=f"{prefix}_strict_oracle",
                            selected_key=_oracle_choice(
                                root,
                                allowed_keys={
                                    option.action_key for option in strict_options
                                },
                            ),
                        ),
                        _decision_row(
                            root,
                            method=f"{prefix}_repaired_oracle",
                            selected_key=_oracle_choice(
                                root,
                                allowed_keys={
                                    option.action_key for option in repaired_options
                                },
                            ),
                        ),
                        _decision_row(
                            root,
                            method=(f"{prefix}_repaired_learned_world_oracle_energy"),
                            selected_key=_oracle_choice(
                                root, allowed_keys=repaired_world_keys
                            ),
                        ),
                        _decision_row(
                            root,
                            method=(f"{prefix}_repaired_world_heuristic"),
                            selected_key=_controller_choice(
                                root,
                                observation=observation,
                                generation=repaired_generation,
                                world_model=world,
                                learned_energy=None,
                                maximum_depth=3,
                            ),
                        ),
                        _decision_row(
                            root,
                            method=(f"{prefix}_repaired_world_learned_ebm"),
                            selected_key=_controller_choice(
                                root,
                                observation=observation,
                                generation=repaired_generation,
                                world_model=world,
                                learned_energy=ebm,
                                maximum_depth=3,
                            ),
                        ),
                        _decision_row(
                            root,
                            method=(f"{prefix}_repaired_world_no_hierarchy"),
                            selected_key=_controller_choice(
                                root,
                                observation=observation,
                                generation=repaired_generation,
                                world_model=world,
                                learned_energy=ebm,
                                maximum_depth=1,
                            ),
                        ),
                    )
                )

    _write_jsonl(destination / "decisions.jsonl", decisions)
    _write_jsonl(destination / "folds.jsonl", fold_rows)
    methods = sorted({row["method"] for row in decisions})
    summaries = {
        method: _summarize_rows([row for row in decisions if row["method"] == method])
        for method in methods
    }

    sample_action = [
        row
        for row in decisions
        if row["method"] == "action_only" and row["root_key"] in qwen_keys
    ]
    sample_template = [
        row
        for row in decisions
        if row["method"] == "template_world_heuristic" and row["root_key"] in qwen_keys
    ]
    stronger_sample = max(
        (sample_action, sample_template),
        key=lambda rows_: _summarize_rows(rows_)["mean_utility"],
    )
    full_qwen = [
        row for row in decisions if row["method"] == "qwen_repaired_world_learned_ebm"
    ]
    qwen_world_oracle = [
        row
        for row in decisions
        if row["method"] == "qwen_repaired_learned_world_oracle_energy"
    ]
    direct_oracle = [row for row in decisions if row["method"] == "oracle_direct"]
    full_action = [row for row in decisions if row["method"] == "action_only"]
    full_left = [row for row in decisions if row["method"] == "deterministic_left"]
    stronger_full = max(
        (full_action, full_left),
        key=lambda rows_: _summarize_rows(rows_)["mean_utility"],
    )
    qwen_gain = _paired_bootstrap(
        full_qwen,
        stronger_sample,
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED,
    )
    qwen_oracle_gain = _paired_bootstrap(
        qwen_world_oracle,
        stronger_sample,
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 1,
    )
    headroom = _paired_bootstrap(
        direct_oracle,
        stronger_full,
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 2,
    )

    original_by_root = {row["root_key"]: row for row in full_qwen}
    shuffled_full = [
        row
        for row in decisions
        if row["method"] == "qwen_relation_shuffle_repaired_world_learned_ebm"
    ]
    shuffle_changes = [
        original_by_root[row["root_key"]]["executed_action_key"]
        != row["executed_action_key"]
        for row in shuffled_full
    ]
    shuffle_utility = _paired_bootstrap(
        full_qwen,
        shuffled_full,
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 3,
    )

    baseline_game_values = {
        game: metrics["mean_utility"]
        for game, metrics in _summarize_rows(stronger_sample)["per_game"].items()
    }
    qwen_game_values = summaries["qwen_repaired_world_learned_ebm"]["per_game"]
    nonnegative_games = sum(
        qwen_game_values[game]["mean_utility"] >= baseline_game_values[game]
        for game in qwen_game_values
    )

    oracle_pipeline_accuracy = summaries["oracle_pipeline"]["oracle_action_accuracy"]
    if headroom["mean_gain"] <= 0.0 or oracle_pipeline_accuracy < 0.95:
        verdict = "ARCHITECTURE_REFUTED_IN_SCOPE"
    elif qwen_gain["mean_gain"] > 0.0 and nonnegative_games >= 6:
        verdict = "EXPLORATORY_SUPPORT"
    elif qwen_oracle_gain["mean_gain"] > 0.0:
        verdict = "PROMISING_BUT_LEARNED_COMPONENT_BOTTLENECK"
    else:
        verdict = "GLOBAL_CHAIN_NEGATIVE_PROPOSAL_OR_GROUNDING_BOTTLENECK"

    qwen_summary = _read_json(destination / "qwen_summary.json")
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "verdict": verdict,
        "confirmatory_gate": False,
        "authority_promoted": False,
        "holdout_opened": False,
        "live_environment_opened": False,
        "roots": len(roots),
        "qwen_roots": len(qwen_keys),
        "qwen_outputs": qwen_summary,
        "metrics": summaries,
        "comparisons": {
            "oracle_headroom_over_stronger_simple_baseline": headroom,
            "qwen_world_oracle_over_stronger_same_root_baseline": (qwen_oracle_gain),
            "full_qwen_over_stronger_same_root_baseline": qwen_gain,
            "full_qwen_over_relation_shuffle": shuffle_utility,
            "relation_shuffle_action_change_rate": (
                float(np.mean(shuffle_changes)) if shuffle_changes else 0.0
            ),
            "full_qwen_nonnegative_games": nonnegative_games,
            "full_qwen_games": len(qwen_game_values),
        },
        "game_signature_probe": _identity_probe(roots, graph_cache),
        "artifact_sha256": {
            "decisions": _file_sha256(destination / "decisions.jsonl"),
            "folds": _file_sha256(destination / "folds.jsonl"),
            "qwen_outputs": _file_sha256(destination / "qwen_outputs.jsonl"),
        },
        "interpretation": {
            "oracle_pipeline_accuracy": oracle_pipeline_accuracy,
            "first_ladder_collapse": _first_ladder_collapse(summaries),
            "global_architecture_refuted": (verdict == "ARCHITECTURE_REFUTED_IN_SCOPE"),
            "scope": (
                "offline source-only V4.3 action trees and the current "
                "Qwen/compiler/world-model/EBM implementation"
            ),
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def _first_ladder_collapse(
    summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    ladder = (
        "oracle_pipeline",
        "qwen_repaired_oracle",
        "qwen_repaired_learned_world_oracle_energy",
        "qwen_repaired_world_learned_ebm",
    )
    previous = None
    for method in ladder:
        metrics = summaries[method]
        if (
            previous is not None
            and metrics["mean_normalized_utility"] + 1e-9
            < previous["mean_normalized_utility"]
        ):
            return method
        previous = metrics
    return "none"


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
        payload = freeze_manifest(output_dir=args.output_dir, v43_dir=args.v43_dir)
    elif args.command == "generate-qwen":
        payload = generate_qwen(
            output_dir=args.output_dir,
            v43_dir=args.v43_dir,
            device=args.device,
        )
    elif args.command == "evaluate":
        payload = evaluate(output_dir=args.output_dir, v43_dir=args.v43_dir)
    else:
        freeze_manifest(output_dir=args.output_dir, v43_dir=args.v43_dir)
        generate_qwen(
            output_dir=args.output_dir,
            v43_dir=args.v43_dir,
            device=args.device,
        )
        payload = evaluate(output_dir=args.output_dir, v43_dir=args.v43_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExecutedRoot",
    "evaluate",
    "freeze_manifest",
    "generate_qwen",
    "load_complete_roots",
    "load_manifest",
    "repair_qwen_hypotheses",
    "step_utility",
    "train_world_model",
]
