"""SAGE12 V4.16 morpho-topological transformation pipeline.

Commands::

    python -m theory.sage12.morpho_topological_v4_16 freeze
    python -m theory.sage12.morpho_topological_v4_16 compile
    python -m theory.sage12.morpho_topological_v4_16 train
    python -m theory.sage12.morpho_topological_v4_16 cluster
    python -m theory.sage12.morpho_topological_v4_16 evaluate
    python -m theory.sage12.morpho_topological_v4_16 shadow
    python -m theory.sage12.morpho_topological_v4_16 run-all

The future state is teacher-only.  Runtime retrieval always starts from the
causal ``before + candidate action`` query encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .action_target_data import (
    build_action_target_trace,
    build_observation,
    grid_sha256,
)
from .mt.advisor import (
    MorphoTopologicalAnalogyAdvisor,
    SageMTConfig,
    SageMTMode,
    SageMTShadowWriter,
)
from .mt.clustering import (
    ClusterRegistry,
    TransformationPrototypeMemory,
    fit_cluster_registry,
)
from .mt.model import (
    DELTA_BUCKETS,
    MTModelConfig,
    TransformationEmbedding,
    checkpoint_payload,
    encode_transitions,
    fit_mt_model,
    load_mt_model,
    predict_graph_details,
)
from .mt.transition import MTTransitionRecord, compile_mt_transition
from .semantic_teacher_v4_9 import compile_semantics

FORMAT_VERSION = "sage12-morpho-topological-v4.16"
MANIFEST_VERSION = "sage12-morpho-topological-manifest-v4.16"
RESULT_VERSION = "sage12-morpho-topological-result-v4.16"
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage12" / "morpho_topological_v4_16"
)
DEFAULT_HUMAN_TRACES_DIR = Path("human_traces")
DEFAULT_V411_DIR = Path("training") / "sage12" / "counterfactual_semantics_v4_11"

HUMAN_TRAIN_GAMES = ("ar25", "bp35", "cd82", "cn04", "dc22", "ft09")
TRANSFER_GAMES = ("g50t", "ka59", "lf52", "lp85", "sp80", "su15", "tr87", "tu93")
ACTIVE_VALIDATION_GAMES = ("re86", "ls20", "sc25")
FINAL_HOLDOUT_GAMES = ("s5i5", "vc33", "m0r0", "sk48", "r11l")
FORBIDDEN_STUDENT_FIELDS = (
    "game_id",
    "source_game_id",
    "frame_before",
    "frame_after",
    "row",
    "col",
    "x",
    "y",
    "object_id",
    "color",
    "colour",
    "value",
    "grid_hash",
    "scene_signature",
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
    )


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


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
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _trace_paths(directory: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(directory.glob("*.steps.jsonl"))
        if path.name.split("-", 1)[0] in HUMAN_TRAIN_GAMES
    )


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    human_traces_dir: str | Path = DEFAULT_HUMAN_TRACES_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    traces = _trace_paths(Path(human_traces_dir))
    panel_paths = tuple(
        Path(v411_dir) / "source_train_shards" / f"{game}.jsonl"
        for game in TRANSFER_GAMES
    )
    sources = (
        traces
        + panel_paths
        + (Path(v411_dir) / "frozen_manifest.json",)
    )
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(path.as_posix() for path in missing))
    model = MTModelConfig()
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "split": {
            "human_train": list(HUMAN_TRAIN_GAMES),
            "transfer_evaluation": list(TRANSFER_GAMES),
            "active_validation_closed": list(ACTIVE_VALIDATION_GAMES),
            "final_holdout_closed": list(FINAL_HOLDOUT_GAMES),
            "v4_15_untouched": True,
        },
        "source_fingerprints": {
            path.as_posix(): _fingerprint(path)
            for path in sources
        },
        "representation": {
            "student_input": "identity_free_morpho_topological_graph",
            "teacher_input": "observed_before_action_after",
            "future_state_deployable": False,
            "maximum_components": 64,
            "forbidden_student_fields": list(FORBIDDEN_STUDENT_FIELDS),
        },
        "model": asdict(model),
        "clustering": {
            "algorithm": "hdbscan",
            "parameter_grid": [[16, 5], [16, 10], [32, 5], [32, 10], [64, 5], [64, 10]],
            "bootstrap_samples": 20,
            "bootstrap_fraction": 0.80,
            "minimum_support": 20,
            "minimum_games": 3,
            "maximum_retrieval_matches": 8,
        },
        "gates": {
            "augmentation_cosine_median_minimum": 0.95,
            "augmentation_cluster_consistency_minimum": 0.99,
            "cluster_stability_ari_minimum": 0.70,
            "eligible_cluster_coverage_minimum": 0.60,
            "identity_gain_over_majority_maximum": 0.10,
            "nonnegative_transfer_games_minimum": 5,
            "completion_capture_absolute_minimum": 1,
            "completion_capture_fraction_minimum": 0.50,
        },
        "authority": {
            "shadow_only": True,
            "bounded_allowed": False,
            "active_allowed": False,
            "holdout_opened": False,
        },
        "result_observed_at_freeze": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    payload = _read_json(Path(output_dir) / "frozen_manifest.json")
    if payload.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported SAGE-MT V4.16 manifest")
    expected = str(payload["manifest_checksum"])
    clean = dict(payload)
    clean.pop("manifest_checksum")
    if _checksum(clean) != expected:
        raise ValueError("SAGE-MT V4.16 manifest checksum mismatch")
    return payload


def _action_data(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if item is not None
    }


def _observed_semantic_outcome(
    *,
    game: str,
    before: Any,
    after: Any,
    action: str,
    action_data: Mapping[str, Any],
    available_actions: Sequence[str],
    game_state_before: str,
    game_state_after: str,
    levels_before: int,
    levels_after: int,
    step: int,
) -> tuple[bool, bool, tuple[int, int] | None, tuple[int, int] | None]:
    """Compile observed utility and actor anchors from the existing teacher."""

    legal = tuple(
        sorted(
            {str(item).strip().upper() for item in available_actions}
            | {action}
        )
    )
    trace = build_action_target_trace(
        game_id=game,
        source_split="source_train",
        policy_seed=0,
        reset_index=0,
        step_index=step,
        collection_phase="v4_16_existing_corpus",
        available_action_names=legal,
        selected_action_name=action,
        selected_action_data=action_data,
        frame_before=before,
        frame_after=after,
        game_state_before=game_state_before,
        game_state_after=game_state_after,
        levels_completed_before=levels_before,
        levels_completed_after=levels_after,
    )
    labels, _, _, _ = compile_semantics(trace)
    observation_before = build_observation(
        before,
        available_actions=legal,
        game_state=game_state_before,
        levels_completed=levels_before,
        infer_players=True,
    )
    observation_after = build_observation(
        after,
        available_actions=legal,
        game_state=game_state_after,
        levels_completed=levels_after,
        infer_players=True,
        prev_player_hypotheses=observation_before.player_candidates,
    )
    player_before = observation_before.best_player
    player_after = observation_after.best_player
    return (
        bool(labels["productive"]),
        bool(labels["risk"]),
        tuple(player_before.position) if player_before is not None else None,
        tuple(player_after.position) if player_after is not None else None,
    )


def _compile_human_records(
    directory: Path,
    *,
    maximum_records: int | None = None,
) -> list[MTTransitionRecord]:
    output: list[MTTransitionRecord] = []
    seen: set[str] = set()
    previous_levels: dict[tuple[str, str], int] = defaultdict(int)
    for path in _trace_paths(directory):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                action = str(row.get("action", "")).strip().upper()
                if action in {"", "RESET"}:
                    continue
                game = str(row["game_id"]).split("-", 1)[0]
                if game not in HUMAN_TRAIN_GAMES:
                    continue
                before = row["frame_before"]
                after = row["frame_after"]
                if np.asarray(before).shape != np.asarray(after).shape:
                    continue
                episode = str(row.get("episode_id", ""))
                previous = previous_levels[(game, episode)]
                current = int(row.get("levels_completed_after", previous))
                level_complete = current > previous or str(
                    row.get("game_state_after", "")
                ).upper() == "WIN"
                previous_levels[(game, episode)] = current
                data = _action_data(row.get("action_args"))
                (
                    productive,
                    risk,
                    player_before,
                    player_after,
                ) = _observed_semantic_outcome(
                    game=game,
                    before=before,
                    after=after,
                    action=action,
                    action_data=data,
                    available_actions=tuple(row.get("available_actions", ())),
                    game_state_before="NOT_FINISHED",
                    game_state_after=str(
                        row.get("game_state_after", "NOT_FINISHED")
                    ),
                    levels_before=previous,
                    levels_after=current,
                    step=int(row.get("step", 0)),
                )
                exact = _checksum(
                    {
                        "game": game,
                        "before": grid_sha256(before),
                        "action": action,
                        "data": data,
                        "after": grid_sha256(after),
                    }
                )
                if exact in seen:
                    continue
                seen.add(exact)
                output.append(
                    compile_mt_transition(
                        before,
                        action,
                        after,
                        action_data=data,
                        source_game_id=game,
                        player_position_before=player_before,
                        player_position_after=player_after,
                        productive=productive,
                        risk=risk,
                        audit={
                            "source": path.as_posix(),
                            "episode_id": episode,
                            "step": int(row.get("step", 0)),
                            "frame_before_sha256": grid_sha256(before),
                            "frame_after_sha256": grid_sha256(after),
                            "level_complete": level_complete,
                        },
                    )
                )
                if maximum_records and len(output) >= maximum_records:
                    return output
    return output


def _compile_transfer_records(
    directory: Path,
    *,
    maximum_records: int | None = None,
) -> list[MTTransitionRecord]:
    output: list[MTTransitionRecord] = []
    seen: set[str] = set()
    for game in TRANSFER_GAMES:
        path = directory / "source_train_shards" / f"{game}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                panel = json.loads(line)
                for arm in panel.get("arms", ()):
                    trace = dict(arm["immediate_trace"])
                    before = trace["frame_before"]
                    after = trace["frame_after"]
                    action = str(trace["selected_action_name"]).strip().upper()
                    data = _action_data(trace.get("selected_action_data"))
                    exact = _checksum(
                        {
                            "game": game,
                            "before": trace.get(
                                "frame_before_sha256",
                                grid_sha256(before),
                            ),
                            "action": action,
                            "data": data,
                            "after": trace.get(
                                "frame_after_sha256",
                                grid_sha256(after),
                            ),
                        }
                    )
                    if exact in seen:
                        continue
                    seen.add(exact)
                    effects = dict(trace.get("effects", {}))
                    level_complete = bool(
                        effects.get("level_complete", False)
                        or int(trace.get("levels_completed_after", 0))
                        > int(trace.get("levels_completed_before", 0))
                        or str(trace.get("game_state_after", "")).upper()
                        == "WIN"
                    )
                    (
                        productive,
                        risk,
                        player_before,
                        player_after,
                    ) = _observed_semantic_outcome(
                        game=game,
                        before=before,
                        after=after,
                        action=action,
                        action_data=data,
                        available_actions=tuple(
                            trace.get("available_action_names", ())
                        ),
                        game_state_before=str(
                            trace.get("game_state_before", "NOT_FINISHED")
                        ),
                        game_state_after=str(
                            trace.get("game_state_after", "NOT_FINISHED")
                        ),
                        levels_before=int(
                            trace.get("levels_completed_before", 0)
                        ),
                        levels_after=int(
                            trace.get("levels_completed_after", 0)
                        ),
                        step=int(trace.get("step_index", 0)),
                    )
                    output.append(
                        compile_mt_transition(
                            before,
                            action,
                            after,
                            action_data=data,
                            source_game_id=game,
                            player_position_before=player_before,
                            player_position_after=player_after,
                            productive=productive,
                            risk=risk,
                            audit={
                                "source": path.as_posix(),
                                "panel_id": str(panel["panel_id"]),
                                "arm_index": int(arm["arm_index"]),
                                "frame_before_sha256": str(
                                    trace.get(
                                        "frame_before_sha256",
                                        grid_sha256(before),
                                    )
                                ),
                                "frame_after_sha256": str(
                                    trace.get(
                                        "frame_after_sha256",
                                        grid_sha256(after),
                                    )
                                ),
                                "level_complete": level_complete,
                            },
                        )
                    )
                    if maximum_records and len(output) >= maximum_records:
                        return output
    return output


def compile_corpus(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    human_traces_dir: str | Path = DEFAULT_HUMAN_TRACES_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    maximum_train_records: int | None = None,
    maximum_transfer_records: int | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(output_dir)
    destination = Path(output_dir)
    train = _compile_human_records(
        Path(human_traces_dir),
        maximum_records=maximum_train_records,
    )
    transfer = _compile_transfer_records(
        Path(v411_dir),
        maximum_records=maximum_transfer_records,
    )
    if not train or not transfer:
        raise RuntimeError("SAGE-MT compilation produced an empty split")
    _write_jsonl(
        destination / "train_transitions.jsonl",
        (record.to_dict() for record in train),
    )
    _write_jsonl(
        destination / "transfer_transitions.jsonl",
        (record.to_dict() for record in transfer),
    )
    views = [record.student_view() for record in train + transfer]
    encoded_views = [_canonical(view).lower() for view in views]
    forbidden_hits = sorted(
        {
            field
            for field in FORBIDDEN_STUDENT_FIELDS
            if any(f'"{field}"' in encoded for encoded in encoded_views)
        }
    )
    qa = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "train_records": len(train),
        "transfer_records": len(transfer),
        "train_games": sorted({record.source_game_id for record in train}),
        "transfer_games": sorted({record.source_game_id for record in transfer}),
        "delta_signatures_train": len({record.delta_signature for record in train}),
        "delta_signatures_transfer": len(
            {record.delta_signature for record in transfer}
        ),
        "forbidden_student_field_hits": forbidden_hits,
        "student_view_safe": not forbidden_hits,
        "active_validation_opened": False,
        "holdout_opened": False,
    }
    _write_json(destination / "teacher_qa.json", qa)
    if forbidden_hits:
        raise RuntimeError("SAGE-MT student view failed leakage QA")
    return qa


def load_records(path: str | Path) -> tuple[MTTransitionRecord, ...]:
    return tuple(
        MTTransitionRecord.from_dict(row)
        for row in _read_jsonl(Path(path))
    )


def train_model(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    qa = _read_json(destination / "teacher_qa.json")
    if not qa.get("student_view_safe"):
        raise RuntimeError("SAGE-MT training blocked by teacher QA")
    records = load_records(destination / "train_transitions.jsonl")
    config = MTModelConfig(**dict(manifest["model"]))
    model, metadata = fit_mt_model(records, config=config, device=device)
    torch.save(
        checkpoint_payload(model, metadata),
        destination / "mt_model.pt",
    )
    embeddings = encode_transitions(
        model,
        records,
        config=config,
        device=device,
    )
    _write_jsonl(
        destination / "train_embeddings.jsonl",
        (asdict(item) for item in embeddings),
    )
    result = {
        **dict(metadata),
        "checkpoint": "mt_model.pt",
        "embedding_rows": len(embeddings),
        "device": device,
    }
    _write_json(destination / "training_result.json", result)
    return result


def _load_embeddings(path: Path) -> tuple[TransformationEmbedding, ...]:
    return tuple(
        TransformationEmbedding(
            transition_id=str(row["transition_id"]),
            vector=tuple(float(item) for item in row["vector"]),
            predicted_vector=tuple(
                float(item) for item in row["predicted_vector"]
            ),
            uncertainty=float(row["uncertainty"]),
            delta_signature=str(row["delta_signature"]),
            source_game_id=str(row["source_game_id"]),
        )
        for row in _read_jsonl(path)
    )


def cluster_embeddings(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    records = load_records(destination / "train_transitions.jsonl")
    embeddings = _load_embeddings(destination / "train_embeddings.jsonl")
    parameters = manifest["clustering"]
    registry = fit_cluster_registry(
        embeddings,
        records,
        minimum_support=int(parameters["minimum_support"]),
        minimum_games=int(parameters["minimum_games"]),
        bootstrap_samples=int(parameters["bootstrap_samples"]),
        parameter_grid=tuple(
            (int(row[0]), int(row[1]))
            for row in parameters["parameter_grid"]
        ),
        seed=int(manifest["model"]["seed"]),
    )
    payload = registry.to_dict()
    payload["manifest_checksum"] = manifest["manifest_checksum"]
    _write_json(destination / "cluster_registry.json", payload)
    return payload


def _event_vector(record: MTTransitionRecord) -> np.ndarray:
    from .mt.model import _event_targets

    return _event_targets(record, DELTA_BUCKETS)


def _recall_at_prototypes(
    records: Sequence[MTTransitionRecord],
    predictions: Sequence[Any],
    memory: TransformationPrototypeMemory,
) -> tuple[float, list[bool]]:
    hits = []
    by_id = {
        prototype.prototype_id: prototype
        for prototype in memory.registry.prototypes
    }
    for record, prediction in zip(records, predictions):
        matches = memory.retrieve(
            prediction.vector,
            action_family=record.graph_before.action_family,
            uncertainty=prediction.uncertainty,
            maximum_matches=8,
        )
        hits.append(
            any(
                record.delta_signature
                in by_id[match.prototype_id].dominant_delta_signatures
                for match in matches
            )
        )
    return float(np.mean(hits)) if hits else 0.0, hits


def _baseline_signature_tables(
    records: Sequence[MTTransitionRecord],
) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, tuple[str, ...]], tuple[str, ...]]:
    action: dict[str, Counter[str]] = defaultdict(Counter)
    state: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for record in records:
        action[record.action_name][record.delta_signature] += 1
        state[record.graph_before.signature][record.delta_signature] += 1
        global_counts[record.delta_signature] += 1
    return (
        {
            key: tuple(item for item, _ in values.most_common(8))
            for key, values in action.items()
        },
        {
            key: tuple(item for item, _ in values.most_common(8))
            for key, values in state.items()
        },
        tuple(item for item, _ in global_counts.most_common(8)),
    )


def _identity_probe(
    matrix: np.ndarray,
    labels: Sequence[str],
) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    counts = Counter(labels)
    majority = max(counts.values()) / max(1, len(labels))
    minimum = min(counts.values())
    if len(counts) < 2 or minimum < 2:
        return majority, majority
    folds = StratifiedKFold(
        n_splits=min(3, minimum),
        shuffle=True,
        random_state=5_160,
    )
    score = float(
        np.mean(
            cross_val_score(
                LogisticRegression(max_iter=1_000),
                matrix,
                np.asarray(labels),
                cv=folds,
            )
        )
    )
    return score, float(majority)


def _paired_bootstrap_lower(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 1_000,
    seed: int = 5_161,
) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = [
        float(np.mean(delta[rng.integers(0, len(delta), size=len(delta))]))
        for _ in range(max(1, int(samples)))
    ]
    return float(np.quantile(values, 0.025))


def _analogy_score(
    prediction: Any,
    record: MTTransitionRecord,
    memory: TransformationPrototypeMemory,
) -> float:
    matches = memory.retrieve(
        prediction.vector,
        action_family=record.graph_before.action_family,
        uncertainty=prediction.uncertainty,
        maximum_matches=8,
    )
    if not matches:
        return -prediction.uncertainty
    weights = np.exp(
        np.asarray([item.similarity for item in matches]) / 0.10
    )
    weights /= weights.sum()
    return float(
        sum(
            weight
            * (match.productive_probability - match.risk_probability)
            for weight, match in zip(weights, matches)
        )
        - 0.10 * prediction.uncertainty
    )


def _utility(record: MTTransitionRecord) -> float:
    return (
        float(record.productive is True)
        - float(record.risk is True)
        + 4.0 * float(bool(record.audit.get("level_complete", False)))
    )


def evaluate(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch

    destination = Path(output_dir)
    manifest = load_manifest(destination)
    train = load_records(destination / "train_transitions.jsonl")
    transfer = load_records(destination / "transfer_transitions.jsonl")
    checkpoint = torch.load(
        destination / "mt_model.pt",
        map_location=device,
        weights_only=False,
    )
    model, config, _ = load_mt_model(checkpoint, device=device)
    registry = ClusterRegistry.from_dict(
        _read_json(destination / "cluster_registry.json")
    )
    memory = TransformationPrototypeMemory(registry)
    predictions = predict_graph_details(
        model,
        [record.graph_before for record in transfer],
        config=config,
        device=device,
    )
    shuffled = predict_graph_details(
        model,
        [record.graph_before.without_relations() for record in transfer],
        config=config,
        device=device,
    )
    recall, recall_hits = _recall_at_prototypes(transfer, predictions, memory)
    shuffled_recall, _ = _recall_at_prototypes(transfer, shuffled, memory)
    action_table, state_table, global_table = _baseline_signature_tables(train)
    action_hits = [
        record.delta_signature
        in action_table.get(record.action_name, global_table)
        for record in transfer
    ]
    state_hits = [
        record.delta_signature
        in state_table.get(record.graph_before.signature, global_table)
        for record in transfer
    ]
    baseline_recall = max(float(np.mean(action_hits)), float(np.mean(state_hits)))
    recall_gain_lower = _paired_bootstrap_lower(
        [float(value) for value in recall_hits],
        [
            float(action or state)
            for action, state in zip(action_hits, state_hits)
        ],
    )

    targets = np.stack([_event_vector(record) for record in transfer])
    probabilities = np.asarray(
        [prediction.delta_probabilities for prediction in predictions]
    )
    model_brier = float(np.mean((probabilities - targets) ** 2))
    action_probabilities: dict[str, np.ndarray] = {}
    global_probability = np.mean(
        np.stack([_event_vector(record) for record in train]),
        axis=0,
    )
    for action in {record.action_name for record in train}:
        rows = [
            _event_vector(record)
            for record in train
            if record.action_name == action
        ]
        action_probabilities[action] = np.mean(np.stack(rows), axis=0)
    baseline_probability_rows = np.stack(
        [
            action_probabilities.get(record.action_name, global_probability)
            for record in transfer
        ]
    )
    action_brier = float(
        np.mean((baseline_probability_rows - targets) ** 2)
    )

    train_embeddings = _load_embeddings(destination / "train_embeddings.jsonl")
    train_matrix = np.asarray(
        [item.vector for item in train_embeddings],
        dtype=np.float32,
    )
    identity_accuracy, identity_majority = _identity_probe(
        train_matrix,
        [item.source_game_id for item in train_embeddings],
    )
    sample = transfer[: min(128, len(transfer))]
    original_sample = predict_graph_details(
        model,
        [record.graph_before for record in sample],
        config=config,
        device=device,
    )
    permuted_sample = predict_graph_details(
        model,
        [
            record.graph_before.permuted(
                tuple(reversed(range(len(record.graph_before.nodes))))
            )
            for record in sample
        ],
        config=config,
        device=device,
    )
    augmentation_cosines = [
        float(np.dot(left.vector, right.vector))
        for left, right in zip(original_sample, permuted_sample)
    ]
    augmentation_cluster_consistency = float(
        np.mean(
            [
                memory.assign(
                    left.vector,
                    action_family=record.graph_before.action_family,
                    uncertainty=left.uncertainty,
                )
                == memory.assign(
                    right.vector,
                    action_family=record.graph_before.action_family,
                    uncertainty=right.uncertainty,
                )
                for record, left, right in zip(
                    sample,
                    original_sample,
                    permuted_sample,
                )
            ]
        )
    )

    train_value_by_action: dict[str, float] = {}
    for action in {record.action_name for record in train}:
        train_value_by_action[action] = float(
            np.mean(
                [
                    _utility(record)
                    for record in train
                    if record.action_name == action
                ]
            )
        )
    panels: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(transfer):
        panels[str(record.audit.get("panel_id", record.transition_id))].append(index)
    analogy_utilities = []
    baseline_utilities = []
    oracle_utilities = []
    analogy_completions = 0
    oracle_completion_panels = 0
    game_gains: dict[str, list[float]] = defaultdict(list)
    for indices in panels.values():
        analogy_index = max(
            indices,
            key=lambda index: (
                _analogy_score(predictions[index], transfer[index], memory),
                transfer[index].transition_id,
            ),
        )
        baseline_index = max(
            indices,
            key=lambda index: (
                train_value_by_action.get(transfer[index].action_name, 0.0),
                transfer[index].transition_id,
            ),
        )
        oracle_index = max(indices, key=lambda index: _utility(transfer[index]))
        analogy_value = _utility(transfer[analogy_index])
        baseline_value = _utility(transfer[baseline_index])
        analogy_utilities.append(analogy_value)
        baseline_utilities.append(baseline_value)
        oracle_utilities.append(_utility(transfer[oracle_index]))
        game_gains[transfer[analogy_index].source_game_id].append(
            analogy_value - baseline_value
        )
        panel_has_completion = any(
            bool(transfer[index].audit.get("level_complete", False))
            for index in indices
        )
        oracle_completion_panels += int(panel_has_completion)
        analogy_completions += int(
            bool(transfer[analogy_index].audit.get("level_complete", False))
        )
    utility_gain_lower = _paired_bootstrap_lower(
        analogy_utilities,
        baseline_utilities,
    )
    nonnegative_games = sum(
        float(np.mean(values)) >= 0.0
        for values in game_gains.values()
    )
    completion_fraction = (
        analogy_completions / oracle_completion_panels
        if oracle_completion_panels
        else 0.0
    )

    thresholds = manifest["gates"]
    gates = {
        "student_view_safe": bool(
            _read_json(destination / "teacher_qa.json")[
                "student_view_safe"
            ]
        ),
        "augmentation_invariant": bool(
            np.median(augmentation_cosines)
            >= thresholds["augmentation_cosine_median_minimum"]
            and augmentation_cluster_consistency
            >= thresholds["augmentation_cluster_consistency_minimum"]
        ),
        "clusters_stable": bool(
            registry.stability_ari
            >= thresholds["cluster_stability_ari_minimum"]
            and registry.eligible_coverage
            >= thresholds["eligible_cluster_coverage_minimum"]
        ),
        "retrieval_beats_baselines": bool(
            recall > baseline_recall and recall_gain_lower > 0.0
        ),
        "query_brier_beats_action_only": bool(model_brier < action_brier),
        "topology_is_used": bool(recall > shuffled_recall),
        "identity_leakage_bounded": bool(
            identity_accuracy - identity_majority
            <= thresholds["identity_gain_over_majority_maximum"]
        ),
        "global_utility_supported": bool(
            utility_gain_lower > 0.0
            and nonnegative_games
            >= thresholds["nonnegative_transfer_games_minimum"]
            and analogy_completions
            >= thresholds["completion_capture_absolute_minimum"]
            and completion_fraction
            >= thresholds["completion_capture_fraction_minimum"]
        ),
    }
    shadow_authorized = all(gates.values())
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "verdict": (
            "SAGE_MT_SHADOW_SUPPORTED"
            if shadow_authorized
            else "SAGE_MT_NOT_YET_SUPPORTED"
        ),
        "shadow_authorized": shadow_authorized,
        "metrics": {
            "cross_game_recall_at_8": recall,
            "best_baseline_recall_at_8": baseline_recall,
            "recall_gain_bootstrap_lower_95": recall_gain_lower,
            "relation_removed_recall_at_8": shuffled_recall,
            "query_delta_brier": model_brier,
            "action_only_delta_brier": action_brier,
            "identity_accuracy": identity_accuracy,
            "identity_majority": identity_majority,
            "augmentation_cosine_median": float(
                np.median(augmentation_cosines)
            ),
            "augmentation_cluster_consistency": augmentation_cluster_consistency,
            "cluster_stability_ari": registry.stability_ari,
            "eligible_cluster_coverage": registry.eligible_coverage,
            "utility_gain_bootstrap_lower_95": utility_gain_lower,
            "nonnegative_transfer_games": nonnegative_games,
            "analogy_completions": analogy_completions,
            "oracle_completion_panels": oracle_completion_panels,
            "completion_capture_fraction": completion_fraction,
        },
        "gates": gates,
        "boundaries": {
            "active_validation_opened": False,
            "holdout_opened": False,
            "controller_authority_promoted": False,
            "v4_15_modified": False,
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def load_shadow_advisor(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
    writer_path: str | Path | None = None,
    require_authorized: bool = True,
) -> MorphoTopologicalAnalogyAdvisor:
    import torch

    destination = Path(output_dir)
    result = _read_json(destination / "result.json")
    if require_authorized and not result.get("shadow_authorized", False):
        raise RuntimeError("SAGE-MT shadow gates are closed")
    checkpoint = torch.load(
        destination / "mt_model.pt",
        map_location=device,
        weights_only=False,
    )
    model, config, _ = load_mt_model(checkpoint, device=device)
    registry = ClusterRegistry.from_dict(
        _read_json(destination / "cluster_registry.json")
    )
    return MorphoTopologicalAnalogyAdvisor(
        model=model,
        model_config=config,
        memory=TransformationPrototypeMemory(registry),
        config=SageMTConfig(mode=SageMTMode.SHADOW),
        device=device,
        writer=(
            SageMTShadowWriter(writer_path)
            if writer_path is not None
            else None
        ),
    )


def prepare_shadow(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    result = _read_json(destination / "result.json")
    payload = {
        "format_version": FORMAT_VERSION,
        "shadow_enabled": bool(result.get("shadow_authorized", False)),
        "mode": "shadow" if result.get("shadow_authorized", False) else "off",
        "controller_authority": False,
        "active_validation_opened": False,
        "holdout_opened": False,
        "result_checksum": result["result_checksum"],
    }
    _write_json(destination / "shadow_activation.json", payload)
    return payload


def run_all(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
) -> dict[str, Any]:
    destination = Path(output_dir)
    if not (destination / "frozen_manifest.json").exists():
        freeze_manifest(output_dir=destination)
    compile_corpus(output_dir=destination)
    train_model(output_dir=destination, device=device)
    cluster_embeddings(output_dir=destination)
    result = evaluate(output_dir=destination, device=device)
    prepare_shadow(output_dir=destination)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "compile",
            "train",
            "cluster",
            "evaluate",
            "shadow",
            "run-all",
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-train-records", type=int)
    parser.add_argument("--maximum-transfer-records", type=int)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_manifest(output_dir=args.output_dir)
    elif args.command == "compile":
        result = compile_corpus(
            output_dir=args.output_dir,
            maximum_train_records=args.maximum_train_records,
            maximum_transfer_records=args.maximum_transfer_records,
        )
    elif args.command == "train":
        result = train_model(output_dir=args.output_dir, device=args.device)
    elif args.command == "cluster":
        result = cluster_embeddings(output_dir=args.output_dir)
    elif args.command == "evaluate":
        result = evaluate(output_dir=args.output_dir, device=args.device)
    elif args.command == "shadow":
        result = prepare_shadow(output_dir=args.output_dir)
    else:
        result = run_all(output_dir=args.output_dir, device=args.device)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTIVE_VALIDATION_GAMES",
    "DEFAULT_OUTPUT_DIR",
    "FINAL_HOLDOUT_GAMES",
    "FORMAT_VERSION",
    "HUMAN_TRAIN_GAMES",
    "MANIFEST_VERSION",
    "RESULT_VERSION",
    "TRANSFER_GAMES",
    "cluster_embeddings",
    "compile_corpus",
    "evaluate",
    "freeze_manifest",
    "load_manifest",
    "load_records",
    "load_shadow_advisor",
    "prepare_shadow",
    "run_all",
    "train_model",
]
