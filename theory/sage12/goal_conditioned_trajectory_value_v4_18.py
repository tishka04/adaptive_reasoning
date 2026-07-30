"""SAGE12 V4.18 goal-conditioned trajectory value.

The implementation deliberately streams the human corpus, keeps checkpoints
in an ignored bounded cache and wraps every CLI command in the shared storage
guard. True future outcomes are used only by offline oracle diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from theory.sage11.splits import NEURO_HOLDOUT_V1

from .action_target_data import build_action_target_trace, grid_sha256
from .artifact_budget import BudgetLimits, StorageGuard
from .counterfactual_semantic_panels_v4_11 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V411_DIR,
)
from .demonstration_milestone_policy_v4_15 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V415_DIR,
)
from .demonstration_milestone_policy_v4_15 import (
    PolicyBelief,
    _active_metrics,
    _advance_policy_belief,
    _file_fingerprint,
    _load_policy_checkpoint,
    _score_candidate_graphs,
    _zscore,
    train_demonstration_policy,
)
from .human_temporal_semantics_v4_14 import (
    ACTIVE_VALIDATION_GAMES,
    HUMAN_TRAIN_GAMES,
    TRANSFER_GAMES,
    TemporalBeliefState,
    _action_sequence_tables,
    _candidate_action_plan,
    _graph_for_action,
    _live_action_signature,
    _live_candidate_graph,
    _load_active_ebm,
    _paired_bootstrap_rows,
    _predict_candidate_rollouts,
    _prediction_features,
    evaluate_transfer_and_global_chain,
    train_temporal_student,
)
from .human_temporal_semantics_v4_14 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V414_DIR,
)
from .human_temporal_semantics_v4_14 import (
    _load_checkpoint as _load_temporal_checkpoint,
)
from .human_temporal_semantics_v4_14 import (
    load_teacher_records as load_temporal_records,
)
from .semantic_teacher_v4_9 import (
    ObjectRelativeGraph,
    _checksum,
    _file_sha256,
    _read_json,
    _write_json,
    _write_jsonl,
    compile_semantics,
)
from .sequence_transformation_policy_v4_17 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V417_DIR,
)

FORMAT_VERSION = "sage12-goal-conditioned-trajectory-value-v4.18"
MANIFEST_VERSION = "sage12-trajectory-value-manifest-v4.18"
TEACHER_VERSION = "sage12-trajectory-credit-record-v4.18"
CHECKPOINT_VERSION = "sage12-trajectory-critic-checkpoint-v4.18"
RESULT_VERSION = "sage12-trajectory-value-result-v4.18"
ACTIVE_VERSION = "sage12-trajectory-value-active-v4.18"

DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage12" / "goal_conditioned_trajectory_value_v4_18"
)
DEFAULT_CACHE_DIR = Path(".sage12_cache") / "v4_18"
PROTOCOL_PATH = (
    Path("reports")
    / "SAGE12_GOAL_CONDITIONED_TRAJECTORY_VALUE_V4_18_PROTOCOL.md"
)

SEED = 5_180
FEATURE_WIDTH = 512
HIDDEN_WIDTH = 128
LATENT_WIDTH = 64
HORIZONS = (8, 16, 32, 64)
GAMMA = 0.97
GOALS = (
    "motion",
    "object_change",
    "topology",
    "access",
    "terminal_progress",
    "risk",
    "overall",
)
IMMEDIATE_FACTORS = GOALS[:-1]
RELATION_FIELDS = frozenset(
    {
        "actor_relation",
        "actor_relative_direction",
        "path_status",
        "direction",
        "proximity",
        "aligned_row",
        "aligned_col",
        "is_actor",
        "relative_size",
    }
)
FORBIDDEN_GRAPH_FIELDS = frozenset(
    {
        "action_name",
        "game_id",
        "palette",
        "x",
        "y",
        "row",
        "col",
        "position",
        "absolute_x",
        "absolute_y",
    }
)
VALUE_COEFFICIENT = 0.5
TEMPORAL_EBM_COEFFICIENT = 0.5
BOOTSTRAP_SAMPLES = 2_000
ACTIVE_SEEDS = (0, 1, 2)
ACTIVE_ACTION_BUDGET = 1_000
ACTIVE_MAXIMUM_RESETS = 14
MODEL_PARAMETERS = {
    "epochs": 20,
    "fold_epochs": 8,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "value_loss_weight": 0.70,
    "factor_loss_weight": 0.20,
    "ranking_loss_weight": 0.10,
}


def _repo_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / ".git").exists():
        raise RuntimeError("V4.18 commands must run from the repository root")
    return root


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _required_sources(
    *,
    v415_dir: Path,
    v414_dir: Path,
    v411_dir: Path,
    v417_dir: Path,
) -> tuple[Path, ...]:
    return (
        v415_dir / "frozen_manifest.json",
        v415_dir / "teacher_qa.json",
        v415_dir / "demonstration_choices.jsonl",
        v415_dir / "logo_predictions.jsonl",
        v415_dir / "active_runs.jsonl",
        v414_dir / "frozen_manifest.json",
        v414_dir / "teacher_qa.json",
        v414_dir / "teacher_corpus.jsonl",
        v414_dir / "logo_predictions.jsonl",
        v411_dir / "frozen_manifest.json",
        v411_dir / "teacher_panels.jsonl",
        v417_dir / "result.json",
        v417_dir / "offline_predictions.jsonl",
        v417_dir / "offline_decisions.jsonl",
        v417_dir / "active_runs.jsonl",
        Path(__file__).resolve(),
        Path(__file__).with_name("artifact_budget.py").resolve(),
        PROTOCOL_PATH.resolve(),
        Path(".gitignore").resolve(),
    )


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    v415_dir: str | Path = DEFAULT_V415_DIR,
    v414_dir: str | Path = DEFAULT_V414_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    v417_dir: str | Path = DEFAULT_V417_DIR,
) -> dict[str, Any]:
    """Freeze all sources, comparison lanes and storage limits before fitting."""

    root = _repo_root()
    destination = _resolve(root, output_dir)
    sources = _required_sources(
        v415_dir=_resolve(root, v415_dir),
        v414_dir=_resolve(root, v414_dir),
        v411_dir=_resolve(root, v411_dir),
        v417_dir=_resolve(root, v417_dir),
    )
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(path.as_posix() for path in missing))
    limits = BudgetLimits()
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "split": {
            "human_train": list(HUMAN_TRAIN_GAMES),
            "offline_transfer": list(TRANSFER_GAMES),
            "active_validation": list(ACTIVE_VALIDATION_GAMES),
            "final_confirmation_closed": list(NEURO_HOLDOUT_V1),
        },
        "source_fingerprints": {
            path.relative_to(root).as_posix(): _file_fingerprint(path)
            for path in sources
        },
        "storage": {
            "cache_dir": _resolve(root, cache_dir).relative_to(root).as_posix(),
            "maximum_scratch_bytes": limits.maximum_scratch_bytes,
            "maximum_local_cache_bytes": limits.maximum_local_cache_bytes,
            "maximum_derived_file_bytes": limits.maximum_derived_file_bytes,
            "maximum_repository_bytes": limits.maximum_repository_bytes,
            "minimum_free_bytes": limits.minimum_free_bytes,
            "regenerable_v4_16_corpora_forbidden": [
                "train_embeddings.jsonl",
                "train_transitions.jsonl",
                "transfer_transitions.jsonl",
            ],
        },
        "checkpoint_regeneration": {
            "components": [
                "v4_14_temporal_student",
                "v4_14_temporal_ebm",
                "v4_15_demonstration_policy",
            ],
            "tracked_sources_only": True,
            "tracked_outputs_overwritten": False,
            "parity_diagnostic_not_gate": True,
        },
        "teacher": {
            "records": 5_661,
            "sequences": 41,
            "horizons": list(HORIZONS),
            "gamma": GAMMA,
            "goals": list(GOALS),
            "streaming": True,
            "unexecuted_regression_targets": False,
        },
        "model": {
            **MODEL_PARAMETERS,
            "feature_width": FEATURE_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "latent_width": LATENT_WIDTH,
            "raw_action_identity": False,
            "game_identity": False,
            "absolute_coordinates": False,
            "palette": False,
        },
        "offline": {
            "panels": 768,
            "arms": 2_831,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "value_coefficient": VALUE_COEFFICIENT,
            "temporal_ebm_coefficient": TEMPORAL_EBM_COEFFICIENT,
            "conditions": [
                "v4_15_policy",
                "v4_17_hybrid",
                "action_only",
                "v4_18_learned",
                "v4_18_without_relations",
                "trajectory_oracle_hybrid",
                "trajectory_oracle",
                "exact_oracle",
            ],
        },
        "active": {
            "games": list(ACTIVE_VALIDATION_GAMES),
            "seeds": list(ACTIVE_SEEDS),
            "fresh_runs": 9,
            "action_budget": ACTIVE_ACTION_BUDGET,
            "maximum_resets": ACTIVE_MAXIMUM_RESETS,
            "online_oracle_executed": False,
        },
        "authority": {
            "holdout_opened": False,
            "controller_authority_promoted": False,
        },
        "result_observed_at_freeze": False,
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root()
    payload = _read_json(_resolve(root, output_dir) / "frozen_manifest.json")
    if payload.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.18 manifest")
    expected = str(payload["manifest_checksum"])
    clean = dict(payload)
    clean.pop("manifest_checksum")
    if _checksum(clean) != expected:
        raise ValueError("V4.18 manifest checksum mismatch")
    return payload


def _verify_frozen_sources(manifest: Mapping[str, Any]) -> None:
    root = _repo_root()
    for relative, expected in manifest["source_fingerprints"].items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        if _file_fingerprint(path) != expected:
            raise ValueError(f"V4.18 frozen source drift: {relative}")


def _copy_minimal_sources(source: Path, destination: Path, names: Sequence[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source / name, destination / name)


def _v414_parity(rebuilt: Path, published: Path) -> dict[str, Any]:
    effects_total = 0
    effect_bits_equal = 0
    probability_delta = 0.0
    records = 0
    for fresh, reference in zip(
        _jsonl_rows(rebuilt / "logo_predictions.jsonl"),
        _jsonl_rows(published / "logo_predictions.jsonl"),
        strict=True,
    ):
        if fresh["example_id"] != reference["example_id"]:
            raise ValueError("V4.14 rebuilt prediction order drift")
        fresh_temporal = fresh["probabilities"]["temporal"]
        old_temporal = reference["probabilities"]["temporal"]
        for effect in sorted(old_temporal):
            new_value = float(fresh_temporal[effect])
            old_value = float(old_temporal[effect])
            effects_total += 1
            effect_bits_equal += int((new_value >= 0.5) == (old_value >= 0.5))
            probability_delta += abs(new_value - old_value)
        records += 1
    return {
        "records": records,
        "effect_bit_agreement": effect_bits_equal / max(effects_total, 1),
        "mean_probability_delta": probability_delta / max(effects_total, 1),
    }


def _v415_parity(rebuilt: Path, published: Path) -> dict[str, Any]:
    selected_equal = 0
    milestone_equal = 0
    records = 0
    for fresh, reference in zip(
        _jsonl_rows(rebuilt / "logo_predictions.jsonl"),
        _jsonl_rows(published / "logo_predictions.jsonl"),
        strict=True,
    ):
        if fresh["example_id"] != reference["example_id"]:
            raise ValueError("V4.15 rebuilt prediction order drift")
        selected_equal += int(
            fresh["predictions"]["learned_milestone"]["selected_index"]
            == reference["predictions"]["learned_milestone"]["selected_index"]
        )
        milestone_equal += int(
            int(np.argmax(fresh["milestone_probabilities"]))
            == int(np.argmax(reference["milestone_probabilities"]))
        )
        records += 1
    return {
        "records": records,
        "selected_action_agreement": selected_equal / max(records, 1),
        "milestone_agreement": milestone_equal / max(records, 1),
    }


def rebuild_required_checkpoints(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Rebuild only V4.14/V4.15 checkpoints into the ignored bounded cache."""

    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    v414_source = _resolve(root, DEFAULT_V414_DIR)
    v415_source = _resolve(root, DEFAULT_V415_DIR)
    v414_cache = cache / "v4_14"
    v415_cache = cache / "v4_15"
    _copy_minimal_sources(
        v414_source,
        v414_cache,
        ("frozen_manifest.json", "teacher_qa.json", "teacher_corpus.jsonl"),
    )
    _copy_minimal_sources(
        v415_source,
        v415_cache,
        ("frozen_manifest.json", "teacher_qa.json", "demonstration_choices.jsonl"),
    )
    started = time.perf_counter()
    temporal = train_temporal_student(
        output_dir=v414_cache,
        requested_device=requested_device,
    )
    temporal_evaluation = evaluate_transfer_and_global_chain(
        output_dir=v414_cache,
        requested_device=requested_device,
    )
    policy = train_demonstration_policy(
        output_dir=v415_cache,
        requested_device=requested_device,
    )
    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "elapsed_seconds": time.perf_counter() - started,
        "v4_14": {
            "selected_device": temporal.get("selected_device"),
            "semantic_result_checksum": temporal.get("result_checksum"),
            "global_result_checksum": temporal_evaluation.get("result_checksum"),
            "parity": _v414_parity(v414_cache, v414_source),
        },
        "v4_15": {
            "selected_device": policy.get("selected_device"),
            "result_checksum": policy.get("result_checksum"),
            "parity": _v415_parity(v415_cache, v415_source),
        },
        "checkpoints": {},
        "tracked_outputs_overwritten": False,
        "v4_16_corpora_regenerated": False,
    }
    for name, path in {
        "v4_14_temporal_student": v414_cache / "temporal_student.pt",
        "v4_14_temporal_ebm": v414_cache / "trajectory_ebm.pt",
        "v4_15_policy": v415_cache / "demonstration_policy.pt",
    }.items():
        result["checkpoints"][name] = {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "checkpoint_regeneration.json", result)
    return result


def _stable_bucket(token: str) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FEATURE_WIDTH, (
        1.0 if digest[8] & 1 else -1.0
    )


def _graph_tokens(
    graph: Mapping[str, Any] | ObjectRelativeGraph,
    *,
    remove_relations: bool = False,
) -> tuple[str, ...]:
    root = graph.root if isinstance(graph, ObjectRelativeGraph) else graph["root"]
    neighbors = (
        graph.neighbors if isinstance(graph, ObjectRelativeGraph) else graph["neighbors"]
    )
    tokens: list[str] = []
    root_fields = []
    for key, value in sorted(root.items()):
        if key in FORBIDDEN_GRAPH_FIELDS or (
            remove_relations and key in RELATION_FIELDS
        ):
            continue
        tokens.append(f"root:{key}={value}")
        root_fields.append(f"{key}={value}")
    tokens.append("root_joint:" + "|".join(root_fields))
    for neighbor in neighbors:
        fields = []
        for key, value in sorted(neighbor.items()):
            if key in FORBIDDEN_GRAPH_FIELDS or (
                remove_relations and key in RELATION_FIELDS
            ):
                continue
            tokens.append(f"neighbor:{key}={value}")
            fields.append(f"{key}={value}")
        tokens.append("neighbor_joint:" + "|".join(fields))
    tokens.append(f"neighbor_count:{min(len(neighbors), 16)}")
    return tuple(tokens)


def graph_feature_vector(
    graph: Mapping[str, Any] | ObjectRelativeGraph,
    *,
    remove_relations: bool = False,
) -> np.ndarray:
    counts = Counter(_graph_tokens(graph, remove_relations=remove_relations))
    vector = np.zeros(FEATURE_WIDTH, dtype=np.float32)
    for token, count in counts.items():
        index, sign = _stable_bucket(token)
        vector[index] += sign * math.sqrt(float(count))
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _factor_events(row: Mapping[str, Any]) -> dict[str, bool]:
    effects = row["teacher"]["observed_effects"]
    return {
        "motion": any(
            bool(effects.get(name, False))
            for name in ("moved", "target_moved", "actor_approached_root")
        ),
        "object_change": any(
            bool(effects.get(name, False))
            for name in (
                "changed",
                "local_change",
                "target_created",
                "target_removed",
                "target_moved",
            )
        ),
        "topology": any(
            bool(effects.get(name, False))
            for name in (
                "contact_gained",
                "contact_lost",
                "path_opened",
                "path_closed",
                "reachable_area_increased",
                "reachable_area_decreased",
            )
        ),
        "access": any(
            bool(effects.get(name, False))
            for name in ("path_opened", "contact_gained", "reachable_area_increased")
        ),
        "terminal_progress": bool(effects.get("level_complete", False)),
        "risk": bool(effects.get("risk", False) or effects.get("game_over", False)),
    }


def _overall_reward(events: Mapping[str, bool], *, productive: bool) -> float:
    return float(
        0.05 * events["motion"]
        + 0.10 * events["object_change"]
        + 0.15 * events["topology"]
        + 0.35 * events["access"]
        + 1.00 * events["terminal_progress"]
        + 0.05 * productive
        - 1.00 * events["risk"]
    )


def _trajectory_targets(
    sequence: Sequence[Mapping[str, Any]],
    index: int,
) -> np.ndarray:
    output = np.zeros((len(GOALS), len(HORIZONS)), dtype=np.float32)
    events = [_factor_events(row) for row in sequence[index:]]
    for horizon_index, horizon in enumerate(HORIZONS):
        window = events[:horizon]
        for goal_index, goal in enumerate(IMMEDIATE_FACTORS):
            occurrences = [
                GAMMA**offset for offset, event in enumerate(window) if event[goal]
            ]
            value = max(occurrences, default=0.0)
            output[goal_index, horizon_index] = -value if goal == "risk" else value
        overall = sum(
            (GAMMA**offset)
            * _overall_reward(
                event,
                productive=bool(sequence[index + offset]["teacher"]["productive"]),
            )
            for offset, event in enumerate(window)
        )
        output[GOALS.index("overall"), horizon_index] = float(
            np.clip(overall, -1.0, 1.0)
        )
    return output


def _sparse(vector: np.ndarray) -> list[list[float]]:
    return [
        [int(index), float(vector[index])]
        for index in np.flatnonzero(vector)
    ]


def _dense(sparse: Sequence[Sequence[float]]) -> np.ndarray:
    vector = np.zeros(FEATURE_WIDTH, dtype=np.float32)
    for index, value in sparse:
        vector[int(index)] = float(value)
    return vector


def _sequence_stream(path: Path) -> Iterator[list[dict[str, Any]]]:
    current_key: tuple[str, str] | None = None
    current: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in _jsonl_rows(path):
        audit = row["audit"]
        key = (str(audit["source_file"]), str(audit["episode_id"]))
        if current_key is None:
            current_key = key
        if key != current_key:
            if key in seen:
                raise ValueError("V4.18 source sequence reappeared after closure")
            seen.add(current_key)
            yield current
            current = []
            current_key = key
        current.append(row)
    if current:
        yield current


def compile_trajectory_credit(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scratch_dir: str | Path,
) -> dict[str, Any]:
    """Stream sequence-level credit into compact sparse training records."""

    root = _repo_root()
    destination = _resolve(root, output_dir)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    source = _resolve(root, DEFAULT_V415_DIR) / "demonstration_choices.jsonl"
    temporary = Path(scratch_dir) / "trajectory_credit.jsonl"
    records = 0
    sequences = 0
    events: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    games: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for sequence in _sequence_stream(source):
            sequences += 1
            for index, row in enumerate(sequence):
                selected_index = int(row["teacher"]["selected_index"])
                candidates = row["student_view"]["candidates"]
                negative_index = (
                    selected_index
                    if len(candidates) == 1
                    else (selected_index + 1) % len(candidates)
                )
                selected = candidates[selected_index]
                negative = candidates[negative_index]
                immediate = _factor_events(row)
                events.update(name for name, active in immediate.items() if active)
                targets = _trajectory_targets(sequence, index)
                for horizon_index, horizon in enumerate(HORIZONS):
                    positives[str(horizon)] += int(
                        targets[GOALS.index("overall"), horizon_index] > 0.0
                    )
                game_id = str(row["audit"]["game_id"]).split("-", 1)[0]
                games[game_id] += 1
                compiled = {
                    "format_version": TEACHER_VERSION,
                    "student_view": {
                        "selected": _sparse(graph_feature_vector(selected)),
                        "selected_without_relations": _sparse(
                            graph_feature_vector(selected, remove_relations=True)
                        ),
                        "negative": _sparse(graph_feature_vector(negative)),
                    },
                    "teacher": {
                        "immediate_factors": [
                            int(immediate[name]) for name in IMMEDIATE_FACTORS
                        ],
                        "trajectory_values": targets.tolist(),
                        "success_weight": float(row["teacher"]["success_weight"]),
                    },
                    "audit": {
                        "example_id": row["audit"]["example_id"],
                        "game_id": game_id,
                        "sequence_key": "|".join(
                            (
                                str(row["audit"]["source_file"]),
                                str(row["audit"]["episode_id"]),
                            )
                        ),
                        "sequence_index": int(row["audit"]["sequence_index"]),
                        "action_family": str(
                            selected["root"].get("action_family", "other")
                        ),
                    },
                }
                output.write(
                    json.dumps(
                        compiled,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                output.write("\n")
                records += 1
    if records != manifest["teacher"]["records"]:
        raise ValueError(f"V4.18 teacher record drift: {records}")
    if sequences != manifest["teacher"]["sequences"]:
        raise ValueError(f"V4.18 teacher sequence drift: {sequences}")
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / "trajectory_credit.jsonl"
    os.replace(temporary, final)
    qa: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "records": records,
        "sequences": sequences,
        "games": dict(games),
        "event_counts": dict(events),
        "positive_overall_by_horizon": dict(positives),
        "streamed": True,
        "unexecuted_regression_targets": 0,
        "student_view_safe": True,
        "artifact_bytes": final.stat().st_size,
        "artifact_sha256": _file_sha256(final),
        "teacher_ready": set(games) == set(HUMAN_TRAIN_GAMES),
    }
    qa["qa_checksum"] = _checksum(qa)
    _write_json(destination / "teacher_qa.json", qa)
    return qa


def _critic_type() -> type[Any]:
    import torch

    class GoalConditionedTrajectoryCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(FEATURE_WIDTH, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, LATENT_WIDTH),
                torch.nn.ReLU(),
            )
            self.value_head = torch.nn.Linear(
                LATENT_WIDTH,
                len(GOALS) * len(HORIZONS),
            )
            self.factor_head = torch.nn.Linear(
                LATENT_WIDTH,
                len(IMMEDIATE_FACTORS),
            )

        def forward(self, features: Any) -> tuple[Any, Any, Any]:
            latent = self.encoder(features)
            values = torch.tanh(self.value_head(latent)).reshape(
                -1,
                len(GOALS),
                len(HORIZONS),
            )
            return values, self.factor_head(latent), latent

    return GoalConditionedTrajectoryCritic


def _credit_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        "selected": np.asarray(
            [_dense(row["student_view"]["selected"]) for row in rows],
            dtype=np.float32,
        ),
        "removed": np.asarray(
            [
                _dense(row["student_view"]["selected_without_relations"])
                for row in rows
            ],
            dtype=np.float32,
        ),
        "negative": np.asarray(
            [_dense(row["student_view"]["negative"]) for row in rows],
            dtype=np.float32,
        ),
        "values": np.asarray(
            [row["teacher"]["trajectory_values"] for row in rows],
            dtype=np.float32,
        ),
        "factors": np.asarray(
            [row["teacher"]["immediate_factors"] for row in rows],
            dtype=np.float32,
        ),
    }


def _select_device(requested: str, batch_size: int) -> tuple[str, dict[str, Any]]:
    import torch

    requested = str(requested).lower()
    result: dict[str, Any] = {
        "requested": requested,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for V4.18 but unavailable")
        result["selected_reason"] = "explicit"
        return requested, result
    if not torch.cuda.is_available():
        result["selected_reason"] = "cuda_unavailable"
        return "cpu", result
    generator = np.random.default_rng(SEED)
    sample = generator.normal(size=(batch_size, FEATURE_WIDTH)).astype(np.float32)
    timings: dict[str, float] = {}
    for device in ("cpu", "cuda:0"):
        torch.manual_seed(SEED)
        model = _critic_type()().to(device)
        tensor = torch.as_tensor(sample, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            values, logits, _latent = model(tensor)
            loss = values.square().mean() + logits.square().mean()
            loss.backward()
            optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings[device] = time.perf_counter() - started
    speedup = timings["cpu"] / max(timings["cuda:0"], 1e-9)
    selected = "cuda:0" if speedup >= 1.10 else "cpu"
    result.update(
        {
            "timings_seconds": timings,
            "cuda_speedup": speedup,
            "selection_threshold": 1.10,
            "selected_reason": (
                "cuda_effectively_faster"
                if selected.startswith("cuda")
                else "cpu_not_slower_for_compact_model"
            ),
        }
    )
    return selected, result


def _fit_critic(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[Any, list[dict[str, float]]]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _critic_type()().to(device)
    tensors = {
        name: torch.as_tensor(values, device=device)
        for name, values in arrays.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_PARAMETERS["learning_rate"],
        weight_decay=MODEL_PARAMETERS["weight_decay"],
    )
    generator = np.random.default_rng(seed)
    overall = GOALS.index("overall")
    horizon = HORIZONS.index(32)
    history = []
    model.train()
    for epoch in range(epochs):
        order = generator.permutation(indices)
        losses = []
        for start in range(0, len(order), batch_size):
            batch_indices = torch.as_tensor(
                order[start : start + batch_size],
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            values, logits, _latent = model(tensors["selected"][batch_indices])
            negative_values, _negative_logits, _negative_latent = model(
                tensors["negative"][batch_indices]
            )
            value_loss = torch.nn.functional.smooth_l1_loss(
                values,
                tensors["values"][batch_indices],
            )
            factor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                tensors["factors"][batch_indices],
            )
            weights = tensors["values"][batch_indices, overall, horizon].clamp_min(0)
            ranking = torch.relu(
                0.10
                + negative_values[:, overall, horizon]
                - values[:, overall, horizon]
            )
            ranking_loss = (ranking * weights).sum() / weights.sum().clamp_min(1.0)
            loss = (
                MODEL_PARAMETERS["value_loss_weight"] * value_loss
                + MODEL_PARAMETERS["factor_loss_weight"] * factor_loss
                + MODEL_PARAMETERS["ranking_loss_weight"] * ranking_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(
                (
                    float(loss.detach().cpu()),
                    float(value_loss.detach().cpu()),
                    float(factor_loss.detach().cpu()),
                    float(ranking_loss.detach().cpu()),
                )
            )
        mean = np.mean(losses, axis=0)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(mean[0]),
                "value_loss": float(mean[1]),
                "factor_loss": float(mean[2]),
                "ranking_loss": float(mean[3]),
            }
        )
    return model, history


def _predict_critic(
    model: Any,
    features: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    outputs = []
    factors = []
    latents = []
    with torch.no_grad():
        for start in range(0, len(features), 1024):
            values, logits, latent = model(
                torch.as_tensor(
                    features[start : start + 1024],
                    dtype=torch.float32,
                    device=device,
                )
            )
            outputs.append(values.cpu().numpy())
            factors.append(torch.sigmoid(logits).cpu().numpy())
            latents.append(latent.cpu().numpy())
    return (
        np.concatenate(outputs),
        np.concatenate(factors),
        np.concatenate(latents),
    )


def _action_value_baseline(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    by_action: dict[str, list[np.ndarray]] = defaultdict(list)
    all_values = []
    for index in indices:
        row = rows[int(index)]
        values = np.asarray(row["teacher"]["trajectory_values"], dtype=np.float32)
        all_values.append(values)
        by_action[str(row["audit"]["action_family"])].append(values)
    global_mean = np.mean(all_values, axis=0)
    return {
        "global": global_mean.tolist(),
        "by_action_family": {
            action: np.mean(values, axis=0).tolist()
            for action, values in sorted(by_action.items())
        },
    }


def _baseline_values(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    baseline: Mapping[str, Any],
) -> np.ndarray:
    global_mean = np.asarray(baseline["global"], dtype=np.float32)
    by_action = baseline["by_action_family"]
    return np.asarray(
        [
            by_action.get(
                str(rows[int(index)]["audit"]["action_family"]),
                global_mean,
            )
            for index in indices
        ],
        dtype=np.float32,
    )


def _macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    scores = []
    predicted = probabilities >= 0.5
    for index in range(labels.shape[1]):
        truth = labels[:, index] >= 0.5
        guess = predicted[:, index]
        tp = int(np.sum(truth & guess))
        fp = int(np.sum(~truth & guess))
        fn = int(np.sum(truth & ~guess))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _identity_probe(
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    games = np.asarray([str(row["audit"]["game_id"]) for row in rows])
    sequences = np.asarray([str(row["audit"]["sequence_key"]) for row in rows])
    folds = np.asarray(
        [
            int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 5
            for value in sequences
        ]
    )
    correct = 0
    total = 0
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        if not np.any(test):
            continue
        centroids = {}
        for game in HUMAN_TRAIN_GAMES:
            mask = train & (games == game)
            if np.any(mask):
                centroid = embeddings[mask].mean(axis=0)
                centroids[game] = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        normalized = embeddings[test] / np.maximum(
            np.linalg.norm(embeddings[test], axis=1, keepdims=True),
            1e-9,
        )
        predicted = [
            max(
                centroids,
                key=lambda game: float(np.dot(vector, centroids[game])),
            )
            for vector in normalized
        ]
        correct += sum(
            predicted_game == actual_game
            for predicted_game, actual_game in zip(
                predicted,
                games[test],
                strict=True,
            )
        )
        total += int(np.sum(test))
    majority = max(Counter(games).values()) / len(games)
    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "majority": float(majority),
        "increment": accuracy - float(majority),
    }


def train_trajectory_critic(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Run six human-game LOGO folds and fit the compact final critic."""

    import torch

    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir) / "critic"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    qa = _read_json(destination / "teacher_qa.json")
    if not qa["teacher_ready"]:
        raise RuntimeError("V4.18 teacher QA failed")
    rows = list(_jsonl_rows(destination / "trajectory_credit.jsonl"))
    arrays = _credit_arrays(rows)
    device, benchmark = _select_device(
        requested_device,
        int(MODEL_PARAMETERS["batch_size"]),
    )
    predictions = np.zeros_like(arrays["values"])
    removed_predictions = np.zeros_like(arrays["values"])
    factor_predictions = np.zeros_like(arrays["factors"])
    baseline_predictions = np.zeros_like(arrays["values"])
    logo_rows = []
    folds = []
    started = time.perf_counter()
    for fold_index, held_game in enumerate(HUMAN_TRAIN_GAMES):
        train_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["audit"]["game_id"] != held_game
            ],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["audit"]["game_id"] == held_game
            ],
            dtype=np.int64,
        )
        model, history = _fit_critic(
            arrays,
            train_indices,
            device=device,
            epochs=int(MODEL_PARAMETERS["fold_epochs"]),
            batch_size=int(MODEL_PARAMETERS["batch_size"]),
            seed=SEED + fold_index,
        )
        predicted, predicted_factors, _latent = _predict_critic(
            model,
            arrays["selected"][test_indices],
            device=device,
        )
        predicted_removed, _removed_factors, _removed_latent = _predict_critic(
            model,
            arrays["removed"][test_indices],
            device=device,
        )
        baseline = _action_value_baseline(rows, train_indices)
        predicted_baseline = _baseline_values(rows, test_indices, baseline)
        predictions[test_indices] = predicted
        removed_predictions[test_indices] = predicted_removed
        factor_predictions[test_indices] = predicted_factors
        baseline_predictions[test_indices] = predicted_baseline
        folds.append(
            {
                "held_game": held_game,
                "records": len(test_indices),
                "value_mae": float(
                    np.mean(np.abs(predicted - arrays["values"][test_indices]))
                ),
                "action_only_value_mae": float(
                    np.mean(
                        np.abs(
                            predicted_baseline - arrays["values"][test_indices]
                        )
                    )
                ),
                "without_relations_value_mae": float(
                    np.mean(
                        np.abs(
                            predicted_removed - arrays["values"][test_indices]
                        )
                    )
                ),
                "factor_macro_f1": _macro_f1(
                    arrays["factors"][test_indices],
                    predicted_factors,
                ),
                "final_loss": history[-1],
            }
        )
        for local_index, row_index in enumerate(test_indices):
            logo_rows.append(
                {
                    "format_version": FORMAT_VERSION,
                    "example_id": rows[int(row_index)]["audit"]["example_id"],
                    "game_id": held_game,
                    "predicted_values": predicted[local_index].tolist(),
                    "without_relations_values": predicted_removed[
                        local_index
                    ].tolist(),
                    "action_only_values": predicted_baseline[local_index].tolist(),
                    "target_values": arrays["values"][row_index].tolist(),
                }
            )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    all_indices = np.arange(len(rows), dtype=np.int64)
    full_model, history = _fit_critic(
        arrays,
        all_indices,
        device=device,
        epochs=int(MODEL_PARAMETERS["epochs"]),
        batch_size=int(MODEL_PARAMETERS["batch_size"]),
        seed=SEED + 100,
    )
    _full_values, _full_factors, embeddings = _predict_critic(
        full_model,
        arrays["selected"],
        device=device,
    )
    checkpoint_path = cache / "trajectory_critic.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "state_dict": full_model.state_dict(),
            "parameters": {
                "feature_width": FEATURE_WIDTH,
                "hidden_width": HIDDEN_WIDTH,
                "latent_width": LATENT_WIDTH,
                "goals": list(GOALS),
                "horizons": list(HORIZONS),
            },
        },
        checkpoint_path,
    )
    _write_jsonl(destination / "logo_predictions.jsonl", logo_rows)
    _write_jsonl(destination / "folds.jsonl", folds)
    action_baseline = _action_value_baseline(rows, all_indices)
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "selected_device": device,
        "device_benchmark": benchmark,
        "training_seconds": time.perf_counter() - started,
        "final_loss": history[-1],
        "leave_one_game_out": {
            "value_mae": float(np.mean(np.abs(predictions - arrays["values"]))),
            "action_only_value_mae": float(
                np.mean(np.abs(baseline_predictions - arrays["values"]))
            ),
            "without_relations_value_mae": float(
                np.mean(np.abs(removed_predictions - arrays["values"]))
            ),
            "factor_macro_f1": _macro_f1(
                arrays["factors"],
                factor_predictions,
            ),
            "per_game": folds,
        },
        "game_identity_probe": _identity_probe(embeddings, rows),
        "action_only_baseline": action_baseline,
        "checkpoint": {
            "path": checkpoint_path.relative_to(root).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _file_sha256(checkpoint_path),
        },
        "teacher_sha256": _file_sha256(destination / "trajectory_credit.jsonl"),
    }
    metadata["training_checksum"] = _checksum(metadata)
    _write_json(destination / "checkpoint_metadata.json", metadata)
    return metadata


def _load_critic(
    cache_dir: Path,
    *,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    checkpoint = torch.load(
        cache_dir / "critic" / "trajectory_critic.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported V4.18 checkpoint")
    model = _critic_type()().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _resolved_device(destination: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return str(_read_json(destination / "checkpoint_metadata.json")["selected_device"])


def _critic_scores(
    model: Any,
    graphs: Sequence[Mapping[str, Any] | ObjectRelativeGraph],
    *,
    device: str,
    remove_relations: bool = False,
) -> np.ndarray:
    features = np.asarray(
        [
            graph_feature_vector(graph, remove_relations=remove_relations)
            for graph in graphs
        ],
        dtype=np.float32,
    )
    values, _factors, _latents = _predict_critic(
        model,
        features,
        device=device,
    )
    return values[:, GOALS.index("overall"), HORIZONS.index(32)]


def _action_only_scores(
    graphs: Sequence[Mapping[str, Any] | ObjectRelativeGraph],
    baseline: Mapping[str, Any],
) -> np.ndarray:
    global_values = np.asarray(baseline["global"], dtype=np.float32)
    by_action = baseline["by_action_family"]
    scores = []
    for graph in graphs:
        root = graph.root if isinstance(graph, ObjectRelativeGraph) else graph["root"]
        values = np.asarray(
            by_action.get(str(root.get("action_family", "other")), global_values),
            dtype=np.float32,
        )
        scores.append(values[GOALS.index("overall"), HORIZONS.index(32)])
    return np.asarray(scores, dtype=np.float64)


def _compose_scores(
    policy: Sequence[float],
    value: Sequence[float],
    energy: Sequence[float],
) -> np.ndarray:
    return (
        _zscore(policy)
        + VALUE_COEFFICIENT * _zscore(value)
        - TEMPORAL_EBM_COEFFICIENT * _zscore(energy)
    )


def _selected_index(scores: Sequence[float], arm_indices: Sequence[int]) -> int:
    return max(
        range(len(scores)),
        key=lambda index: (float(scores[index]), -int(arm_indices[index])),
    )


def _decision_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_game = {}
    for game in sorted({str(row["game_id"]) for row in rows}):
        selected = [row for row in rows if row["game_id"] == game]
        per_game[game] = {
            "panels": len(selected),
            "mean_utility": float(
                np.mean([float(row["utility"]) for row in selected])
            ),
            "mean_regret": float(
                np.mean([float(row["regret"]) for row in selected])
            ),
            "completion_arms": int(
                sum(bool(row["completion"]) for row in selected)
            ),
            "exact_oracle_accuracy": float(
                np.mean([bool(row["exact_oracle_action"]) for row in selected])
            ),
        }
    return {
        "panels": len(rows),
        "mean_utility": float(
            np.mean([float(row["utility"]) for row in rows])
        ),
        "mean_regret": float(np.mean([float(row["regret"]) for row in rows])),
        "completion_arms": int(sum(bool(row["completion"]) for row in rows)),
        "exact_oracle_accuracy": float(
            np.mean([bool(row["exact_oracle_action"]) for row in rows])
        ),
        "per_game": per_game,
    }


def evaluate_offline(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Evaluate every registered lane on the immutable V4.11 transfer panels."""

    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    device = _resolved_device(destination, requested_device)
    model, checkpoint = _load_critic(cache, device=device)
    if checkpoint["manifest_checksum"] != manifest["manifest_checksum"]:
        raise ValueError("V4.18 critic/manifest mismatch")
    metadata = _read_json(destination / "checkpoint_metadata.json")
    raw_panels = {
        row["panel_id"]: row
        for row in _jsonl_rows(_resolve(root, DEFAULT_V411_DIR) / "teacher_panels.jsonl")
        if row["audit"]["game_id"] in TRANSFER_GAMES
    }
    component_rows = list(
        _jsonl_rows(_resolve(root, DEFAULT_V417_DIR) / "offline_predictions.jsonl")
    )
    conditions = list(manifest["offline"]["conditions"])
    decisions: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in conditions
    }
    prediction_rows = []
    started = time.perf_counter()
    for component in component_rows:
        panel_id = str(component["panel_id"])
        panel = raw_panels[panel_id]
        raw_arms = sorted(panel["arms"], key=lambda arm: int(arm["arm_index"]))
        component_arms = sorted(
            component["arms"],
            key=lambda arm: int(arm["arm_index"]),
        )
        arm_indices = [int(arm["arm_index"]) for arm in component_arms]
        if arm_indices != [int(arm["arm_index"]) for arm in raw_arms]:
            raise ValueError(f"V4.18 arm alignment drift: {panel_id}")
        graphs = [arm["model_graph"] for arm in raw_arms]
        learned = _critic_scores(model, graphs, device=device)
        removed = _critic_scores(
            model,
            graphs,
            device=device,
            remove_relations=True,
        )
        action_only = _action_only_scores(
            graphs,
            metadata["action_only_baseline"],
        )
        policy = np.asarray(
            [float(arm["policy_score"]) for arm in component_arms],
            dtype=np.float64,
        )
        energy = np.asarray(
            [float(arm["temporal_energy"]) for arm in component_arms],
            dtype=np.float64,
        )
        v417 = np.asarray(
            [float(arm["hybrid_score"]) for arm in component_arms],
            dtype=np.float64,
        )
        utility = np.asarray(
            [float(arm["utility"]) for arm in component_arms],
            dtype=np.float64,
        )
        trajectory_oracle = np.asarray(
            [float(arm["teacher"]["horizon_return"]) for arm in raw_arms],
            dtype=np.float64,
        )
        completion = [bool(arm["completion"]) for arm in component_arms]
        score_by_condition = {
            "v4_15_policy": policy,
            "v4_17_hybrid": v417,
            "action_only": action_only,
            "v4_18_learned": _compose_scores(policy, learned, energy),
            "v4_18_without_relations": _compose_scores(
                policy,
                removed,
                energy,
            ),
            "trajectory_oracle_hybrid": _compose_scores(
                policy,
                trajectory_oracle,
                energy,
            ),
            "trajectory_oracle": trajectory_oracle,
            "exact_oracle": utility,
        }
        exact_index = _selected_index(utility, arm_indices)
        maximum_utility = float(utility[exact_index])
        for condition in conditions:
            selected = _selected_index(score_by_condition[condition], arm_indices)
            decisions[condition].append(
                {
                    "format_version": FORMAT_VERSION,
                    "condition": condition,
                    "panel_id": panel_id,
                    "game_id": component["game_id"],
                    "selected_arm": arm_indices[selected],
                    "utility": float(utility[selected]),
                    "regret": maximum_utility - float(utility[selected]),
                    "completion": completion[selected],
                    "exact_oracle_action": selected == exact_index,
                }
            )
        prediction_rows.append(
            {
                "format_version": FORMAT_VERSION,
                "panel_id": panel_id,
                "game_id": component["game_id"],
                "arms": [
                    {
                        "arm_index": arm_indices[index],
                        "learned_value": float(learned[index]),
                        "without_relations_value": float(removed[index]),
                        "action_only_value": float(action_only[index]),
                        "trajectory_oracle_value": float(trajectory_oracle[index]),
                        "utility": float(utility[index]),
                        "completion": completion[index],
                    }
                    for index in range(len(arm_indices))
                ],
            }
        )
    all_decisions = [
        row for condition in conditions for row in decisions[condition]
    ]
    predictions_path = destination / "offline_predictions.jsonl"
    decisions_path = destination / "offline_decisions.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(decisions_path, all_decisions)
    summaries = {
        condition: _decision_summary(decisions[condition])
        for condition in conditions
    }
    v415 = decisions["v4_15_policy"]
    comparisons = {
        condition: _paired_bootstrap_rows(
            decisions[condition],
            v415,
            samples=BOOTSTRAP_SAMPLES,
            seed=SEED + index,
        )
        for index, condition in enumerate(conditions)
        if condition != "v4_15_policy"
    }
    learned_vs_action_only = _paired_bootstrap_rows(
        decisions["v4_18_learned"],
        decisions["action_only"],
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 99,
    )
    learned_vs_without_relations = _paired_bootstrap_rows(
        decisions["v4_18_learned"],
        decisions["v4_18_without_relations"],
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 100,
    )
    learned_by_game = summaries["v4_18_learned"]["per_game"]
    baseline_by_game = summaries["v4_15_policy"]["per_game"]
    nonnegative_games = sum(
        learned_by_game[game]["mean_utility"]
        >= baseline_by_game[game]["mean_utility"]
        for game in TRANSFER_GAMES
    )
    oracle_supported = bool(
        comparisons["trajectory_oracle_hybrid"]["ci_low"] > 0.0
        and summaries["trajectory_oracle_hybrid"]["completion_arms"]
        >= summaries["v4_15_policy"]["completion_arms"]
    )
    oracle_gain = comparisons["trajectory_oracle_hybrid"]["mean_gain"]
    learned_gain = comparisons["v4_18_learned"]["mean_gain"]
    oracle_capture = (
        learned_gain / oracle_gain if oracle_gain > 0.0 else float("-inf")
    )
    learned_supported = bool(
        comparisons["v4_18_learned"]["ci_low"] > 0.0
        and learned_vs_action_only["ci_low"] > 0.0
        and nonnegative_games >= 5
        and learned_vs_without_relations["ci_low"] > 0.0
        and oracle_capture >= 0.25
        and summaries["v4_18_learned"]["completion_arms"] >= 1
        and summaries["v4_18_learned"]["completion_arms"]
        >= math.ceil(
            summaries["trajectory_oracle_hybrid"]["completion_arms"] / 2
        )
        and metadata["leave_one_game_out"]["value_mae"]
        < metadata["leave_one_game_out"]["action_only_value_mae"]
        and metadata["game_identity_probe"]["increment"] <= 0.10
    )
    if not oracle_supported:
        verdict = "OBJECTIVE_OR_INTEGRATION_BOTTLENECK"
    elif not learned_supported:
        verdict = "REPRESENTATION_OR_DATA_BOTTLENECK"
    else:
        verdict = "OFFLINE_SUPPORTED_ACTIVE_PENDING"
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "status": "OFFLINE_COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "elapsed_seconds": time.perf_counter() - started,
        "device": device,
        "conditions": summaries,
        "paired_against_v4_15": comparisons,
        "learned_vs_action_only": learned_vs_action_only,
        "learned_vs_without_relations": learned_vs_without_relations,
        "oracle_gain_capture": oracle_capture,
        "nonnegative_transfer_games": nonnegative_games,
        "objective_and_integration_supported": oracle_supported,
        "learned_trajectory_value_supported": learned_supported,
        "verdict": verdict,
        "all_offline_conditions_executed": (
            set(decisions) == set(conditions)
            and all(len(rows) == 768 for rows in decisions.values())
        ),
        "holdout_opened": False,
        "authority_promoted": False,
        "artifacts": {
            "predictions": {
                "bytes": predictions_path.stat().st_size,
                "sha256": _file_sha256(predictions_path),
            },
            "decisions": {
                "bytes": decisions_path.stat().st_size,
                "sha256": _file_sha256(decisions_path),
            },
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def _run_goal_critic_controller(
    *,
    game_id: str,
    seed: int,
    action_budget: int,
    maximum_resets: int,
    policy_model: Any,
    policy_parameters: Mapping[str, Any],
    critic_model: Any,
    temporal_model: Any,
    temporal_parameters: Mapping[str, Any],
    temporal_ebm: Any,
    sequence_table: Mapping[tuple[str, ...], float],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from theory.m1.polymorphic_a25_adapter import _step_env_action
    from theory.m2.m3_execution_smoke import _reset_env
    from theory.non_ar25_active_micro_run import _env_dir
    from theory.real_env_option_adapter import snapshot_frame
    from theory.sage12.bound_mechanic_pilot import _legal_actions
    from theory.unified_cognition_ab_benchmark import (
        _is_terminal,
        _make_real_env,
    )

    controller = "v4_18_goal_critic"
    run_id = f"{game_id}:{seed}:{controller}"
    environment = _make_real_env(game_id, _env_dir())
    frame = _reset_env(environment)
    resets = 1
    actions_executed = 0
    episode_steps = 0
    levels = 0
    wins = 0
    game_overs = 0
    illegal_proposals = 0
    policy_belief = PolicyBelief()
    temporal_belief = TemporalBeliefState()
    decision_latencies = []
    execution_latencies = []
    candidate_counts = []
    traces = []
    stop_reason = "action_budget"
    while actions_executed < action_budget:
        before = snapshot_frame(frame)
        if _is_terminal(before.game_state):
            if resets >= maximum_resets:
                stop_reason = "maximum_resets"
                break
            frame = _reset_env(environment)
            resets += 1
            episode_steps = 0
            policy_belief = PolicyBelief()
            temporal_belief = TemporalBeliefState()
            continue
        legal = tuple(_legal_actions(environment))
        if not legal:
            stop_reason = "no_legal_actions"
            break
        candidate_counts.append(len(legal))
        decision_started = time.perf_counter()
        available = tuple(
            sorted({str(getattr(action, "name", "")).upper() for action in legal})
        )
        policy_graphs = [
            _live_candidate_graph(
                game_id=game_id,
                policy_seed=seed,
                reset_index=resets - 1,
                step_index=actions_executed,
                frame=frame,
                legal=legal,
                action=action,
            )
            for action in legal
        ]
        policy = _score_candidate_graphs(
            policy_model,
            policy_graphs,
            parameters=policy_parameters,
            device=device,
            belief=policy_belief,
        )
        value_scores = _critic_scores(
            critic_model,
            policy_graphs,
            device=device,
        )
        action_plans = [
            _candidate_action_plan(
                str(getattr(action, "name", "")).upper(),
                available,
                sequence_table,
            )
            for action in legal
        ]
        graph_plans = [
            tuple(
                graph if offset == 0 else _graph_for_action(graph, action_name)
                for offset, action_name in enumerate(action_plan)
            )
            for graph, action_plan in zip(
                policy_graphs,
                action_plans,
                strict=True,
            )
        ]
        temporal_predictions = _predict_candidate_rollouts(
            temporal_model,
            graph_plans,
            parameters=temporal_parameters,
            device=device,
            initial_belief=temporal_belief,
        )
        energies = np.asarray(
            [
                temporal_ebm.energies((_prediction_features(prediction),))[0]
                for prediction in temporal_predictions
            ],
            dtype=np.float64,
        )
        scores = _compose_scores(
            policy["learned_scores"],
            value_scores,
            energies,
        )
        maximum = float(np.max(scores))
        tied = [
            index
            for index, value in enumerate(scores)
            if abs(float(value) - maximum) <= 1e-12
        ]
        selected_index = min(
            tied,
            key=lambda index: hashlib.sha256(
                (
                    f"{run_id}:{actions_executed}:"
                    f"{_live_action_signature(legal[index])}"
                ).encode()
            ).hexdigest(),
        )
        decision_latencies.append(time.perf_counter() - decision_started)
        selected = legal[selected_index]
        execution_started = time.perf_counter()
        try:
            next_frame = _step_env_action(environment, selected)
        except Exception as exc:  # noqa: BLE001 - external game boundary.
            illegal_proposals += 1
            traces.append(
                {
                    "format_version": FORMAT_VERSION,
                    "run_id": run_id,
                    "action_index": actions_executed,
                    "execution_error": f"{type(exc).__name__}:{exc}",
                }
            )
            stop_reason = "execution_error"
            break
        execution_latencies.append(time.perf_counter() - execution_started)
        after = snapshot_frame(next_frame)
        executed_trace = build_action_target_trace(
            game_id=game_id,
            source_split="source_validation",
            policy_seed=seed,
            reset_index=resets - 1,
            step_index=actions_executed,
            collection_phase="v4_18_active",
            available_action_names=available,
            selected_action_name=str(getattr(selected, "name", "")).upper(),
            selected_action_data=dict(getattr(selected, "action_args", {}) or {}),
            frame_before=before.grid,
            frame_after=after.grid,
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            levels_completed_before=before.levels_completed,
            levels_completed_after=after.levels_completed,
        )
        observed_effects, _applicable, _productive, _evidence = compile_semantics(
            executed_trace
        )
        policy_belief = _advance_policy_belief(
            policy_model,
            policy_graphs[selected_index],
            observed_effects,
            parameters=policy_parameters,
            device=device,
            belief=policy_belief,
        )
        temporal_belief = temporal_predictions[selected_index][0].next_belief
        level_delta = max(
            0,
            int(after.levels_completed) - int(before.levels_completed),
        )
        is_win = str(after.game_state).upper() == "WIN"
        is_game_over = str(after.game_state).upper() == "GAME_OVER"
        levels += level_delta
        wins += int(is_win)
        game_overs += int(is_game_over)
        traces.append(
            {
                "format_version": FORMAT_VERSION,
                "run_id": run_id,
                "controller": controller,
                "game_id": game_id,
                "seed": seed,
                "reset_index": resets - 1,
                "action_index": actions_executed,
                "episode_step": episode_steps,
                "pre_state_sha256": grid_sha256(before.grid),
                "post_state_sha256": grid_sha256(after.grid),
                "candidate_count": len(legal),
                "selected_action": _live_action_signature(selected),
                "policy_score": float(policy["learned_scores"][selected_index]),
                "trajectory_value": float(value_scores[selected_index]),
                "temporal_energy": float(energies[selected_index]),
                "hybrid_score": float(scores[selected_index]),
                "levels_completed_after": after.levels_completed,
                "game_state_after": after.game_state,
                "decision_seconds": decision_latencies[-1],
                "execution_seconds": execution_latencies[-1],
            }
        )
        actions_executed += 1
        episode_steps += 1
        frame = next_frame
        if _is_terminal(after.game_state):
            if resets >= maximum_resets:
                stop_reason = "maximum_resets"
                break
            frame = _reset_env(environment)
            resets += 1
            episode_steps = 0
            policy_belief = PolicyBelief()
            temporal_belief = TemporalBeliefState()
    return (
        {
            "format_version": FORMAT_VERSION,
            "run_id": run_id,
            "controller": controller,
            "game_id": game_id,
            "seed": seed,
            "action_budget": action_budget,
            "maximum_resets": maximum_resets,
            "actions_executed": actions_executed,
            "resets": resets,
            "levels_completed": levels,
            "wins": wins,
            "game_overs": game_overs,
            "illegal_proposals": illegal_proposals,
            "stop_reason": stop_reason,
            "mean_candidates": float(np.mean(candidate_counts))
            if candidate_counts
            else 0.0,
            "decision_latency_seconds": {
                "mean": float(np.mean(decision_latencies))
                if decision_latencies
                else 0.0,
                "p95": float(np.quantile(decision_latencies, 0.95))
                if decision_latencies
                else 0.0,
            },
            "execution_latency_seconds": {
                "mean": float(np.mean(execution_latencies))
                if execution_latencies
                else 0.0,
            },
        },
        traces,
    )


def run_active_validation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Run nine bounded active trajectories and finalize the V4.18 verdict."""

    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    result = _read_json(destination / "result.json")
    device = _resolved_device(destination, requested_device)
    critic_model, critic_checkpoint = _load_critic(cache, device=device)
    if critic_checkpoint["manifest_checksum"] != manifest["manifest_checksum"]:
        raise ValueError("V4.18 active critic/manifest mismatch")
    v414_cache = cache / "v4_14"
    v415_cache = cache / "v4_15"
    policy_model, policy_checkpoint = _load_policy_checkpoint(
        v415_cache / "demonstration_policy.pt",
        device=device,
    )
    temporal_model, temporal_checkpoint = _load_temporal_checkpoint(
        v414_cache / "temporal_student.pt",
        device=device,
    )
    temporal_ebm = _load_active_ebm(
        v414_cache / "trajectory_ebm.pt",
        device=device,
    )
    temporal_records = load_temporal_records(v414_cache)
    _action_table, sequence_table, _global_value = _action_sequence_tables(
        temporal_records
    )
    reused = [
        {
            **row,
            "source": "reused_v4_17_content_addressed",
        }
        for row in _jsonl_rows(
            _resolve(root, DEFAULT_V417_DIR) / "active_runs.jsonl"
        )
        if row["controller"]
        in {"milestone_policy_temporal_ebm", "v4_17_hybrid"}
    ]
    if len(reused) != 18:
        raise ValueError("V4.18 expected 18 frozen V4.15/V4.17 active runs")
    runs = list(reused)
    traces = []
    started = time.perf_counter()
    for game_id in ACTIVE_VALIDATION_GAMES:
        for seed in ACTIVE_SEEDS:
            run, run_traces = _run_goal_critic_controller(
                game_id=game_id,
                seed=seed,
                action_budget=ACTIVE_ACTION_BUDGET,
                maximum_resets=ACTIVE_MAXIMUM_RESETS,
                policy_model=policy_model,
                policy_parameters=policy_checkpoint["parameters"],
                critic_model=critic_model,
                temporal_model=temporal_model,
                temporal_parameters=temporal_checkpoint["parameters"],
                temporal_ebm=temporal_ebm,
                sequence_table=sequence_table,
                device=device,
            )
            runs.append(run)
            traces.extend(run_traces)
    runs_path = destination / "active_runs.jsonl"
    traces_path = destination / "active_traces.jsonl"
    _write_jsonl(runs_path, runs)
    _write_jsonl(traces_path, traces)
    metrics = _active_metrics(runs)
    baselines = {
        (str(row["controller"]), str(row["game_id"]), int(row["seed"])): row
        for row in reused
    }
    paired = []
    for row in runs:
        if row["controller"] != "v4_18_goal_critic":
            continue
        for baseline_controller in (
            "milestone_policy_temporal_ebm",
            "v4_17_hybrid",
        ):
            baseline = baselines[
                (baseline_controller, str(row["game_id"]), int(row["seed"]))
            ]
            paired.append(
                {
                    "baseline": baseline_controller,
                    "game_id": row["game_id"],
                    "seed": row["seed"],
                    "level_gain": int(row["levels_completed"])
                    - int(baseline["levels_completed"]),
                    "win_gain": int(row["wins"]) - int(baseline["wins"]),
                    "game_over_delta": int(row["game_overs"])
                    - int(baseline["game_overs"]),
                }
            )
    learned_metrics = metrics["v4_18_goal_critic"]
    active_progress = bool(
        int(learned_metrics["levels"]) > 0
        and int(learned_metrics["illegal_proposals"]) == 0
    )
    if not result["objective_and_integration_supported"]:
        verdict = "OBJECTIVE_OR_INTEGRATION_BOTTLENECK"
    elif not result["learned_trajectory_value_supported"]:
        verdict = "REPRESENTATION_OR_DATA_BOTTLENECK"
    elif not active_progress:
        verdict = "PLANNING_OR_EXECUTION_BOTTLENECK"
    else:
        verdict = "GOAL_CONDITIONED_TRAJECTORY_VALUE_SUPPORTED"
    active: dict[str, Any] = {
        "format_version": ACTIVE_VERSION,
        "status": "COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "games": list(ACTIVE_VALIDATION_GAMES),
        "seeds": list(ACTIVE_SEEDS),
        "fresh_runs": 9,
        "reused_runs": 18,
        "total_runs": len(runs),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "paired_comparisons": paired,
        "active_progress": active_progress,
        "holdout_opened": False,
        "authority_promoted": False,
        "artifacts": {
            "runs": {
                "bytes": runs_path.stat().st_size,
                "sha256": _file_sha256(runs_path),
            },
            "traces": {
                "bytes": traces_path.stat().st_size,
                "sha256": _file_sha256(traces_path),
            },
        },
    }
    active["active_checksum"] = _checksum(active)
    active_path = destination / "active_validation.json"
    _write_json(active_path, active)
    result["status"] = "COMPLETE"
    result["verdict"] = verdict
    result["active_validation"] = active
    result["all_conditions_executed"] = True
    result["holdout_opened"] = False
    result["authority_promoted"] = False
    result["artifacts"]["active_validation"] = {
        "bytes": active_path.stat().st_size,
        "sha256": _file_sha256(active_path),
    }
    result.pop("result_checksum", None)
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return active


def _guarded_command(
    command: str,
    output_dir: str | Path,
    operation: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    root = _repo_root()
    with StorageGuard(
        repo_root=root,
        output_dir=output_dir,
        command=command,
    ) as guard:
        return operation(guard.scratch_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "rebuild", "compile", "train", "evaluate", "active"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    def dispatch(scratch: Path) -> dict[str, Any]:
        if args.command == "freeze":
            return freeze_manifest(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
            )
        if args.command == "rebuild":
            return rebuild_required_checkpoints(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                requested_device=args.device,
            )
        if args.command == "compile":
            return compile_trajectory_credit(
                output_dir=args.output_dir,
                scratch_dir=scratch,
            )
        if args.command == "train":
            return train_trajectory_critic(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                requested_device=args.device,
            )
        if args.command == "evaluate":
            return evaluate_offline(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                requested_device=args.device,
            )
        return run_active_validation(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            requested_device=args.device,
        )

    payload = _guarded_command(args.command, args.output_dir, dispatch)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
