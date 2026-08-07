"""SAGE12 V4.19 topological causal invariants and bounded control.

Commands::

    python -m theory.sage12.topological_causal_control_v4_19 freeze
    python -m theory.sage12.topological_causal_control_v4_19 compile
    python -m theory.sage12.topological_causal_control_v4_19 train
    python -m theory.sage12.topological_causal_control_v4_19 evaluate
    python -m theory.sage12.topological_causal_control_v4_19 active

The implementation never persists frames, full scene graphs, embeddings, or
the regenerable V4.16 transition corpora. Every command runs under the shared
strict storage guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from theory.live_transition_loop import build_observation
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
)
from .goal_conditioned_trajectory_value_v4_18 import (
    DEFAULT_CACHE_DIR as DEFAULT_V418_CACHE_DIR,
)
from .goal_conditioned_trajectory_value_v4_18 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V418_DIR,
)
from .goal_conditioned_trajectory_value_v4_18 import (
    _decision_summary,
    _identity_probe,
    _jsonl_rows,
    _macro_f1,
    _repo_root,
    _resolve,
    _selected_index,
)
from .human_temporal_semantics_v4_14 import (
    ACTIVE_VALIDATION_GAMES,
    HUMAN_TRAIN_GAMES,
    TRANSFER_GAMES,
    TemporalBeliefState,
    _action_names,
    _action_sequence_tables,
    _candidate_action_plan,
    _graph_for_action,
    _live_action_signature,
    _live_candidate_graph,
    _load_active_ebm,
    _paired_bootstrap_rows,
    _predict_candidate_rollouts,
    _prediction_features,
    _step_rows,
)
from .human_temporal_semantics_v4_14 import (
    _load_checkpoint as _load_temporal_checkpoint,
)
from .human_temporal_semantics_v4_14 import (
    load_teacher_records as load_temporal_records,
)
from .morpho_topological_v4_16 import (
    _action_data,
    _observed_semantic_outcome,
)
from .mt.graph import build_mt_graph
from .mt.transition import compile_mt_transition
from .semantic_teacher_v4_9 import (
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
from .topological_invariants_v4_19 import (
    FACTOR_NAMES,
    FEATURE_WIDTH,
    compile_topological_transition,
    dense_vector,
    feature_vector,
    forbidden_field_hits,
    permutation_invariant,
    sparse_vector,
)

FORMAT_VERSION = "sage12-topological-causal-control-v4.19"
MANIFEST_VERSION = "sage12-topological-causal-manifest-v4.19"
TEACHER_VERSION = "sage12-topological-credit-record-v4.19"
CHECKPOINT_VERSION = "sage12-topological-predictor-checkpoint-v4.19"
RESULT_VERSION = "sage12-topological-causal-result-v4.19"
ACTIVE_VERSION = "sage12-topological-causal-active-v4.19"

DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage12" / "topological_causal_control_v4_19"
)
DEFAULT_CACHE_DIR = Path(".sage12_cache") / "v4_19"
PROTOCOL_PATH = (
    Path("reports")
    / "SAGE12_TOPOLOGICAL_CAUSAL_INVARIANTS_V4_19_PROTOCOL.md"
)
RESULT_REPORT_PATH = (
    Path("reports")
    / "SAGE12_TOPOLOGICAL_CAUSAL_INVARIANTS_V4_19_RESULT.md"
)
HUMAN_TRACES_DIR = Path("human_traces")

SEED = 5_190
HORIZONS = (8, 16, 32, 64)
GAMMA = 0.97
HIDDEN_WIDTH = 128
LATENT_WIDTH = 64
UNCERTAINTY_COEFFICIENT = 0.25
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
    "factor_loss_weight": 0.65,
    "value_loss_weight": 0.35,
}


def _sources(
    *,
    v415_dir: Path,
    v411_dir: Path,
    v417_dir: Path,
    v418_dir: Path,
) -> tuple[Path, ...]:
    return (
        v415_dir / "frozen_manifest.json",
        v415_dir / "teacher_qa.json",
        v415_dir / "demonstration_choices.jsonl",
        v411_dir / "frozen_manifest.json",
        v411_dir / "teacher_panels.jsonl",
        v417_dir / "result.json",
        v417_dir / "offline_predictions.jsonl",
        v417_dir / "active_runs.jsonl",
        v418_dir / "result.json",
        v418_dir / "offline_predictions.jsonl",
        v418_dir / "active_runs.jsonl",
        Path(__file__).resolve(),
        Path(__file__).with_name("topological_invariants_v4_19.py").resolve(),
        Path(__file__).with_name("artifact_budget.py").resolve(),
        PROTOCOL_PATH.resolve(),
        Path(".gitignore").resolve(),
    )


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir)
    sources = _sources(
        v415_dir=_resolve(root, DEFAULT_V415_DIR),
        v411_dir=_resolve(root, DEFAULT_V411_DIR),
        v417_dir=_resolve(root, DEFAULT_V417_DIR),
        v418_dir=_resolve(root, DEFAULT_V418_DIR),
    )
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(path.as_posix() for path in missing))
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
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
            "limits": BudgetLimits().__dict__,
            "raw_frames_persisted": False,
            "full_graphs_persisted": False,
            "embeddings_persisted": False,
            "v4_16_giant_corpora_regenerated": False,
        },
        "teacher": {
            "expected_records": 5_661,
            "expected_sequences": 41,
            "horizons": list(HORIZONS),
            "gamma": GAMMA,
            "factors": list(FACTOR_NAMES),
            "new_environment_collection": 0,
        },
        "model": {
            "feature_width": FEATURE_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "latent_width": LATENT_WIDTH,
            **MODEL_PARAMETERS,
            "raw_action_identity": False,
            "game_identity": False,
            "palette": False,
            "absolute_coordinates": False,
            "object_identity": False,
        },
        "representation_gates": {
            "minimum_confident_correspondence_fraction": 0.90,
            "maximum_fully_ambiguous_fraction": 0.10,
            "minimum_factor_f1_gain": 0.10,
            "minimum_binding_swap_degradation": 0.05,
            "minimum_relation_degradation": 0.05,
            "minimum_nonnegative_human_games": 5,
            "maximum_identity_increment": 0.10,
            "node_permutation_exact": True,
        },
        "offline": {
            "panels": 768,
            "arms": 2_831,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "value_coefficient": VALUE_COEFFICIENT,
            "temporal_ebm_coefficient": TEMPORAL_EBM_COEFFICIENT,
            "uncertainty_coefficient": UNCERTAINTY_COEFFICIENT,
            "minimum_nonnegative_games": 5,
            "minimum_oracle_gain_capture": 0.30,
            "conditions": [
                "v4_15_policy",
                "v4_17_hybrid",
                "v4_18_learned",
                "action_only",
                "static_invariants",
                "v4_19_learned",
                "v4_19_without_relations",
                "v4_19_binding_swapped",
                "local_topology_oracle",
                "multi_horizon_topology_oracle",
                "exact_oracle",
            ],
        },
        "active": {
            "games": list(ACTIVE_VALIDATION_GAMES),
            "seeds": list(ACTIVE_SEEDS),
            "action_budget": ACTIVE_ACTION_BUDGET,
            "maximum_resets": ACTIVE_MAXIMUM_RESETS,
            "fresh_runs": 9,
            "reused_controllers": [
                "milestone_policy_temporal_ebm",
                "v4_17_hybrid",
                "v4_18_goal_critic",
            ],
        },
        "authority": {
            "holdout_opened": False,
            "controller_authority_promoted": False,
        },
        "result_observed_at_freeze": False,
        "pre_fit_amendments": [
            {
                "id": "persistent_contact_relations",
                "trigger": (
                    "first_compact_corpus_qa_contact_added_on_every_transition"
                ),
                "fit_observed_before_amendment": False,
                "change": (
                    "contact deltas restricted to confident one-to-one "
                    "persistent object correspondences"
                ),
                "thresholds_changed": False,
            }
        ],
    }
    manifest["manifest_checksum"] = _checksum(manifest)
    _write_json(destination / "frozen_manifest.json", manifest)
    return manifest


def load_manifest(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = _repo_root()
    payload = _read_json(_resolve(root, output_dir) / "frozen_manifest.json")
    if payload.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V4.19 manifest")
    expected = str(payload["manifest_checksum"])
    clean = dict(payload)
    clean.pop("manifest_checksum")
    if _checksum(clean) != expected:
        raise ValueError("V4.19 manifest checksum mismatch")
    return payload


def _verify_sources(manifest: Mapping[str, Any]) -> None:
    root = _repo_root()
    for relative, expected in manifest["source_fingerprints"].items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        if _file_fingerprint(path) != expected:
            raise ValueError(f"V4.19 frozen source drift: {relative}")


def _future_values(
    local_values: Sequence[float],
    *,
    index: int,
) -> list[float]:
    output = []
    for horizon in HORIZONS:
        value = sum(
            (GAMMA**offset) * float(item)
            for offset, item in enumerate(
                local_values[index : index + horizon]
            )
        )
        output.append(float(np.clip(value, -1.0, 1.0)))
    return output


def _student_payload(graph: Any) -> dict[str, Any]:
    return {
        "full": sparse_vector(feature_vector(graph)),
        "without_relations": sparse_vector(
            feature_vector(graph, remove_relations=True)
        ),
        "binding_swapped": sparse_vector(
            feature_vector(graph, swap_binding=True)
        ),
        "static_invariants": sparse_vector(
            feature_vector(graph, static_only=True)
        ),
    }


def _compile_segment(
    *,
    game: str,
    episode_id: str,
    segment_index: int,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    compiled = []
    previous_level = 0
    previous_state = "NOT_FINISHED"
    for index, raw in enumerate(rows):
        action = str(raw["action"]).strip().upper()
        data = _action_data(raw.get("action_args"))
        current_level = int(raw.get("levels_completed_after", previous_level))
        state_after = str(raw.get("game_state_after", "NOT_FINISHED"))
        before = raw["frame_before"]
        after = raw["frame_after"]
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
            available_actions=_action_names(raw.get("available_actions") or ()),
            game_state_before=previous_state,
            game_state_after=state_after,
            levels_before=previous_level,
            levels_after=current_level,
            step=int(raw.get("step", index)),
        )
        terminal = bool(
            current_level > previous_level or state_after.upper() == "WIN"
        )
        transition = compile_mt_transition(
            before,
            action,
            after,
            action_data=data,
            source_game_id=game,
            player_position_before=player_before,
            player_position_after=player_after,
            productive=productive,
            risk=risk,
            audit={},
        )
        topological = compile_topological_transition(
            transition,
            terminal_progress=terminal,
            risk=risk,
        )
        payload = _student_payload(transition.graph_before)
        if forbidden_field_hits(payload):
            raise ValueError("V4.19 student payload leaked an identity field")
        compiled.append(
            {
                "format_version": TEACHER_VERSION,
                "student_view": payload,
                "teacher": {
                    "factors": [
                        float(topological.factors[name])
                        for name in FACTOR_NAMES
                    ],
                    "local_value": topological.local_value,
                    "future_values": [],
                },
                "audit": {
                    "game_id": game,
                    "sequence_key": f"{game}:{episode_id}:segment{segment_index}",
                    "sequence_index": index,
                    "action_family": transition.graph_before.action_family,
                    "pre_state_sha256": grid_sha256(before),
                    "post_state_sha256": grid_sha256(after),
                    "correspondence": dict(topological.correspondence),
                    "permutation_invariant": permutation_invariant(
                        transition.graph_before
                    ),
                },
            }
        )
        previous_level = current_level
        previous_state = state_after
    local_values = [row["teacher"]["local_value"] for row in compiled]
    for index, row in enumerate(compiled):
        row["teacher"]["future_values"] = _future_values(
            local_values,
            index=index,
        )
    return compiled


def compile_topological_credit(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    traces_dir: str | Path = HUMAN_TRACES_DIR,
) -> dict[str, Any]:
    root = _repo_root()
    destination = _resolve(root, output_dir)
    manifest = load_manifest(destination)
    _verify_sources(manifest)
    grouped = _step_rows(_resolve(root, traces_dir))
    output_path = destination / "topological_credit.jsonl"
    rows_written = 0
    sequences = 0
    confidence_sum = 0.0
    confidence_count = 0
    fully_ambiguous = 0
    permutation_failures = 0
    by_game: Counter[str] = Counter()
    factor_counts = np.zeros(len(FACTOR_NAMES), dtype=np.int64)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for (game, episode_id), raw_rows in sorted(grouped.items()):
            if game not in HUMAN_TRAIN_GAMES:
                continue
            segments: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            for raw in raw_rows:
                if str(raw.get("action", "")).strip().upper() == "RESET":
                    if current:
                        segments.append(current)
                        current = []
                    continue
                current.append(dict(raw))
            if current:
                segments.append(current)
            for segment_index, segment in enumerate(segments):
                if not segment:
                    continue
                compiled = _compile_segment(
                    game=game,
                    episode_id=episode_id,
                    segment_index=segment_index,
                    rows=segment,
                )
                sequences += 1
                for row in compiled:
                    encoded = json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    handle.write(encoded)
                    handle.write("\n")
                    rows_written += 1
                    by_game[game] += 1
                    factor_counts += np.asarray(
                        row["teacher"]["factors"],
                        dtype=np.int64,
                    )
                    quality = row["audit"]["correspondence"]
                    structural = int(quality["structural_correspondences"])
                    confidence_sum += (
                        float(quality["confident_fraction"]) * structural
                    )
                    confidence_count += structural
                    fully_ambiguous += int(quality["fully_ambiguous"])
                    permutation_failures += int(
                        not row["audit"]["permutation_invariant"]
                    )
    expected = int(manifest["teacher"]["expected_records"])
    expected_sequences = int(manifest["teacher"]["expected_sequences"])
    if rows_written != expected or sequences != expected_sequences:
        raise ValueError(
            "V4.19 human corpus drift: "
            f"{rows_written}/{sequences} != {expected}/{expected_sequences}"
        )
    qa: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "records": rows_written,
        "sequences": sequences,
        "by_game": dict(sorted(by_game.items())),
        "factor_counts": {
            name: int(factor_counts[index])
            for index, name in enumerate(FACTOR_NAMES)
        },
        "correspondence": {
            "confident_fraction": confidence_sum / max(confidence_count, 1),
            "fully_ambiguous_fraction": fully_ambiguous / rows_written,
            "structural_correspondences": confidence_count,
        },
        "permutation_failures": permutation_failures,
        "forbidden_student_field_hits": [],
        "raw_frames_persisted": False,
        "full_graphs_persisted": False,
        "artifact": {
            "bytes": output_path.stat().st_size,
            "sha256": _file_sha256(output_path),
        },
        "holdout_opened": False,
    }
    qa["corpus_ready"] = bool(
        qa["correspondence"]["confident_fraction"] >= 0.90
        and qa["correspondence"]["fully_ambiguous_fraction"] < 0.10
        and permutation_failures == 0
    )
    qa["qa_checksum"] = _checksum(qa)
    _write_json(destination / "teacher_qa.json", qa)
    return qa


def _credit_rows(path: Path) -> Iterator[dict[str, Any]]:
    for row in _jsonl_rows(path):
        if row.get("format_version") != TEACHER_VERSION:
            raise ValueError("unsupported V4.19 teacher record")
        yield row


def _arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(
            [
                dense_vector(row["student_view"][key])
                for row in rows
            ],
            dtype=np.float32,
        )
        for key in (
            "full",
            "without_relations",
            "binding_swapped",
            "static_invariants",
        )
    } | {
        "factors": np.asarray(
            [row["teacher"]["factors"] for row in rows],
            dtype=np.float32,
        ),
        "values": np.asarray(
            [row["teacher"]["future_values"] for row in rows],
            dtype=np.float32,
        ),
    }


def _model_type() -> type[Any]:
    import torch

    class TopologicalPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(FEATURE_WIDTH, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, LATENT_WIDTH),
                torch.nn.ReLU(),
            )
            self.factor_head = torch.nn.Linear(
                LATENT_WIDTH,
                len(FACTOR_NAMES),
            )
            self.value_head = torch.nn.Linear(
                LATENT_WIDTH,
                len(HORIZONS),
            )

        def forward(self, features: Any) -> tuple[Any, Any, Any]:
            latent = self.encoder(features)
            return (
                self.factor_head(latent),
                torch.tanh(self.value_head(latent)),
                latent,
            )

    return TopologicalPredictor


def _fit_model(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: str,
    epochs: int,
    seed: int,
) -> tuple[Any, list[dict[str, float]]]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _model_type()().to(device)
    features = torch.as_tensor(arrays["full"], device=device)
    factors = torch.as_tensor(arrays["factors"], device=device)
    values = torch.as_tensor(arrays["values"], device=device)
    positive = np.sum(arrays["factors"][indices], axis=0)
    negative = len(indices) - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1.0), 1.0, 20.0)
    weights = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(MODEL_PARAMETERS["learning_rate"]),
        weight_decay=float(MODEL_PARAMETERS["weight_decay"]),
    )
    generator = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = generator.permutation(indices)
        losses = []
        model.train()
        for start in range(
            0,
            len(order),
            int(MODEL_PARAMETERS["batch_size"]),
        ):
            batch_indices = torch.as_tensor(
                order[start : start + int(MODEL_PARAMETERS["batch_size"])],
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_values, _latent = model(
                features[batch_indices]
            )
            factor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                factors[batch_indices],
                pos_weight=weights,
            )
            value_loss = torch.nn.functional.smooth_l1_loss(
                predicted_values,
                values[batch_indices],
            )
            loss = (
                float(MODEL_PARAMETERS["factor_loss_weight"]) * factor_loss
                + float(MODEL_PARAMETERS["value_loss_weight"]) * value_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(
                (
                    float(loss.detach().cpu()),
                    float(factor_loss.detach().cpu()),
                    float(value_loss.detach().cpu()),
                )
            )
        mean = np.mean(losses, axis=0)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(mean[0]),
                "factor_loss": float(mean[1]),
                "value_loss": float(mean[2]),
            }
        )
    return model, history


def _predict(
    model: Any,
    features: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    with torch.no_grad():
        logits, values, latent = model(
            torch.as_tensor(features, dtype=torch.float32, device=device)
        )
        probabilities = torch.sigmoid(logits)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-7))
            + (1.0 - probabilities)
            * torch.log((1.0 - probabilities).clamp_min(1e-7))
        ).mean(dim=1)
    return (
        probabilities.cpu().numpy(),
        values.cpu().numpy(),
        entropy.cpu().numpy(),
        latent.cpu().numpy(),
    )


def _device(requested: str) -> tuple[str, dict[str, Any]]:
    import torch

    available = bool(torch.cuda.is_available())
    if requested != "auto":
        if requested.startswith("cuda") and not available:
            raise RuntimeError("CUDA requested for V4.19 but unavailable")
        return requested, {
            "requested": requested,
            "cuda_available": available,
            "selected": requested,
        }
    selected = "cuda:0" if available else "cpu"
    return selected, {
        "requested": requested,
        "cuda_available": available,
        "selected": selected,
        "reason": (
            "cuda_available_for_compact_predictor"
            if available
            else "cuda_unavailable_compact_cpu_fit"
        ),
    }


def _action_baseline(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    grouped_factors: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_values: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in indices:
        row = rows[int(index)]
        family = str(row["audit"]["action_family"])
        grouped_factors[family].append(
            np.asarray(row["teacher"]["factors"], dtype=np.float32)
        )
        grouped_values[family].append(
            np.asarray(row["teacher"]["future_values"], dtype=np.float32)
        )
    all_factors = np.asarray(
        [rows[int(index)]["teacher"]["factors"] for index in indices],
        dtype=np.float32,
    )
    all_values = np.asarray(
        [rows[int(index)]["teacher"]["future_values"] for index in indices],
        dtype=np.float32,
    )
    return {
        "global_factors": np.mean(all_factors, axis=0).tolist(),
        "global_values": np.mean(all_values, axis=0).tolist(),
        "by_action_family": {
            family: {
                "factors": np.mean(grouped_factors[family], axis=0).tolist(),
                "values": np.mean(grouped_values[family], axis=0).tolist(),
            }
            for family in sorted(grouped_factors)
        },
    }


def _baseline_predictions(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    baseline: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    factors = []
    values = []
    by_action = baseline["by_action_family"]
    for index in indices:
        family = str(rows[int(index)]["audit"]["action_family"])
        item = by_action.get(family)
        factors.append(
            item["factors"] if item else baseline["global_factors"]
        )
        values.append(item["values"] if item else baseline["global_values"])
    return (
        np.asarray(factors, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    )


def train_predictor(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    import torch

    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(destination)
    _verify_sources(manifest)
    rows = list(_credit_rows(destination / "topological_credit.jsonl"))
    arrays = _arrays(rows)
    device, benchmark = _device(requested_device)
    predictions = np.zeros_like(arrays["factors"])
    removed_predictions = np.zeros_like(arrays["factors"])
    swapped_predictions = np.zeros_like(arrays["factors"])
    static_predictions = np.zeros_like(arrays["factors"])
    value_predictions = np.zeros_like(arrays["values"])
    baseline_factor_predictions = np.zeros_like(arrays["factors"])
    baseline_value_predictions = np.zeros_like(arrays["values"])
    fold_rows = []
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
        model, history = _fit_model(
            arrays,
            train_indices,
            device=device,
            epochs=int(MODEL_PARAMETERS["fold_epochs"]),
            seed=SEED + fold_index,
        )
        predicted, values, _uncertainty, _latent = _predict(
            model,
            arrays["full"][test_indices],
            device=device,
        )
        removed, _removed_values, _removed_uncertainty, _ = _predict(
            model,
            arrays["without_relations"][test_indices],
            device=device,
        )
        swapped, _swapped_values, _swapped_uncertainty, _ = _predict(
            model,
            arrays["binding_swapped"][test_indices],
            device=device,
        )
        static, _static_values, _static_uncertainty, _ = _predict(
            model,
            arrays["static_invariants"][test_indices],
            device=device,
        )
        baseline = _action_baseline(rows, train_indices)
        baseline_factors, baseline_values = _baseline_predictions(
            rows,
            test_indices,
            baseline,
        )
        predictions[test_indices] = predicted
        removed_predictions[test_indices] = removed
        swapped_predictions[test_indices] = swapped
        static_predictions[test_indices] = static
        value_predictions[test_indices] = values
        baseline_factor_predictions[test_indices] = baseline_factors
        baseline_value_predictions[test_indices] = baseline_values
        full_f1 = _macro_f1(arrays["factors"][test_indices], predicted)
        baseline_f1 = _macro_f1(
            arrays["factors"][test_indices],
            baseline_factors,
        )
        fold_rows.append(
            {
                "held_game": held_game,
                "records": len(test_indices),
                "factor_macro_f1": full_f1,
                "action_only_macro_f1": baseline_f1,
                "factor_gain": full_f1 - baseline_f1,
                "without_relations_macro_f1": _macro_f1(
                    arrays["factors"][test_indices],
                    removed,
                ),
                "binding_swapped_macro_f1": _macro_f1(
                    arrays["factors"][test_indices],
                    swapped,
                ),
                "static_invariants_macro_f1": _macro_f1(
                    arrays["factors"][test_indices],
                    static,
                ),
                "value_mae": float(
                    np.mean(
                        np.abs(values - arrays["values"][test_indices])
                    )
                ),
                "action_only_value_mae": float(
                    np.mean(
                        np.abs(
                            baseline_values - arrays["values"][test_indices]
                        )
                    )
                ),
                "final_loss": history[-1],
            }
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    all_indices = np.arange(len(rows), dtype=np.int64)
    final_model, history = _fit_model(
        arrays,
        all_indices,
        device=device,
        epochs=int(MODEL_PARAMETERS["epochs"]),
        seed=SEED + 100,
    )
    _factors, _values, _uncertainty, embeddings = _predict(
        final_model,
        arrays["full"],
        device=device,
    )
    checkpoint_path = cache / "topological_predictor.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "state_dict": final_model.state_dict(),
            "parameters": {
                "feature_width": FEATURE_WIDTH,
                "hidden_width": HIDDEN_WIDTH,
                "latent_width": LATENT_WIDTH,
                "factors": list(FACTOR_NAMES),
                "horizons": list(HORIZONS),
            },
        },
        checkpoint_path,
    )
    action_baseline = _action_baseline(rows, all_indices)
    full_f1 = _macro_f1(arrays["factors"], predictions)
    action_f1 = _macro_f1(arrays["factors"], baseline_factor_predictions)
    removed_f1 = _macro_f1(arrays["factors"], removed_predictions)
    swapped_f1 = _macro_f1(arrays["factors"], swapped_predictions)
    static_f1 = _macro_f1(arrays["factors"], static_predictions)
    identity = _identity_probe(embeddings, rows)
    qa = _read_json(destination / "teacher_qa.json")
    nonnegative_games = sum(row["factor_gain"] >= 0.0 for row in fold_rows)
    representation_supported = bool(
        qa["correspondence"]["confident_fraction"] >= 0.90
        and qa["correspondence"]["fully_ambiguous_fraction"] < 0.10
        and qa["permutation_failures"] == 0
        and full_f1 - action_f1 >= 0.10
        and full_f1 - swapped_f1 >= 0.05
        and full_f1 - removed_f1 >= 0.05
        and nonnegative_games >= 5
        and identity["increment"] <= 0.10
    )
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "selected_device": device,
        "device_benchmark": benchmark,
        "training_seconds": time.perf_counter() - started,
        "leave_one_game_out": {
            "factor_macro_f1": full_f1,
            "action_only_macro_f1": action_f1,
            "factor_gain": full_f1 - action_f1,
            "without_relations_macro_f1": removed_f1,
            "relation_degradation": full_f1 - removed_f1,
            "binding_swapped_macro_f1": swapped_f1,
            "binding_swap_degradation": full_f1 - swapped_f1,
            "static_invariants_macro_f1": static_f1,
            "value_mae": float(
                np.mean(np.abs(value_predictions - arrays["values"]))
            ),
            "action_only_value_mae": float(
                np.mean(
                    np.abs(
                        baseline_value_predictions - arrays["values"]
                    )
                )
            ),
            "nonnegative_games": nonnegative_games,
            "per_game": fold_rows,
        },
        "game_identity_probe": identity,
        "action_only_baseline": action_baseline,
        "representation_supported": representation_supported,
        "checkpoint": {
            "path": checkpoint_path.relative_to(root).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _file_sha256(checkpoint_path),
        },
        "teacher_sha256": _file_sha256(
            destination / "topological_credit.jsonl"
        ),
        "final_loss": history[-1],
    }
    metadata["training_checksum"] = _checksum(metadata)
    _write_json(destination / "checkpoint_metadata.json", metadata)
    _write_jsonl(destination / "folds.jsonl", fold_rows)
    return metadata


def _load_predictor(
    cache_dir: Path,
    *,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    checkpoint = torch.load(
        cache_dir / "topological_predictor.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported V4.19 predictor checkpoint")
    model = _model_type()().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _trace_topology(
    trace: Mapping[str, Any],
    *,
    game: str,
) -> tuple[Any, Any]:
    before = trace["frame_before"]
    after = trace["frame_after"]
    action = str(trace["selected_action_name"]).strip().upper()
    data = _action_data(trace.get("selected_action_data"))
    levels_before = int(trace.get("levels_completed_before", 0))
    levels_after = int(trace.get("levels_completed_after", levels_before))
    state_before = str(trace.get("game_state_before", "NOT_FINISHED"))
    state_after = str(trace.get("game_state_after", "NOT_FINISHED"))
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
        available_actions=tuple(trace.get("available_action_names", ())),
        game_state_before=state_before,
        game_state_after=state_after,
        levels_before=levels_before,
        levels_after=levels_after,
        step=int(trace.get("step_index", 0)),
    )
    transition = compile_mt_transition(
        before,
        action,
        after,
        action_data=data,
        source_game_id=game,
        player_position_before=player_before,
        player_position_after=player_after,
        productive=productive,
        risk=risk,
        audit={},
    )
    topological = compile_topological_transition(
        transition,
        terminal_progress=bool(
            levels_after > levels_before or state_after.upper() == "WIN"
        ),
        risk=risk,
    )
    return transition, topological


def _continuation_topology_value(
    arm: Mapping[str, Any],
    *,
    game: str,
    immediate_value: float,
) -> float:
    values = []
    for continuation in arm.get("continuations", ()):
        total = float(immediate_value)
        for offset, trace in enumerate(continuation, start=1):
            _transition, compiled = _trace_topology(trace, game=game)
            total += (GAMMA**offset) * compiled.local_value
        values.append(float(np.clip(total, -1.0, 1.0)))
    return float(np.mean(values)) if values else float(immediate_value)


def _predict_graphs(
    model: Any,
    graphs: Sequence[Any],
    *,
    device: str,
    remove_relations: bool = False,
    swap_binding: bool = False,
    static_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            feature_vector(
                graph,
                remove_relations=remove_relations,
                swap_binding=swap_binding,
                static_only=static_only,
            )
            for graph in graphs
        ],
        dtype=np.float32,
    )
    factors, values, uncertainty, _latent = _predict(
        model,
        features,
        device=device,
    )
    return factors, values, uncertainty


def _action_only_graph_scores(
    graphs: Sequence[Any],
    baseline: Mapping[str, Any],
) -> np.ndarray:
    by_action = baseline["by_action_family"]
    global_values = baseline["global_values"]
    return np.asarray(
        [
            (
                by_action.get(graph.action_family, {})
                .get("values", global_values)[HORIZONS.index(32)]
            )
            for graph in graphs
        ],
        dtype=np.float64,
    )


def _compose(
    policy: Sequence[float],
    value: Sequence[float],
    energy: Sequence[float],
    uncertainty: Sequence[float] | None = None,
) -> np.ndarray:
    score = (
        _zscore(policy)
        + VALUE_COEFFICIENT * _zscore(value)
        - TEMPORAL_EBM_COEFFICIENT * _zscore(energy)
    )
    if uncertainty is not None:
        score -= UNCERTAINTY_COEFFICIENT * _zscore(uncertainty)
    return score


def _source_panels(root: Path) -> Iterator[dict[str, Any]]:
    directory = _resolve(root, DEFAULT_V411_DIR) / "source_train_shards"
    for game in TRANSFER_GAMES:
        yield from _jsonl_rows(directory / f"{game}.jsonl")


def evaluate_offline(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    requested_device: str = "auto",
) -> dict[str, Any]:
    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir)
    manifest = load_manifest(destination)
    _verify_sources(manifest)
    metadata = _read_json(destination / "checkpoint_metadata.json")
    device = (
        str(metadata["selected_device"])
        if requested_device == "auto"
        else requested_device
    )
    model, checkpoint = _load_predictor(cache, device=device)
    if checkpoint["manifest_checksum"] != manifest["manifest_checksum"]:
        raise ValueError("V4.19 predictor/manifest mismatch")
    raw_panels = {
        str(row["panel_id"]): row for row in _source_panels(root)
    }
    v417_rows = list(
        _jsonl_rows(
            _resolve(root, DEFAULT_V417_DIR) / "offline_predictions.jsonl"
        )
    )
    v418_rows = {
        str(row["panel_id"]): row
        for row in _jsonl_rows(
            _resolve(root, DEFAULT_V418_DIR) / "offline_predictions.jsonl"
        )
    }
    conditions = list(manifest["offline"]["conditions"])
    decisions: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in conditions
    }
    prediction_rows = []
    started = time.perf_counter()
    for component in v417_rows:
        panel_id = str(component["panel_id"])
        game = str(component["game_id"])
        panel = raw_panels[panel_id]
        raw_arms = sorted(
            panel["arms"],
            key=lambda item: int(item["arm_index"]),
        )
        component_arms = sorted(
            component["arms"],
            key=lambda item: int(item["arm_index"]),
        )
        v418_arms = sorted(
            v418_rows[panel_id]["arms"],
            key=lambda item: int(item["arm_index"]),
        )
        arm_indices = [int(arm["arm_index"]) for arm in component_arms]
        if arm_indices != [int(arm["arm_index"]) for arm in raw_arms]:
            raise ValueError(f"V4.19 raw arm alignment drift: {panel_id}")
        if arm_indices != [int(arm["arm_index"]) for arm in v418_arms]:
            raise ValueError(f"V4.19 V4.18 arm alignment drift: {panel_id}")
        graphs = []
        local_oracle = []
        horizon_oracle = []
        for raw_arm in raw_arms:
            transition, topological = _trace_topology(
                raw_arm["immediate_trace"],
                game=game,
            )
            graphs.append(transition.graph_before)
            local_oracle.append(topological.local_value)
            horizon_oracle.append(
                _continuation_topology_value(
                    raw_arm,
                    game=game,
                    immediate_value=topological.local_value,
                )
            )
        _factors, learned_values, uncertainty = _predict_graphs(
            model,
            graphs,
            device=device,
        )
        _removed_factors, removed_values, removed_uncertainty = _predict_graphs(
            model,
            graphs,
            device=device,
            remove_relations=True,
        )
        _swapped_factors, swapped_values, swapped_uncertainty = _predict_graphs(
            model,
            graphs,
            device=device,
            swap_binding=True,
        )
        _static_factors, static_values, static_uncertainty = _predict_graphs(
            model,
            graphs,
            device=device,
            static_only=True,
        )
        learned = learned_values[:, HORIZONS.index(32)]
        removed = removed_values[:, HORIZONS.index(32)]
        swapped = swapped_values[:, HORIZONS.index(32)]
        static = static_values[:, HORIZONS.index(32)]
        action_only = _action_only_graph_scores(
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
        v418 = np.asarray(
            [float(arm["learned_value"]) for arm in v418_arms],
            dtype=np.float64,
        )
        utility = np.asarray(
            [float(arm["utility"]) for arm in component_arms],
            dtype=np.float64,
        )
        completion = [
            bool(arm["completion"]) for arm in component_arms
        ]
        local = np.asarray(local_oracle, dtype=np.float64)
        horizon = np.asarray(horizon_oracle, dtype=np.float64)
        zero_uncertainty = np.zeros(len(graphs), dtype=np.float64)
        score_by_condition = {
            "v4_15_policy": policy,
            "v4_17_hybrid": v417,
            "v4_18_learned": _compose(policy, v418, energy),
            "action_only": _compose(policy, action_only, energy),
            "static_invariants": _compose(
                policy,
                static,
                energy,
                static_uncertainty,
            ),
            "v4_19_learned": _compose(
                policy,
                learned,
                energy,
                uncertainty,
            ),
            "v4_19_without_relations": _compose(
                policy,
                removed,
                energy,
                removed_uncertainty,
            ),
            "v4_19_binding_swapped": _compose(
                policy,
                swapped,
                energy,
                swapped_uncertainty,
            ),
            "local_topology_oracle": _compose(
                policy,
                local,
                energy,
                zero_uncertainty,
            ),
            "multi_horizon_topology_oracle": _compose(
                policy,
                horizon,
                energy,
                zero_uncertainty,
            ),
            "exact_oracle": utility,
        }
        exact_index = _selected_index(utility, arm_indices)
        maximum_utility = float(utility[exact_index])
        for condition in conditions:
            selected = _selected_index(
                score_by_condition[condition],
                arm_indices,
            )
            decisions[condition].append(
                {
                    "format_version": FORMAT_VERSION,
                    "condition": condition,
                    "panel_id": panel_id,
                    "game_id": game,
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
                "game_id": game,
                "arms": [
                    {
                        "arm_index": arm_indices[index],
                        "learned_value": float(learned[index]),
                        "without_relations_value": float(removed[index]),
                        "binding_swapped_value": float(swapped[index]),
                        "static_value": float(static[index]),
                        "uncertainty": float(uncertainty[index]),
                        "local_topology_oracle": float(local[index]),
                        "multi_horizon_topology_oracle": float(horizon[index]),
                        "utility": float(utility[index]),
                        "completion": completion[index],
                    }
                    for index in range(len(arm_indices))
                ],
            }
        )
    all_decisions = [
        row
        for condition in conditions
        for row in decisions[condition]
    ]
    predictions_path = destination / "offline_predictions.jsonl"
    decisions_path = destination / "offline_decisions.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(decisions_path, all_decisions)
    summaries = {
        condition: _decision_summary(decisions[condition])
        for condition in conditions
    }
    baseline = decisions["v4_18_learned"]
    comparisons = {
        condition: _paired_bootstrap_rows(
            decisions[condition],
            baseline,
            samples=BOOTSTRAP_SAMPLES,
            seed=SEED + index,
        )
        for index, condition in enumerate(conditions)
        if condition != "v4_18_learned"
    }
    learned_vs_action = _paired_bootstrap_rows(
        decisions["v4_19_learned"],
        decisions["action_only"],
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 90,
    )
    learned_vs_removed = _paired_bootstrap_rows(
        decisions["v4_19_learned"],
        decisions["v4_19_without_relations"],
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 91,
    )
    learned_vs_swapped = _paired_bootstrap_rows(
        decisions["v4_19_learned"],
        decisions["v4_19_binding_swapped"],
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED + 92,
    )
    learned_by_game = summaries["v4_19_learned"]["per_game"]
    baseline_by_game = summaries["v4_18_learned"]["per_game"]
    nonnegative_games = sum(
        learned_by_game[game]["mean_utility"]
        >= baseline_by_game[game]["mean_utility"]
        for game in TRANSFER_GAMES
    )
    oracle_comparison = comparisons["multi_horizon_topology_oracle"]
    oracle_supported = bool(
        oracle_comparison["ci_low"] > 0.0
        and summaries["multi_horizon_topology_oracle"]["completion_arms"]
        >= summaries["v4_18_learned"]["completion_arms"]
    )
    oracle_gain = float(oracle_comparison["mean_gain"])
    learned_gain = float(comparisons["v4_19_learned"]["mean_gain"])
    oracle_capture = (
        learned_gain / oracle_gain
        if oracle_gain > 0.0
        else float("-inf")
    )
    representation_supported = bool(metadata["representation_supported"])
    value_supported = bool(
        comparisons["v4_19_learned"]["ci_low"] > 0.0
        and learned_vs_action["ci_low"] > 0.0
        and learned_vs_removed["ci_low"] > 0.0
        and learned_vs_swapped["ci_low"] > 0.0
        and nonnegative_games >= 5
        and oracle_capture >= 0.30
        and summaries["v4_19_learned"]["completion_arms"] >= 1
        and summaries["v4_19_learned"]["completion_arms"]
        >= math.ceil(
            summaries["multi_horizon_topology_oracle"]["completion_arms"]
            / 2
        )
    )
    if not oracle_supported:
        verdict = "TOPOLOGICAL_OBJECTIVE_BOTTLENECK"
    elif not representation_supported:
        verdict = "CORRESPONDENCE_OR_REPRESENTATION_BOTTLENECK"
    elif not value_supported:
        verdict = "VALUE_LEARNING_BOTTLENECK"
    else:
        verdict = "OFFLINE_SUPPORTED_ACTIVE_PENDING"
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "status": "OFFLINE_COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "elapsed_seconds": time.perf_counter() - started,
        "device": device,
        "conditions": summaries,
        "paired_against_v4_18": comparisons,
        "learned_vs_action_only": learned_vs_action,
        "learned_vs_without_relations": learned_vs_removed,
        "learned_vs_binding_swapped": learned_vs_swapped,
        "nonnegative_transfer_games": nonnegative_games,
        "oracle_gain_capture": oracle_capture,
        "topological_objective_supported": oracle_supported,
        "topological_representation_supported": representation_supported,
        "topological_value_supported": value_supported,
        "verdict": verdict,
        "all_offline_conditions_executed": bool(
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


def _run_topological_controller(
    *,
    game_id: str,
    seed: int,
    action_budget: int,
    maximum_resets: int,
    policy_model: Any,
    policy_parameters: Mapping[str, Any],
    predictor: Any,
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

    controller = "v4_19_topological_control"
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
    decision_latencies: list[float] = []
    execution_latencies: list[float] = []
    candidate_counts: list[int] = []
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
            sorted(
                {
                    str(getattr(action, "name", "")).upper()
                    for action in legal
                }
            )
        )
        observation = build_observation(
            before.grid,
            available_actions=available,
            game_state=before.game_state,
            levels_completed=before.levels_completed,
            infer_players=True,
        )
        player = observation.best_player
        player_position = (
            tuple(player.position) if player is not None else None
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
        topology_graphs = [
            build_mt_graph(
                before.grid,
                action_name=str(getattr(action, "name", "")).upper(),
                action_data=dict(
                    getattr(action, "action_args", {}) or {}
                ),
                player_position=player_position,
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
        _factor_probabilities, values, uncertainty = _predict_graphs(
            predictor,
            topology_graphs,
            device=device,
        )
        value_scores = values[:, HORIZONS.index(32)]
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
                graph
                if offset == 0
                else _graph_for_action(graph, action_name)
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
                temporal_ebm.energies(
                    (_prediction_features(prediction),)
                )[0]
                for prediction in temporal_predictions
            ],
            dtype=np.float64,
        )
        scores = _compose(
            policy["learned_scores"],
            value_scores,
            energies,
            uncertainty,
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
            collection_phase="v4_19_active",
            available_action_names=available,
            selected_action_name=str(
                getattr(selected, "name", "")
            ).upper(),
            selected_action_data=dict(
                getattr(selected, "action_args", {}) or {}
            ),
            frame_before=before.grid,
            frame_after=after.grid,
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            levels_completed_before=before.levels_completed,
            levels_completed_after=after.levels_completed,
        )
        observed_effects, _applicable, _productive, _evidence = (
            compile_semantics(executed_trace)
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
                "policy_score": float(
                    policy["learned_scores"][selected_index]
                ),
                "topological_value": float(value_scores[selected_index]),
                "uncertainty": float(uncertainty[selected_index]),
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
            "mean_candidates": (
                float(np.mean(candidate_counts))
                if candidate_counts
                else 0.0
            ),
            "decision_latency_seconds": {
                "mean": (
                    float(np.mean(decision_latencies))
                    if decision_latencies
                    else 0.0
                ),
                "p95": (
                    float(np.quantile(decision_latencies, 0.95))
                    if decision_latencies
                    else 0.0
                ),
            },
            "execution_latency_seconds": {
                "mean": (
                    float(np.mean(execution_latencies))
                    if execution_latencies
                    else 0.0
                )
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
    root = _repo_root()
    destination = _resolve(root, output_dir)
    cache = _resolve(root, cache_dir)
    manifest = load_manifest(destination)
    _verify_sources(manifest)
    result = _read_json(destination / "result.json")
    metadata = _read_json(destination / "checkpoint_metadata.json")
    device = (
        str(metadata["selected_device"])
        if requested_device == "auto"
        else requested_device
    )
    predictor, predictor_checkpoint = _load_predictor(cache, device=device)
    if predictor_checkpoint["manifest_checksum"] != manifest["manifest_checksum"]:
        raise ValueError("V4.19 active predictor/manifest mismatch")
    v418_cache = _resolve(root, DEFAULT_V418_CACHE_DIR)
    policy_model, policy_checkpoint = _load_policy_checkpoint(
        v418_cache / "v4_15" / "demonstration_policy.pt",
        device=device,
    )
    temporal_model, temporal_checkpoint = _load_temporal_checkpoint(
        v418_cache / "v4_14" / "temporal_student.pt",
        device=device,
    )
    temporal_ebm = _load_active_ebm(
        v418_cache / "v4_14" / "trajectory_ebm.pt",
        device=device,
    )
    temporal_records = load_temporal_records(v418_cache / "v4_14")
    _action_table, sequence_table, _global_value = _action_sequence_tables(
        temporal_records
    )
    reused_v417 = [
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
    reused_v418 = [
        {
            **row,
            "source": "reused_v4_18_content_addressed",
        }
        for row in _jsonl_rows(
            _resolve(root, DEFAULT_V418_DIR) / "active_runs.jsonl"
        )
        if row["controller"] == "v4_18_goal_critic"
    ]
    if len(reused_v417) != 18 or len(reused_v418) != 9:
        raise ValueError("V4.19 expected 27 frozen comparator runs")
    reused = [*reused_v417, *reused_v418]
    runs = list(reused)
    traces = []
    started = time.perf_counter()
    for game_id in ACTIVE_VALIDATION_GAMES:
        for seed in ACTIVE_SEEDS:
            run, run_traces = _run_topological_controller(
                game_id=game_id,
                seed=seed,
                action_budget=ACTIVE_ACTION_BUDGET,
                maximum_resets=ACTIVE_MAXIMUM_RESETS,
                policy_model=policy_model,
                policy_parameters=policy_checkpoint["parameters"],
                predictor=predictor,
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
    baseline_runs = {
        (str(row["game_id"]), int(row["seed"])): row
        for row in reused_v418
    }
    paired = []
    fresh = [
        row
        for row in runs
        if row["controller"] == "v4_19_topological_control"
    ]
    for row in fresh:
        baseline = baseline_runs[(str(row["game_id"]), int(row["seed"]))]
        paired.append(
            {
                "game_id": row["game_id"],
                "seed": row["seed"],
                "level_gain": int(row["levels_completed"])
                - int(baseline["levels_completed"]),
                "win_gain": int(row["wins"]) - int(baseline["wins"]),
                "game_over_delta": int(row["game_overs"])
                - int(baseline["game_overs"]),
            }
        )
    learned = metrics["v4_19_topological_control"]
    baseline = metrics["v4_18_goal_critic"]
    active_progress = bool(
        int(learned["levels"]) > 0
        and int(learned["illegal_proposals"]) == 0
        and int(learned["game_overs"]) <= int(baseline["game_overs"])
    )
    if not result["topological_objective_supported"]:
        verdict = "TOPOLOGICAL_OBJECTIVE_BOTTLENECK"
    elif not result["topological_representation_supported"]:
        verdict = "CORRESPONDENCE_OR_REPRESENTATION_BOTTLENECK"
    elif not result["topological_value_supported"]:
        verdict = "VALUE_LEARNING_BOTTLENECK"
    elif not active_progress:
        verdict = "PLANNING_OR_EXECUTION_BOTTLENECK"
    else:
        verdict = "TOPOLOGICAL_CAUSAL_CONTROL_SUPPORTED"
    active: dict[str, Any] = {
        "format_version": ACTIVE_VERSION,
        "status": "COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "games": list(ACTIVE_VALIDATION_GAMES),
        "seeds": list(ACTIVE_SEEDS),
        "fresh_runs": len(fresh),
        "reused_runs": len(reused),
        "total_runs": len(runs),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "paired_against_v4_18": paired,
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
        scratch_namespace="v4_19",
    ) as guard:
        return operation(guard.scratch_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "compile", "train", "evaluate", "active"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    def dispatch(_scratch: Path) -> dict[str, Any]:
        if args.command == "freeze":
            return freeze_manifest(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
            )
        if args.command == "compile":
            return compile_topological_credit(
                output_dir=args.output_dir,
            )
        if args.command == "train":
            return train_predictor(
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
