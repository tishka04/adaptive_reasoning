"""SAGE12 V4.17 sequence-conditioned transformation policy.

Commands::

    python -m theory.sage12.sequence_transformation_policy_v4_17 freeze
    python -m theory.sage12.sequence_transformation_policy_v4_17 prepare --device cuda:0
    python -m theory.sage12.sequence_transformation_policy_v4_17 evaluate --device cuda:0
    python -m theory.sage12.sequence_transformation_policy_v4_17 active --device cuda:0

V4.17 composes, without transfer-set fitting, the frozen V4.15 sequence
policy, the V4.16 causal transformation predictor and the V4.14 temporal EBM.
Future outcomes are scoring-only and the final confirmation split stays
closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from theory.live_transition_loop import build_observation
from theory.sage11.splits import NEURO_HOLDOUT_V1

from .action_target_data import build_action_target_trace, grid_sha256
from .counterfactual_semantic_panels_v4_11 import load_teacher_panels
from .demonstration_milestone_policy_v4_15 import (
    ACTIVE_VALIDATION_GAMES,
    DEFAULT_OUTPUT_DIR as DEFAULT_V415_DIR,
    EBM_COEFFICIENT,
    PolicyBelief,
    TRANSFER_GAMES,
    _action_sequence_tables,
    _active_metrics,
    _advance_policy_belief,
    _candidate_action_plan,
    _file_fingerprint,
    _graph_for_action,
    _live_action_signature,
    _live_candidate_graph,
    _load_active_ebm,
    _load_policy_checkpoint,
    _load_temporal_checkpoint,
    _paired_bootstrap_rows,
    _predict_candidate_rollouts,
    _prediction_features,
    _read_jsonl,
    _score_candidate_graphs,
    _summarize_decisions,
    _zscore,
    load_temporal_records,
)
from .human_temporal_semantics_v4_14 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_V414_DIR,
    TemporalBeliefState,
)
from .morpho_topological_v4_16 import (
    DEFAULT_HUMAN_TRACES_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_V416_DIR,
    DEFAULT_V411_DIR,
    cluster_embeddings,
    compile_corpus,
    evaluate as evaluate_v416,
    freeze_manifest as freeze_v416_manifest,
    prepare_shadow,
    train_model as train_v416_model,
)
from .mt.clustering import ClusterRegistry, TransformationPrototypeMemory
from .mt.graph import MorphoTopologicalGraph, build_mt_graph
from .mt.model import (
    encode_transitions,
    load_mt_model,
    predict_graph_details,
)
from .mt.transition import MTTransitionRecord
from .semantic_teacher_v4_9 import (
    _checksum,
    _file_sha256,
    _read_json,
    _write_json,
    _write_jsonl,
    compile_semantics,
)


FORMAT_VERSION = "sage12-sequence-transformation-policy-v4.17"
MANIFEST_VERSION = "sage12-sequence-transformation-manifest-v4.17"
RESULT_VERSION = "sage12-sequence-transformation-result-v4.17"
ACTIVE_VERSION = "sage12-sequence-transformation-active-v4.17"
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage12" / "sequence_transformation_policy_v4_17"
)
PROTOCOL_PATH = (
    Path("reports") / "SAGE12_SEQUENCE_TRANSFORMATION_POLICY_V4_17_PROTOCOL.md"
)
SEED = 5_170
TRANSFORMATION_COEFFICIENT = 0.5
TEMPORAL_EBM_COEFFICIENT = EBM_COEFFICIENT
BOOTSTRAP_SAMPLES = 2_000
ACTIVE_SEEDS = (0, 1, 2)
ACTIVE_ACTION_BUDGET = 1_000
ACTIVE_MAXIMUM_RESETS = 14


def _required_sources(
    *,
    v415_dir: Path,
    v414_dir: Path,
    v411_dir: Path,
    traces_dir: Path,
) -> tuple[Path, ...]:
    human = tuple(
        path
        for path in sorted(traces_dir.glob("*.steps.jsonl"))
        if path.name.split("-", 1)[0]
        in {"ar25", "bp35", "cd82", "cn04", "dc22", "ft09"}
    )
    v411 = tuple(
        v411_dir / "source_train_shards" / f"{game}.jsonl" for game in TRANSFER_GAMES
    )
    v415 = tuple(
        v415_dir / name
        for name in (
            "frozen_manifest.json",
            "teacher_qa.json",
            "semantic_result.json",
            "checkpoint_metadata.json",
            "demonstration_policy.pt",
            "transfer_predictions.jsonl",
            "transfer_decisions.jsonl",
            "active_runs.jsonl",
            "active_validation.json",
            "result.json",
        )
    )
    v414 = tuple(
        v414_dir / name
        for name in (
            "temporal_student.pt",
            "trajectory_ebm.pt",
            "transfer_predictions.jsonl",
            "teacher_corpus.jsonl",
        )
    )
    implementation = (
        Path(__file__),
        Path("theory") / "sage12" / "morpho_topological_v4_16.py",
        *tuple(sorted((Path("theory") / "sage12" / "mt").glob("*.py"))),
        Path("reports") / "SAGE12_MORPHO_TOPOLOGICAL_V4_16_PROTOCOL.md",
        PROTOCOL_PATH,
    )
    return (
        *human,
        *v411,
        v411_dir / "frozen_manifest.json",
        *v415,
        *v414,
        *implementation,
    )


def freeze_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v415_dir: str | Path = DEFAULT_V415_DIR,
    v414_dir: str | Path = DEFAULT_V414_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    traces_dir: str | Path = DEFAULT_HUMAN_TRACES_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir)
    sources = _required_sources(
        v415_dir=Path(v415_dir),
        v414_dir=Path(v414_dir),
        v411_dir=Path(v411_dir),
        traces_dir=Path(traces_dir),
    )
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(path.as_posix() for path in missing))
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "split": {
            "human_train": ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09"],
            "offline_transfer": list(TRANSFER_GAMES),
            "active_validation": list(ACTIVE_VALIDATION_GAMES),
            "final_confirmation_closed": list(NEURO_HOLDOUT_V1),
        },
        "source_fingerprints": {
            path.as_posix(): _file_fingerprint(path) for path in sources
        },
        "composition": {
            "policy": "v4_15_learned_milestone_score",
            "transformation": "v4_16_causal_prototype_value",
            "trajectory": "v4_14_depth_three_temporal_energy",
            "policy_coefficient": 1.0,
            "transformation_coefficient": TRANSFORMATION_COEFFICIENT,
            "temporal_ebm_coefficient": TEMPORAL_EBM_COEFFICIENT,
            "normalization": "within_candidate_set_zscore",
            "fit_on_transfer": False,
            "future_frames_deployable": False,
            "prototype_updates_during_validation": False,
        },
        "offline": {
            "panels": 768,
            "arms": 2_831,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "minimum_nonnegative_games": 5,
            "completion_absolute_minimum": 1,
            "completion_fraction_minimum": 0.5,
            "conditions": [
                "v4_15_learned_policy",
                "v4_15_policy_temporal_ebm",
                "v4_16_transformation_only",
                "sequence_plus_transformation",
                "v4_17_hybrid",
                "hybrid_without_topological_relations",
                "hybrid_permuted_transform",
                "oracle_transformation_hybrid",
                "true_world_learned_ebm",
                "exact_oracle",
            ],
        },
        "active": {
            "seeds": list(ACTIVE_SEEDS),
            "action_budget": ACTIVE_ACTION_BUDGET,
            "maximum_resets": ACTIVE_MAXIMUM_RESETS,
            "reuse_v4_15_runs_by_checksum": True,
            "fresh_controller": "v4_17_hybrid",
        },
        "authority": {
            "holdout_opened": False,
            "controller_authority_promoted": False,
            "active_descriptive_only": True,
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
        raise ValueError("unsupported V4.17 manifest")
    expected = str(payload["manifest_checksum"])
    clean = dict(payload)
    clean.pop("manifest_checksum")
    if _checksum(clean) != expected:
        raise ValueError("V4.17 manifest checksum mismatch")
    return payload


def _verify_frozen_sources(manifest: Mapping[str, Any]) -> None:
    for raw_path, expected in manifest["source_fingerprints"].items():
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        actual = _file_fingerprint(path)
        if actual != expected:
            raise ValueError(f"V4.17 frozen source drift: {path.as_posix()}")


def prepare_components(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v416_dir: str | Path = DEFAULT_V416_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    component = Path(v416_dir)
    started = time.perf_counter()
    if not (component / "frozen_manifest.json").exists():
        freeze_v416_manifest(output_dir=component)
    teacher = compile_corpus(output_dir=component)
    training = train_v416_model(output_dir=component, device=device)
    clustering = cluster_embeddings(output_dir=component)
    result = evaluate_v416(output_dir=component, device=device)
    shadow = prepare_shadow(output_dir=component)
    paths = (
        component / "frozen_manifest.json",
        component / "teacher_qa.json",
        component / "training_result.json",
        component / "train_embeddings.jsonl",
        component / "cluster_registry.json",
        component / "result.json",
        component / "shadow_activation.json",
        component / "mt_model.pt",
    )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "component": "sage_mt_v4_16",
        "device": device,
        "teacher": teacher,
        "training": training,
        "clustering": {
            "prototype_count": len(clustering.get("prototypes", ())),
            "selected_parameters": clustering.get("selected_parameters", {}),
            "stability_ari": clustering.get("stability_ari", 0.0),
            "eligible_coverage": clustering.get("eligible_coverage", 0.0),
        },
        "component_result": result,
        "shadow": shadow,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_sha256": {path.name: _file_sha256(path) for path in paths},
    }
    payload["preparation_checksum"] = _checksum(payload)
    _write_json(destination / "preparation.json", payload)
    return payload


def _load_mt_runtime(
    directory: Path,
    *,
    device: str,
) -> tuple[Any, Any, TransformationPrototypeMemory]:
    import torch

    checkpoint = torch.load(
        directory / "mt_model.pt",
        map_location=device,
        weights_only=False,
    )
    model, config, _metadata = load_mt_model(checkpoint, device=device)
    registry = ClusterRegistry.from_dict(
        _read_json(directory / "cluster_registry.json")
    )
    return model, config, TransformationPrototypeMemory(registry)


def _mt_value(
    vector: Sequence[float],
    uncertainty: float,
    graph: MorphoTopologicalGraph,
    memory: TransformationPrototypeMemory,
) -> tuple[float, int]:
    matches = memory.retrieve(
        vector,
        action_family=graph.action_family,
        uncertainty=float(uncertainty),
        maximum_matches=8,
    )
    if not matches:
        return -float(uncertainty), 0
    weights = np.exp(
        np.asarray([match.similarity for match in matches], dtype=np.float64) / 0.10
    )
    weights /= weights.sum()
    value = sum(
        float(weight) * (match.productive_probability - match.risk_probability)
        for weight, match in zip(weights, matches)
    )
    return float(value - 0.10 * float(uncertainty)), len(matches)


def _compose_scores(
    policy: Sequence[float],
    transformation: Sequence[float],
    temporal_energy: Sequence[float] | None = None,
) -> np.ndarray:
    output = _zscore(policy) + TRANSFORMATION_COEFFICIENT * _zscore(transformation)
    if temporal_energy is not None:
        output -= TEMPORAL_EBM_COEFFICIENT * _zscore(temporal_energy)
    return output


def _deterministic_permutation(
    values: Sequence[float],
    *,
    key: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        return array.copy()
    shift = 1 + (
        int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (len(array) - 1)
    )
    return np.roll(array, shift)


def _argmax(values: Sequence[float], arm_indices: Sequence[int]) -> int:
    return max(
        range(len(values)),
        key=lambda index: (float(values[index]), -int(arm_indices[index])),
    )


def _transfer_record_map(
    records: Sequence[MTTransitionRecord],
) -> dict[tuple[str, int], MTTransitionRecord]:
    output = {}
    for record in records:
        panel_id = str(record.audit.get("panel_id", ""))
        arm_index = int(record.audit.get("arm_index", -1))
        if panel_id and arm_index >= 0:
            output[(panel_id, arm_index)] = record
    return output


def evaluate_offline(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    v415_dir: str | Path = DEFAULT_V415_DIR,
    v416_dir: str | Path = DEFAULT_V416_DIR,
    v411_dir: str | Path = DEFAULT_V411_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    preparation = _read_json(destination / "preparation.json")
    v415_path = Path(v415_dir)
    component = Path(v416_dir)
    panels = tuple(
        panel
        for panel in load_teacher_panels(v411_dir)
        if panel.game_id in TRANSFER_GAMES
    )
    if len(panels) != int(manifest["offline"]["panels"]):
        raise ValueError("V4.17 offline panel count drift")
    mt_records = tuple(
        MTTransitionRecord.from_dict(row)
        for row in _read_jsonl(component / "transfer_transitions.jsonl")
    )
    record_map = _transfer_record_map(mt_records)
    selected_records = []
    keys = []
    for panel in panels:
        for arm in panel.arms:
            key = (panel.panel_id, int(arm.arm_index))
            if key not in record_map:
                raise ValueError(f"missing V4.16 arm for V4.17: {key}")
            keys.append(key)
            selected_records.append(record_map[key])
    model, config, memory = _load_mt_runtime(component, device=device)
    predictions = predict_graph_details(
        model,
        [record.graph_before for record in selected_records],
        config=config,
        device=device,
    )
    no_relation_predictions = predict_graph_details(
        model,
        [record.graph_before.without_relations() for record in selected_records],
        config=config,
        device=device,
    )
    teacher_embeddings = encode_transitions(
        model,
        selected_records,
        config=config,
        device=device,
    )
    mt_by_key = {}
    for key, record, prediction, no_relation, teacher in zip(
        keys,
        selected_records,
        predictions,
        no_relation_predictions,
        teacher_embeddings,
    ):
        predicted_value, match_count = _mt_value(
            prediction.vector,
            prediction.uncertainty,
            record.graph_before,
            memory,
        )
        no_relation_value, no_relation_matches = _mt_value(
            no_relation.vector,
            no_relation.uncertainty,
            record.graph_before,
            memory,
        )
        teacher_value, teacher_matches = _mt_value(
            teacher.vector,
            0.0,
            record.graph_before,
            memory,
        )
        mt_by_key[key] = {
            "predicted_value": predicted_value,
            "uncertainty": float(prediction.uncertainty),
            "match_count": match_count,
            "no_relation_value": no_relation_value,
            "no_relation_matches": no_relation_matches,
            "teacher_value": teacher_value,
            "teacher_matches": teacher_matches,
        }
    v415_predictions = {
        str(row["panel_id"]): row
        for row in _read_jsonl(v415_path / "transfer_predictions.jsonl")
    }
    v415_decisions = _read_jsonl(v415_path / "transfer_decisions.jsonl")
    old_selection = {
        (str(row["panel_id"]), str(row["method"])): int(row["selected_arm"])
        for row in v415_decisions
    }
    decisions = []
    prediction_rows = []
    for panel in panels:
        arm_indices = [int(arm.arm_index) for arm in panel.arms]
        by_arm = {
            int(row["arm_index"]): row
            for row in v415_predictions[panel.panel_id]["arms"]
        }
        policy = [float(by_arm[index]["learned_score"]) for index in arm_indices]
        energy = [float(by_arm[index]["temporal_energy"]) for index in arm_indices]
        transformation = [
            float(mt_by_key[(panel.panel_id, index)]["predicted_value"])
            for index in arm_indices
        ]
        no_relation = [
            float(mt_by_key[(panel.panel_id, index)]["no_relation_value"])
            for index in arm_indices
        ]
        teacher = [
            float(mt_by_key[(panel.panel_id, index)]["teacher_value"])
            for index in arm_indices
        ]
        permuted = _deterministic_permutation(
            transformation,
            key=panel.panel_id,
        )
        sequence_transform = _compose_scores(policy, transformation)
        hybrid = _compose_scores(policy, transformation, energy)
        hybrid_no_relation = _compose_scores(policy, no_relation, energy)
        hybrid_permuted = _compose_scores(policy, permuted, energy)
        oracle_transform_hybrid = _compose_scores(policy, teacher, energy)
        selected = {
            "v4_15_learned_policy": old_selection[
                (panel.panel_id, "learned_milestone_policy")
            ],
            "v4_15_policy_temporal_ebm": old_selection[
                (panel.panel_id, "policy_temporal_ebm")
            ],
            "v4_16_transformation_only": arm_indices[
                _argmax(transformation, arm_indices)
            ],
            "sequence_plus_transformation": arm_indices[
                _argmax(sequence_transform, arm_indices)
            ],
            "v4_17_hybrid": arm_indices[_argmax(hybrid, arm_indices)],
            "hybrid_without_topological_relations": arm_indices[
                _argmax(hybrid_no_relation, arm_indices)
            ],
            "hybrid_permuted_transform": arm_indices[
                _argmax(hybrid_permuted, arm_indices)
            ],
            "oracle_transformation_hybrid": arm_indices[
                _argmax(oracle_transform_hybrid, arm_indices)
            ],
            "true_world_learned_ebm": old_selection[
                (panel.panel_id, "true_world_learned_ebm")
            ],
            "exact_oracle": old_selection[(panel.panel_id, "oracle_energy")],
        }
        arm_by_index = {int(arm.arm_index): arm for arm in panel.arms}
        oracle = arm_by_index[selected["exact_oracle"]]
        for method, selected_index in selected.items():
            arm = arm_by_index[selected_index]
            decisions.append(
                {
                    "format_version": FORMAT_VERSION,
                    "panel_id": panel.panel_id,
                    "game_id": panel.game_id,
                    "method": method,
                    "selected_arm": selected_index,
                    "oracle_arm": int(oracle.arm_index),
                    "utility": float(arm.horizon_return),
                    "oracle_utility": float(oracle.horizon_return),
                    "regret": float(oracle.horizon_return) - float(arm.horizon_return),
                    "oracle_action": selected_index == int(oracle.arm_index),
                    "completion_selected": bool(arm.labels["level_complete"]),
                    "completion_available": any(
                        bool(row.labels["level_complete"]) for row in panel.arms
                    ),
                }
            )
        prediction_rows.append(
            {
                "format_version": FORMAT_VERSION,
                "panel_id": panel.panel_id,
                "game_id": panel.game_id,
                "arms": [
                    {
                        "arm_index": arm_index,
                        "utility": float(arm_by_index[arm_index].horizon_return),
                        "completion": bool(
                            arm_by_index[arm_index].labels["level_complete"]
                        ),
                        "policy_score": policy[index],
                        "transformation_score": transformation[index],
                        "transformation_uncertainty": mt_by_key[
                            (panel.panel_id, arm_index)
                        ]["uncertainty"],
                        "prototype_matches": mt_by_key[(panel.panel_id, arm_index)][
                            "match_count"
                        ],
                        "no_relation_score": no_relation[index],
                        "teacher_transformation_score": teacher[index],
                        "temporal_energy": energy[index],
                        "hybrid_score": float(hybrid[index]),
                    }
                    for index, arm_index in enumerate(arm_indices)
                ],
            }
        )
    decisions_path = destination / "offline_decisions.jsonl"
    predictions_path = destination / "offline_predictions.jsonl"
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(predictions_path, prediction_rows)
    methods = sorted({str(row["method"]) for row in decisions})
    by_method = {
        method: [row for row in decisions if row["method"] == method]
        for method in methods
    }
    metrics = {method: _summarize_decisions(rows) for method, rows in by_method.items()}
    comparisons = {
        "hybrid_over_v4_15_ebm": _paired_bootstrap_rows(
            by_method["v4_17_hybrid"],
            by_method["v4_15_policy_temporal_ebm"],
            samples=BOOTSTRAP_SAMPLES,
            seed=SEED,
        ),
        "hybrid_over_transformation_only": _paired_bootstrap_rows(
            by_method["v4_17_hybrid"],
            by_method["v4_16_transformation_only"],
            samples=BOOTSTRAP_SAMPLES,
            seed=SEED + 1,
        ),
        "relation_degradation": _paired_bootstrap_rows(
            by_method["v4_17_hybrid"],
            by_method["hybrid_without_topological_relations"],
            samples=BOOTSTRAP_SAMPLES,
            seed=SEED + 2,
        ),
        "permutation_degradation": _paired_bootstrap_rows(
            by_method["v4_17_hybrid"],
            by_method["hybrid_permuted_transform"],
            samples=BOOTSTRAP_SAMPLES,
            seed=SEED + 3,
        ),
        "oracle_transformation_headroom": _paired_bootstrap_rows(
            by_method["oracle_transformation_hybrid"],
            by_method["v4_17_hybrid"],
            samples=BOOTSTRAP_SAMPLES,
            seed=SEED + 4,
        ),
    }
    baseline_game = metrics["v4_15_policy_temporal_ebm"]["per_game"]
    hybrid_game = metrics["v4_17_hybrid"]["per_game"]
    nonnegative_games = sum(
        hybrid_game[game]["mean_utility"] >= baseline_game[game]["mean_utility"]
        for game in TRANSFER_GAMES
    )
    completion = {
        method: sum(bool(row["completion_selected"]) for row in rows)
        for method, rows in by_method.items()
    }
    oracle_completion = completion["exact_oracle"]
    required_completion = max(
        int(manifest["offline"]["completion_absolute_minimum"]),
        math.ceil(
            float(manifest["offline"]["completion_fraction_minimum"])
            * oracle_completion
        ),
    )
    checks = {
        "hybrid_over_v4_15_ebm_ci_positive": (
            comparisons["hybrid_over_v4_15_ebm"]["ci_low"] > 0.0
        ),
        "hybrid_over_transformation_ci_positive": (
            comparisons["hybrid_over_transformation_only"]["ci_low"] > 0.0
        ),
        "nonnegative_transfer_games": (
            nonnegative_games >= int(manifest["offline"]["minimum_nonnegative_games"])
        ),
        "topological_relations_used": (
            comparisons["relation_degradation"]["ci_low"] > 0.0
        ),
        "completion_absolute_and_fraction": (
            completion["v4_17_hybrid"] >= required_completion
        ),
        "all_registered_conditions_executed": set(methods)
        == set(manifest["offline"]["conditions"]),
        "future_outcomes_scoring_only": True,
    }
    offline_supported = all(checks.values())
    component_result = preparation["component_result"]
    result: dict[str, Any] = {
        "format_version": RESULT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "preparation_checksum": preparation["preparation_checksum"],
        "verdict": (
            "HYBRID_OFFLINE_SUPPORTED_ACTIVE_PENDING"
            if offline_supported
            else (
                "TRANSFORMATION_COMPONENT_BOTTLENECK"
                if not component_result["shadow_authorized"]
                else "SEQUENCE_TRANSFORMATION_BOTTLENECK"
            )
        ),
        "component_v4_16_supported": bool(component_result["shadow_authorized"]),
        "hybrid_offline_supported": offline_supported,
        "metrics": metrics,
        "comparisons": comparisons,
        "completion_capture": {
            "oracle": oracle_completion,
            "required": required_completion,
            "selected_by_method": completion,
        },
        "nonnegative_transfer_games": nonnegative_games,
        "checks": checks,
        "panels": len(panels),
        "arms": sum(len(panel.arms) for panel in panels),
        "topology": {
            "policy_coefficient": 1.0,
            "transformation_coefficient": TRANSFORMATION_COEFFICIENT,
            "temporal_ebm_coefficient": TEMPORAL_EBM_COEFFICIENT,
            "future_outcomes_scoring_only": True,
            "teacher_transform_oracle_only": True,
        },
        "active_validation": {
            "status": "PENDING_BOUNDED_RUN",
            "games": list(ACTIVE_VALIDATION_GAMES),
            "seeds": list(ACTIVE_SEEDS),
        },
        "all_conditions_executed": False,
        "holdout_opened": False,
        "authority_promoted": False,
        "artifact_sha256": {
            "offline_decisions": _file_sha256(decisions_path),
            "offline_predictions": _file_sha256(predictions_path),
            "v4_16_result": _file_sha256(component / "result.json"),
        },
    }
    result["result_checksum"] = _checksum(result)
    _write_json(destination / "result.json", result)
    return result


def _run_hybrid_controller(
    *,
    game_id: str,
    seed: int,
    action_budget: int,
    maximum_resets: int,
    policy_model: Any,
    policy_parameters: Mapping[str, Any],
    mt_model: Any,
    mt_config: Any,
    mt_memory: TransformationPrototypeMemory,
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

    controller = "v4_17_hybrid"
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
    prototype_coverages = []
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
        observation = build_observation(
            before.grid,
            available_actions=available,
            game_state=before.game_state,
            levels_completed=before.levels_completed,
            infer_players=True,
        )
        player_position = (
            tuple(observation.best_player.position)
            if observation.best_player is not None
            else None
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
        mt_graphs = [
            build_mt_graph(
                before.grid,
                action_name=str(getattr(action, "name", "")).upper(),
                action_data=dict(getattr(action, "action_args", {}) or {}),
                player_position=player_position,
            )
            for action in legal
        ]
        mt_predictions = predict_graph_details(
            mt_model,
            mt_graphs,
            config=mt_config,
            device=device,
        )
        mt_details = [
            _mt_value(
                prediction.vector,
                prediction.uncertainty,
                graph,
                mt_memory,
            )
            for prediction, graph in zip(mt_predictions, mt_graphs)
        ]
        mt_scores = [row[0] for row in mt_details]
        prototype_coverages.append(float(np.mean([row[1] > 0 for row in mt_details])))
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
            for graph, action_plan in zip(policy_graphs, action_plans)
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
            mt_scores,
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
            collection_phase="v4_17_active",
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
                "selected_plan": list(action_plans[selected_index]),
                "policy_score": float(policy["learned_scores"][selected_index]),
                "transformation_score": float(mt_scores[selected_index]),
                "transformation_uncertainty": float(
                    mt_predictions[selected_index].uncertainty
                ),
                "prototype_matches": int(mt_details[selected_index][1]),
                "temporal_energy": float(energies[selected_index]),
                "hybrid_score": float(scores[selected_index]),
                "predicted_milestone": policy["predicted_milestone"],
                "levels_completed_before": before.levels_completed,
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
            "prototype_coverage": float(np.mean(prototype_coverages))
            if prototype_coverages
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
    v415_dir: str | Path = DEFAULT_V415_DIR,
    v416_dir: str | Path = DEFAULT_V416_DIR,
    v414_dir: str | Path = DEFAULT_V414_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest = load_manifest(destination)
    _verify_frozen_sources(manifest)
    result_path = destination / "result.json"
    result = _read_json(result_path)
    v415_path = Path(v415_dir)
    v416_path = Path(v416_dir)
    v414_path = Path(v414_dir)
    policy_model, policy_checkpoint = _load_policy_checkpoint(
        v415_path / "demonstration_policy.pt",
        device=device,
    )
    policy_parameters = policy_checkpoint["parameters"]
    mt_model, mt_config, mt_memory = _load_mt_runtime(
        v416_path,
        device=device,
    )
    temporal_model, temporal_checkpoint = _load_temporal_checkpoint(
        v414_path / "temporal_student.pt",
        device=device,
    )
    temporal_parameters = temporal_checkpoint["parameters"]
    temporal_ebm = _load_active_ebm(
        v414_path / "trajectory_ebm.pt",
        device=device,
    )
    temporal_records = load_temporal_records(v414_path)
    _action_table, sequence_table, _global_value = _action_sequence_tables(
        temporal_records
    )
    baseline_path = v415_path / "active_runs.jsonl"
    expected = manifest["source_fingerprints"][baseline_path.as_posix()]["sha256"]
    if _file_sha256(baseline_path) != expected:
        raise ValueError("V4.17 V4.15 active baseline drift")
    reused = [
        {
            **row,
            "source": "reused_v4_15_content_addressed",
        }
        for row in _read_jsonl(baseline_path)
    ]
    if len(reused) != 27:
        raise ValueError("V4.17 expected 27 V4.15 active runs")
    runs = list(reused)
    traces = []
    started = time.perf_counter()
    for game_id in ACTIVE_VALIDATION_GAMES:
        for seed in ACTIVE_SEEDS:
            run, run_traces = _run_hybrid_controller(
                game_id=game_id,
                seed=seed,
                action_budget=ACTIVE_ACTION_BUDGET,
                maximum_resets=ACTIVE_MAXIMUM_RESETS,
                policy_model=policy_model,
                policy_parameters=policy_parameters,
                mt_model=mt_model,
                mt_config=mt_config,
                mt_memory=mt_memory,
                temporal_model=temporal_model,
                temporal_parameters=temporal_parameters,
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
    baseline_by_key = {
        (str(row["game_id"]), int(row["seed"])): row
        for row in reused
        if row["controller"] == "milestone_policy_temporal_ebm"
    }
    paired = []
    for row in runs:
        if row["controller"] != "v4_17_hybrid":
            continue
        baseline = baseline_by_key[(str(row["game_id"]), int(row["seed"]))]
        paired.append(
            {
                "game_id": row["game_id"],
                "seed": row["seed"],
                "level_gain": int(row["levels_completed"])
                - int(baseline["levels_completed"]),
                "win_gain": int(row["wins"]) - int(baseline["wins"]),
                "game_over_delta": int(row["game_overs"]) - int(baseline["game_overs"]),
            }
        )
    hybrid_metrics = metrics["v4_17_hybrid"]
    active_progress = bool(
        int(hybrid_metrics["levels"]) > 0
        and int(hybrid_metrics["illegal_proposals"]) == 0
    )
    active: dict[str, Any] = {
        "format_version": ACTIVE_VERSION,
        "status": "COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "games": list(ACTIVE_VALIDATION_GAMES),
        "seeds": list(ACTIVE_SEEDS),
        "fresh_runs": 9,
        "reused_runs": 27,
        "total_runs": len(runs),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "paired_against_v4_15_milestone_ebm": paired,
        "hybrid_active_progress": active_progress,
        "descriptive_only": True,
        "holdout_opened": False,
        "authority_promoted": False,
        "artifact_sha256": {
            "runs": _file_sha256(runs_path),
            "traces": _file_sha256(traces_path),
        },
    }
    active["active_checksum"] = _checksum(active)
    active_path = destination / "active_validation.json"
    _write_json(active_path, active)
    offline_supported = bool(result["hybrid_offline_supported"])
    component_supported = bool(result["component_v4_16_supported"])
    if offline_supported and active_progress:
        verdict = "SEQUENCE_TRANSFORMATION_SUPPORTED"
    elif active_progress:
        verdict = "LIVE_PROGRESS_WITH_CAUSAL_GATES_FAILED"
    elif not component_supported:
        verdict = "TRANSFORMATION_COMPONENT_BOTTLENECK"
    else:
        verdict = "SEQUENCE_TRANSFORMATION_BOTTLENECK"
    result["verdict"] = verdict
    result["active_validation"] = active
    result["hybrid_active_progress"] = active_progress
    result["all_conditions_executed"] = True
    result["holdout_opened"] = False
    result["authority_promoted"] = False
    result["artifact_sha256"].update(
        {
            "active_runs": _file_sha256(runs_path),
            "active_traces": _file_sha256(traces_path),
            "active_validation": _file_sha256(active_path),
        }
    )
    result.pop("result_checksum", None)
    result["result_checksum"] = _checksum(result)
    _write_json(result_path, result)
    return active


def run_all(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = Path(output_dir)
    if not (destination / "frozen_manifest.json").exists():
        freeze_manifest(output_dir=destination)
    prepare_components(output_dir=destination, device=device)
    evaluate_offline(output_dir=destination, device=device)
    return run_active_validation(output_dir=destination, device=device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "prepare", "evaluate", "active", "run-all"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        payload = freeze_manifest(output_dir=args.output_dir)
    elif args.command == "prepare":
        payload = prepare_components(
            output_dir=args.output_dir,
            device=args.device,
        )
    elif args.command == "evaluate":
        payload = evaluate_offline(
            output_dir=args.output_dir,
            device=args.device,
        )
    elif args.command == "active":
        payload = run_active_validation(
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        payload = run_all(output_dir=args.output_dir, device=args.device)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
