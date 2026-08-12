"""T12.4 offline training and active evaluation of neural novelty scoring."""

from __future__ import annotations

import io
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from theory.unified_cognition_ab_benchmark import _is_terminal

from .archive import abstract_state_from_payload
from .burst_experiment import BurstExcursion, BurstRun
from .contracts import GroundedAction
from .experiment import (
    RunStorageBudget,
    _file_sha256,
    _read_json,
    _write_json_once,
)
from .graph_experiment import (
    _grounded_actions,
    _make_env,
    _record_root,
    _restore_variant,
    _symbolic_state,
    _write_archive,
)
from .lineage_archive import LineagePreservingArchive
from .neural_novelty_protocol import (
    NeuralNoveltyProtocol,
    load_neural_novelty_dataset,
    load_neural_novelty_manifest,
    load_neural_novelty_receipt,
    neural_novelty_phase_receipt,
)
from .novelty import (
    OnlineNoveltyPredictor,
    brier_score,
    encode_state_action,
    expected_calibration_error,
)
from .shield_model import ProgressProtectedTerminalShield

EnvFactory = Callable[[str], Any]


def _resolve_manifest_path(path: str) -> Path:
    candidate = Path(path)
    return (
        candidate
        if candidate.is_absolute()
        else Path(__file__).resolve().parents[3] / candidate
    )


def _example_inputs(
    dataset: Mapping[str, Any],
    *,
    split: str,
) -> list[tuple[Any, GroundedAction, bool, bool, str]]:
    states = dict(dataset["states"])
    output = []
    for row in dataset["examples"]:
        if row["split"] != split:
            continue
        state_id = str(row["source_state_id"])
        output.append(
            (
                abstract_state_from_payload(states[state_id]),
                GroundedAction(
                    str(row["action"]["action_name"]),
                    dict(row["action"].get("action_data", {}) or {}),
                ),
                bool(row["semantic_changed"]),
                bool(row["novel"]),
                str(row["example_id"]),
            )
        )
    return output


def _action_priors(
    examples: Sequence[tuple[Any, GroundedAction, bool, bool, str]],
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    totals: dict[str, list[int]] = {}
    for _, action, changed, novel, _ in examples:
        row = totals.setdefault(action.key, [0, 0, 0])
        row[0] += 1
        row[1] += int(changed)
        row[2] += int(novel)
    priors = {
        key: ((row[1] + 1) / (row[0] + 2), (row[2] + 1) / (row[0] + 2))
        for key, row in totals.items()
    }
    count = len(examples)
    global_prior = (
        (sum(int(item[2]) for item in examples) + 1) / (count + 2),
        (sum(int(item[3]) for item in examples) + 1) / (count + 2),
    )
    return priors, global_prior


def _mean_brier(
    probabilities: Sequence[tuple[float, float]],
    targets: Sequence[tuple[float, float]],
) -> float:
    return 0.5 * (
        brier_score([row[0] for row in probabilities], [row[0] for row in targets])
        + brier_score(
            [row[1] for row in probabilities], [row[1] for row in targets]
        )
    )


def train_neural_novelty_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = load_neural_novelty_manifest(manifest_path)
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4 training requires a clean scientific freeze")
    if not manifest.get("firewall", {}).get(
        "neural_novelty_training_authorized", False
    ):
        raise ValueError("T12.4 neural training is not authorized")
    protocol = NeuralNoveltyProtocol(**dict(manifest["protocol"]))
    dataset_path = _resolve_manifest_path(str(manifest["dataset"]["path"]))
    dataset = load_neural_novelty_dataset(dataset_path, protocol=protocol)
    training = _example_inputs(dataset, split="train")
    validation = _example_inputs(dataset, split="validation")
    if not training or not validation:
        raise ValueError("T12.4 train/validation split is empty")

    predictor = OnlineNoveltyPredictor(
        seed=protocol.torch_seed,
        hidden_dim=protocol.hidden_dim,
        maximum_examples=4_096,
        batch_size=protocol.batch_size,
        learning_rate=protocol.learning_rate,
    )
    training_features = torch.tensor(
        [encode_state_action(item[0], item[1]) for item in training],
        dtype=torch.float32,
    )
    training_targets = torch.tensor(
        [[float(item[2]), float(item[3])] for item in training],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(protocol.torch_seed)
    for _ in range(protocol.training_epochs):
        order = torch.randperm(len(training), generator=generator)
        for start in range(0, len(training), protocol.batch_size):
            indices = order[start : start + protocol.batch_size]
            predictor.model.train()
            predictor.optimizer.zero_grad(set_to_none=True)
            logits = predictor.model(training_features[indices])
            loss = F.binary_cross_entropy_with_logits(
                logits,
                training_targets[indices],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.0)
            predictor.optimizer.step()
            predictor.updates += 1

    priors, global_prior = _action_priors(training)
    probabilities = []
    shuffled_probabilities = []
    baseline_probabilities = []
    targets = []
    prediction_rows = []
    for index, (state, action, changed, novel, example_id) in enumerate(validation):
        prediction = predictor.predict(state, action)
        shuffled_state = validation[(index + 1) % len(validation)][0]
        shuffled = predictor.predict(shuffled_state, action)
        predicted = (
            prediction.change_probability,
            prediction.novelty_probability,
        )
        shuffled_predicted = (
            shuffled.change_probability,
            shuffled.novelty_probability,
        )
        baseline = priors.get(action.key, global_prior)
        target = (float(changed), float(novel))
        probabilities.append(predicted)
        shuffled_probabilities.append(shuffled_predicted)
        baseline_probabilities.append(baseline)
        targets.append(target)
        prediction_rows.append(
            {
                "example_id": example_id,
                "action_key": action.key,
                "change_probability": predicted[0],
                "novelty_probability": predicted[1],
                "action_only_change_probability": baseline[0],
                "action_only_novelty_probability": baseline[1],
                "semantic_changed": changed,
                "novel": novel,
            }
        )

    model_brier = _mean_brier(probabilities, targets)
    baseline_brier = _mean_brier(baseline_probabilities, targets)
    shuffled_brier = _mean_brier(shuffled_probabilities, targets)
    change_ece = expected_calibration_error(
        [row[0] for row in probabilities], [row[0] for row in targets]
    )
    novelty_ece = expected_calibration_error(
        [row[1] for row in probabilities], [row[1] for row in targets]
    )
    metrics = {
        "training_examples": len(training),
        "validation_examples": len(validation),
        "training_epochs": protocol.training_epochs,
        "optimizer_updates": predictor.updates,
        "parameter_count": predictor.parameter_count,
        "train_semantic_change_prevalence": dataset["qa"]["train"][
            "semantic_changed_prevalence"
        ],
        "train_novelty_prevalence": dataset["qa"]["train"][
            "novelty_prevalence"
        ],
        "validation_semantic_change_prevalence": dataset["qa"]["validation"][
            "semantic_changed_prevalence"
        ],
        "validation_novelty_prevalence": dataset["qa"]["validation"][
            "novelty_prevalence"
        ],
        "model_mean_brier": model_brier,
        "action_only_mean_brier": baseline_brier,
        "brier_gain": baseline_brier - model_brier,
        "state_shuffle_mean_brier": shuffled_brier,
        "state_shuffle_degradation": shuffled_brier - model_brier,
        "change_ece": change_ece,
        "novelty_ece": novelty_ece,
        "maximum_ece": max(change_ece, novelty_ece),
    }
    passed = bool(
        metrics["training_examples"] >= protocol.minimum_training_examples
        and metrics["validation_examples"] >= protocol.minimum_validation_examples
        and metrics["parameter_count"] <= protocol.maximum_parameters
        and metrics["brier_gain"] >= protocol.minimum_brier_gain
        and metrics["state_shuffle_degradation"]
        >= protocol.minimum_state_shuffle_degradation
        and metrics["maximum_ece"] <= protocol.maximum_ece
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    checkpoint_path = destination / "neural_novelty_predictor.pt"
    buffer = io.BytesIO()
    torch.save(
        {
            "format_version": "sage-t12.1-online-novelty-v1",
            "seed": predictor.seed,
            "hidden_dim": predictor.hidden_dim,
            "maximum_examples": predictor.maximum_examples,
            "batch_size": predictor.batch_size,
            "learning_rate": predictor.learning_rate,
            "state_dict": predictor.model.state_dict(),
            "examples": [item.to_dict() for item in predictor.examples],
            "updates": predictor.updates,
            "metadata": {
                **metrics,
                "manifest_checksum": manifest["manifest_checksum"],
                "dataset_checksum": dataset["dataset_checksum"],
            },
        },
        buffer,
    )
    encoded = buffer.getvalue()
    storage.reserve(len(encoded))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as handle:
        handle.write(encoded)
    predictions_path = destination / "validation_predictions.json"
    _write_json_once(
        predictions_path,
        {
            "format_version": "sage-t12.4-neural-novelty-predictions-v1",
            "rows": prediction_rows,
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_4_NEURAL_FIT_GATE"
        if passed
        else "FAIL_T12_4_NEURAL_FIT_GATE"
    )
    report = {
        "format_version": "sage-t12.4-neural-novelty-training-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "training_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = neural_novelty_phase_receipt(
        manifest=manifest,
        phase="train",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _file_sha256(checkpoint_path),
            },
            "validation_predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": _file_sha256(predictions_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(
        destination / "training_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def run_neural_novelty_arm(
    *,
    game_id: str,
    seed: int,
    sdk_call_budget: int,
    burst_schedule: tuple[int, ...],
    environments_dir: str | Path,
    shield_payload: Mapping[str, Any],
    env_factory: EnvFactory | None = None,
    novelty_scorer: OnlineNoveltyPredictor | None = None,
    maximum_cells: int = 50_000,
) -> tuple[BurstRun, ProgressProtectedTerminalShield, dict[str, Any]]:
    if tuple(int(value) for value in burst_schedule) != (4, 8, 16):
        raise ValueError("T12.4 active runner requires the 4/8/16 schedule")
    archive = LineagePreservingArchive(maximum_cells=maximum_cells, seed=seed)
    shield = ProgressProtectedTerminalShield.from_dict(shield_payload)
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    archive.sdk_calls = 1
    _record_root(archive, env, frame)
    excursions = []
    decision_latencies = []
    neural_action_changes = 0
    scored_decisions = 0
    excursion_index = 0
    while archive.sdk_calls < sdk_call_budget:
        cell = archive.select_cell(
            remaining_sdk_calls=sdk_call_budget - archive.sdk_calls
        )
        if cell is None:
            break
        variant = cell.best_variant(archive.prefixes)
        env, frame, exact, restoration_calls = _restore_variant(
            archive=archive,
            variant=variant,
            game_id=game_id,
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        archive.sdk_calls += restoration_calls
        archive.note_replay(exact=exact)
        horizon = int(burst_schedule[excursion_index % len(burst_schedule)])
        executed = 0
        progress_edges = 0
        terminal_failures = 0
        reason = "burst_complete"
        if not exact:
            updated = replace(variant, replay_failures=variant.replay_failures + 1)
            cell.variants[variant.exact_hash] = updated
            if updated.replay_failures >= 2:
                cell.blocked = True
            reason = "restore_mismatch"
        else:
            source_cell = cell
            source_exact_hash = variant.exact_hash
            executed_prefix_id = variant.prefix_id
            executed_path_edge_ids = variant.path_edge_ids
            for _ in range(horizon):
                if archive.sdk_calls >= sdk_call_budget:
                    reason = "sdk_budget"
                    break
                before = snapshot_frame(frame)
                if _is_terminal(before.game_state):
                    reason = "terminal_source"
                    break
                candidates = _grounded_actions(env)
                allowed = tuple(
                    action
                    for action in candidates
                    if shield.allows(source_cell.cell_id, action)
                )
                default_action = archive.choose_action(source_cell, allowed)
                started = time.perf_counter()
                action = archive.choose_action(
                    source_cell,
                    allowed,
                    novelty_scorer=novelty_scorer,
                )
                if novelty_scorer is not None:
                    decision_latencies.append(
                        1_000.0 * (time.perf_counter() - started)
                    )
                    scored_decisions += int(action is not None)
                    neural_action_changes += int(
                        action is not None
                        and default_action is not None
                        and action.key != default_action.key
                    )
                if action is None:
                    source_cell.blocked = True
                    reason = "no_shield_allowed_action"
                    break
                selected = select_live_action(
                    env,
                    action.action_name,
                    action_args=action.action_data,
                )
                if selected is None:
                    source_cell.action_attempts[action.key] = (
                        source_cell.action_attempts.get(action.key, 0) + 1
                    )
                    reason = "action_unavailable"
                    break
                after_frame = _step_env_action(env, selected)
                archive.sdk_calls += 1
                executed += 1
                after = snapshot_frame(
                    after_frame,
                    fallback_available_actions=before.available_actions,
                )
                target_hash = state_signature_from_frame(after_frame)
                level_delta = max(
                    0, int(after.levels_completed) - int(before.levels_completed)
                )
                success = bool(
                    level_delta > 0
                    or str(after.game_state).upper() in {"WIN", "WON", "VICTORY"}
                )
                terminal = _is_terminal(after.game_state)
                edge = archive.add_lineage_transition(
                    source_cell_id=source_cell.cell_id,
                    source_exact_hash=source_exact_hash,
                    source_prefix_id=executed_prefix_id,
                    source_path_edge_ids=executed_path_edge_ids,
                    action=action,
                    target_state=_symbolic_state(after_frame),
                    target_exact_hash=target_hash,
                    target_level=int(after.levels_completed),
                    target_legal_actions=_grounded_actions(env),
                    terminal=terminal,
                    success=success,
                    changed=source_exact_hash != target_hash,
                )
                executed_prefix_id = edge.prefix_id
                executed_path_edge_ids = executed_path_edge_ids + (edge.edge_id,)
                progress_edges += int(edge.level_delta > 0 or edge.success)
                terminal_failures += int(edge.terminal and not edge.success)
                frame = after_frame
                source_cell = archive.cells[edge.target_cell_id]
                source_exact_hash = edge.target_exact_hash
                if edge.level_delta > 0 or edge.success:
                    reason = "progress"
                    break
                if edge.terminal:
                    reason = "terminal_failure"
                    break
        excursions.append(
            BurstExcursion(
                excursion_index=excursion_index,
                requested_horizon=horizon,
                executed_actions=executed,
                restoration_calls=restoration_calls,
                exact_restoration=exact,
                start_cell_id=cell.cell_id,
                start_exact_hash=variant.exact_hash,
                stopped_reason=reason,
                progress_edges=progress_edges,
                terminal_failures=terminal_failures,
            )
        )
        excursion_index += 1
    neural_metrics = {
        "scored_decisions": scored_decisions,
        "neural_action_changes": neural_action_changes,
        "mean_decision_latency_ms": (
            0.0
            if not decision_latencies
            else sum(decision_latencies) / len(decision_latencies)
        ),
        "p95_decision_latency_ms": _percentile_95(decision_latencies),
    }
    return BurstRun(archive=archive, excursions=tuple(excursions)), shield, neural_metrics


def _aggregate_active_gate(
    *,
    protocol: NeuralNoveltyProtocol,
    conditions: Sequence[Mapping[str, Any]],
    sdk_calls: int,
) -> tuple[bool, dict[str, Any]]:
    control_cells = 0
    treatment_cells = 0
    control_sdk = 0
    treatment_sdk = 0
    control_terminal = 0
    treatment_terminal = 0
    control_actions = 0
    treatment_actions = 0
    control_progress = 0
    treatment_progress = 0
    terminal_regressions = 0
    progress_regressions = 0
    minimum_coverage_ratio = float("inf")
    minimum_replay_exact = 1.0
    action_changes = 0
    maximum_p95_latency = 0.0
    vetoes = 0
    per_seed = []
    for condition in conditions:
        arms = dict(condition["arms"])
        control = dict(arms["lineage_shield_control"])
        treatment = dict(arms["lineage_shield_neural"])
        c = dict(control["metrics"])
        t = dict(treatment["metrics"])
        c_actions = int(c["exploration_actions"])
        t_actions = int(t["exploration_actions"])
        c_terminal = int(c["terminal_edges"])
        t_terminal = int(t["terminal_edges"])
        c_rate = c_terminal / max(1, c_actions)
        t_rate = t_terminal / max(1, t_actions)
        c_coverage = float(c["symbolic_cells_per_1000_sdk_calls"])
        t_coverage = float(t["symbolic_cells_per_1000_sdk_calls"])
        coverage_ratio = t_coverage / max(1e-12, c_coverage)
        terminal_regressions += int(t_rate > c_rate)
        progress_regressions += int(
            int(t["progress_edges"]) < int(c["progress_edges"])
        )
        minimum_coverage_ratio = min(minimum_coverage_ratio, coverage_ratio)
        minimum_replay_exact = min(
            minimum_replay_exact,
            float(c["replay_exact_rate"]),
            float(t["replay_exact_rate"]),
        )
        control_cells += int(c["symbolic_cells"])
        treatment_cells += int(t["symbolic_cells"])
        control_sdk += int(c["sdk_calls"])
        treatment_sdk += int(t["sdk_calls"])
        control_terminal += c_terminal
        treatment_terminal += t_terminal
        control_actions += c_actions
        treatment_actions += t_actions
        control_progress += int(c["progress_edges"])
        treatment_progress += int(t["progress_edges"])
        neural_metrics = dict(treatment["neural_metrics"])
        action_changes += int(neural_metrics["neural_action_changes"])
        maximum_p95_latency = max(
            maximum_p95_latency,
            float(neural_metrics["p95_decision_latency_ms"]),
        )
        vetoes += int(treatment["shield_metrics"].get("vetoes", 0))
        per_seed.append(
            {
                "seed": int(condition["seed"]),
                "control_coverage": c_coverage,
                "treatment_coverage": t_coverage,
                "coverage_ratio": coverage_ratio,
                "control_terminal_rate": c_rate,
                "treatment_terminal_rate": t_rate,
                "control_progress": int(c["progress_edges"]),
                "treatment_progress": int(t["progress_edges"]),
                "neural_action_changes": int(
                    neural_metrics["neural_action_changes"]
                ),
                "p95_decision_latency_ms": float(
                    neural_metrics["p95_decision_latency_ms"]
                ),
            }
        )
    control_coverage = 1_000.0 * control_cells / max(1, control_sdk)
    treatment_coverage = 1_000.0 * treatment_cells / max(1, treatment_sdk)
    coverage_ratio = treatment_coverage / max(1e-12, control_coverage)
    coverage_gain = coverage_ratio - 1.0
    control_terminal_rate = control_terminal / max(1, control_actions)
    treatment_terminal_rate = treatment_terminal / max(1, treatment_actions)
    terminal_rate_ratio = (
        1.0
        if control_terminal_rate == 0.0 and treatment_terminal_rate == 0.0
        else (
            1_000_000_000.0
            if control_terminal_rate == 0.0
            else treatment_terminal_rate / control_terminal_rate
        )
    )
    metrics = {
        "control_coverage": control_coverage,
        "treatment_coverage": treatment_coverage,
        "coverage_ratio": coverage_ratio,
        "relative_coverage_gain": coverage_gain,
        "minimum_per_seed_coverage_ratio": minimum_coverage_ratio,
        "control_terminal_rate": control_terminal_rate,
        "treatment_terminal_rate": treatment_terminal_rate,
        "terminal_rate_ratio": terminal_rate_ratio,
        "terminal_regression_seeds": terminal_regressions,
        "control_progress_edges": control_progress,
        "treatment_progress_edges": treatment_progress,
        "progress_regression_seeds": progress_regressions,
        "minimum_replay_exact_rate": minimum_replay_exact,
        "neural_action_changes": action_changes,
        "maximum_p95_decision_latency_ms": maximum_p95_latency,
        "treatment_shield_vetoes": vetoes,
        "sdk_calls": sdk_calls,
        "maximum_total_sdk_calls": protocol.maximum_total_sdk_calls,
        "per_seed": per_seed,
    }
    passed = bool(
        coverage_gain >= protocol.minimum_relative_coverage_gain
        and minimum_coverage_ratio >= protocol.minimum_per_seed_coverage_ratio
        and terminal_rate_ratio <= protocol.maximum_terminal_rate_ratio
        and terminal_regressions <= protocol.maximum_terminal_regression_seeds
        and treatment_progress >= control_progress
        and progress_regressions <= protocol.maximum_progress_regression_seeds
        and minimum_replay_exact >= protocol.minimum_replay_exact_rate
        and action_changes >= protocol.minimum_neural_action_changes
        and maximum_p95_latency <= protocol.maximum_p95_decision_latency_ms
        and vetoes >= 1
        and sdk_calls <= protocol.maximum_total_sdk_calls
    )
    return passed, metrics


def evaluate_neural_novelty_experiment(
    *,
    manifest_path: str | Path,
    training_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_neural_novelty_manifest(
        manifest_path, verify_code=env_factory is None
    )
    training_receipt = load_neural_novelty_receipt(
        training_receipt_path,
        manifest=manifest,
        require_passed=True,
    )
    if training_receipt.get("status") != "PASS_T12_4_NEURAL_FIT_GATE":
        raise ValueError("T12.4 active evaluation requires a passed neural fit")
    protocol = NeuralNoveltyProtocol(**dict(manifest["protocol"]))
    checkpoint = _resolve_manifest_path(
        str(training_receipt["artifacts"]["checkpoint"]["path"])
    )
    shield_payload = _read_json(
        _resolve_manifest_path(str(manifest["shield"]["path"]))
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    archive_artifacts = []
    game_id = str(manifest["game_id"])
    for seed in protocol.evaluation_seeds:
        arms = {}
        for arm in protocol.evaluation_arms:
            scorer = (
                None
                if arm == "lineage_shield_control"
                else OnlineNoveltyPredictor.load(checkpoint)
            )
            run, shield, neural_metrics = run_neural_novelty_arm(
                game_id=game_id,
                seed=seed,
                sdk_call_budget=protocol.sdk_calls_per_evaluation_arm,
                burst_schedule=protocol.burst_schedule,
                environments_dir=environments_dir,
                shield_payload=shield_payload,
                env_factory=env_factory,
                novelty_scorer=scorer,
                maximum_cells=protocol.maximum_cells,
            )
            archive_path = destination / game_id / str(seed) / f"{arm}.json"
            artifact = _write_archive(
                archive_path, run.archive, storage_budget=storage
            )
            artifact.update({"game_id": game_id, "seed": seed, "arm": arm})
            archive_artifacts.append(artifact)
            excursions_path = destination / game_id / str(seed) / f"{arm}_excursions.json"
            _write_json_once(
                excursions_path,
                {
                    "format_version": "sage-t12.4-neural-novelty-excursions-v1",
                    "game_id": game_id,
                    "seed": seed,
                    "arm": arm,
                    "excursions": [item.to_dict() for item in run.excursions],
                },
                storage_budget=storage,
            )
            arms[arm] = {
                "metrics": run.metrics(),
                "shield_metrics": shield.metrics(),
                "neural_metrics": neural_metrics,
                "archive": artifact,
                "excursions": {
                    "path": str(excursions_path.resolve()),
                    "sha256": _file_sha256(excursions_path),
                },
            }
        conditions.append({"game_id": game_id, "seed": seed, "arms": arms})
    sdk_calls = sum(
        int(arm["metrics"]["sdk_calls"])
        for condition in conditions
        for arm in condition["arms"].values()
    )
    passed, metrics = _aggregate_active_gate(
        protocol=protocol,
        conditions=conditions,
        sdk_calls=sdk_calls,
    )
    evaluation_path = destination / "paired_evaluation.json"
    _write_json_once(
        evaluation_path,
        {
            "format_version": "sage-t12.4-neural-novelty-paired-evaluation-v1",
            "conditions": conditions,
            "archives": archive_artifacts,
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_4_NEURAL_ACTIVE_GATE"
        if passed
        else "FAIL_T12_4_NEURAL_ACTIVE_GATE"
    )
    report = {
        "format_version": "sage-t12.4-neural-novelty-active-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "training_receipt_checksum": training_receipt["receipt_checksum"],
        "metrics": metrics,
        "conditions": conditions,
        "storage": storage.snapshot(),
    }
    report_path = destination / "active_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = neural_novelty_phase_receipt(
        manifest=manifest,
        phase="evaluate",
        passed=passed,
        status=status,
        metrics=metrics,
        parent_receipt=training_receipt,
        artifacts={
            "paired_evaluation": {
                "path": str(evaluation_path.resolve()),
                "sha256": _file_sha256(evaluation_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "checkpoint": dict(training_receipt["artifacts"]["checkpoint"]),
        },
    )
    _write_json_once(
        destination / "active_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def neural_novelty_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_neural_novelty_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_neural_novelty_receipt(receipt_path, manifest=manifest)
    )
    freeze_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4_FREEZE"
    )
    fit_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4_NEURAL_FIT_GATE"
    )
    active_passed = bool(
        receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_4_NEURAL_ACTIVE_GATE"
    )
    return {
        "format_version": "sage-t12.4-neural-novelty-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3e_status": manifest["parent"]["receipt"]["status"],
        "receipt": (
            None
            if receipt is None
            else {
                "phase": receipt["phase"],
                "passed": receipt["passed"],
                "status": receipt["status"],
                "receipt_checksum": receipt["receipt_checksum"],
            }
        ),
        "next_phase_authorized": freeze_passed or fit_passed or active_passed,
        "firewall": {
            **dict(manifest["firewall"]),
            "neural_active_evaluation_authorized": fit_passed,
            "t12_5_freeze_authorized": active_passed,
        },
    }


__all__ = [
    "evaluate_neural_novelty_experiment",
    "neural_novelty_experiment_status",
    "run_neural_novelty_arm",
    "train_neural_novelty_experiment",
]
