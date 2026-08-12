"""T12.3e paired terminal-shield retest on lineage-preserving archives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from theory.unified_cognition_ab_benchmark import _is_terminal

from .burst_experiment import BurstExcursion, BurstRun
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
from .lineage_shield_protocol import (
    LineageShieldProtocol,
    lineage_shield_phase_receipt,
    load_lineage_shield_manifest,
    load_lineage_shield_receipt,
    load_lineage_shield_registry,
)
from .shield_experiment import WitnessShieldTrial, replay_witness_with_shield
from .shield_model import ProgressProtectedTerminalShield
from .witness_protocol import load_witness_registry

EnvFactory = Callable[[str], Any]


def run_lineage_shield_arm(
    *,
    game_id: str,
    seed: int,
    sdk_call_budget: int,
    burst_schedule: tuple[int, ...],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
    shield: ProgressProtectedTerminalShield | None = None,
    maximum_cells: int = 50_000,
) -> tuple[BurstRun, ProgressProtectedTerminalShield | None]:
    if tuple(int(value) for value in burst_schedule) != (4, 8, 16):
        raise ValueError("T12.3e burst runner requires the 4/8/16 schedule")
    archive = LineagePreservingArchive(maximum_cells=maximum_cells, seed=seed)
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    archive.sdk_calls = 1
    _record_root(archive, env, frame)
    excursions = []
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
                action = archive.choose_action(
                    source_cell,
                    _grounded_actions(env),
                    shield=shield,
                )
                if action is None:
                    source_cell.blocked = True
                    reason = "no_shield_allowed_action" if shield else "no_legal_action"
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
    return BurstRun(archive=archive, excursions=tuple(excursions)), shield


def _aggregate_gate(
    *,
    protocol: LineageShieldProtocol,
    source_registry: Mapping[str, Any],
    source_shield: ProgressProtectedTerminalShield,
    witness_trials: Sequence[WitnessShieldTrial],
    conditions: Sequence[Mapping[str, Any]],
    sdk_calls: int,
) -> tuple[bool, dict[str, Any]]:
    per_seed = []
    control_terminal = 0
    treatment_terminal = 0
    control_actions = 0
    treatment_actions = 0
    control_cells = 0
    treatment_cells = 0
    control_sdk = 0
    treatment_sdk = 0
    control_progress = 0
    treatment_progress = 0
    terminal_regression_seeds = 0
    progress_regression_seeds = 0
    vetoes = 0
    minimum_replay_exact_rate = 1.0
    minimum_per_seed_coverage_ratio = float("inf")
    lineage_attached = 0
    rebases_avoided = 0
    lineage_rebased = 0
    for condition in conditions:
        arms = dict(condition["arms"])
        control = dict(arms["lineage_control"])
        treatment = dict(arms["lineage_terminal_shield"])
        control_metrics = dict(control["metrics"])
        treatment_metrics = dict(treatment["metrics"])
        c_actions = int(control_metrics["exploration_actions"])
        t_actions = int(treatment_metrics["exploration_actions"])
        c_terminal = int(control_metrics["terminal_edges"])
        t_terminal = int(treatment_metrics["terminal_edges"])
        c_rate = c_terminal / max(1, c_actions)
        t_rate = t_terminal / max(1, t_actions)
        c_progress = int(control_metrics["progress_edges"])
        t_progress = int(treatment_metrics["progress_edges"])
        terminal_regression_seeds += int(t_rate > c_rate)
        progress_regression_seeds += int(t_progress < c_progress)
        control_terminal += c_terminal
        treatment_terminal += t_terminal
        control_actions += c_actions
        treatment_actions += t_actions
        control_cells += int(control_metrics["symbolic_cells"])
        treatment_cells += int(treatment_metrics["symbolic_cells"])
        control_sdk += int(control_metrics["sdk_calls"])
        treatment_sdk += int(treatment_metrics["sdk_calls"])
        control_progress += c_progress
        treatment_progress += t_progress
        vetoes += int(treatment["shield_metrics"].get("vetoes", 0))
        minimum_replay_exact_rate = min(
            minimum_replay_exact_rate,
            float(control_metrics["replay_exact_rate"]),
            float(treatment_metrics["replay_exact_rate"]),
        )
        coverage_ratio = float(
            treatment_metrics["symbolic_cells_per_1000_sdk_calls"]
        ) / max(
            1e-12,
            float(control_metrics["symbolic_cells_per_1000_sdk_calls"]),
        )
        minimum_per_seed_coverage_ratio = min(
            minimum_per_seed_coverage_ratio, coverage_ratio
        )
        for metrics in (control_metrics, treatment_metrics):
            lineage_attached += int(metrics["lineage_attached_transitions"])
            rebases_avoided += int(metrics["shortest_prefix_rebases_avoided"])
            lineage_rebased += int(metrics["lineage_rebased_transitions"])
        per_seed.append(
            {
                "seed": int(condition["seed"]),
                "control_terminal_rate": c_rate,
                "treatment_terminal_rate": t_rate,
                "control_progress": c_progress,
                "treatment_progress": t_progress,
                "control_coverage": float(
                    control_metrics["symbolic_cells_per_1000_sdk_calls"]
                ),
                "treatment_coverage": float(
                    treatment_metrics["symbolic_cells_per_1000_sdk_calls"]
                ),
                "coverage_ratio": coverage_ratio,
                "control_replay_exact_rate": float(
                    control_metrics["replay_exact_rate"]
                ),
                "treatment_replay_exact_rate": float(
                    treatment_metrics["replay_exact_rate"]
                ),
                "vetoes": int(treatment["shield_metrics"].get("vetoes", 0)),
            }
        )
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
    control_coverage = 1000.0 * control_cells / max(1, control_sdk)
    treatment_coverage = 1000.0 * treatment_cells / max(1, treatment_sdk)
    coverage_ratio = treatment_coverage / max(1e-12, control_coverage)
    witness_confirmations = sum(item.confirmed for item in witness_trials)
    expected_witness_confirmations = (
        protocol.expected_witnesses * protocol.witness_repetitions
    )
    source_shield_metrics = source_shield.metrics()
    metrics = {
        "source_terminal_candidates": len(
            source_registry["terminal_candidate_ids"]
        ),
        "source_terminal_confirmation_rate": 1.0,
        "source_shield": source_shield_metrics,
        "witness_confirmations": witness_confirmations,
        "expected_witness_confirmations": expected_witness_confirmations,
        "witness_vetoes": sum(item.vetoed_actions for item in witness_trials),
        "all_witness_actions_protected": all(
            item.all_actions_protected for item in witness_trials
        ),
        "control_terminal_rate": control_terminal_rate,
        "treatment_terminal_rate": treatment_terminal_rate,
        "terminal_rate_ratio": terminal_rate_ratio,
        "terminal_regression_seeds": terminal_regression_seeds,
        "progress_regression_seeds": progress_regression_seeds,
        "control_coverage": control_coverage,
        "treatment_coverage": treatment_coverage,
        "coverage_ratio": coverage_ratio,
        "minimum_per_seed_coverage_ratio": minimum_per_seed_coverage_ratio,
        "control_progress_edges": control_progress,
        "treatment_progress_edges": treatment_progress,
        "evaluation_vetoes": vetoes,
        "minimum_evaluation_replay_exact_rate": minimum_replay_exact_rate,
        "lineage_attached_transitions": lineage_attached,
        "shortest_prefix_rebases_avoided": rebases_avoided,
        "lineage_rebased_transitions": lineage_rebased,
        "sdk_calls": sdk_calls,
        "maximum_total_sdk_calls": protocol.maximum_total_sdk_calls,
        "per_seed": per_seed,
    }
    passed = bool(
        metrics["source_terminal_candidates"]
        == protocol.expected_terminal_candidates
        and source_shield_metrics["confirmed_terminal_traces"]
        == protocol.expected_terminal_candidates
        and source_shield_metrics["multi_step_hazard_observed"]
        and source_shield_metrics["confirmed_unsafe_actions"] >= 1
        and source_shield_metrics["protected_action_pairs"]
        == protocol.expected_protected_action_pairs
        and witness_confirmations == expected_witness_confirmations
        and metrics["witness_vetoes"] == 0
        and metrics["all_witness_actions_protected"]
        and vetoes >= protocol.minimum_vetoes
        and terminal_rate_ratio <= protocol.maximum_terminal_rate_ratio
        and terminal_regression_seeds <= protocol.maximum_terminal_regression_seeds
        and coverage_ratio >= protocol.minimum_coverage_ratio
        and minimum_per_seed_coverage_ratio >= protocol.minimum_coverage_ratio
        and treatment_progress >= control_progress
        and progress_regression_seeds
        <= protocol.maximum_progress_regression_seeds
        and minimum_replay_exact_rate
        >= protocol.minimum_evaluation_replay_exact_rate
        and lineage_attached > 0
        and rebases_avoided >= protocol.minimum_rebases_avoided
        and lineage_rebased == 0
        and sdk_calls <= protocol.maximum_total_sdk_calls
    )
    return passed, metrics


def _resolve_manifest_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(__file__).resolve().parents[3] / candidate


def run_lineage_shield_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_lineage_shield_manifest(
        manifest_path, verify_code=env_factory is None
    )
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.3e run requires a clean scientific freeze")
    if not manifest.get("firewall", {}).get(
        "lineage_shield_experiment_authorized", False
    ):
        raise ValueError("T12.3e lineage-shield experiment is not authorized")
    protocol = LineageShieldProtocol(**dict(manifest["protocol"]))
    source_registry = load_lineage_shield_registry(
        _resolve_manifest_path(str(manifest["source_registry"]["path"])),
        protocol=protocol,
    )
    shield_payload = _read_json(
        _resolve_manifest_path(str(source_registry["terminal_shield"]["path"]))
    )
    source_shield = ProgressProtectedTerminalShield.from_dict(shield_payload)
    _, witnesses = load_witness_registry(
        _resolve_manifest_path(
            str(manifest["witness_source_t12_3a"]["registry"]["path"])
        )
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)

    witness_trials = []
    for witness in witnesses:
        for repetition in range(protocol.witness_repetitions):
            witness_trials.append(
                replay_witness_with_shield(
                    witness=witness,
                    shield_payload=shield_payload,
                    repetition=repetition,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
            )

    conditions = []
    archive_artifacts = []
    game_id = str(manifest["game_id"])
    for seed in protocol.evaluation_seeds:
        arms = {}
        for arm in protocol.evaluation_arms:
            arm_shield = (
                None
                if arm == "lineage_control"
                else ProgressProtectedTerminalShield.from_dict(shield_payload)
            )
            run, used_shield = run_lineage_shield_arm(
                game_id=game_id,
                seed=seed,
                sdk_call_budget=protocol.sdk_calls_per_evaluation_arm,
                burst_schedule=protocol.burst_schedule,
                environments_dir=environments_dir,
                env_factory=env_factory,
                shield=arm_shield,
                maximum_cells=protocol.maximum_cells,
            )
            archive_path = destination / game_id / str(seed) / f"{arm}.json"
            artifact = _write_archive(
                archive_path, run.archive, storage_budget=storage
            )
            artifact.update({"game_id": game_id, "seed": seed, "arm": arm})
            archive_artifacts.append(artifact)
            excursions_path = (
                destination / game_id / str(seed) / f"{arm}_excursions.json"
            )
            _write_json_once(
                excursions_path,
                {
                    "format_version": "sage-t12.3e-lineage-shield-excursions-v1",
                    "game_id": game_id,
                    "seed": seed,
                    "arm": arm,
                    "excursions": [item.to_dict() for item in run.excursions],
                },
                storage_budget=storage,
            )
            arms[arm] = {
                "metrics": run.metrics(),
                "shield_metrics": (
                    {} if used_shield is None else used_shield.metrics()
                ),
                "archive": artifact,
                "excursions": {
                    "path": str(excursions_path.resolve()),
                    "sha256": _file_sha256(excursions_path),
                },
            }
        conditions.append({"game_id": game_id, "seed": seed, "arms": arms})

    sdk_calls = sum(item.calls for item in witness_trials) + sum(
        int(arm["metrics"]["sdk_calls"])
        for condition in conditions
        for arm in condition["arms"].values()
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        source_registry=source_registry,
        source_shield=source_shield,
        witness_trials=witness_trials,
        conditions=conditions,
        sdk_calls=sdk_calls,
    )
    witness_path = destination / "witness_non_regression.json"
    _write_json_once(
        witness_path,
        {
            "format_version": "sage-t12.3e-witness-non-regression-v1",
            "trials": [item.to_dict() for item in witness_trials],
        },
        storage_budget=storage,
    )
    evaluation_path = destination / "paired_evaluation.json"
    _write_json_once(
        evaluation_path,
        {
            "format_version": "sage-t12.3e-paired-lineage-shield-evaluation-v1",
            "conditions": conditions,
            "archives": archive_artifacts,
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_3E_LINEAGE_SHIELD_GATE"
        if passed
        else "FAIL_T12_3E_LINEAGE_SHIELD_GATE"
    )
    report = {
        "format_version": "sage-t12.3e-lineage-shield-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3d_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "source_t12_3b_receipt_checksum": manifest["source_t12_3b"][
            "receipt"
        ]["receipt_checksum"],
        "metrics": metrics,
        "conditions": conditions,
        "storage": storage.snapshot(),
    }
    report_path = destination / "lineage_shield_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = lineage_shield_phase_receipt(
        manifest=manifest,
        phase="lineage_shield",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "witness_non_regression": {
                "path": str(witness_path.resolve()),
                "sha256": _file_sha256(witness_path),
            },
            "paired_evaluation": {
                "path": str(evaluation_path.resolve()),
                "sha256": _file_sha256(evaluation_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "source_registry": dict(manifest["source_registry"]),
        },
    )
    _write_json_once(
        destination / "lineage_shield_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def lineage_shield_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_lineage_shield_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_lineage_shield_receipt(receipt_path, manifest=manifest)
    )
    passed = bool(
        manifest.get("scientific_claims_authorized", False)
        and receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_3E_LINEAGE_SHIELD_GATE"
    )
    return {
        "format_version": "sage-t12.3e-lineage-shield-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3d_status": manifest["parent"]["receipt"]["status"],
        "source_t12_3b_status": manifest["source_t12_3b"]["receipt"]["status"],
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
        "next_phase_authorized": passed,
        "firewall": {
            **dict(manifest["firewall"]),
            "t12_4_freeze_authorized": passed,
        },
    }


__all__ = [
    "lineage_shield_experiment_status",
    "run_lineage_shield_arm",
    "run_lineage_shield_experiment",
]
