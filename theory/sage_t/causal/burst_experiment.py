"""Paired SAGE.T12.2 one-step versus burst symbolic exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from theory.unified_cognition_ab_benchmark import _is_terminal

from .archive import GoExploreArchive
from .burst_protocol import (
    BurstExploreProtocol,
    burst_phase_receipt,
    load_burst_manifest,
    load_burst_receipt,
)
from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import (
    _grounded_actions,
    _intervention_bundles,
    _make_env,
    _record_root,
    _restore_variant,
    _symbolic_state,
    _write_archive,
    run_go_explore_arm,
)

EnvFactory = Callable[[str], Any]


@dataclass(frozen=True)
class BurstExcursion:
    excursion_index: int
    requested_horizon: int
    executed_actions: int
    restoration_calls: int
    exact_restoration: bool
    start_cell_id: str
    start_exact_hash: str
    stopped_reason: str
    progress_edges: int
    terminal_failures: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "excursion_index": self.excursion_index,
            "requested_horizon": self.requested_horizon,
            "executed_actions": self.executed_actions,
            "restoration_calls": self.restoration_calls,
            "exact_restoration": self.exact_restoration,
            "start_cell_id": self.start_cell_id,
            "start_exact_hash": self.start_exact_hash,
            "stopped_reason": self.stopped_reason,
            "progress_edges": self.progress_edges,
            "terminal_failures": self.terminal_failures,
        }


@dataclass(frozen=True)
class BurstRun:
    archive: GoExploreArchive
    excursions: tuple[BurstExcursion, ...]

    def metrics(self) -> dict[str, Any]:
        archive_metrics = self.archive.metrics()
        edges = int(archive_metrics["edges"])
        sdk_calls = int(archive_metrics["sdk_calls"])
        return {
            **archive_metrics,
            "excursions": len(self.excursions),
            "exploration_actions": edges,
            "exploration_action_fraction": (
                0.0 if sdk_calls <= 0 else edges / sdk_calls
            ),
            "mean_actions_per_excursion": (
                0.0
                if not self.excursions
                else sum(item.executed_actions for item in self.excursions)
                / len(self.excursions)
            ),
            "completed_bursts": sum(
                item.executed_actions == item.requested_horizon
                for item in self.excursions
            ),
            "progress_excursions": sum(
                item.progress_edges > 0 for item in self.excursions
            ),
        }


def run_burst_go_explore_arm(
    *,
    game_id: str,
    seed: int,
    sdk_call_budget: int,
    burst_schedule: tuple[int, ...] = (4, 8, 16),
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    maximum_cells: int = 50_000,
) -> BurstRun:
    """Restore once, then archive every state in a bounded action burst."""

    if tuple(int(value) for value in burst_schedule) != (4, 8, 16):
        raise ValueError("burst runner requires the preregistered 4/8/16 schedule")
    archive = GoExploreArchive(maximum_cells=maximum_cells, seed=seed)
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
            updated = replace(
                variant, replay_failures=variant.replay_failures + 1
            )
            cell.variants[variant.exact_hash] = updated
            if updated.replay_failures >= 2:
                cell.blocked = True
            reason = "restore_mismatch"
        else:
            source_cell = cell
            source_exact_hash = variant.exact_hash
            for _ in range(horizon):
                if archive.sdk_calls >= sdk_call_budget:
                    reason = "sdk_budget"
                    break
                before = snapshot_frame(frame)
                if _is_terminal(before.game_state):
                    reason = "terminal_source"
                    break
                candidates = _grounded_actions(env)
                action = archive.choose_action(source_cell, candidates)
                if action is None:
                    source_cell.blocked = True
                    reason = "no_legal_action"
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
                    0,
                    int(after.levels_completed) - int(before.levels_completed),
                )
                success = bool(
                    level_delta > 0
                    or str(after.game_state).upper() in {"WIN", "WON", "VICTORY"}
                )
                terminal = _is_terminal(after.game_state)
                edge = archive.add_transition(
                    source_cell_id=source_cell.cell_id,
                    source_exact_hash=source_exact_hash,
                    action=action,
                    target_state=_symbolic_state(after_frame),
                    target_exact_hash=target_hash,
                    target_level=int(after.levels_completed),
                    target_legal_actions=_grounded_actions(env),
                    terminal=terminal,
                    success=success,
                    changed=source_exact_hash != target_hash,
                )
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
    return BurstRun(archive=archive, excursions=tuple(excursions))


def _arm_metrics(archive: GoExploreArchive) -> dict[str, Any]:
    metrics = archive.metrics()
    edges = int(metrics["edges"])
    sdk_calls = int(metrics["sdk_calls"])
    return {
        **metrics,
        "exploration_actions": edges,
        "exploration_action_fraction": 0.0 if sdk_calls <= 0 else edges / sdk_calls,
        "excursions": int(metrics["replay_attempts"]),
        "mean_actions_per_excursion": (
            0.0 if metrics["replay_attempts"] <= 0 else edges / metrics["replay_attempts"]
        ),
    }


def _paired_gate(conditions: list[Mapping[str, Any]], protocol: BurstExploreProtocol):
    per_seed = []
    for condition in conditions:
        control = condition["arms"]["one_step_archive"]["metrics"]
        treatment = condition["arms"]["burst_archive"]["metrics"]
        control_fraction = float(control["exploration_action_fraction"])
        treatment_fraction = float(treatment["exploration_action_fraction"])
        control_coverage = float(control["symbolic_cells_per_1000_sdk_calls"])
        treatment_coverage = float(treatment["symbolic_cells_per_1000_sdk_calls"])
        per_seed.append(
            {
                "seed": condition["seed"],
                "control_action_fraction": control_fraction,
                "burst_action_fraction": treatment_fraction,
                "action_efficiency_ratio": (
                    math.inf
                    if control_fraction <= 0.0 and treatment_fraction > 0.0
                    else 0.0
                    if control_fraction <= 0.0
                    else treatment_fraction / control_fraction
                ),
                "control_coverage": control_coverage,
                "burst_coverage": treatment_coverage,
                "relative_coverage_gain": (
                    math.inf
                    if control_coverage <= 0.0 and treatment_coverage > 0.0
                    else 0.0
                    if control_coverage <= 0.0
                    else treatment_coverage / control_coverage - 1.0
                ),
                "control_progress": int(control["progress_edges"]),
                "burst_progress": int(treatment["progress_edges"]),
                "control_terminal": int(control["terminal_edges"]),
                "burst_terminal": int(treatment["terminal_edges"]),
                "burst_exploration_actions": int(treatment["exploration_actions"]),
            }
        )
    mean_control_fraction = sum(row["control_action_fraction"] for row in per_seed) / len(per_seed)
    mean_burst_fraction = sum(row["burst_action_fraction"] for row in per_seed) / len(per_seed)
    mean_control_coverage = sum(row["control_coverage"] for row in per_seed) / len(per_seed)
    mean_burst_coverage = sum(row["burst_coverage"] for row in per_seed) / len(per_seed)
    total_burst_actions = sum(row["burst_exploration_actions"] for row in per_seed)
    total_burst_terminal = sum(row["burst_terminal"] for row in per_seed)
    metrics = {
        "per_seed": per_seed,
        "aggregate_action_efficiency_ratio": (
            math.inf
            if mean_control_fraction <= 0.0 and mean_burst_fraction > 0.0
            else 0.0
            if mean_control_fraction <= 0.0
            else mean_burst_fraction / mean_control_fraction
        ),
        "aggregate_relative_coverage_gain": (
            math.inf
            if mean_control_coverage <= 0.0 and mean_burst_coverage > 0.0
            else 0.0
            if mean_control_coverage <= 0.0
            else mean_burst_coverage / mean_control_coverage - 1.0
        ),
        "positive_action_efficiency_every_seed": all(
            row["action_efficiency_ratio"] > 1.0 for row in per_seed
        ),
        "burst_progress_edges": sum(row["burst_progress"] for row in per_seed),
        "burst_terminal_failure_rate": (
            0.0
            if total_burst_actions <= 0
            else total_burst_terminal / total_burst_actions
        ),
        "exact_replay_all_arms": all(
            float(arm["metrics"]["replay_exact_rate"]) == 1.0
            for condition in conditions
            for arm in condition["arms"].values()
        ),
    }
    passed = bool(
        metrics["aggregate_action_efficiency_ratio"]
        >= protocol.minimum_exploration_action_ratio
        and metrics["aggregate_relative_coverage_gain"]
        >= protocol.minimum_relative_coverage_gain
        and metrics["positive_action_efficiency_every_seed"]
        and metrics["burst_progress_edges"] >= protocol.minimum_progress_edges
        and metrics["burst_terminal_failure_rate"]
        <= protocol.maximum_terminal_failure_rate
        and metrics["exact_replay_all_arms"]
    )
    return passed, metrics


def run_burst_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_burst_manifest(
        manifest_path, verify_code=env_factory is None
    )
    protocol = BurstExploreProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    artifacts = []
    bundles = []
    game_id = str(manifest["game_id"])
    for seed in protocol.seeds:
        one_step, _ = run_go_explore_arm(
            game_id=game_id,
            seed=seed,
            sdk_call_budget=protocol.sdk_call_budget_per_seed_arm,
            environments_dir=environments_dir,
            env_factory=env_factory,
            maximum_cells=protocol.maximum_cells,
        )
        burst = run_burst_go_explore_arm(
            game_id=game_id,
            seed=seed,
            sdk_call_budget=protocol.sdk_call_budget_per_seed_arm,
            burst_schedule=protocol.burst_schedule,
            environments_dir=environments_dir,
            env_factory=env_factory,
            maximum_cells=protocol.maximum_cells,
        )
        arms = {}
        for arm, archive, metrics in (
            ("one_step_archive", one_step, _arm_metrics(one_step)),
            ("burst_archive", burst.archive, burst.metrics()),
        ):
            path = destination / game_id / str(seed) / f"{arm}.json"
            artifact = _write_archive(path, archive, storage_budget=storage)
            artifact.update({"game_id": game_id, "seed": seed, "arm": arm})
            artifacts.append(artifact)
            arms[arm] = {"metrics": metrics, "artifact": artifact}
        excursion_path = destination / game_id / str(seed) / "burst_excursions.json"
        _write_json_once(
            excursion_path,
            {
                "format_version": "sage-t12.2-burst-excursions-v1",
                "game_id": game_id,
                "seed": seed,
                "schedule": list(protocol.burst_schedule),
                "excursions": [item.to_dict() for item in burst.excursions],
            },
            storage_budget=storage,
        )
        excursion_meta = {
            "path": str(excursion_path.resolve()),
            "sha256": _file_sha256(excursion_path),
            "game_id": game_id,
            "seed": seed,
        }
        artifacts.append({**excursion_meta, "arm": "burst_excursions"})
        arms["burst_archive"]["excursions"] = excursion_meta
        bundles.extend(_intervention_bundles(burst.archive, game_id=game_id, seed=seed))
        conditions.append({"game_id": game_id, "seed": seed, "arms": arms})
    passed, metrics = _paired_gate(conditions, protocol)
    bundle_path = destination / "intervention_bundles.json"
    _write_json_once(
        bundle_path,
        {
            "format_version": "sage-t12.2-exact-prefix-bundles-v1",
            "bundles": bundles,
        },
        storage_budget=storage,
    )
    metrics["exact_prefix_intervention_bundles"] = len(bundles)
    report = {
        "format_version": "sage-t12.2-burst-report-v1",
        "status": "PASS_T12_2_BURST_GATE" if passed else "FAIL_T12_2_BURST_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_1_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "conditions": conditions,
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "burst_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = burst_phase_receipt(
        manifest=manifest,
        phase="burst_archive",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        artifacts={
            "archives": artifacts,
            "intervention_bundles": {
                "path": str(bundle_path.resolve()),
                "sha256": _file_sha256(bundle_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(destination / "burst_receipt.json", receipt, storage_budget=storage)
    return report


def burst_experiment_status(
    *, manifest_path: str | Path, receipt_path: str | Path | None = None
) -> dict[str, Any]:
    manifest = load_burst_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_burst_receipt(receipt_path, manifest=manifest)
    )
    return {
        "format_version": "sage-t12.2-burst-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_1_status": manifest["parent"]["receipt"]["status"],
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
        "next_phase_authorized": bool(
            manifest.get("scientific_claims_authorized", False)
            and receipt is not None
            and receipt.get("passed") is True
        ),
        "firewall": dict(manifest["firewall"]),
    }


__all__ = [
    "BurstExcursion",
    "BurstRun",
    "burst_experiment_status",
    "run_burst_experiment",
    "run_burst_go_explore_arm",
]
