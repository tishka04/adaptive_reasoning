"""T12.3c stepwise replay audit and paired lineage-preserving evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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

from .archive import GoExploreArchive
from .burst_experiment import BurstExcursion, BurstRun
from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import (
    _grounded_actions,
    _make_env,
    _record_root,
    _restore_variant,
    _symbolic_state,
    _write_archive,
)
from .lineage_archive import LineagePreservingArchive
from .lineage_protocol import (
    ReplayAuditCase,
    ReplayLineageProtocol,
    lineage_phase_receipt,
    load_lineage_manifest,
    load_lineage_receipt,
    load_lineage_registry,
)

EnvFactory = Callable[[str], Any]


@dataclass(frozen=True)
class ReplayAuditTrial:
    case_id: str
    case_kind: str
    repetition: int
    exact: bool
    calls: int
    first_divergence_step: int | None
    first_divergence_kind: str
    expected_hash: str
    observed_hash: str
    events: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_kind": self.case_kind,
            "repetition": self.repetition,
            "exact": self.exact,
            "calls": self.calls,
            "first_divergence_step": self.first_divergence_step,
            "first_divergence_kind": self.first_divergence_kind,
            "expected_hash": self.expected_hash,
            "observed_hash": self.observed_hash,
            "events": [dict(item) for item in self.events],
        }


def replay_audit_case(
    *,
    case: ReplayAuditCase,
    repetition: int,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> ReplayAuditTrial:
    env = _make_env(case.game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    observed = state_signature_from_frame(frame)
    expected = case.expected_hashes[0]
    divergence_step: int | None = None
    divergence_kind = ""
    events: list[dict[str, Any]] = [
        {
            "step": 0,
            "kind": "reset",
            "expected_hash": expected,
            "observed_hash": observed,
            "exact": observed == expected,
        }
    ]
    if observed != expected:
        divergence_step = 0
        divergence_kind = "reset_hash"
    else:
        for index, action in enumerate(case.actions, start=1):
            selected = select_live_action(
                env,
                action.action_name,
                action_args=action.action_data,
            )
            expected = case.expected_hashes[index]
            if selected is None:
                divergence_step = index
                divergence_kind = "action_unavailable"
                observed = state_signature_from_frame(frame)
                events.append(
                    {
                        "step": index,
                        "kind": "action_unavailable",
                        "action_key": action.key,
                        "expected_hash": expected,
                        "observed_hash": observed,
                        "exact": False,
                    }
                )
                break
            frame = _step_env_action(env, selected)
            calls += 1
            observed = state_signature_from_frame(frame)
            exact = observed == expected
            events.append(
                {
                    "step": index,
                    "kind": "action",
                    "action_key": action.key,
                    "expected_hash": expected,
                    "observed_hash": observed,
                    "exact": exact,
                }
            )
            if not exact:
                divergence_step = index
                divergence_kind = "state_hash"
                break
    return ReplayAuditTrial(
        case_id=case.case_id,
        case_kind=case.case_kind,
        repetition=int(repetition),
        exact=divergence_step is None,
        calls=calls,
        first_divergence_step=divergence_step,
        first_divergence_kind=divergence_kind,
        expected_hash=expected,
        observed_hash=observed,
        events=tuple(events),
    )


def run_lineage_burst_arm(
    *,
    game_id: str,
    seed: int,
    sdk_call_budget: int,
    burst_schedule: tuple[int, ...],
    preserve_lineage: bool,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
    maximum_cells: int = 50_000,
) -> BurstRun:
    if tuple(int(value) for value in burst_schedule) != (4, 8, 16):
        raise ValueError("T12.3c burst runner requires the 4/8/16 schedule")
    archive: GoExploreArchive
    if preserve_lineage:
        archive = LineagePreservingArchive(maximum_cells=maximum_cells, seed=seed)
    else:
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
                action = archive.choose_action(source_cell, _grounded_actions(env))
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
                    0, int(after.levels_completed) - int(before.levels_completed)
                )
                success = bool(
                    level_delta > 0
                    or str(after.game_state).upper() in {"WIN", "WON", "VICTORY"}
                )
                terminal = _is_terminal(after.game_state)
                common = {
                    "source_cell_id": source_cell.cell_id,
                    "source_exact_hash": source_exact_hash,
                    "action": action,
                    "target_state": _symbolic_state(after_frame),
                    "target_exact_hash": target_hash,
                    "target_level": int(after.levels_completed),
                    "target_legal_actions": _grounded_actions(env),
                    "terminal": terminal,
                    "success": success,
                    "changed": source_exact_hash != target_hash,
                }
                if preserve_lineage:
                    lineage_archive = archive
                    if not isinstance(lineage_archive, LineagePreservingArchive):
                        raise TypeError("lineage arm lacks its lineage archive")
                    edge = lineage_archive.add_lineage_transition(
                        **common,
                        source_prefix_id=executed_prefix_id,
                        source_path_edge_ids=executed_path_edge_ids,
                    )
                    executed_prefix_id = edge.prefix_id
                    executed_path_edge_ids = executed_path_edge_ids + (edge.edge_id,)
                else:
                    edge = archive.add_transition(**common)
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


def _aggregate_gate(
    *,
    protocol: ReplayLineageProtocol,
    audit_trials: Sequence[ReplayAuditTrial],
    conditions: Sequence[Mapping[str, Any]],
    sdk_calls: int,
) -> tuple[bool, dict[str, Any]]:
    failed_case_ids = {
        item.case_id for item in audit_trials if item.case_kind == "failed"
    }
    reproduced = {
        item.case_id
        for item in audit_trials
        if item.case_kind == "failed" and not item.exact
    }
    matched = [item for item in audit_trials if item.case_kind == "matched_control"]
    matched_exact_rate = (
        0.0 if not matched else sum(item.exact for item in matched) / len(matched)
    )
    divergence_steps = Counter(
        item.first_divergence_step
        for item in audit_trials
        if item.first_divergence_step is not None
    )
    per_seed = []
    minimum_treatment_exact = 1.0
    coverage_ratios = []
    progress_regressions = 0
    lineage_attached = 0
    rebases_avoided = 0
    lineage_rebased = 0
    calibration_gain = float("-inf")
    for condition in conditions:
        seed = int(condition["seed"])
        arms = dict(condition["arms"])
        control = dict(arms["shortest_prefix_control"]["metrics"])
        treatment = dict(arms["lineage_preserving"]["metrics"])
        control_exact = float(control["replay_exact_rate"])
        treatment_exact = float(treatment["replay_exact_rate"])
        control_coverage = float(control["symbolic_cells_per_1000_sdk_calls"])
        treatment_coverage = float(treatment["symbolic_cells_per_1000_sdk_calls"])
        coverage_ratio = (
            1.0 if control_coverage <= 0.0 else treatment_coverage / control_coverage
        )
        control_progress = int(control["progress_edges"])
        treatment_progress = int(treatment["progress_edges"])
        minimum_treatment_exact = min(minimum_treatment_exact, treatment_exact)
        coverage_ratios.append(coverage_ratio)
        progress_regressions += int(treatment_progress < control_progress)
        lineage_attached += int(treatment["lineage_attached_transitions"])
        rebases_avoided += int(treatment["shortest_prefix_rebases_avoided"])
        lineage_rebased += int(treatment["lineage_rebased_transitions"])
        gain = treatment_exact - control_exact
        if seed == 6803:
            calibration_gain = gain
        per_seed.append(
            {
                "seed": seed,
                "control_replay_exact_rate": control_exact,
                "treatment_replay_exact_rate": treatment_exact,
                "replay_gain": gain,
                "control_coverage": control_coverage,
                "treatment_coverage": treatment_coverage,
                "coverage_ratio": coverage_ratio,
                "control_progress": control_progress,
                "treatment_progress": treatment_progress,
            }
        )
    minimum_coverage = min(coverage_ratios, default=0.0)
    metrics = {
        "audit_failed_cases": len(failed_case_ids),
        "reproduced_parent_failures": len(reproduced),
        "matched_control_exact_rate": matched_exact_rate,
        "first_divergence_steps": {
            str(key): value for key, value in sorted(divergence_steps.items())
        },
        "minimum_treatment_replay_exact_rate": minimum_treatment_exact,
        "calibration_seed_replay_gain": calibration_gain,
        "minimum_per_seed_coverage_ratio": minimum_coverage,
        "progress_regression_seeds": progress_regressions,
        "lineage_attached_transitions": lineage_attached,
        "shortest_prefix_rebases_avoided": rebases_avoided,
        "lineage_rebased_transitions": lineage_rebased,
        "sdk_calls": sdk_calls,
        "maximum_total_sdk_calls": protocol.maximum_total_sdk_calls,
        "per_seed": per_seed,
    }
    passed = bool(
        len(reproduced) >= protocol.minimum_reproduced_parent_failures
        and matched_exact_rate >= protocol.minimum_treatment_replay_exact_rate
        and minimum_treatment_exact >= protocol.minimum_treatment_replay_exact_rate
        and calibration_gain >= protocol.minimum_calibration_seed_replay_gain
        and minimum_coverage >= protocol.minimum_coverage_ratio
        and progress_regressions <= protocol.maximum_progress_regression_seeds
        and lineage_attached > 0
        and rebases_avoided > 0
        and lineage_rebased == 0
        and sdk_calls <= protocol.maximum_total_sdk_calls
    )
    return passed, metrics


def _registry_path(manifest: Mapping[str, Any]) -> Path:
    path = Path(str(manifest["audit_registry"]["path"]))
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def run_lineage_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_lineage_manifest(
        manifest_path, verify_code=env_factory is None
    )
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.3c run requires a clean scientific freeze")
    if not manifest.get("firewall", {}).get(
        "replay_lineage_experiment_authorized", False
    ):
        raise ValueError("T12.3c replay-lineage experiment is not authorized")
    protocol = ReplayLineageProtocol(**dict(manifest["protocol"]))
    _, cases = load_lineage_registry(_registry_path(manifest), protocol=protocol)
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)

    audit_trials = []
    for case in cases:
        for repetition in range(protocol.audit_repetitions):
            audit_trials.append(
                replay_audit_case(
                    case=case,
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
            run = run_lineage_burst_arm(
                game_id=game_id,
                seed=seed,
                sdk_call_budget=protocol.sdk_calls_per_evaluation_arm,
                burst_schedule=protocol.burst_schedule,
                preserve_lineage=arm == "lineage_preserving",
                environments_dir=environments_dir,
                env_factory=env_factory,
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
                    "format_version": "sage-t12.3c-lineage-excursions-v1",
                    "game_id": game_id,
                    "seed": seed,
                    "arm": arm,
                    "excursions": [item.to_dict() for item in run.excursions],
                },
                storage_budget=storage,
            )
            arms[arm] = {
                "metrics": run.metrics(),
                "archive": artifact,
                "excursions": {
                    "path": str(excursions_path.resolve()),
                    "sha256": _file_sha256(excursions_path),
                },
            }
        conditions.append({"game_id": game_id, "seed": seed, "arms": arms})

    sdk_calls = sum(item.calls for item in audit_trials) + sum(
        int(arm["metrics"]["sdk_calls"])
        for condition in conditions
        for arm in condition["arms"].values()
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        audit_trials=audit_trials,
        conditions=conditions,
        sdk_calls=sdk_calls,
    )
    audit_path = destination / "stepwise_replay_audit.json"
    _write_json_once(
        audit_path,
        {
            "format_version": "sage-t12.3c-stepwise-replay-audit-v1",
            "trials": [item.to_dict() for item in audit_trials],
        },
        storage_budget=storage,
    )
    evaluation_path = destination / "paired_evaluation.json"
    _write_json_once(
        evaluation_path,
        {
            "format_version": "sage-t12.3c-paired-lineage-evaluation-v1",
            "conditions": conditions,
            "archives": archive_artifacts,
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_3C_REPLAY_LINEAGE_GATE"
        if passed
        else "FAIL_T12_3C_REPLAY_LINEAGE_GATE"
    )
    report = {
        "format_version": "sage-t12.3c-replay-lineage-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3b_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "metrics": metrics,
        "conditions": conditions,
        "storage": storage.snapshot(),
    }
    report_path = destination / "lineage_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = lineage_phase_receipt(
        manifest=manifest,
        phase="replay_lineage",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "stepwise_replay_audit": {
                "path": str(audit_path.resolve()),
                "sha256": _file_sha256(audit_path),
            },
            "paired_evaluation": {
                "path": str(evaluation_path.resolve()),
                "sha256": _file_sha256(evaluation_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "audit_registry": dict(manifest["audit_registry"]),
        },
    )
    _write_json_once(
        destination / "lineage_receipt.json", receipt, storage_budget=storage
    )
    return report


def lineage_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_lineage_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_lineage_receipt(receipt_path, manifest=manifest)
    )
    passed = bool(
        manifest.get("scientific_claims_authorized", False)
        and receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_3C_REPLAY_LINEAGE_GATE"
    )
    return {
        "format_version": "sage-t12.3c-replay-lineage-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3b_status": manifest["parent"]["receipt"]["status"],
        "parent_failure_class": manifest["parent"]["receipt"]["failure_class"],
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
            "t12_3b_child_rerun_authorized": passed,
        },
    }


__all__ = [
    "ReplayAuditTrial",
    "lineage_experiment_status",
    "replay_audit_case",
    "run_lineage_burst_arm",
    "run_lineage_experiment",
]
