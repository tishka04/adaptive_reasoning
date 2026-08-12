"""Executable phased experiment for SAGE.T12.1 symbolic Go-Explore.

Expensive environment access is confined to archive, shield, option extraction,
and transfer phases.  Training and compilation consume immutable artifacts.
"""

from __future__ import annotations

import io
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch

from theory.live_transition_loop import build_observation
from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    _make_real_env,
    select_live_action,
    state_signature_from_frame,
)
from theory.sage_t.compiler import compile_causal_observation
from theory.unified_cognition_ab_benchmark import (
    SharedLegacyProposalPolicy,
    _available_action_names,
    _is_terminal,
    _materialize_decision,
)
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from .adapters import causal_state_from_abstract
from .archive import (
    ArchiveEdge,
    ArchiveStateVariant,
    GoExploreArchive,
    ROOT_PREFIX_ID,
)
from .contracts import (
    CausalProgram,
    CausalState,
    GroundedAction,
    StructuredDelta,
    TransitionEvidence,
    ValueDistribution,
    causal_program_from_dict,
)
from .executor import CausalExecutor
from .experiment import (
    RunStorageBudget,
    _file_sha256,
    _read_json,
    _verify_signed,
    _write_json_once,
)
from .graph_protocol import (
    GraphExploreProtocol,
    load_graph_manifest,
    load_graph_receipt,
    phase_receipt,
)
from .novelty import (
    OnlineNoveltyPredictor,
    brier_score,
    expected_calibration_error,
)
from .options import (
    CausalOptionCompiler,
    CompiledCausalOption,
    MinimalCausalOption,
    MinimalOptionExtractor,
    OptionMechanismRegistry,
    PosteriorOptionProvider,
)
from .posterior import CausalPosterior
from .terminal_shield import MultiStepTerminalShield

EnvFactory = Callable[[str], Any]


def _require_source_train(manifest: Mapping[str, Any], phase: str) -> None:
    if manifest.get("stage") != "source_train":
        raise ValueError(
            f"{phase} is a fitting/adaptation phase and is restricted to source_train"
        )


def _require_receipt_phase(receipt: Mapping[str, Any], phase: str) -> None:
    if receipt.get("phase") != phase:
        raise ValueError(
            f"expected upstream {phase} receipt, observed {receipt.get('phase')}"
        )


def _grounded_actions(env: Any) -> tuple[GroundedAction, ...]:
    output = {}
    for action in _valid_actions(env):
        name = str(getattr(action, "name", "")).strip().upper()
        if not name or name == "RESET":
            continue
        grounded = GroundedAction(
            name,
            dict(getattr(action, "action_args", {}) or {}),
        )
        output[grounded.key] = grounded
    return tuple(output[key] for key in sorted(output))


def _symbolic_state(frame: Any):  # type: ignore[no-untyped-def]
    snapshot = snapshot_frame(frame)
    observation = build_observation(
        snapshot.grid,
        available_actions=snapshot.available_actions,
        game_state=snapshot.game_state,
        levels_completed=snapshot.levels_completed,
    )
    return compile_causal_observation(observation)


def _make_env(game_id: str, environments_dir: str | Path, env_factory: EnvFactory | None):
    return (
        env_factory(game_id)
        if env_factory is not None
        else _make_real_env(game_id, environments_dir)
    )


def _record_root(
    archive: GoExploreArchive,
    env: Any,
    frame: Any,
    *,
    prefix_id: str = ROOT_PREFIX_ID,
    path_edge_ids: Sequence[str] = (),
) -> tuple[str, str]:
    snapshot = snapshot_frame(frame)
    actions = _grounded_actions(env)
    exact_hash = state_signature_from_frame(frame)
    cell, _ = archive.observe_state(
        state=_symbolic_state(frame),
        exact_hash=exact_hash,
        level=int(snapshot.levels_completed),
        legal_actions=actions,
        prefix_id=prefix_id,
        path_edge_ids=path_edge_ids,
        terminal=_is_terminal(snapshot.game_state),
    )
    return cell.cell_id, exact_hash


def _restore_variant(
    *,
    archive: GoExploreArchive,
    variant: ArchiveStateVariant,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> tuple[Any, Any, bool, int]:
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    for action in archive.prefixes.actions(variant.prefix_id):
        selected = select_live_action(
            env,
            action.action_name,
            action_args=action.action_data,
        )
        if selected is None:
            return env, frame, False, calls
        frame = _step_env_action(env, selected)
        calls += 1
    return env, frame, state_signature_from_frame(frame) == variant.exact_hash, calls


def _confirm_terminal_variant(
    *,
    archive: GoExploreArchive,
    variant: ArchiveStateVariant,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> tuple[bool, int]:
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    expected_edges = archive.path_edges(variant)
    actions = archive.prefixes.actions(variant.prefix_id)
    if len(actions) != len(expected_edges):
        return False, calls
    for action, edge in zip(actions, expected_edges):
        selected = select_live_action(
            env,
            action.action_name,
            action_args=action.action_data,
        )
        if selected is None:
            return False, calls
        frame = _step_env_action(env, selected)
        calls += 1
        if state_signature_from_frame(frame) != edge.target_exact_hash:
            return False, calls
    terminal = _is_terminal(snapshot_frame(frame).game_state)
    return bool(terminal and expected_edges and expected_edges[-1].terminal), calls


def _baseline_action(
    *,
    controller: UnifiedCognitiveController,
    policy: SharedLegacyProposalPolicy,
    env: Any,
    frame: Any,
) -> Any | None:
    before = snapshot_frame(frame)
    legal = tuple(_valid_actions(env))
    proposal = policy.select(legal)
    if proposal is None:
        return None
    decision = controller.select_action(
        current_grid=before.grid,
        available_actions=_available_action_names(legal),
        legacy_action=str(getattr(proposal, "name", "")),
        legacy_action_data=dict(getattr(proposal, "action_args", {}) or {}),
        available_action_candidates=legal,
        game_state=before.game_state,
        levels_completed=before.levels_completed,
    )
    return _materialize_decision(legal, decision) or proposal


def _observe_controller_transition(
    controller: UnifiedCognitiveController,
    selected: Any,
    before: Any,
    after: Any,
    legal: Sequence[Any],
) -> None:
    controller.observe_transition(
        action=str(getattr(selected, "name", "")),
        action_data=dict(getattr(selected, "action_args", {}) or {}),
        grid_before=before.grid,
        grid_after=after.grid,
        available_actions=_available_action_names(legal),
        game_state_before=before.game_state,
        game_state_after=after.game_state,
        levels_completed_before=before.levels_completed,
        levels_completed_after=after.levels_completed,
    )
    if int(after.levels_completed) > int(before.levels_completed) and not _is_terminal(
        after.game_state
    ):
        controller.on_level_change()


def run_baseline_arm(
    *,
    game_id: str,
    seed: int,
    sdk_call_budget: int,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
    use_historical_controller: bool = True,
) -> GoExploreArchive:
    archive = GoExploreArchive()
    reset_index = 0
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    archive.sdk_calls = 1
    source_cell_id, source_exact_hash = _record_root(archive, env, frame)
    controller = UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
    )
    policy = SharedLegacyProposalPolicy(
        game_id=game_id,
        seed=seed,
        reset_index=reset_index,
    )
    while archive.sdk_calls < sdk_call_budget:
        before = snapshot_frame(frame)
        if _is_terminal(before.game_state):
            if archive.sdk_calls + 1 > sdk_call_budget:
                break
            reset_index += 1
            env = _make_env(game_id, environments_dir, env_factory)
            frame = _reset_env(env)
            archive.sdk_calls += 1
            source_cell_id, source_exact_hash = _record_root(archive, env, frame)
            controller = UnifiedCognitiveController(
                game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
            )
            policy = SharedLegacyProposalPolicy(
                game_id=game_id,
                seed=seed,
                reset_index=reset_index,
            )
            continue
        legal_raw = tuple(_valid_actions(env))
        selected = (
            _baseline_action(
                controller=controller,
                policy=policy,
                env=env,
                frame=frame,
            )
            if use_historical_controller
            else policy.select(legal_raw)
        )
        if selected is None:
            break
        grounded = GroundedAction(
            str(getattr(selected, "name", "")),
            dict(getattr(selected, "action_args", {}) or {}),
        )
        after_frame = _step_env_action(env, selected)
        archive.sdk_calls += 1
        after = snapshot_frame(
            after_frame,
            fallback_available_actions=before.available_actions,
        )
        target_actions = _grounded_actions(env)
        edge = archive.add_transition(
            source_cell_id=source_cell_id,
            source_exact_hash=source_exact_hash,
            action=grounded,
            target_state=_symbolic_state(after_frame),
            target_exact_hash=state_signature_from_frame(after_frame),
            target_level=int(after.levels_completed),
            target_legal_actions=target_actions,
            terminal=_is_terminal(after.game_state),
            success=(
                int(after.levels_completed) > int(before.levels_completed)
                or str(after.game_state).upper() in {"WIN", "WON", "VICTORY"}
            ),
            changed=source_exact_hash != state_signature_from_frame(after_frame),
        )
        if use_historical_controller:
            _observe_controller_transition(controller, selected, before, after, legal_raw)
        frame = after_frame
        source_cell_id = edge.target_cell_id
        source_exact_hash = edge.target_exact_hash
    return archive


def run_go_explore_arm(
    *,
    game_id: str,
    seed: int = 0,
    sdk_call_budget: int,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
    shield: MultiStepTerminalShield | None = None,
    learn_shield: bool = False,
    apply_shield: bool = True,
    predictor: OnlineNoveltyPredictor | None = None,
    update_predictor: bool = True,
    maximum_cells: int = 50_000,
) -> tuple[GoExploreArchive, MultiStepTerminalShield | None]:
    archive = GoExploreArchive(maximum_cells=maximum_cells, seed=seed)
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    archive.sdk_calls = 1
    _record_root(archive, env, frame)
    while archive.sdk_calls < sdk_call_budget:
        cell = archive.select_cell(
            remaining_sdk_calls=sdk_call_budget - archive.sdk_calls
        )
        if cell is None:
            break
        variant = cell.best_variant(archive.prefixes)
        env, frame, exact, calls = _restore_variant(
            archive=archive,
            variant=variant,
            game_id=game_id,
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        archive.sdk_calls += calls
        archive.note_replay(exact=exact)
        if not exact:
            cell.variants[variant.exact_hash] = replace(
                variant, replay_failures=variant.replay_failures + 1
            )
            if cell.variants[variant.exact_hash].replay_failures >= 2:
                cell.blocked = True
            continue
        candidates = _grounded_actions(env)
        action = archive.choose_action(
            cell,
            candidates,
            shield=shield if apply_shield else None,
            novelty_scorer=predictor,
        )
        if action is None:
            cell.blocked = True
            continue
        selected = select_live_action(
            env,
            action.action_name,
            action_args=action.action_data,
        )
        if selected is None:
            cell.action_attempts[action.key] = cell.action_attempts.get(action.key, 0) + 1
            continue
        if archive.sdk_calls + 1 > sdk_call_budget:
            break
        before_snapshot = snapshot_frame(frame)
        after_frame = _step_env_action(env, selected)
        archive.sdk_calls += 1
        after_snapshot = snapshot_frame(
            after_frame,
            fallback_available_actions=before_snapshot.available_actions,
        )
        target_state = _symbolic_state(after_frame)
        target_exact_hash = state_signature_from_frame(after_frame)
        target_actions = _grounded_actions(env)
        target_key = archive.cell_key(
            target_state,
            level=int(after_snapshot.levels_completed),
            legal_actions=target_actions,
        )
        changed = target_exact_hash != variant.exact_hash
        novel = target_key not in archive.cells
        if predictor is not None:
            predictor.observe(
                cell.state,
                action,
                changed=changed,
                novel=novel,
                update=update_predictor,
            )
        edge = archive.add_transition(
            source_cell_id=cell.cell_id,
            source_exact_hash=variant.exact_hash,
            action=action,
            target_state=target_state,
            target_exact_hash=target_exact_hash,
            target_level=int(after_snapshot.levels_completed),
            target_legal_actions=target_actions,
            terminal=_is_terminal(after_snapshot.game_state),
            success=(
                int(after_snapshot.levels_completed)
                > int(before_snapshot.levels_completed)
                or str(after_snapshot.game_state).upper() in {"WIN", "WON", "VICTORY"}
            ),
            changed=changed,
        )
        if learn_shield and shield is not None and (edge.level_delta > 0 or edge.success):
            shield.observe_progress(edge)
        if learn_shield and shield is not None and edge.terminal and not edge.success:
            target_cell = archive.cells[edge.target_cell_id]
            target_variant = target_cell.variants[edge.target_exact_hash]
            confirmation_cost = 1 + archive.prefixes.depth(target_variant.prefix_id)
            confirmed = False
            if archive.sdk_calls + confirmation_cost <= sdk_call_budget:
                confirmed, confirmation_calls = _confirm_terminal_variant(
                    archive=archive,
                    variant=target_variant,
                    game_id=game_id,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
                archive.sdk_calls += confirmation_calls
            shield.record_terminal_trace(
                archive.path_edges(target_variant),
                exact_replay_confirmed=confirmed,
            )
    return archive, shield


def _write_archive(
    path: Path,
    archive: GoExploreArchive,
    *,
    storage_budget: RunStorageBudget,
) -> dict[str, Any]:
    _write_json_once(path, archive.to_dict(), storage_budget=storage_budget)
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _intervention_bundles(
    archive: GoExploreArchive,
    *,
    game_id: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], dict[str, ArchiveEdge]] = {}
    for edge in sorted(archive.edges.values(), key=lambda item: item.ordinal):
        grouped.setdefault(
            (edge.source_cell_id, edge.source_exact_hash), {}
        ).setdefault(edge.action.key, edge)
    bundles = []
    for (cell_id, exact_hash), by_action in sorted(grouped.items()):
        if len(by_action) < 2:
            continue
        variant = archive.cells[cell_id].variants[exact_hash]
        bundles.append(
            {
                "bundle_id": f"bundle_{game_id}_{seed}_{len(bundles):06d}",
                "game_id": game_id,
                "seed": seed,
                "source_cell_id": cell_id,
                "prefix_hash": exact_hash,
                "prefix": [
                    {
                        "action_name": action.action_name,
                        "action_data": dict(action.action_data),
                    }
                    for action in archive.prefixes.actions(variant.prefix_id)
                ],
                "branches": [
                    {
                        "action_name": edge.action.action_name,
                        "action_data": dict(edge.action.action_data),
                        "target_exact_hash": edge.target_exact_hash,
                        "target_cell_id": edge.target_cell_id,
                        "changed": edge.changed,
                        "novel": edge.novel,
                        "terminal": edge.terminal,
                        "progress": edge.level_delta > 0 or edge.success,
                    }
                    for edge in sorted(by_action.values(), key=lambda item: item.action.key)
                ],
            }
        )
    return tuple(bundles)


def _paired_gate(
    conditions: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str,
    treatment_arm: str,
    minimum_relative_gain: float,
) -> tuple[bool, dict[str, Any]]:
    per_seed = []
    for condition in conditions:
        baseline = condition["arms"][baseline_arm]["metrics"]
        treatment = condition["arms"][treatment_arm]["metrics"]
        base_rate = float(baseline["symbolic_cells_per_1000_sdk_calls"])
        treatment_rate = float(treatment["symbolic_cells_per_1000_sdk_calls"])
        relative = (
            math.inf
            if base_rate <= 0.0 and treatment_rate > 0.0
            else 0.0
            if base_rate <= 0.0
            else treatment_rate / base_rate - 1.0
        )
        per_seed.append(
            {
                "seed": condition["seed"],
                "baseline_rate": base_rate,
                "treatment_rate": treatment_rate,
                "relative_gain": relative,
                "baseline_progress": int(baseline["progress_edges"]),
                "treatment_progress": int(treatment["progress_edges"]),
                "baseline_terminal": int(baseline["terminal_edges"]),
                "treatment_terminal": int(treatment["terminal_edges"]),
            }
        )
    mean_baseline = sum(item["baseline_rate"] for item in per_seed) / len(per_seed)
    mean_treatment = sum(item["treatment_rate"] for item in per_seed) / len(per_seed)
    aggregate_gain = (
        math.inf
        if mean_baseline <= 0.0 and mean_treatment > 0.0
        else 0.0
        if mean_baseline <= 0.0
        else mean_treatment / mean_baseline - 1.0
    )
    metrics = {
        "per_seed": per_seed,
        "aggregate_relative_coverage_gain": aggregate_gain,
        "positive_coverage_gain_every_seed": all(
            item["relative_gain"] > 0.0 for item in per_seed
        ),
        "treatment_progress_edges": sum(item["treatment_progress"] for item in per_seed),
        "safety_regressions": sum(
            item["treatment_terminal"] > item["baseline_terminal"]
            for item in per_seed
        ),
    }
    passed = bool(
        aggregate_gain >= minimum_relative_gain
        and metrics["positive_coverage_gain_every_seed"]
        and metrics["treatment_progress_edges"] >= 1
    )
    return passed, metrics


def run_archive_phase(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    use_historical_controller: bool = True,
) -> dict[str, Any]:
    manifest = load_graph_manifest(
        manifest_path, verify_code=env_factory is None
    )
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    artifacts = []
    intervention_bundles = []
    learned_shield = MultiStepTerminalShield(
        horizon=protocol.terminal_horizon,
        minimum_support=protocol.terminal_minimum_support,
    )
    for game_id in manifest["games"]:
        for seed in protocol.archive_seeds:
            arms: dict[str, Any] = {}
            baseline = run_baseline_arm(
                game_id=str(game_id),
                seed=seed,
                sdk_call_budget=protocol.sdk_call_budget_per_seed_arm,
                environments_dir=environments_dir,
                env_factory=env_factory,
                use_historical_controller=use_historical_controller,
            )
            baseline_path = destination / str(game_id) / str(seed) / "baseline.json"
            artifact = _write_archive(baseline_path, baseline, storage_budget=storage)
            artifact.update({"game_id": game_id, "seed": seed, "arm": "baseline"})
            artifacts.append(artifact)
            arms["baseline"] = {"metrics": baseline.metrics(), "artifact": artifact}
            explorer, _ = run_go_explore_arm(
                game_id=str(game_id),
                seed=seed,
                sdk_call_budget=protocol.sdk_call_budget_per_seed_arm,
                environments_dir=environments_dir,
                env_factory=env_factory,
                shield=learned_shield,
                learn_shield=True,
                apply_shield=False,
                maximum_cells=protocol.maximum_cells,
            )
            explorer_path = (
                destination / str(game_id) / str(seed) / "symbolic_archive.json"
            )
            artifact = _write_archive(explorer_path, explorer, storage_budget=storage)
            artifact.update(
                {"game_id": game_id, "seed": seed, "arm": "symbolic_archive"}
            )
            artifacts.append(artifact)
            intervention_bundles.extend(
                _intervention_bundles(
                    explorer, game_id=str(game_id), seed=int(seed)
                )
            )
            arms["symbolic_archive"] = {
                "metrics": explorer.metrics(),
                "artifact": artifact,
            }
            conditions.append({"game_id": game_id, "seed": seed, "arms": arms})
    passed, gate_metrics = _paired_gate(
        conditions,
        baseline_arm="baseline",
        treatment_arm="symbolic_archive",
        minimum_relative_gain=protocol.archive_minimum_relative_coverage_gain,
    )
    integrity = all(
        arm["metrics"]["replay_exact_rate"] == 1.0
        for condition in conditions
        for name, arm in condition["arms"].items()
        if name == "symbolic_archive"
    )
    passed = bool(passed and integrity)
    shield_path = destination / "terminal_shield.json"
    _write_json_once(shield_path, learned_shield.to_dict(), storage_budget=storage)
    bundle_path = destination / "intervention_bundles.json"
    _write_json_once(
        bundle_path,
        {
            "format_version": "sage-t12.1-exact-prefix-bundles-v1",
            "bundles": intervention_bundles,
        },
        storage_budget=storage,
    )
    report = {
        "format_version": "sage-t12.1-archive-report-v1",
        "status": "PASS_ARCHIVE_GATE" if passed else "FAIL_ARCHIVE_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "conditions": conditions,
        "metrics": {
            **gate_metrics,
            "replay_integrity": integrity,
            "terminal_shield": learned_shield.metrics(),
            "exact_prefix_intervention_bundles": len(intervention_bundles),
        },
        "storage": storage.snapshot(),
    }
    report_path = destination / "archive_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = phase_receipt(
        manifest=manifest,
        phase="archive",
        passed=passed,
        status=report["status"],
        metrics=report["metrics"],
        artifacts={
            "archives": artifacts,
            "shield": {"path": str(shield_path.resolve()), "sha256": _file_sha256(shield_path)},
            "intervention_bundles": {
                "path": str(bundle_path.resolve()),
                "sha256": _file_sha256(bundle_path),
            },
            "report": {"path": str(report_path.resolve()), "sha256": _file_sha256(report_path)},
        },
    )
    _write_json_once(destination / "archive_receipt.json", receipt, storage_budget=storage)
    return report


def _load_artifact(path: str, expected_sha256: str) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file() or _file_sha256(artifact_path) != expected_sha256:
        raise ValueError(f"artifact checksum mismatch: {artifact_path}")
    return _read_json(artifact_path)


def run_shield_phase(
    *,
    manifest_path: str | Path,
    archive_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_graph_manifest(manifest_path, verify_code=env_factory is None)
    _require_source_train(manifest, "shield")
    parent = load_graph_receipt(
        archive_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(parent, "archive")
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    shield_meta = dict(parent["artifacts"]["shield"])
    shield_payload = _load_artifact(shield_meta["path"], shield_meta["sha256"])
    frozen_shield = MultiStepTerminalShield.from_dict(shield_payload)
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    artifacts = []
    for game_id in manifest["games"]:
        for seed in protocol.shield_seeds:
            arms = {}
            for arm in protocol.shield_arms:
                shield = (
                    None
                    if arm == "symbolic_archive"
                    else MultiStepTerminalShield.from_dict(shield_payload)
                )
                archive, used_shield = run_go_explore_arm(
                    game_id=str(game_id),
                    seed=seed,
                    sdk_call_budget=protocol.sdk_call_budget_per_seed_arm,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                    shield=shield,
                    learn_shield=False,
                    maximum_cells=protocol.maximum_cells,
                )
                path = destination / str(game_id) / str(seed) / f"{arm}.json"
                artifact = _write_archive(path, archive, storage_budget=storage)
                artifact.update({"game_id": game_id, "seed": seed, "arm": arm})
                artifacts.append(artifact)
                arms[arm] = {
                    "metrics": archive.metrics(),
                    "shield_metrics": (
                        {} if used_shield is None else used_shield.metrics()
                    ),
                    "artifact": artifact,
                }
            conditions.append({"game_id": game_id, "seed": seed, "arms": arms})
    safety_regressions = 0
    progress_regressions = 0
    total_vetoes = 0
    for condition in conditions:
        control = condition["arms"]["symbolic_archive"]["metrics"]
        treatment = condition["arms"]["archive_shield"]
        safety_regressions += int(
            treatment["metrics"]["terminal_edges"] > control["terminal_edges"]
        )
        progress_regressions += int(
            treatment["metrics"]["progress_edges"] < control["progress_edges"]
        )
        total_vetoes += int(treatment["shield_metrics"].get("vetoes", 0))
    source_metrics = frozen_shield.metrics()
    passed = bool(
        source_metrics["multi_step_hazard_observed"]
        and safety_regressions == 0
        and progress_regressions == 0
        and total_vetoes >= 1
    )
    metrics = {
        "source_shield": source_metrics,
        "safety_regressions": safety_regressions,
        "progress_regressions": progress_regressions,
        "vetoes": total_vetoes,
    }
    report = {
        "format_version": "sage-t12.1-shield-report-v1",
        "status": "PASS_SHIELD_GATE" if passed else "FAIL_SHIELD_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "conditions": conditions,
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "shield_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = phase_receipt(
        manifest=manifest,
        phase="shield",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        parent_receipt=parent,
        artifacts={
            "archives": artifacts,
            "shield": shield_meta,
            "archive_receipt": {
                "path": str(Path(archive_receipt_path).resolve()),
                "sha256": _file_sha256(archive_receipt_path),
            },
            "report": {"path": str(report_path.resolve()), "sha256": _file_sha256(report_path)},
        },
    )
    _write_json_once(destination / "shield_receipt.json", receipt, storage_budget=storage)
    return report


def _action_priors(examples: Sequence[tuple[str, bool, bool]]) -> dict[str, tuple[float, float]]:
    totals: dict[str, list[int]] = {}
    for action_key, changed, novel in examples:
        row = totals.setdefault(action_key, [0, 0, 0])
        row[0] += 1
        row[1] += int(changed)
        row[2] += int(novel)
    return {
        key: ((row[1] + 1) / (row[0] + 2), (row[2] + 1) / (row[0] + 2))
        for key, row in totals.items()
    }


def train_novelty_phase(
    *,
    manifest_path: str | Path,
    shield_receipt_path: str | Path,
    output_dir: str | Path,
    archive_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_graph_manifest(manifest_path)
    _require_source_train(manifest, "train_novelty")
    parent = load_graph_receipt(
        shield_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(parent, "shield")
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    archive_receipt_meta = dict(parent["artifacts"]["archive_receipt"])
    if archive_receipt_path is not None:
        supplied = Path(archive_receipt_path)
        if (
            supplied.resolve() != Path(archive_receipt_meta["path"]).resolve()
            or _file_sha256(supplied) != archive_receipt_meta["sha256"]
        ):
            raise ValueError("supplied archive receipt does not match shield ancestry")
    archive_receipt = load_graph_receipt(
        archive_receipt_meta["path"], manifest=manifest, require_passed=True
    )
    _require_receipt_phase(archive_receipt, "archive")
    if parent.get("parent_receipt_checksum") != archive_receipt.get(
        "receipt_checksum"
    ):
        raise ValueError("shield receipt is not descended from the archive receipt")
    archive_rows = sorted(
        (
            dict(item)
            for item in archive_receipt["artifacts"]["archives"]
            if item.get("arm") == "symbolic_archive"
        ),
        key=lambda item: (str(item["game_id"]), int(item["seed"])),
    )
    if len(archive_rows) < 3:
        raise ValueError("novelty training needs three source archive seeds")
    loaded = [
        GoExploreArchive.from_dict(_load_artifact(row["path"], row["sha256"]))
        for row in archive_rows
    ]
    predictor = OnlineNoveltyPredictor(seed=7101)
    training_priors: list[tuple[str, bool, bool]] = []
    for archive in loaded[:-1]:
        for edge in sorted(archive.edges.values(), key=lambda item: item.ordinal):
            predictor.observe(
                archive.cells[edge.source_cell_id].state,
                edge.action,
                changed=edge.changed,
                novel=edge.novel,
                update=True,
            )
            training_priors.append((edge.action.key, edge.changed, edge.novel))
    priors = _action_priors(training_priors)
    global_change = (sum(item[1] for item in training_priors) + 1) / (
        len(training_priors) + 2
    )
    global_novel = (sum(item[2] for item in training_priors) + 1) / (
        len(training_priors) + 2
    )
    validation = loaded[-1]
    probabilities: list[tuple[float, float]] = []
    shuffled_probabilities: list[tuple[float, float]] = []
    targets: list[tuple[float, float]] = []
    baseline_probabilities: list[tuple[float, float]] = []
    ordered_edges = sorted(validation.edges.values(), key=lambda item: item.ordinal)
    states = [validation.cells[edge.source_cell_id].state for edge in ordered_edges]
    for index, edge in enumerate(ordered_edges):
        prediction = predictor.predict(states[index], edge.action)
        shuffled = predictor.predict(states[(index + 1) % len(states)], edge.action)
        probabilities.append(
            (prediction.change_probability, prediction.novelty_probability)
        )
        shuffled_probabilities.append(
            (shuffled.change_probability, shuffled.novelty_probability)
        )
        targets.append((float(edge.changed), float(edge.novel)))
        baseline_probabilities.append(
            priors.get(edge.action.key, (global_change, global_novel))
        )
    if not targets:
        raise ValueError("novelty validation archive contains no edges")

    def mean_brier(rows: Sequence[tuple[float, float]]) -> float:
        return 0.5 * (
            brier_score([row[0] for row in rows], [row[0] for row in targets])
            + brier_score([row[1] for row in rows], [row[1] for row in targets])
        )

    model_brier = mean_brier(probabilities)
    baseline_brier = mean_brier(baseline_probabilities)
    shuffled_brier = mean_brier(shuffled_probabilities)
    change_prevalence = sum(row[0] for row in targets) / len(targets)
    novelty_prevalence = sum(row[1] for row in targets) / len(targets)
    maximum_ece = max(
        expected_calibration_error(
            [row[0] for row in probabilities], [row[0] for row in targets]
        ),
        expected_calibration_error(
            [row[1] for row in probabilities], [row[1] for row in targets]
        ),
    )
    metrics = {
        "training_examples": len(training_priors),
        "validation_examples": len(targets),
        "parameter_count": predictor.parameter_count,
        "change_prevalence": change_prevalence,
        "novelty_prevalence": novelty_prevalence,
        "model_mean_brier": model_brier,
        "action_only_mean_brier": baseline_brier,
        "brier_gain": baseline_brier - model_brier,
        "state_shuffle_degradation": shuffled_brier - model_brier,
        "maximum_ece": maximum_ece,
    }
    passed = bool(
        len(training_priors) >= protocol.neural_minimum_examples
        and 0.05 <= change_prevalence <= 0.95
        and 0.05 <= novelty_prevalence <= 0.95
        and metrics["brier_gain"] >= protocol.neural_minimum_brier_gain
        and metrics["state_shuffle_degradation"]
        >= protocol.neural_minimum_state_shuffle_degradation
        and maximum_ece <= protocol.neural_maximum_ece
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    checkpoint_path = destination / "novelty_predictor.pt"
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
            "metadata": metrics,
        },
        buffer,
    )
    encoded = buffer.getvalue()
    storage.reserve(len(encoded))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as handle:
        handle.write(encoded)
    report = {
        "format_version": "sage-t12.1-novelty-report-v1",
        "status": "PASS_NOVELTY_GATE" if passed else "FAIL_NOVELTY_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "novelty_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = phase_receipt(
        manifest=manifest,
        phase="train_novelty",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        parent_receipt=parent,
        artifacts={
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _file_sha256(checkpoint_path),
            },
            "shield": parent["artifacts"]["shield"],
            "archive_receipt": archive_receipt_meta,
            "report": {"path": str(report_path.resolve()), "sha256": _file_sha256(report_path)},
        },
    )
    _write_json_once(destination / "novelty_receipt.json", receipt, storage_budget=storage)
    return report


def run_neural_phase(
    *,
    manifest_path: str | Path,
    novelty_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    """Compare the frozen symbolic archive with and without neural ordering."""

    manifest = load_graph_manifest(manifest_path, verify_code=env_factory is None)
    _require_source_train(manifest, "neural_ordering")
    parent = load_graph_receipt(
        novelty_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(parent, "train_novelty")
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    checkpoint_meta = dict(parent["artifacts"]["checkpoint"])
    shield_meta = dict(parent["artifacts"]["shield"])
    _load_artifact(shield_meta["path"], shield_meta["sha256"])
    checkpoint_path = Path(checkpoint_meta["path"])
    if (
        not checkpoint_path.is_file()
        or _file_sha256(checkpoint_path) != checkpoint_meta["sha256"]
    ):
        raise ValueError("novelty checkpoint checksum mismatch")
    shield_payload = _read_json(shield_meta["path"])
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    artifacts = []
    for game_id in manifest["games"]:
        for seed in protocol.neural_seeds:
            arms = {}
            for arm in protocol.neural_arms:
                predictor = (
                    OnlineNoveltyPredictor.load(checkpoint_path)
                    if arm == "archive_shield_neural"
                    else None
                )
                archive, used_shield = run_go_explore_arm(
                    game_id=str(game_id),
                    seed=seed,
                    sdk_call_budget=protocol.sdk_call_budget_per_seed_arm,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                    shield=MultiStepTerminalShield.from_dict(shield_payload),
                    learn_shield=False,
                    predictor=predictor,
                    update_predictor=False,
                    maximum_cells=protocol.maximum_cells,
                )
                path = destination / str(game_id) / str(seed) / f"{arm}.json"
                artifact = _write_archive(path, archive, storage_budget=storage)
                artifact.update({"game_id": game_id, "seed": seed, "arm": arm})
                artifacts.append(artifact)
                arms[arm] = {
                    "metrics": archive.metrics(),
                    "shield_metrics": (
                        {} if used_shield is None else used_shield.metrics()
                    ),
                    "artifact": artifact,
                }
            conditions.append({"game_id": game_id, "seed": seed, "arms": arms})
    passed, metrics = _paired_gate(
        conditions,
        baseline_arm="archive_shield",
        treatment_arm="archive_shield_neural",
        minimum_relative_gain=protocol.neural_minimum_relative_coverage_gain,
    )
    positive_seeds = sum(
        item["relative_gain"] > 0.0 for item in metrics["per_seed"]
    )
    progress_regressions = sum(
        item["treatment_progress"] < item["baseline_progress"]
        for item in metrics["per_seed"]
    )
    safety_regressions = sum(
        item["treatment_terminal"] > item["baseline_terminal"]
        for item in metrics["per_seed"]
    )
    replay_integrity = all(
        arm["metrics"]["replay_exact_rate"] == 1.0
        for condition in conditions
        for arm in condition["arms"].values()
    )
    metrics.update(
        {
            "positive_coverage_seeds": positive_seeds,
            "progress_regressions": progress_regressions,
            "safety_regressions": safety_regressions,
            "replay_integrity": replay_integrity,
        }
    )
    passed = bool(
        passed
        and positive_seeds >= 2
        and progress_regressions == 0
        and safety_regressions == 0
        and replay_integrity
    )
    report = {
        "format_version": "sage-t12.1-neural-ordering-report-v1",
        "status": "PASS_NEURAL_ORDERING_GATE" if passed else "FAIL_NEURAL_ORDERING_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "conditions": conditions,
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "neural_ordering_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = phase_receipt(
        manifest=manifest,
        phase="neural_ordering",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        parent_receipt=parent,
        artifacts={
            "archives": artifacts,
            "checkpoint": checkpoint_meta,
            "shield": shield_meta,
            "archive_receipt": parent["artifacts"]["archive_receipt"],
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(
        destination / "neural_ordering_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def _archive_artifacts(receipt: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in receipt.get("artifacts", {}).get("archives", ()))


def _load_archive_artifact(meta: Mapping[str, Any]) -> GoExploreArchive:
    return GoExploreArchive.from_dict(
        _load_artifact(str(meta["path"]), str(meta["sha256"]))
    )


def _first_progress_trace(
    archive_receipt: Mapping[str, Any],
) -> tuple[str, GoExploreArchive, ArchiveStateVariant, tuple[ArchiveEdge, ...]] | None:
    candidates = []
    for meta in _archive_artifacts(archive_receipt):
        if meta.get("arm") != "symbolic_archive":
            continue
        archive = _load_archive_artifact(meta)
        for edge in archive.edges.values():
            if not (edge.level_delta > 0 or edge.success):
                continue
            target = archive.cells[edge.target_cell_id]
            variant = target.variants[edge.target_exact_hash]
            candidates.append(
                (
                    edge.ordinal,
                    str(meta.get("game_id", "")),
                    int(meta.get("seed", 0)),
                    archive,
                    variant,
                    archive.path_edges(variant),
                )
            )
    if not candidates:
        return None
    _, game_id, _, archive, variant, edges = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return game_id, archive, variant, edges


def _replay_from_variant(
    *,
    archive: GoExploreArchive,
    initiation: ArchiveStateVariant,
    actions: tuple[GroundedAction, ...],
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> tuple[bool, int, str]:
    env, frame, exact, calls = _restore_variant(
        archive=archive,
        variant=initiation,
        game_id=game_id,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if not exact:
        return False, calls, "INITIATION_REPLAY_MISMATCH"
    initial = snapshot_frame(frame)
    for action in actions:
        selected = select_live_action(
            env, action.action_name, action_args=action.action_data
        )
        if selected is None:
            return False, calls, "ACTION_UNAVAILABLE"
        frame = _step_env_action(env, selected)
        calls += 1
    final = snapshot_frame(frame)
    progressed = int(final.levels_completed) > int(initial.levels_completed)
    return progressed, calls, "PROGRESS" if progressed else "NO_PROGRESS"


def extract_option_phase(
    *,
    manifest_path: str | Path,
    archive_receipt_path: str | Path,
    parent_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    """Minimize the earliest symbolic-archive progression by exact replay."""

    manifest = load_graph_manifest(manifest_path, verify_code=env_factory is None)
    _require_source_train(manifest, "option_extraction")
    archive_receipt = load_graph_receipt(
        archive_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(archive_receipt, "archive")
    parent = load_graph_receipt(
        parent_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(parent, "neural_ordering")
    expected_archive = dict(parent["artifacts"]["archive_receipt"])
    if (
        Path(expected_archive["path"]).resolve()
        != Path(archive_receipt_path).resolve()
        or expected_archive["sha256"] != _file_sha256(archive_receipt_path)
    ):
        raise ValueError("option extraction archive is outside neural receipt ancestry")
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    first = _first_progress_trace(archive_receipt)
    option: MinimalCausalOption | None = None
    calls_used = 0
    failure = ""
    if first is None:
        failure = "OPTION_NO_ARCHIVE_PROGRESS"
    else:
        game_id, archive, _, path_edges = first
        progress_index = len(path_edges) - 1
        previous_progress = max(
            (
                index
                for index, edge in enumerate(path_edges[:progress_index])
                if edge.level_delta > 0 or edge.success
            ),
            default=-1,
        )
        suffix = path_edges[previous_progress + 1 :]
        suffix = suffix[-protocol.option_maximum_horizon :]
        if not suffix:
            failure = "OPTION_EMPTY_SUFFIX"
        else:
            source_cell = archive.cells[suffix[0].source_cell_id]
            initiation = source_cell.variants[suffix[0].source_exact_hash]
            states_before = tuple(
                archive.cells[edge.source_cell_id].state for edge in suffix
            )

            def replay_progress(actions: tuple[GroundedAction, ...]) -> bool:
                nonlocal calls_used
                remaining = protocol.sdk_call_budget_per_seed_arm - calls_used
                estimated = 1 + archive.prefixes.depth(initiation.prefix_id) + len(actions)
                if estimated > remaining:
                    return False
                progressed, calls, _ = _replay_from_variant(
                    archive=archive,
                    initiation=initiation,
                    actions=actions,
                    game_id=game_id,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
                calls_used += calls
                return progressed

            try:
                option = MinimalOptionExtractor(
                    maximum_horizon=protocol.option_maximum_horizon
                ).extract(
                    initiation_state=source_cell.state,
                    initiation_exact_hash=initiation.exact_hash,
                    actions=tuple(edge.action for edge in suffix),
                    states_before=states_before,
                    replay_progress=replay_progress,
                    expected_effects=tuple(
                        "level_progress"
                        if edge.level_delta > 0 or edge.success
                        else "state_change"
                        if edge.changed
                        else "no_change"
                        for edge in suffix
                    ),
                    source_evidence_ids=tuple(edge.edge_id for edge in suffix),
                )
            except (RuntimeError, ValueError) as exc:
                failure = f"OPTION_EXTRACTION_FAILED:{type(exc).__name__}:{exc}"
    passed = option is not None
    artifacts: dict[str, Any] = {
        "archive_receipt": {
            "path": str(Path(archive_receipt_path).resolve()),
            "sha256": _file_sha256(archive_receipt_path),
        }
    }
    if option is not None:
        option_path = destination / "minimal_option.json"
        option_payload = option.safe_payload
        option_payload["option_checksum"] = option.checksum
        _write_json_once(option_path, option_payload, storage_budget=storage)
        artifacts["option"] = {
            "path": str(option_path.resolve()),
            "sha256": _file_sha256(option_path),
        }
    metrics = {
        "sdk_calls": calls_used,
        "option_found": passed,
        "option_length": 0 if option is None else len(option.steps),
        "reproduction_count": 0 if option is None else option.reproduction_count,
        "minimization_evaluations": (
            0 if option is None else option.minimization_evaluations
        ),
        "failure": failure,
    }
    report = {
        "format_version": "sage-t12.1-option-extraction-report-v1",
        "status": "PASS_OPTION_EXTRACTION_GATE" if passed else "FAIL_OPTION_EXTRACTION_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "option_extraction_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    artifacts["report"] = {
        "path": str(report_path.resolve()),
        "sha256": _file_sha256(report_path),
    }
    receipt = phase_receipt(
        manifest=manifest,
        phase="option_extraction",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        parent_receipt=parent,
        artifacts=artifacts,
    )
    _write_json_once(destination / "option_receipt.json", receipt, storage_budget=storage)
    return report


def _load_option_artifact(meta: Mapping[str, Any]) -> MinimalCausalOption:
    payload = _load_artifact(str(meta["path"]), str(meta["sha256"]))
    expected = str(payload.pop("option_checksum", ""))
    option = MinimalCausalOption.from_dict(payload)
    if not expected or option.checksum != expected:
        raise ValueError("minimal option checksum mismatch")
    return option


def _programs_from_graph_manifest(
    manifest: Mapping[str, Any], game_id: str
) -> tuple[CausalProgram, ...]:
    meta = manifest.get("program_registry")
    if not isinstance(meta, Mapping):
        raise ValueError("option compilation requires a frozen program registry")
    path = Path(str(meta["path"]))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    if not path.is_file() or _file_sha256(path) != str(meta["sha256"]):
        raise ValueError("frozen program registry checksum mismatch")
    registry = _read_json(path)
    _verify_signed(registry, "registry_checksum")
    if registry.get("format_version") != "sage-t11-causal-program-registry-v1":
        raise ValueError("unsupported causal program registry")
    game = dict(registry.get("games", {})).get(str(game_id))
    if not isinstance(game, Mapping):
        raise ValueError(f"program registry has no entry for {game_id}")
    return tuple(
        causal_program_from_dict(dict(item)) for item in game.get("programs", ())
    )


def _state_for_program(state: Any, program: CausalProgram) -> CausalState:
    causal = causal_state_from_abstract(state)
    values = dict(causal.variables)
    for variable in program.variables:
        if variable.variable_id in values:
            continue
        default = variable.domain[0] if variable.domain else None
        values[variable.variable_id] = ValueDistribution.deterministic(default)
    return CausalState(
        variables=values,
        entities=causal.entities,
        relations=causal.relations,
        observation_hash=causal.observation_hash,
        confidence=causal.confidence,
    )


def _execute_option_automaton(
    executor: CausalExecutor,
    program: CausalProgram,
    state: CausalState,
    actions: Sequence[GroundedAction],
    complete_variable: str,
) -> tuple[bool, tuple[int, ...]]:
    phases = []
    current = state
    phase_variable = complete_variable.removesuffix(".complete") + ".phase"
    for action in actions:
        prediction = executor.predict_step(program, current, action)
        current = prediction.state_after
        raw_phase = current.value(phase_variable).mode
        phases.append(int(raw_phase or 0))
    return bool(current.value(complete_variable).mode), tuple(phases)


def compile_option_phase(
    *,
    manifest_path: str | Path,
    option_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile the minimized sequence into complete causal-program particles."""

    manifest = load_graph_manifest(manifest_path)
    _require_source_train(manifest, "option_compilation")
    parent = load_graph_receipt(
        option_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(parent, "option_extraction")
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    option = _load_option_artifact(dict(parent["artifacts"]["option"]))
    archive_receipt_meta = dict(parent["artifacts"]["archive_receipt"])
    archive_receipt = load_graph_receipt(
        archive_receipt_meta["path"], manifest=manifest, require_passed=True
    )
    first = _first_progress_trace(archive_receipt)
    if first is None:
        raise ValueError("option compilation lost its source progress trace")
    game_id, archive, _, _ = first
    parents = _programs_from_graph_manifest(manifest, game_id)
    if not parents:
        raise ValueError("option compilation needs rival parent programs")
    # Keep the common parent posterior bounded before creating one child per
    # complete dynamics+goal particle.
    parent_executor = CausalExecutor()
    parent_posterior = CausalPosterior(
        executor=parent_executor,
        maximum_particles=8,
        maximum_repair_parents=0,
    )
    parent_posterior.seed(parents)
    selected_parents = tuple(item.program for item in parent_posterior.top(8))
    children, compiled = CausalOptionCompiler().compile(option, selected_parents)
    executor = CausalExecutor(mechanism_registry=OptionMechanismRegistry())
    for child in children:
        executor.compile(child)
    posterior = CausalPosterior(
        executor=executor,
        maximum_particles=max(2, len(children)),
        minimum_particles=1,
        maximum_repair_parents=0,
    )
    posterior.seed(children)
    provider = PosteriorOptionProvider(
        compiled,
        minimum_posterior_mass=protocol.option_minimum_posterior_mass,
    )
    initiation_cell = next(
        (
            cell
            for cell in archive.cells.values()
            if option.initiation_exact_hash in cell.variants
        ),
        None,
    )
    if initiation_cell is None:
        raise ValueError("compiled option initiation state not found in archive")
    materialized = provider.materialize(initiation_cell.state, posterior)
    complete_variable = next(
        variable.variable_id
        for variable in children[0].variables
        if variable.variable_id.startswith(f"option.{option.option_id}.")
        and variable.variable_id.endswith(".complete")
    )
    initial = _state_for_program(initiation_cell.state, children[0])
    completed, phases = _execute_option_automaton(
        executor, children[0], initial, materialized, complete_variable
    )
    deletion_controls = []
    for index in range(len(materialized)):
        deleted = materialized[:index] + materialized[index + 1 :]
        deleted_complete, _ = _execute_option_automaton(
            executor, children[0], initial, deleted, complete_variable
        )
        deletion_controls.append(deleted_complete)
    reversed_complete = None
    if len(materialized) > 1:
        reversed_complete, _ = _execute_option_automaton(
            executor,
            children[0],
            initial,
            tuple(reversed(materialized)),
            complete_variable,
        )
    passed = bool(
        completed
        and len(materialized) == len(option.steps)
        and not any(deletion_controls)
        and (reversed_complete is not True or len({a.key for a in materialized}) == 1)
        and provider.owner_mass(posterior) + 1e-12
        >= protocol.option_minimum_posterior_mass
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    registry_path = destination / "compiled_option.json"
    children_path = destination / "option_programs.json"
    _write_json_once(registry_path, compiled.to_dict(), storage_budget=storage)
    _write_json_once(
        children_path,
        {
            "format_version": "sage-t12.1-option-programs-v1",
            "game_id": game_id,
            "programs": [child.to_dict() for child in children],
        },
        storage_budget=storage,
    )
    metrics = {
        "parent_programs": len(selected_parents),
        "child_programs": len(children),
        "posterior_owner_mass": provider.owner_mass(posterior),
        "option_length": len(option.steps),
        "materialized_length": len(materialized),
        "completed": completed,
        "phase_trace": list(phases),
        "deletion_controls_complete": deletion_controls,
        "reversed_control_complete": reversed_complete,
    }
    report = {
        "format_version": "sage-t12.1-option-compilation-report-v1",
        "status": "PASS_OPTION_COMPILATION_GATE" if passed else "FAIL_OPTION_COMPILATION_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "option_compilation_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = phase_receipt(
        manifest=manifest,
        phase="option_compilation",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        parent_receipt=parent,
        artifacts={
            "compiled_option": {
                "path": str(registry_path.resolve()),
                "sha256": _file_sha256(registry_path),
            },
            "option_programs": {
                "path": str(children_path.resolve()),
                "sha256": _file_sha256(children_path),
            },
            "archive_receipt": parent["artifacts"]["archive_receipt"],
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(
        destination / "option_compilation_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def _transition_evidence(
    *,
    before_state: Any,
    after_state: Any,
    action: GroundedAction,
    before_level: int,
    after_level: int,
    terminal: bool,
    game_id: str,
    ordinal: int,
) -> TransitionEvidence:
    before = causal_state_from_abstract(before_state)
    after = causal_state_from_abstract(after_state)
    changes = {
        key: value
        for key, value in after.variables.items()
        if value.total_variation(before.value(key)) > 1e-12
    }
    level_change = max(0, int(after_level) - int(before_level))
    return TransitionEvidence(
        evidence_id=f"ev-{game_id}-{ordinal}",
        state_before=before,
        action=action,
        state_after=after,
        observed_delta=StructuredDelta(
            variable_changes=changes,
            progress=min(1.0, float(level_change > 0)),
        ),
        terminal=terminal,
        success=level_change > 0,
        level_change=level_change,
        game_id=game_id,
    )


def _load_compiled_option(
    receipt: Mapping[str, Any],
) -> tuple[CompiledCausalOption, tuple[CausalProgram, ...]]:
    compiled_payload = _load_artifact(
        **{
            "path": str(receipt["artifacts"]["compiled_option"]["path"]),
            "expected_sha256": str(
                receipt["artifacts"]["compiled_option"]["sha256"]
            ),
        }
    )
    compiled = CompiledCausalOption.from_dict(compiled_payload)
    programs_payload = _load_artifact(
        **{
            "path": str(receipt["artifacts"]["option_programs"]["path"]),
            "expected_sha256": str(
                receipt["artifacts"]["option_programs"]["sha256"]
            ),
        }
    )
    programs = tuple(
        causal_program_from_dict(dict(item))
        for item in programs_payload.get("programs", ())
    )
    if {item.canonical_hash for item in programs} != set(
        compiled.owner_program_hashes
    ):
        raise ValueError("compiled option owners do not match child programs")
    return compiled, programs


def _run_transfer_arm(
    *,
    arm: str,
    game_id: str,
    seed: int,
    archive: GoExploreArchive,
    entry: ArchiveStateVariant,
    compiled: CompiledCausalOption,
    programs: Sequence[CausalProgram],
    shield_payload: Mapping[str, Any],
    checkpoint_path: Path,
    protocol: GraphExploreProtocol,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> dict[str, Any]:
    env, frame, exact, calls = _restore_variant(
        archive=archive,
        variant=entry,
        game_id=game_id,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    initial = snapshot_frame(frame)
    initial_hash = state_signature_from_frame(frame)
    shield = MultiStepTerminalShield.from_dict(shield_payload)
    predictor = OnlineNoveltyPredictor.load(checkpoint_path)
    executor = CausalExecutor(mechanism_registry=OptionMechanismRegistry())
    posterior = CausalPosterior(
        executor=executor,
        maximum_particles=max(2, len(programs)),
        minimum_particles=1,
        maximum_repair_parents=0,
    )
    posterior.seed(programs)
    provider = PosteriorOptionProvider(
        compiled,
        minimum_posterior_mass=protocol.option_minimum_posterior_mass,
    )
    updates_enabled = arm == "causal_option_full"
    option_enabled = arm != "archive_no_option"
    option_authorized = arm == "raw_option" or bool(
        provider.owner_mass(posterior) + 1e-12
        >= protocol.option_minimum_posterior_mass
    )
    option_index = 0
    actions = []
    observations = []
    levels_gained = 0
    terminal_failures = 0
    horizon = max(1, protocol.transfer_levels * len(compiled.option.steps))
    while calls < protocol.sdk_call_budget_per_seed_arm and len(actions) < horizon:
        before_snapshot = snapshot_frame(frame)
        if _is_terminal(before_snapshot.game_state):
            terminal_failures += 1
            break
        before_state = _symbolic_state(frame)
        legal = _grounded_actions(env)
        if not legal:
            break
        cell_id = GoExploreArchive.cell_key(
            before_state,
            level=int(before_snapshot.levels_completed),
            legal_actions=legal,
        )
        selected_action: GroundedAction | None = None
        source = "archive"
        if option_enabled and option_authorized:
            step = compiled.option.steps[option_index % len(compiled.option.steps)]
            try:
                candidate = step.materialize(before_state)
            except ValueError:
                candidate = None
            if candidate is not None and any(item.key == candidate.key for item in legal):
                if shield.allows(cell_id, candidate):
                    selected_action = candidate
                    source = "option"
        if selected_action is None:
            allowed = [action for action in legal if shield.allows(cell_id, action)]
            if not allowed:
                break
            selected_action = min(
                allowed,
                key=lambda action: (
                    -predictor.score(before_state, action)[1],
                    -predictor.score(before_state, action)[0],
                    hashlib.sha256(
                        f"{seed}:{action.key}".encode("utf-8")
                    ).hexdigest(),
                    action.key,
                ),
            )
            source = "neural_archive"
        selected = select_live_action(
            env,
            selected_action.action_name,
            action_args=selected_action.action_data,
        )
        if selected is None:
            break
        before_level = int(before_snapshot.levels_completed)
        after_frame = _step_env_action(env, selected)
        calls += 1
        after_snapshot = snapshot_frame(
            after_frame,
            fallback_available_actions=before_snapshot.available_actions,
        )
        after_level = int(after_snapshot.levels_completed)
        after_state = _symbolic_state(after_frame)
        delta = max(0, after_level - before_level)
        levels_gained += delta
        actions.append(
            {
                "action_name": selected_action.action_name,
                "action_data": dict(selected_action.action_data),
                "source": source,
            }
        )
        observations.append(
            {
                "before_hash": state_signature_from_frame(frame),
                "after_hash": state_signature_from_frame(after_frame),
                "level_delta": delta,
                "terminal": _is_terminal(after_snapshot.game_state),
            }
        )
        if updates_enabled:
            posterior.update(
                _transition_evidence(
                    before_state=before_state,
                    after_state=after_state,
                    action=selected_action,
                    before_level=before_level,
                    after_level=after_level,
                    terminal=_is_terminal(after_snapshot.game_state),
                    game_id=game_id,
                    ordinal=len(actions),
                )
            )
            option_authorized = bool(
                provider.owner_mass(posterior) + 1e-12
                >= protocol.option_minimum_posterior_mass
            )
        if source == "option":
            option_index = 0 if delta > 0 else option_index + 1
        if _is_terminal(after_snapshot.game_state):
            terminal_failures += int(delta == 0)
            frame = after_frame
            break
        if delta > 0 and levels_gained >= protocol.transfer_levels:
            frame = after_frame
            break
        frame = after_frame
    return {
        "arm": arm,
        "entry_exact": exact,
        "entry_hash": initial_hash,
        "entry_level": int(initial.levels_completed),
        "actions": actions,
        "observations": observations,
        "sdk_calls": calls,
        "levels_gained": levels_gained,
        "terminal_failures": terminal_failures,
        "posterior_updates": len(posterior.evidence),
        "final_owner_mass": provider.owner_mass(posterior),
        "option_authorized": option_authorized,
    }


def run_transfer_phase(
    *,
    manifest_path: str | Path,
    compilation_receipt_path: str | Path,
    archive_receipt_path: str | Path,
    novelty_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    """Run exact-entry paired transfer across the next three level changes."""

    manifest = load_graph_manifest(manifest_path, verify_code=env_factory is None)
    _require_source_train(manifest, "option_transfer")
    compilation = load_graph_receipt(
        compilation_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(compilation, "option_compilation")
    archive_receipt = load_graph_receipt(
        archive_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(archive_receipt, "archive")
    novelty_receipt = load_graph_receipt(
        novelty_receipt_path, manifest=manifest, require_passed=True
    )
    _require_receipt_phase(novelty_receipt, "train_novelty")
    expected_archive = dict(compilation["artifacts"]["archive_receipt"])
    if (
        Path(expected_archive["path"]).resolve()
        != Path(archive_receipt_path).resolve()
        or expected_archive["sha256"] != _file_sha256(archive_receipt_path)
    ):
        raise ValueError("transfer archive is outside compilation ancestry")
    protocol = GraphExploreProtocol(**dict(manifest["protocol"]))
    compiled, programs = _load_compiled_option(compilation)
    first = _first_progress_trace(archive_receipt)
    if first is None:
        raise ValueError("transfer needs an archive progression entry")
    game_id, archive, entry, _ = first
    checkpoint_meta = dict(novelty_receipt["artifacts"]["checkpoint"])
    checkpoint_path = Path(checkpoint_meta["path"])
    if (
        not checkpoint_path.is_file()
        or _file_sha256(checkpoint_path) != checkpoint_meta["sha256"]
    ):
        raise ValueError("transfer novelty checkpoint mismatch")
    shield_meta = dict(novelty_receipt["artifacts"]["shield"])
    shield_payload = _load_artifact(shield_meta["path"], shield_meta["sha256"])
    conditions = []
    for seed in protocol.transfer_seeds:
        arms = {
            arm: _run_transfer_arm(
                arm=arm,
                game_id=game_id,
                seed=seed,
                archive=archive,
                entry=entry,
                compiled=compiled,
                programs=programs,
                shield_payload=shield_payload,
                checkpoint_path=checkpoint_path,
                protocol=protocol,
                environments_dir=environments_dir,
                env_factory=env_factory,
            )
            for arm in protocol.transfer_arms
        }
        conditions.append({"game_id": game_id, "seed": seed, "arms": arms})
    exact_pairing = all(
        len({arm["entry_hash"] for arm in condition["arms"].values()}) == 1
        and all(arm["entry_exact"] for arm in condition["arms"].values())
        for condition in conditions
    )
    full_levels = sum(
        condition["arms"]["causal_option_full"]["levels_gained"]
        for condition in conditions
    )
    raw_levels = sum(
        condition["arms"]["raw_option"]["levels_gained"]
        for condition in conditions
    )
    no_update_levels = sum(
        condition["arms"]["causal_option_no_posterior_update"]["levels_gained"]
        for condition in conditions
    )
    archive_levels = sum(
        condition["arms"]["archive_no_option"]["levels_gained"]
        for condition in conditions
    )
    full_terminal = sum(
        condition["arms"]["causal_option_full"]["terminal_failures"]
        for condition in conditions
    )
    control_terminal = min(
        sum(condition["arms"][arm]["terminal_failures"] for condition in conditions)
        for arm in ("archive_no_option", "raw_option", "causal_option_no_posterior_update")
    )
    passed = bool(
        exact_pairing
        and full_levels >= 2
        and full_levels >= max(raw_levels, no_update_levels, archive_levels)
        and full_terminal <= control_terminal
    )
    metrics = {
        "exact_pairing": exact_pairing,
        "causal_option_full_levels": full_levels,
        "raw_option_levels": raw_levels,
        "no_posterior_update_levels": no_update_levels,
        "archive_no_option_levels": archive_levels,
        "causal_option_full_terminal_failures": full_terminal,
        "best_control_terminal_failures": control_terminal,
        "memory_reset_between_conditions": True,
    }
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    bundles_path = destination / "transfer_bundles.json"
    _write_json_once(
        bundles_path,
        {
            "format_version": "sage-t12.1-transfer-bundles-v1",
            "conditions": conditions,
        },
        storage_budget=storage,
    )
    report = {
        "format_version": "sage-t12.1-transfer-report-v1",
        "status": "PASS_OPTION_TRANSFER_GATE" if passed else "FAIL_OPTION_TRANSFER_GATE",
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "conditions": conditions,
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "transfer_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = phase_receipt(
        manifest=manifest,
        phase="option_transfer",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        parent_receipt=compilation,
        artifacts={
            "bundles": {
                "path": str(bundles_path.resolve()),
                "sha256": _file_sha256(bundles_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(destination / "transfer_receipt.json", receipt, storage_budget=storage)
    return report


def graph_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_paths: Sequence[str | Path],
) -> dict[str, Any]:
    manifest = load_graph_manifest(manifest_path)
    receipts = [
        load_graph_receipt(path, manifest=manifest) for path in receipt_paths
    ]
    chain_valid = all(
        receipt.get("parent_receipt_checksum")
        == previous.get("receipt_checksum")
        for previous, receipt in zip(receipts, receipts[1:])
    )
    return {
        "format_version": "sage-t12.1-graph-experiment-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "scientific_claims_authorized": bool(
            manifest.get("scientific_claims_authorized", False)
        ),
        "stage": manifest["stage"],
        "firewall": dict(manifest["firewall"]),
        "receipt_chain_valid": chain_valid,
        "receipts": [
            {
                "phase": receipt["phase"],
                "passed": receipt["passed"],
                "status": receipt["status"],
                "receipt_checksum": receipt["receipt_checksum"],
            }
            for receipt in receipts
        ],
        "next_phase_authorized": bool(
            manifest.get("scientific_claims_authorized", False)
            and chain_valid
            and (not receipts or receipts[-1]["passed"])
        ),
    }
