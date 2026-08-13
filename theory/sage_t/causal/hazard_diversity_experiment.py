"""Offline compile and prospective T12.4a.4d.1 paired experiment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from theory.unified_cognition_ab_benchmark import _is_terminal

from .archive import _action_from_payload, abstract_state_from_payload
from .experiment import (
    RunStorageBudget,
    _file_sha256,
    _read_json,
    _write_json_once,
)
from .graph_experiment import (
    _grounded_actions,
    _intervention_bundles,
    _restore_variant,
    _symbolic_state,
    _write_archive,
)
from .hazard_diversity_model import (
    AbstractHazardModel,
    HazardObservation,
    StructuralActionDiversityPolicy,
)
from .hazard_diversity_protocol import (
    HazardDiversityProtocol,
    hazard_diversity_receipt,
    load_hazard_diversity_manifest,
    load_hazard_diversity_receipt,
)
from .shield_model import ProgressProtectedTerminalShield
from .target_regrounding_experiment import (
    AnchoredLineageArchive,
    ContractRegroundingScorer,
    _catalog_checksum,
    _confirm_suffix,
    _discovered_witnesses,
    _load_frozen_inputs,
    _replay_anchor,
)
from .target_regrounding_protocol import (
    load_target_regrounding_manifest,
    load_target_regrounding_receipt,
)
from .witness_protocol import ProgressWitness

EnvFactory = Callable[[str], Any]


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _parent_artifacts(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_manifest_path = _resolve(manifest["parent"]["manifest"]["path"], root=root)
    parent_receipt_path = _resolve(manifest["parent"]["receipt"]["path"], root=root)
    parent_manifest = load_target_regrounding_manifest(
        parent_manifest_path,
        root=root,
    )
    parent_receipt = load_target_regrounding_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=root,
    )
    return parent_manifest, parent_receipt


def _extract_hazard_observations(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    allowed_search_seeds: Sequence[int],
) -> tuple[HazardObservation, ...]:
    allowed = {int(value) for value in allowed_search_seeds}
    observations: list[HazardObservation] = []
    for meta in receipt["artifacts"]["archives"]:
        search_seed = int(meta["search_seed"])
        if search_seed not in allowed:
            raise ValueError("parent archive includes an unregistered compile seed")
        lineage_seed = int(meta["lineage_seed"])
        archive_path = _resolve(meta["path"], root=root)
        payload = _read_json(archive_path)
        cells = {
            str(row["cell_id"]): abstract_state_from_payload(dict(row["state"]))
            for row in payload["cells"]
        }
        for edge in payload["edges"]:
            observations.append(
                HazardObservation(
                    search_seed=search_seed,
                    lineage_seed=lineage_seed,
                    source_exact_hash=str(edge["source_exact_hash"]),
                    state=cells[str(edge["source_cell_id"])],
                    action=_action_from_payload(dict(edge["action"])),
                    terminal=bool(edge["terminal"]),
                )
            )
    deduplicated: dict[str, HazardObservation] = {}
    for item in observations:
        previous = deduplicated.get(item.observation_key)
        if previous is not None and previous.terminal != item.terminal:
            raise ValueError("conflicting labels in parent intervention evidence")
        deduplicated[item.observation_key] = item
    return tuple(deduplicated[key] for key in sorted(deduplicated))


def _binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    true_positive = sum(y and p for y, p in zip(labels, predictions, strict=True))
    false_positive = sum(
        (not y) and p for y, p in zip(labels, predictions, strict=True)
    )
    false_negative = sum(
        y and (not p) for y, p in zip(labels, predictions, strict=True)
    )
    true_negative = sum(
        (not y) and (not p) for y, p in zip(labels, predictions, strict=True)
    )
    return {
        "examples": len(labels),
        "false_negative": false_negative,
        "false_positive": false_positive,
        "false_positive_rate": false_positive / max(1, false_positive + true_negative),
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "terminal_examples": sum(labels),
        "true_negative": true_negative,
        "true_positive": true_positive,
    }


def _cross_fit(
    observations: Sequence[HazardObservation],
    *,
    protocol: HazardDiversityProtocol,
) -> dict[str, Any]:
    folds = []
    all_labels: list[bool] = []
    all_predictions: list[bool] = []
    for holdout in protocol.compile_search_seeds:
        training = tuple(
            item for item in observations if item.search_seed != holdout
        )
        validation = tuple(
            item for item in observations if item.search_seed == holdout
        )
        model = AbstractHazardModel.fit(
            training,
            radius=protocol.local_hazard_radius,
            minimum_support=protocol.minimum_hazard_support,
            unsafe_rate_threshold=protocol.unsafe_rate_threshold,
        )
        labels = [item.terminal for item in validation]
        predictions = [model.is_unsafe(item.state, item.action) for item in validation]
        metrics = _binary_metrics(labels, predictions)
        folds.append(
            {
                "holdout_search_seed": holdout,
                "metrics": metrics,
                "training_examples": len(training),
                "unsafe_signatures": len(model.unsafe_signatures),
            }
        )
        all_labels.extend(labels)
        all_predictions.extend(predictions)
    return {
        "folds": folds,
        "micro_metrics": _binary_metrics(all_labels, all_predictions),
    }


def compile_hazard_diversity(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = load_hazard_diversity_manifest(manifest_path)
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.4d.1 compile requires a clean freeze")
    if not manifest["firewall"]["hazard_compile_authorized"]:
        raise ValueError("T12.4a.4d.1 hazard compile is not authorized")
    protocol = HazardDiversityProtocol(**dict(manifest["protocol"]))
    root = Path(__file__).resolve().parents[3]
    _, parent_receipt = _parent_artifacts(manifest, root=root)
    observations = _extract_hazard_observations(
        parent_receipt,
        root=root,
        allowed_search_seeds=protocol.compile_search_seeds,
    )
    cross_fit = _cross_fit(observations, protocol=protocol)
    model = AbstractHazardModel.fit(
        observations,
        radius=protocol.local_hazard_radius,
        minimum_support=protocol.minimum_hazard_support,
        unsafe_rate_threshold=protocol.unsafe_rate_threshold,
    )
    micro = cross_fit["micro_metrics"]
    passing_folds = sum(
        float(item["metrics"]["recall"]) >= protocol.minimum_crossfit_recall
        for item in cross_fit["folds"]
    )
    checks = {
        "all_compile_seeds_present": {
            item.search_seed for item in observations
        }
        == set(protocol.compile_search_seeds),
        "crossfit_false_positive_rate_bounded": (
            float(micro["false_positive_rate"])
            <= protocol.maximum_crossfit_false_positive_rate
        ),
        "crossfit_precision_sufficient": (
            float(micro["precision"]) >= protocol.minimum_crossfit_precision
        ),
        "crossfit_recall_sufficient": (
            float(micro["recall"]) >= protocol.minimum_crossfit_recall
        ),
        "enough_recall_passing_folds": (
            passing_folds >= protocol.minimum_recall_passing_folds
        ),
        "model_has_unsafe_signatures": bool(model.unsafe_signatures),
        "terminal_examples_present": int(micro["terminal_examples"]) > 0,
    }
    passed = all(checks.values())
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable compile: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    model_path = destination / "abstract_hazard_model.sealed.json"
    report_path = destination / "hazard_crossfit_report.json"
    _write_json_once(model_path, model.to_dict(), storage_budget=storage)
    report = {
        "checks": checks,
        "cross_fit": cross_fit,
        "format_version": "sage-t12.4a.4d.1-hazard-crossfit-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "model": {
            "support_signatures": len(model.support),
            "unsafe_signatures": len(model.unsafe_signatures),
        },
        "observation_count": len(observations),
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": (
            "PASS_T12_4A_4D_1_HAZARD_COMPILE_GATE"
            if passed
            else "FAIL_T12_4A_4D_1_HAZARD_COMPILE_GATE"
        ),
    }
    _write_json_once(report_path, report, storage_budget=storage)
    metrics = {
        "checks": checks,
        "crossfit_micro": micro,
        "passing_recall_folds": passing_folds,
        "support_signatures": len(model.support),
        "unsafe_signatures": len(model.unsafe_signatures),
        "storage": storage.snapshot(),
    }
    status = report["status"]
    receipt = hazard_diversity_receipt(
        manifest=manifest,
        phase="compile",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "hazard_model": {
                "path": str(model_path.resolve()),
                "sha256": _file_sha256(model_path),
            },
            "crossfit_report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(destination / "compile_receipt.json", receipt, storage_budget=storage)
    return receipt


@dataclass(frozen=True)
class HazardDiversityRun:
    archive: AnchoredLineageArchive
    arm: str
    lineage_seed: int
    search_seed: int
    entry_exact: bool
    entry_hash: str
    entry_descriptor: Mapping[str, Any]
    candidate_catalog_checksum: str
    applicable_mass: float
    materialized_option_actions: int
    excursions: int
    progress_edge_id: str | None
    progress_suffix: tuple[Any, ...]
    progress_sdk_calls: int | None
    diversity_metrics: Mapping[str, Any]

    def metrics(self) -> dict[str, Any]:
        metrics = dict(self.archive.metrics())
        actions = int(metrics["edges"])
        terminal = int(metrics["terminal_edges"])
        families = Counter(edge.action.action_name for edge in self.archive.edges.values())
        maximum_share = max(families.values(), default=0) / max(1, actions)
        return {
            **metrics,
            **dict(self.diversity_metrics),
            "action_family_counts": dict(sorted(families.items())),
            "arm": self.arm,
            "candidate_catalog_checksum": self.candidate_catalog_checksum,
            "entry_exact": self.entry_exact,
            "entry_hash": self.entry_hash,
            "excursions": self.excursions,
            "exploration_actions": actions,
            "first_progress_sdk_calls": self.progress_sdk_calls,
            "materialized_option_actions": self.materialized_option_actions,
            "maximum_action_family_share": maximum_share,
            "option_applicable_mass": self.applicable_mass,
            "progress_suffix_length": len(self.progress_suffix),
            "terminal_failure_rate": terminal / max(1, actions),
        }


def run_hazard_diversity_arm(
    *,
    game_id: str,
    witness: ProgressWitness,
    registry: Any,
    posterior: Any,
    shield: ProgressProtectedTerminalShield,
    hazard_model: AbstractHazardModel,
    arm: str,
    search_seed: int,
    sdk_call_budget: int,
    maximum_excursions: int,
    maximum_cells: int,
    burst_schedule: tuple[int, ...],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> HazardDiversityRun:
    valid_arms = {
        "local_archive_control",
        "diversity_control",
        "abstract_hazard_diversity",
    }
    if arm not in valid_arms:
        raise ValueError(f"unsupported hazard-diversity arm: {arm}")
    archive = AnchoredLineageArchive(
        anchor_prefix_depth=len(witness.steps),
        maximum_cells=maximum_cells,
        seed=search_seed,
    )
    env, frame, entry_exact, calls, _, entry_hash = _replay_anchor(
        witness=witness,
        archive=archive,
        game_id=game_id,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    archive.sdk_calls = calls
    empty = StructuralActionDiversityPolicy(seed=search_seed)
    if not entry_exact:
        return HazardDiversityRun(
            archive=archive,
            arm=arm,
            lineage_seed=witness.source_seed,
            search_seed=search_seed,
            entry_exact=False,
            entry_hash=entry_hash,
            entry_descriptor={},
            candidate_catalog_checksum="",
            applicable_mass=1.0,
            materialized_option_actions=0,
            excursions=0,
            progress_edge_id=None,
            progress_suffix=(),
            progress_sdk_calls=None,
            diversity_metrics=empty.metrics(),
        )
    # Import lazily to retain the frozen T12.4a.4d descriptor/provider contract.
    from .option_applicability_experiment import _state_descriptor
    from .option_contracts import ContractedOptionProvider

    entry_snapshot = snapshot_frame(frame)
    entry_state = _symbolic_state(frame)
    descriptor = _state_descriptor(
        entry_state,
        exact_hash=entry_hash,
        level=int(entry_snapshot.levels_completed),
        game_state=str(entry_snapshot.game_state),
    )
    provider = ContractedOptionProvider(registry)
    applicable_mass = provider.applicable_mass(descriptor, posterior)
    materialized = provider.materialize(descriptor, entry_state, posterior)
    catalog = _grounded_actions(env)
    catalog_checksum = _catalog_checksum(catalog)
    scorer = ContractRegroundingScorer(registry)
    diversity = StructuralActionDiversityPolicy(seed=search_seed)

    excursion_index = 0
    progress_edge_id = None
    progress_suffix: tuple[Any, ...] = ()
    progress_sdk_calls = None
    while (
        archive.sdk_calls < sdk_call_budget
        and excursion_index < maximum_excursions
        and progress_edge_id is None
    ):
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
        if not exact:
            updated = replace(variant, replay_failures=variant.replay_failures + 1)
            cell.variants[variant.exact_hash] = updated
            if updated.replay_failures >= 2:
                cell.blocked = True
            excursion_index += 1
            continue
        source_cell = cell
        source_hash = variant.exact_hash
        source_prefix_id = variant.prefix_id
        source_path_ids = variant.path_edge_ids
        horizon = int(burst_schedule[excursion_index % len(burst_schedule)])
        for _ in range(horizon):
            if archive.sdk_calls >= sdk_call_budget:
                break
            before = snapshot_frame(frame)
            if _is_terminal(before.game_state):
                break
            candidates = _grounded_actions(env)
            if arm == "local_archive_control":
                action = archive.choose_action(source_cell, candidates, shield=shield)
            else:
                action = diversity.choose(
                    source_cell,
                    candidates,
                    static_shield=shield,
                    hazard_model=(
                        hazard_model
                        if arm == "abstract_hazard_diversity"
                        else None
                    ),
                    novelty_scorer=scorer,
                )
            if action is None:
                source_cell.blocked = True
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
                break
            after_frame = _step_env_action(env, selected)
            archive.sdk_calls += 1
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
            edge = archive.add_anchored_transition(
                source_cell_id=source_cell.cell_id,
                source_exact_hash=source_hash,
                source_prefix_id=source_prefix_id,
                source_path_edge_ids=source_path_ids,
                action=action,
                target_state=_symbolic_state(after_frame),
                target_exact_hash=target_hash,
                target_level=int(after.levels_completed),
                target_legal_actions=_grounded_actions(env),
                terminal=terminal,
                success=success,
                changed=source_hash != target_hash,
            )
            source_prefix_id = edge.prefix_id
            source_path_ids = source_path_ids + (edge.edge_id,)
            frame = after_frame
            source_cell = archive.cells[edge.target_cell_id]
            source_hash = edge.target_exact_hash
            if edge.level_delta > 0 or edge.success:
                progress_edge_id = edge.edge_id
                actions = archive.prefixes.actions(edge.prefix_id)
                progress_suffix = tuple(actions[len(witness.steps) :])
                progress_sdk_calls = archive.sdk_calls
                break
            if edge.terminal:
                break
        excursion_index += 1
    return HazardDiversityRun(
        archive=archive,
        arm=arm,
        lineage_seed=witness.source_seed,
        search_seed=search_seed,
        entry_exact=True,
        entry_hash=entry_hash,
        entry_descriptor=descriptor,
        candidate_catalog_checksum=catalog_checksum,
        applicable_mass=applicable_mass,
        materialized_option_actions=len(materialized),
        excursions=excursion_index,
        progress_edge_id=progress_edge_id,
        progress_suffix=progress_suffix,
        progress_sdk_calls=progress_sdk_calls,
        diversity_metrics=diversity.metrics(),
    )


def _active_gate(
    *,
    protocol: HazardDiversityProtocol,
    conditions: Sequence[Mapping[str, Any]],
    confirmation_trials: Sequence[Mapping[str, Any]],
    candidate: HazardDiversityRun | None,
    total_sdk_calls: int,
) -> tuple[bool, bool, dict[str, Any]]:
    all_arms = [arm for item in conditions for arm in item["arms"].values()]
    paired_catalogs = all(
        len(
            {
                item["arms"][arm]["metrics"]["candidate_catalog_checksum"]
                for arm in protocol.search_arms
            }
        )
        == 1
        for item in conditions
    )
    actions = {
        arm: sum(
            int(item["arms"][arm]["metrics"]["exploration_actions"])
            for item in conditions
        )
        for arm in protocol.search_arms
    }
    terminal = {
        arm: sum(
            int(item["arms"][arm]["metrics"]["terminal_edges"])
            for item in conditions
        )
        for arm in protocol.search_arms
    }
    progress = {
        arm: sum(
            int(item["arms"][arm]["metrics"]["progress_edges"])
            for item in conditions
        )
        for arm in protocol.search_arms
    }
    terminal_rates = {
        arm: terminal[arm] / max(1, actions[arm]) for arm in protocol.search_arms
    }
    family_counts: dict[str, Counter[str]] = {
        arm: Counter() for arm in protocol.search_arms
    }
    for item in conditions:
        for arm in protocol.search_arms:
            family_counts[arm].update(
                item["arms"][arm]["metrics"]["action_family_counts"]
            )
    family_shares = {
        arm: max(counts.values(), default=0) / max(1, sum(counts.values()))
        for arm, counts in family_counts.items()
    }
    abstract_vetoes = sum(
        int(
            item["arms"]["abstract_hazard_diversity"]["metrics"].get(
                "abstract_hazard_vetoes", 0
            )
        )
        for item in conditions
    )
    expected_confirmations = (
        len(protocol.source_lineages) * protocol.confirmation_repetitions_per_lineage
    )
    confirmation_exact_rate = sum(
        bool(item["prefix_exact"]) for item in confirmation_trials
    ) / max(1, expected_confirmations)
    final_hashes = {str(item["final_exact_hash"]) for item in confirmation_trials}
    confirmed = bool(
        len(confirmation_trials) == expected_confirmations
        and confirmation_exact_rate >= protocol.minimum_confirmation_exact_rate
        and all(
            item["prefix_exact"]
            and item["available"]
            and item["progressed"]
            and not item["terminal_failure"]
            for item in confirmation_trials
        )
        and len(final_hashes) == 1
    )
    all_options_blocked = all(
        int(item["metrics"]["materialized_option_actions"]) == 0
        and 1.0 - float(item["metrics"]["option_applicable_mass"])
        >= protocol.minimum_contract_block_margin
        for item in all_arms
    )
    checks = {
        "abstract_hazard_exercised": abstract_vetoes > 0,
        "all_anchor_replays_exact": all(
            bool(item["metrics"]["entry_exact"]) for item in all_arms
        ),
        "all_archive_replays_exact": all(
            float(item["metrics"]["replay_exact_rate"]) == 1.0
            for item in all_arms
        ),
        "contracted_option_blocked_at_every_anchor": all_options_blocked,
        "discovered_suffix_within_bound": bool(
            candidate
            and 0 < len(candidate.progress_suffix)
            <= protocol.maximum_witness_suffix_actions
        ),
        "diversity_arms_do_not_collapse": all(
            family_shares[arm] <= protocol.maximum_action_family_share
            for arm in ("diversity_control", "abstract_hazard_diversity")
        ),
        "exact_cross_lineage_confirmation": confirmed,
        "paired_candidate_catalogs_identical": paired_catalogs,
        "progress_witness_discovered": candidate is not None,
        "sdk_budget_respected": (
            all(
                int(item["metrics"]["sdk_calls"])
                <= protocol.sdk_calls_per_search_arm
                for item in all_arms
            )
            and total_sdk_calls <= protocol.maximum_total_sdk_calls
        ),
        "treatment_terminal_failure_rate_bounded": (
            terminal_rates["abstract_hazard_diversity"]
            <= protocol.maximum_treatment_terminal_failure_rate
        ),
        "treatment_not_riskier_than_diversity_control": (
            terminal_rates["abstract_hazard_diversity"]
            <= terminal_rates["diversity_control"] + 1e-12
        ),
    }
    guidance_passed = bool(
        progress["abstract_hazard_diversity"] >= protocol.minimum_progress_edges
        and terminal_rates["abstract_hazard_diversity"]
        <= terminal_rates["diversity_control"] + 1e-12
        and family_shares["abstract_hazard_diversity"]
        <= protocol.maximum_action_family_share
    )
    return all(checks.values()), guidance_passed, {
        "abstract_hazard_vetoes": abstract_vetoes,
        "action_family_counts": {
            arm: dict(sorted(value.items())) for arm, value in family_counts.items()
        },
        "actions": actions,
        "checks": checks,
        "confirmation_count": len(confirmation_trials),
        "confirmation_exact_rate": confirmation_exact_rate,
        "expected_confirmation_count": expected_confirmations,
        "final_hash_count": len(final_hashes),
        "guidance_claim_passed": guidance_passed,
        "maximum_action_family_shares": family_shares,
        "progress_edges": progress,
        "sdk_calls_used": total_sdk_calls,
        "terminal_edges": terminal,
        "terminal_failure_rates": terminal_rates,
    }


def run_hazard_diversity_experiment(
    *,
    manifest_path: str | Path,
    compile_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_hazard_diversity_manifest(
        manifest_path,
        verify_code=env_factory is None,
    )
    compile_receipt = load_hazard_diversity_receipt(
        compile_receipt_path,
        manifest=manifest,
    )
    if not (
        compile_receipt.get("passed") is True
        and compile_receipt.get("phase") == "compile"
        and compile_receipt.get("status")
        == "PASS_T12_4A_4D_1_HAZARD_COMPILE_GATE"
    ):
        raise ValueError("T12.4a.4d.1 active run requires the passed compile gate")
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.4d.1 active run requires a clean freeze")
    protocol = HazardDiversityProtocol(**dict(manifest["protocol"]))
    root = Path(__file__).resolve().parents[3]
    parent_manifest, _ = _parent_artifacts(manifest, root=root)
    witnesses, registry, posterior, frozen_shield = _load_frozen_inputs(
        parent_manifest,
        root=root,
    )
    model_path = _resolve(
        compile_receipt["artifacts"]["hazard_model"]["path"], root=root
    )
    hazard_model = AbstractHazardModel.from_dict(_read_json(model_path))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    runs: list[tuple[HazardDiversityRun, dict[str, Any]]] = []
    archive_artifacts = []
    bundles = []
    for search_seed in protocol.active_search_seeds:
        for witness in witnesses:
            arms = {}
            for arm in protocol.search_arms:
                shield = ProgressProtectedTerminalShield.from_dict(
                    frozen_shield.to_dict()
                )
                run = run_hazard_diversity_arm(
                    game_id=str(manifest["game_id"]),
                    witness=witness,
                    registry=registry,
                    posterior=posterior,
                    shield=shield,
                    hazard_model=hazard_model,
                    arm=arm,
                    search_seed=search_seed,
                    sdk_call_budget=protocol.sdk_calls_per_search_arm,
                    maximum_excursions=protocol.maximum_excursions_per_arm,
                    maximum_cells=protocol.maximum_cells,
                    burst_schedule=protocol.burst_schedule,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
                path = (
                    destination
                    / str(manifest["game_id"])
                    / str(search_seed)
                    / str(witness.source_seed)
                    / f"{arm}.json"
                )
                artifact = _write_archive(path, run.archive, storage_budget=storage)
                artifact.update(
                    {
                        "arm": arm,
                        "lineage_seed": witness.source_seed,
                        "search_seed": search_seed,
                    }
                )
                archive_artifacts.append(artifact)
                arms[arm] = {
                    "artifact": artifact,
                    "metrics": run.metrics(),
                    "shield_metrics": shield.metrics(),
                }
                runs.append((run, artifact))
                for index, item in enumerate(
                    _intervention_bundles(
                        run.archive,
                        game_id=str(manifest["game_id"]),
                        seed=search_seed,
                    )
                ):
                    bundles.append(
                        {
                            **item,
                            "arm": arm,
                            "bundle_id": (
                                f"bundle_{manifest['game_id']}_{search_seed}_"
                                f"{witness.source_seed}_{arm}_{index:06d}"
                            ),
                            "lineage_seed": witness.source_seed,
                            "source_bundle_id": item["bundle_id"],
                        }
                    )
            conditions.append(
                {
                    "arms": arms,
                    "game_id": manifest["game_id"],
                    "lineage_seed": witness.source_seed,
                    "search_seed": search_seed,
                }
            )
    candidates = [
        (run, artifact)
        for run, artifact in runs
        if run.progress_edge_id is not None and run.progress_suffix
    ]
    selected = min(
        candidates,
        key=lambda item: (
            int(item[0].progress_sdk_calls or 10**9),
            len(item[0].progress_suffix),
            item[0].search_seed,
            item[0].lineage_seed,
            item[0].arm,
        ),
        default=None,
    )
    confirmation_trials = []
    if selected is not None:
        candidate, _ = selected
        for witness in witnesses:
            for repetition in range(protocol.confirmation_repetitions_per_lineage):
                confirmation_trials.append(
                    _confirm_suffix(
                        witness=witness,
                        suffix=candidate.progress_suffix,
                        repetition=repetition,
                        game_id=str(manifest["game_id"]),
                        environments_dir=environments_dir,
                        env_factory=env_factory,
                    )
                )
    total_sdk_calls = sum(run.archive.sdk_calls for run, _ in runs) + sum(
        int(item["calls"]) for item in confirmation_trials
    )
    passed, guidance_passed, gate_metrics = _active_gate(
        protocol=protocol,
        conditions=conditions,
        confirmation_trials=confirmation_trials,
        candidate=None if selected is None else selected[0],
        total_sdk_calls=total_sdk_calls,
    )
    trials_path = destination / "confirmation_trials.json"
    bundles_path = destination / "intervention_bundles.json"
    registry_path = destination / "progress_witnesses.sealed.json"
    report_path = destination / "hazard_diversity_report.json"
    _write_json_once(
        trials_path,
        {
            "format_version": "sage-t12.4a.4d.1-confirmation-trials-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "trials": confirmation_trials,
        },
        storage_budget=storage,
    )
    _write_json_once(
        bundles_path,
        {
            "format_version": "sage-t12.4a.4d.1-intervention-bundles-v1",
            "bundles": bundles,
            "manifest_checksum": manifest["manifest_checksum"],
        },
        storage_budget=storage,
    )
    discovered: tuple[ProgressWitness, ...] = ()
    selected_payload = None
    if passed and selected is not None:
        candidate, artifact = selected
        discovered = _discovered_witnesses(
            witnesses=witnesses,
            trials=confirmation_trials,
            archive_sha256=str(artifact["sha256"]),
            progress_edge_id=str(candidate.progress_edge_id),
            source_arm=candidate.arm,
        )
        selected_payload = {
            "arm": candidate.arm,
            "archive_sha256": artifact["sha256"],
            "lineage_seed": candidate.lineage_seed,
            "progress_edge_id": candidate.progress_edge_id,
            "search_seed": candidate.search_seed,
            "suffix": [
                {
                    "action_name": action.action_name,
                    "action_data": dict(action.action_data),
                }
                for action in candidate.progress_suffix
            ],
        }
    from .target_regrounding_protocol import _checksum as _parent_checksum

    witness_payload = {
        "format_version": "sage-t12.4a.4d.1-progress-witness-registry-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "parent_compile_receipt_checksum": compile_receipt["receipt_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "selected_discovery": selected_payload,
        "witnesses": [item.to_dict() for item in discovered],
    }
    witness_registry = {
        **witness_payload,
        "registry_checksum": _parent_checksum(witness_payload),
    }
    _write_json_once(registry_path, witness_registry, storage_budget=storage)
    metrics = {
        **gate_metrics,
        "discovered_witness_count": len(discovered),
        "exact_prefix_intervention_bundles": len(bundles),
        "paired_condition_count": len(conditions),
        "selected_discovery": selected_payload,
        "storage": storage.snapshot(),
    }
    status = (
        "PASS_T12_4A_4D_1_HAZARD_DIVERSITY_GATE"
        if passed
        else "FAIL_T12_4A_4D_1_HAZARD_DIVERSITY_GATE"
    )
    report = {
        "conditions": conditions,
        "format_version": "sage-t12.4a.4d.1-hazard-diversity-report-v1",
        "guidance_claim_authorized": guidance_passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
        "storage": storage.snapshot(),
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = hazard_diversity_receipt(
        manifest=manifest,
        phase="active",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "archives": archive_artifacts,
            "confirmation_trials": {
                "path": str(trials_path.resolve()),
                "sha256": _file_sha256(trials_path),
            },
            "intervention_bundles": {
                "path": str(bundles_path.resolve()),
                "sha256": _file_sha256(bundles_path),
            },
            "progress_witness_registry": {
                "path": str(registry_path.resolve()),
                "sha256": _file_sha256(registry_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
        parent_receipt_checksum=compile_receipt["receipt_checksum"],
    )
    _write_json_once(
        destination / "hazard_diversity_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def hazard_diversity_status(
    *,
    manifest_path: str | Path,
    compile_receipt_path: str | Path | None = None,
    active_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_hazard_diversity_manifest(manifest_path)
    compile_receipt = None
    if compile_receipt_path is not None and Path(compile_receipt_path).is_file():
        compile_receipt = load_hazard_diversity_receipt(
            compile_receipt_path,
            manifest=manifest,
        )
    compile_passed = bool(
        compile_receipt
        and compile_receipt.get("passed") is True
        and compile_receipt.get("phase") == "compile"
        and compile_receipt.get("status")
        == "PASS_T12_4A_4D_1_HAZARD_COMPILE_GATE"
    )
    active_receipt = None
    if active_receipt_path is not None and Path(active_receipt_path).is_file():
        active_receipt = load_hazard_diversity_receipt(
            active_receipt_path,
            manifest=manifest,
        )
    active_passed = bool(
        active_receipt
        and active_receipt.get("passed") is True
        and active_receipt.get("phase") == "active"
        and active_receipt.get("status")
        == "PASS_T12_4A_4D_1_HAZARD_DIVERSITY_GATE"
        and active_receipt.get("parent_receipt_checksum")
        == (compile_receipt or {}).get("receipt_checksum")
    )
    return {
        "active_receipt": active_receipt,
        "compile_receipt": compile_receipt,
        "firewall": {
            "hazard_compile_authorized": manifest["firewall"][
                "hazard_compile_authorized"
            ],
            "hazard_diversity_active_run_authorized": compile_passed,
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_4a_4e_option_extraction_freeze_authorized": active_passed,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.4a.4d.1-hazard-diversity-status-v1",
        "guidance_claim_authorized": bool(
            active_passed
            and active_receipt.get("metrics", {}).get("guidance_claim_passed", False)
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": active_passed,
        "parent_t12_4a_4d_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
    }


__all__ = [
    "HazardDiversityRun",
    "compile_hazard_diversity",
    "hazard_diversity_status",
    "run_hazard_diversity_arm",
    "run_hazard_diversity_experiment",
]
