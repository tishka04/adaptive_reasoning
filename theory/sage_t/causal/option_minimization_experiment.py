from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from theory.sage.live_prefix_counterfactual_collector import (
    _step_env_action,
    select_live_action,
    snapshot_frame,
    state_signature_from_frame,
)

from .archive import AbstractState, GoExploreArchive
from .contracts import causal_program_from_dict
from .executor import CausalExecutor
from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import (
    _execute_option_automaton,
    _make_env,
    _state_for_program,
)
from .option_minimization_protocol import (
    CONTEXTUAL_OPTION_FORMAT,
    OptionMinimizationProtocol,
    _checksum,
    _read_json,
    _signed,
    _verify_signed,
    load_option_minimization_manifest,
    load_option_minimization_receipt,
    option_minimization_receipt,
)
from .options import (
    CausalOptionCompiler,
    MinimalCausalOption,
    MinimalOptionStep,
    OptionMechanismRegistry,
    PosteriorOptionProvider,
)
from .posterior import CausalPosterior
from .witness_experiment import (
    _execute_expected_steps,
    _reset_env,
)
from .witness_reconfirmation_protocol import load_reconfirmation_registry


@dataclass
class OptionSdkCallBudget:
    maximum: int
    used: int = 0

    def consume(self, count: int = 1, *, reason: str = "replay") -> None:
        self.used += int(count)
        if self.used > self.maximum:
            raise RuntimeError(
                f"T12.4a.3 SDK call budget exceeded during {reason}: "
                f"{self.used}>{self.maximum}"
            )


@dataclass(frozen=True)
class AblationContext:
    seed: int
    witness_id: str
    prefix: tuple[Any, ...]
    candidate: tuple[Any, ...]
    initial_exact_hash: str
    initiation_exact_hash: str
    initiation_signature: str
    target_exact_hash: str
    target_level: int
    archive_checksum: str


@dataclass(frozen=True)
class AblationTrial:
    trial_id: str
    context_seed: int
    repetition: int
    branch_kind: str
    selected_indices: tuple[int, ...]
    action_names: tuple[str, ...]
    prefix_exact: bool
    branch_available: bool
    initial_exact_hash: str
    prefix_exact_hash: str
    final_exact_hash: str
    target_exact_hash: str
    initial_level: int
    final_level: int
    target_level: int
    target_reached: bool
    progressed: bool
    off_target_progression: bool
    sdk_calls_after: int
    branch_trace: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_names": list(self.action_names),
            "branch_available": self.branch_available,
            "branch_kind": self.branch_kind,
            "branch_trace": [dict(item) for item in self.branch_trace],
            "context_seed": self.context_seed,
            "final_exact_hash": self.final_exact_hash,
            "final_level": self.final_level,
            "initial_exact_hash": self.initial_exact_hash,
            "initial_level": self.initial_level,
            "off_target_progression": self.off_target_progression,
            "prefix_exact": self.prefix_exact,
            "prefix_exact_hash": self.prefix_exact_hash,
            "progressed": self.progressed,
            "repetition": self.repetition,
            "sdk_calls_after": self.sdk_calls_after,
            "selected_indices": list(self.selected_indices),
            "target_exact_hash": self.target_exact_hash,
            "target_level": self.target_level,
            "target_reached": self.target_reached,
            "trial_id": self.trial_id,
        }


def _archive_state_for_exact_hash(
    archive: GoExploreArchive,
    exact_hash: str,
) -> AbstractState:
    for cell in archive.cells.values():
        for variant in cell.variants.values():
            if variant.exact_hash == exact_hash:
                return cell.state
    raise ValueError(f"archive has no exact state {exact_hash}")


def _load_contexts(
    manifest: Mapping[str, Any],
) -> tuple[tuple[AblationContext, GoExploreArchive], ...]:
    registry_path = Path(str(manifest["parent"]["witness_registry"]["path"]))
    _, witnesses = load_reconfirmation_registry(registry_path)
    source_by_seed = {
        int(item["seed"]): item for item in manifest["source_archives"]
    }
    contexts: list[tuple[AblationContext, GoExploreArchive]] = []
    protocol = OptionMinimizationProtocol(**dict(manifest["protocol"]))
    suffix_length = protocol.candidate_action_count
    for seed in protocol.source_seeds:
        witness = next(item for item in witnesses if item.source_seed == int(seed))
        prefix = tuple(witness.steps[:-suffix_length])
        candidate = tuple(witness.steps[-suffix_length:])
        source = source_by_seed[int(seed)]
        archive = GoExploreArchive.from_dict(
            _read_json(Path(str(source["path"])))
        )
        initiation_state = _archive_state_for_exact_hash(
            archive,
            candidate[0].expected_source_hash,
        )
        contexts.append(
            (
                AblationContext(
                    seed=int(seed),
                    witness_id=witness.witness_id,
                    prefix=prefix,
                    candidate=candidate,
                    initial_exact_hash=witness.initial_exact_hash,
                    initiation_exact_hash=candidate[0].expected_source_hash,
                    initiation_signature=initiation_state.signature,
                    target_exact_hash=witness.target_exact_hash,
                    target_level=witness.target_level,
                    archive_checksum=str(source["sha256"]),
                ),
                archive,
            )
        )
    return tuple(contexts)


def _mask_indices(mask: int, length: int) -> tuple[int, ...]:
    return tuple(index for index in range(length) if mask & (1 << index))


def _execute_branch(
    *,
    game_id: str,
    environments_dir: Path,
    context: AblationContext,
    indices: Sequence[int],
    branch_kind: str,
    repetition: int,
    budget: OptionSdkCallBudget,
) -> AblationTrial:
    env = _make_env(game_id, environments_dir, None)
    budget.consume(reason="reset")
    frame = _reset_env(env)
    initial_hash = state_signature_from_frame(frame)
    initial_snapshot = snapshot_frame(frame)
    frame, _, prefix_divergence = _execute_expected_steps(
        env=env,
        frame=frame,
        steps=context.prefix,
        phase=f"t12_4a_3_prefix_seed_{context.seed}",
        start_index=0,
        budget=budget,
    )
    prefix_signature = state_signature_from_frame(frame)
    prefix_exact = (
        initial_hash == context.initial_exact_hash
        and not prefix_divergence
        and prefix_signature == context.initiation_exact_hash
    )

    selected = tuple(context.candidate[index] for index in indices)
    if branch_kind == "reverse":
        selected = tuple(reversed(context.candidate))
    trace: list[dict[str, Any]] = []
    branch_available = True
    for position, step in enumerate(selected):
        source_hash = state_signature_from_frame(frame)
        selected_action = select_live_action(
            env,
            step.action.action_name,
            action_args=step.action.action_data,
        )
        if selected_action is None:
            branch_available = False
            trace.append(
                {
                    "action_name": step.action.action_name,
                    "available": False,
                    "position": position,
                    "source_exact_hash": source_hash,
                }
            )
            break
        budget.consume(1, reason=f"branch_{branch_kind}")
        frame = _step_env_action(env, selected_action)
        after = snapshot_frame(frame)
        trace.append(
            {
                "action_name": step.action.action_name,
                "available": True,
                "position": position,
                "source_exact_hash": source_hash,
                "target_exact_hash": state_signature_from_frame(frame),
                "target_level": int(after.levels_completed),
            }
        )

    final_snapshot = snapshot_frame(frame)
    final_hash = state_signature_from_frame(frame)
    initial_level = int(initial_snapshot.levels_completed)
    final_level = int(final_snapshot.levels_completed)
    progressed = final_level > initial_level
    target_reached = (
        progressed
        and final_level == context.target_level
        and final_hash == context.target_exact_hash
    )
    action_names = tuple(step.action.action_name for step in selected)
    selected_id = "-".join(str(item) for item in indices) or "empty"
    return AblationTrial(
        trial_id=(
            f"seed_{context.seed}_{branch_kind}_{selected_id}_rep_{repetition}"
        ),
        context_seed=context.seed,
        repetition=repetition,
        branch_kind=branch_kind,
        selected_indices=tuple(indices),
        action_names=action_names,
        prefix_exact=prefix_exact,
        branch_available=branch_available,
        initial_exact_hash=initial_hash,
        prefix_exact_hash=prefix_signature,
        final_exact_hash=final_hash,
        target_exact_hash=context.target_exact_hash,
        initial_level=initial_level,
        final_level=final_level,
        target_level=context.target_level,
        target_reached=target_reached,
        progressed=progressed,
        off_target_progression=progressed and not target_reached,
        sdk_calls_after=budget.used,
        branch_trace=tuple(trace),
    )


def _aggregate_matrix(
    trials: Sequence[AblationTrial],
    *,
    contexts: Sequence[AblationContext],
    repetitions: int,
    candidate_length: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    successful_masks: list[int] = []
    for mask in range(1 << candidate_length):
        indices = _mask_indices(mask, candidate_length)
        context_rows: list[dict[str, Any]] = []
        common_success = True
        for context in contexts:
            selected = [
                trial
                for trial in trials
                if trial.branch_kind == "subsequence"
                and trial.context_seed == context.seed
                and trial.selected_indices == indices
            ]
            confirmations = sum(trial.target_reached for trial in selected)
            common_success = common_success and confirmations == repetitions
            context_rows.append(
                {
                    "available_trials": sum(trial.branch_available for trial in selected),
                    "context_seed": context.seed,
                    "final_exact_hashes": sorted(
                        {trial.final_exact_hash for trial in selected}
                    ),
                    "off_target_progressions": sum(
                        trial.off_target_progression for trial in selected
                    ),
                    "prefix_exact_trials": sum(trial.prefix_exact for trial in selected),
                    "progressions": sum(trial.progressed for trial in selected),
                    "target_confirmations": confirmations,
                    "trials": len(selected),
                }
            )
        if common_success:
            successful_masks.append(mask)
        rows.append(
            {
                "action_names": [
                    contexts[0].candidate[index].action.action_name for index in indices
                ],
                "common_success": common_success,
                "contexts": context_rows,
                "length": len(indices),
                "mask": mask,
                "selected_indices": list(indices),
            }
        )
    minimum_length = (
        min(int(mask).bit_count() for mask in successful_masks)
        if successful_masks
        else None
    )
    minimal_masks = (
        [mask for mask in successful_masks if int(mask).bit_count() == minimum_length]
        if minimum_length is not None
        else []
    )
    return {
        "candidate_length": candidate_length,
        "minimum_successful_length": minimum_length,
        "minimal_masks": minimal_masks,
        "rows": rows,
        "successful_masks": successful_masks,
    }


def _build_contextual_option(
    *,
    manifest: Mapping[str, Any],
    contexts: Sequence[AblationContext],
    selected_indices: Sequence[int],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    primary = contexts[0]
    selected_steps = tuple(primary.candidate[index] for index in selected_indices)
    option_steps = tuple(
        MinimalOptionStep(
            action_name=step.action.action_name,
            static_action_data=step.action.action_data,
            expected_effect=(
                "level_progress" if position == len(selected_steps) - 1 else "state_transition"
            ),
        )
        for position, step in enumerate(selected_steps)
    )
    evidence_ids = tuple(
        [f"t12_4a_2:{context.witness_id}" for context in contexts]
        + [f"t12_4a_3:mask:{'-'.join(map(str, selected_indices))}"]
    )
    option = MinimalCausalOption(
        initiation_signature=primary.initiation_signature,
        initiation_exact_hash=primary.initiation_exact_hash,
        steps=option_steps,
        source="t12_4a_3_exhaustive_two_context_ablation",
        reproduction_count=(
            len(contexts)
            * int(manifest["protocol"]["repetitions_per_candidate_context"])
        ),
        minimization_evaluations=(
            int(manifest["protocol"]["exhaustive_subsequence_count"])
            * len(contexts)
            * int(manifest["protocol"]["repetitions_per_candidate_context"])
        ),
        source_evidence_ids=evidence_ids,
    )
    option_payload = option.safe_payload
    payload = {
        "parent_t12_4a_2_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "context_bindings": [
            {
                "archive_checksum": context.archive_checksum,
                "initiation_exact_hash": context.initiation_exact_hash,
                "initiation_signature": context.initiation_signature,
                "prefix_length": len(context.prefix),
                "seed": context.seed,
                "target_exact_hash": context.target_exact_hash,
                "target_level": context.target_level,
                "witness_id": context.witness_id,
            }
            for context in contexts
        ],
        "format_version": CONTEXTUAL_OPTION_FORMAT,
        "manifest_checksum": manifest["manifest_checksum"],
        "minimality": {
            "exhaustive": True,
            "minimum_successful_length": matrix["minimum_successful_length"],
            "selected_indices": list(selected_indices),
            "unique_minimal": len(matrix["minimal_masks"]) == 1,
        },
        "option": option_payload,
        "option_checksum": _checksum(option_payload),
        "protocol_checksum": manifest["protocol_checksum"],
    }
    return _signed(payload, "contextual_option_checksum")


def run_option_ablation(
    manifest_path: Path,
    *,
    output_dir: Path,
    environments_dir: Path,
) -> dict[str, Any]:
    manifest = load_option_minimization_manifest(manifest_path)
    if not manifest["firewall"]["option_ablation_authorized"]:
        raise ValueError("T12.4a.3 option ablation is not authorized")

    protocol = OptionMinimizationProtocol(**dict(manifest["protocol"]))
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {output_dir}")
    storage = RunStorageBudget(
        output_dir,
        protocol.maximum_artifact_bytes_per_run,
    )
    contexts_with_archives = _load_contexts(manifest)
    contexts = tuple(item[0] for item in contexts_with_archives)
    environments_path = environments_dir.resolve()
    budget = OptionSdkCallBudget(maximum=protocol.maximum_sdk_calls)
    trials: list[AblationTrial] = []
    candidate_length = protocol.candidate_action_count

    for context in contexts:
        for mask in range(protocol.exhaustive_subsequence_count):
            indices = _mask_indices(mask, candidate_length)
            for repetition in range(protocol.repetitions_per_candidate_context):
                trials.append(
                    _execute_branch(
                        game_id=str(manifest["game_id"]),
                        environments_dir=environments_path,
                        context=context,
                        indices=indices,
                        branch_kind="subsequence",
                        repetition=repetition,
                        budget=budget,
                    )
                )
        reverse_indices = tuple(reversed(range(candidate_length)))
        for repetition in range(protocol.repetitions_per_candidate_context):
            trials.append(
                _execute_branch(
                    game_id=str(manifest["game_id"]),
                    environments_dir=environments_path,
                    context=context,
                    indices=reverse_indices,
                    branch_kind="reverse",
                    repetition=repetition,
                    budget=budget,
                )
            )

    matrix = _aggregate_matrix(
        trials,
        contexts=contexts,
        repetitions=protocol.repetitions_per_candidate_context,
        candidate_length=candidate_length,
    )
    full_mask = (1 << candidate_length) - 1
    full_row = next(row for row in matrix["rows"] if row["mask"] == full_mask)
    minimal_masks = list(matrix["minimal_masks"])
    unique_minimal = len(minimal_masks) == 1
    selected_mask = minimal_masks[0] if unique_minimal else None
    selected_indices = (
        _mask_indices(selected_mask, candidate_length)
        if selected_mask is not None
        else ()
    )
    proper_subsequence_successes = [
        mask
        for mask in matrix["successful_masks"]
        if selected_mask is not None
        and mask != selected_mask
        and (mask & selected_mask) == mask
    ]
    reverse_trials = [trial for trial in trials if trial.branch_kind == "reverse"]
    prefix_exact_trials = sum(trial.prefix_exact for trial in trials)
    available_trials = sum(trial.branch_available for trial in trials)
    off_target_progressions = sum(trial.off_target_progression for trial in trials)
    reverse_progressions = sum(trial.progressed for trial in reverse_trials)
    expected_trials = (
        (protocol.exhaustive_subsequence_count + 1)
        * len(contexts)
        * protocol.repetitions_per_candidate_context
    )
    checks = {
        "all_branches_available": available_trials == expected_trials,
        "all_prefixes_exact": prefix_exact_trials == expected_trials,
        "full_candidate_reproduced_both_contexts": bool(full_row["common_success"]),
        "no_off_target_progression": (
            off_target_progressions <= protocol.maximum_off_target_progressions
        ),
        "proper_subsequences_fail": not proper_subsequence_successes,
        "reverse_control_no_progress": (
            not protocol.require_reversed_no_progress or reverse_progressions == 0
        ),
        "sdk_budget_respected": budget.used <= protocol.maximum_sdk_calls,
        "trial_count_exact": len(trials) == expected_trials,
        "unique_minimal_common_option": unique_minimal,
    }
    passed = all(checks.values())
    trials_path = output_dir / "ablation_trials.json"
    matrix_path = output_dir / "ablation_matrix.json"
    report_path = output_dir / "option_ablation_report.json"
    receipt_path = output_dir / "option_ablation_receipt.json"
    _write_json_once(
        trials_path,
        {
            "format_version": "sage-t12.4a.3-ablation-trials-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [trial.to_dict() for trial in trials],
        },
        storage_budget=storage,
    )
    _write_json_once(
        matrix_path,
        {
            "format_version": "sage-t12.4a.3-ablation-matrix-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "matrix": matrix,
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    report = {
        "checks": checks,
        "expected_trials": expected_trials,
        "format_version": "sage-t12.4a.3-option-ablation-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": {
            "available_trials": available_trials,
            "candidate_length": candidate_length,
            "context_count": len(contexts),
            "minimum_successful_length": matrix["minimum_successful_length"],
            "off_target_progressions": off_target_progressions,
            "prefix_exact_trials": prefix_exact_trials,
            "proper_subsequence_successes": proper_subsequence_successes,
            "reverse_progressions": reverse_progressions,
            "sdk_calls_used": budget.used,
            "selected_indices": list(selected_indices),
            "successful_mask_count": len(matrix["successful_masks"]),
            "trial_count": len(trials),
        },
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": (
            "PASS_T12_4A_3_OPTION_ABLATION_GATE"
            if passed
            else "FAIL_T12_4A_3_OPTION_ABLATION_GATE"
        ),
    }
    option_path: Path | None = None
    if passed:
        contextual = _build_contextual_option(
            manifest=manifest,
            contexts=contexts,
            selected_indices=selected_indices,
            matrix=matrix,
        )
        option_path = output_dir / "minimal_option.json"
        _write_json_once(option_path, contextual, storage_budget=storage)
    report["storage"] = storage.snapshot()
    _write_json_once(report_path, report, storage_budget=storage)
    artifacts = {
        "ablation_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": _file_sha256(matrix_path),
        },
        "ablation_trials": {
            "path": str(trials_path.resolve()),
            "sha256": _file_sha256(trials_path),
        },
        "report": {
            "path": str(report_path.resolve()),
            "sha256": _file_sha256(report_path),
        },
    }
    if option_path is not None:
        artifacts["minimal_option"] = {
            "path": str(option_path.resolve()),
            "sha256": _file_sha256(option_path),
        }
    receipt = option_minimization_receipt(
        manifest=manifest,
        phase="option_ablation",
        passed=passed,
        status=report["status"],
        metrics={
            **report["metrics"],
            "checks": checks,
            "storage": storage.snapshot(),
        },
        artifacts=artifacts,
    )
    _write_json_once(receipt_path, receipt, storage_budget=storage)
    return receipt


def _load_contextual_option(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("format_version") != CONTEXTUAL_OPTION_FORMAT:
        raise ValueError("unsupported T12.4a.3 contextual option")
    _verify_signed(payload, "contextual_option_checksum")
    if _checksum(payload["option"]) != payload["option_checksum"]:
        raise ValueError("T12.4a.3 option checksum mismatch")
    return payload


def compile_option_shadow(
    manifest_path: Path,
    *,
    ablation_receipt_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = load_option_minimization_manifest(manifest_path)
    ablation_receipt = load_option_minimization_receipt(
        ablation_receipt_path,
        manifest=manifest,
        require_passed=True,
    )
    if ablation_receipt.get("phase") != "option_ablation":
        raise ValueError("T12.4a.3 shadow compile requires the ablation receipt")
    option_payload = _load_contextual_option(
        Path(str(ablation_receipt["artifacts"]["minimal_option"]["path"]))
    )
    option = MinimalCausalOption.from_dict(option_payload["option"])

    registry_meta = manifest["program_registry"]
    registry = _read_json(Path(str(registry_meta["path"])))
    _verify_signed(registry, "registry_checksum")
    parent_programs = tuple(
        causal_program_from_dict(dict(item))
        for item in registry["games"][manifest["game_id"]]["programs"]
    )
    protocol = OptionMinimizationProtocol(**dict(manifest["protocol"]))
    parent_executor = CausalExecutor()
    parent_posterior = CausalPosterior(
        executor=parent_executor,
        maximum_particles=protocol.maximum_parent_particles,
        maximum_repair_parents=0,
    )
    parent_posterior.seed(parent_programs)
    selected_parents = tuple(
        particle.program
        for particle in parent_posterior.top(protocol.maximum_parent_particles)
    )
    child_programs, compiled = CausalOptionCompiler().compile(
        option,
        selected_parents,
    )
    executor = CausalExecutor(mechanism_registry=OptionMechanismRegistry())
    for child in child_programs:
        executor.compile(child)
    child_posterior = CausalPosterior(
        executor=executor,
        maximum_particles=protocol.maximum_parent_particles,
        maximum_repair_parents=0,
    )
    child_posterior.seed(child_programs)
    provider = PosteriorOptionProvider(
        compiled,
        minimum_posterior_mass=protocol.minimum_posterior_owner_mass,
    )
    owner_mass = provider.owner_mass(child_posterior)
    primary_context, primary_archive = _load_contexts(manifest)[0]
    abstract_state = _archive_state_for_exact_hash(
        primary_archive,
        primary_context.initiation_exact_hash,
    )
    actions = provider.materialize(abstract_state, child_posterior)
    positive_results: list[bool] = []
    deletion_results: list[bool] = []
    reverse_results: list[bool] = []
    for child in child_programs:
        complete_variable = next(
            variable.variable_id
            for variable in child.variables
            if variable.variable_id.startswith(f"option.{option.option_id}.")
            and variable.variable_id.endswith(".complete")
        )
        initial = _state_for_program(abstract_state, child)
        positive_results.append(
            _execute_option_automaton(
                executor,
                child,
                initial,
                actions,
                complete_variable,
            )[0]
        )
        for index in range(len(actions)):
            deletion = actions[:index] + actions[index + 1 :]
            deletion_results.append(
                _execute_option_automaton(
                    executor,
                    child,
                    initial,
                    deletion,
                    complete_variable,
                )[0]
            )
        reverse_results.append(
            _execute_option_automaton(
                executor,
                child,
                initial,
                tuple(reversed(actions)),
                complete_variable,
            )[0]
        )
    checks = {
        "all_parent_programs_compiled": (
            len(child_programs) == len(selected_parents) > 0
        ),
        "all_programs_owned_by_option_provider": owner_mass >= 1.0 - 1e-12,
        "context_evidence_count": len(option_payload["context_bindings"]) >= 2,
        "deletion_controls_fail": not any(deletion_results),
        "option_checksum_bound": option_payload["option_checksum"] == option.checksum,
        "owner_mass_gate": owner_mass >= protocol.minimum_posterior_owner_mass,
        "positive_automaton_completes": all(positive_results),
        "reverse_control_fails": not any(reverse_results),
        "shadow_only_no_environment_calls": True,
    }
    passed = all(checks.values())
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {output_dir}")
    storage = RunStorageBudget(output_dir, protocol.maximum_artifact_bytes_per_run)
    compiled_path = output_dir / "compiled_option_registry.json"
    programs_path = output_dir / "option_programs.json"
    posterior_path = output_dir / "posterior_snapshot.json"
    report_path = output_dir / "shadow_compile_report.json"
    receipt_path = output_dir / "shadow_compile_receipt.json"
    _write_json_once(compiled_path, compiled.to_dict(), storage_budget=storage)
    _write_json_once(
        programs_path,
        {
            "format_version": "sage-t12.4a.3-option-programs-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "programs": [program.to_dict() for program in child_programs],
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    snapshot = child_posterior.snapshot()
    _write_json_once(posterior_path, snapshot, storage_budget=storage)
    report = {
        "checks": checks,
        "format_version": "sage-t12.4a.3-shadow-compile-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": {
            "compiled_program_count": len(child_programs),
            "context_evidence_count": len(option_payload["context_bindings"]),
            "deletion_completion_count": sum(deletion_results),
            "option_action_count": len(actions),
            "posterior_owner_mass": owner_mass,
            "posterior_particle_count": len(child_posterior.particles),
            "reverse_completion_count": sum(reverse_results),
        },
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": (
            "PASS_T12_4A_3_SHADOW_COMPILE_GATE"
            if passed
            else "FAIL_T12_4A_3_SHADOW_COMPILE_GATE"
        ),
    }
    report["storage"] = storage.snapshot()
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = option_minimization_receipt(
        manifest=manifest,
        phase="shadow_compile",
        passed=passed,
        status=report["status"],
        metrics={
            **report["metrics"],
            "checks": checks,
            "storage": storage.snapshot(),
        },
        parent_receipt=ablation_receipt,
        artifacts={
            "compiled_option_registry": {
                "path": str(compiled_path.resolve()),
                "sha256": _file_sha256(compiled_path),
            },
            "option_programs": {
                "path": str(programs_path.resolve()),
                "sha256": _file_sha256(programs_path),
            },
            "posterior_snapshot": {
                "path": str(posterior_path.resolve()),
                "sha256": _file_sha256(posterior_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(receipt_path, receipt, storage_budget=storage)
    return receipt


def option_minimization_status(
    manifest_path: Path,
    *,
    receipt_path: Path | None = None,
    compile_receipt_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_option_minimization_manifest(manifest_path)
    receipt: dict[str, Any] | None = None
    compile_receipt: dict[str, Any] | None = None
    if receipt_path is not None and receipt_path.exists():
        receipt = load_option_minimization_receipt(receipt_path)
    if compile_receipt_path is not None and compile_receipt_path.exists():
        compile_receipt = load_option_minimization_receipt(
            compile_receipt_path,
            manifest=manifest,
        )

    ablation_passed = bool(
        receipt
        and receipt.get("passed")
        and receipt.get("phase") == "option_ablation"
        and receipt.get("status") == "PASS_T12_4A_3_OPTION_ABLATION_GATE"
    )
    compile_passed = bool(
        ablation_passed
        and compile_receipt
        and compile_receipt.get("passed")
        and compile_receipt.get("phase") == "shadow_compile"
        and compile_receipt.get("status") == "PASS_T12_4A_3_SHADOW_COMPILE_GATE"
        and compile_receipt.get("parent_receipt_checksum")
        == receipt.get("receipt_checksum")
    )
    return {
        "firewall": {
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "option_ablation_authorized": manifest["firewall"][
                "option_ablation_authorized"
            ],
            "option_compilation_authorized": ablation_passed,
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_4a_4_transfer_freeze_authorized": compile_passed,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.4a.3-option-minimization-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": compile_passed,
        "parent_t12_4a_2_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "receipt": receipt,
        "shadow_compile_receipt": compile_receipt,
    }
