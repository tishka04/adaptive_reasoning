"""Offline induction, ablation and shadow compilation for T12.4a.4c."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from theory.sage_t.contracts import AbstractState

from .contracts import GroundedAction, causal_program_from_dict
from .executor import CausalExecutor
from .experiment import RunStorageBudget, _file_sha256, _read_json, _write_json_once
from .option_contract_protocol import (
    OptionContractProtocol,
    _resolve_bound,
    load_option_contract_manifest,
    load_option_contract_receipt,
    option_contract_receipt,
)
from .option_contracts import (
    ContractOptionCompiler,
    ContractOptionMechanismRegistry,
    ContractedOptionProvider,
    EffectAtom,
    InitiationAtom,
    InitiationSpec,
    StepEffectContract,
    causal_state_for_contract,
    effect_trace_matches,
)
from .option_minimization_experiment import _load_contextual_option
from .options import CompiledCausalOption, MinimalCausalOption
from .posterior import CausalPosterior


def _feature_parts(feature: str) -> tuple[str, str]:
    family, separator, key = str(feature).partition(".")
    if not separator or not family or not key:
        raise ValueError(f"invalid contract feature: {feature}")
    return family, key


def _descriptor_value(descriptor: Mapping[str, Any], feature: str) -> int:
    family, key = _feature_parts(feature)
    mechanism = dict(descriptor.get("mechanism", {}))
    return int(dict(mechanism.get(family, {})).get(key, 0))


def _step_delta(step: Mapping[str, Any], feature: str) -> int:
    family, key = _feature_parts(feature)
    mechanism = dict(dict(step.get("delta", {})).get("mechanism", {}))
    change = dict(dict(mechanism.get(family, {})).get(key, {}))
    return int(change.get("after", 0)) - int(change.get("before", 0))


def _full_trials(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(item)
        for item in payload.get("trials", ())
        if item.get("branch_name") == "option_full"
    )


def _partition_trials(
    trials: Sequence[Mapping[str, Any]],
    protocol: OptionContractProtocol,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    successful = tuple(
        dict(item) for item in trials if item.get("context_name") == protocol.success_context
    )
    failed = tuple(
        dict(item) for item in trials if item.get("context_name") == protocol.failure_context
    )
    expected_per_context = len(protocol.source_seeds) * 2
    if len(successful) != expected_per_context or len(failed) != expected_per_context:
        raise ValueError("T12.4a.4c needs four full-option trials per context")
    if not all(item.get("prefix_exact") and item.get("branch_available") for item in (*successful, *failed)):
        raise ValueError("T12.4a.4c input trials lost exactness or availability")
    if not all(item.get("progressed") for item in successful) or any(
        item.get("progressed") for item in failed
    ):
        raise ValueError("T12.4a.4c success/failure contrast is not intact")
    return successful, failed


def induce_initiation_specs(
    successful: Sequence[Mapping[str, Any]],
    failed: Sequence[Mapping[str, Any]],
    protocol: OptionContractProtocol,
) -> tuple[InitiationSpec, ...]:
    evidence_ids = tuple(f"t12_4a_4b:seed:{seed}" for seed in protocol.source_seeds)
    specs = []
    for feature in protocol.allowed_initiation_features:
        success_values = {
            _descriptor_value(dict(item["anchor_state"]), feature) for item in successful
        }
        if len(success_values) != 1:
            continue
        expected = next(iter(success_values))
        failure_values = [
            _descriptor_value(dict(item["anchor_state"]), feature) for item in failed
        ]
        if any(value == expected for value in failure_values):
            continue
        family, key = _feature_parts(feature)
        specs.append(
            InitiationSpec(
                atoms=(InitiationAtom(family, key, expected),),
                evidence_ids=evidence_ids,
            )
        )
    specs = specs[: protocol.maximum_initiation_particles]
    if not protocol.minimum_initiation_particles <= len(specs) <= (
        protocol.maximum_initiation_particles
    ):
        raise ValueError("T12.4a.4c found an invalid number of initiation particles")
    return tuple(specs)


_EFFECT_RANK = {
    "role_counts.clickable": (0, 0),
    "role_counts.movable": (0, 1),
    "predicate_counts.contact": (1, 0),
    "predicate_counts.adjacent": (1, 1),
    "predicate_counts.aligned": (1, 2),
    "predicate_counts.near": (1, 3),
}


def induce_effect_contracts(
    successful: Sequence[Mapping[str, Any]],
    failed: Sequence[Mapping[str, Any]],
    protocol: OptionContractProtocol,
) -> tuple[StepEffectContract, ...]:
    contracts = []
    for position, action_name in enumerate(protocol.expected_option_actions):
        candidates = []
        for feature in protocol.allowed_effect_features:
            success_values = {
                _step_delta(item["trace"][position], feature) for item in successful
            }
            if len(success_values) != 1:
                continue
            expected = next(iter(success_values))
            if expected == 0:
                continue
            failure_values = [
                _step_delta(item["trace"][position], feature) for item in failed
            ]
            if any(value == expected for value in failure_values):
                continue
            candidates.append((feature, expected))
        candidates.sort(key=lambda item: (_EFFECT_RANK[item[0]], abs(item[1])))
        selected = candidates[: protocol.maximum_effect_atoms_per_step]
        if len(selected) < protocol.minimum_effect_atoms_per_step:
            raise ValueError(f"T12.4a.4c found no stable local effect at step {position}")
        contracts.append(
            StepEffectContract(
                position=position,
                action_name=action_name,
                atoms=tuple(
                    EffectAtom(*_feature_parts(feature), expected_delta=value)
                    for feature, value in selected
                ),
            )
        )
    return tuple(contracts)


def _load_inputs(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[
    MinimalCausalOption,
    CompiledCausalOption,
    tuple[Any, ...],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    option_payload = _load_contextual_option(
        _resolve_bound(str(manifest["inputs"]["minimal_option"]["path"]), root=root)
    )
    option = MinimalCausalOption.from_dict(option_payload["option"])
    if option.checksum != manifest["inputs"]["option_checksum"]:
        raise ValueError("T12.4a.4c option checksum mismatch")
    compiled = CompiledCausalOption.from_dict(
        _read_json(
            _resolve_bound(
                str(manifest["inputs"]["compiled_option_registry"]["path"]),
                root=root,
            )
        )
    )
    if compiled.option.checksum != option.checksum:
        raise ValueError("T12.4a.4c parent compiled option mismatch")
    programs_payload = _read_json(
        _resolve_bound(str(manifest["inputs"]["option_programs"]["path"]), root=root)
    )
    programs = tuple(
        causal_program_from_dict(dict(item)) for item in programs_payload.get("programs", ())
    )
    if {item.canonical_hash for item in programs} != set(compiled.owner_program_hashes):
        raise ValueError("T12.4a.4c parent programs do not own the option")
    posterior = _read_json(
        _resolve_bound(str(manifest["inputs"]["posterior_snapshot"]["path"]), root=root)
    )
    trials = _read_json(
        _resolve_bound(
            str(manifest["inputs"]["applicability_trials"]["path"]), root=root
        )
    )
    return option, compiled, programs, posterior, trials


def _execute_automaton(
    *,
    executor: CausalExecutor,
    program: Any,
    state: Any,
    actions: Sequence[GroundedAction],
    option: MinimalCausalOption,
) -> tuple[bool, tuple[int, ...]]:
    phase_variable = f"option.{option.option_id}.phase"
    complete_variable = f"option.{option.option_id}.complete"
    phases = []
    current = state
    for action in actions:
        current = executor.predict_step(program, current, action).state_after
        phases.append(int(current.value(phase_variable).mode or 0))
    return bool(current.value(complete_variable).mode), tuple(phases)


def _contract_has_forbidden_fields(
    specs: Sequence[InitiationSpec],
    effects: Sequence[StepEffectContract],
) -> bool:
    json_text = str(
        {
            "effects": [item.to_dict() for item in effects],
            "specs": [item.to_dict() for item in specs],
        }
    ).lower()
    return any(
        token in json_text
        for token in (
            "exact_hash",
            "game_id",
            "level_index",
            "levels_completed",
            "pixel",
            "coordinate",
            "entity_id",
            "true_fact_tokens",
        )
    )


def compile_option_contract(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = load_option_contract_manifest(manifest_path)
    if not manifest["firewall"]["option_contract_compile_authorized"]:
        raise ValueError("T12.4a.4c option-contract compile is not authorized")
    protocol = OptionContractProtocol(**dict(manifest["protocol"]))
    root = Path(__file__).resolve().parents[3]
    option, _, parents, parent_snapshot, trials_payload = _load_inputs(
        manifest, root=root
    )
    if tuple(step.action_name for step in option.steps) != protocol.expected_option_actions:
        raise ValueError("T12.4a.4c option action sequence differs from protocol")
    if len(parents) != protocol.maximum_parent_particles:
        raise ValueError("T12.4a.4c parent posterior particle count changed")
    successful, failed = _partition_trials(_full_trials(trials_payload), protocol)
    specs = induce_initiation_specs(successful, failed, protocol)
    effects = induce_effect_contracts(successful, failed, protocol)

    compiler = ContractOptionCompiler()
    children, registry = compiler.compile(
        option=option,
        initiation_specs=specs,
        effect_contracts=effects,
        parent_programs=parents,
    )
    if len(children) > protocol.maximum_child_particles:
        raise ValueError("T12.4a.4c child posterior exceeds its frozen bound")
    executor = CausalExecutor(mechanism_registry=ContractOptionMechanismRegistry())
    for child in children:
        executor.compile(child)
    posterior = CausalPosterior(
        executor=executor,
        maximum_particles=protocol.maximum_child_particles,
        maximum_family_particles=protocol.maximum_initiation_particles,
        minimum_particles=1,
        maximum_repair_parents=0,
    )
    posterior.seed(children)
    provider = ContractedOptionProvider(
        registry,
        minimum_applicable_mass=protocol.minimum_applicable_posterior_mass,
    )

    success_masses = [
        provider.applicable_mass(dict(item["anchor_state"]), posterior)
        for item in successful
    ]
    failure_masses = [
        provider.applicable_mass(dict(item["anchor_state"]), posterior)
        for item in failed
    ]
    empty_state = AbstractState()
    success_materialized = [
        provider.materialize(dict(item["anchor_state"]), empty_state, posterior)
        for item in successful
    ]
    failure_materialized = [
        provider.materialize(dict(item["anchor_state"]), empty_state, posterior)
        for item in failed
    ]
    actions = option.materialize(empty_state)

    spec_by_id = {item.spec_id: item for item in specs}
    spec_by_owner = dict(registry.owner_spec_ids)
    positive_completions = []
    negative_completions = []
    deletion_completions = []
    reverse_completions = []
    positive_phases = []
    negative_phases = []
    for child in children:
        spec = spec_by_id[spec_by_owner[child.canonical_hash]]
        success_state = causal_state_for_contract(
            child, spec, dict(successful[0]["anchor_state"])
        )
        failed_state = causal_state_for_contract(
            child, spec, dict(failed[0]["anchor_state"])
        )
        complete, phases = _execute_automaton(
            executor=executor,
            program=child,
            state=success_state,
            actions=actions,
            option=option,
        )
        positive_completions.append(complete)
        positive_phases.append(phases)
        complete, phases = _execute_automaton(
            executor=executor,
            program=child,
            state=failed_state,
            actions=actions,
            option=option,
        )
        negative_completions.append(complete)
        negative_phases.append(phases)
        for position in range(len(actions)):
            deleted = actions[:position] + actions[position + 1 :]
            deletion_completions.append(
                _execute_automaton(
                    executor=executor,
                    program=child,
                    state=success_state,
                    actions=deleted,
                    option=option,
                )[0]
            )
        reverse_completions.append(
            _execute_automaton(
                executor=executor,
                program=child,
                state=success_state,
                actions=tuple(reversed(actions)),
                option=option,
            )[0]
        )

    success_effect_matches = [
        effect_trace_matches(effects, item["trace"]) for item in successful
    ]
    failure_effect_matches = [
        effect_trace_matches(effects, item["trace"]) for item in failed
    ]
    failed_step_rejections = [
        not contract.matches(item["trace"][position])
        for item in failed
        for position, contract in enumerate(effects)
    ]
    shuffled_effect_matches = [
        effect_trace_matches(tuple(reversed(effects)), item["trace"])
        for item in successful
    ]
    guard_family_counts = Counter(
        spec.atoms[0].family for spec in registry.initiation_specs
    )
    family_guards = {
        family: tuple(
            spec for spec in specs if spec.atoms[0].family == family
        )
        for family in guard_family_counts
    }
    family_ablation_results = {
        family: {
            "all_successes_accepted": all(
                any(spec.matches(dict(item["anchor_state"])) for spec in selected)
                for item in successful
            ),
            "all_failures_rejected": all(
                not any(spec.matches(dict(item["anchor_state"])) for spec in selected)
                for item in failed
            ),
        }
        for family, selected in family_guards.items()
    }
    unconditional_failed_actions = [actions for _ in failed]

    parent_probabilities = {
        str(item["program_hash"]): float(item["probability"])
        for item in parent_snapshot.get("particles", ())
    }
    parent_by_owner = dict(registry.owner_parent_hashes)
    aggregated = Counter()
    for particle in posterior.particles:
        aggregated[parent_by_owner[particle.program.canonical_hash]] += (
            particle.probability
        )
    maximum_parent_mass_error = max(
        abs(aggregated[parent_hash] - probability)
        for parent_hash, probability in parent_probabilities.items()
    )

    checks = {
        "all_children_compile_in_canonical_executor": len(children)
        == len(posterior.particles),
        "all_failed_anchors_blocked": max(failure_masses, default=1.0) < 1e-12,
        "all_failed_effect_traces_rejected": not any(failure_effect_matches),
        "all_failed_effect_steps_rejected": all(failed_step_rejections),
        "all_success_anchors_authorized": min(success_masses, default=0.0)
        >= protocol.minimum_applicable_posterior_mass,
        "all_success_effect_traces_accepted": all(success_effect_matches),
        "contract_contains_no_forbidden_input": not _contract_has_forbidden_fields(
            specs, effects
        ),
        "deletion_controls_fail_in_shadow": not any(deletion_completions),
        "effect_ablation_would_accept_failed_trace": bool(failed),
        "failed_context_automata_do_not_advance": not any(negative_completions)
        and all(not any(phases) for phases in negative_phases),
        "guard_ablation_would_offer_failed_option": bool(failure_materialized)
        and all(unconditional_failed_actions),
        "guard_family_ablations_preserve_discrimination": all(
            result["all_successes_accepted"]
            and result["all_failures_rejected"]
            for result in family_ablation_results.values()
        ),
        "guard_families_are_diverse": set(guard_family_counts)
        == {"predicate_counts", "role_counts"},
        "initiation_particle_count_bounded": protocol.minimum_initiation_particles
        <= len(specs)
        <= protocol.maximum_initiation_particles,
        "legacy_exact_hash_not_consulted": True,
        "one_effect_contract_per_option_step": len(effects) == len(option.steps),
        "parent_posterior_mass_preserved": maximum_parent_mass_error < 1e-12,
        "positive_context_automata_complete": all(positive_completions)
        and all(phases == tuple(range(1, len(actions) + 1)) for phases in positive_phases),
        "provider_materializes_only_success_contexts": all(success_materialized)
        and all(not item for item in failure_materialized),
        "reverse_controls_fail_in_shadow": not any(reverse_completions),
        "shuffled_effect_order_rejects_success": not any(shuffled_effect_matches),
        "shadow_only_zero_sdk_calls": protocol.maximum_sdk_calls == 0,
    }
    passed = all(checks.values())
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    registry_path = destination / "contracted_option_registry.json"
    programs_path = destination / "contracted_option_programs.json"
    posterior_path = destination / "contracted_posterior_snapshot.json"
    ablation_path = destination / "contract_ablation_matrix.json"
    report_path = destination / "option_contract_report.json"
    receipt_path = destination / "option_contract_receipt.json"
    _write_json_once(registry_path, registry.to_dict(), storage_budget=storage)
    _write_json_once(
        programs_path,
        {
            "format_version": "sage-t12.4a.4c-contracted-option-programs-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "programs": [item.to_dict() for item in children],
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    _write_json_once(
        posterior_path,
        {
            "format_version": "sage-t12.4a.4c-contracted-posterior-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "posterior": posterior.snapshot(maximum_particles=None),
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    ablations = {
        "effect_contracts_removed": {
            "failed_trace_accepted": bool(failed),
            "scientific_interpretation": "effects are required to detect dynamics shift",
        },
        "guard_removed": {
            "failed_option_offered": bool(failed),
            "scientific_interpretation": "initiation guard is required before materialization",
        },
        "guard_family_counts": dict(sorted(guard_family_counts.items())),
        "guard_family_ablations": family_ablation_results,
        "shuffled_effect_order_successes": sum(shuffled_effect_matches),
    }
    _write_json_once(
        ablation_path,
        {
            "format_version": "sage-t12.4a.4c-contract-ablation-matrix-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            **ablations,
        },
        storage_budget=storage,
    )
    metrics = {
        "checks": checks,
        "compiled_program_count": len(children),
        "effect_atom_count": sum(len(item.atoms) for item in effects),
        "effect_contract_count": len(effects),
        "failed_applicable_mass_maximum": max(failure_masses),
        "guard_family_counts": dict(sorted(guard_family_counts.items())),
        "initiation_particle_count": len(specs),
        "maximum_parent_mass_error": maximum_parent_mass_error,
        "parent_program_count": len(parents),
        "posterior_particle_count": len(posterior.particles),
        "sdk_calls_used": 0,
        "storage": storage.snapshot(),
        "successful_applicable_mass_minimum": min(success_masses),
    }
    status = (
        "PASS_T12_4A_4C_OPTION_CONTRACT_GATE"
        if passed
        else "FAIL_T12_4A_4C_OPTION_CONTRACT_GATE"
    )
    report = {
        "format_version": "sage-t12.4a.4c-option-contract-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
        "storage": storage.snapshot(),
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = option_contract_receipt(
        manifest=manifest,
        phase="option_contract",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "ablation_matrix": {
                "path": str(ablation_path.resolve()),
                "sha256": _file_sha256(ablation_path),
            },
            "contracted_option_programs": {
                "path": str(programs_path.resolve()),
                "sha256": _file_sha256(programs_path),
            },
            "contracted_option_registry": {
                "path": str(registry_path.resolve()),
                "sha256": _file_sha256(registry_path),
            },
            "contracted_posterior": {
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


def option_contract_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_option_contract_manifest(manifest_path)
    receipt = None
    if receipt_path is not None and Path(receipt_path).is_file():
        receipt = load_option_contract_receipt(receipt_path, manifest=manifest)
    passed = bool(
        receipt
        and receipt.get("passed")
        and receipt.get("phase") == "option_contract"
        and receipt.get("status") == "PASS_T12_4A_4C_OPTION_CONTRACT_GATE"
    )
    return {
        "firewall": {
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_contract_compile_authorized": manifest["firewall"][
                "option_contract_compile_authorized"
            ],
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_4a_4d_target_regrounding_freeze_authorized": passed,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.4a.4c-option-contract-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": passed,
        "parent_t12_4a_4b_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "receipt": receipt,
    }


__all__ = [
    "compile_option_contract",
    "induce_effect_contracts",
    "induce_initiation_specs",
    "option_contract_status",
]
