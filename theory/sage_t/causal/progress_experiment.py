"""Offline compilation and source replication for T12.5 causal progress."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import CausalProgram, causal_program_from_dict
from .experiment import RunStorageBudget, _file_sha256, _read_json, _write_json_once
from .option_contracts import ContractedCausalOption
from .progress import (
    CausalProgressActionEvaluator,
    CausalProgressExecutor,
    CausalProgressProgram,
    JointCausalProgressPosterior,
    ProgressEvidence,
    rival_progress_programs,
)
from .progress_protocol import (
    CausalProgressProtocol,
    _resolve_bound,
    causal_progress_receipt,
    load_causal_progress_manifest,
    load_causal_progress_receipt,
)

PROGRESS_REGISTRY_FORMAT = "sage-t12.5-causal-progress-registry-v1"
PROGRESS_EVIDENCE_FORMAT = "sage-t12.5-progress-evidence-summary-v1"
PROGRESS_REPORT_FORMAT = "sage-t12.5-causal-progress-report-v1"


def _artifact_path(manifest: Mapping[str, Any], name: str, *, root: Path) -> Path:
    return _resolve_bound(str(manifest["inputs"][name]["path"]), root=root)


def _load_inputs(
    manifest: Mapping[str, Any], *, root: Path
) -> tuple[
    ContractedCausalOption,
    tuple[CausalProgram, ...],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    registry = ContractedCausalOption.from_dict(
        _read_json(_artifact_path(manifest, "contracted_option_registry", root=root))
    )
    programs_payload = _read_json(
        _artifact_path(manifest, "contracted_option_programs", root=root)
    )
    programs = tuple(
        causal_program_from_dict(dict(item))
        for item in programs_payload.get("programs", ())
    )
    posterior = _read_json(
        _artifact_path(manifest, "contracted_posterior", root=root)
    )
    applicability = _read_json(
        _artifact_path(manifest, "applicability_trials", root=root)
    )
    ablations = _read_json(
        _artifact_path(manifest, "ablation_trials", root=root)
    )
    minimal_option = _read_json(
        _artifact_path(manifest, "minimal_option", root=root)
    )
    return registry, programs, posterior, applicability, ablations, minimal_option


def _owner_probabilities(
    programs: Sequence[CausalProgram], posterior_payload: Mapping[str, Any]
) -> dict[str, float]:
    known = {program.canonical_hash for program in programs}
    particles = tuple(dict(item) for item in posterior_payload["posterior"]["particles"])
    if {str(item["program_hash"]) for item in particles} != known:
        raise ValueError("T12.5 posterior/program ownership mismatch")
    probabilities = {
        str(item["program_hash"]): float(item["probability"]) for item in particles
    }
    if len(probabilities) != len(programs):
        raise ValueError("T12.5 posterior contains duplicate program owners")
    return probabilities


def _typed_transition_evidence(
    payload: Mapping[str, Any], *, lineage_seed: int
) -> tuple[ProgressEvidence, ...]:
    selected: dict[tuple[str, str], ProgressEvidence] = {}
    for raw in payload.get("trials", ()):
        item = dict(raw)
        if int(item.get("lineage_seed", -1)) != int(lineage_seed):
            continue
        if item.get("branch_name") != "option_full":
            continue
        if not item.get("prefix_exact") or not item.get("branch_available"):
            raise ValueError("T12.5 typed evidence lost exact replay integrity")
        context = str(item["context_name"])
        key = (context, str(item["trace_checksum"]))
        trace = tuple(dict(step) for step in item.get("trace", ()))
        selected.setdefault(
            key,
            ProgressEvidence(
                evidence_id=f"typed:{lineage_seed}:{context}:{item['trace_checksum']}",
                lineage_id=f"lineage:{lineage_seed}",
                steps=trace,
                progressed=bool(item["progressed"]),
                modality="observed_typed_transition_trace",
                action_names=tuple(str(step["action_name"]) for step in trace),
            ),
        )
    contexts = {item.evidence_id.split(":")[2] for item in selected.values()}
    if contexts != {"successful_level0", "failed_level1"}:
        raise ValueError("T12.5 needs positive and same-action failed typed contexts")
    return tuple(selected[key] for key in sorted(selected))


def _irrelevant_step(action_name: str, position: int) -> dict[str, Any]:
    return {
        "action_name": str(action_name).upper(),
        "delta": {"mechanism": {}},
        "position": int(position),
    }


def _order_intervention_evidence(
    payload: Mapping[str, Any],
    *,
    lineage_seed: int,
    minimal_indices: Sequence[int],
    templates: Sequence[CausalProgressProgram],
) -> tuple[ProgressEvidence, ...]:
    ordered = next(item for item in templates if item.progress_kind == "ordered_effects")
    if len(minimal_indices) != len(ordered.milestones):
        raise ValueError("T12.5 minimal indices do not match causal milestones")
    milestone_by_index = {
        int(source_index): milestone
        for source_index, milestone in zip(minimal_indices, ordered.milestones)
    }
    selected: dict[tuple[str, tuple[int, ...]], ProgressEvidence] = {}
    for raw in payload.get("trials", ()):
        item = dict(raw)
        if int(item.get("context_seed", -1)) != int(lineage_seed):
            continue
        if not item.get("prefix_exact") or not item.get("branch_available"):
            raise ValueError("T12.5 order evidence lost exact replay integrity")
        indices = tuple(int(value) for value in item.get("selected_indices", ()))
        kind = str(item["branch_kind"])
        key = (kind, indices)
        action_names = tuple(map(str, item.get("action_names", ())))
        steps = []
        for position, source_index in enumerate(indices):
            milestone = milestone_by_index.get(source_index)
            action_name = action_names[position] if position < len(action_names) else "ACTION0"
            steps.append(
                _irrelevant_step(action_name, position)
                if milestone is None
                else milestone.as_step(position=position, action_name=action_name)
            )
        selected.setdefault(
            key,
            ProgressEvidence(
                evidence_id=(
                    f"order:{lineage_seed}:{kind}:"
                    + ("-".join(map(str, indices)) or "empty")
                ),
                lineage_id=f"lineage:{lineage_seed}",
                steps=tuple(steps),
                progressed=bool(item["progressed"]),
                modality="exact_replay_action_order_proxy",
                action_names=action_names,
            ),
        )
    if len(selected) != 65:
        raise ValueError(
            f"T12.5 requires 64 deletion masks plus reverse, got {len(selected)}"
        )
    return tuple(selected[key] for key in sorted(selected))


def _accuracy(
    predictions: Sequence[bool], evidence: Sequence[ProgressEvidence]
) -> float:
    if len(predictions) != len(evidence) or not evidence:
        raise ValueError("invalid T12.5 accuracy inputs")
    return sum(
        prediction is item.progressed
        for prediction, item in zip(predictions, evidence)
    ) / len(evidence)


def _program_accuracy(
    program: CausalProgressProgram, evidence: Sequence[ProgressEvidence]
) -> float:
    executor = CausalProgressExecutor()
    return _accuracy(
        [executor.evaluate_trace(program, item.steps).predicted_success for item in evidence],
        evidence,
    )


def _contains_actions(actions: Sequence[str], target: Sequence[str]) -> bool:
    if not target or len(actions) < len(target):
        return False
    return any(
        tuple(actions[index : index + len(target)]) == tuple(target)
        for index in range(len(actions) - len(target) + 1)
    )


def _baseline_metrics(
    evidence: Sequence[ProgressEvidence],
    templates: Sequence[CausalProgressProgram],
    target_actions: Sequence[str],
) -> dict[str, float]:
    by_kind = {item.progress_kind: item for item in templates}
    metrics = {
        f"{kind}_accuracy": _program_accuracy(by_kind[kind], evidence)
        for kind in ("terminal_only", "change_count", "unordered_effects")
    }
    metrics["action_only_accuracy"] = _accuracy(
        [_contains_actions(item.action_names, target_actions) for item in evidence],
        evidence,
    )
    # A state-only initiation classifier sees the same successful anchor for
    # every ablation branch, so it cannot distinguish useful histories there.
    metrics["state_only_initiation_accuracy"] = _accuracy(
        [
            item.modality == "exact_replay_action_order_proxy"
            or "successful_level0" in item.evidence_id
            for item in evidence
        ],
        evidence,
    )
    return metrics


def _strictly_increasing(values: Sequence[float]) -> bool:
    return all(after > before + 1e-12 for before, after in zip(values, values[1:]))


def _max_mass_error(
    before: Mapping[str, float], after: Mapping[str, float]
) -> float:
    return max(abs(float(before[key]) - float(after[key])) for key in before)


def _forbidden_semantics(programs: Sequence[CausalProgressProgram]) -> bool:
    payload = json.dumps(
        [
            {
                "goal_predicate": item.goal_predicate,
                "milestones": [
                    milestone.semantic_payload for milestone in item.milestones
                ],
                "progress_kind": item.progress_kind,
            }
            for item in programs
        ],
        sort_keys=True,
    ).lower()
    return any(
        token in payload
        for token in (
            "exact_hash",
            "absolute_coordinate",
            "entity_id",
            "game_id",
            "pixel",
            "raw_fact_token",
        )
    )


def _write_budgeted(
    path: Path, payload: Mapping[str, Any], budget: RunStorageBudget
) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    budget.reserve(len(encoded) + 1)
    _write_json_once(path, payload)


def compile_causal_progress(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_causal_progress_manifest(manifest_path, root=repo_root)
    protocol = CausalProgressProtocol(**dict(manifest["protocol"]))
    if not manifest["firewall"].get("causal_progress_compile_authorized", False):
        raise ValueError("T12.5 manifest does not authorize compilation")

    registry, programs, parent_posterior, applicability, ablations, minimal = (
        _load_inputs(manifest, root=repo_root)
    )
    if len(programs) != protocol.maximum_owner_programs:
        raise ValueError("T12.5 owner-program count changed")
    if len(registry.effect_contracts) != protocol.expected_milestone_count:
        raise ValueError("T12.5 milestone count changed")
    owners = _owner_probabilities(programs, parent_posterior)
    program_by_hash = {item.canonical_hash: item for item in programs}
    if set(registry.owner_program_hashes) != set(program_by_hash):
        raise ValueError("T12.5 contracted option owner set changed")

    evidence_ids = (
        f"t12_4a_4c:{manifest['parent']['receipt']['receipt_checksum']}",
        f"t12_4a_3:{manifest['parent']['ablation_receipt']['receipt_checksum']}",
    )
    progress_programs = tuple(
        progress
        for owner_hash, owner in sorted(program_by_hash.items())
        for progress in rival_progress_programs(
            owner_program_hash=owner_hash,
            effect_contracts=registry.effect_contracts,
            goal_predicate=owner.goal.success_predicate,
            failure_predicate=owner.goal.failure_predicate,
            evidence_ids=evidence_ids,
        )
    )
    if len(progress_programs) != protocol.maximum_joint_particles:
        raise ValueError("T12.5 joint progress-program count changed")
    templates = tuple(
        item
        for item in progress_programs
        if item.owner_program_hash == sorted(program_by_hash)[0]
    )
    minimal_indices = tuple(int(value) for value in minimal["minimality"]["selected_indices"])
    target_actions = tuple(
        str(item["action_name"]) for item in minimal["option"]["steps"]
    )

    train_evidence = (
        *_typed_transition_evidence(
            applicability, lineage_seed=protocol.induction_lineage_seed
        ),
        *_order_intervention_evidence(
            ablations,
            lineage_seed=protocol.induction_lineage_seed,
            minimal_indices=minimal_indices,
            templates=templates,
        ),
    )
    replication_evidence = (
        *_typed_transition_evidence(
            applicability, lineage_seed=protocol.replication_lineage_seed
        ),
        *_order_intervention_evidence(
            ablations,
            lineage_seed=protocol.replication_lineage_seed,
            minimal_indices=minimal_indices,
            templates=templates,
        ),
    )
    posterior = JointCausalProgressPosterior.from_factorized_prior(
        owner_probabilities=owners,
        progress_programs=progress_programs,
        mdl_beta=protocol.mdl_beta,
        match_probability=protocol.likelihood_match_probability,
    )
    owner_mass_before = posterior.mass_by_owner()
    posterior.update_many(train_evidence)
    induction_mass_by_kind = posterior.mass_by_kind()

    ordered = next(item for item in templates if item.progress_kind == "ordered_effects")
    ordered_accuracy = _program_accuracy(ordered, replication_evidence)
    posterior_accuracy = _accuracy(
        [posterior.success_probability(item.steps) >= 0.5 for item in replication_evidence],
        replication_evidence,
    )
    baselines = _baseline_metrics(replication_evidence, templates, target_actions)
    # Replication metrics above are computed before the replication lineage can
    # update any weight.  Once measured, that independent evidence is allowed
    # to consolidate the common posterior used by subsequent decisions.
    posterior.update_many(replication_evidence)
    owner_mass_error = _max_mass_error(owner_mass_before, posterior.mass_by_owner())
    mass_by_kind = posterior.mass_by_kind()

    positive = next(
        item
        for item in replication_evidence
        if item.modality == "observed_typed_transition_trace" and item.progressed
    )
    failed = next(
        item
        for item in replication_evidence
        if item.modality == "observed_typed_transition_trace" and not item.progressed
    )
    positive_prefix_scores = [0.0] + [
        posterior.expected_potential(positive.steps[:length])
        for length in range(1, len(positive.steps) + 1)
    ]
    failed_prefix_scores = [0.0] + [
        posterior.expected_potential(failed.steps[:length])
        for length in range(1, len(failed.steps) + 1)
    ]
    relabeled = tuple(
        {**dict(step), "action_name": f"RELABEL_{index}"}
        for index, step in enumerate(positive.steps)
    )
    relabel_error = abs(
        posterior.expected_potential(positive.steps)
        - posterior.expected_potential(relabeled)
    )
    next_index = 2
    correct_next = ordered.milestones[next_index].as_step(
        action_name="ACTION_CORRECT"
    )
    out_of_order = ordered.milestones[-1].as_step(action_name="ACTION_DISTRACTOR")
    irrelevant = _irrelevant_step("ACTION_IRRELEVANT", 0)
    ranking = CausalProgressActionEvaluator.rank(
        posterior,
        {
            "correct_next_effect": (correct_next,),
            "out_of_order_effect": (out_of_order,),
            "irrelevant_effect": (irrelevant,),
        },
        prefix=positive.steps[:next_index],
    )

    checks = {
        "all_source_replays_exact": True,
        "causal_progress_is_history_dependent": _strictly_increasing(
            positive_prefix_scores
        ),
        "failed_same_action_trace_stays_flat": max(failed_prefix_scores) <= 1e-12,
        "action_relabeling_preserves_progress": relabel_error <= 1e-12,
        "correct_next_effect_ranked_first": ranking[0][0] == "correct_next_effect",
        "ordered_program_replicates_perfectly": (
            ordered_accuracy >= protocol.minimum_replication_accuracy
        ),
        "posterior_replicates_perfectly": (
            posterior_accuracy >= protocol.minimum_replication_accuracy
        ),
        "ordered_posterior_mass_concentrated": (
            mass_by_kind["ordered_effects"]
            >= protocol.minimum_ordered_posterior_mass
        ),
        "ordered_beats_every_preregistered_baseline": ordered_accuracy
        > max(baselines.values()),
        "parent_world_program_mass_preserved": (
            owner_mass_error <= protocol.maximum_parent_mass_error
        ),
        "joint_particles_are_bounded": len(progress_programs)
        == protocol.maximum_joint_particles,
        "one_milestone_per_sealed_effect_contract": len(ordered.milestones)
        == len(registry.effect_contracts),
        "progress_semantics_contain_no_forbidden_input": not _forbidden_semantics(
            progress_programs
        ),
        "evidence_modalities_remain_separated": {
            item.modality for item in (*train_evidence, *replication_evidence)
        }
        == {"observed_typed_transition_trace", "exact_replay_action_order_proxy"},
        "offline_zero_sdk_calls": protocol.maximum_sdk_calls == 0,
    }

    output = Path(output_dir)
    budget = RunStorageBudget(output, protocol.maximum_artifact_bytes_per_run)
    registry_payload = {
        "format_version": PROGRESS_REGISTRY_FORMAT,
        "manifest_checksum": manifest["manifest_checksum"],
        "programs": [item.to_dict() for item in progress_programs],
        "protocol_checksum": manifest["protocol_checksum"],
        "shared_progress_program_count": len(protocol.rival_progress_kinds),
        "virtual_joint_particle_count": len(progress_programs),
    }
    posterior_payload = {
        "manifest_checksum": manifest["manifest_checksum"],
        "posterior": posterior.snapshot(),
        "protocol_checksum": manifest["protocol_checksum"],
    }
    evidence_payload = {
        "format_version": PROGRESS_EVIDENCE_FORMAT,
        "manifest_checksum": manifest["manifest_checksum"],
        "claim_boundary": manifest["claim_boundary"],
        "induction": [
            {
                "action_names": list(item.action_names),
                "evidence_id": item.evidence_id,
                "lineage_id": item.lineage_id,
                "modality": item.modality,
                "progressed": item.progressed,
                "step_count": len(item.steps),
            }
            for item in train_evidence
        ],
        "replication": [
            {
                "action_names": list(item.action_names),
                "evidence_id": item.evidence_id,
                "lineage_id": item.lineage_id,
                "modality": item.modality,
                "progressed": item.progressed,
                "step_count": len(item.steps),
            }
            for item in replication_evidence
        ],
        "protocol_checksum": manifest["protocol_checksum"],
    }
    report = {
        "format_version": PROGRESS_REPORT_FORMAT,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "claim_boundary": manifest["claim_boundary"],
        "metrics": {
            "baselines": baselines,
            "checks": checks,
            "failed_prefix_scores": failed_prefix_scores,
            "induction_evidence_count": len(train_evidence),
            "joint_particle_count": len(progress_programs),
            "maximum_parent_mass_error": owner_mass_error,
            "induction_posterior_mass_by_kind": induction_mass_by_kind,
            "ordered_replication_accuracy": ordered_accuracy,
            "posterior_mass_by_kind": mass_by_kind,
            "posterior_replication_accuracy": posterior_accuracy,
            "positive_prefix_scores": positive_prefix_scores,
            "ranking_probe": [list(item) for item in ranking],
            "relabeling_value_error": relabel_error,
            "replication_evidence_count": len(replication_evidence),
            "sdk_calls_used": 0,
        },
    }
    paths = {
        "program_registry": output / "progress_program_registry.json",
        "posterior": output / "joint_progress_posterior.json",
        "evidence": output / "progress_evidence_summary.json",
        "report": output / "causal_progress_report.json",
    }
    for name, payload in (
        ("program_registry", registry_payload),
        ("posterior", posterior_payload),
        ("evidence", evidence_payload),
        ("report", report),
    ):
        _write_budgeted(paths[name], payload, budget)
    storage = {
        "maximum_artifact_bytes": protocol.maximum_artifact_bytes_per_run,
        "remaining_artifact_bytes": (
            protocol.maximum_artifact_bytes_per_run - budget.used_bytes
        ),
        "used_artifact_bytes": budget.used_bytes,
        "within_budget": budget.used_bytes <= protocol.maximum_artifact_bytes_per_run,
    }
    checks["storage_within_budget"] = storage["within_budget"]
    report["metrics"]["storage"] = storage
    # The report was intentionally written before storage accounting.  The
    # receipt is the authoritative signed carrier of final storage metrics.
    passed = all(checks.values())
    receipt = causal_progress_receipt(
        manifest=manifest,
        phase="compile",
        passed=passed,
        status=(
            "PASS_T12_5_CAUSAL_PROGRESS_GATE"
            if passed
            else "FAIL_T12_5_CAUSAL_PROGRESS_GATE"
        ),
        metrics={**report["metrics"], "checks": checks, "storage": storage},
        artifacts={
            name: {"path": str(path.resolve()), "sha256": _file_sha256(path)}
            for name, path in paths.items()
        },
    )
    _write_budgeted(output / "causal_progress_receipt.json", receipt, budget)
    return receipt


def causal_progress_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_causal_progress_manifest(manifest_path, root=repo_root)
    receipt = (
        None
        if receipt_path is None or not Path(receipt_path).is_file()
        else load_causal_progress_receipt(
            receipt_path, manifest=manifest, root=repo_root
        )
    )
    passed = bool(
        receipt
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_5_CAUSAL_PROGRESS_GATE"
    )
    return {
        "format_version": "sage-t12.5-causal-progress-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_4a_4c_status": manifest["parent"]["receipt"]["status"],
        "receipt": (
            None
            if receipt is None
            else {
                "passed": receipt["passed"],
                "phase": receipt["phase"],
                "receipt_checksum": receipt["receipt_checksum"],
                "status": receipt["status"],
            }
        ),
        "next_phase_authorized": passed,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "neural_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "option_control_authorized": False,
            "causal_progress_shadow_experiment_authorized": passed,
            "causal_progress_control_authorized": False,
            "t12_6_freeze_authorized": False,
        },
        "claim_boundary": manifest["claim_boundary"],
    }


__all__ = ["causal_progress_status", "compile_causal_progress"]
