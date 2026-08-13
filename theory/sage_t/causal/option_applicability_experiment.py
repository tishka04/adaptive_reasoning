"""Paired symbolic audit for the failed T12.4a.4 option transfer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from theory.sage.live_prefix_counterfactual_collector import (
    _step_env_action,
    select_live_action,
    snapshot_frame,
    state_signature_from_frame,
)
from theory.sage_t.contracts import AbstractEntity, AbstractState, GroundFact

from .contracts import GroundedAction
from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import _is_terminal, _make_env, _symbolic_state
from .option_applicability_protocol import (
    AUTHORIZED_DIAGNOSES,
    OptionApplicabilityProtocol,
    _checksum,
    _resolve_bound,
    load_option_applicability_manifest,
    load_option_applicability_receipt,
    option_applicability_receipt,
)
from .option_minimization_experiment import _load_contextual_option
from .options import MinimalCausalOption
from .witness_experiment import _execute_expected_steps, _reset_env
from .witness_protocol import ProgressWitness, WitnessStep
from .witness_reconfirmation_protocol import load_reconfirmation_registry

EnvFactory = Callable[[str], Any]
_GOAL_PREDICATES = frozenset({"level_complete", "game_over", "progress"})


@dataclass
class ApplicabilitySdkBudget:
    maximum: int
    used: int = 0

    def consume(self, count: int = 1) -> None:
        additional = max(0, int(count))
        if self.used + additional > self.maximum:
            raise RuntimeError(
                "T12.4a.4b SDK call budget exceeded: "
                f"used={self.used} additional={additional} maximum={self.maximum}"
            )
        self.used += additional

    def snapshot(self) -> dict[str, Any]:
        return {
            "maximum_sdk_calls": self.maximum,
            "remaining_sdk_calls": self.maximum - self.used,
            "used_sdk_calls": self.used,
            "within_budget": self.used <= self.maximum,
        }


def _entity_token(entity: AbstractEntity) -> str:
    return _checksum(
        {"attributes": list(entity.attributes), "roles": list(entity.roles)}
    )[:20]


def _fact_token(
    fact: GroundFact,
    *,
    entity_tokens: Mapping[str, str],
) -> str:
    terms = tuple(entity_tokens.get(term, f"literal:{term}") for term in fact.terms)
    return f"{fact.predicate}|{'|'.join(terms) if terms else '-'}|{fact.value}"


def _state_descriptor(
    state: AbstractState,
    *,
    exact_hash: str,
    level: int,
    game_state: str,
) -> dict[str, Any]:
    entity_by_id = {entity.entity_id: _entity_token(entity) for entity in state.entities}
    entities = sorted(entity_by_id.values())
    true_facts = sorted(
        _fact_token(fact, entity_tokens=entity_by_id) for fact in state.true_facts
    )
    false_facts = sorted(
        _fact_token(fact, entity_tokens=entity_by_id) for fact in state.false_facts
    )
    role_counts = Counter(role for entity in state.entities for role in entity.roles)
    attribute_counts = Counter(
        f"{key}={value}"
        for entity in state.entities
        for key, value in entity.attributes
    )
    predicate_counts = Counter(fact.predicate for fact in state.true_facts)
    counters = {key: float(value) for key, value in state.counters}
    full = {
        "attribute_counts": dict(sorted(attribute_counts.items())),
        "counters": dict(sorted(counters.items())),
        "entity_tokens": entities,
        "false_fact_tokens": false_facts,
        "predicate_counts": dict(sorted(predicate_counts.items())),
        "registers": [list(item) for item in state.registers],
        "regime_index": state.regime_index,
        "role_counts": dict(sorted(role_counts.items())),
        "topology": [list(item) for item in state.topology],
        "true_fact_tokens": true_facts,
    }
    mechanism = {
        **full,
        "counters": {
            key: value for key, value in counters.items() if key != "levels_completed"
        },
        "false_fact_tokens": [
            token
            for token in false_facts
            if token.split("|", 1)[0] not in _GOAL_PREDICATES
        ],
        "predicate_counts": {
            key: value
            for key, value in predicate_counts.items()
            if key not in _GOAL_PREDICATES
        },
        "true_fact_tokens": [
            token
            for token in true_facts
            if token.split("|", 1)[0] not in _GOAL_PREDICATES
        ],
    }
    return {
        "abstract_state_signature": state.signature,
        "exact_hash": exact_hash,
        "full": full,
        "full_checksum": _checksum(full),
        "game_state": game_state,
        "level": int(level),
        "mechanism": mechanism,
        "mechanism_checksum": _checksum(mechanism),
    }


def _counter_delta(before: Sequence[str], after: Sequence[str]) -> dict[str, list[str]]:
    before_count = Counter(before)
    after_count = Counter(after)
    added = list((after_count - before_count).elements())
    removed = list((before_count - after_count).elements())
    return {"added": sorted(added), "removed": sorted(removed)}


def _mapping_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    output = {}
    for key in sorted(set(before) | set(after)):
        previous = before.get(key, 0)
        current = after.get(key, 0)
        if previous != current:
            output[key] = {"after": current, "before": previous}
    return output


def _structured_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_full = dict(before["full"])
    after_full = dict(after["full"])
    before_mechanism = dict(before["mechanism"])
    after_mechanism = dict(after["mechanism"])

    def delta_payload(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "attribute_counts": _mapping_delta(
                dict(left["attribute_counts"]), dict(right["attribute_counts"])
            ),
            "counters": _mapping_delta(dict(left["counters"]), dict(right["counters"])),
            "entity_tokens": _counter_delta(
                list(left["entity_tokens"]), list(right["entity_tokens"])
            ),
            "false_fact_tokens": _counter_delta(
                list(left["false_fact_tokens"]), list(right["false_fact_tokens"])
            ),
            "predicate_counts": _mapping_delta(
                dict(left["predicate_counts"]), dict(right["predicate_counts"])
            ),
            "regime_index": _mapping_delta(
                {"value": left["regime_index"]}, {"value": right["regime_index"]}
            ),
            "role_counts": _mapping_delta(
                dict(left["role_counts"]), dict(right["role_counts"])
            ),
            "topology": _counter_delta(
                [str(item) for item in left["topology"]],
                [str(item) for item in right["topology"]],
            ),
            "true_fact_tokens": _counter_delta(
                list(left["true_fact_tokens"]), list(right["true_fact_tokens"])
            ),
        }

    full = delta_payload(before_full, after_full)
    mechanism = delta_payload(before_mechanism, after_mechanism)
    empty_mechanism = all(
        not value.get("added")
        and not value.get("removed")
        if isinstance(value, dict) and set(value) == {"added", "removed"}
        else not value
        for value in mechanism.values()
    )
    return {
        "exact_changed": before["exact_hash"] != after["exact_hash"],
        "full": full,
        "full_checksum": _checksum(full),
        "game_state_changed": before["game_state"] != after["game_state"],
        "level_delta": int(after["level"]) - int(before["level"]),
        "mechanism": mechanism,
        "mechanism_checksum": _checksum(mechanism),
        "mechanism_empty": bool(empty_mechanism),
    }


@dataclass(frozen=True)
class ApplicabilityTrial:
    trial_id: str
    context_name: str
    branch_name: str
    repetition: int
    lineage_seed: int
    anchor_level: int
    expected_anchor_hash: str
    observed_anchor_hash: str
    prefix_exact: bool
    first_divergence: str
    branch_available: bool
    anchor_state: Mapping[str, Any]
    trace: tuple[Mapping[str, Any], ...]
    trace_checksum: str
    progressed: bool
    level_delta: int
    progress_action_count: int
    final_exact_hash: str
    terminal: bool
    terminal_failure: bool
    sdk_calls_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_level": self.anchor_level,
            "anchor_state": dict(self.anchor_state),
            "branch_available": self.branch_available,
            "branch_name": self.branch_name,
            "context_name": self.context_name,
            "expected_anchor_hash": self.expected_anchor_hash,
            "final_exact_hash": self.final_exact_hash,
            "first_divergence": self.first_divergence,
            "level_delta": self.level_delta,
            "lineage_seed": self.lineage_seed,
            "observed_anchor_hash": self.observed_anchor_hash,
            "prefix_exact": self.prefix_exact,
            "progress_action_count": self.progress_action_count,
            "progressed": self.progressed,
            "repetition": self.repetition,
            "sdk_calls_after": self.sdk_calls_after,
            "terminal": self.terminal,
            "terminal_failure": self.terminal_failure,
            "trace": [dict(item) for item in self.trace],
            "trace_checksum": self.trace_checksum,
            "trial_id": self.trial_id,
        }


def _load_inputs(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[tuple[ProgressWitness, ...], MinimalCausalOption]:
    witness_path = _resolve_bound(
        str(manifest["inputs"]["witness_registry"]["path"]), root=root
    )
    _, witnesses = load_reconfirmation_registry(witness_path)
    option_path = _resolve_bound(
        str(manifest["inputs"]["minimal_option"]["path"]), root=root
    )
    contextual = _load_contextual_option(option_path)
    option = MinimalCausalOption.from_dict(contextual["option"])
    if option.checksum != manifest["inputs"]["option_checksum"]:
        raise ValueError("T12.4a.4b option differs from manifest")
    return witnesses, option


def _materialize_option(option: MinimalCausalOption) -> tuple[GroundedAction, ...]:
    empty = AbstractState()
    return tuple(step.materialize(empty) for step in option.steps)


def _run_trial(
    *,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    route: Sequence[WitnessStep],
    actions: Sequence[GroundedAction],
    context_name: str,
    branch_name: str,
    repetition: int,
    lineage_seed: int,
    expected_anchor_hash: str,
    anchor_level: int,
    budget: ApplicabilitySdkBudget,
) -> ApplicabilityTrial:
    env = _make_env(game_id, environments_dir, env_factory)
    budget.consume()
    frame = _reset_env(env)
    initial_hash = state_signature_from_frame(frame)
    initial_exact = bool(route and initial_hash == route[0].expected_source_hash)
    divergence = "" if initial_exact else "reset:initial"
    if initial_exact:
        frame, _, divergence = _execute_expected_steps(
            env=env,
            frame=frame,
            steps=route,
            phase=f"applicability_{context_name}_{branch_name}",
            start_index=0,
            budget=budget,
        )
    anchor_snapshot = snapshot_frame(frame)
    observed_anchor_hash = state_signature_from_frame(frame)
    observed_anchor_level = int(anchor_snapshot.levels_completed)
    prefix_exact = bool(
        initial_exact
        and not divergence
        and observed_anchor_hash == expected_anchor_hash
        and observed_anchor_level == anchor_level
    )
    anchor_state = _state_descriptor(
        _symbolic_state(frame),
        exact_hash=observed_anchor_hash,
        level=observed_anchor_level,
        game_state=str(anchor_snapshot.game_state),
    )

    trace: list[dict[str, Any]] = []
    branch_available = prefix_exact
    progressed = False
    progress_action_count = 0
    if prefix_exact:
        for position, action in enumerate(actions):
            before_snapshot = snapshot_frame(frame)
            before_hash = state_signature_from_frame(frame)
            before_state = _state_descriptor(
                _symbolic_state(frame),
                exact_hash=before_hash,
                level=int(before_snapshot.levels_completed),
                game_state=str(before_snapshot.game_state),
            )
            selected = select_live_action(
                env,
                action.action_name,
                action_args=action.action_data,
            )
            if selected is None:
                branch_available = False
                trace.append(
                    {
                        "action_data": dict(action.action_data),
                        "action_name": action.action_name,
                        "available": False,
                        "position": position,
                        "source_state": before_state,
                    }
                )
                break
            budget.consume()
            frame = _step_env_action(env, selected)
            after_snapshot = snapshot_frame(frame)
            after_hash = state_signature_from_frame(frame)
            after_state = _state_descriptor(
                _symbolic_state(frame),
                exact_hash=after_hash,
                level=int(after_snapshot.levels_completed),
                game_state=str(after_snapshot.game_state),
            )
            delta = _structured_delta(before_state, after_state)
            terminal = _is_terminal(after_snapshot.game_state)
            trace.append(
                {
                    "action_data": dict(action.action_data),
                    "action_name": action.action_name,
                    "available": True,
                    "delta": delta,
                    "position": position,
                    "source_state": before_state,
                    "target_state": after_state,
                    "terminal": terminal,
                }
            )
            if int(after_snapshot.levels_completed) > anchor_level:
                progressed = True
                progress_action_count = position + 1
                break
            if terminal:
                break

    final_snapshot = snapshot_frame(frame)
    final_level = int(final_snapshot.levels_completed)
    final_hash = state_signature_from_frame(frame)
    level_delta = max(0, final_level - anchor_level)
    terminal = _is_terminal(final_snapshot.game_state)
    terminal_failure = bool(terminal and level_delta == 0)
    diagnostic_trace = [
        {
            "action_name": item.get("action_name"),
            "available": item.get("available"),
            "delta": item.get("delta"),
            "position": item.get("position"),
            "source_exact_hash": item.get("source_state", {}).get("exact_hash"),
            "target_exact_hash": item.get("target_state", {}).get("exact_hash"),
        }
        for item in trace
    ]
    return ApplicabilityTrial(
        trial_id=(
            f"{context_name}_{branch_name}_seed_{lineage_seed}_rep_{repetition}"
        ),
        context_name=context_name,
        branch_name=branch_name,
        repetition=repetition,
        lineage_seed=lineage_seed,
        anchor_level=anchor_level,
        expected_anchor_hash=expected_anchor_hash,
        observed_anchor_hash=observed_anchor_hash,
        prefix_exact=prefix_exact,
        first_divergence=divergence,
        branch_available=branch_available,
        anchor_state=anchor_state,
        trace=tuple(trace),
        trace_checksum=_checksum(diagnostic_trace),
        progressed=progressed,
        level_delta=level_delta,
        progress_action_count=progress_action_count,
        final_exact_hash=final_hash,
        terminal=terminal,
        terminal_failure=terminal_failure,
        sdk_calls_after=budget.used,
    )


def _descriptor_contrast(
    successful: Mapping[str, Any],
    failed: Mapping[str, Any],
) -> dict[str, Any]:
    left = dict(successful["mechanism"])
    right = dict(failed["mechanism"])
    return {
        "attribute_counts": _mapping_delta(
            dict(left["attribute_counts"]), dict(right["attribute_counts"])
        ),
        "entity_tokens": _counter_delta(
            list(left["entity_tokens"]), list(right["entity_tokens"])
        ),
        "false_fact_tokens": _counter_delta(
            list(left["false_fact_tokens"]), list(right["false_fact_tokens"])
        ),
        "predicate_counts": _mapping_delta(
            dict(left["predicate_counts"]), dict(right["predicate_counts"])
        ),
        "role_counts": _mapping_delta(
            dict(left["role_counts"]), dict(right["role_counts"])
        ),
        "true_fact_tokens": _counter_delta(
            list(left["true_fact_tokens"]), list(right["true_fact_tokens"])
        ),
    }


def classify_applicability(
    successful_trials: Sequence[ApplicabilityTrial],
    failed_trials: Sequence[ApplicabilityTrial],
) -> dict[str, Any]:
    """Apply the frozen mutually-exclusive diagnostic decision tree."""

    successful = {
        (trial.lineage_seed, trial.repetition): trial for trial in successful_trials
    }
    failed = {(trial.lineage_seed, trial.repetition): trial for trial in failed_trials}
    keys = sorted(set(successful) & set(failed))
    if not keys or set(successful) != set(failed):
        return {
            "classification": "AUDIT_INTEGRITY_FAILURE",
            "reason": "unpaired successful and failed contexts",
        }

    anchor_matches: list[bool] = []
    paired_step_matches: list[bool] = []
    divergence_positions: list[int] = []
    structured_changes = False
    exact_changes = False
    contrasts = {}
    for key in keys:
        left = successful[key]
        right = failed[key]
        anchor_matches.append(
            left.anchor_state["mechanism_checksum"]
            == right.anchor_state["mechanism_checksum"]
        )
        contrasts[f"seed_{key[0]}_rep_{key[1]}"] = _descriptor_contrast(
            left.anchor_state, right.anchor_state
        )
        if len(left.trace) != len(right.trace):
            divergence_positions.append(min(len(left.trace), len(right.trace)))
        for position, (left_step, right_step) in enumerate(
            zip(left.trace, right.trace, strict=False)
        ):
            left_delta = dict(left_step.get("delta", {}))
            right_delta = dict(right_step.get("delta", {}))
            matches = (
                left_step.get("action_name") == right_step.get("action_name")
                and left_delta.get("mechanism_checksum")
                == right_delta.get("mechanism_checksum")
            )
            paired_step_matches.append(matches)
            if not matches:
                divergence_positions.append(position)
            structured_changes = structured_changes or not bool(
                left_delta.get("mechanism_empty", True)
            ) or not bool(right_delta.get("mechanism_empty", True))
            exact_changes = exact_changes or bool(
                left_delta.get("exact_changed", False)
            ) or bool(right_delta.get("exact_changed", False))

    all_anchors_match = all(anchor_matches)
    all_deltas_match = bool(paired_step_matches) and all(paired_step_matches)
    earliest = min(divergence_positions) if divergence_positions else None
    outcomes_contrast = all(trial.progressed for trial in successful_trials) and not any(
        trial.progressed for trial in failed_trials
    )
    if not outcomes_contrast:
        classification = "AUDIT_INTEGRITY_FAILURE"
        reason = "the preregistered success/failure contrast was not reproduced"
    elif not structured_changes and exact_changes:
        classification = "REPRESENTATION_INSUFFICIENT"
        reason = "pixels changed but the object-centric state exposed no mechanism delta"
    elif not all_deltas_match:
        if earliest == 0 and not all_anchors_match:
            classification = "INITIATION_AND_DYNAMICS_SHIFT"
            reason = "anchor structure and the first option mechanism delta both changed"
        else:
            classification = "DYNAMICS_CONTEXT_SHIFT"
            reason = "the option's structured transition effects diverged across contexts"
    elif not all_anchors_match:
        classification = "INITIATION_GOAL_CONTEXT_SHIFT"
        reason = "transition effects matched but the initiation context and outcome differed"
    else:
        classification = "TERMINATION_PREDICATE_CONTEXT_SHIFT"
        reason = "anchors and transition effects matched while completion did not"
    return {
        "all_anchor_mechanisms_match": all_anchors_match,
        "all_paired_mechanism_deltas_match": all_deltas_match,
        "anchor_contrasts": contrasts,
        "classification": classification,
        "earliest_mechanism_divergence_position": earliest,
        "exact_changes_observed": exact_changes,
        "outcome_contrast_reproduced": outcomes_contrast,
        "paired_step_comparisons": len(paired_step_matches),
        "paired_step_match_rate": (
            sum(paired_step_matches) / len(paired_step_matches)
            if paired_step_matches
            else 0.0
        ),
        "reason": reason,
        "structured_mechanism_changes_observed": structured_changes,
    }


def _determinism_check(trials: Sequence[ApplicabilityTrial]) -> bool:
    groups: dict[tuple[str, str, int], set[str]] = {}
    for trial in trials:
        groups.setdefault(
            (trial.context_name, trial.branch_name, trial.lineage_seed), set()
        ).add(trial.trace_checksum)
    return all(len(values) == 1 for values in groups.values())


def run_option_applicability(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_option_applicability_manifest(
        manifest_path, verify_code=env_factory is None
    )
    if not manifest["firewall"]["option_applicability_audit_authorized"]:
        raise ValueError("T12.4a.4b applicability audit is not authorized")
    protocol = OptionApplicabilityProtocol(**dict(manifest["protocol"]))
    root = Path(__file__).resolve().parents[3]
    witnesses, option = _load_inputs(manifest, root=root)
    option_actions = _materialize_option(option)
    if tuple(action.action_name for action in option_actions) != (
        protocol.expected_option_actions
    ):
        raise ValueError("T12.4a.4b materialized option differs from protocol")
    branches = {"option_full": option_actions, "null": ()}
    budget = ApplicabilitySdkBudget(protocol.maximum_sdk_calls)
    trials: list[ApplicabilityTrial] = []
    witnesses_by_seed = {item.source_seed: item for item in witnesses}
    for context_name in protocol.context_names:
        for branch_name in protocol.branch_names:
            for lineage_seed in protocol.source_seeds:
                witness = witnesses_by_seed[lineage_seed]
                if context_name == "successful_level0":
                    prefix_length = int(
                        manifest["inputs"]["successful_prefix_lengths"][str(lineage_seed)]
                    )
                    route = tuple(witness.steps[:prefix_length])
                    expected_hash = str(
                        manifest["inputs"]["successful_anchor_hashes"][str(lineage_seed)]
                    )
                    anchor_level = 0
                else:
                    route = tuple(witness.steps)
                    expected_hash = str(manifest["inputs"]["failed_anchor_hash"])
                    anchor_level = int(manifest["inputs"]["failed_anchor_level"])
                for repetition in range(
                    protocol.repetitions_per_context_lineage_branch
                ):
                    trials.append(
                        _run_trial(
                            game_id=str(manifest["game_id"]),
                            environments_dir=environments_dir,
                            env_factory=env_factory,
                            route=route,
                            actions=branches[branch_name],
                            context_name=context_name,
                            branch_name=branch_name,
                            repetition=repetition,
                            lineage_seed=lineage_seed,
                            expected_anchor_hash=expected_hash,
                            anchor_level=anchor_level,
                            budget=budget,
                        )
                    )

    successful_full = tuple(
        trial
        for trial in trials
        if trial.context_name == "successful_level0"
        and trial.branch_name == "option_full"
    )
    failed_full = tuple(
        trial
        for trial in trials
        if trial.context_name == "failed_level1" and trial.branch_name == "option_full"
    )
    null_trials = tuple(trial for trial in trials if trial.branch_name == "null")
    diagnosis = classify_applicability(successful_full, failed_full)
    prefix_exact = sum(trial.prefix_exact for trial in trials)
    terminal_failures = sum(trial.terminal_failure for trial in trials)
    checks = {
        "all_actions_available": all(trial.branch_available for trial in trials),
        "all_prefixes_exact": prefix_exact == len(trials),
        "expected_failed_context_reproduced": not any(
            trial.progressed for trial in failed_full
        ),
        "expected_success_context_reproduced": all(
            trial.progressed and trial.level_delta == 1 for trial in successful_full
        ),
        "mutually_exclusive_diagnosis_identified": diagnosis.get("classification")
        in AUTHORIZED_DIAGNOSES,
        "no_terminal_failures": terminal_failures
        <= protocol.maximum_terminal_failures,
        "null_controls_preserve_anchor": all(
            trial.final_exact_hash == trial.expected_anchor_hash for trial in null_trials
        ),
        "repetitions_deterministic_within_lineage": _determinism_check(trials),
        "sdk_budget_respected": budget.used <= protocol.maximum_sdk_calls,
        "trial_count_exact": len(trials) == protocol.expected_trial_count,
    }
    passed = all(checks.values())
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    trials_path = destination / "applicability_trials.json"
    diagnostic_path = destination / "applicability_diagnosis.json"
    report_path = destination / "applicability_report.json"
    receipt_path = destination / "applicability_receipt.json"
    _write_json_once(
        trials_path,
        {
            "format_version": "sage-t12.4a.4b-applicability-trials-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [trial.to_dict() for trial in trials],
        },
        storage_budget=storage,
    )
    _write_json_once(
        diagnostic_path,
        {
            "format_version": "sage-t12.4a.4b-applicability-diagnosis-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "option_checksum": manifest["inputs"]["option_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            **diagnosis,
        },
        storage_budget=storage,
    )
    metrics = {
        "checks": checks,
        "classification": diagnosis.get("classification"),
        "diagnosis": diagnosis,
        "failed_context_progressions": sum(trial.progressed for trial in failed_full),
        "prefix_exact_trials": prefix_exact,
        "sdk_calls": budget.snapshot(),
        "storage": storage.snapshot(),
        "successful_context_progressions": sum(
            trial.progressed for trial in successful_full
        ),
        "terminal_failures": terminal_failures,
        "trial_count": len(trials),
    }
    status = (
        "PASS_T12_4A_4B_APPLICABILITY_AUDIT_GATE"
        if passed
        else "FAIL_T12_4A_4B_APPLICABILITY_AUDIT_GATE"
    )
    report = {
        "format_version": "sage-t12.4a.4b-option-applicability-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
        "storage": storage.snapshot(),
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = option_applicability_receipt(
        manifest=manifest,
        phase="option_applicability_audit",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "diagnosis": {
                "path": str(diagnostic_path.resolve()),
                "sha256": _file_sha256(diagnostic_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "trials": {
                "path": str(trials_path.resolve()),
                "sha256": _file_sha256(trials_path),
            },
        },
    )
    _write_json_once(receipt_path, receipt, storage_budget=storage)
    return receipt


def option_applicability_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_option_applicability_manifest(manifest_path)
    receipt = None
    if receipt_path is not None and Path(receipt_path).is_file():
        receipt = load_option_applicability_receipt(receipt_path, manifest=manifest)
    passed = bool(
        receipt
        and receipt.get("passed")
        and receipt.get("phase") == "option_applicability_audit"
        and receipt.get("status") == "PASS_T12_4A_4B_APPLICABILITY_AUDIT_GATE"
    )
    classification = (
        str(receipt.get("metrics", {}).get("classification", "")) if receipt else ""
    )
    representation_route = bool(passed and classification == "REPRESENTATION_INSUFFICIENT")
    contract_route = bool(passed and classification in AUTHORIZED_DIAGNOSES and not representation_route)
    return {
        "firewall": {
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_applicability_audit_authorized": manifest["firewall"][
                "option_applicability_audit_authorized"
            ],
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_4a_4c_option_contract_freeze_authorized": contract_route,
            "t12_4a_4c_representation_extension_freeze_authorized": representation_route,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.4a.4b-option-applicability-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": bool(contract_route or representation_route),
        "parent_t12_4a_4_status": manifest["parent"]["negative_receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "receipt": receipt,
    }


__all__ = [
    "ApplicabilityTrial",
    "classify_applicability",
    "option_applicability_status",
    "run_option_applicability",
]
