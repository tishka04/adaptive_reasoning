"""A34.3 scoped utility probe for the A33.4 parameterized relation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.a33.parameterized_relational_registry import (
    A33_4_ENTRY_ADDED,
    A33_4_SCHEMA_VERSION,
    A33_4_TRUTH_STATUS,
    CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST,
    DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH,
)
from theory.non_ar25_active_micro_run import _env_dir
from theory.sage.live_mini_frontier_m3_executor import (
    EnvFactory,
    _execute_request_arm,
)
from theory.sage.third_unknown_game_parameterized_execution import (
    budget_from_request,
)
from theory.sage.third_unknown_game_parameterized_frontier import (
    DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH,
    SAGE7A_FRONTIER_GENERATED,
    SAGE7A_SCHEMA_VERSION,
    SAGE7A_TRUTH_STATUS,
)

from .control_dependent_relational_usage_probe import (
    _arm_signal,
    _signature_payload,
    _signature_won,
)


DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH = (
    Path("diagnostics") / "a34" / "parameterized_relational_usage_probe.json"
)

A34_3_SCHEMA_VERSION = "a34.parameterized_relational_usage_probe.v1"
A34_3_TRUTH_STATUS = "NOT_REEVALUATED_BY_A34_3"
A34_3_RELATION_USEFUL = "A34_PARAMETERIZED_RELATION_CONTEXTUALLY_USEFUL"
A34_3_RELATION_NOT_USEFUL = "A34_PARAMETERIZED_RELATION_NOT_USEFUL"
UTILITY_ASSESSMENT = "USEFUL_AGAINST_RECORDED_DIFFERENTIATING_CONTROL"
BASELINE_POLICY = "RECORDED_DIFFERENTIATING_PARAMETER_VARIANT"
REGISTRY_POLICY = "A33_4_EXACT_SCOPE_PARAMETERIZED_RELATIONAL_PRIORITY"
EQUIVALENCE_AUDIT_POLICY = "RECORDED_EQUIVALENT_PARAMETER_VARIANT_AUDIT"


@dataclass(frozen=True)
class ParameterizedRelationalUsageProbeResult:
    """One exact-context comparison among three ACTION6 parameter variants."""

    context_snapshot_hash: str
    source_request_id: str
    source_step: int
    budget: int
    technical_source_request_ids: Tuple[str, ...]
    game_id: str
    metric: str
    action_family: str
    baseline_action_args: Dict[str, Any]
    registry_action_args: Dict[str, Any]
    equivalent_action_args: Dict[str, Any]
    baseline_arm: Dict[str, Any]
    registry_arm: Dict[str, Any]
    equivalent_arm: Dict[str, Any]
    baseline_signal: float
    registry_signal: float
    equivalent_signal: float
    registry_gain_over_baseline: float
    registry_gain_over_equivalent: float
    utility_assessment: str
    parameter_choice_changed: bool
    scope_match: bool
    replay_exact_all_arms: bool
    functional_local_progress: bool
    levels_completed_before: int
    baseline_levels_completed_after: int
    registry_levels_completed_after: int
    equivalent_levels_completed_after: int
    registry_levels_completed_delta: int
    baseline_win: bool
    registry_win: bool
    equivalent_win: bool
    truth_status: str = A34_3_TRUTH_STATUS
    scientific_verdict_performed: bool = False
    support_counted: int = 0
    parameterized_variants_counted_as_distinct_actions: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_snapshot_hash": self.context_snapshot_hash,
            "source_request_id": self.source_request_id,
            "source_step": int(self.source_step),
            "budget": int(self.budget),
            "technical_source_request_ids": list(self.technical_source_request_ids),
            "game_id": self.game_id,
            "metric": self.metric,
            "baseline_policy": BASELINE_POLICY,
            "registry_policy": REGISTRY_POLICY,
            "equivalence_audit_policy": EQUIVALENCE_AUDIT_POLICY,
            "action_family": self.action_family,
            "baseline_action_args": dict(self.baseline_action_args),
            "registry_action_args": dict(self.registry_action_args),
            "equivalent_action_args": dict(self.equivalent_action_args),
            "baseline_arm": dict(self.baseline_arm),
            "registry_arm": dict(self.registry_arm),
            "equivalent_arm": dict(self.equivalent_arm),
            "baseline_signal": float(self.baseline_signal),
            "registry_signal": float(self.registry_signal),
            "equivalent_signal": float(self.equivalent_signal),
            "registry_gain_over_baseline": float(self.registry_gain_over_baseline),
            "registry_gain_over_equivalent": float(self.registry_gain_over_equivalent),
            "utility_assessment": self.utility_assessment,
            "parameter_choice_changed": self.parameter_choice_changed,
            "scope_match": self.scope_match,
            "replay_exact_all_arms": self.replay_exact_all_arms,
            "functional_local_progress": self.functional_local_progress,
            "levels_completed_before": int(self.levels_completed_before),
            "baseline_levels_completed_after": int(
                self.baseline_levels_completed_after
            ),
            "registry_levels_completed_after": int(
                self.registry_levels_completed_after
            ),
            "equivalent_levels_completed_after": int(
                self.equivalent_levels_completed_after
            ),
            "registry_levels_completed_delta": int(
                self.registry_levels_completed_delta
            ),
            "baseline_win": self.baseline_win,
            "registry_win": self.registry_win,
            "equivalent_win": self.equivalent_win,
            "truth_status": self.truth_status,
            "scientific_verdict_performed": self.scientific_verdict_performed,
            "support_counted": int(self.support_counted),
            "parameterized_variants_counted_as_distinct_actions": (
                self.parameterized_variants_counted_as_distinct_actions
            ),
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_parameterized_relational_usage_probe(
    *,
    registry_path: str | Path = (
        DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH
    ),
    source_sage7a_path: str | Path = DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Compare no-memory and A33.4 parameter choices in eight exact contexts."""
    registry = _load_json(registry_path)
    source_sage7a = _load_json(source_sage7a_path)
    validate_a34_3_sources(registry, source_sage7a)
    entry = dict(registry["parameterized_relational_contrasts"][0])
    contexts = build_a34_3_replay_contexts(entry, source_sage7a)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    probes = tuple(
        execute_a34_3_context_probe(
            entry,
            context,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        for context in contexts
    )
    gate = build_a34_3_gate(entry, contexts, probes)
    if not gate or not all(gate.values()):
        raise ValueError("A34.3 parameterized usage gate did not pass")
    summary = summarize_a34_3(probes, gate)
    payload = {
        "config": {
            "schema_version": A34_3_SCHEMA_VERSION,
            "registry_path": str(registry_path),
            "source_sage7a_path": str(source_sage7a_path),
            "environments_dir": str(env_dir),
            "inputs_read": ["A33.4", "SAGE.7a_REPLAY"],
            "baseline_policy": BASELINE_POLICY,
            "registry_policy": REGISTRY_POLICY,
            "equivalence_audit_policy": EQUIVALENCE_AUDIT_POLICY,
            "canonical_replay_selection": (
                "LOWEST_BUDGET_THEN_STEP_THEN_REQUEST_ID_WITHOUT_OUTCOME_READ"
            ),
            "artifacts_not_modified": ["A32.7", "A33.4", "SAGE.7a"],
            "utility_policy": {
                "registry_decides_parameter_relation": True,
                "sage_source_reconstructs_context_only": True,
                "same_live_prefix_for_all_arms": True,
                "all_registered_contexts_required": True,
                "technical_replays_not_counted_as_contexts": True,
                "parameter_variants_are_one_action_family": True,
                "local_signal_gain_is_not_level_completion": True,
                "a34_does_not_reevaluate_truth": True,
                "a34_does_not_count_support": True,
            },
        },
        "registry_entry": entry,
        "replay_contexts": [
            {key: value for key, value in row.items() if key != "request"}
            for row in contexts
        ],
        "usage_probes": [probe.to_dict() for probe in probes],
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": summary["status"],
        "truth_status": A34_3_TRUTH_STATUS,
        "utility_evaluation_performed": True,
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "support": 0,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "parameterized_variants_counted_as_distinct_actions": False,
        "scope_generalization_performed": False,
        "levels_completed": summary["registry_levels_completed_max"],
        "win_rate": summary["registry_win_rate"],
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_parameterized_relational_usage_probe(payload, output_path)
    return payload


def build_a34_3_replay_contexts(
    entry: Mapping[str, Any],
    source_sage7a: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Select one canonical outcome-blind replay request per registered hash."""
    target_args = dict(entry.get("target_action_args", {}) or {})
    requests = [
        dict(row)
        for row in source_sage7a.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping)
        and str(row.get("target_action", "")) == str(entry.get("target_action", ""))
        and dict(row.get("target_action_args", {}) or {}) == target_args
    ]
    contexts: List[Dict[str, Any]] = []
    for context_hash in entry.get("context_snapshot_hashes", []) or []:
        matching = [
            row
            for row in requests
            if str(row.get("context_snapshot_hash", "")) == str(context_hash)
        ]
        if not matching:
            raise ValueError("A33.4 context is missing from SAGE.7a replay provenance")
        ordered = sorted(
            matching,
            key=lambda row: (
                budget_from_request(row),
                int(row.get("source_step", 0) or 0),
                str(row.get("request_id", "")),
            ),
        )
        selected = ordered[0]
        contexts.append(
            {
                "context_snapshot_hash": str(context_hash),
                "source_request_id": str(selected.get("request_id", "")),
                "source_step": int(selected.get("source_step", 0) or 0),
                "budget": budget_from_request(selected),
                "technical_source_request_ids": [
                    str(row.get("request_id", "")) for row in ordered
                ],
                "technical_replay_count": len(ordered) - 1,
                "game_id": str(selected.get("game_id", "")),
                "metric": str(selected.get("metric", "")),
                "context_replay": list(selected.get("context_replay", []) or []),
                "context_replay_args": [
                    dict(row) for row in selected.get("context_replay_args", []) or []
                ],
                "request": selected,
                "canonical_request_selected_without_outcome_read": True,
                "provenance_used_for_parameter_choice": False,
                "registry_used_for_parameter_choice": True,
            }
        )
    return tuple(contexts)


def execute_a34_3_context_probe(
    entry: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> ParameterizedRelationalUsageProbeResult:
    """Execute the differentiating, target and equivalent variants."""
    request = dict(context.get("request", {}) or {})
    action_family = str(entry.get("target_action", ""))
    target_args = dict(entry.get("target_action_args", {}) or {})
    baseline_args = dict(
        entry.get("differentiating_control_variants", [])[0].get("action_args", {})
    )
    equivalent_args = dict(
        entry.get("equivalent_control_variants", [])[0].get("action_args", {})
    )
    baseline = _execute_request_arm(
        request,
        action_name=action_family,
        action_args=baseline_args,
        arm="no_memory_parameter_baseline",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    treatment = _execute_request_arm(
        request,
        action_name=action_family,
        action_args=target_args,
        arm="a33_4_parameterized_registry_policy",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    equivalent = _execute_request_arm(
        request,
        action_name=action_family,
        action_args=equivalent_args,
        arm="equivalent_parameter_control_audit",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    arms = (baseline, treatment, equivalent)
    if any(str(arm.get("status", "")) != "EXECUTED" for arm in arms):
        reasons = [str(arm.get("reason", "")) for arm in arms]
        raise ValueError(f"A34.3 exact replay arm blocked: {reasons}")
    baseline_signal = _arm_signal(baseline)
    registry_signal = _arm_signal(treatment)
    equivalent_signal = _arm_signal(equivalent)
    before_state = _signature_payload(treatment.get("before_signature", ""))
    baseline_after = _signature_payload(baseline.get("after_signature", ""))
    treatment_after = _signature_payload(treatment.get("after_signature", ""))
    equivalent_after = _signature_payload(equivalent.get("after_signature", ""))
    before_levels = int(before_state.get("levels_completed", 0) or 0)
    registry_levels = int(treatment_after.get("levels_completed", 0) or 0)
    scope_match = (
        str(context.get("game_id", "")) == str(entry.get("game_id", ""))
        and str(context.get("metric", "")) == str(entry.get("predicted_metric", ""))
        and str(context.get("context_snapshot_hash", ""))
        in set(entry.get("context_snapshot_hashes", []) or [])
    )
    return ParameterizedRelationalUsageProbeResult(
        context_snapshot_hash=str(context.get("context_snapshot_hash", "")),
        source_request_id=str(context.get("source_request_id", "")),
        source_step=int(context.get("source_step", 0) or 0),
        budget=int(context.get("budget", 0) or 0),
        technical_source_request_ids=tuple(
            str(value)
            for value in context.get("technical_source_request_ids", []) or []
        ),
        game_id=str(context.get("game_id", "")),
        metric=str(context.get("metric", "")),
        action_family=action_family,
        baseline_action_args=baseline_args,
        registry_action_args=target_args,
        equivalent_action_args=equivalent_args,
        baseline_arm=dict(baseline),
        registry_arm=dict(treatment),
        equivalent_arm=dict(equivalent),
        baseline_signal=baseline_signal,
        registry_signal=registry_signal,
        equivalent_signal=equivalent_signal,
        registry_gain_over_baseline=registry_signal - baseline_signal,
        registry_gain_over_equivalent=registry_signal - equivalent_signal,
        utility_assessment=(
            UTILITY_ASSESSMENT
            if registry_signal > baseline_signal
            and registry_signal == equivalent_signal
            else "NOT_USEFUL"
        ),
        parameter_choice_changed=baseline_args != target_args,
        scope_match=scope_match,
        replay_exact_all_arms=all(
            bool(arm.get("context_snapshot_hash_verified", False)) for arm in arms
        ),
        functional_local_progress=registry_signal > baseline_signal,
        levels_completed_before=before_levels,
        baseline_levels_completed_after=int(
            baseline_after.get("levels_completed", 0) or 0
        ),
        registry_levels_completed_after=registry_levels,
        equivalent_levels_completed_after=int(
            equivalent_after.get("levels_completed", 0) or 0
        ),
        registry_levels_completed_delta=registry_levels - before_levels,
        baseline_win=_signature_won(baseline_after),
        registry_win=_signature_won(treatment_after),
        equivalent_win=_signature_won(equivalent_after),
    )


def build_a34_3_gate(
    entry: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    probes: Sequence[ParameterizedRelationalUsageProbeResult],
) -> Dict[str, bool]:
    expected_hashes = list(entry.get("context_snapshot_hashes", []) or [])
    return {
        "all_registered_contexts_reconstructed_once": len(contexts)
        == len(probes)
        == len(expected_hashes)
        == 8
        and {str(row.get("context_snapshot_hash", "")) for row in contexts}
        == set(expected_hashes),
        "canonical_requests_selected_without_outcomes": all(
            bool(row.get("canonical_request_selected_without_outcome_read", False))
            for row in contexts
        ),
        "technical_replays_not_recounted_as_contexts": sum(
            int(row.get("technical_replay_count", 0) or 0) for row in contexts
        )
        == 5,
        "all_replays_exact_for_all_arms": all(
            probe.replay_exact_all_arms for probe in probes
        ),
        "registry_policy_scope_matched": all(probe.scope_match for probe in probes),
        "registry_changes_parameter_choice": all(
            probe.parameter_choice_changed for probe in probes
        ),
        "registry_beats_differentiating_control": all(
            probe.registry_gain_over_baseline == 2.0 for probe in probes
        ),
        "registry_preserves_equivalent_control_limit": all(
            probe.registry_gain_over_equivalent == 0.0 for probe in probes
        ),
        "utility_is_relational_not_autonomous": all(
            probe.utility_assessment == UTILITY_ASSESSMENT for probe in probes
        ),
        "parameter_variants_remain_one_action_family": all(
            not probe.parameterized_variants_counted_as_distinct_actions
            and probe.action_family == "ACTION6"
            for probe in probes
        ),
        "no_scientific_verdict_or_support_created": all(
            not probe.scientific_verdict_performed and probe.support_counted == 0
            for probe in probes
        ),
        "wrong_confirmations_zero": all(
            probe.wrong_confirmations == 0 for probe in probes
        ),
    }


def summarize_a34_3(
    probes: Sequence[ParameterizedRelationalUsageProbeResult],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    useful = [
        probe for probe in probes if probe.utility_assessment == UTILITY_ASSESSMENT
    ]
    registry_wins = sum(probe.registry_win for probe in probes)
    outcome = (
        A34_3_RELATION_USEFUL
        if len(useful) == len(probes)
        else A34_3_RELATION_NOT_USEFUL
    )
    return {
        "registered_relations_probed": 1 if probes else 0,
        "exact_contexts_probed": len(probes),
        "technical_replay_requests_preserved": sum(
            len(probe.technical_source_request_ids) - 1 for probe in probes
        ),
        "action_families": sorted({probe.action_family for probe in probes}),
        "distinct_action_families": len({probe.action_family for probe in probes}),
        "parameter_choices_changed": sum(
            probe.parameter_choice_changed for probe in probes
        ),
        "contextual_relational_utility_events": len(useful),
        "registry_gain_over_baseline": [
            probe.registry_gain_over_baseline for probe in probes
        ],
        "registry_gain_over_equivalent": [
            probe.registry_gain_over_equivalent for probe in probes
        ],
        "functional_local_progress_events": sum(
            probe.functional_local_progress for probe in probes
        ),
        "baseline_levels_completed_max": max(
            (probe.baseline_levels_completed_after for probe in probes), default=0
        ),
        "registry_levels_completed_max": max(
            (probe.registry_levels_completed_after for probe in probes), default=0
        ),
        "registry_levels_completed_delta_total": sum(
            probe.registry_levels_completed_delta for probe in probes
        ),
        "baseline_wins": sum(probe.baseline_win for probe in probes),
        "registry_wins": registry_wins,
        "equivalent_wins": sum(probe.equivalent_win for probe in probes),
        "registry_win_rate": registry_wins / len(probes) if probes else 0.0,
        "level_or_win_progress_demonstrated": any(
            probe.registry_levels_completed_delta > 0 or probe.registry_win
            for probe in probes
        ),
        "parameterized_variants_counted_as_distinct_actions": False,
        "a34_truth_reevaluations": 0,
        "support_counted": 0,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "status": "CONTEXTUALLY_USEFUL" if useful else "NOT_USEFUL",
        "outcome_status": outcome,
    }


def validate_a34_3_sources(
    registry: Mapping[str, Any],
    source_sage7a: Mapping[str, Any],
) -> None:
    """Require exact A33.4 scope and candidate-only SAGE.7a replay data."""
    registry_config = dict(registry.get("config", {}) or {})
    if str(registry_config.get("schema_version", "")) != A33_4_SCHEMA_VERSION:
        raise ValueError("A34.3 requires the A33.4 registry schema")
    if (
        str(registry.get("outcome_status", "")) != A33_4_ENTRY_ADDED
        or str(registry.get("truth_status", "")) != A33_4_TRUTH_STATUS
        or str(registry.get("status", "")) != "REGISTERED"
        or not bool(registry.get("registration_performed", False))
        or bool(registry.get("confirmation_performed", True))
        or int(registry.get("wrong_confirmations", 0) or 0) != 0
    ):
        raise ValueError("A34.3 requires the completed non-reevaluating A33.4 registry")
    if not bool(registry.get("summary", {}).get("gate_passed", False)) or not all(
        bool(value) for value in registry.get("gate", {}).values()
    ):
        raise ValueError("every A33.4 registry gate must pass")
    entries = [
        row
        for row in registry.get("parameterized_relational_contrasts", []) or []
        if isinstance(row, Mapping)
    ]
    if len(entries) != 1:
        raise ValueError("A34.3 requires exactly one A33.4 relation")
    entry = entries[0]
    if (
        str(entry.get("registry_entry_type", ""))
        != CONTROL_DEPENDENT_PARAMETERIZED_RELATIONAL_CONTRAST
        or str(entry.get("game_id", "")) != "tn36-ab4f63cc"
        or str(entry.get("target_action", "")) != "ACTION6"
        or dict(entry.get("target_action_args", {}) or {}) != {"x": 25, "y": 42}
        or list(entry.get("differentiating_control_variants", []) or [])
        != [{"action": "ACTION6", "action_args": {"x": 34, "y": 51}}]
        or list(entry.get("equivalent_control_variants", []) or [])
        != [{"action": "ACTION6", "action_args": {"x": 41, "y": 44}}]
        or str(entry.get("predicted_metric", "")) != "local_patch_before_after"
        or int(entry.get("confirmed_support", 0) or 0) != 8
        or int(entry.get("contradictions", 0) or 0) != 0
        or len(entry.get("context_snapshot_hashes", []) or []) != 8
    ):
        raise ValueError("A33.4 relation identity and scope must remain exact")
    if (
        not bool(entry.get("scope_game_locked", False))
        or not bool(entry.get("scope_metric_locked", False))
        or not bool(entry.get("scope_target_parameter_locked", False))
        or not bool(entry.get("scope_control_parameters_locked", False))
        or not bool(entry.get("scope_contexts_locked", False))
        or not bool(entry.get("autonomous_target_effect_excluded", False))
        or str(entry.get("autonomous_target_effect_status", "")) != "unresolved"
    ):
        raise ValueError("A33.4 parameterized scope and exclusions must remain locked")
    sage7a_config = dict(source_sage7a.get("config", {}) or {})
    if (
        str(sage7a_config.get("schema_version", "")) != SAGE7A_SCHEMA_VERSION
        or str(source_sage7a.get("outcome_status", "")) != SAGE7A_FRONTIER_GENERATED
        or str(source_sage7a.get("truth_status", "")) != SAGE7A_TRUTH_STATUS
        or int(source_sage7a.get("support", 0) or 0) != 0
        or bool(source_sage7a.get("revision_performed", True))
        or bool(
            source_sage7a.get(
                "parameterized_controls_counted_as_distinct_actions", True
            )
        )
    ):
        raise ValueError("A34.3 requires candidate-only SAGE.7a replay provenance")
    contexts = build_a34_3_replay_contexts(entry, source_sage7a)
    hashes = [str(row.get("context_snapshot_hash", "")) for row in contexts]
    if hashes != list(entry.get("context_snapshot_hashes", []) or []):
        raise ValueError(
            "A34.3 replay contexts must exactly match A33.4 order and scope"
        )
    expected_controls = [
        {"action": "ACTION6", "action_args": {"x": 34, "y": 51}},
        {"action": "ACTION6", "action_args": {"x": 41, "y": 44}},
    ]
    if any(
        str(row.get("game_id", "")) != "tn36-ab4f63cc"
        or str(row.get("metric", "")) != "local_patch_before_after"
        or len(row.get("context_replay", []) or []) != int(row.get("source_step", 0))
        or [
            {
                "action": str(control.get("action", "")),
                "action_args": dict(control.get("action_args", {}) or {}),
            }
            for control in row.get("request", {}).get(
                "pre_registered_parameterized_control_variants", []
            )
        ]
        != expected_controls
        for row in contexts
    ):
        raise ValueError("A34.3 replay provenance must remain exact and complete")


def write_parameterized_relational_usage_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_A33_PARAMETERIZED_RELATIONAL_REGISTRY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--source-sage7a",
        default=str(DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH),
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH),
    )
    args = parser.parse_args(argv)
    payload = run_parameterized_relational_usage_probe(
        registry_path=args.registry,
        source_sage7a_path=args.source_sage7a,
        environments_dir=args.environments_dir,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
