"""A34.2 scoped utility probe for the A33.3 relational registry.

The registry decides which action relation may be used. SAGE provenance is read
only to reconstruct the exact live-prefix contexts already locked by A33.3.
A34.2 measures utility; it does not re-evaluate the relation's truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.a33.control_dependent_relational_registry import (
    A33_3_ENTRY_ADDED,
    A33_3_SCHEMA_VERSION,
    A33_3_TRUTH_STATUS,
    CONTROL_DEPENDENT_RELATIONAL_CONTRAST,
    DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH,
)
from theory.non_ar25_active_micro_run import _env_dir
from theory.sage.live_mini_frontier_m3_executor import (
    EnvFactory,
    _execute_request_arm,
)
from theory.sage.second_unknown_game_control_dependence_consolidation import (
    DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH,
    SAGE6F_A32_REVIEW_ELIGIBLE,
    SAGE6F_SCHEMA_VERSION,
    SAGE6F_TRUTH_STATUS,
)
from theory.sage.second_unknown_game_switch_frontier import (
    DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH,
    SAGE6A_FRONTIER_GENERATED,
    SAGE6A_SCHEMA_VERSION,
    SAGE6A_TRUTH_STATUS,
)


DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH = (
    Path("diagnostics") / "a34" / "control_dependent_relational_usage_probe.json"
)

A34_2_SCHEMA_VERSION = "a34.control_dependent_relational_usage_probe.v1"
A34_2_TRUTH_STATUS = "NOT_REEVALUATED_BY_A34_2"
A34_2_RELATION_USEFUL = "A34_CONTROL_DEPENDENT_RELATION_CONTEXTUALLY_USEFUL"
A34_2_RELATION_NOT_USEFUL = "A34_CONTROL_DEPENDENT_RELATION_NOT_USEFUL"
UTILITY_ASSESSMENT = "USEFUL_AGAINST_RECORDED_LOWER_EFFECT_CONTROL"
BASELINE_POLICY = "RECORDED_LOWER_EFFECT_CONTROL"
REGISTRY_POLICY = "A33_3_EXACT_SCOPE_RELATIONAL_PRIORITY"
EQUIVALENCE_AUDIT_POLICY = "RECORDED_EQUIVALENT_CONTROL_AUDIT"


@dataclass(frozen=True)
class ControlDependentRelationalUsageProbeResult:
    """One exact-context baseline/treatment/equivalence comparison."""

    context_cluster_id: str
    context_snapshot_hash: str
    source_request_id: str
    source_step: int
    budget: int
    game_id: str
    metric: str
    baseline_action: str
    registry_action: str
    equivalent_action: str
    baseline_arm: Dict[str, Any]
    registry_arm: Dict[str, Any]
    equivalent_arm: Dict[str, Any]
    baseline_signal: float
    registry_signal: float
    equivalent_signal: float
    registry_gain_over_baseline: float
    registry_gain_over_equivalent: float
    utility_assessment: str
    action_choice_changed: bool
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
    truth_status: str = A34_2_TRUTH_STATUS
    scientific_verdict_performed: bool = False
    support_counted: int = 0
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_cluster_id": self.context_cluster_id,
            "context_snapshot_hash": self.context_snapshot_hash,
            "source_request_id": self.source_request_id,
            "source_step": int(self.source_step),
            "budget": int(self.budget),
            "game_id": self.game_id,
            "metric": self.metric,
            "baseline_policy": BASELINE_POLICY,
            "registry_policy": REGISTRY_POLICY,
            "equivalence_audit_policy": EQUIVALENCE_AUDIT_POLICY,
            "baseline_action": self.baseline_action,
            "registry_action": self.registry_action,
            "equivalent_action": self.equivalent_action,
            "baseline_arm": dict(self.baseline_arm),
            "registry_arm": dict(self.registry_arm),
            "equivalent_arm": dict(self.equivalent_arm),
            "baseline_signal": float(self.baseline_signal),
            "registry_signal": float(self.registry_signal),
            "equivalent_signal": float(self.equivalent_signal),
            "registry_gain_over_baseline": float(self.registry_gain_over_baseline),
            "registry_gain_over_equivalent": float(self.registry_gain_over_equivalent),
            "utility_assessment": self.utility_assessment,
            "action_choice_changed": self.action_choice_changed,
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
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_control_dependent_relational_usage_probe(
    *,
    registry_path: str | Path = (
        DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH
    ),
    source_sage6f_path: str | Path = (
        DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH
    ),
    source_sage6a_path: str | Path = DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Compare no-memory and A33.3 choices in all three registered contexts."""
    registry = _load_json(registry_path)
    source_sage6f = _load_json(source_sage6f_path)
    source_sage6a = _load_json(source_sage6a_path)
    validate_a34_2_sources(registry, source_sage6f, source_sage6a)
    entry = dict(registry["control_dependent_relational_contrasts"][0])
    contexts = build_a34_2_replay_contexts(entry, source_sage6f, source_sage6a)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    probes = tuple(
        execute_a34_2_context_probe(
            entry,
            context,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        for context in contexts
    )
    gate = build_a34_2_gate(entry, contexts, probes)
    if not gate or not all(gate.values()):
        raise ValueError("A34.2 relational usage gate did not pass")
    summary = summarize_a34_2(probes, gate)
    payload = {
        "config": {
            "schema_version": A34_2_SCHEMA_VERSION,
            "registry_path": str(registry_path),
            "source_sage6f_path": str(source_sage6f_path),
            "source_sage6a_path": str(source_sage6a_path),
            "environments_dir": str(env_dir),
            "inputs_read": ["A33.3", "SAGE.6f_PROVENANCE", "SAGE.6a_REPLAY"],
            "baseline_policy": BASELINE_POLICY,
            "registry_policy": REGISTRY_POLICY,
            "equivalence_audit_policy": EQUIVALENCE_AUDIT_POLICY,
            "artifacts_not_modified": [
                "A32.6",
                "A33.1",
                "A33.2",
                "A33.3",
                "SAGE.6a",
                "SAGE.6f",
            ],
            "utility_policy": {
                "registry_decides_action_relation": True,
                "sage_sources_reconstruct_context_only": True,
                "same_live_prefix_for_all_arms": True,
                "all_registered_contexts_required": True,
                "local_signal_gain_is_not_level_completion": True,
                "a34_does_not_reevaluate_truth": True,
                "a34_does_not_count_support": True,
            },
        },
        "registry_entry": entry,
        "replay_contexts": [dict(row) for row in contexts],
        "usage_probes": [probe.to_dict() for probe in probes],
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": summary["status"],
        "truth_status": A34_2_TRUTH_STATUS,
        "utility_evaluation_performed": True,
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "support": 0,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "scope_generalization_performed": False,
        "levels_completed": summary["registry_levels_completed_max"],
        "win_rate": summary["registry_win_rate"],
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_control_dependent_relational_usage_probe(payload, output_path)
    return payload


def build_a34_2_replay_contexts(
    entry: Mapping[str, Any],
    source_sage6f: Mapping[str, Any],
    source_sage6a: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Join A33.3 hashes to provenance and replay requests without using outcomes."""
    clusters_by_hash = {
        str(row.get("context_snapshot_hash", "")): row
        for row in source_sage6f.get("context_cluster_manifest", []) or []
        if isinstance(row, Mapping) and bool(row.get("paired_control_context", False))
    }
    requests_by_id = {
        str(row.get("request_id", "")): row
        for row in source_sage6a.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping)
    }
    contexts: List[Dict[str, Any]] = []
    for context_hash in entry.get("paired_context_snapshot_hashes", []) or []:
        cluster = dict(clusters_by_hash.get(str(context_hash), {}) or {})
        source_ids = list(cluster.get("source_request_ids", []) or [])
        if len(source_ids) != 1:
            raise ValueError("each A33.3 context must map to one replay request")
        request = dict(requests_by_id.get(str(source_ids[0]), {}) or {})
        if not request:
            raise ValueError("A33.3 replay request is missing from SAGE.6a")
        budgets = list(cluster.get("budgets", []) or [])
        contexts.append(
            {
                "context_cluster_id": str(cluster.get("context_cluster_id", "")),
                "context_snapshot_hash": str(context_hash),
                "source_request_id": str(source_ids[0]),
                "source_step": int(request.get("source_step", 0) or 0),
                "budget": int(budgets[0]) if len(budgets) == 1 else 0,
                "game_id": str(request.get("game_id", "")),
                "metric": str(request.get("metric", "")),
                "context_replay": list(request.get("context_replay", []) or []),
                "context_replay_args": [
                    dict(row) for row in request.get("context_replay_args", []) or []
                ],
                "request": request,
                "provenance_used_for_action_choice": False,
                "registry_used_for_action_choice": True,
            }
        )
    return tuple(contexts)


def execute_a34_2_context_probe(
    entry: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> ControlDependentRelationalUsageProbeResult:
    """Execute baseline, registry and equivalence actions from one exact prefix."""
    request = dict(context.get("request", {}) or {})
    controls = [str(value) for value in entry.get("control_actions", []) or []]
    baseline_action = controls[0]
    equivalent_action = controls[1]
    registry_action = str(entry.get("target_action", ""))
    baseline = _execute_request_arm(
        request,
        action_name=baseline_action,
        action_args={},
        arm="no_memory_baseline",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    treatment = _execute_request_arm(
        request,
        action_name=registry_action,
        action_args={},
        arm="a33_3_registry_policy",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    equivalent = _execute_request_arm(
        request,
        action_name=equivalent_action,
        action_args={},
        arm="equivalent_control_audit",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    arms = (baseline, treatment, equivalent)
    if any(str(arm.get("status", "")) != "EXECUTED" for arm in arms):
        reasons = [str(arm.get("reason", "")) for arm in arms]
        raise ValueError(f"A34.2 exact replay arm blocked: {reasons}")
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
        in set(entry.get("paired_context_snapshot_hashes", []) or [])
    )
    return ControlDependentRelationalUsageProbeResult(
        context_cluster_id=str(context.get("context_cluster_id", "")),
        context_snapshot_hash=str(context.get("context_snapshot_hash", "")),
        source_request_id=str(context.get("source_request_id", "")),
        source_step=int(context.get("source_step", 0) or 0),
        budget=int(context.get("budget", 0) or 0),
        game_id=str(context.get("game_id", "")),
        metric=str(context.get("metric", "")),
        baseline_action=baseline_action,
        registry_action=registry_action,
        equivalent_action=equivalent_action,
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
        action_choice_changed=baseline_action != registry_action,
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


def build_a34_2_gate(
    entry: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    probes: Sequence[ControlDependentRelationalUsageProbeResult],
) -> Dict[str, bool]:
    expected_hashes = list(entry.get("paired_context_snapshot_hashes", []) or [])
    return {
        "all_registered_contexts_reconstructed_once": len(contexts)
        == len(probes)
        == len(expected_hashes)
        == 3
        and {str(row.get("context_snapshot_hash", "")) for row in contexts}
        == set(expected_hashes),
        "all_replays_exact_for_all_arms": all(
            probe.replay_exact_all_arms for probe in probes
        ),
        "registry_policy_scope_matched": all(probe.scope_match for probe in probes),
        "registry_changes_baseline_choice": all(
            probe.action_choice_changed for probe in probes
        ),
        "registry_beats_recorded_lower_effect_control": all(
            probe.registry_gain_over_baseline == 32.0 for probe in probes
        ),
        "registry_preserves_equivalent_control_limit": all(
            probe.registry_gain_over_equivalent == 0.0 for probe in probes
        ),
        "utility_is_relational_not_autonomous": all(
            probe.utility_assessment == UTILITY_ASSESSMENT for probe in probes
        ),
        "no_scientific_verdict_or_support_created": all(
            not probe.scientific_verdict_performed and probe.support_counted == 0
            for probe in probes
        ),
        "wrong_confirmations_zero": all(
            probe.wrong_confirmations == 0 for probe in probes
        ),
    }


def summarize_a34_2(
    probes: Sequence[ControlDependentRelationalUsageProbeResult],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    useful = [
        probe for probe in probes if probe.utility_assessment == UTILITY_ASSESSMENT
    ]
    registry_wins = sum(probe.registry_win for probe in probes)
    outcome = (
        A34_2_RELATION_USEFUL
        if len(useful) == len(probes)
        else A34_2_RELATION_NOT_USEFUL
    )
    return {
        "registered_relations_probed": 1 if probes else 0,
        "exact_contexts_probed": len(probes),
        "baseline_actions": _counts(probe.baseline_action for probe in probes),
        "registry_actions": _counts(probe.registry_action for probe in probes),
        "equivalent_actions": _counts(probe.equivalent_action for probe in probes),
        "action_choices_changed": sum(probe.action_choice_changed for probe in probes),
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
        "a34_truth_reevaluations": 0,
        "support_counted": 0,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "status": "CONTEXTUALLY_USEFUL" if useful else "NOT_USEFUL",
        "outcome_status": outcome,
    }


def validate_a34_2_sources(
    registry: Mapping[str, Any],
    source_sage6f: Mapping[str, Any],
    source_sage6a: Mapping[str, Any],
) -> None:
    """Require exact registered scope and candidate-only replay provenance."""
    registry_config = dict(registry.get("config", {}) or {})
    if str(registry_config.get("schema_version", "")) != A33_3_SCHEMA_VERSION:
        raise ValueError("A34.2 requires the A33.3 registry schema")
    if (
        str(registry.get("outcome_status", "")) != A33_3_ENTRY_ADDED
        or str(registry.get("truth_status", "")) != A33_3_TRUTH_STATUS
        or str(registry.get("status", "")) != "REGISTERED"
        or not bool(registry.get("registration_performed", False))
        or bool(registry.get("confirmation_performed", True))
        or int(registry.get("wrong_confirmations", 0) or 0) != 0
    ):
        raise ValueError("A34.2 requires the completed non-reevaluating A33.3 registry")
    if not bool(registry.get("summary", {}).get("gate_passed", False)) or not all(
        bool(value) for value in registry.get("gate", {}).values()
    ):
        raise ValueError("every A33.3 registry gate must pass")
    entries = [
        row
        for row in registry.get("control_dependent_relational_contrasts", []) or []
        if isinstance(row, Mapping)
    ]
    if len(entries) != 1:
        raise ValueError("A34.2 requires exactly one A33.3 relation")
    entry = entries[0]
    if (
        str(entry.get("registry_entry_type", ""))
        != CONTROL_DEPENDENT_RELATIONAL_CONTRAST
        or str(entry.get("game_id", "")) != "wa30-ee6fef47"
        or str(entry.get("target_action", "")) != "ACTION2"
        or list(entry.get("control_actions", []) or []) != ["ACTION1", "ACTION3"]
        or str(entry.get("predicted_metric", "")) != "local_patch_before_after"
        or int(entry.get("confirmed_support", 0) or 0) != 3
        or int(entry.get("contradictions", 0) or 0) != 0
        or len(entry.get("paired_context_snapshot_hashes", []) or []) != 3
    ):
        raise ValueError("A33.3 relation identity and scope must remain exact")
    if (
        not bool(entry.get("scope_game_locked", False))
        or not bool(entry.get("scope_contexts_locked", False))
        or not bool(entry.get("scope_target_action_locked", False))
        or not bool(entry.get("scope_control_actions_locked", False))
        or not bool(entry.get("scope_metric_locked", False))
        or not bool(entry.get("standalone_action2_effect_excluded", False))
        or str(entry.get("standalone_action2_effect_status", "")) != "unresolved"
    ):
        raise ValueError("A33.3 relation scope and exclusions must remain locked")
    sage6f_config = dict(source_sage6f.get("config", {}) or {})
    if (
        str(sage6f_config.get("schema_version", "")) != SAGE6F_SCHEMA_VERSION
        or str(source_sage6f.get("outcome_status", "")) != SAGE6F_A32_REVIEW_ELIGIBLE
        or str(source_sage6f.get("truth_status", "")) != SAGE6F_TRUTH_STATUS
        or int(source_sage6f.get("support", 0) or 0) != 0
        or bool(source_sage6f.get("revision_performed", True))
    ):
        raise ValueError("A34.2 requires candidate-only SAGE.6f provenance")
    sage6a_config = dict(source_sage6a.get("config", {}) or {})
    if (
        str(sage6a_config.get("schema_version", "")) != SAGE6A_SCHEMA_VERSION
        or str(source_sage6a.get("outcome_status", "")) != SAGE6A_FRONTIER_GENERATED
        or str(source_sage6a.get("truth_status", "")) != SAGE6A_TRUTH_STATUS
        or int(source_sage6a.get("support", 0) or 0) != 0
        or bool(source_sage6a.get("revision_performed", True))
    ):
        raise ValueError("A34.2 requires candidate-only SAGE.6a replay provenance")
    contexts = build_a34_2_replay_contexts(entry, source_sage6f, source_sage6a)
    hashes = [str(row.get("context_snapshot_hash", "")) for row in contexts]
    if hashes != list(entry.get("paired_context_snapshot_hashes", []) or []):
        raise ValueError(
            "A34.2 replay contexts must exactly match A33.3 order and scope"
        )
    if any(
        str(row.get("game_id", "")) != "wa30-ee6fef47"
        or str(row.get("metric", "")) != "local_patch_before_after"
        or len(row.get("context_replay", []) or []) != int(row.get("source_step", 0))
        or not row.get("context_snapshot_hash", "")
        for row in contexts
    ):
        raise ValueError("A34.2 replay provenance must remain exact and complete")


def write_control_dependent_relational_usage_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _arm_signal(arm: Mapping[str, Any]) -> float:
    return float(
        arm.get("measurement_for_delta", {}).get("local_changed_pixels", 0) or 0
    )


def _signature_payload(value: Any) -> Dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _signature_won(signature: Mapping[str, Any]) -> bool:
    state = str(signature.get("game_state", "")).upper()
    return state in {"WIN", "WON", "VICTORY"}


def _counts(values: Sequence[str] | Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--source-sage6f",
        default=str(DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH),
    )
    parser.add_argument(
        "--source-sage6a",
        default=str(DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH),
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH),
    )
    args = parser.parse_args(argv)
    payload = run_control_dependent_relational_usage_probe(
        registry_path=args.registry,
        source_sage6f_path=args.source_sage6f,
        source_sage6a_path=args.source_sage6a,
        environments_dir=args.environments_dir,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
