"""P3.G2 contextual post-stop conversion policy probe.

This module consumes M3.G4 as policy material. It does not rerun the M3
objective-conversion cells and does not turn a policy utility result into a
mechanic verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p3.abstract_mechanic_policy_probe import (
    ABSTRACT_MECHANIC_POLICY,
    TRUTH_STATUS,
)


DEFAULT_M3G4_DIVERSE_VALIDATION_PATH = (
    Path("diagnostics")
    / "m3"
    / "objective_conversion_diverse_safe_stop_validation.json"
)
DEFAULT_P3G0_POLICY_PROBE_PATH = (
    Path("diagnostics") / "p3" / "abstract_mechanic_policy_probe.json"
)
DEFAULT_P3G1_CONSOLIDATION_PATH = (
    Path("diagnostics")
    / "p3"
    / "objective_aware_abstract_policy_utility_consolidation.json"
)
DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH = (
    Path("diagnostics")
    / "p3"
    / "contextual_post_stop_conversion_policy_adapter.json"
)
DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH = (
    Path("diagnostics")
    / "p3"
    / "contextual_post_stop_conversion_policy_probe.json"
)

P3G2_SCHEMA_VERSION = "p3.contextual_post_stop_conversion_policy_probe.v1"

HOLD_OR_STOP_STATE = "hold_or_stop_state"
ACTION6_ONLY = "ACTION6"
ACTION6_ACTION3 = "ACTION6,ACTION3"
ACTION6_ACTION4 = "ACTION6,ACTION4"
CONTEXTUAL_POLICY = "contextual_post_stop_conversion_policy"

POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY = (
    "POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY"
)
POST_STOP_CONTEXTUAL_POLICY_NEUTRAL_CANDIDATE_ONLY = (
    "POST_STOP_CONTEXTUAL_POLICY_NEUTRAL_CANDIDATE_ONLY"
)
POST_STOP_CONTEXTUAL_POLICY_RISKY_CANDIDATE_ONLY = (
    "POST_STOP_CONTEXTUAL_POLICY_RISKY_CANDIDATE_ONLY"
)
POST_STOP_CONTEXTUAL_POLICY_HARMFUL_CANDIDATE_ONLY = (
    "POST_STOP_CONTEXTUAL_POLICY_HARMFUL_CANDIDATE_ONLY"
)
POST_STOP_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY = (
    "POST_STOP_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY"
)

DEFAULT_EXTENSION_SEQUENCES = (("ACTION6", "ACTION3"), ("ACTION6", "ACTION4"))
DEFAULT_FALLBACK_SEQUENCE = ("ACTION6",)
DEFAULT_MIN_EXTENSION_MARGIN_VS_ACTION6 = 5.0


def build_contextual_post_stop_conversion_policy_adapter(
    *,
    source_m3g4_path: str | Path = DEFAULT_M3G4_DIVERSE_VALIDATION_PATH,
    min_extension_margin_vs_action6: float = DEFAULT_MIN_EXTENSION_MARGIN_VS_ACTION6,
) -> Dict[str, Any]:
    payload = _load_json(source_m3g4_path)
    validate_m3g4_source(payload)
    safe_stop_records = _safe_stop_records(payload)
    risk_model = derive_extension_risk_model(safe_stop_records)
    adapter = {
        "policy_adapter_id": "p3g2::bp35::contextual_post_stop_conversion_policy_adapter",
        "source_m3g4_path": str(source_m3g4_path),
        "adapter_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        "decision_options": [
            HOLD_OR_STOP_STATE,
            ACTION6_ONLY,
            ACTION6_ACTION3,
            ACTION6_ACTION4,
        ],
        "fallback_sequence": list(DEFAULT_FALLBACK_SEQUENCE),
        "extension_sequences": [list(seq) for seq in DEFAULT_EXTENSION_SEQUENCES],
        "extension_sequence_preference": [ACTION6_ACTION3, ACTION6_ACTION4],
        "decision_features": [
            "sampling_family",
            "terminal_horizon_remaining",
            "terminal_horizon_band",
            "hold_baseline_terminal_adjusted_progress",
            "hold_baseline_band",
            "relation_progress_control_summary.best_relation_taps",
            "relation_progress_control_summary.any_relation_control_terminal_reentry",
            "candidate_record.new_relation_states_when_available",
            "changed_pixels_when_available",
        ],
        "selector_rule": {
            "default": "use ACTION6 if it is nonterminal and beats hold",
            "extension_gate": (
                "allow ACTION6,ACTION3/ACTION6,ACTION4 only when the "
                "sampling_family+terminal_horizon_band scope and hold_baseline_band "
                "did not show terminal re-entry in M3.G4"
            ),
            "margin_rule": (
                "extension terminal_adjusted_progress must exceed ACTION6 by "
                f"at least {float(min_extension_margin_vs_action6)}"
            ),
            "fallback_if_blocked": "ACTION6, otherwise hold_or_stop_state",
        },
        "min_extension_margin_vs_action6": float(min_extension_margin_vs_action6),
        "risk_model": risk_model,
        "source_m3g4_validation_outcome_status": str(
            payload.get("validation_outcome_status", "")
            or (payload.get("summary", {}) or {}).get("validation_outcome_status", "")
        ),
        "source_m3g4_counted_as_mechanic_confirmation": False,
        "policy_adapter_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    return {
        "config": {
            "schema_version": P3G2_SCHEMA_VERSION,
            "stage": "P3.G2.1",
            "source_m3g4_path": str(source_m3g4_path),
            "execution_performed": False,
        },
        "summary": {
            "safe_stops_available": len(safe_stop_records),
            "fallback_sequence": list(DEFAULT_FALLBACK_SEQUENCE),
            "extension_sequences": [list(seq) for seq in DEFAULT_EXTENSION_SEQUENCES],
            "blocked_sampling_family_horizon_pairs": len(
                risk_model["blocked_sampling_family_horizon_pairs"]
            ),
            "blocked_hold_baseline_bands": len(
                risk_model["blocked_hold_baseline_bands"]
            ),
            "unsafe_extension_options_observed": risk_model[
                "unsafe_extension_options_observed"
            ],
            "ready_for_contextual_policy_probe": bool(safe_stop_records),
            "source_m3g4_counted_as_mechanic_confirmation": False,
            "policy_adapter_counted_as_confirmation": False,
            "policy_result_counted_as_scientific_verdict": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "contextual_post_stop_policy_adapter": adapter,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_adapter_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def run_contextual_post_stop_conversion_policy_probe(
    *,
    source_m3g4_path: str | Path = DEFAULT_M3G4_DIVERSE_VALIDATION_PATH,
    adapter_path: str | Path = DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH,
    p3g0_policy_probe_path: str | Path | None = DEFAULT_P3G0_POLICY_PROBE_PATH,
    p3g1_consolidation_path: str | Path | None = DEFAULT_P3G1_CONSOLIDATION_PATH,
) -> Dict[str, Any]:
    source_payload = _load_json(source_m3g4_path)
    validate_m3g4_source(source_payload)
    adapter_payload = _load_json(adapter_path)
    validate_adapter_source(adapter_payload)
    adapter = dict(adapter_payload.get("contextual_post_stop_policy_adapter", {}) or {})
    safe_stop_records = _safe_stop_records(source_payload)

    policy_records = [
        select_contextual_post_stop_option(record, adapter=adapter)
        for record in safe_stop_records
    ]
    baseline_records = build_baseline_policy_records(safe_stop_records)
    baseline_aggregates = {
        condition: aggregate_option_records(condition, records)
        for condition, records in sorted(baseline_records.items())
    }
    contextual_aggregate = aggregate_option_records(CONTEXTUAL_POLICY, policy_records)
    policy_status = contextual_policy_status(
        contextual_aggregate=contextual_aggregate,
        action6_aggregate=baseline_aggregates.get(ACTION6_ONLY, {}),
    )
    p3_reference = load_policy_reference_comparison(
        p3g0_policy_probe_path=p3g0_policy_probe_path,
        p3g1_consolidation_path=p3g1_consolidation_path,
        contextual_aggregate=contextual_aggregate,
    )

    action6_mean = float(
        baseline_aggregates.get(ACTION6_ONLY, {}).get(
            "mean_terminal_adjusted_progress", 0.0
        )
        or 0.0
    )
    selected_mean = float(
        contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    unsafe_options_avoided = sum(
        int(record.get("unsafe_extension_options_avoided", 0) or 0)
        for record in policy_records
    )
    unsafe_safe_stops_avoided = len(
        {
            str(record.get("safe_stop_id", ""))
            for record in policy_records
            if int(record.get("unsafe_extension_options_avoided", 0) or 0) > 0
        }
    )

    summary = {
        "policy_utility_status": policy_status,
        "safe_stops_evaluated": len(policy_records),
        "selected_hold_count": sum(
            1 for record in policy_records if record.get("selected_option") == HOLD_OR_STOP_STATE
        ),
        "selected_action6_only_count": sum(
            1 for record in policy_records if record.get("selected_option") == ACTION6_ONLY
        ),
        "selected_extension_count": sum(
            1
            for record in policy_records
            if str(record.get("selected_option", "")).startswith("ACTION6,ACTION")
        ),
        "abstention_or_hold_rate": round(
            mean(
                1.0 if record.get("selected_option") == HOLD_OR_STOP_STATE else 0.0
                for record in policy_records
            ),
            6,
        ),
        "terminal_rate": contextual_aggregate["terminal_rate"],
        "mean_terminal_adjusted_progress": contextual_aggregate[
            "mean_terminal_adjusted_progress"
        ],
        "mean_delta_vs_hold": contextual_aggregate["mean_delta_vs_hold"],
        "mean_delta_vs_action6_only": round(selected_mean - action6_mean, 6),
        "improvement_over_action6_only": selected_mean > action6_mean,
        "unsafe_extension_options_avoided": unsafe_options_avoided,
        "unsafe_extension_safe_stops_avoided": unsafe_safe_stops_avoided,
        "objective_completion_signal": bool(
            contextual_aggregate["objective_completion_runs"] > 0
        ),
        "objective_completion_runs": contextual_aggregate["objective_completion_runs"],
        "baseline_aggregates": baseline_aggregates,
        "contextual_policy_aggregate": contextual_aggregate,
        "p3_reference_comparison": p3_reference,
        "source_m3g4_validation_outcome_status": str(
            source_payload.get("validation_outcome_status", "")
            or (source_payload.get("summary", {}) or {}).get(
                "validation_outcome_status", ""
            )
        ),
        "policy_result_counted_as_scientific_verdict": False,
        "source_m3g4_counted_as_mechanic_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    return {
        "config": {
            "schema_version": P3G2_SCHEMA_VERSION,
            "stage": "P3.G2.2",
            "source_m3g4_path": str(source_m3g4_path),
            "adapter_path": str(adapter_path),
            "p3g0_policy_probe_path": (
                None if p3g0_policy_probe_path is None else str(p3g0_policy_probe_path)
            ),
            "p3g1_consolidation_path": (
                None
                if p3g1_consolidation_path is None
                else str(p3g1_consolidation_path)
            ),
            "execution_performed": False,
            "source_cells_rerun": False,
            "policy_options": [
                HOLD_OR_STOP_STATE,
                ACTION6_ONLY,
                ACTION6_ACTION3,
                ACTION6_ACTION4,
            ],
        },
        "summary": summary,
        "policy_decision_records": policy_records,
        "baseline_aggregates": baseline_aggregates,
        "contextual_policy_aggregate": contextual_aggregate,
        "policy_utility_status": policy_status,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "source_m3g4_counted_as_mechanic_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def derive_extension_risk_model(
    safe_stop_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    terminal_extension_records: list[Dict[str, Any]] = []
    family_horizon_pairs: set[str] = set()
    hold_bands: set[str] = set()
    family_horizon_hold_triples: set[str] = set()
    for safe_stop in safe_stop_records:
        family = str(safe_stop.get("sampling_family", ""))
        horizon_band = str(safe_stop.get("terminal_horizon_band", ""))
        hold_band = str(safe_stop.get("hold_baseline_band", ""))
        for candidate in _candidate_records(safe_stop):
            sequence = sequence_key(candidate.get("action_or_sequence", []))
            if sequence == ACTION6_ONLY:
                continue
            if not bool(candidate.get("candidate_terminal_reentry", False)):
                continue
            family_horizon_pairs.add(scope_key(family, horizon_band))
            hold_bands.add(hold_band)
            family_horizon_hold_triples.add(scope_key(family, horizon_band, hold_band))
            terminal_extension_records.append(
                {
                    "safe_stop_id": str(safe_stop.get("safe_stop_id", "")),
                    "selected_if_unblocked": sequence,
                    "sampling_family": family,
                    "terminal_horizon_band": horizon_band,
                    "hold_baseline_band": hold_band,
                    "candidate_terminal_adjusted_progress_after_stop": float(
                        candidate.get(
                            "candidate_terminal_adjusted_progress_after_stop", 0.0
                        )
                        or 0.0
                    ),
                    "delta_terminal_adjusted_progress_vs_hold": float(
                        candidate.get("delta_terminal_adjusted_progress_vs_hold", 0.0)
                        or 0.0
                    ),
                }
            )
    return {
        "blocked_sampling_family_horizon_pairs": sorted(family_horizon_pairs),
        "blocked_hold_baseline_bands": sorted(hold_bands),
        "blocked_sampling_family_horizon_hold_triples": sorted(
            family_horizon_hold_triples
        ),
        "terminal_extension_records": terminal_extension_records,
        "unsafe_extension_options_observed": len(terminal_extension_records),
        "risk_model_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def select_contextual_post_stop_option(
    safe_stop: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
) -> Dict[str, Any]:
    hold = hold_option_record(safe_stop)
    candidates = {sequence_key(row.get("action_or_sequence", [])): row for row in _candidate_records(safe_stop)}
    action6 = dict(candidates.get(ACTION6_ONLY, {}))
    fallback = hold
    if is_viable_action6(action6, hold):
        fallback = option_from_candidate(action6, selected_reason="safe_action6_fallback")

    min_margin = float(
        adapter.get("min_extension_margin_vs_action6", DEFAULT_MIN_EXTENSION_MARGIN_VS_ACTION6)
        or DEFAULT_MIN_EXTENSION_MARGIN_VS_ACTION6
    )
    blocked_extensions: list[Dict[str, Any]] = []
    extension_options: list[Dict[str, Any]] = []
    action6_taps = float(fallback.get("terminal_adjusted_progress", 0.0) or 0.0)
    for sequence in adapter.get("extension_sequence_preference", []) or []:
        candidate = dict(candidates.get(str(sequence), {}) or {})
        if not candidate:
            continue
        blockers = extension_blockers(
            safe_stop,
            candidate,
            adapter=adapter,
            action6_taps=action6_taps,
            min_margin=min_margin,
        )
        if blockers:
            blocked_extensions.append(
                blocked_extension_record(candidate, blockers=blockers)
            )
            continue
        extension_options.append(
            option_from_candidate(
                candidate,
                selected_reason="contextual_extension_allowed_and_beats_action6",
            )
        )

    selected = max(
        extension_options,
        key=lambda row: (
            float(row.get("terminal_adjusted_progress", 0.0) or 0.0),
            -sequence_preference_index(str(row.get("selected_option", "")), adapter),
        ),
        default=fallback,
    )
    unsafe_avoided = [
        row
        for row in _candidate_records(safe_stop)
        if sequence_key(row.get("action_or_sequence", [])) != ACTION6_ONLY
        and bool(row.get("candidate_terminal_reentry", False))
        and sequence_key(row.get("action_or_sequence", []))
        != str(selected.get("selected_option", ""))
    ]
    hold_taps = float(hold.get("terminal_adjusted_progress", 0.0) or 0.0)
    return {
        "safe_stop_id": str(safe_stop.get("safe_stop_id", "")),
        "sampling_family": str(safe_stop.get("sampling_family", "")),
        "terminal_horizon_remaining": safe_stop.get("terminal_horizon_remaining"),
        "terminal_horizon_band": str(safe_stop.get("terminal_horizon_band", "")),
        "hold_baseline_terminal_adjusted_progress": hold_taps,
        "hold_baseline_band": str(safe_stop.get("hold_baseline_band", "")),
        "relation_progress_best_taps": float(
            (safe_stop.get("relation_progress_control_summary", {}) or {}).get(
                "best_relation_taps", 0.0
            )
            or 0.0
        ),
        **selected,
        "delta_vs_hold": round(
            float(selected.get("terminal_adjusted_progress", 0.0) or 0.0) - hold_taps,
            6,
        ),
        "delta_vs_action6_only": round(
            float(selected.get("terminal_adjusted_progress", 0.0) or 0.0)
            - float(action6.get("candidate_terminal_adjusted_progress_after_stop", hold_taps) or hold_taps),
            6,
        ),
        "blocked_extension_records": blocked_extensions,
        "unsafe_extension_options_avoided": len(unsafe_avoided),
        "policy_decision_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def extension_blockers(
    safe_stop: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
    action6_taps: float,
    min_margin: float,
) -> list[str]:
    risk_model = dict(adapter.get("risk_model", {}) or {})
    family = str(safe_stop.get("sampling_family", ""))
    horizon_band = str(safe_stop.get("terminal_horizon_band", ""))
    hold_band = str(safe_stop.get("hold_baseline_band", ""))
    blockers: list[str] = []
    if scope_key(family, horizon_band) in set(
        risk_model.get("blocked_sampling_family_horizon_pairs", []) or []
    ):
        blockers.append("blocked_sampling_family_horizon_pair")
    if hold_band in set(risk_model.get("blocked_hold_baseline_bands", []) or []):
        blockers.append("blocked_hold_baseline_band")
    if bool(candidate.get("candidate_terminal_reentry", False)):
        blockers.append("candidate_terminal_reentry")
    candidate_taps = float(
        candidate.get("candidate_terminal_adjusted_progress_after_stop", 0.0) or 0.0
    )
    if candidate_taps - action6_taps < min_margin:
        blockers.append("extension_margin_vs_action6_too_small")
    if float(candidate.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0) <= 0.0:
        blockers.append("does_not_beat_hold")
    return blockers


def build_baseline_policy_records(
    safe_stop_records: Sequence[Mapping[str, Any]],
) -> Dict[str, list[Dict[str, Any]]]:
    baselines = {
        HOLD_OR_STOP_STATE: [],
        ACTION6_ONLY: [],
        ACTION6_ACTION3: [],
        ACTION6_ACTION4: [],
    }
    for safe_stop in safe_stop_records:
        baselines[HOLD_OR_STOP_STATE].append(hold_option_record(safe_stop))
        candidates = {sequence_key(row.get("action_or_sequence", [])): row for row in _candidate_records(safe_stop)}
        for sequence in (ACTION6_ONLY, ACTION6_ACTION3, ACTION6_ACTION4):
            if sequence in candidates:
                baselines[sequence].append(
                    option_from_candidate(
                        candidates[sequence],
                        selected_reason=f"baseline_always_{sequence}",
                    )
                )
    return baselines


def aggregate_option_records(
    condition: str,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "condition": condition,
        "runs": len(records),
        "mean_terminal_adjusted_progress": round(
            mean(row.get("terminal_adjusted_progress", 0.0) for row in records),
            6,
        ),
        "mean_delta_vs_hold": round(
            mean(row.get("delta_vs_hold", 0.0) for row in records),
            6,
        ),
        "terminal_rate": round(
            mean(1.0 if row.get("terminal_reentry") else 0.0 for row in records),
            6,
        ),
        "objective_completion_runs": sum(
            1 for row in records if bool(row.get("objective_completion_signal", False))
        ),
        "mean_levels_completed": round(
            mean(row.get("levels_completed_after_rollout", 0) for row in records),
            6,
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def contextual_policy_status(
    *,
    contextual_aggregate: Mapping[str, Any],
    action6_aggregate: Mapping[str, Any],
) -> str:
    if int(contextual_aggregate.get("objective_completion_runs", 0) or 0) > 0:
        return POST_STOP_OBJECTIVE_COMPLETION_POLICY_CANDIDATE_ONLY
    if float(contextual_aggregate.get("terminal_rate", 0.0) or 0.0) > 0.0:
        return POST_STOP_CONTEXTUAL_POLICY_RISKY_CANDIDATE_ONLY
    contextual_taps = float(
        contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    action6_taps = float(
        action6_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    if contextual_taps > action6_taps:
        return POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY
    if contextual_taps == action6_taps:
        return POST_STOP_CONTEXTUAL_POLICY_NEUTRAL_CANDIDATE_ONLY
    return POST_STOP_CONTEXTUAL_POLICY_HARMFUL_CANDIDATE_ONLY


def load_policy_reference_comparison(
    *,
    p3g0_policy_probe_path: str | Path | None,
    p3g1_consolidation_path: str | Path | None,
    contextual_aggregate: Mapping[str, Any],
) -> Dict[str, Any]:
    contextual_terminal = float(contextual_aggregate.get("terminal_rate", 0.0) or 0.0)
    contextual_taps = float(
        contextual_aggregate.get("mean_terminal_adjusted_progress", 0.0) or 0.0
    )
    p3g0 = load_p3g0_reference(p3g0_policy_probe_path)
    p3g1 = load_p3g1_reference(p3g1_consolidation_path)
    return {
        "p3g0_reference": p3g0,
        "p3g1_reference": p3g1,
        "terminal_rate_delta_vs_p3g0": (
            None
            if p3g0.get("terminal_rate") is None
            else round(contextual_terminal - float(p3g0["terminal_rate"]), 6)
        ),
        "terminal_rate_delta_vs_p3g1": (
            None
            if p3g1.get("terminal_rate") is None
            else round(contextual_terminal - float(p3g1["terminal_rate"]), 6)
        ),
        "terminal_adjusted_progress_delta_vs_p3g0": (
            None
            if p3g0.get("terminal_adjusted_progress") is None
            else round(contextual_taps - float(p3g0["terminal_adjusted_progress"]), 6)
        ),
        "terminal_adjusted_progress_delta_vs_p3g1": (
            None
            if p3g1.get("terminal_adjusted_progress") is None
            else round(contextual_taps - float(p3g1["terminal_adjusted_progress"]), 6)
        ),
        "comparison_scale_note": (
            "P3.G2 is evaluated on post-stop safe-stop records; P3.G0/P3.G1 "
            "references are full rollout aggregates and are terminal-rate "
            "references, not mechanic confirmation."
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def load_p3g0_reference(path: str | Path | None) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"available": False, "terminal_rate": None, "terminal_adjusted_progress": None}
    payload = _load_json(path)
    validate_policy_reference_source(payload, source_name="P3.G0")
    aggregate = (
        (payload.get("summary", {}) or {})
        .get("condition_aggregates", {})
        .get(ABSTRACT_MECHANIC_POLICY, {})
    )
    return {
        "available": bool(aggregate),
        "condition": ABSTRACT_MECHANIC_POLICY,
        "terminal_rate": aggregate.get("terminal_rate"),
        "terminal_adjusted_progress": aggregate.get("mean_terminal_adjusted_progress"),
        "mean_progress_proxy": aggregate.get("mean_progress_proxy"),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def load_p3g1_reference(path: str | Path | None) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"available": False, "terminal_rate": None, "terminal_adjusted_progress": None}
    payload = _load_json(path)
    validate_policy_reference_source(payload, source_name="P3.G1")
    summary = dict(payload.get("summary", {}) or {})
    return {
        "available": bool(summary),
        "condition": str(summary.get("best_objective_aware_condition", "")),
        "terminal_rate": summary.get("best_objective_terminal_rate"),
        "terminal_adjusted_progress": summary.get(
            "best_objective_terminal_adjusted_progress"
        ),
        "mean_progress_proxy": summary.get("best_objective_mean_progress_proxy"),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def hold_option_record(safe_stop: Mapping[str, Any]) -> Dict[str, Any]:
    hold_taps = float(
        safe_stop.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
    )
    return {
        "selected_option": HOLD_OR_STOP_STATE,
        "selected_action_or_sequence": [],
        "selected_reason": "hold_or_stop_state_baseline",
        "terminal_adjusted_progress": round(hold_taps, 6),
        "terminal_reentry": False,
        "objective_completion_signal": False,
        "levels_completed_after_rollout": 0,
        "delta_vs_hold": 0.0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def option_from_candidate(
    candidate: Mapping[str, Any],
    *,
    selected_reason: str,
) -> Dict[str, Any]:
    sequence = [str(action) for action in candidate.get("action_or_sequence", []) or []]
    selected_option = sequence_key(sequence)
    return {
        "selected_option": selected_option,
        "selected_action_or_sequence": sequence,
        "selected_reason": selected_reason,
        "terminal_adjusted_progress": round(
            float(candidate.get("candidate_terminal_adjusted_progress_after_stop", 0.0) or 0.0),
            6,
        ),
        "terminal_reentry": bool(candidate.get("candidate_terminal_reentry", False)),
        "objective_completion_signal": bool(
            candidate.get("objective_completion_signal", False)
        ),
        "levels_completed_after_rollout": int(
            candidate.get("candidate_levels_completed_after_rollout", 0) or 0
        ),
        "delta_vs_hold": round(
            float(candidate.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0),
            6,
        ),
        "beats_relation_progress_policy": bool(
            candidate.get("beats_relation_progress_policy", False)
        ),
        "signal_strength": str(candidate.get("signal_strength", "")),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def blocked_extension_record(
    candidate: Mapping[str, Any],
    *,
    blockers: Sequence[str],
) -> Dict[str, Any]:
    return {
        "action_or_sequence": [str(action) for action in candidate.get("action_or_sequence", []) or []],
        "sequence_key": sequence_key(candidate.get("action_or_sequence", [])),
        "blockers": list(blockers),
        "candidate_terminal_reentry": bool(candidate.get("candidate_terminal_reentry", False)),
        "candidate_terminal_adjusted_progress_after_stop": float(
            candidate.get("candidate_terminal_adjusted_progress_after_stop", 0.0) or 0.0
        ),
        "delta_terminal_adjusted_progress_vs_hold": float(
            candidate.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def is_viable_action6(
    action6: Mapping[str, Any],
    hold: Mapping[str, Any],
) -> bool:
    if not action6:
        return False
    if bool(action6.get("candidate_terminal_reentry", False)):
        return False
    candidate_taps = float(
        action6.get("candidate_terminal_adjusted_progress_after_stop", 0.0) or 0.0
    )
    hold_taps = float(hold.get("terminal_adjusted_progress", 0.0) or 0.0)
    return candidate_taps >= hold_taps


def sequence_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(action) for action in value or [])


def scope_key(*parts: str) -> str:
    return "|".join(str(part) for part in parts)


def sequence_preference_index(sequence: str, adapter: Mapping[str, Any]) -> int:
    preference = [str(value) for value in adapter.get("extension_sequence_preference", []) or []]
    try:
        return preference.index(sequence)
    except ValueError:
        return len(preference)


def mean(values: Sequence[Any] | Any) -> float:
    vals = [float(value or 0.0) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _safe_stop_records(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        dict(record)
        for record in payload.get("per_safe_stop_validation_records", []) or []
        if isinstance(record, Mapping)
    ]


def _candidate_records(safe_stop: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        dict(candidate)
        for candidate in safe_stop.get("candidate_records", []) or []
        if isinstance(candidate, Mapping)
    ]


def validate_m3g4_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G4 source support must remain 0")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("M3.G4 source must not write A32/A33")
    if bool(payload.get("validation_outcome_status_counted_as_scientific_verdict", False)):
        raise ValueError("M3.G4 source outcome cannot be a scientific verdict")
    if not payload.get("per_safe_stop_validation_records"):
        raise ValueError("P3.G2 requires M3.G4 per-safe-stop records")


def validate_adapter_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G2 adapter support must remain 0")
    if bool(payload.get("policy_adapter_counted_as_confirmation", False)):
        raise ValueError("P3.G2 adapter cannot be confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("P3.G2 adapter cannot be a verdict")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("P3.G2 adapter must not write A32/A33")
    if not bool(summary.get("ready_for_contextual_policy_probe", False)):
        raise ValueError("P3.G2 adapter is not ready")


def validate_policy_reference_source(
    payload: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError(f"{source_name} support must remain 0")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError(f"{source_name} policy reference cannot be a verdict")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError(f"{source_name} reference must not write A32/A33")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate P3.G2 contextual post-stop policy.",
    )
    parser.add_argument("--stage", choices=("adapter", "probe", "all"), default="all")
    parser.add_argument("--source-m3g4", type=Path, default=DEFAULT_M3G4_DIVERSE_VALIDATION_PATH)
    parser.add_argument("--adapter-out", type=Path, default=DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_CONTEXTUAL_POST_STOP_ADAPTER_OUTPUT_PATH)
    parser.add_argument("--probe-out", type=Path, default=DEFAULT_CONTEXTUAL_POST_STOP_PROBE_OUTPUT_PATH)
    parser.add_argument("--p3g0-policy-probe", type=Path, default=DEFAULT_P3G0_POLICY_PROBE_PATH)
    parser.add_argument("--p3g1-consolidation", type=Path, default=DEFAULT_P3G1_CONSOLIDATION_PATH)
    parser.add_argument(
        "--min-extension-margin-vs-action6",
        type=float,
        default=DEFAULT_MIN_EXTENSION_MARGIN_VS_ACTION6,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    adapter_path = args.adapter
    if args.stage in {"adapter", "all"}:
        adapter = build_contextual_post_stop_conversion_policy_adapter(
            source_m3g4_path=args.source_m3g4,
            min_extension_margin_vs_action6=args.min_extension_margin_vs_action6,
        )
        write_json(adapter, args.adapter_out)
        adapter_path = args.adapter_out
    if args.stage in {"probe", "all"}:
        probe = run_contextual_post_stop_conversion_policy_probe(
            source_m3g4_path=args.source_m3g4,
            adapter_path=adapter_path,
            p3g0_policy_probe_path=args.p3g0_policy_probe,
            p3g1_consolidation_path=args.p3g1_consolidation,
        )
        write_json(probe, args.probe_out)
        print(
            json.dumps(
                {
                    "adapter_out": str(args.adapter_out),
                    "probe_out": str(args.probe_out),
                    "policy_utility_status": probe["summary"]["policy_utility_status"],
                    "safe_stops_evaluated": probe["summary"]["safe_stops_evaluated"],
                    "mean_terminal_adjusted_progress": probe["summary"][
                        "mean_terminal_adjusted_progress"
                    ],
                    "terminal_rate": probe["summary"]["terminal_rate"],
                    "mean_delta_vs_action6_only": probe["summary"][
                        "mean_delta_vs_action6_only"
                    ],
                    "unsafe_extension_options_avoided": probe["summary"][
                        "unsafe_extension_options_avoided"
                    ],
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
