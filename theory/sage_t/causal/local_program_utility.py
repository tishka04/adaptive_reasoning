"""Risk-aware short-program audit primitives for SAGE.T12.5b.4.

This module has no environment access. Labels use observed level and terminal
outcomes, while the frozen causal-progress posterior is inspected only after
the useful and distractor programs have been registered.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .progress import JointCausalProgressPosterior
from .progress_shadow import projection_vector


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def program_id(actions: Sequence[str]) -> str:
    values = tuple(str(item).upper() for item in actions)
    if not values:
        raise ValueError("local program cannot be empty")
    return ">".join(values)


def _summary_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(item["program_length"]), tuple(item["program_actions"]))


def _program_summaries(
    *,
    trials: Sequence[Mapping[str, Any]],
    expected_programs: Sequence[Sequence[str]],
    repetitions_per_program: int,
    transport_actions: Sequence[str],
    features: Sequence[str],
    posterior: JointCausalProgressPosterior,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(item) for item in trials]
    expected = {
        program_id(actions): tuple(str(item).upper() for item in actions)
        for actions in expected_programs
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["program_id"])].append(row)
    schedule = {
        (identifier, repetition)
        for identifier in expected
        for repetition in range(int(repetitions_per_program))
    }
    observed = {
        (str(row["program_id"]), int(row["repetition"])) for row in rows
    }
    schedule_complete = observed == schedule and len(rows) == len(schedule)
    prefix_keys = {_canonical(row.get("prefix_steps", ())) for row in rows}
    context_hashes = {str(row.get("detour_context_hash", "")) for row in rows}
    context_replay_exact = bool(
        rows
        and len(prefix_keys) == 1
        and len(context_hashes) == 1
        and all(row.get("original_prefix_exact") is True for row in rows)
        and all(row.get("prefix_exact") is True for row in rows)
        and all(row.get("detour_available") is True for row in rows)
        and all(row.get("detour_neutral") is True for row in rows)
        and all(row.get("detour_terminal") is False for row in rows)
    )
    transport = {str(item).upper() for item in transport_actions}
    summaries = []
    for identifier, actions in expected.items():
        records = sorted(grouped.get(identifier, ()), key=lambda item: item["repetition"])
        repetitions_exact = bool(
            len(records) == int(repetitions_per_program)
            and {int(item["repetition"]) for item in records}
            == set(range(int(repetitions_per_program)))
        )
        actions_exact = bool(
            records
            and all(
                tuple(str(value).upper() for value in item.get("program_actions", ()))
                == actions
                for item in records
            )
        )
        completion_values = {bool(item.get("program_complete")) for item in records}
        executed_values = {int(item.get("executed_action_count", 0)) for item in records}
        availability_deterministic = bool(
            repetitions_exact
            and actions_exact
            and len(completion_values) == 1
            and len(executed_values) == 1
        )
        effect_keys = {
            _canonical(item.get("candidate_steps", ())) for item in records
        }
        outcomes = {
            (
                int(item.get("level_delta", 0)),
                bool(item.get("terminal")),
                bool(item.get("terminal_failure")),
                str(item.get("terminal_state", "")),
                bool(item.get("program_complete")),
            )
            for item in records
        }
        effect_deterministic = bool(repetitions_exact and len(effect_keys) == 1)
        outcome_deterministic = bool(repetitions_exact and len(outcomes) == 1)
        complete = bool(records and all(item.get("program_complete") for item in records))
        representative = records[0] if records else {}
        candidate_steps = tuple(
            dict(item) for item in representative.get("candidate_steps", ())
        )
        prefix_steps = tuple(dict(item) for item in representative.get("prefix_steps", ()))
        magnitude = None
        causal_gain = None
        if complete and effect_deterministic and outcome_deterministic:
            magnitude = float(
                sum(
                    abs(value)
                    for step in candidate_steps
                    for value in projection_vector(step, features=features)
                )
            )
            baseline = posterior.expected_potential(prefix_steps)
            causal_gain = float(
                posterior.expected_potential((*prefix_steps, *candidate_steps)) - baseline
            )
        level_delta = int(representative.get("level_delta", 0))
        terminal = bool(representative.get("terminal"))
        terminal_failure = bool(representative.get("terminal_failure"))
        deterministic = bool(
            availability_deterministic and effect_deterministic and outcome_deterministic
        )
        transport_eligible = set(actions).issubset(transport)
        safe_progress = bool(
            complete and deterministic and level_delta > 0 and not terminal_failure
        )
        safe_nonprogress = bool(
            complete and deterministic and level_delta == 0 and not terminal
        )
        unsafe = bool(deterministic and terminal_failure)
        summaries.append(
            {
                "availability_deterministic": availability_deterministic,
                "causal_gain": causal_gain,
                "effect_deterministic": effect_deterministic,
                "evidence_ids": [str(item.get("trial_id")) for item in records],
                "executable": complete,
                "level_delta": level_delta,
                "magnitude": magnitude,
                "outcome_deterministic": outcome_deterministic,
                "program_actions": list(actions),
                "program_complete": complete,
                "program_id": identifier,
                "program_length": len(actions),
                "repetition_count": len(records),
                "repetitions_exact": repetitions_exact,
                "safe_nonprogress": safe_nonprogress,
                "safe_progress": safe_progress,
                "terminal": terminal,
                "terminal_failure": terminal_failure,
                "terminal_state": str(representative.get("terminal_state", "")),
                "transport_eligible": transport_eligible,
                "unsafe": unsafe,
            }
        )
    summaries.sort(key=_summary_sort_key)
    integrity = {
        "availability_is_deterministic": all(
            item["availability_deterministic"] for item in summaries
        ),
        "context_replay_is_exact": context_replay_exact,
        "effects_are_deterministic": all(
            item["effect_deterministic"] for item in summaries
        ),
        "effects_are_deterministic_when_complete": all(
            item["effect_deterministic"] for item in summaries if item["executable"]
        ),
        "fixed_program_schedule_completed": schedule_complete,
        "outcomes_are_deterministic": all(
            item["outcome_deterministic"] for item in summaries
        ),
        "repetition_count_is_exact": all(
            item["repetitions_exact"] for item in summaries
        ),
    }
    return summaries, integrity


def audit_calibration_trials(
    *,
    trials: Sequence[Mapping[str, Any]],
    expected_programs: Sequence[Sequence[str]],
    repetitions_per_program: int,
    transport_actions: Sequence[str],
    features: Sequence[str],
    posterior: JointCausalProgressPosterior,
    minimum_distractor_magnitude_gap: float,
) -> dict[str, Any]:
    summaries, integrity = _program_summaries(
        trials=trials,
        expected_programs=expected_programs,
        repetitions_per_program=repetitions_per_program,
        transport_actions=transport_actions,
        features=features,
        posterior=posterior,
    )
    progress_candidates = sorted(
        (
            item
            for item in summaries
            if item["transport_eligible"] and item["safe_progress"]
        ),
        key=_summary_sort_key,
    )
    selected_progress = progress_candidates[0] if progress_candidates else None
    distractor_candidates = []
    if selected_progress is not None:
        threshold = float(selected_progress["magnitude"] or 0.0) + float(
            minimum_distractor_magnitude_gap
        )
        distractor_candidates = sorted(
            (
                item
                for item in summaries
                if item["transport_eligible"]
                and item["safe_nonprogress"]
                and float(item["magnitude"] or 0.0) >= threshold
            ),
            key=lambda item: (
                -float(item["magnitude"] or 0.0),
                int(item["program_length"]),
                tuple(item["program_actions"]),
            ),
        )
    selected_distractor = (
        distractor_candidates[0] if distractor_candidates else None
    )
    causal_correct = bool(
        selected_progress is not None
        and selected_distractor is not None
        and float(selected_progress["causal_gain"] or 0.0)
        > float(selected_distractor["causal_gain"] or 0.0)
    )
    selection = {
        "causal_score_used_for_program_selection": False,
        "distractor": selected_distractor,
        "distractor_rule": (
            "largest magnitude safe non-progress transport program with fixed "
            "minimum gap; then shortest and lexicographic"
        ),
        "progress": selected_progress,
        "progress_rule": "shortest safe-progress transport program; lexicographic tie-break",
    }
    metrics = {
        **integrity,
        "candidate_program_count": len(summaries),
        "causal_contrast_correct": causal_correct,
        "complete_program_count": sum(item["executable"] for item in summaries),
        "hard_utility_contrast_count": int(selected_distractor is not None),
        "missing_program_count": sum(not item["executable"] for item in summaries),
        "safe_nonprogress_program_count": sum(
            item["safe_nonprogress"] for item in summaries
        ),
        "safe_progress_program_count": sum(item["safe_progress"] for item in summaries),
        "transport_safe_progress_program_count": len(progress_candidates),
        "unsafe_program_count": sum(item["unsafe"] for item in summaries),
    }
    return {
        "metrics": metrics,
        "program_registry": {
            "format_version": "sage-t12.5b.4-local-program-registry-v1",
            "programs": summaries,
            "selection": selection,
        },
        "selection": selection,
    }


def evaluation_registry_payload(
    *,
    manifest_checksum: str,
    protocol_checksum: str,
    calibration_evidence_checksum: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    progress = selection.get("progress")
    distractor = selection.get("distractor")
    if not isinstance(progress, Mapping) or not isinstance(distractor, Mapping):
        raise ValueError("T12.5b.4 calibration has no evaluable contrast")
    payload = {
        "format_version": "sage-t12.5b.4-evaluation-registry-v1",
        "manifest_checksum": str(manifest_checksum),
        "protocol_checksum": str(protocol_checksum),
        "calibration_evidence_checksum": str(calibration_evidence_checksum),
        "selection_rule_uses_causal_score": False,
        "programs": {
            "progress": {
                "program_actions": list(progress["program_actions"]),
                "program_id": str(progress["program_id"]),
                "calibration_causal_gain": float(progress["causal_gain"]),
                "calibration_magnitude": float(progress["magnitude"]),
            },
            "distractor": {
                "program_actions": list(distractor["program_actions"]),
                "program_id": str(distractor["program_id"]),
                "calibration_causal_gain": float(distractor["causal_gain"]),
                "calibration_magnitude": float(distractor["magnitude"]),
            },
        },
    }
    return {**payload, "registry_checksum": _checksum(payload)}


def audit_evaluation_trials(
    *,
    trials: Sequence[Mapping[str, Any]],
    evaluation_registry: Mapping[str, Any],
    repetitions_per_program: int,
    transport_actions: Sequence[str],
    features: Sequence[str],
    posterior: JointCausalProgressPosterior,
) -> dict[str, Any]:
    registered = dict(evaluation_registry["programs"])
    expected = (
        tuple(registered["progress"]["program_actions"]),
        tuple(registered["distractor"]["program_actions"]),
    )
    summaries, integrity = _program_summaries(
        trials=trials,
        expected_programs=expected,
        repetitions_per_program=repetitions_per_program,
        transport_actions=transport_actions,
        features=features,
        posterior=posterior,
    )
    by_id = {item["program_id"]: item for item in summaries}
    progress = by_id[str(registered["progress"]["program_id"])]
    distractor = by_id[str(registered["distractor"]["program_id"])]
    progress_transferred = bool(progress["safe_progress"])
    distractor_stable = bool(distractor["safe_nonprogress"])
    causal_transferred = bool(
        progress["causal_gain"] is not None
        and distractor["causal_gain"] is not None
        and float(progress["causal_gain"]) > float(distractor["causal_gain"])
    )
    return {
        "metrics": {
            **integrity,
            "causal_utility_transferred": causal_transferred,
            "distractor_stable_safe_nonprogress": distractor_stable,
            "evaluation_program_count": len(summaries),
            "progress_program_transferred": progress_transferred,
        },
        "program_registry": {
            "format_version": "sage-t12.5b.4-evaluation-program-registry-v1",
            "programs": summaries,
        },
        "registered_distractor": distractor,
        "registered_progress": progress,
    }


__all__ = [
    "audit_calibration_trials",
    "audit_evaluation_trials",
    "evaluation_registry_payload",
    "program_id",
]
