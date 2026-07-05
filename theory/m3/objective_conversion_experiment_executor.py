"""M3.G1.2 executor for stop-state objective-conversion experiments.

Reaches the central ``safe_stop_state`` by strictly replaying the P3.G1
terminal-safe-but-passive policy (``objective_aware_abstract_policy_lambda_0``),
then applies each candidate action/short-sequence vs the two controls
(``hold_or_stop_state`` zero-action, and horizon-matched
``relation_progress_policy``).

Discipline:
- This executor produces RAW MEASUREMENTS ONLY. It does NOT interpret,
  aggregate into outcome statuses, confirm, refute, or revise. Interpretation
  (candidate-only outcome statuses) is M3.G1.3's job.
- Every artifact keeps ``support=0``, ``revision_status=CANDIDATE_ONLY``,
  ``truth_status=NOT_EVALUATED_BY_M3``, and never writes A32/A33.

Determinism / testability:
- ``safe_stop_capturer`` and ``cell_executor`` are injectable seams so tests
  run deterministically without the live offline environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_conversion_experiment_planner import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    HOLD_OR_STOP_STATE_CONTROL,
    READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT,
    RELATION_PROGRESS_POLICY_CONTROL,
    SAFE_STOP_POLICY_CONDITION,
)


DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "objective_conversion_experiment_results.json"
)
RESULTS_SCHEMA_VERSION = "m3.objective_conversion_experiment_results.v1"

OBJECTIVE_CONVERSION_CELL_EXECUTED = "OBJECTIVE_CONVERSION_CELL_EXECUTED"
HOLD_OR_STOP_STATE_OBSERVED = "HOLD_OR_STOP_STATE_OBSERVED"

# Dedicated block statuses (no ambiguous result on replay/availability failure).
SAFE_STOP_REPLAY_MISMATCH_BLOCKED = "SAFE_STOP_REPLAY_MISMATCH_BLOCKED"
SAFE_STOP_NOT_REACHED_BLOCKED = "SAFE_STOP_NOT_REACHED_BLOCKED"
CANDIDATE_ACTION_UNAVAILABLE_BLOCKED = "CANDIDATE_ACTION_UNAVAILABLE_BLOCKED"
TERMINAL_DURING_SAFE_STOP_PREFIX_BLOCKED = "TERMINAL_DURING_SAFE_STOP_PREFIX_BLOCKED"
BLOCKED_STATUSES = (
    SAFE_STOP_REPLAY_MISMATCH_BLOCKED,
    SAFE_STOP_NOT_REACHED_BLOCKED,
    CANDIDATE_ACTION_UNAVAILABLE_BLOCKED,
    TERMINAL_DURING_SAFE_STOP_PREFIX_BLOCKED,
)

LOW_SINGLE_SAFE_STOP = "LOW_SINGLE_SAFE_STOP"

DEFAULT_SAFE_STOP_CAPTURE_CONFIG = {
    "selection_rule": "best_p3g1_condition_from_utility_consolidation",
    "fallback_condition": SAFE_STOP_POLICY_CONDITION,
    "fallback_budget": 64,
    "fallback_tie_break_seed": 0,
    "selection_counted_as_support": False,
}


@dataclass(frozen=True)
class CapturedSafeStop:
    """The single safe-stop endpoint produced by replaying the P3.G1 policy."""

    prefix: Tuple[Dict[str, Any], ...]
    captured_prefix_hash: str
    safe_stop_state_hash: str
    hold_baseline_terminal_adjusted_progress: float
    hold_baseline_levels_completed: int
    hold_baseline_terminal: bool
    provenance: Dict[str, Any]
    capture_config: Dict[str, Any]
    adapter: Dict[str, Any] = field(default_factory=dict)
    prefix_step_dicts: Tuple[Dict[str, Any], ...] = ()

    def base_step_dicts(self) -> List[Dict[str, Any]]:
        """Real safe-stop prefix step dicts used as the shared metric base.

        Candidate and relation-progress taps are both summarized over this same
        base + their post-stop steps, so ``delta_vs_hold`` isolates the true
        marginal effect of acting (and terminal re-entry zeroes the whole taps).
        """
        if self.prefix_step_dicts:
            return [dict(step) for step in self.prefix_step_dicts]
        return _build_prefix_step_dicts(self)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "captured_prefix": [dict(step) for step in self.prefix],
            "captured_prefix_len": len(self.prefix),
            "captured_prefix_hash": self.captured_prefix_hash,
            "safe_stop_state_hash": self.safe_stop_state_hash,
            "hold_baseline_terminal_adjusted_progress": float(
                self.hold_baseline_terminal_adjusted_progress
            ),
            "hold_baseline_levels_completed": int(
                self.hold_baseline_levels_completed
            ),
            "hold_baseline_terminal": bool(self.hold_baseline_terminal),
            "provenance": dict(self.provenance),
            "safe_stop_capture_config": dict(self.capture_config),
            "safe_stop_context_diversity": LOW_SINGLE_SAFE_STOP,
        }


@dataclass(frozen=True)
class ObjectiveConversionCell:
    """One unique objective-conversion execution cell, shared by ≥1 hypotheses."""

    cell_signature: str
    game_id: str
    condition_kind: str  # "candidate" | "hold" | "relation_progress_policy"
    condition_id: str
    action_or_sequence: Tuple[str, ...] | None
    post_stop_horizon: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_signature": self.cell_signature,
            "game_id": self.game_id,
            "condition_kind": self.condition_kind,
            "condition_id": self.condition_id,
            "action_or_sequence": (
                None
                if self.action_or_sequence is None
                else list(self.action_or_sequence)
            ),
            "post_stop_horizon": int(self.post_stop_horizon),
        }


def run_objective_conversion_experiment_execution(
    *,
    requests_path: str | Path = (
        DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    tie_break_seed: int | None = None,
    max_cells: int | None = None,
    safe_stop_capturer: Callable[[Sequence[Mapping[str, Any]]], CapturedSafeStop]
    | None = None,
    cell_executor: Callable[
        [ObjectiveConversionCell, CapturedSafeStop], Mapping[str, Any]
    ]
    | None = None,
) -> Dict[str, Any]:
    request_payload = _load_json(requests_path)
    _validate_source_request_payload(request_payload)
    requests = ready_objective_conversion_requests(request_payload)

    if safe_stop_capturer is None:
        safe_stop_capturer = _make_default_safe_stop_capturer(
            environments_dir=environments_dir,
            tie_break_seed=tie_break_seed,
        )
    captured = safe_stop_capturer(requests)

    planned_cells, links = build_execution_cells(requests)
    unique_cells = unique_execution_cells(planned_cells)
    if max_cells is not None:
        unique_cells = unique_cells[: max(0, int(max_cells))]
    executed_signatures = {cell.cell_signature for cell in unique_cells}
    executable_links = [
        link
        for link in links
        if str(link.get("cell_signature", "")) in executed_signatures
    ]

    if cell_executor is None:
        cell_executor = _make_default_cell_executor(
            environments_dir=environments_dir,
        )

    cell_results = [dict(cell_executor(cell, captured)) for cell in unique_cells]
    result_by_signature = {
        str(row.get("cell_signature", "")): dict(row) for row in cell_results
    }
    linked_measurements = [
        build_hypothesis_measurement_link(
            link,
            result_by_signature.get(str(link.get("cell_signature", "")), {}),
        )
        for link in executable_links
    ]
    return build_results_payload(
        requests_path=requests_path,
        environments_dir=environments_dir,
        captured=captured,
        requests=requests,
        planned_cells=planned_cells,
        unique_cells=unique_cells,
        cell_results=cell_results,
        hypothesis_measurement_links=linked_measurements,
    )


# --------------------------------------------------------------------------- #
# Cell construction + dedup + hypothesis linking
# --------------------------------------------------------------------------- #


def build_execution_cells(
    requests: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[ObjectiveConversionCell, ...], Tuple[Dict[str, Any], ...]]:
    cells: List[ObjectiveConversionCell] = []
    links: List[Dict[str, Any]] = []
    for request in requests:
        game_id = str(request.get("game_id", ""))
        source_hypothesis_id = str(request.get("source_hypothesis_id", ""))
        family = str(request.get("hypothesis_family", ""))
        request_cells: List[ObjectiveConversionCell] = []
        for condition in request.get("candidate_conditions", []) or []:
            request_cells.append(_candidate_cell(game_id, condition))
        for control in request.get("control_conditions", []) or []:
            request_cells.append(_control_cell(game_id, control))
        for cell in request_cells:
            cells.append(cell)
            links.append(
                {
                    "request_id": str(request.get("request_id", "")),
                    "source_hypothesis_id": source_hypothesis_id,
                    "hypothesis_family": family,
                    "cell_signature": cell.cell_signature,
                    "condition_kind": cell.condition_kind,
                    "condition_id": cell.condition_id,
                    "post_stop_horizon": int(cell.post_stop_horizon),
                    "duplicate_execution_cell_counted_as_independent": False,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                    "wrong_confirmations": 0,
                }
            )
    return tuple(cells), tuple(links)


def _candidate_cell(
    game_id: str, condition: Mapping[str, Any]
) -> ObjectiveConversionCell:
    actions = tuple(str(a) for a in condition.get("action_or_sequence", []) or [])
    horizon = int(condition.get("post_stop_horizon", len(actions)) or len(actions))
    return ObjectiveConversionCell(
        cell_signature=execution_cell_signature(
            game_id=game_id,
            condition_kind="candidate",
            action_or_sequence=list(actions),
            post_stop_horizon=horizon,
        ),
        game_id=game_id,
        condition_kind="candidate",
        condition_id=str(condition.get("condition_id", "")),
        action_or_sequence=actions,
        post_stop_horizon=horizon,
    )


def _control_cell(
    game_id: str, control: Mapping[str, Any]
) -> ObjectiveConversionCell:
    family = str(control.get("condition_family", ""))
    horizon = int(control.get("post_stop_horizon", 0) or 0)
    if family == HOLD_OR_STOP_STATE_CONTROL:
        return ObjectiveConversionCell(
            cell_signature=execution_cell_signature(
                game_id=game_id,
                condition_kind="hold",
                action_or_sequence=None,
                post_stop_horizon=0,
            ),
            game_id=game_id,
            condition_kind="hold",
            condition_id=HOLD_OR_STOP_STATE_CONTROL,
            action_or_sequence=None,
            post_stop_horizon=0,
        )
    return ObjectiveConversionCell(
        cell_signature=execution_cell_signature(
            game_id=game_id,
            condition_kind="relation_progress_policy",
            action_or_sequence=None,
            post_stop_horizon=horizon,
        ),
        game_id=game_id,
        condition_kind="relation_progress_policy",
        condition_id=str(control.get("condition_id", "")),
        action_or_sequence=None,
        post_stop_horizon=horizon,
    )


def unique_execution_cells(
    planned_cells: Sequence[ObjectiveConversionCell],
) -> Tuple[ObjectiveConversionCell, ...]:
    by_signature: Dict[str, ObjectiveConversionCell] = {}
    for cell in planned_cells:
        by_signature.setdefault(cell.cell_signature, cell)
    return tuple(by_signature[key] for key in by_signature)


def execution_cell_signature(
    *,
    game_id: str,
    condition_kind: str,
    action_or_sequence: Sequence[str] | None,
    post_stop_horizon: int,
) -> str:
    raw = {
        "action_or_sequence": (
            None if action_or_sequence is None else list(action_or_sequence)
        ),
        "condition_kind": condition_kind,
        "game_id": game_id,
        "post_stop_horizon": int(post_stop_horizon),
    }
    return "m3_g1::" + json.dumps(raw, sort_keys=True, separators=(",", ":"))


def build_hypothesis_measurement_link(
    link: Mapping[str, Any],
    cell_result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(link),
        "cell_status": str(cell_result.get("status", "")),
        "execution_performed": bool(cell_result.get("execution_performed", False)),
        "safe_stop_replay_exact": bool(
            cell_result.get("safe_stop_replay_exact", False)
        ),
        "measurements": dict(cell_result.get("measurements", {}) or {}),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "cell_link_counted_as_independent_execution": False,
        "cell_link_counted_as_confirmation": False,
    }


# --------------------------------------------------------------------------- #
# Result payload (measurements only)
# --------------------------------------------------------------------------- #


def build_results_payload(
    *,
    requests_path: str | Path,
    environments_dir: str | Path | None,
    captured: CapturedSafeStop,
    requests: Sequence[Mapping[str, Any]],
    planned_cells: Sequence[ObjectiveConversionCell],
    unique_cells: Sequence[ObjectiveConversionCell],
    cell_results: Sequence[Mapping[str, Any]],
    hypothesis_measurement_links: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = summarize_execution(
        requests=requests,
        planned_cells=planned_cells,
        unique_cells=unique_cells,
        cell_results=cell_results,
        links=hypothesis_measurement_links,
        captured=captured,
    )
    return {
        "config": {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "requests_path": str(requests_path),
            "environments_dir": (
                None if environments_dir is None else str(environments_dir)
            ),
            "inputs_read": ["M3.G1.1", "P3.G1"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "central_experimental_unit": (
                "safe_stop_state -> candidate_action_or_sequence"
            ),
            "stage_produces": "raw_measurements_only_no_interpretation",
            "interpretation_deferred_to": "M3.G1.3",
        },
        "safe_stop_capture": captured.to_public_dict(),
        "summary": summary,
        "execution_cells": [dict(row) for row in cell_results],
        "hypothesis_measurement_links": [
            dict(row) for row in hypothesis_measurement_links
        ],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "safe_stop_context_diversity": LOW_SINGLE_SAFE_STOP,
        "duplicate_execution_cells_counted_as_independent": False,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def summarize_execution(
    *,
    requests: Sequence[Mapping[str, Any]],
    planned_cells: Sequence[ObjectiveConversionCell],
    unique_cells: Sequence[ObjectiveConversionCell],
    cell_results: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
    captured: CapturedSafeStop,
) -> Dict[str, Any]:
    executed = [row for row in cell_results if bool(row.get("execution_performed"))]
    blocked = [row for row in cell_results if not bool(row.get("execution_performed"))]
    block_counts = {
        status: len([row for row in blocked if row.get("status") == status])
        for status in BLOCKED_STATUSES
    }
    return {
        "objective_conversion_requests_consumed": len(requests),
        "planned_cells": len(planned_cells),
        "unique_execution_cells": len(unique_cells),
        "duplicate_execution_cells": max(
            0, len(planned_cells) - len(unique_cells)
        ),
        "duplicate_execution_cells_counted_as_independent": False,
        "hypothesis_measurement_links": len(links),
        "cells_executed": len(executed),
        "cells_blocked": len(blocked),
        "block_status_counts": block_counts,
        "candidate_cells_executed": len(
            [r for r in executed if r.get("condition_kind") == "candidate"]
        ),
        "hold_cells_executed": len(
            [r for r in executed if r.get("condition_kind") == "hold"]
        ),
        "relation_progress_cells_executed": len(
            [
                r
                for r in executed
                if r.get("condition_kind") == "relation_progress_policy"
            ]
        ),
        "safe_stop_replay_exact_cells": len(
            [r for r in cell_results if bool(r.get("safe_stop_replay_exact"))]
        ),
        "safe_stop_context_diversity": LOW_SINGLE_SAFE_STOP,
        "safe_stop_policy_condition": captured.provenance.get(
            "safe_stop_policy_condition", SAFE_STOP_POLICY_CONDITION
        ),
        "produces_aggregated_outcome_enum": False,
        "execution_performed": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


# --------------------------------------------------------------------------- #
# Cell result builders (used by default + injected executors)
# --------------------------------------------------------------------------- #


def measured_cell_result(
    cell: ObjectiveConversionCell,
    *,
    captured: CapturedSafeStop,
    candidate_terminal_adjusted_progress: float,
    candidate_levels_completed: int,
    candidate_terminal_reentry: bool,
    objective_completion_signal: bool,
    diagnostics: Mapping[str, Any],
    replayed_prefix_hash: str,
    safe_stop_state_hash: str,
    post_stop_steps_executed: int,
) -> Dict[str, Any]:
    hold = float(captured.hold_baseline_terminal_adjusted_progress)
    delta = float(candidate_terminal_adjusted_progress) - hold
    replay_exact = bool(
        replayed_prefix_hash == captured.captured_prefix_hash
        and safe_stop_state_hash == captured.safe_stop_state_hash
    )
    status = (
        HOLD_OR_STOP_STATE_OBSERVED
        if cell.condition_kind == "hold"
        else OBJECTIVE_CONVERSION_CELL_EXECUTED
    )
    return {
        **cell.to_dict(),
        "status": status,
        "execution_performed": True,
        "captured_prefix_hash": captured.captured_prefix_hash,
        "replayed_prefix_hash": replayed_prefix_hash,
        "safe_stop_state_hash": safe_stop_state_hash,
        "safe_stop_replay_exact": replay_exact,
        "post_stop_steps_executed": int(post_stop_steps_executed),
        "blocked_reason": None,
        "measurements": {
            "candidate_terminal_adjusted_progress_after_stop": float(
                candidate_terminal_adjusted_progress
            ),
            "hold_or_stop_state_terminal_adjusted_progress_after_stop": hold,
            "delta_terminal_adjusted_progress_vs_hold": round(delta, 6),
            "candidate_terminal_reentry": bool(candidate_terminal_reentry),
            "candidate_levels_completed_after_rollout": int(
                candidate_levels_completed
            ),
            "objective_completion_signal": bool(objective_completion_signal),
            **{str(k): v for k, v in dict(diagnostics).items()},
        },
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "cell_result_counted_as_confirmation": False,
    }


def blocked_cell_result(
    cell: ObjectiveConversionCell,
    *,
    captured: CapturedSafeStop,
    status: str,
    reason: str,
    replayed_prefix_hash: str | None = None,
    safe_stop_state_hash: str | None = None,
    safe_stop_replay_exact: bool = False,
) -> Dict[str, Any]:
    return {
        **cell.to_dict(),
        "status": status,
        "execution_performed": False,
        "captured_prefix_hash": captured.captured_prefix_hash,
        "replayed_prefix_hash": replayed_prefix_hash,
        "safe_stop_state_hash": safe_stop_state_hash,
        "safe_stop_replay_exact": bool(safe_stop_replay_exact),
        "post_stop_steps_executed": 0,
        "blocked_reason": reason,
        "measurements": {},
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "blocked_cell_counted_as_contradiction": False,
        "cell_result_counted_as_confirmation": False,
    }


# --------------------------------------------------------------------------- #
# Default (live offline env) seams - imported lazily to keep module light
# --------------------------------------------------------------------------- #


def _make_default_safe_stop_capturer(
    *,
    environments_dir: str | Path | None,
    tie_break_seed: int | None,
) -> Callable[[Sequence[Mapping[str, Any]]], CapturedSafeStop]:
    def capturer(requests: Sequence[Mapping[str, Any]]) -> CapturedSafeStop:
        return capture_safe_stop_prefix(
            requests,
            environments_dir=environments_dir,
            tie_break_seed=tie_break_seed,
        )

    return capturer


def _make_default_cell_executor(
    *,
    environments_dir: str | Path | None,
) -> Callable[[ObjectiveConversionCell, CapturedSafeStop], Mapping[str, Any]]:
    def executor(
        cell: ObjectiveConversionCell, captured: CapturedSafeStop
    ) -> Mapping[str, Any]:
        return execute_objective_conversion_cell(
            cell,
            captured=captured,
            environments_dir=environments_dir,
        )

    return executor


def capture_safe_stop_prefix(
    requests: Sequence[Mapping[str, Any]],
    *,
    environments_dir: str | Path | None,
    tie_break_seed: int | None,
    utility_consolidation_path: str | Path | None = None,
    objective_adapter_path: str | Path | None = None,
) -> CapturedSafeStop:
    """Replay the best P3.G1 terminal-safe-but-passive policy once.

    Selects the best condition from the P3.G1 utility consolidation, falling
    back to ``objective_aware_abstract_policy_lambda_0`` / budget 64 / seed 0.
    """
    from theory.p3.objective_aware_abstract_policy_probe import (  # noqa: WPS433
        DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH,
        DEFAULT_OBJECTIVE_AWARE_CONSOLIDATION_OUTPUT_PATH,
        execute_objective_aware_condition,
        summarize_objective_aware_steps,
    )
    from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

    adapter_path = (
        Path(objective_adapter_path)
        if objective_adapter_path is not None
        else DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH
    )
    consolidation_path = (
        Path(utility_consolidation_path)
        if utility_consolidation_path is not None
        else DEFAULT_OBJECTIVE_AWARE_CONSOLIDATION_OUTPUT_PATH
    )
    capture_config = dict(DEFAULT_SAFE_STOP_CAPTURE_CONFIG)
    condition = capture_config["fallback_condition"]
    budget = int(capture_config["fallback_budget"])
    seed = int(
        tie_break_seed
        if tie_break_seed is not None
        else capture_config["fallback_tie_break_seed"]
    )
    selection_source = "fallback"
    try:
        consolidation = _load_json(consolidation_path)
        best = str(
            (consolidation.get("summary", {}) or {}).get(
                "best_objective_aware_condition", ""
            )
        )
        if best.startswith("objective_aware_abstract_policy_lambda_"):
            condition = best
            selection_source = "best_p3g1_condition_from_utility_consolidation"
    except (FileNotFoundError, ValueError):
        selection_source = "fallback"

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    adapter_payload = _load_json(adapter_path)
    adapter = dict(adapter_payload.get("objective_aware_policy_adapter", {}) or {})
    game_id = _game_id_from_requests(requests)

    steps, stop_event = execute_objective_aware_condition(
        condition=condition,
        adapter=adapter,
        budget=budget,
        tie_break_seed=seed,
        environments_dir=env_dir,
        game_id=game_id,
    )
    prefix = tuple(
        {
            "action": str(step.get("policy_selected_action", "")),
            "action_args": dict(step.get("action_args", {}) or {}),
        }
        for step in steps
    )
    baseline = summarize_objective_aware_steps(
        condition=condition,
        steps=steps,
        budget=budget,
        tie_break_seed=seed,
        stop_event=stop_event,
    )
    safe_stop_state_hash = (
        str(steps[-1].get("state_signature_after", ""))
        if steps
        else "safe_stop::initial"
    )
    provenance = {
        "safe_stop_state_source": "P3.G1",
        "safe_stop_policy_condition": condition,
        "safe_stop_policy_selection_source": selection_source,
        "stop_trigger_reason": str(
            stop_event.get("trigger_reason")
            or "objective_aware_terminal_risk_score_stop"
        ),
        "terminal_horizon_source": str(stop_event.get("terminal_horizon_source", "")),
        "base_state_family": "terminal_safe_stop_or_avoidance_state",
        "budget": budget,
        "tie_break_seed": seed,
    }
    return CapturedSafeStop(
        prefix=prefix,
        captured_prefix_hash=_prefix_hash(prefix),
        safe_stop_state_hash=safe_stop_state_hash,
        hold_baseline_terminal_adjusted_progress=float(
            baseline.get("terminal_adjusted_progress", 0.0) or 0.0
        ),
        hold_baseline_levels_completed=int(
            baseline.get("final_levels_completed", 0) or 0
        ),
        hold_baseline_terminal=bool(
            baseline.get("terminal_state_after_rollout", False)
        ),
        provenance=provenance,
        capture_config=capture_config,
        adapter=adapter,
        prefix_step_dicts=tuple(dict(step) for step in steps),
    )


def execute_objective_conversion_cell(
    cell: ObjectiveConversionCell,
    *,
    captured: CapturedSafeStop,
    environments_dir: str | Path | None,
) -> Dict[str, Any]:
    """Strict-replay the captured safe-stop prefix, then apply the condition."""
    from theory.m2.m3_execution_smoke import _make_env, _reset_env  # noqa: WPS433
    from theory.m1.polymorphic_a25_adapter import _step_env_action  # noqa: WPS433
    from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
    from theory.p3.abstract_mechanic_policy_probe import (
        concrete_named_action,
        is_game_over,
    )
    from theory.p1.bp35_sage_candidate_policy_probe import state_signature
    from theory.real_env_option_adapter import snapshot_frame

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    env = _make_env(cell.game_id, env_dir)
    current = _reset_env(env)

    # Strict replay of captured safe-stop prefix.
    replayed_signatures: List[str] = []
    before = snapshot_frame(current)
    for step in captured.prefix:
        if is_game_over(before.game_state):
            return blocked_cell_result(
                cell,
                captured=captured,
                status=TERMINAL_DURING_SAFE_STOP_PREFIX_BLOCKED,
                reason="terminal_reached_during_safe_stop_prefix_replay",
            )
        selected = concrete_named_action(
            list(_valid_actions(env)),
            str(step.get("action", "")),
            dict(step.get("action_args", {}) or {}),
        )
        if selected is None:
            return blocked_cell_result(
                cell,
                captured=captured,
                status=SAFE_STOP_REPLAY_MISMATCH_BLOCKED,
                reason="safe_stop_prefix_action_unavailable_during_replay",
            )
        after_frame = _step_env_action(env, selected)
        if after_frame is None:
            return blocked_cell_result(
                cell,
                captured=captured,
                status=SAFE_STOP_REPLAY_MISMATCH_BLOCKED,
                reason="env_step_returned_no_frame_during_replay",
            )
        current = after_frame
        after = _safe_snapshot(current, before)
        replayed_signatures.append(
            state_signature(after.grid, after.levels_completed, after.game_state)
        )
        before = after

    safe_stop_frame = before
    safe_stop_state_hash = (
        replayed_signatures[-1]
        if replayed_signatures
        else "safe_stop::initial"
    )
    replayed_prefix_hash = captured.captured_prefix_hash  # replay mirrors capture
    if safe_stop_state_hash != captured.safe_stop_state_hash:
        return blocked_cell_result(
            cell,
            captured=captured,
            status=SAFE_STOP_REPLAY_MISMATCH_BLOCKED,
            reason="safe_stop_state_hash_mismatch_after_replay",
            replayed_prefix_hash=replayed_prefix_hash,
            safe_stop_state_hash=safe_stop_state_hash,
        )
    if is_game_over(safe_stop_frame.game_state):
        return blocked_cell_result(
            cell,
            captured=captured,
            status=SAFE_STOP_NOT_REACHED_BLOCKED,
            reason="safe_stop_endpoint_terminal",
            replayed_prefix_hash=replayed_prefix_hash,
            safe_stop_state_hash=safe_stop_state_hash,
        )

    if cell.condition_kind == "hold":
        return measured_cell_result(
            cell,
            captured=captured,
            candidate_terminal_adjusted_progress=(
                captured.hold_baseline_terminal_adjusted_progress
            ),
            candidate_levels_completed=captured.hold_baseline_levels_completed,
            candidate_terminal_reentry=False,
            objective_completion_signal=(
                captured.hold_baseline_levels_completed > 0
            ),
            diagnostics=_empty_diagnostics(),
            replayed_prefix_hash=replayed_prefix_hash,
            safe_stop_state_hash=safe_stop_state_hash,
            post_stop_steps_executed=0,
        )

    # Determine the post-stop action plan.
    if cell.condition_kind == "candidate":
        action_plan = list(cell.action_or_sequence or ())
        result = _roll_post_stop_actions(
            env=env,
            current=current,
            captured=captured,
            action_plan=action_plan,
        )
    else:  # relation_progress_policy
        result = _roll_relation_progress(
            env=env,
            current=current,
            captured=captured,
            horizon=int(cell.post_stop_horizon),
        )

    if result.get("blocked_status"):
        return blocked_cell_result(
            cell,
            captured=captured,
            status=str(result["blocked_status"]),
            reason=str(result.get("blocked_reason", "")),
            replayed_prefix_hash=replayed_prefix_hash,
            safe_stop_state_hash=safe_stop_state_hash,
            safe_stop_replay_exact=True,
        )

    return measured_cell_result(
        cell,
        captured=captured,
        candidate_terminal_adjusted_progress=float(result["terminal_adjusted_progress"]),
        candidate_levels_completed=int(result["levels_completed"]),
        candidate_terminal_reentry=bool(result["terminal_reentry"]),
        objective_completion_signal=bool(result["objective_completion_signal"]),
        diagnostics=result["diagnostics"],
        replayed_prefix_hash=replayed_prefix_hash,
        safe_stop_state_hash=safe_stop_state_hash,
        post_stop_steps_executed=int(result["post_stop_steps_executed"]),
    )


def _roll_post_stop_actions(
    *,
    env: Any,
    current: Any,
    captured: CapturedSafeStop,
    action_plan: Sequence[str],
) -> Dict[str, Any]:
    from theory.non_ar25_active_micro_run import _valid_actions
    from theory.p3.abstract_mechanic_policy_probe import concrete_named_action

    post_steps = captured.base_step_dicts()
    frame = current
    executed = 0
    for action_name in action_plan:
        selected = concrete_named_action(
            list(_valid_actions(env)), str(action_name), {}
        )
        if selected is None:
            return {
                "blocked_status": CANDIDATE_ACTION_UNAVAILABLE_BLOCKED,
                "blocked_reason": f"candidate_action_unavailable::{action_name}",
            }
        after_frame, step = _step_and_build(env, frame, selected, captured)
        if after_frame is None:
            return {
                "blocked_status": CANDIDATE_ACTION_UNAVAILABLE_BLOCKED,
                "blocked_reason": f"env_step_returned_no_frame::{action_name}",
            }
        frame = after_frame
        post_steps.append(step)
        executed += 1
        if bool(step.get("terminal_state_after")):
            break
    return _summarize_post_stop(captured, post_steps, executed)


def _roll_relation_progress(
    *,
    env: Any,
    current: Any,
    captured: CapturedSafeStop,
    horizon: int,
) -> Dict[str, Any]:
    from collections import Counter

    from theory.non_ar25_active_micro_run import _valid_actions
    from theory.p3.abstract_mechanic_policy_probe import (
        STOP_ACTION,
        concrete_action_for_decision,
        is_game_over,
        select_abstract_model_decision,
    )
    from theory.p3.terminal_horizon_estimator import estimate_terminal_horizon
    from theory.real_env_option_adapter import snapshot_frame

    post_steps = captured.base_step_dicts()
    frame = current
    executed = 0
    action_counts: Counter = Counter()
    for _ in range(max(0, int(horizon))):
        before = snapshot_frame(frame)
        if is_game_over(before.game_state):
            break
        horizon_estimate = estimate_terminal_horizon(
            observation=before.grid,
            history=[],
            policy_state={"env_actions_executed": len(post_steps)},
            terminal_budget_estimate=64,
        ).to_dict()
        decision = select_abstract_model_decision(
            adapter=captured.adapter,
            valid_actions=list(_valid_actions(env)),
            action_counts=action_counts,
            tie_break_seed=0,
            horizon_estimate=horizon_estimate,
        )
        if str(decision.action_name) == STOP_ACTION:
            break
        selected = concrete_action_for_decision(list(_valid_actions(env)), decision)
        if selected is None:
            break
        after_frame, step = _step_and_build(env, frame, selected, captured)
        if after_frame is None:
            break
        frame = after_frame
        post_steps.append(step)
        action_counts[str(decision.action_name)] += 1
        executed += 1
        if bool(step.get("terminal_state_after")):
            break
    return _summarize_post_stop(captured, post_steps, executed)


def _safe_snapshot(frame: Any, before: Any) -> Any:
    """Snapshot a frame, tolerating empty/terminal frames (game over).

    The offline env can return an empty grid on a terminal transition; rather
    than crash, we treat it as a terminal endpoint carrying the prior grid.
    """
    from theory.real_env_option_adapter import (
        EnvFrameSnapshot,
        _safe_int,
        _state_name,
        snapshot_frame,
    )

    try:
        return snapshot_frame(
            frame, fallback_available_actions=getattr(before, "available_actions", [])
        )
    except (ValueError, TypeError):
        state = _state_name(getattr(frame, "state", "GAME_OVER"))
        if state in ("NOT_FINISHED", "RUNNING"):
            state = "GAME_OVER"
        return EnvFrameSnapshot(
            grid=before.grid,
            available_actions=list(getattr(before, "available_actions", []) or []),
            game_state=state,
            levels_completed=_safe_int(
                getattr(frame, "levels_completed", before.levels_completed)
            ),
        )


def _step_and_build(
    env: Any,
    frame: Any,
    selected: Any,
    captured: CapturedSafeStop,
) -> Tuple[Any, Dict[str, Any]]:
    from theory.m1.polymorphic_a25_adapter import _step_env_action
    from theory.p1.bp35_sage_candidate_policy_probe import (
        measure_probe_metrics,
        state_signature,
    )
    from theory.p3.abstract_mechanic_policy_probe import (
        action_has_actor_effect,
        action_has_relation_effect,
        is_game_over,
    )
    from theory.real_env_option_adapter import snapshot_frame

    before = snapshot_frame(frame)
    after_frame = _step_env_action(env, selected)
    if after_frame is None:
        return None, {}
    after = _safe_snapshot(after_frame, before)
    action_name = str(getattr(selected, "name", ""))
    action_args = dict(getattr(selected, "action_args", {}) or {})
    measurements = measure_probe_metrics(before.grid, after.grid, action_args)
    after_signature = state_signature(
        after.grid, after.levels_completed, after.game_state
    )
    changed_pixels = float(
        measurements["changed_pixels"].get("changed_pixels", 0) or 0
    )
    relation_expected = action_has_relation_effect(captured.adapter, action_name)
    actor_effect_expected = action_has_actor_effect(captured.adapter, action_name)
    useful_new_state = bool(
        changed_pixels > 0
        and after.levels_completed >= before.levels_completed
        and not is_game_over(after.game_state)
    )
    step = {
        "policy_selected_action": action_name,
        "action_args": action_args,
        "changed_pixels": changed_pixels,
        "actor_relation_delta_count": int(relation_expected and changed_pixels > 0),
        "action_effect_usefulness": int(actor_effect_expected and changed_pixels > 0),
        "new_relation_state": int(relation_expected and useful_new_state),
        "useful_new_state": useful_new_state,
        "dead_end_or_cycle": False,
        "levels_after": int(after.levels_completed),
        "game_state_after": str(after.game_state),
        "terminal_state_after": is_game_over(after.game_state),
        "state_signature_after": after_signature,
        "measurements": measurements,
    }
    return after_frame, step


def _summarize_post_stop(
    captured: CapturedSafeStop,
    post_steps: Sequence[Mapping[str, Any]],
    executed: int,
) -> Dict[str, Any]:
    from theory.p3.objective_aware_abstract_policy_probe import (
        summarize_objective_aware_steps,
    )

    summary = summarize_objective_aware_steps(
        condition="m3_g1_post_stop",
        steps=list(post_steps),
        budget=len(post_steps),
        tie_break_seed=0,
        stop_event={},
    )
    terminal = bool(summary.get("terminal_state_after_rollout", False))
    levels = int(summary.get("final_levels_completed", 0) or 0)
    return {
        "terminal_adjusted_progress": float(
            summary.get("terminal_adjusted_progress", 0.0) or 0.0
        ),
        "levels_completed": levels,
        "terminal_reentry": terminal,
        "objective_completion_signal": (
            levels > captured.hold_baseline_levels_completed
        ),
        "post_stop_steps_executed": int(executed),
        "diagnostics": {
            "relation_delta_after_stop": int(
                summary.get("actor_relation_delta_count", 0) or 0
            ),
            "changed_pixels": float(
                sum(float(s.get("changed_pixels", 0) or 0) for s in post_steps[
                    len(captured.prefix):
                ])
            ),
            "new_relation_states": int(summary.get("new_relation_states", 0) or 0),
            "distance_decreases_count": int(
                summary.get("distance_decreases_count", 0) or 0
            ),
            "objective_readiness_signature_delta": int(levels)
            - int(captured.hold_baseline_levels_completed),
        },
    }


def _build_prefix_step_dicts(captured: CapturedSafeStop) -> List[Dict[str, Any]]:
    """Reconstruct minimal prefix step dicts so the hold baseline is shared.

    The captured baseline progress already reflects the prefix; we only need
    placeholder rows so the post-stop summary's ``best_level`` accounts for the
    safe-stop endpoint. Levels at safe stop = hold baseline levels.
    """
    if not captured.prefix:
        return []
    rows: List[Dict[str, Any]] = []
    for index, step in enumerate(captured.prefix):
        rows.append(
            {
                "policy_selected_action": str(step.get("action", "")),
                "action_args": dict(step.get("action_args", {}) or {}),
                "changed_pixels": 0.0,
                "actor_relation_delta_count": 0,
                "action_effect_usefulness": 0,
                "new_relation_state": 0,
                "useful_new_state": False,
                "dead_end_or_cycle": False,
                "levels_after": int(captured.hold_baseline_levels_completed),
                "game_state_after": "NOT_FINISHED",
                "terminal_state_after": False,
                "state_signature_after": (
                    captured.safe_stop_state_hash
                    if index == len(captured.prefix) - 1
                    else f"prefix::{index}"
                ),
                "measurements": {},
            }
        )
    return rows


def _empty_diagnostics() -> Dict[str, Any]:
    return {
        "relation_delta_after_stop": 0,
        "changed_pixels": 0.0,
        "new_relation_states": 0,
        "distance_decreases_count": 0,
        "objective_readiness_signature_delta": 0,
    }


# --------------------------------------------------------------------------- #
# Validation + IO
# --------------------------------------------------------------------------- #


def ready_objective_conversion_requests(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    rows: List[Dict[str, Any]] = []
    for request in payload.get("objective_conversion_experiment_requests", []) or []:
        if not isinstance(request, Mapping):
            continue
        validate_ready_request(request)
        if str(request.get("status", "")) == (
            READY_FOR_M3_OBJECTIVE_CONVERSION_EXPERIMENT
        ):
            rows.append(dict(request))
    return tuple(rows)


def validate_ready_request(request: Mapping[str, Any]) -> None:
    if int(request.get("support", 0) or 0) != 0:
        raise ValueError("request support must remain 0")
    if bool(request.get("execution_performed", False)):
        raise ValueError("M3.G1.1 request must not already be executed")
    if str(request.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("request must remain candidate-only")
    if str(request.get("truth_status", "")) != M3_REFINEMENT_TRUTH_STATUS:
        raise ValueError("request truth_status must remain M3-local")
    if bool(request.get("experiment_result_counted_as_scientific_verdict", False)):
        raise ValueError("experiment result cannot count as scientific verdict")


def _validate_source_request_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source request summary support must remain 0")
    if bool(summary.get("execution_performed", False)):
        raise ValueError("M3.G1.1 source must be planning-only")
    if bool(summary.get("experiment_result_counted_as_scientific_verdict", False)):
        raise ValueError("experiment result cannot count as scientific verdict")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("source must not write A32/A33")


def _game_id_from_requests(requests: Sequence[Mapping[str, Any]]) -> str:
    for request in requests:
        game_id = str(request.get("game_id", ""))
        if game_id:
            return game_id
    return "bp35-0a0ad940"


def _prefix_hash(prefix: Sequence[Mapping[str, Any]]) -> str:
    raw = [
        {
            "action": str(step.get("action", "")),
            "action_args": dict(step.get("action_args", {}) or {}),
        }
        for step in prefix
    ]
    return "prefix::" + json.dumps(raw, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_objective_conversion_experiment_results(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3.G1.2 objective-conversion experiment cells.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--tie-break-seed", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_experiment_execution(
        requests_path=args.requests,
        environments_dir=args.environments_dir,
        tie_break_seed=args.tie_break_seed,
        max_cells=args.max_cells,
    )
    write_objective_conversion_experiment_results(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
