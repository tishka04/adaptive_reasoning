"""HUD.2 validation of TerminalHorizonObserver on real visual histories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
)
from theory.m3.objective_stop_switch_experiment_executor import (
    execute_decision_step,
    is_terminal_game_state,
    update_rollout_memory,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.p1.bp35_sage_candidate_policy_probe import (
    DEFAULT_GAME_ID,
    PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
    ProbeStep,
    candidate_policy_memory_from_scope,
    select_probe_decision,
    state_signature,
)
from theory.p3.terminal_horizon_estimator import (
    TerminalHorizonObserver,
    detect_budget_bar_candidates,
    score_budget_bar_candidate,
)
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_HUD_VALIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "bp35_terminal_horizon_hud_validation.json"
)
HUD_VALIDATION_SCHEMA_VERSION = "hud.terminal_horizon_validation.v1"
TRUTH_STATUS = "NOT_EVALUATED_BY_HUD_VALIDATION"


def env_actions_used(steps: Sequence[ProbeStep]) -> int:
    return sum(int(getattr(step, "env_actions", 1) or 0) for step in steps)


def capture_bp35_visual_history(
    *,
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    max_steps: int = 96,
    tie_break_seed: int = 0,
) -> Dict[str, Any]:
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    memory = candidate_policy_memory_from_scope(_load_json(scope_consolidation_path))
    env = _make_env(game_id, env_dir)
    current_frame = _reset_env(env)
    initial = snapshot_frame(current_frame)
    grid_history = [initial.grid]
    action_history: list[str] = []
    used_action6_args: list[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    seen_states: set[str] = {
        state_signature(initial.grid, initial.levels_completed, initial.game_state)
    }
    steps: list[ProbeStep] = []

    for step_index in range(max(0, int(max_steps))):
        before = snapshot_frame(current_frame)
        if is_terminal_game_state(before.game_state):
            break
        decision = select_probe_decision(
            condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            memory=memory,
            before_grid=before.grid,
            valid_actions=list(_valid_actions(env)),
            action_history=tuple(action_history),
            used_action6_args=tuple(used_action6_args),
            action_counts=action_counts,
            previous_steps=tuple(steps),
            tie_break_seed=tie_break_seed,
        )
        current_frame, step = execute_decision_step(
            env,
            current_frame,
            decision=decision,
            memory=memory,
            condition_label=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            step_index=step_index,
            seen_states=seen_states,
        )
        steps.append(step)
        after = snapshot_frame(current_frame)
        grid_history.append(after.grid)
        seen_states.add(step.state_signature_after)
        update_rollout_memory(
            step,
            action_history=action_history,
            used_action6_args=used_action6_args,
            action_counts=action_counts,
        )
        if is_terminal_game_state(after.game_state):
            break

    final_state = steps[-1].game_state_after if steps else initial.game_state
    return {
        "game_id": game_id,
        "policy": PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        "grid_history": grid_history,
        "action_history": action_history,
        "steps": steps,
        "terminal_reached": is_terminal_game_state(final_state),
        "terminal_step": len(steps) if is_terminal_game_state(final_state) else None,
        "final_game_state": final_state,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def top_budget_bar_candidate_log(
    *,
    grid: Any,
    history: Sequence[Any],
) -> Dict[str, Any] | None:
    scored = [
        score_budget_bar_candidate(candidate, observation=grid, history=history)
        for candidate in detect_budget_bar_candidates(grid)
    ]
    if not scored:
        return None
    return max(scored, key=lambda row: float(row.get("score", 0.0) or 0.0))


def estimate_sequence_monotone(values: Sequence[int | None]) -> bool:
    cleaned = [int(value) for value in values if value is not None]
    if len(cleaned) < 2:
        return False
    return all(cleaned[index] >= cleaned[index + 1] for index in range(len(cleaned) - 1))


def validate_hud_history(
    *,
    grid_history: Sequence[Any],
    action_history: Sequence[str],
    terminal_step: int | None,
    observer: TerminalHorizonObserver | None = None,
) -> Dict[str, Any]:
    observer = observer or TerminalHorizonObserver(terminal_budget_estimate=terminal_step)
    estimates: list[Dict[str, Any]] = []
    for index, grid in enumerate(grid_history):
        history_prefix = list(grid_history[:index])
        estimate = observer.estimate(
            observation=grid,
            history=history_prefix,
            policy_state={"env_actions_executed": index},
        )
        candidate = top_budget_bar_candidate_log(grid=grid, history=history_prefix)
        actual_remaining = None
        actual_next_remaining = None
        if terminal_step is not None:
            actual_remaining = max(0, int(terminal_step) - index)
            actual_next_remaining = max(0, int(terminal_step) - index - 1)
        predicted = None
        if estimate.estimated_moves_remaining is not None:
            predicted = max(0, int(estimate.estimated_moves_remaining) - 1)
        evidence = dict(estimate.evidence or {})
        estimates.append(
            {
                "step": index,
                "action_before_step": action_history[index - 1] if index > 0 and index - 1 < len(action_history) else None,
                "source": estimate.source,
                "confidence": estimate.confidence,
                "observed": estimate.observed,
                "estimated_moves_remaining": estimate.estimated_moves_remaining,
                "terminal_fraction_remaining": estimate.terminal_fraction_remaining,
                "bar_bbox": evidence.get("bar_bbox"),
                "orientation": evidence.get("bar_orientation") or evidence.get("orientation"),
                "bar_semantics": evidence.get("bar_semantics"),
                "estimated_remaining_rule": evidence.get("estimated_remaining_rule"),
                "monotonicity_score": evidence.get("score"),
                "monotonic_delta_observed": evidence.get("monotonic_delta_observed"),
                "ticks_lost_per_action": evidence.get("ticks_lost_per_action"),
                "ticks_changed_per_action": evidence.get("ticks_changed_per_action"),
                "predicted_next_remaining": predicted,
                "actual_next_remaining_proxy": actual_next_remaining,
                "actual_remaining_proxy": actual_remaining,
                "top_candidate": candidate,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
            }
        )

    source_counts = Counter(str(row["source"]) for row in estimates)
    hud_rows = [row for row in estimates if row["source"] == "hud_bar"]
    bbox_counts = Counter(
        json.dumps(row.get("bar_bbox"), sort_keys=True)
        for row in hud_rows
        if row.get("bar_bbox") is not None
    )
    dominant_bbox = json.loads(bbox_counts.most_common(1)[0][0]) if bbox_counts else None
    stable_bbox_steps = bbox_counts.most_common(1)[0][1] if bbox_counts else 0
    dominant_hud_rows = [
        row
        for row in hud_rows
        if row.get("bar_bbox") is not None and row.get("bar_bbox") == dominant_bbox
    ]
    hud_remaining_values = [
        row.get("estimated_moves_remaining") for row in dominant_hud_rows
    ]
    monotone_hud = estimate_sequence_monotone(hud_remaining_values)
    hud_detected_steps = len(hud_rows)
    stable_hud = bool(hud_detected_steps >= 3 and stable_bbox_steps >= max(3, hud_detected_steps // 2) and monotone_hud)
    fallback_steps = source_counts.get("empirical_fallback", 0)
    summary = {
        "real_history_steps": len(grid_history),
        "actions_observed": len(action_history),
        "terminal_step": terminal_step,
        "terminal_reached": terminal_step is not None,
        "source_counts": dict(source_counts),
        "hud_bar_source_active_steps": hud_detected_steps,
        "hud_bar_detected_on_real_sequence": bool(hud_detected_steps > 0),
        "stable_hud_bar_sequence_detected": stable_hud,
        "dominant_hud_bar_bbox": dominant_bbox,
        "dominant_hud_bar_stable_steps": stable_bbox_steps,
        "hud_estimated_remaining_nonincreasing": monotone_hud,
        "empirical_fallback_steps": fallback_steps,
        "source_remained_empirical_fallback": bool(hud_detected_steps == 0 and fallback_steps > 0),
        "ready_for_hud_p3_2b": stable_hud,
        "fallback_remains_active": bool(fallback_steps > 0),
        "false_positive_guard": "hud_bar_source_requires_monotone_stable_edge_history",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "hud_result_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }
    return {
        "estimates": estimates,
        "summary": summary,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def run_hud_visual_history_validation(
    *,
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    max_steps: int = 96,
    tie_break_seed: int = 0,
) -> Dict[str, Any]:
    capture = capture_bp35_visual_history(
        scope_consolidation_path=scope_consolidation_path,
        environments_dir=environments_dir,
        game_id=game_id,
        max_steps=max_steps,
        tie_break_seed=tie_break_seed,
    )
    terminal_step = capture["terminal_step"]
    observer = TerminalHorizonObserver(terminal_budget_estimate=terminal_step)
    validation = validate_hud_history(
        grid_history=capture["grid_history"],
        action_history=capture["action_history"],
        terminal_step=terminal_step,
        observer=observer,
    )
    return {
        "config": {
            "schema_version": HUD_VALIDATION_SCHEMA_VERSION,
            "game_id": game_id,
            "policy": PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            "max_steps": int(max_steps),
            "tie_break_seed": int(tie_break_seed),
            "inputs_read": ["M3.24", "real_bp35_visual_history"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33"],
            "observer": "TerminalHorizonObserver",
            "fusion_priority": [
                "environment_metadata",
                "hud_bar",
                "empirical_fallback",
                "unknown",
            ],
        },
        "trajectory_summary": {
            "terminal_reached": capture["terminal_reached"],
            "terminal_step": capture["terminal_step"],
            "final_game_state": capture["final_game_state"],
            "actions_observed": len(capture["action_history"]),
            "grid_history_steps": len(capture["grid_history"]),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        },
        "hud_validation": validation,
        "summary": validation["summary"],
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "hud_result_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a33_ready": False,
    }


def write_hud_visual_history_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_HUD_VALIDATION_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TerminalHorizonObserver HUD source on real bp35 visual history.",
    )
    parser.add_argument(
        "--scope-consolidation",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--tie-break-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_HUD_VALIDATION_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_hud_visual_history_validation(
        scope_consolidation_path=args.scope_consolidation,
        environments_dir=args.environments_dir,
        game_id=args.game_id,
        max_steps=args.max_steps,
        tie_break_seed=args.tie_break_seed,
    )
    write_hud_visual_history_validation(payload, args.out)


if __name__ == "__main__":
    main()
