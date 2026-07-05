"""P1.9/P1.11 frontier trigger after exhausted policy refresh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p1.bp35_sage_candidate_policy_probe import (
    CONDITIONAL_ACTION4_REFRESH_POLICY,
    CONDITIONAL_MOVEMENT_REFRESH_POLICY,
    DEFAULT_GAME_ID,
    DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_CONDITIONAL_REFRESH_OUTPUT_PATH,
    DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MOVEMENT_REFRESH_OUTPUT_PATH,
    MOVEMENT_REFRESH_CANDIDATES,
    TRUTH_STATUS,
)


DEFAULT_P1_BP35_POLICY_FRONTIER_TRIGGER_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_policy_frontier_trigger.json"
)
DEFAULT_P1_BP35_MOVEMENT_POLICY_FRONTIER_TRIGGER_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_movement_policy_frontier_trigger.json"
)
ACTION4_FRONTIER_REASON = (
    "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_REFRESH"
)
MOVEMENT_FRONTIER_REASON = (
    "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH"
)
FRONTIER_REASON = MOVEMENT_FRONTIER_REASON


REFRESH_MODE_SPECS = {
    "action4": {
        "condition": CONDITIONAL_ACTION4_REFRESH_POLICY,
        "default_matrix_path": DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_CONDITIONAL_REFRESH_OUTPUT_PATH,
        "frontier_reason": ACTION4_FRONTIER_REASON,
        "no_saturation_reason": "NO_REFRESH_SATURATION_OBSERVED",
        "no_rollouts_reason": "NO_CONDITIONAL_REFRESH_ROLLOUTS_AVAILABLE",
        "source_label": "p1_8_conditional_refresh_matrix",
        "trigger_field": "conditional_refresh_triggers",
        "action6_after_field": "action6_after_conditional_refresh_steps",
        "useful_action6_after_field": "useful_action6_after_conditional_refresh_steps",
        "new_action6_after_field": "new_action6_affordances_after_refresh",
        "counted_as_confirmation_field": "conditional_refresh_counted_as_confirmation",
        "frontier_context_id": "after_soft_stale_and_conditional_refresh_exhaustion",
        "schema_version": "p1.synthetic_action4_saturation_fixture.v1",
        "refresh_candidates": ["ACTION4"],
    },
    "movement": {
        "condition": CONDITIONAL_MOVEMENT_REFRESH_POLICY,
        "default_matrix_path": DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MOVEMENT_REFRESH_OUTPUT_PATH,
        "frontier_reason": MOVEMENT_FRONTIER_REASON,
        "no_saturation_reason": "NO_MOVEMENT_REFRESH_SATURATION_OBSERVED",
        "no_rollouts_reason": "NO_CONDITIONAL_MOVEMENT_REFRESH_ROLLOUTS_AVAILABLE",
        "source_label": "p1_10_conditional_movement_refresh_matrix",
        "trigger_field": "conditional_movement_refresh_triggers",
        "action6_after_field": "action6_after_conditional_movement_refresh_steps",
        "useful_action6_after_field": "useful_action6_after_conditional_movement_refresh_steps",
        "new_action6_after_field": "new_action6_affordances_after_movement_refresh",
        "counted_as_confirmation_field": "movement_refresh_counted_as_confirmation",
        "frontier_context_id": "after_soft_stale_and_conditional_movement_refresh_exhaustion",
        "schema_version": "p1.synthetic_movement_saturation_fixture.v1",
        "refresh_candidates": list(MOVEMENT_REFRESH_CANDIDATES),
    },
}


def run_bp35_policy_frontier_trigger(
    *,
    refresh_matrix_path: str | Path | None = None,
    conditional_refresh_matrix_path: str | Path | None = None,
    movement_refresh_matrix_path: str | Path | None = None,
    refresh_mode: str = "movement",
    include_synthetic_fixture: bool = True,
    game_id: str = DEFAULT_GAME_ID,
) -> Dict[str, Any]:
    spec = _refresh_spec(refresh_mode)
    matrix_path = _resolve_matrix_path(
        spec,
        refresh_matrix_path=refresh_matrix_path,
        conditional_refresh_matrix_path=conditional_refresh_matrix_path,
        movement_refresh_matrix_path=movement_refresh_matrix_path,
    )
    matrix_payload = _load_json(matrix_path)
    real_rollout_evaluation = evaluate_policy_frontier_trigger(
        matrix_payload,
        game_id=game_id,
        refresh_mode=refresh_mode,
        synthetic_fixture=False,
        source_label=str(spec["source_label"]),
    )
    synthetic_fixture = (
        evaluate_policy_frontier_trigger(
            synthetic_saturation_matrix_fixture(
                game_id=game_id,
                refresh_mode=refresh_mode,
            ),
            game_id=game_id,
            refresh_mode=refresh_mode,
            synthetic_fixture=True,
            source_label="synthetic_saturation_fixture",
        )
        if include_synthetic_fixture
        else {}
    )
    return {
        "config": {
            "refresh_matrix_path": str(matrix_path),
            "refresh_mode": refresh_mode,
            "exhausted_policy": str(spec["condition"]),
            "refresh_candidates": list(spec["refresh_candidates"]),
            "game_id": game_id,
            "schema_version": "p1.bp35_policy_frontier_trigger.v1",
            "trigger_condition": str(spec["frontier_reason"]),
            "include_synthetic_fixture": bool(include_synthetic_fixture),
            "inputs_read": ["P1.10" if refresh_mode == "movement" else "P1.8"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["M3", "A32", "A33"],
        },
        "real_rollout_evaluation": real_rollout_evaluation,
        "synthetic_saturation_fixture": synthetic_fixture,
        "summary": {
            "real_frontier_triggered": bool(
                real_rollout_evaluation.get("frontier_triggered")
            ),
            "real_ready_for_p2_frontier_extraction": bool(
                real_rollout_evaluation.get("ready_for_p2_frontier_extraction")
            ),
            "synthetic_fixture_frontier_triggered": bool(
                synthetic_fixture.get("frontier_triggered")
            ),
            "synthetic_fixture_validates_trigger_logic": bool(
                synthetic_fixture.get("frontier_triggered")
                and synthetic_fixture.get("synthetic_fixture")
            ),
            "frontier_triggered": bool(
                real_rollout_evaluation.get("frontier_triggered")
            ),
            "ready_for_p2_frontier_extraction": bool(
                real_rollout_evaluation.get("ready_for_p2_frontier_extraction")
            ),
            "policy_result_counted_as_scientific_verdict": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "frontier_trigger_counted_as_confirmation": False,
    }


def evaluate_policy_frontier_trigger(
    matrix_payload: Mapping[str, Any],
    *,
    game_id: str = DEFAULT_GAME_ID,
    refresh_mode: str = "movement",
    synthetic_fixture: bool = False,
    source_label: str = "",
) -> Dict[str, Any]:
    spec = _refresh_spec(refresh_mode)
    _validate_candidate_only_matrix(matrix_payload)
    refresh_summaries = _refresh_summaries(matrix_payload, spec)
    exhausted = [
        row
        for row in refresh_summaries
        if _summary_indicates_exhausted_refresh(row, spec)
    ]
    refresh_triggered = [
        row
        for row in refresh_summaries
        if int(row.get(str(spec["trigger_field"]), 0) or 0) > 0
    ]
    frontier_triggered = bool(exhausted)
    frontier_reason = (
        str(spec["frontier_reason"])
        if frontier_triggered
        else _no_frontier_reason(
        refresh_summaries,
        refresh_triggered,
        spec,
    )
    )
    frontier_record = (
        _frontier_record(
            game_id=game_id,
            source_label=source_label,
            spec=spec,
            synthetic_fixture=synthetic_fixture,
            exhausted_summaries=exhausted,
        )
        if frontier_triggered
        else None
    )
    return {
        "source_label": source_label,
        "game_id": game_id,
        "refresh_mode": refresh_mode,
        "synthetic_fixture": bool(synthetic_fixture),
        "frontier_triggered": frontier_triggered,
        "frontier_reason": frontier_reason,
        "exhausted_policy": str(spec["condition"]),
        "refresh_candidates": list(spec["refresh_candidates"]),
        "refresh_summaries_seen": len(refresh_summaries),
        "refresh_runs_with_triggers": len(refresh_triggered),
        "exhausted_refresh_runs": len(exhausted),
        "refresh_total_triggers": sum(
            int(row.get(str(spec["trigger_field"]), 0) or 0)
            for row in refresh_summaries
        ),
        "new_action6_affordances_after_refresh_total": sum(
            int(row.get(str(spec["new_action6_after_field"]), 0) or 0)
            for row in refresh_summaries
        ),
        "ready_for_p2_frontier_extraction": bool(frontier_triggered),
        "frontier_record": frontier_record,
        "synthetic_fixture_not_for_scientific_handoff": bool(synthetic_fixture),
        "policy_result_counted_as_scientific_verdict": False,
        "frontier_trigger_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def synthetic_saturation_matrix_fixture(
    *,
    game_id: str = DEFAULT_GAME_ID,
    refresh_mode: str = "movement",
) -> Dict[str, Any]:
    spec = _refresh_spec(refresh_mode)
    exhausted_summary = {
        "condition": str(spec["condition"]),
        "policy_steps": 3,
        str(spec["trigger_field"]): 1,
        str(spec["action6_after_field"]): 1,
        str(spec["useful_action6_after_field"]): 0,
        str(spec["new_action6_after_field"]): 0,
        "neutral_or_unhelpful_action6_steps": 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }
    return {
        "config": {
            "game_id": game_id,
            "refresh_mode": refresh_mode,
            "schema_version": str(spec["schema_version"]),
        },
        "summary": {
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "aggregate": {
            "refresh_mode": refresh_mode,
            "refresh_runs": 1,
            "refresh_total_triggers": 1,
            "refresh_runs_with_triggers": 1,
            str(spec["counted_as_confirmation_field"]): False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "budget_runs": [
            {
                "budget": 8,
                "tie_break_seed": 0,
                "conditions_run": [str(spec["condition"])],
                "condition_summaries": [exhausted_summary],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                str(spec["counted_as_confirmation_field"]): False,
            }
        ],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _summary_indicates_exhausted_refresh(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> bool:
    return bool(
        int(row.get(str(spec["trigger_field"]), 0) or 0) > 0
        and int(row.get(str(spec["useful_action6_after_field"]), 0) or 0)
        == 0
        and int(row.get(str(spec["new_action6_after_field"]), 0) or 0) == 0
    )


def _refresh_summaries(
    matrix_payload: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for run in matrix_payload.get("budget_runs", []) or []:
        if not isinstance(run, Mapping):
            continue
        for summary in run.get("condition_summaries", []) or []:
            if not isinstance(summary, Mapping):
                continue
            if str(summary.get("condition", "")) == str(spec["condition"]):
                rows.append(dict(summary))
    return rows


def _no_frontier_reason(
    refresh_summaries: Sequence[Mapping[str, Any]],
    refresh_triggered: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> str:
    if not refresh_summaries:
        return str(spec["no_rollouts_reason"])
    if not refresh_triggered:
        return str(spec["no_saturation_reason"])
    return "REFRESH_DID_NOT_MEET_EXHAUSTED_FRONTIER_CRITERION"


def _frontier_record(
    *,
    game_id: str,
    source_label: str,
    spec: Mapping[str, Any],
    synthetic_fixture: bool,
    exhausted_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    condition = str(spec["condition"])
    return {
        "frontier_id": (
            f"p1::bp35::{condition}::exhausted_refresh"
            + ("::synthetic_fixture" if synthetic_fixture else "")
        ),
        "source_label": source_label,
        "game_id": game_id,
        "frontier_context_id": str(spec["frontier_context_id"]),
        "frontier_reason": str(spec["frontier_reason"]),
        "exhausted_policy": condition,
        "refresh_candidates": list(spec["refresh_candidates"]),
        "exhausted_runs": len(exhausted_summaries),
        "ready_for_p2_frontier_extraction": True,
        "policy_result_counted_as_scientific_verdict": False,
        "frontier_trigger_counted_as_confirmation": False,
        "synthetic_fixture": bool(synthetic_fixture),
        "synthetic_fixture_not_for_scientific_handoff": bool(synthetic_fixture),
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _validate_candidate_only_matrix(matrix_payload: Mapping[str, Any]) -> None:
    for section_name in ("summary", "aggregate"):
        section = matrix_payload.get(section_name, {}) or {}
        if not isinstance(section, Mapping):
            continue
        if int(section.get("support", 0) or 0) != 0:
            raise ValueError(f"{section_name}.support must remain 0")
        if bool(section.get("revision_performed", False)):
            raise ValueError(f"{section_name}.revision_performed must be false")
        if int(section.get("wrong_confirmations", 0) or 0) != 0:
            raise ValueError(f"{section_name}.wrong_confirmations must remain 0")
        if bool(section.get("conditional_refresh_counted_as_confirmation", False)):
            raise ValueError(
                f"{section_name}.conditional_refresh_counted_as_confirmation must be false"
            )
        if bool(section.get("movement_refresh_counted_as_confirmation", False)):
            raise ValueError(
                f"{section_name}.movement_refresh_counted_as_confirmation must be false"
            )


def _refresh_spec(refresh_mode: str) -> Mapping[str, Any]:
    mode = str(refresh_mode)
    if mode not in REFRESH_MODE_SPECS:
        raise ValueError(f"unknown refresh mode: {mode}")
    return REFRESH_MODE_SPECS[mode]


def _resolve_matrix_path(
    spec: Mapping[str, Any],
    *,
    refresh_matrix_path: str | Path | None,
    conditional_refresh_matrix_path: str | Path | None,
    movement_refresh_matrix_path: str | Path | None,
) -> Path:
    if refresh_matrix_path is not None:
        return Path(refresh_matrix_path)
    condition = str(spec["condition"])
    if condition == CONDITIONAL_MOVEMENT_REFRESH_POLICY and movement_refresh_matrix_path:
        return Path(movement_refresh_matrix_path)
    if condition == CONDITIONAL_ACTION4_REFRESH_POLICY and conditional_refresh_matrix_path:
        return Path(conditional_refresh_matrix_path)
    return Path(spec["default_matrix_path"])


def write_bp35_policy_frontier_trigger(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P1_BP35_POLICY_FRONTIER_TRIGGER_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P1 bp35 policy frontier trigger evaluation.",
    )
    parser.add_argument(
        "--refresh-mode",
        choices=sorted(REFRESH_MODE_SPECS),
        default="movement",
        help="Refresh policy family to evaluate.",
    )
    parser.add_argument(
        "--refresh-matrix",
        type=Path,
        default=None,
        help="Explicit refresh matrix path. Overrides mode-specific defaults.",
    )
    parser.add_argument(
        "--conditional-refresh-matrix",
        type=Path,
        default=None,
        help="Legacy P1.8 ACTION4-only refresh matrix path.",
    )
    parser.add_argument(
        "--movement-refresh-matrix",
        type=Path,
        default=None,
        help="P1.10 movement-refresh matrix path.",
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument(
        "--no-synthetic-fixture",
        action="store_true",
        help="Do not include the synthetic saturation contract fixture.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P1_BP35_MOVEMENT_POLICY_FRONTIER_TRIGGER_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_bp35_policy_frontier_trigger(
        refresh_matrix_path=args.refresh_matrix,
        conditional_refresh_matrix_path=args.conditional_refresh_matrix,
        movement_refresh_matrix_path=args.movement_refresh_matrix,
        refresh_mode=args.refresh_mode,
        include_synthetic_fixture=not bool(args.no_synthetic_fixture),
        game_id=args.game_id,
    )
    write_bp35_policy_frontier_trigger(payload, args.out)
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
