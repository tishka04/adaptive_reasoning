"""M3 controlled-experiment planning loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m1.controlled_followup_experiment import (
    DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH,
    ControlledExperiment,
    execute_controlled_followup,
)
from theory.m1.scientific_integration_pretest import (
    DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .next_experiment_selector import (
    DEFAULT_PREFERRED_CONTROLS,
    PlannedControlledExperiment,
    select_next_experiment,
)
from .scientific_planner_state import (
    ScientificPlanningState,
    build_scientific_planning_state_from_payloads,
    updated_ledger_entries_from_state,
)


DEFAULT_M3_OUTPUT_PATH = Path("diagnostics") / "m3" / "scientific_planning_bp35.json"
DEFAULT_M3_GAME_ID = "bp35-0a0ad940"
DEFAULT_M3_BUDGET = 3


def run_scientific_planning_loop(
    *,
    scientific_integration_path: str | Path = DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
    controlled_results_paths: Sequence[str | Path] | None = None,
    environments_dir: str | Path | None = None,
    game_id: str = DEFAULT_M3_GAME_ID,
    budget: int = DEFAULT_M3_BUDGET,
    preferred_controls: Sequence[str] = DEFAULT_PREFERRED_CONTROLS,
) -> Dict[str, Any]:
    """Run the minimal M3 scientific planning loop."""
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    ledger_payload = _load_json(scientific_integration_path)
    controlled_payloads = [
        _load_json(path)
        for path in _controlled_result_paths(controlled_results_paths)
        if Path(path).exists()
    ]
    initial_state = build_scientific_planning_state_from_payloads(
        ledger_payload=ledger_payload,
        controlled_payloads=controlled_payloads,
        budget=budget,
        game_id=game_id,
    )
    experiments = [dict(item) for item in initial_state.controlled_experiment_results]
    skipped_controls = [dict(item) for item in initial_state.skipped_controls]
    open_questions = list(initial_state.open_questions)
    planned_experiments: list[Dict[str, Any]] = []

    while True:
        state = build_scientific_planning_state_from_payloads(
            ledger_payload=ledger_payload,
            budget=budget,
            game_id=game_id,
            extra_controlled_experiments=experiments,
            skipped_controls=skipped_controls,
            open_questions=open_questions,
        )
        if state.remaining_budget <= 0:
            break

        live_actions = load_live_available_action_names(game_id, env_dir)
        plan = select_next_experiment(
            state,
            live_available_actions=live_actions,
            preferred_controls=preferred_controls,
        )
        planned_experiments.append(plan.to_dict())
        skipped_controls.extend(plan.skipped_controls)
        open_questions.extend(plan.open_questions)

        ledger_entry = _ledger_entry_for_plan(state, plan)
        experiment = execute_planned_controlled_experiment(
            plan,
            ledger_entry,
            environments_dir=env_dir,
            prior_state=state,
        )
        experiments.append(experiment)
        if int(experiment.get("controlled_experiments_run", 0) or 0) <= 0:
            open_questions.append("controlled_experiment_execution_failed")
            break

    experiments = annotate_support_independence(experiments)
    final_state = build_scientific_planning_state_from_payloads(
        ledger_payload=ledger_payload,
        budget=budget,
        game_id=game_id,
        extra_controlled_experiments=experiments,
        skipped_controls=skipped_controls,
        open_questions=open_questions,
    )
    summary = summarize_scientific_planning_loop(
        final_state,
        planned_experiments=planned_experiments,
    )
    return {
        "config": {
            "scientific_integration_path": str(scientific_integration_path),
            "controlled_results_paths": [
                str(path) for path in _controlled_result_paths(controlled_results_paths)
            ],
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "budget": int(budget),
            "preferred_controls": list(preferred_controls),
        },
        "summary": summary,
        "planning_state": final_state.to_dict(),
        "planned_experiments": planned_experiments,
        "controlled_experiments": experiments,
        "updated_ledger_entries": [
            dict(item) for item in updated_ledger_entries_from_state(final_state)
        ],
        "skipped_controls": [dict(item) for item in final_state.skipped_controls],
        "open_questions": list(final_state.open_questions),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "propose_ready_for_A15_revision": summary[
            "propose_ready_for_A15_revision"
        ],
        "propose_followup_disambiguation": summary[
            "propose_followup_disambiguation"
        ],
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def execute_planned_controlled_experiment(
    plan: PlannedControlledExperiment,
    ledger_entry: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    prior_state: ScientificPlanningState,
) -> Dict[str, Any]:
    """Execute a planned experiment through the M1.3l controlled runner."""
    result = execute_controlled_followup(
        ledger_entry,
        environments_dir=environments_dir,
        control_actions=(plan.control_action,),
    )
    experiment = result.to_dict() if isinstance(result, ControlledExperiment) else dict(result)
    experiment.update(
        {
            "planned_priority": round(float(plan.priority), 4),
            "planning_reason": plan.reason,
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "contradictions": 0,
            "controlled_test_required": True,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "trace_support_counted_as_proof": False,
            "prior_counted_as_proof": False,
            "observation_counted_as_confirmation": False,
        }
    )
    if plan.control_reuse_reason:
        experiment["control_reuse_reason"] = plan.control_reuse_reason
    independent, reused = support_independence_for_experiment(
        experiment,
        prior_state=prior_state,
    )
    experiment["independent_support_events"] = independent
    experiment["reused_control_support_events"] = reused
    return experiment


def support_independence_for_experiment(
    experiment: Mapping[str, Any],
    *,
    prior_state: ScientificPlanningState,
) -> tuple[int, int]:
    support = int(experiment.get("support_events", 0) or 0)
    if support <= 0:
        return 0, 0
    key = str(experiment.get("hypothesis_key", "") or "")
    control = str(experiment.get("control_action", "") or "")
    reuse_reason = str(experiment.get("control_reuse_reason", "") or "")
    prior_controls = set(prior_state.support_controls_by_key.get(key, ()))
    if control and not reuse_reason and control not in prior_controls:
        return 1, max(0, support - 1)
    return 0, support


def annotate_support_independence(
    experiments: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Add per-experiment raw/independent/reused support fields."""
    rows: list[Dict[str, Any]] = []
    support_controls_by_key: Dict[str, set[str]] = {}
    for experiment in experiments:
        row = dict(experiment)
        support = int(row.get("support_events", 0) or 0)
        key = str(row.get("hypothesis_key", "") or "")
        control = str(row.get("control_action", "") or "")
        reuse_reason = str(row.get("control_reuse_reason", "") or "")
        independent = 0
        reused = 0
        if support > 0:
            support_controls = support_controls_by_key.setdefault(key, set())
            if control and not reuse_reason and control not in support_controls:
                independent = 1
                reused = max(0, support - 1)
                support_controls.add(control)
            else:
                reused = support
        row["independent_support_events"] = independent
        row["reused_control_support_events"] = reused
        rows.append(row)
    return rows


def summarize_scientific_planning_loop(
    state: ScientificPlanningState,
    *,
    planned_experiments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    state_summary = state.summary()
    support_events = int(state_summary["support_events_total"])
    independent_support_events = int(
        state_summary["independent_support_events_total"]
    )
    reused_control_support_events = int(
        state_summary["reused_control_support_events_total"]
    )
    contradiction_events = int(state_summary["contradiction_events_total"])
    ready = (
        independent_support_events >= 2
        and support_events >= 3
        and contradiction_events == 0
    )
    return {
        "planned_experiments": len(planned_experiments),
        "executed_experiments": int(state_summary["controlled_experiments_run"]),
        "controlled_experiments_run": int(
            state_summary["controlled_experiments_run"]
        ),
        "support_events": support_events,
        "independent_support_events": independent_support_events,
        "reused_control_support_events": reused_control_support_events,
        "contradiction_events": contradiction_events,
        "unresolved_records": int(state_summary["open_hypotheses"]),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "controlled_test_required": True,
        "propose_ready_for_A15_revision": ready,
        "propose_followup_disambiguation": contradiction_events >= 1,
        "revision_performed": False,
        "observation_counted_as_confirmation": False,
        "wrong_confirmations": 0,
    }


def load_live_available_action_names(
    game_id: str,
    environments_dir: str | Path,
) -> tuple[str, ...]:
    """Return action names available at reset for ``game_id``."""
    from theory.m1.live_anchor_ranking import _load_live_grid_and_actions

    env_dir = Path(environments_dir)
    _configure_offline_env(env_dir)
    _, valid_actions = _load_live_grid_and_actions(game_id, env_dir)
    return tuple(
        sorted(
            {
                str(getattr(action, "name", ""))
                for action in valid_actions
                if str(getattr(action, "name", ""))
            }
        )
    )


def write_scientific_planning_result(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M3_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ledger_entry_for_plan(
    state: ScientificPlanningState,
    plan: PlannedControlledExperiment,
) -> Dict[str, Any]:
    for entry in state.ledger_entries:
        if str(entry.get("key", "")) == plan.hypothesis_key:
            return dict(entry)
    raise ValueError(f"planned hypothesis not found in ledger: {plan.hypothesis_key}")


def _controlled_result_paths(
    controlled_results_paths: Sequence[str | Path] | None,
) -> tuple[str | Path, ...]:
    if controlled_results_paths is not None:
        return tuple(controlled_results_paths)
    path = DEFAULT_CONTROLLED_EXPERIMENT_RESULTS_OUTPUT_PATH
    return (path,) if Path(path).exists() else ()


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M3 minimal scientific planning loop.",
    )
    parser.add_argument(
        "--scientific-integration",
        type=Path,
        default=DEFAULT_SCIENTIFIC_INTEGRATION_PRETEST_OUTPUT_PATH,
    )
    parser.add_argument(
        "--controlled-results",
        action="append",
        default=None,
        help="Prior controlled-result artifact. Can be repeated.",
    )
    parser.add_argument(
        "--no-prior-controlled-results",
        action="store_true",
        help="Ignore the default M1.3l controlled-result artifact.",
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_M3_GAME_ID)
    parser.add_argument("--budget", type=int, default=DEFAULT_M3_BUDGET)
    parser.add_argument("--out", type=Path, default=DEFAULT_M3_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    controlled_results: Sequence[str | Path] | None
    if args.no_prior_controlled_results:
        controlled_results = ()
    elif args.controlled_results is None:
        controlled_results = None
    else:
        controlled_results = tuple(Path(path) for path in args.controlled_results)
    payload = run_scientific_planning_loop(
        scientific_integration_path=args.scientific_integration,
        controlled_results_paths=controlled_results,
        environments_dir=args.environments_dir,
        game_id=args.game_id,
        budget=args.budget,
    )
    write_scientific_planning_result(payload, args.out)
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
