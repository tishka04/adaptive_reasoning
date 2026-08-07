"""SAGE.T9.3b bounded retry with live spatial macro grounding."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import bounded_active_v9_3 as r1
from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6j_r3 as repair_r3
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v7 as live_i
from . import trajectory_planning_v9_2 as t9_2
from .contracts import ActionCandidate
from .controller import SageTConfig
from .decision import BayesianDecision
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor
from .synthesis import FragmentProposal, ProgramAssembler
from .terminal_calibration_v9 import T9_1_POLICIES

FORMAT_VERSION = "sage-t9.3b-live-grounded-bounded-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_3b_bounded_manifest.json"
)
DEFAULT_PARENT_REPORT = r1.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "bounded_v9_3b"


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "bounded_active_v9_3b.py": v86c._file_sha256(
            directory / "bounded_active_v9_3b.py"
        )
    }


def _load_parent_report(path: str | Path = DEFAULT_PARENT_REPORT) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3 report checksum mismatch")
    if report.get("status") != "T9_3_FAILED_CLOSED":
        raise ValueError("T9.3b requires the failed T9.3 pilot")
    checks = report.get("checks", {})
    allowed_failures = {"real_progress", "level_completed", "false_high_terminal"}
    if any(not value and name not in allowed_failures for name, value in checks.items()):
        raise ValueError("T9.3 failed outside reachability/reporting")
    if int(report.get("metrics", {}).get("levels_completed", -1)) != 0:
        raise ValueError("T9.3b requires the zero-progress failure")
    if report.get("t9_4_authorized") is not False:
        raise ValueError("T9.4 opened after failed T9.3")
    return report


def freeze_manifest(*, output_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    parent = _load_parent_report()
    base = r1.load_manifest()
    payload = json.loads(json.dumps(base))
    payload.pop("manifest_checksum", None)
    payload["format_version"] = FORMAT_VERSION
    payload["status"] = "FROZEN_BEFORE_T9_3B_BOUNDED_SOURCE_TRAIN"
    payload["parent_t9_3_report_checksum"] = parent["report_checksum"]
    payload["code_sha256"] = _code_hashes()
    payload["registered_changes"] = [
        "fallback repeat macros from materialized spatial legal actions",
        "westmost tie-break only among equal-utility admissible actions",
        "false-high metric matched to the action actually executed",
    ]
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3b manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.3b manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_3B_BOUNDED_SOURCE_TRAIN":
        raise ValueError("T9.3b manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.3b code drifted")
    if payload.get("parent_t9_3_report_checksum") != _load_parent_report()["report_checksum"]:
        raise ValueError("T9.3b parent report drifted")
    if payload.get("registered_changes") != [
        "fallback repeat macros from materialized spatial legal actions",
        "westmost tie-break only among equal-utility admissible actions",
        "false-high metric matched to the action actually executed",
    ]:
        raise ValueError("T9.3b changes drifted")
    firewall = payload.get("firewall", {})
    if any(
        bool(firewall.get(key))
        for key in (
            "source_validation_opened",
            "ar25_opened",
            "holdout_opened",
            "active_authority",
        )
    ):
        raise ValueError("T9.3b firewall is open")
    return payload


def live_spatial_macros(
    legal_actions: Sequence[ActionCandidate],
    *,
    maximum: int = 8,
) -> tuple[tuple[ActionCandidate, ...], ...]:
    """Repeat materialized spatial actions, ordered by relative x position."""

    spatial = {
        action.key: action
        for action in legal_actions
        if {"x", "y"}.issubset(action.action_data)
    }
    ordered = sorted(
        spatial.values(),
        key=lambda action: (
            float(action.action_data["x"]),
            float(action.action_data["y"]),
            action.key,
        ),
    )
    macros = []
    for action in ordered:
        for length in (3, 2):
            macros.append(tuple(action for _ in range(length)))
            if len(macros) >= max(0, int(maximum)):
                return tuple(macros)
    return tuple(macros)


class LiveGroundedDecisionEngine(t9_2.StructuralTrajectoryDecisionEngine):
    """Prefer the relative westmost action only when utility is tied."""

    def decide(self, *args: Any, **kwargs: Any) -> BayesianDecision:
        decision = super().decide(*args, **kwargs)
        if decision.chosen is None:
            return decision
        tied = [
            item
            for item in decision.assessments
            if not item.veto
            and abs(item.utility - decision.chosen.utility) <= 1e-9
            and "x" in item.first_action.action_data
        ]
        if not tied:
            return decision
        chosen = min(
            tied,
            key=lambda item: (
                float(item.first_action.action_data["x"]),
                -len(item.candidate.actions),
                item.candidate.key,
            ),
        )
        return replace(decision, chosen=chosen)


class LiveGroundedBoundedController(r1.BoundedTrajectoryController):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_engine = LiveGroundedDecisionEngine(
            executor=self.executor,
            maximum_sequences=self.config.maximum_sequences,
            maximum_particles=self.config.maximum_particles_per_decision,
            ordinary_horizon=self.config.ordinary_horizon,
            calibrator=self.terminal_calibrator,
            repeat_bonus_per_extra_action=self.repeat_bonus_per_extra_action,
        )

    def _ensure_programs(self, **kwargs: Any) -> None:
        super()._ensure_programs(**kwargs)
        fallback = live_spatial_macros(
            kwargs["candidates"],
            maximum=self.maximum_structural_macros,
        )
        combined = []
        seen = set()
        for macro in (*self._latest_proposal.plan_sequences, *fallback):
            key = tuple(action.key for action in macro)
            if key in seen:
                continue
            seen.add(key)
            combined.append(tuple(macro))
        self._latest_proposal = FragmentProposal(
            fragments=self._latest_proposal.fragments,
            plan_sequences=tuple(combined[: self.maximum_structural_macros]),
        )


def build_controller(manifest: Mapping[str, Any]) -> LiveGroundedBoundedController:
    caps = manifest["controller_caps"]
    executor = StructuralRoleProgramExecutor()
    t7 = v86.load_t7_manifest(verify_code=True)
    posterior_config = t7["posterior"]
    posterior = BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=T8_6G_POLICIES[live_i.SELECTED_POLICY].with_repair_v2(),
        maximum_particles=int(posterior_config["maximum_particles"]),
        channel_weights=v86._weights("joint"),
        unknown_coverage_penalty=float(posterior_config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(posterior_config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            posterior_config["repair_log_likelihood_threshold"]
        ),
        maximum_repair_contexts=repair_r3.MAXIMUM_REPAIR_CONTEXTS,
    )
    authority = manifest["authority"]
    return LiveGroundedBoundedController(
        executor=executor,
        posterior=posterior,
        proposer=live_i.StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=SageTConfig(
            mode="bounded",
            counterfactual_gate_passed=True,
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(caps["maximum_particles_per_decision"]),
            ordinary_horizon=int(caps["ordinary_horizon"]),
            bounded_maximum_interventions_per_reset=int(
                authority["maximum_interventions_per_reset"]
            ),
            bounded_maximum_terminal_risk=float(
                authority["maximum_marginal_terminal_risk"]
            ),
        ),
        terminal_policy=T9_1_POLICIES[str(manifest["selected_terminal_policy"])],
        maximum_structural_macros=int(caps["maximum_structural_macros"]),
        repeat_bonus_per_extra_action=0.35,
        strong_surprise_threshold=float(authority["strong_surprise_lockout_threshold"]),
    )


def _factory(manifest: Mapping[str, Any], registry: dict[str, LiveGroundedBoundedController]):
    def factory(game_id: str) -> UnifiedCognitiveController:
        sage_t = build_controller(manifest)
        registry[str(game_id)] = sage_t
        return UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(
                sage_t_authority_mode="bounded",
                sage_t_counterfactual_gate_passed=True,
            ),
            sage_t_controller=sage_t,
        )

    return factory


def _action_key(record: Mapping[str, Any]) -> str:
    action = record.get("action", {})
    return ActionCandidate(
        str(action.get("name", "")),
        dict(action.get("data", {}) or {}),
    ).key


def _actual_action_terminal_rows(
    controller: LiveGroundedBoundedController,
) -> tuple[tuple[float, bool], ...]:
    decisions = [
        record for record in controller.compact_records if record.get("kind") == "decision"
    ]
    observations = [
        record for record in controller.compact_records if record.get("kind") == "observation"
    ]
    rows = []
    for decision, observation in zip(decisions, observations):
        key = _action_key(decision)
        matches = [
            item
            for item in decision.get("sequences", ()) or ()
            if item.get("sequence") and item["sequence"][0] == key
        ]
        if not matches:
            continue
        shortest = min(matches, key=lambda item: len(item["sequence"]))
        terminal = "game_over" in tuple(observation.get("events", ()) or ())
        rows.append((float(shortest.get("terminal_risk", 0.0) or 0.0), terminal))
    return tuple(rows)


def run_pilot(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    runtime = live_base.runtime_capabilities()
    destination = Path(output_dir)
    if not runtime.get("ready"):
        report = {
            "format_version": FORMAT_VERSION,
            "status": "BLOCKED_RUNTIME",
            "manifest_checksum": manifest["manifest_checksum"],
            "runtime": runtime,
            "t9_4_authorized": False,
        }
        report["report_checksum"] = v86c._checksum(report)
        v86c._write_json(destination / "report.json", report)
        return report
    started = time.perf_counter()
    conditions = []
    registry: dict[str, LiveGroundedBoundedController] = {}
    for game_id in manifest["source_train_games"]:
        for seed in manifest["seeds"]:
            common = {
                "arm": "unified",
                "game_id": str(game_id),
                "seed": int(seed),
                "action_budget_per_reset": int(manifest["action_budget_per_reset"]),
                "resets": int(manifest["resets"]),
                "env_dir": Path(environments_dir),
                "env_factory": None,
            }
            off = live_base._run_arm(controller_factory=r1._off_factory, **common)
            bounded = live_base._run_arm(
                controller_factory=_factory(manifest, registry),
                **common,
            )
            controller = registry[str(game_id)]
            conditions.append(
                {
                    "game": str(game_id),
                    "seed": int(seed),
                    "off": r1._arm_metrics(off),
                    "bounded": r1._arm_metrics(bounded),
                    "intervention": r1._intervention_metrics(bounded, controller),
                    "controller_errors": len(tuple(bounded.get("controller_errors", ()) or ())),
                    "environment_errors": sum(
                        str(attempt.get("failure_cause", "")).startswith("environment_")
                        for attempt in bounded.get("attempts", ()) or ()
                    ),
                    "illegal_actions": sum(
                        "unavailable_decision" in str(error)
                        for error in bounded.get("controller_errors", ()) or ()
                    ),
                    "bounded_safety": controller.summary()["bounded_safety"],
                    "terminal_calibration": controller.terminal_calibrator.snapshot(),
                }
            )
    elapsed = time.perf_counter() - started
    terminal_rows = tuple(
        row
        for controller in registry.values()
        for row in _actual_action_terminal_rows(controller)
    )
    false_high_rate = (
        mean(not actual and risk >= 0.8 for risk, actual in terminal_rows)
        if terminal_rows
        else 0.0
    )
    decisions = [
        value for controller in registry.values() for value in controller.decision_latencies_ms
    ]
    observations = [
        value
        for controller in registry.values()
        for value in controller.observation_latencies_ms
    ]
    bounded_actions = sum(row["bounded"]["actions"] for row in conditions)
    progress = sum(row["bounded"]["progress_events"] for row in conditions)
    levels = sum(row["bounded"]["levels_completed"] for row in conditions)
    game_over_delta = sum(
        row["bounded"]["game_over_actions"] - row["off"]["game_over_actions"]
        for row in conditions
    )
    gate = manifest["gate"]
    checks = {
        "minimum_actions": bounded_actions >= int(gate["minimum_total_actions"]),
        "real_progress": progress >= int(gate["minimum_real_progress_events"]),
        "level_completed": levels >= int(gate["minimum_levels_completed"]),
        "game_over_nonincrease": game_over_delta <= int(gate["maximum_game_over_delta"]),
        "false_high_terminal": false_high_rate
        <= float(gate["maximum_false_high_terminal_rate"]),
        "decision_p95": r1._quantile(decisions, 0.95)
        <= float(gate["maximum_decision_p95_ms"]),
        "observation_p95": r1._quantile(observations, 0.95)
        <= float(gate["maximum_observation_p95_ms"]),
        "wall_time": elapsed <= float(gate["maximum_wall_seconds"]),
        "zero_illegal_actions": sum(row["illegal_actions"] for row in conditions) == 0,
        "zero_controller_errors": sum(row["controller_errors"] for row in conditions) == 0,
        "zero_environment_errors": sum(row["environment_errors"] for row in conditions) == 0,
        "bounded_budget": all(
            int(row["bounded_safety"]["maximum_interventions_in_branch"])
            <= int(manifest["authority"]["maximum_interventions_per_reset"])
            for row in conditions
        ),
        "firewall_closed": True,
    }
    passed = all(checks.values())
    report = {
        "format_version": FORMAT_VERSION,
        "status": "T9_3B_PASSED" if passed else "T9_3B_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "runtime": runtime,
        "checks": checks,
        "metrics": {
            "bounded_actions": bounded_actions,
            "progress_events": progress,
            "levels_completed": levels,
            "game_over_delta": game_over_delta,
            "false_high_terminal_rate": false_high_rate,
            "decision_p95_ms": r1._quantile(decisions, 0.95),
            "observation_p95_ms": r1._quantile(observations, 0.95),
            "wall_seconds": elapsed,
            "interventions": sum(
                row["intervention"]["interventions"] for row in conditions
            ),
            "useful_interventions": sum(
                row["intervention"]["useful_interventions"] for row in conditions
            ),
            "wasted_interventions": sum(
                row["intervention"]["wasted_interventions"] for row in conditions
            ),
        },
        "conditions": conditions,
        "t9_4_authorized": passed,
        "active_authority_authorized": False,
        "source_validation_opened": False,
        "holdout_opened": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_jsonl(destination / "conditions.jsonl", conditions)
    v86c._write_json(destination / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        result = freeze_manifest(output_path=args.manifest)
    else:
        result = run_pilot(
            manifest_path=args.manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze or result.get("status") == "T9_3B_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "LiveGroundedBoundedController",
    "LiveGroundedDecisionEngine",
    "build_controller",
    "freeze_manifest",
    "live_spatial_macros",
    "load_manifest",
    "main",
    "run_pilot",
]
